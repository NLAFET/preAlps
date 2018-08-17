/*
============================================================================
Name        : test_lorasc.c
Author      : Simplice Donfack
Version     : 0.1
Description : Preconditioner based on Schur complement
Date        : Mai 15, 2017
============================================================================
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#ifdef PETSC
/* Petsc */
#include <petsc_interface.h>
#include <petscksp.h>
#endif

//#include <mat_load_mm.h>

#include "preAlps_utils.h"
#include "preAlps_doublevector.h"
#include <cplm_utils.h>
#include "cplm_matcsr.h"
#include "precond.h"
#include "preAlps_preconditioner.h"
#include "lorasc.h"
#include "block_jacobi.h"
#include "ecg.h"
#include "operator.h"

//#define USE_OPERATORBUILD 0

//#define USE_OPERATOR_MATMULT_GATHERV 1 //debug only, use gather the vector to perform the matvec operation

//#define USE_SPMSV_DBG 1

#ifdef USE_SPMSV_DBG
#include "spmsv_dbg/preAlps_spmsv_dbg.c"
#endif //SPMSV




static int checkArgs(MPI_Comm comm, int precond_num, Prec_Type_t precond_type, int npLevel1){
  int my_rank, nbprocs;

  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &nbprocs);

  if(my_rank==0){
    if(precond_num<0 || precond_num>2) preAlps_abort("Unknown preconditioner");

    if(precond_type != PREALPS_LORASC && npLevel1!=nbprocs){
      preAlps_abort("This preconditioner does not support more than 1 block per process. Do not set -npLevel1");
    }

    if(nbprocs%npLevel1!=0){
      preAlps_abort("npLevel1: %d should be a multiple of p: %d", npLevel1, nbprocs);
    }
  }


  return 0;
}

static void help_show(){
  printf(" Purpose\n");
  printf(" =======\n");
  printf(" Preconditioner based on Schur complement, \n");
  printf(" more details...\n");
  printf("\n");
  printf(" Usage\n");
  printf(" =========\n");
  printf(" mpirun -np <nbprocs> ./test_lorasc -m <matrix_file_name> -r <rhs_file_name> -t <ECG_enlargedFactor> -p <preconditioner_num>\n");
  printf("\n");
  printf(" Arguments\n");
  printf(" =========\n");
  printf(" -m: the matrix file\n");
  printf("    the matrix stored in matrix market format\n");
  printf(" -r: the right hand side file\n");
  printf("     the right hand side stored in a text file\n");
  printf(" -t: the enlarged factor (default :1)\n");
  printf(" -npLevel1: the number of processors at the first level if greater then 0 (default : 0 = <all processors>)\n");
  printf(" -p: preconditioner \n");
  printf("     0: no prec, 1: blockJacobi, 2: Lorasc\n");
}

/**/
int main(int argc, char** argv){

  MPI_Comm comm, comm_masterGroup, comm_localGroup;
  int nbprocs, my_rank, root = 0, groupLevel = 0, local_root=0;
  int npLevel1 = 0; //For multilevel algorithms, the number of processors at the first level
  //int npLevel2 = 1; //For multilevel algorithms, the number of processors at the second level

  int i, ierr = 0;
  char matrix_filename[150]="", rhs_filename[150]="";
  CPLM_Mat_CSR_t A = CPLM_MatCSRNULL(), AOrigin = CPLM_MatCSRNULL();
  CPLM_Mat_CSR_t locAP = CPLM_MatCSRNULL();
  double *R = NULL, *C = NULL; //For the scaling


  double *b = NULL, *rhs = NULL, *sol = NULL, *x = NULL, *rhsOrigin = NULL;
  int m = 0, n, nnz, mloc, offsetloc, rhs_size = 0;

  double ttemp, tPartition =0.0, tPrec = 0.0, tSolve = 0.0, tTotal;


  /* Required by block Jacobi*/
  int* rowPos = NULL;
  int* colPos = NULL;
  int sizeRowPos, sizeColPos;

  // Generic preconditioner type and object
  Prec_Type_t precond_type = PREALPS_LORASC; //PREALPS_NOPREC, PREALPS_BLOCKJACOBI
  PreAlps_preconditioner_t *precond = NULL;

  #ifdef USE_OPERATOR_MATMULT_GATHERV //DEBUG ONLY
    double *vs;
  #endif

  // Lorasc preconditioner
  preAlps_Lorasc_t *lorascA = NULL;

  /* Program default parameters */
  int doScale = 1, monitorResidual = 0, doSolutionCheck = 1;
  int precond_num = 2; /* 0: no prec, 1: blockJacobi, 2: Lorasc */
  int ecg_enlargedFactor = 1, ecg_maxIter = 30000;
  int ecg_ortho_alg = ORTHOMIN;
  double ecg_tol = 1e-5;


  // Start MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_dup(MPI_COMM_WORLD, &comm);

  // Let me know who I am
  MPI_Comm_size(comm, &nbprocs);
  MPI_Comm_rank(comm, &my_rank);

#ifdef DEBUG
  printf("I am proc %d over %d processors\n", my_rank, nbprocs);
#endif

  // Get user parameters
  for(i=1;i<argc;i+=2){
    if (strcmp(argv[i],"-m") == 0) strcpy(matrix_filename,argv[i+1]);
    if (strcmp(argv[i],"-r") == 0) strcpy(rhs_filename,argv[i+1]);
    if (strcmp(argv[i],"-t") == 0) ecg_enlargedFactor = atoi(argv[i+1]);
    if (strcmp(argv[i],"-p") == 0) precond_num = atoi(argv[i+1]);
    if (strcmp(argv[i],"-npLevel1") == 0) npLevel1 = atoi(argv[i+1]);
    if (strcmp(argv[i],"-h") == 0){
      if(my_rank==0){
        help_show();
      }
      MPI_Finalize();
      return EXIT_SUCCESS;
    }
  }


  switch(precond_num){
    case 0:
      precond_type = PREALPS_NOPREC;
      if(my_rank==root) printf("Preconditioner: NONE\n");
    break;
    case 1:
      precond_type = PREALPS_BLOCKJACOBI;
      if(my_rank==root) printf("Preconditioner: BLOCKJACOBI\n");
    break;
    case 2:
      precond_type = PREALPS_LORASC;
      if(my_rank==root) printf("Preconditioner: LORASC\n");
    break;
  }

  // Check args
  if(npLevel1<=0 || npLevel1>nbprocs) npLevel1 = nbprocs;

  checkArgs(comm, precond_num, precond_type, npLevel1);

  // Print Summary
  if(my_rank==0){

    if(strlen(matrix_filename)==0){
      preAlps_abort("Error: unknown Matrix. ./test_lorasc -h for usage");
    }

    printf("Matrix name: %s\nEnlarged factor: %d\n", matrix_filename, ecg_enlargedFactor);
    printf("nbprocs: %d, nbprocs Level 1: %d\n", nbprocs, npLevel1);
    printf("Reading matrix ...\n");
  }

  // Load the matrix on proc 0
  if(my_rank==root){

    const char* ext = preAlps_get_filename_extension(matrix_filename);

    if (strcmp(ext,"mtx") == 0) {
      CPLM_LoadMatrixMarket(matrix_filename, &A);
    }else{
      #ifdef PETSC
        Mat A_petsc;
        petscMatLoad(&A_petsc,matrix_filename,PETSC_COMM_SELF);
        petscCreateMatCSR(A_petsc,&A);
        MatDestroy(&A_petsc);
      #else
        CPLM_Abort("Please Compile with PETSC to read other matrix file type");
      #endif
    }

    if(doSolutionCheck){
      //Save the matrix for the solution check
      CPLM_MatCSRCopy(&A, &AOrigin);
    }

    CPLM_MatCSRPrintInfo(&A);
    CPLM_MatCSRPrintCoords(&A, "Loaded matrix");
  }

  // Load the Right-hand side on proc 0
  if(my_rank==root){

    if(strlen(rhs_filename)==0){

      if ( !(rhs  = (double *) malloc(A.info.m*sizeof(double))) ) preAlps_abort("Malloc fails for rhs[].");
      /*
      // rhs not provided, generate x and compute rhs = Ax
      double *xTmp;
      xTmp = (double*) malloc(A.info.m*sizeof(double));
      for (int k = 0 ; k < A.info.m ; k++) xTmp [k] = 1.0;

      CPLM_MatCSRMatrixVector(&A, 1.0, xTmp, 0.0, rhs);
      free(xTmp);
      */
      // Generate a random vector
      srand(11);
      for(int i=0;i<A.info.m;i++)
        rhs[i] = ((double) rand() / (double) RAND_MAX);
    }else{
      // Read rhs on proc 0
      preAlps_doubleVector_load(rhs_filename, &rhs, &rhs_size);

      if(rhs_size!=A.info.m){
        preAlps_abort("Error: the matrix and rhs size does not match. Matrix size: %d x %d, rhs size: %d", A.info.m, A.info.n, rhs_size);
      }

    }
    //preAlps_doubleVector_printSynchronized(b, A.info.m, "b0", "b", MPI_COMM_SELF);
    if(doSolutionCheck){
      rhsOrigin = (double*) malloc(A.info.m*sizeof(double));
      for (int k = 0 ; k < A.info.m ; k++) rhsOrigin [k] = rhs[k];
    }
  }

  // Scale the matrix and the rhs
  if(doScale && my_rank==0){

      if ( !(R  = (double *) malloc(A.info.m * sizeof(double))) ) preAlps_abort("Malloc fails for R[].");
      if ( !(C  = (double *) malloc(A.info.n * sizeof(double))) ) preAlps_abort("Malloc fails for C[].");

      CPLM_MatCSRSymRACScaling(&A, R, C);

      preAlps_doubleVector_printSynchronized(R, A.info.m, "R", "R", MPI_COMM_SELF);
      preAlps_doubleVector_printSynchronized(C, A.info.n, "C", "C", MPI_COMM_SELF);

      #ifdef BUILDING_MATRICES_DUMP
        printf("Dumping the matrix ...\n");
        CPLM_MatCSRSave(&A, "dump_AScaled.mtx");
        printf("Dumping the matrix ... done\n");
      #endif

      CPLM_MatCSRPrintCoords(&A, "Scaled matrix");

      //Apply the Scaling factor on the rhs
      if(R) preAlps_doubleVector_pointWiseProductInPlace(R, rhs, A.info.m);

  }



  /* Build the selected preconditionner */

  if(precond_type==PREALPS_LORASC){

    ttemp =  MPI_Wtime();

    // Memory allocation for lorasc preconditioner
    ierr =  preAlps_LorascAlloc(&lorascA); preAlps_checkError(ierr);

    // The number of processor at the first level
    lorascA->npLevel1 = npLevel1;

    // Set parameters for the preconditioners
    lorascA->deflation_tol = 1e-2;//5e-3

    // Change the nrhs before building lorasc (required for analysis by internal solvers such as MUMPS )
    lorascA->nrhs = ecg_enlargedFactor;

    // Build the preconditioner and distribute the matrix
    ierr = preAlps_LorascBuild(lorascA, &A, &locAP, comm); preAlps_checkError(ierr);

    tPrec = MPI_Wtime() - ttemp - lorascA->tPartition;

    if(my_rank==root) printf("Schur-complement size: %d\n", lorascA->sep_nrows);

    tPartition = lorascA->tPartition;

  }else{

    //ttemp =  MPI_Wtime();
    /* permute only the matrix using lorasc, do not build the preconditioner */

    // Memory allocation for lorasc preconditioner
    ierr =  preAlps_LorascAlloc(&lorascA); preAlps_checkError(ierr);

    //permute only the matrix using lorasc, do not build the preconditioner
    lorascA->OptPermuteOnly = 1;

    // permute and distribute the matrix using lorasc
    ierr = preAlps_LorascBuild(lorascA, &A, &locAP, comm); preAlps_checkError(ierr);

    tPartition = lorascA->tPartition; //MPI_Wtime() - ttemp;
  }

  comm_masterGroup = lorascA->comm_masterGroup;
  comm_localGroup  = lorascA->comm_localGroup;

  if(comm_masterGroup!=MPI_COMM_NULL){

    CPLM_MatCSRPrintSynchronizedCoords(&locAP, comm_masterGroup, "locAP", "locAP");

    // Broadcast the global matrix dimension from the root to the other procs in the master groups
    CPLM_MatCSRDimensions_Bcast(&A, root, &m, &n, &nnz, comm_masterGroup);

    // Prepare the operator
    if(lorascA) preAlps_OperatorBuildNoPerm(&locAP, lorascA->partBegin, 1, comm_masterGroup);

    #ifdef USE_OPERATOR_MATMULT_GATHERV //DEBUG ONLY
      if(my_rank==0) printf("[DEBUG] ***** USING MATMULT GATHERV DEBUG **** \n");
      vs = (double*) malloc(m*ecg_enlargedFactor*sizeof(double));
    #endif

  }

  /* Prepare the selected preconditioner */

  if(precond_type==PREALPS_NOPREC){

    // Create a generic preconditioner object compatible with EcgSolver
    preAlps_PreconditionerCreate(&precond, precond_type, NULL);

  } else if(precond_type==PREALPS_BLOCKJACOBI){
    ttemp =  MPI_Wtime();
    // Get row partitioning of A from the operator
    preAlps_OperatorGetRowPosPtr(&rowPos,&sizeRowPos);
    // Get col partitioning induced by this row partitioning
    preAlps_OperatorGetColPosPtr(&colPos,&sizeColPos);
    // Construct the preconditioner
    preAlps_BlockJacobiCreate(&locAP, rowPos, sizeRowPos, colPos, sizeColPos);
    tPrec = MPI_Wtime() - ttemp;
  }else if(precond_type==PREALPS_LORASC){
    // Create a generic preconditioner object compatible with EcgSolver
    preAlps_PreconditionerCreate(&precond, precond_type, (void *) lorascA);
  }

  if(comm_masterGroup!=MPI_COMM_NULL){

    if ( !(b  = (double *) malloc(m*sizeof(double))) ) preAlps_abort("Malloc fails for b[].");

    if(my_rank==root){
      //Apply the permutation on the right hand side
      preAlps_doubleVector_permute(lorascA->perm, rhs, b, m);
    }

    //Distribute the rhs
    MPI_Scatterv(b, lorascA->partCount, lorascA->partBegin, MPI_DOUBLE, my_rank==0?MPI_IN_PLACE:b, locAP.info.m, MPI_DOUBLE, root, comm_masterGroup);
    preAlps_doubleVector_printSynchronized(b, locAP.info.m, "b after the distribution", "b", comm_masterGroup);

  }

  /* Solve the system */

  preAlps_ECG_t ecg;
  // Set parameters
  ecg.comm       = comm_masterGroup;  /* MPI Communicator */
  ecg.globPbSize = m;                 /* Size of the global problem */
  ecg.locPbSize  = locAP.info.m;      /* Size of the local problem */
  ecg.maxIter    = ecg_maxIter;       /* Maximum number of iterations */
  ecg.enlFac     = ecg_enlargedFactor;/* Enlarging factor */
  ecg.tol        = ecg_tol;           /* Tolerance of the method */
  ecg.ortho_alg  = ecg_ortho_alg;     /* Orthogonalization algorithm */
  ecg.bs_red     = NO_BS_RED;         /* Reduction of the search directions */

  int rci_request = 0;
  int stop = 0;

  ttemp = MPI_Wtime();
  // Allocate memory and initialize variables
  if(comm_masterGroup!=MPI_COMM_NULL){
    preAlps_ECGInitialize(&ecg, b, &rci_request);
    preAlps_doubleVectorSet_printSynchronized(ecg.R->val, ecg.R->info.m, ecg.R->info.n, ecg.R->info.lda, "ecg.R", "ecg.R0", comm_masterGroup);
  }

  // Finish initialization
  if(precond_type==PREALPS_BLOCKJACOBI) {
    preAlps_BlockJacobiApply(ecg.R,ecg.P);
  }
  else {
    preAlps_PreconditionerMatApply(precond, ecg.R, ecg.P);
  }


  if(comm_masterGroup!=MPI_COMM_NULL){

    #ifdef USE_OPERATOR_MATMULT_GATHERV
      //Algatherv AP
      MPI_Allgatherv(ecg.P->val, ecg.P->info.nval, MPI_DOUBLE, vs, lorascA->partCount, lorascA->partBegin, MPI_DOUBLE, comm_masterGroup);
      CPLM_MatCSRMatrixCSRDenseMult(&locAP, 1.0, vs, ecg.P->info.n, m, 0.0, ecg.AP->val, ecg.AP->info.lda);
    #else
      preAlps_BlockOperator(ecg.P, ecg.AP);
      //spmsv_dbg(comm_masterGroup, &locAP, ecg.P, lorascA->partCount, ecg.AP);
    #endif
  }

  // Main loop
  while (stop != 1) {

    //Iterate in the master group
    if(comm_masterGroup!=MPI_COMM_NULL) preAlps_ECGIterate(&ecg, &rci_request);

    //Broadcast rci_request to the local group
    MPI_Bcast(&rci_request, 1, MPI_INT, local_root, comm_localGroup);

    if (rci_request == 0) {

      if(comm_masterGroup!=MPI_COMM_NULL){
        #ifdef USE_OPERATOR_MATMULT_GATHERV
          MPI_Allgatherv(ecg.P->val, ecg.P->info.nval, MPI_DOUBLE, vs, lorascA->partCount, lorascA->partBegin, MPI_DOUBLE, comm_masterGroup);
          CPLM_MatCSRMatrixCSRDenseMult(&locAP, 1.0, vs, ecg.P->info.n, m, 0.0, ecg.AP->val, ecg.AP->info.lda);
        #else
          preAlps_BlockOperator(ecg.P, ecg.AP);
          //spmsv_dbg(comm_masterGroup, &locAP, ecg.P, lorascA->partCount, ecg.AP);
        #endif
      }

    }
    else if (rci_request == 1) {

      if(comm_masterGroup!=MPI_COMM_NULL) preAlps_ECGStoppingCriterion(&ecg, &stop);

      //Broadcast stop to the local group
      MPI_Bcast(&stop, 1, MPI_INT, local_root, comm_localGroup);

      if (stop == 1) break;

      if (ecg_ortho_alg == ORTHOMIN){

        if(precond_type==PREALPS_BLOCKJACOBI) {
          preAlps_BlockJacobiApply(ecg.R,ecg.Z);
        } else {
          preAlps_PreconditionerMatApply(precond, ecg.R, ecg.Z);
        }

      }
      else if (ecg_ortho_alg == ORTHODIR){

        if(precond_type==PREALPS_BLOCKJACOBI) {
          preAlps_BlockJacobiApply(ecg.AP, ecg.Z);
        }else{
          preAlps_PreconditionerMatApply(precond, ecg.AP, ecg.Z);
        }

      }

    }

    //preAlps_abort("dbgSolve brk1");

    if(monitorResidual && my_rank==root) printf("Iteration: %d, \tres: %e\n", ecg.iter, ecg.res);
  }

  if(comm_masterGroup!=MPI_COMM_NULL){


    sol = (double*) malloc(locAP.info.m*sizeof(double));

    // Retrieve solution and free memory
    preAlps_ECGFinalize(&ecg, sol);

    preAlps_doubleVector_printSynchronized(sol, locAP.info.m, "sol", "solution", comm_masterGroup);

    // Gather the solution on proc 0

    if(my_rank==0) x = (double*) malloc(m*sizeof(double));
    MPI_Gatherv(sol, locAP.info.m, MPI_DOUBLE, x, lorascA->partCount, lorascA->partBegin, MPI_DOUBLE, root, comm_masterGroup);

    // Post process the solution
    if(my_rank==0) {

      double *xTmp;
      xTmp = (double*) malloc(m*sizeof(double));

      preAlps_doubleVector_invpermute(lorascA->perm, x, xTmp, m);

      //Apply the Scaling factor on the solution
      if(C) preAlps_doubleVector_pointWiseProduct(C, xTmp, x, m);

      preAlps_doubleVector_printSynchronized(x, m, "final solution", "xTmp", MPI_COMM_SELF);

      free(xTmp);
    }


    //Check the solution
    if(doSolutionCheck && my_rank==0){
      double *rTmp, normRes, normRhs;
      rTmp = (double*) malloc(m*sizeof(double));
      for (int k = 0 ; k < m ; k++) rTmp [ k] = rhsOrigin [k] ;
      // compute Ax-b
      CPLM_MatCSRMatrixVector(&AOrigin, 1.0, x, -1.0, rTmp);

      preAlps_doubleVector_printSynchronized(rTmp, m, "err=b-AX", "err", MPI_COMM_SELF);

      normRes = preAlps_doubleVector_norm2(rTmp, m);
      normRhs = preAlps_doubleVector_norm2(rhsOrigin, m);
      printf("norm (b-Ax)/norm(b): %e\n", normRes/normRhs);
      free(rTmp);
    }

    tSolve = MPI_Wtime() - ttemp;

    if (my_rank == 0)
      printf("=== ECG ===\n\tSolver iterations: %d\n\tnorm(res): %e\n",ecg.iter,ecg.res);

    //tTotal = tPartition + tPrec + tSolve;
    tTotal = tPrec + tSolve; //exclude partitioning time

    preAlps_dstats_display(comm_masterGroup, tPartition, "Time partitioning");
    preAlps_dstats_display(comm_masterGroup, tPrec, "Time preconditioner");
    preAlps_dstats_display(comm_masterGroup, tSolve, "Time Solve");
    preAlps_dstats_display(comm_masterGroup, tTotal, "Time Total");



    #ifdef USE_OPERATOR_MATMULT_GATHERV
      free(vs);
    #endif

    //Free memory
    preAlps_OperatorFree();

  }


  //Destroy the preconditioner/the partitioning
  //if(precond_type==PREALPS_LORASC){
  ierr =  preAlps_LorascDestroy(&lorascA); preAlps_checkError(ierr);
  //}

  if(comm_masterGroup!=MPI_COMM_NULL){
    // Destroy the generic preconditioner object
    preAlps_PreconditionerDestroy(&precond);
  }

  //Free memory
  if(b) free(b);
  if(rhs) free(rhs);
  if(sol) free(sol);
  if(x) free(x);
  CPLM_MatCSRFree(&locAP);
  if(my_rank==0){
    CPLM_MatCSRFree(&A);
    CPLM_MatCSRFree(&AOrigin);
    if(rhsOrigin) free(rhsOrigin);
    if(R) free(R);
    if(C) free(C);
  }

  MPI_Finalize();
  return EXIT_SUCCESS;
}
