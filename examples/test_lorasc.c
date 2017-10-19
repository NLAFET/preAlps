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
#include <mat_load_mm.h>
#include <mat_csr.h>

#include "preAlps_utils.h"
#include "preAlps_doublevector.h"
#include "preAlps_cplm_matcsr.h"
#include "precond.h"
#include "preAlps_preconditioner.h"
#include "lorasc.h"
#include "block_jacobi.h"
#include "ecg.h"
#include "operator.h"

//#define USE_OPERATORBUILD 0

//#define MATMULT_GATHERV  //debug

/**/
int main(int argc, char** argv){

  MPI_Comm comm;
  int nbprocs, my_rank, root =0;
  char matrix_filename[150]="", rhs_filename[150]="";
  CPLM_Mat_CSR_t A = CPLM_MatCSRNULL();
  CPLM_Mat_CSR_t locAP = CPLM_MatCSRNULL();
  int i, ierr = 0;

  double *b = NULL, *rhs = NULL;//*x = NULL,
  int m = 0, n, nnz, mloc, offsetloc, rhs_size = 0;

  double ttemp, tPartition =0.0, tPrec = 0.0, tSolve = 0.0, tTotal;

  /* Required by block Jacobi*/
  int* rowPos = NULL;
  int* colPos = NULL;
  int sizeRowPos, sizeColPos;

  // Generic preconditioner type and object
  Prec_Type_t precond_type = PREALPS_LORASC; //PREALPS_NOPREC, PREALPS_BLOCKJACOBI
  PreAlps_preconditioner_t *precond = NULL;


  // Lorasc preconditioner
  preAlps_Lorasc_t *lorascA = NULL;

  /* Program default parameters */
  int doScale = 0, monitorResidual = 0;
  int precond_num = 2; /* 0: no prec, 1: blockJacobi, 2: Lorasc */
  int ecg_enlargedFactor = 1, ecg_maxIter = 10000;
  double ecg_tol = 1e-6;


  // Start MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_dup(MPI_COMM_WORLD, &comm);

  // Let me know who I am
  MPI_Comm_size(comm, &nbprocs);
  MPI_Comm_rank(comm, &my_rank);

#ifdef DEBUG
  printf("I am proc %d over %d processors\n", my_rank, nbprocs);
#endif

  /* Get user parameters */
  for(i=1;i<argc;i+=2){
    if (strcmp(argv[i],"-m") == 0) strcpy(matrix_filename,argv[i+1]);
    if (strcmp(argv[i],"-r") == 0) strcpy(rhs_filename,argv[i+1]);
    if (strcmp(argv[i],"-t") == 0) ecg_enlargedFactor = atoi(argv[i+1]);
    if (strcmp(argv[i],"-p") == 0) precond_num = atoi(argv[i+1]);
    if (strcmp(argv[i],"-h") == 0){
      if(my_rank==0){
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
        printf(" -p: preconditioner \n");
        printf("     0: no prec, 1: blockJacobi, 2: Lorasc\n");

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
    default:
      preAlps_abort("Unknown preconditioner");
  }

  /*
   * Load the matrix on proc 0
   */

  if(my_rank==0){

    if(strlen(matrix_filename)==0){
      preAlps_abort("Error: unknown Matrix. ./test_lorasc -h for usage");
    }

    printf("Matrix name: %s\nEnlarged factor: %d\n", matrix_filename, ecg_enlargedFactor);
    printf("Reading matrix ...\n");
  }

  if(my_rank==root){


    CPLM_LoadMatrixMarket(matrix_filename, &A);

    /* Get the local dimension of A*/
    preAlps_nsplit(A.info.m, nbprocs, my_rank, &mloc, &offsetloc);

    if(doScale){

      /*Scale the matrix*/
      double *R, *C;

      if ( !(R  = (double *) malloc(A.info.m * sizeof(double))) ) preAlps_abort("Malloc fails for R[].");
      if ( !(C  = (double *) malloc(A.info.n * sizeof(double))) ) preAlps_abort("Malloc fails for C[].");

      CPLM_MatCSRSymRACScaling(&A, R, C);

      free(R);
      free(C);
    }

    #ifdef BUILDING_MATRICES_DUMP
      printf("Dumping the matrix ...\n");
      CPLM_MatCSRSave(&A, "dump_AScaled.mtx");
      printf("Dumping the matrix ... done\n");
    #endif

    CPLM_MatCSRPrintCoords(&A, "Scaled matrix");

    //CPLM_MatCSRPrintInfo(&A);
    //CPLM_MatCSRPrintf2D("Loaded matrix", &A);

    CPLM_MatCSRPrintCoords(&A, "Loaded matrix");
  }

  // Broadcast the global matrix dimension from the root to the other procs
  preAlps_matrixDim_Bcast(comm, &A, root, &m, &n, &nnz);

  // Load the Right-hand side
  if ( !(b  = (double *) malloc(m*sizeof(double))) ) preAlps_abort("Malloc fails for b[].");

  if(my_rank==root){
    if(strlen(rhs_filename)==0){
      // Generate a random vector
      srand(11);
      for(int i=0;i<m;i++)
        b[i] = ((double) rand() / (double) RAND_MAX);
    }else{
      // Read rhs on proc 0
      preAlps_doubleVector_load(rhs_filename, &rhs, &rhs_size);

      if(rhs_size!=A.info.m){
        preAlps_abort("Error: the matrix and rhs size does not match. Matrix size: %d x %d, rhs size: %d", A.info.m, A.info.n, rhs_size);
      }
      for(i=0;i<rhs_size;i++) b[i] = rhs[i];
      free(rhs);
    }
  }



  /* Build the selected preconditionner */

  if(precond_type==PREALPS_LORASC){

    ttemp =  MPI_Wtime();

    /* Memory allocation for lorasc preconditioner */
    ierr =  preAlps_LorascAlloc(&lorascA); preAlps_checkError(ierr);

    /* Set parameters for the preconditioners */
    lorascA->deflation_tol = 1e-2;

    /* Change the nrhs before building lorasc (required for analysis by internal solvers and memory allocation ) */
    lorascA->nrhs = ecg_enlargedFactor;

    /* Build the preconditioner and distribute the matrix */
    ierr = preAlps_LorascBuild(lorascA, &A, &locAP, comm); preAlps_checkError(ierr);

    tPrec = MPI_Wtime() - ttemp;

    if(my_rank==root) printf("Schur-complement size: %d\n", lorascA->sep_nrows);

  }else{

    ttemp =  MPI_Wtime();
    /* permute only the matrix using lorasc, do not build the preconditioner */

    // Memory allocation for lorasc preconditioner
    ierr =  preAlps_LorascAlloc(&lorascA); preAlps_checkError(ierr);

    //permute only the matrix using lorasc, do not build the preconditioner
    lorascA->OptPermuteOnly = 1;

    // permute and distribute the matrix using lorasc
    ierr = preAlps_LorascBuild(lorascA, &A, &locAP, comm); preAlps_checkError(ierr);

    tPartition = MPI_Wtime() - ttemp;
  }

  /* Prepare the operator */

  if(lorascA) preAlps_OperatorBuildNoPerm(&locAP, lorascA->partBegin, 1, comm);

  #ifdef MATMULT_GATHERV //DEBUG ONLY
    if(my_rank==0) printf("[DEBUG] ***** MATMULT GATHERV DEBUG **** \n");
    double *vs = (double*) malloc(m*enlargedFactor*sizeof(double));
  #endif


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


  // Broadcast the rhs to all the processors
  MPI_Bcast(b, m, MPI_DOUBLE, root, comm);


  /* Solve the system */

  ECG_t ecg;
  // Set parameters
  ecg.comm       = comm;              /* MPI Communicator */
  ecg.globPbSize = m;                 /* Size of the global problem */
  ecg.locPbSize  = locAP.info.m;      /* Size of the local problem */
  ecg.maxIter    = ecg_maxIter;       /* Maximum number of iterations */
  ecg.enlFac     = ecg_enlargedFactor;/* Enlarging factor */
  ecg.tol        = ecg_tol;           /* Tolerance of the method */
  ecg.ortho_alg  = ORTHOMIN;          /* Orthogonalization algorithm */

  int rci_request = 0;
  int stop = 0;
  double* sol = NULL;
  sol = (double*) malloc(m*sizeof(double));

  ttemp = MPI_Wtime();
  // Allocate memory and initialize variables
  preAlps_ECGInitialize(&ecg, b, &rci_request);
  // Finish initialization
  if(precond_type==PREALPS_BLOCKJACOBI) preAlps_BlockJacobiApply(ecg.R,ecg.P);
  else preAlps_PreconditionerMatApply(precond, ecg.R, ecg.P);

  #ifdef MATMULT_GATHERV
    //Algatherv AP
    MPI_Allgatherv(ecg.P->val, ecg.P->info.nval, MPI_DOUBLE, vs, lorascA->partCount, lorascA->partBegin, MPI_DOUBLE, comm);
    CPLM_MatCSRMatrixCSRDenseMult(&locAP, 1.0, vs, ecg.P->info.n, m, 0.0, ecg.AP->val, ecg.AP->info.lda);
  #else
    preAlps_BlockOperator(ecg.P, ecg.AP);
  #endif

  // Main loop
  while (stop != 1) {

    preAlps_ECGIterate(&ecg, &rci_request);

    if (rci_request == 0) {

      #ifdef MATMULT_GATHERV
        MPI_Allgatherv(ecg.P->val, ecg.P->info.nval, MPI_DOUBLE, vs, lorascA->partCount, lorascA->partBegin, MPI_DOUBLE, comm);
        CPLM_MatCSRMatrixCSRDenseMult(&locAP, 1.0, vs, ecg.P->info.n, m, 0.0, ecg.AP->val, ecg.AP->info.lda);
      #else
        preAlps_BlockOperator(ecg.P, ecg.AP);
      #endif

    }
    else if (rci_request == 1) {

      preAlps_ECGStoppingCriterion(&ecg, &stop);

      if (stop == 1) break;

      if (ecg.ortho_alg == ORTHOMIN){

        if(precond_type==PREALPS_BLOCKJACOBI) preAlps_BlockJacobiApply(ecg.R,ecg.Z);
        else preAlps_PreconditionerMatApply(precond, ecg.R, ecg.Z);

      }
      else if (ecg.ortho_alg == ORTHODIR){

        if(precond_type==PREALPS_BLOCKJACOBI) preAlps_BlockJacobiApply(ecg.AP,ecg.Z);
        else preAlps_PreconditionerMatApply(precond, ecg.AP, ecg.Z);

      }

    }

    if(monitorResidual && my_rank==root) printf("Iteration: %d, \tres: %e\n", ecg.iter, ecg.res);
  }

  // Retrieve solution and free memory
  preAlps_ECGFinalize(&ecg, sol);

  tSolve = MPI_Wtime() - ttemp;

  if (my_rank == 0)
    printf("=== ECG ===\n\titerations: %d\n\tnorm(res): %e\n",ecg.iter,ecg.res);

  tTotal = tPartition + tPrec + tSolve;

  preAlps_dstats_display(comm, tPartition, "Time partitioning");
  preAlps_dstats_display(comm, tPrec, "Time preconditioner");
  preAlps_dstats_display(comm, tSolve, "Time Solve");
  preAlps_dstats_display(comm, tTotal, "Time Total");

  free(sol);

  #ifdef MATMULT_GATHERV
    free(vs);
  #endif

  //Free memory
  preAlps_OperatorFree();

  //Destroy the preconditioner/the partitioning
  //if(precond_type==PREALPS_LORASC){
  ierr =  preAlps_LorascDestroy(&lorascA); preAlps_checkError(ierr);
  //}

  // Destroy the generic preconditioner object
  preAlps_PreconditionerDestroy(&precond);

  if(b) free(b);
  CPLM_MatCSRFree(&locAP);
  if(my_rank==0){
    CPLM_MatCSRFree(&A);
  }

  MPI_Finalize();
  return EXIT_SUCCESS;
}
