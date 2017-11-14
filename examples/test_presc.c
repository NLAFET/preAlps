/*
============================================================================
Name        : test_presc.c
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
#include "presc.h"
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
  int *partBegin=NULL, *perm = NULL;

  double ttemp, tPartition =0.0, tPrec = 0.0, tSolve = 0.0, tTotal;

  /* Required by block Jacobi*/
  int* rowPos = NULL;
  int* colPos = NULL;
  int sizeRowPos, sizeColPos;

  // Generic preconditioner type and object
  Prec_Type_t precond_type = PREALPS_PRESC; //PREALPS_NOPREC, PREALPS_BLOCKJACOBI
  PreAlps_preconditioner_t *precond = NULL;


  // Presc preconditioner
  preAlps_Presc_t *prescA = NULL;

  /* Program default parameters */
  int doScale = 1, monitorResidual = 0;
  int precond_num = 2; /* 0: no prec, 1: blockJacobi, 2: Presc */
  int ecg_enlargedFactor = 1, ecg_maxIter = 30000;
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
        printf(" mpirun -np <nbprocs> ./test_presc -m <matrix_file_name> -r <rhs_file_name> -t <ECG_enlargedFactor> -p <preconditioner_num>\n");
        printf("\n");
        printf(" Arguments\n");
        printf(" =========\n");
        printf(" -m: the matrix file\n");
        printf("    the matrix stored in matrix market format\n");
        printf(" -r: the right hand side file\n");
        printf("     the right hand side stored in a text file\n");
        printf(" -t: the enlarged factor (default :1)\n");
        printf(" -p: preconditioner \n");
        printf("     0: no prec, 1: blockJacobi, 2: presc\n");

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
      precond_type = PREALPS_PRESC;
      if(my_rank==root) printf("Preconditioner: PRESC\n");
    break;
    default:
      preAlps_abort("Unknown preconditioner");
  }

  /*
   * Load the matrix on proc 0
   */

  if(my_rank==0){

    if(strlen(matrix_filename)==0){
      preAlps_abort("Error: unknown Matrix. ./test_presc -h for usage");
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

  // Allocate memory for the permutation array
  if ( !(perm  = (int *) malloc(m*sizeof(int))) ) preAlps_abort("Malloc fails for perm[].");


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


  /*
   * Partition the matrix and create a block row structure where each row
   * with off-diag element is permuted to the bottom.
   */

  ttemp =  MPI_Wtime();
  int locNbDiagRows;
  preAlps_blockDiagODBStructCreate(comm, &A, &locAP, perm, &partBegin, &locNbDiagRows);
  tPartition = MPI_Wtime() - ttemp;


  /*
   * Prepare the operator
   */

  preAlps_OperatorBuildNoPerm(&locAP, partBegin, 1, comm);


  #ifdef MATMULT_GATHERV //DEBUG ONLY
    if(my_rank==0) printf("[DEBUG] ***** MATMULT GATHERV DEBUG **** \n");
    double *vs = (double*) malloc(m*enlargedFactor*sizeof(double));
  #endif


  /*
   * Build the selected preconditioner
   */

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
  }else if(precond_type==PREALPS_PRESC){

    ttemp =  MPI_Wtime();

    /* Memory allocation for presc preconditioner */
    ierr =  preAlps_PrescAlloc(&prescA); preAlps_checkError(ierr);

    /* Set parameters for the preconditioners */
    prescA->deflation_tol = 5e-3;

    /* Set parameters for the preconditioners*/
    prescA->eigs_kind = PRESC_EIGS_SALOC; //PRESC_EIGS_SSLOC, PRESC_EIGS_SALOC
    if(my_rank==0) printf("Presc eigenvalues kind: %s\n", prescA->eigs_kind==PRESC_EIGS_SSLOC?"SSLOC":"SALOC");

    /* Change the nrhs before building presc (required for analysis by internal solvers and memory allocation ) */
    prescA->nrhs = ecg_enlargedFactor;

    /* Build the preconditioner and distribute the matrix */
    ierr = preAlps_PrescBuild(prescA, &locAP, partBegin, locNbDiagRows, comm); preAlps_checkError(ierr);

    tPrec = MPI_Wtime() - ttemp;

    //if(my_rank==root) printf("Schur-complement size: %d\n", prescA->sep_nrows);

    // Create a generic preconditioner object compatible with EcgSolver
    preAlps_PreconditionerCreate(&precond, precond_type, (void *) prescA);
  }

  //if(my_rank==0) preAlps_doubleVector_gathervDump(b, A.info.m, "dump/b0.txt", MPI_COMM_SELF, "b0");

  // Broadcast the rhs to all the processors
  MPI_Bcast(b, m, MPI_DOUBLE, root, comm);


  /*
   * Solve the system
   */

  preAlps_ECG_t ecg;
  // Set parameters
  ecg.comm       = comm;              /* MPI Communicator */
  ecg.globPbSize = m;                 /* Size of the global problem */
  ecg.locPbSize  = locAP.info.m;      /* Size of the local problem */
  ecg.maxIter    = ecg_maxIter;       /* Maximum number of iterations */
  ecg.enlFac     = ecg_enlargedFactor;/* Enlarging factor */
  ecg.tol        = ecg_tol;           /* Tolerance of the method */
  ecg.ortho_alg  = ORTHODIR;          /* Orthogonalization algorithm */

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
    MPI_Allgatherv(ecg.P->val, ecg.P->info.nval, MPI_DOUBLE, vs, prescA->partCount, prescA->partBegin, MPI_DOUBLE, comm);
    CPLM_MatCSRMatrixCSRDenseMult(&locAP, 1.0, vs, ecg.P->info.n, m, 0.0, ecg.AP->val, ecg.AP->info.lda);
  #else
    preAlps_BlockOperator(ecg.P, ecg.AP);
  #endif

  // Main loop
  while (stop != 1) {

    preAlps_ECGIterate(&ecg, &rci_request);

    if (rci_request == 0) {

      #ifdef MATMULT_GATHERV
        MPI_Allgatherv(ecg.P->val, ecg.P->info.nval, MPI_DOUBLE, vs, prescA->partCount, prescA->partBegin, MPI_DOUBLE, comm);
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
  if(precond_type==PREALPS_PRESC){
    ierr =  preAlps_PrescDestroy(&prescA); preAlps_checkError(ierr);
  }

  // Destroy the generic preconditioner object
  preAlps_PreconditionerDestroy(&precond);

  if(b) free(b);
  if(partBegin) free(partBegin);
  if(perm) free(perm);

  CPLM_MatCSRFree(&locAP);
  if(my_rank==0){
    CPLM_MatCSRFree(&A);
  }

  MPI_Finalize();
  return EXIT_SUCCESS;
}
