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
#include "precond.h"
#include "lorasc.h"
#include "preAlps_preconditioner.h"
#include "ecg.h"
#include "operator.h"

#define USE_OPERATORBUILD 1

/**/
int main(int argc, char** argv){

  MPI_Comm comm;
  int nbprocs, my_rank;
  char matrix_filename[150]="", rhs_filename[150]="";
  CPLM_Mat_CSR_t A = CPLM_MatCSRNULL();
  CPLM_Mat_CSR_t locAP = CPLM_MatCSRNULL();
  int i, ierr   = 0;

  double *b = NULL;//*x = NULL,

  int m, mloc, offsetloc, b_size = 0;


  /* Generic preconditioner type and object */
  Prec_Type_t precond_type = PREALPS_LORASC;
  PreAlps_preconditioner_t *precond = NULL;

  /* Lorasc preconditioner */
  preAlps_Lorasc_t *lorascA = NULL;


  /* Start MPI*/
  MPI_Init(&argc, &argv);

  MPI_Comm_dup(MPI_COMM_WORLD, &comm);

  MPI_Comm_size(comm, &nbprocs);

  MPI_Comm_rank(comm, &my_rank);

#ifdef DEBUG
  printf("I am proc %d over %d processors\n", my_rank, nbprocs);
#endif

  /* Get user parameters */
  for(i=1;i<argc;i+=2){
    if (strcmp(argv[i],"-m") == 0) strcpy(matrix_filename,argv[i+1]);
    if (strcmp(argv[i],"-r") == 0) strcpy(rhs_filename,argv[i+1]);
    if (strcmp(argv[i],"-h") == 0){
      if(my_rank==0){
        printf(" Purpose\n");
        printf(" =======\n");
        printf(" Preconditioner based on Schur complement, \n");
        printf(" more details...\n");
        printf("\n");
        printf(" Usage\n");
        printf(" =========\n");
        printf(" mpirun -np <nbprocs> ./test_lorasc -m <matrix_file_name> -r <rhs_file_name>\n");
        printf("\n");
        printf(" Arguments\n");
        printf(" =========\n");
        printf(" -m: the matrix file\n");
        printf("    the matrix stored in matrix market format\n");
        printf(" -r: the right hand side file\n");
        printf("     the right hand side stored in a text file\n");
      }
      MPI_Finalize();
      return EXIT_SUCCESS;
    }
  }

  /*
   * Load the matrix on proc 0
   */
  if(my_rank==0){

    if(strlen(matrix_filename)==0){
      preAlps_abort("Error: unknown Matrix. ./test_lorasc -h for usage");
    }

    printf("Matrix name: %s\n", matrix_filename);

    printf("Reading matrix ...\n");
  }

#if USE_OPERATORBUILD
  /* Use Ecg built-in operator */

  /* Read and partition the matrix */
  preAlps_OperatorBuild(matrix_filename, comm);

  // Get the CSR structure of A
  preAlps_OperatorGetA(&A);

  // Get the sizes of A
  preAlps_OperatorGetSizes(&m, &mloc);


#else
  /* Read the matrix in the conventional way */

  if(my_rank==0){


    CPLM_LoadMatrixMarket(matrix_filename, &A);

    /* Get the local dimension of A*/
    preAlps_nsplit(A.info.m, nbprocs, my_rank, &mloc, &offsetloc);


    /*Scale the matrix*/
    double *R, *C;

    if ( !(R  = (double *) malloc(A.info.m * sizeof(double))) ) preAlps_abort("Malloc fails for R[].");
    if ( !(C  = (double *) malloc(A.info.n * sizeof(double))) ) preAlps_abort("Malloc fails for C[].");

    CPLM_MatCSRSymRACScaling(&A, R, C);

    free(R);
    free(C);

    #ifdef BUILDING_MATRICES_DUMP
      printf("Dumping the matrix ...\n");
      CPLM_MatCSRSave(&A, "dump_AScaled.mtx");
      printf("Dumping the matrix ... done\n");
    #endif

    CPLM_MatCSRPrintCoords(&A, "Scaled matrix");
  }
#endif


CPLM_MatCSRPrintInfo(&A);
//CPLM_MatCSRPrintf2D("Loaded matrix", &A);

CPLM_MatCSRPrintCoords(&A, "Loaded matrix");


/* Read the rhs*/
  if(strlen(rhs_filename)==0){
    /*Generate a random rhs*/
    CPLM_DVector_t rhs = CPLM_DVectorNULL();
    CPLM_DVectorMalloc(&rhs, mloc);
    CPLM_DVectorRandom(&rhs, 11);
    b = rhs.val;
    b_size = rhs.nval;
  }else{

    /* Read rhs on proc 0 and distribute */
    if(my_rank==0){
      preAlps_doubleVector_load(rhs_filename, &b, &b_size);
      printf("Rhs size:%d\n", b_size);
      preAlps_doubleVector_printSynchronized(b, b_size, "b", "rhs", MPI_COMM_SELF);
      if(b_size!=A.info.n){
        preAlps_abort("Error: The matrix and rhs size does not match. Matrix size: %d x %d, rhs size: %d", A.info.m, A.info.n, b_size);
      }
    }

    /* Distribute the rhs */


  }



  if(precond_type==PREALPS_NOPREC){

    if(my_rank==0) printf("Preconditioner: NONE\n");



  }else if(precond_type==PREALPS_LORASC){

    if(my_rank==0) printf("Preconditioner: LORASC\n");

    /* Memory allocation for the preconditioner */
    ierr =  preAlps_LorascAlloc(&lorascA); preAlps_checkError(ierr);

    /* Set parameters for the preconditioners */
    lorascA->deflation_tolerance = 1e-2;

    /* Build the preconditioner */
    preAlps_LorascBuild(lorascA, &A, &locAP, comm);

    /* Create a generic preconditioner object compatible with EcgSolver*/
    preAlps_PreconditionerCreate(&precond, precond_type, (void *) lorascA);
  }

  /* Solve the system */
  //preAlps_PreconditionerMatApply(precond, NULL, NULL);
  ECG_t ecg;
  // Set parameters
  ecg.comm = MPI_COMM_WORLD;      /* MPI Communicator */
  ecg.globPbSize = A.info.m;      /* Size of the global problem */
  ecg.locPbSize = locAP.info.m;   /* Size of the local problem */
  ecg.maxIter = 10000;            /* Maximum number of iterations */
  ecg.enlFac = 1;                 /* Enlarging factor */
  ecg.tol = 1e-8;                 /* Tolerance of the method */
  ecg.ortho_alg = ORTHOMIN;       /* Orthogonalization algorithm */
  //ECGSolve(ecg, precond, &locAP, b, x);


  int rci_request = 0;
  int stop = 0;
  double* sol = NULL;
  sol = (double*) malloc(m*sizeof(double));
  // Allocate memory and initialize variables
  preAlps_ECGInitialize(&ecg, b, &rci_request);
  // Finish initialization
  preAlps_PreconditionerMatApply(precond, ecg.R,ecg.P);
  preAlps_BlockOperator(ecg.P,ecg.AP);
  // Main loop
  while (stop != 1) {
    preAlps_ECGIterate(&ecg,&rci_request);
    if (rci_request == 0) {
      preAlps_BlockOperator(ecg.P,ecg.AP);
    }
    else if (rci_request == 1) {
      preAlps_ECGStoppingCriterion(&ecg,&stop);
      if (stop == 1) break;
      if (ecg.ortho_alg == ORTHOMIN)
        preAlps_PreconditionerMatApply(precond, ecg.R,ecg.Z);
      else if (ecg.ortho_alg == ORTHODIR)
        preAlps_PreconditionerMatApply(precond, ecg.AP,ecg.Z);
    }
  }
  // Retrieve solution and free memory
  preAlps_ECGFinalize(&ecg,sol);


  if (my_rank == 0)
    printf("=== ECG ===\n\titerations: %d\n\tnorm(res): %e\n",ecg.iter,ecg.res);

 free(sol);
#if USE_OPERATORBUILD
  preAlps_OperatorFree();
#endif

  if(precond_type==PREALPS_LORASC){
    /* Destroy Lorasc preconditioner */
    ierr =  preAlps_LorascDestroy(&lorascA); preAlps_checkError(ierr);
  }

  /* Destroy the generic preconditioner object*/
  preAlps_PreconditionerDestroy(&precond);

  if(b) free(b);

  CPLM_MatCSRFree(&locAP);

  if(my_rank==0){
    CPLM_MatCSRFree(&A);
  }


  MPI_Finalize();
  return EXIT_SUCCESS;
}
