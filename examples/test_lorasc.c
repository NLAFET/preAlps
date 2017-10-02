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
//#include "cgsolver.h"

/**/
int main(int argc, char** argv){

  MPI_Comm comm;
  int nbprocs, my_rank;
  char matrix_filename[150]="", rhs_filename[150]="";
  CPLM_Mat_CSR_t A = CPLM_MatCSRNULL();
  CPLM_Mat_CSR_t locAP = CPLM_MatCSRNULL();
  int i, ierr   = 0;

  double *x = NULL, *b = NULL;
  int b_size = 0;


  /* Generic preconditioner type and object */
  Prec_Type_t precond_type = PREALPS_LORASC;
  PreAlps_preconditioner_t *precond = NULL;

  /* Lorasc preconditioner */
  Lorasc_t *lorascA = NULL;


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

    CPLM_LoadMatrixMarket(matrix_filename, &A);
    CPLM_MatCSRPrintInfo(&A);
    //CPLM_MatCSRPrintf2D("Loaded matrix", &A);

    CPLM_MatCSRPrintCoords(&A, "Loaded matrix");


    if(strlen(rhs_filename)==0){
      /*Generate a random rhs*/
      //preAlps_abort("Random rhs not yet implemented"); /* TODO */
      CPLM_DVector_t rhs = CPLM_DVectorNULL();
      CPLM_DVectorMalloc(&rhs, A.info.m);
      CPLM_DVectorRandom(&rhs, 11);
      b = rhs.val;
      b_size = rhs.nval;
    }else{

      /* Read rhs */
      preAlps_doubleVector_load(rhs_filename, &b, &b_size);

    }

    printf("Rhs size:%d\n", b_size);

    //for(i=0;i<b_size;i++) printf("b[%d]: %f\n", i, b[i]);

    preAlps_doubleVector_printSynchronized(b, b_size, "b", "rhs", MPI_COMM_SELF);

    if(b_size!=A.info.n){

      preAlps_abort("Error: The matrix and rhs size does not match. Matrix size: %d x %d, rhs size: %d", A.info.m, A.info.n, b_size);

    }
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

  if(precond_type==PREALPS_NOPREC){

    if(my_rank==0) printf("Preconditioner: NONE\n");



  }else if(precond_type==PREALPS_LORASC){

    if(my_rank==0) printf("Preconditioner: LORASC\n");

    /* Memory allocation for the preconditioner */
    ierr =  Lorasc_alloc(&lorascA); preAlps_checkError(ierr);

    /* Set parameters for the preconditioners */
    lorascA->deflation_tolerance = 1e-2;

    /* Build the preconditioner */
    Lorasc_build(lorascA, &A, &locAP, comm);

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

  if (my_rank == 0)
    printf("=== ECG ===\n\titerations: %d\n\tnorm(res): %e\n",ecg.iter,ecg.res);


  if(precond_type==PREALPS_LORASC){
    /* Destroy Lorasc preconditioner */
    ierr =  Lorasc_destroy(&lorascA); preAlps_checkError(ierr);
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
