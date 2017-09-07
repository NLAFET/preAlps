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
#include "preconditioner.h"
#include "presc.h"
//#include "cgsolver.h"

/**/
int main(int argc, char** argv){

  MPI_Comm comm;
  int nbprocs, my_rank;
  char matrix_filename[150]="", rhs_filename[150]="";
  Mat_CSR_t A = MatCSRNULL();
  Mat_CSR_t locAP = MatCSRNULL();
  int i, ierr   = 0;

  double *x = NULL, *b = NULL;
  int b_size = 0;


  /*Preconditioner object*/
  preAlps_preconditioner_t precond = PREALPS_NOPREC;
  Presc_t *prescA;

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
        printf(" mpirun -np <nbprocs> ./test_presc -m <matrix_file_name> -r <rhs_file_name>\n");
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
      preAlps_abort("Error: unknown Matrix. Usage ./test_presc -m <matrix_file_name>");
    }

    printf("Matrix name: %s\n", matrix_filename);

    printf("Reading matrix ...\n");

    LoadMatrixMarket(matrix_filename, &A);
    MatCSRPrintInfo(&A);
    //MatCSRPrintf2D("Loaded matrix", &A);

    MatCSRPrintCoords(&A, "Loaded matrix");


    if(strlen(rhs_filename)==0){
      /*Generate a random rhs*/
      preAlps_abort("Random rhs not yet implemented"); /* TODO */

    }else{

      /* Read rhs */
      preAlps_doubleVector_load(rhs_filename, &b, &b_size);

    }

    printf("Rhs size:%d\n", b_size);

    for(i=0;i<b_size;i++) printf("b[%d]: %f\n", i, b[i]);

    preAlps_doubleVector_printSynchronized(b, b_size, "b", "rhs", MPI_COMM_SELF);

    if(b_size!=A.info.n){

      preAlps_abort("Error: The matrix and rhs size does not match. Matrix size: %d x %d, rhs size: %d", A.info.m, A.info.n, b_size);

    }
    /*Scale the matrix*/
    double *R, *C;

    if ( !(R  = (double *) malloc(A.info.m * sizeof(double))) ) preAlps_abort("Malloc fails for R[].");
    if ( !(C  = (double *) malloc(A.info.n * sizeof(double))) ) preAlps_abort("Malloc fails for C[].");

    MatCSRSymRACScaling(&A, R, C);

    free(R);
    free(C);

    #ifdef BUILDING_MATRICES_DUMP
      printf("Dumping the matrix ...\n");
      MatCSRSave(&A, "dump_AScaled.mtx");
      printf("Dumping the matrix ... done\n");
    #endif

    MatCSRPrintCoords(&A, "Scaled matrix");


  }

  if(precond==PREALPS_NOPREC){

    if(my_rank==0) printf("Preconditioner: NONE\n");



  }else{

    if(my_rank==0) printf("Preconditioner: PRESC\n");

    /* Memory allocation for the preconditioner */
    ierr =  Presc_alloc(&prescA); preAlps_checkError(ierr);

    /* Set parameters for the preconditioners */
    prescA->eigs_kind = PRESC_EIGS_SLOC; //PRESC_EIGS_SLOC

    /* Build the preconditioner */
    Presc_build(prescA, &A, &locAP, comm);


    /* Destroy the preconditioner */
    ierr =  Presc_destroy(&prescA); preAlps_checkError(ierr);

    MatCSRFree(&locAP);
  }

  /* Fill b */

  /* Solve the system */

  //cgsolver_solve(precond, NULL, &locAP, b, x);



  if(b) free(b);

  if(my_rank==0){
    MatCSRFree(&A);
  }


  MPI_Finalize();
  return EXIT_SUCCESS;
}
