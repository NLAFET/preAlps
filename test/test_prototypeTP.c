/*
 ==========================================================================================
 Name        : test_prototypeTP.c
 Author      : Alan Ayala
 Version     : 0.1
 Description : Performs tournament pivoting on a sparse matrix and gets its approximated
               singular values.
 Date        : Oct 21, 2016
 ==========================================================================================
 */

#include "preAlps_matrix.h"
#include "tournamentPivoting.h"
#include "spTP_utils.h"


int main(int argc, char **argv){

  /* Global variables */
  double *Sval; ; // vector of singular values
  int k; // rank of approximation
  int rank,size;
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  MPI_Comm comm;
  MPI_Comm_dup(MPI_COMM_WORLD, &comm);

  /* Initialize variables (can be donde by stdin too) */
  int ordering = 0; //  if set to 1 -> Uses Metis ordering
  int printSVal = 0; // if set to 1 -> Prints the singular values into a file
  int checkFact = 0; // set to 0 for this example (None factorization is performed)
  int printFact = 0; // set to 0 for this example (None factorization is performed)
  char *matrixName = NULL;


  /* Reading parameters from stdin */
  preAlps_TP_parameters_display(comm,&matrixName,&k,ordering,&printSVal,&checkFact,&printFact,argc,argv);

  /* Reading matrix in Master processor */
  int *xa = NULL, *ia = NULL;
  double *a = NULL;
  int m=0, n=0, nnz=0;

  if(rank ==0){
    preAlps_matrix_readmm_csc(matrixName, &m, &n, &nnz, &xa, &ia, &a);
  }
  free(matrixName);



/* Distribute the matrix among all processors */
  long col_offset=0;
  preAlps_spTP_distribution(comm,&m, &n, &nnz, &xa, &ia, &a, &col_offset, checkFact);


  /* Alocate memory for the vectors Jc and Sval to be used in the tournament pivoting scheme */
  ASSERT(k>0);
  long *Jc;
  if(rank == 0) {
    Jc  = malloc(sizeof(long)*k);
    if(printSVal) Sval  = malloc(sizeof(long)*k);
  }

  /* Call tournamentPivoting and get Jc and the singular values (if required) */
  double t_begin, t_tp;
  t_begin=MPI_Wtime();
  preAlps_tournamentPivoting(comm,xa,ia,a,m,n,nnz,col_offset,k,Jc,&Sval,printSVal,ordering);
  t_tp = MPI_Wtime()-t_begin;


  /* Print the results */
if(rank==0) {
  printf("Time for tournamentPivoting = %f \n", t_tp );

  FILE *Jcf;
  Jcf = fopen ("Jc.txt", "w");
  printf("Vector of selected columns written in Jc.txt \n");
  for (int i = 0; i < k; i++)  fprintf(Jcf, "%ld\n", Jc[i]+1);
  fclose(Jcf);
  free(Jc);

  if(printSVal) {
    FILE *svalues;
    svalues = fopen ("svalues.txt", "w");

    for (int i = 0; i < k; i++) {
       fprintf(svalues, "%d %f\n", i+1, fabs(Sval[i]));
    }
    printf("Singular values written in svalues.txt \n");
    fclose(svalues);

    free(Sval);
  }
}

free(xa);
free(ia);
free(a);

  MPI_Finalize();

  return 0;
}
