/*
 ======================================================================================
 Name        : test_prototypeCUR.c
 Author      : Alan Ayala
 Version     : 0.1
 Description : Performs a CUR factorization using tournament pivoting on a sparse matrix.
               In adition, it gets an approximation of the singular values.
 Date        : Sept 27, 2016
 ======================================================================================
 */

#include "preAlps_matrix.h"
#include "tournamentPivoting.h"
#include "spTP_utils.h"


int main(int argc, char **argv){

  /* Global variables */
  double *Sval; // vector of singular values
  int k; // rank of approximation
  int rank,size;
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);


  /* Initialize variables (can be donde by stdin too) */
  int ordering = 0; //  if set to 1 -> Uses Metis ordering.
  int printSVal = 0; // if set to 1 -> Prints the singular values into a file
  int checkFact = 0; // if set to 1 -> Checks the CUR factorization error (infinty norm)
  int printFact = 0; // if set to 1 -> Prints the CUR factors (matrix U and vectors Jr, Jc)
  char *matrixName = NULL;


  /* Reading parameters from stdin */
  preAlps_TP_parameters_display(&matrixName,&k,ordering, &printSVal,&checkFact,&printFact,argc,argv);

/* Reading matrix in Master processor */
  int *row_indx = NULL, *col_indx = NULL;
  double *a = NULL;
  int m=0, n=0, nnz=0;

  if(rank ==0){
  preAlps_matrix_readmm_csc(matrixName, &m, &n, &nnz, &row_indx, &col_indx, &a);
  }

  free(matrixName);

  /* Alocate memory for the vectors Jc, Jr and the singular values */
  ASSERT(k>0);
  long *Jc,*Jr;
  Jr  = malloc(sizeof(long)*k);
  if(rank == 0) {
    Jc  = malloc(sizeof(long)*k);
    if(printSVal) Sval  = malloc(sizeof(long)*k);
  }

  /* Call tournamentPivotingCUR and get Jc, Jr and the singular values (if required) */
  double t_begin, t_tp;
  t_begin=MPI_Wtime();
  preAlps_tournamentPivotingCUR(row_indx,col_indx,a,m,n,nnz,k,Jr,Jc,&Sval,printSVal,checkFact,printFact,ordering);
  t_tp = MPI_Wtime()-t_begin;


  /* Print the results */
if(rank==0) {
  printf("Time for tournamentPivotingCUR = %f \n",  t_tp );

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


  if(printFact){
    FILE *Jrf,*Jcf;
    Jrf = fopen ("Jr.txt", "w");
    Jcf = fopen ("Jc.txt", "w");
    printf("Vector of selected columns written in Jc.txt \n");
    printf("Vector of selected rows written in Jr.txt \n");
    for (int i = 0; i < k; i++) {
       fprintf(Jrf, "%ld\n", Jr[i]+1); // vectors in matlab format
       fprintf(Jcf, "%ld\n", Jc[i]+1);
    }
    fclose(Jrf);
    fclose(Jcf);
    free(Jc);
    free(Jr);
  }


}


  MPI_Finalize();

  return 0;
}
