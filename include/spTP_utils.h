#ifndef SPTP_UTILS_H
#define SPTP_UTILS_H
#include <string.h>
#include <mpi.h>
#include "SuiteSparseQR_C.h"
void preAlps_TP_parameters_display(MPI_Comm comm, char **matrixName, int *k, int ordering, int *printSVal, int *checkFact,
  int *printFact, int argc, char **argv);

void preAlps_CSC_to_cholmod_l_sparse(int m, int n, int nnz, int *colPtr, int *rowInd, double *a, cholmod_sparse **A, cholmod_common *cc);

void preAlps_l_CSC_to_cholmod_l_sparse(long m, long n, long nnz, long *colPtr, long *rowInd, double *a, cholmod_sparse **A, cholmod_common *cc);

void preAlps_spTP_distribution(MPI_Comm comm, int *m, int *n, int *nnz, int **colPtr, int **rowInd,
  double **a, long *col_offset, int checkFact);


#endif
