#ifndef SPTP_UTILS_H
#define SPTP_UTILS_H
#include <string.h>
#include <mpi.h>
#include "SuiteSparseQR_C.h"
void preAlps_TP_parameters_display( char **matrixName, int *k, int ordering, int *printSVal, int *checkFact,
  int *printFact, int argc, char **argv);
void preAlps_TP_matrix_distribute(int rank, int size, int *row_indx, int *col_indx, double *a, int m, int n, int nnz,
  long *col_offset, cholmod_sparse **A, int checkFact, cholmod_sparse **A_test, cholmod_common *cc);
#endif
