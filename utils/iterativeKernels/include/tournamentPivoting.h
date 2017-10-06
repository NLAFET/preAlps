#ifndef TOURNAMENTPIVOTING_H
#define TOURNAMENTPIVOTING_H

#include <stdio.h>
#include <stdlib.h>
#include "SuiteSparseQR_C.h"
#include <string.h>
#include <math.h>
#include <mpi.h>

#ifdef MKL
  #include <mkl.h>
#else
  #include <lapacke.h>
#endif

#define ASSERT(t_) if(!(t_)) printf("WARNING %s, line %d\n",__FILE__,__LINE__)
#ifndef MAX
#define MAX(a_,b_) ((a_)>(b_)?(a_):(b_))
#endif
#define DBL_EPSILON 1e-15


int preAlps_tournamentPivoting(MPI_Comm comm, int *colPtr, int *rowInd, double *a, int m,  int n,  int nnz, long col_offset, int k, long *Jc,
   double *Sval, int printSVal, int ordering);

int preAlps_tournamentPivotingQR(MPI_Comm comm, int *colPtr, int *rowInd, double *a, int m,  int n,  int nnz,
  long col_offset, int k, long *Jc, double *Sval, int printSVal, int checkFact, int printFact, int ordering);


int preAlps_tournamentPivotingCUR(MPI_Comm comm, int *colPtr, int *rowInd, double *a, int m,  int n,  int nnz,
    long col_offset, int k, long *Jr, long *Jc, double *Sval, int printSVal, int checkFact, int printFact, int ordering);

#endif
