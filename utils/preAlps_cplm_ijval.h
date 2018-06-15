#ifndef PREALPS_CPLM_IJVAL_H
#define PREALPS_CPLM_IJVAL_H
/*
* This file contains functions used to manipulate ij val tuple
*
* Authors : Sebastien Cayrols
*         : Remi Lacroix
* Email   : sebastien.cayrols@[(gmail.com) | (inria.fr)]
*/

#include <mpi.h>

typedef struct {
  int i, j;
  double val;
} CPLM_IJVal_t;

int CPLM_CompareIJVal(const void* e1, const void* e2);

int CPLM_CompareIJValCol(const void* e1, const void* e2);

#ifdef MPIACTIVATE
  void CPLM_RegisterIJValStruct(MPI_Datatype *type);
#endif

#endif
