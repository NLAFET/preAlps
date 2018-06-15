/*
* This file contains functions used to manipulate ij val tuple
*
* Authors : Sebastien Cayrols
*         : Remi Lacroix
* Email   : sebastien.cayrols@[(gmail.com) | (inria.fr)]
*/


#include <mpi.h>
#include "preAlps_cplm_ijval.h"

int CPLM_CompareIJVal(const void* e1, const void* e2)
{
  const CPLM_IJVal_t *ijval1 = (const CPLM_IJVal_t*)e1;
  const CPLM_IJVal_t *ijval2 = (const CPLM_IJVal_t*)e2;

  if (ijval1->i > ijval2->i) {
    return 1;
  } else if (ijval1->i < ijval2->i) {
    return -1;
  } else if (ijval1->j > ijval2->j) {
    return 1;
  } else if (ijval1->j < ijval2->j) {/*Previous version with error } else if (ijval1->j > ijval2->j) { ADDED by sebastien cayrols Date 10/14/2015*/
    return -1;
  }

  return 0;
}

/*
* This function compares 2 ijval structures but used for column only
*/
int CPLM_CompareIJValCol(const void* e1, const void* e2)
{
  const CPLM_IJVal_t *ijval1 = (const CPLM_IJVal_t*)e1;
  const CPLM_IJVal_t *ijval2 = (const CPLM_IJVal_t*)e2;

  if (ijval1->j > ijval2->j) {
    return 1;
  } else if (ijval1->j < ijval2->j) {
    return -1;
  } else if (ijval1->i > ijval2->i) {
    return 1;
  } else if (ijval1->i < ijval2->i) {
    return -1;
  }

  return 0;
}

void CPLM_RegisterIJValStruct(MPI_Datatype *type)
{
  int i;
  MPI_Datatype blocktype[3] = {MPI_INT, MPI_INT, MPI_DOUBLE};
  int blocklen[3] = {1, 1, 1};
  MPI_Aint disp[3];
  MPI_Aint base;
  CPLM_IJVal_t ijval;

  MPI_Get_address(&ijval.i, &disp[0]);
  MPI_Get_address(&ijval.j, &disp[1]);
  MPI_Get_address(&ijval.val, &disp[2]);
  base = disp[0];
  for (i = 0; i < 3; i++) {
    disp[i] -= base;
  }

  MPI_Type_struct(3, blocklen, disp, blocktype, type);

  MPI_Type_commit(type);
}
