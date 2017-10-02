/******************************************************************************/
/* Author     : Olivier Tissot, Simplice Donfack                              */
/* Creation   : 2016/09/05                                                    */
/* Description: Definition of the preconditioner                              */
/******************************************************************************/


/******************************************************************************/
/*                                  INCLUDE                                   */
/******************************************************************************/
#include "precond.h"
#include "block_jacobi.h"
/******************************************************************************/

/******************************************************************************/
/*                                    CODE                                    */
/******************************************************************************/

int PrecondCreate(Prec_Type_t precond_type,
                  CPLM_Mat_CSR_t* A,
                  int* rowPos,
                  int sizeRowPos,
                  int* colPos,
                  int sizeColPos,
                  int* dep,
                  int sizeDep)
{

CPLM_PUSH
CPLM_BEGIN_TIME
  int ierr = 0;

  if(precond_type==PREALPS_BLOCKJACOBI){
    ierr = BlockJacobiCreate(A,
                             rowPos,
                             sizeRowPos,
                             colPos,
                             sizeColPos,
                             dep,
                             sizeDep); CPLM_CHKERR(ierr);
  }
  else
    CPALAMEM_Abort("Unknown preconditioner: %d", precond_type);

CPLM_END_TIME
CPLM_POP
  return ierr;

}

int PrecondApply(Prec_Type_t precond_type, CPLM_Mat_Dense_t* A_in, CPLM_Mat_Dense_t* B_out)
{

CPLM_PUSH
CPLM_BEGIN_TIME
  int ierr = 0;

  if(precond_type==PREALPS_BLOCKJACOBI){
    ierr = BlockJacobiApply(A_in, B_out); CPLM_CHKERR(ierr);
  }
  else
    CPALAMEM_Abort("Unknown preconditioner: %d", precond_type);

CPLM_END_TIME
CPLM_POP
  return ierr;

}

void PrecondFree(Prec_Type_t precond_type)
{

CPLM_PUSH
CPLM_BEGIN_TIME
  if(precond_type==PREALPS_BLOCKJACOBI){
    BlockJacobiFree();
  }
  else
    CPALAMEM_Abort("Unknown preconditioner: %d", precond_type);
CPLM_END_TIME
CPLM_POP
}

/******************************************************************************/
