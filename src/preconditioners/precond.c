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
                  Mat_CSR_t* A,
                  int* rowPos,
                  int sizeRowPos,
                  int* colPos,
                  int sizeColPos,
                  int* dep,
                  int sizeDep)
{

PUSH
BEGIN_TIME
  int ierr = 0;

  if(precond_type==PREALPS_BLOCKJACOBI){
    ierr = BlockJacobiCreate(A,
                             rowPos,
                             sizeRowPos,
                             colPos,
                             sizeColPos,
                             dep,
                             sizeDep); CHKERR(ierr);
  }
  else
    CPALAMEM_Abort("Unknown preconditioner: %d", precond_type);

END_TIME
POP
  return ierr;

}

int PrecondApply(Prec_Type_t precond_type, Mat_Dense_t* A_in, Mat_Dense_t* B_out)
{

PUSH
BEGIN_TIME
  int ierr = 0;

  if(precond_type==PREALPS_BLOCKJACOBI){
    ierr = BlockJacobiApply(A_in, B_out); CHKERR(ierr);
  }
  else
    CPALAMEM_Abort("Unknown preconditioner: %d", precond_type);

END_TIME
POP
  return ierr;

}

void PrecondFree(Prec_Type_t precond_type)
{

PUSH
BEGIN_TIME
  if(precond_type==PREALPS_BLOCKJACOBI){
    BlockJacobiFree();
  }
  else
    CPALAMEM_Abort("Unknown preconditioner: %d", precond_type);
END_TIME
POP
}

/******************************************************************************/
