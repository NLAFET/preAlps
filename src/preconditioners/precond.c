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

//int PrecondBlockOperator(precond_t prec_type, Mat_Dense_t* A_in, Mat_Dense_t* B_out)
int PrecondBlockOperator(Precond_t precond_type, Mat_Dense_t* A_in, Mat_Dense_t* B_out)
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

/******************************************************************************/
