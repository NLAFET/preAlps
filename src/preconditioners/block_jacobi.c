/******************************************************************************/
/* Author     : Olivier Tissot                                                */
/* Creation   : 2016/09/08                                                    */
/* Description: Block Jacobi preconditioner                                   */
/******************************************************************************/


/******************************************************************************/
/*                                  INCLUDE                                   */
/******************************************************************************/
#include "preAlps_cplm_utils.h"
#include "block_jacobi.h"
/******************************************************************************/

/******************************************************************************/
/*                              GLOBAL VARIABLES                              */
/******************************************************************************/
static MKL_INT pardiso_pt_g[64] = {0};
static MKL_INT iparam_g[64] = {0};
static CPLM_Mat_CSR_t Adiag_g = CPLM_MatCSRNULL();
/******************************************************************************/

/******************************************************************************/
/*                                    CODE                                    */
/******************************************************************************/
int preAlps_BlockJacobiCreate(CPLM_Mat_CSR_t* A,
                              int* rowPos,
                              int sizeRowPos,
                              int* colPos,
                              int sizeColPos)
{
CPLM_PUSH
  int size, rank, ierr;
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  CPLM_ASSERT(pardiso_pt_g != NULL);
  CPLM_ASSERT(iparam_g != NULL);

  int mtype = 2, perm = 1;
  CPLM_IVector_t rowPos_s = CPLM_IVectorNULL();
  CPLM_IVector_t colPos_s = CPLM_IVectorNULL();
  CPLM_IVectorCreateFromPtr(&rowPos_s,sizeRowPos,rowPos);
  CPLM_IVectorCreateFromPtr(&colPos_s,sizeColPos,colPos);
  // Construct Cholesky of the diagonal block
  CPLM_MatCSRPARDISOSetParameters(iparam_g);
  iparam_g[4] = 0;
  // Get the diag block
  ierr = CPLM_MatCSRGetDiagBlock(A,
                            &Adiag_g,
                            &rowPos_s,
                            &colPos_s,
                            SYMMETRIC);CPLM_CHKERR(ierr);
  // Call Pardiso
  ierr = CPLM_MatCSRPARDISOFactorization(&Adiag_g,
                                    pardiso_pt_g,
                                    iparam_g,
                                    mtype,
                                    &perm);
  if (ierr != 0)
    CPLM_Abort("PARDISO Cholesky error: %d",ierr);
CPLM_POP
  return ierr;
}

int preAlps_BlockJacobiInitialize(CPLM_DVector_t* rhs)
{
CPLM_PUSH
  int ierr;
  int mtype = 2, perm = 1; // SPD matrix
  CPLM_Mat_Dense_t sol = CPLM_MatDenseNULL();
  // Dirty...
  CPLM_Mat_Dense_t rhs_s = CPLM_MatDenseNULL();
  CPLM_MatDenseSetInfo(&rhs_s, rhs->nval, 1, rhs->nval, 1, COL_MAJOR);
  rhs_s.val = rhs->val;
  CPLM_MatDenseSetInfo(&sol, rhs->nval, 1, rhs->nval, 1, COL_MAJOR);
  CPLM_MatDenseMalloc(&sol);

  // PARDISO solution in place
  iparam_g[5] = 1;
  ierr = CPLM_MatCSRPARDISOSolve(&Adiag_g,
                                 &rhs_s,
                                 &sol,
                                 pardiso_pt_g,
                                 iparam_g,
                                 mtype,
                                 &perm);
  iparam_g[5] = 0;
  CPLM_MatDenseFree(&sol);
CPLM_POP
  return ierr;
}

int preAlps_BlockJacobiApply(CPLM_Mat_Dense_t* A_in, CPLM_Mat_Dense_t* B_out)
{
CPLM_PUSH
  int ierr = 0, mtype = 2, perm = 1;
  CPLM_ASSERT(A_in->val != NULL);
  CPLM_ASSERT(B_out->val != NULL);
  ierr = CPLM_MatCSRPARDISOSolve(&Adiag_g,
                                 A_in,
                                 B_out,
                                 pardiso_pt_g,
                                 iparam_g,
                                 mtype,
                                 &perm);
CPLM_POP
  return ierr;

}


void preAlps_BlockJacobiFree()
{
CPLM_PUSH
  MKL_INT mtype = 2; // SPD matrix
  CPLM_MatCSRPARDISOFree(&Adiag_g,pardiso_pt_g,iparam_g,mtype);
  CPLM_MatCSRFree(&Adiag_g);
CPLM_POP
}

/******************************************************************************/
