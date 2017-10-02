/******************************************************************************/
/* Author     : Olivier Tissot                                                */
/* Creation   : 2016/09/08                                                    */
/* Description: Block Jacobi preconditioner                                   */
/******************************************************************************/


/******************************************************************************/
/*                                  INCLUDE                                   */
/******************************************************************************/
#include "block_jacobi.h"
/******************************************************************************/

/******************************************************************************/
/*                              GLOBAL VARIABLES                              */
/******************************************************************************/
static MKL_INT pardiso_pt_g[64] = {0};
static MKL_INT iparam_g[64] = {0};
static CPLM_Mat_CSR_t Adiag_g = MatCSRNULL();
/******************************************************************************/

/******************************************************************************/
/*                                    CODE                                    */
/******************************************************************************/
int BlockJacobiCreate(CPLM_Mat_CSR_t* A,
                      int* rowPos,
                      int sizeRowPos,
                      int* colPos,
                      int sizeColPos,
                      int* dep,
                      int sizeDep)
{
CPLM_PUSH
CPLM_BEGIN_TIME
  int size, rank, ierr;
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  CPLM_ASSERT(pardiso_pt_g != NULL);
  CPLM_ASSERT(iparam_g != NULL);

  int mtype = 2, perm = 1;
  IVector_t rowPos_s = IVectorNULL();
  IVector_t colPos_s = IVectorNULL();
  IVector_t dep_s    = IVectorNULL();
  IVectorCreateFromPtr(&rowPos_s,sizeRowPos,rowPos);
  IVectorCreateFromPtr(&colPos_s,sizeColPos,colPos);
  IVectorCreateFromPtr(&dep_s,sizeDep,dep);
  // Construct Cholesky of the diagonal block
  MatCSRPARDISOSetParameters(iparam_g);
  iparam_g[4] = 0;
  // Get the diag block
  ierr = MatCSRGetDiagBlock(A,
                            &Adiag_g,
                            &rowPos_s,
                            &colPos_s,
                            SYMMETRIC);CPLM_CHKERR(ierr);
  // Call Pardiso
  ierr = MatCSRPARDISOFactorization(&Adiag_g,
                                    pardiso_pt_g,
                                    iparam_g,
                                    mtype,
                                    &perm);
  if (ierr != 0)
    CPALAMEM_Abort("PARDISO Cholesky error: %d",ierr);
CPLM_END_TIME
CPLM_POP
  return ierr;
}

int BlockJacobiInitialize(CPLM_DVector_t* rhs) {
CPLM_PUSH
  int ierr;
  int mtype = 2, perm = 1; // SPD matrix
  CPLM_Mat_Dense_t sol = MatDenseNULL();
  // Dirty...
  CPLM_Mat_Dense_t rhs_s = MatDenseNULL();
  MatDenseSetInfo(&rhs_s, rhs->nval, 1, rhs->nval, 1, COL_MAJOR);
  rhs_s.val = rhs->val;
  MatDenseSetInfo(&sol, rhs->nval, 1, rhs->nval, 1, COL_MAJOR);
  MatDenseMalloc(&sol);

  // PARDISO solution in place
  iparam_g[5] = 1;
  ierr = MatCSRPARDISOSolve(&Adiag_g,
                            &rhs_s,
                            &sol,
                            pardiso_pt_g,
                            iparam_g,
                            mtype,
                            &perm);
  iparam_g[5] = 0;
  MatDenseFree(&sol);
CPLM_POP
  return ierr;
}

int BlockJacobiApply(CPLM_Mat_Dense_t* A_in, CPLM_Mat_Dense_t* B_out)
{
CPLM_PUSH
CPLM_BEGIN_TIME
  int ierr = 0, mtype = 2, perm = 1;
  CPLM_ASSERT(A_in->val != NULL);
  CPLM_ASSERT(B_out->val != NULL);
  ierr = MatCSRPARDISOSolve(&Adiag_g,
                            A_in,
                            B_out,
                            pardiso_pt_g,
                            iparam_g,
                            mtype,
                            &perm);
CPLM_END_TIME
CPLM_POP
  return ierr;

}


void BlockJacobiFree() {
CPLM_PUSH
  MKL_INT mtype = 2; // SPD matrix
  MatCSRPARDISOFree(&Adiag_g,pardiso_pt_g,iparam_g,mtype);
  MatCSRFree(&Adiag_g);
CPLM_POP
}

/******************************************************************************/
