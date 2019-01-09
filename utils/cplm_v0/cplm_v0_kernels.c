/*
* This file contains functions used to call sequential MKL/BLAS/LAPACK routines
*
* Authors : Sebastien Cayrols
*         : Olivier Tissot
* Email   : sebastien.cayrols@[(gmail.com) | (inria.fr)]
*         : olivier.tissot@inria.fr
*/

/******************************************************************************/
/*                                  INCLUDE                                   */
/******************************************************************************/
/* STD */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
/* MKL/LAPACK */
#ifdef LAPACKEACTIVATE
  #include <lapacke.h>
  #include <cblas.h>
#endif
/* MPI */
#include <mpi.h>
/* CPaLAMeM */
//#include "cpalamem_macro.h"
#undef TIMERACTIVATE
#include "cplm_utils.h"
#include "cplm_v0_timing.h"
#include "cplm_matcsr.h"
#include "cplm_v0_kernels.h"
#include "cplm_QS.h"
//#include "cpalamem_instrumentation.h"
/******************************************************************************/

/******************************************************************************/
/*                                    CODE                                    */
/******************************************************************************/

/*
int CPLM_MatDenseSVD(CPLM_Mat_Dense_t* A_in,
                CPLM_Mat_Dense_t* U_out,
                CPLM_DVector_t*   S_out,
                CPLM_Mat_Dense_t* Vt_out,
                double** work,
                size_t* workSize)
{
CPLM_PUSH
CPLM_BEGIN_TIME
  int ierr          = 0;
  int matrix_layout = 0;
  int m             = 0;
  int n             = 0;
  int minmn         = -1;
  char jobu         = 0;
  char jobvt        = 0;

  CPLM_ASSERT(A_in      != NULL);
  CPLM_ASSERT(A_in->val != NULL);
  m = A_in->info.m;
  n = A_in->info.n;

  if (U_out->val == NULL)
  {
    ierr = CPLM_MatDenseSetInfo(U_out,m,m,m,m,A_in->info.stor_type);CPLM_CHKERR(ierr);
  //ierr = 1 + CPLM_MatDenseSetInfo(U_out,m,m,m,m,A_in->info.stor_type);CPLM_CHKERR(ierr);
    ierr = CPLM_MatDenseMalloc(U_out);CPLM_CHKERR(ierr);
  }
  if (Vt_out->val == NULL)
  {
    ierr = CPLM_MatDenseSetInfo(Vt_out,n,n,n,n,A_in->info.stor_type);CPLM_CHKERR(ierr);
    ierr = CPLM_MatDenseMalloc(Vt_out);CPLM_CHKERR(ierr);
  }
  if (S_out->val == NULL)
  {
    ierr = CPLM_DVectorMalloc(S_out,n);CPLM_CHKERR(ierr);
  }

  CPLM_ASSERT(U_out->info.m  == m && U_out->info.n  == m);
  CPLM_ASSERT(Vt_out->info.m == n && Vt_out->info.n == n);
  CPLM_ASSERT(S_out->nval == n);

  minmn = CPLM_MIN(m,n);

  if (*work == NULL)
  {
    *work = (double*) malloc((minmn - 1)*sizeof(double));
    *workSize = minmn - 1;
  }
  else if (*workSize < minmn - 1)
  {
    *work = realloc(*work,(minmn - 1)*sizeof(double));
    *workSize = minmn - 1;
  }

  matrix_layout = (A_in->info.stor_type == ROW_MAJOR) ? LAPACK_ROW_MAJOR
    : LAPACK_COL_MAJOR;
  jobu  = 'A'; // All the columns of U
  jobvt = 'A'; // All the columns of V^t

  ierr = LAPACKE_dgesvd(matrix_layout,
                        jobu,
                        jobvt,
                        m,
                        n,
                        A_in->val,
                        A_in->info.lda,
                        S_out->val,
                        U_out->val,
                        U_out->info.lda,
                        Vt_out->val,
                        Vt_out->info.lda,
                        *work);

CPLM_HANDLER(CPLM_MatDenseSVDHandler)
CPLM_END_TIME
CPLM_POP
  return ierr;
}
*/


//TODO Remove malloc and explain why H
int CPLM_MatDenseKernelQR( CPLM_Mat_Dense_t*  A_io,
                      CPLM_Mat_Dense_t*  H_io,
                      int           index_i,
                      int           index_j) //to be verified
{
CPLM_PUSH
CPLM_BEGIN_TIME
  int ierr          = 0;
  int matrix_layout = -1;
  CPLM_DVector_t tau = CPLM_DVectorNULL();

  tau.nval = A_io->info.N;
  ierr = CPLM_DVectorMalloc(&tau,A_io->info.N);CPLM_CHKERR(ierr);

  matrix_layout = (A_io->info.stor_type == ROW_MAJOR) ? LAPACK_ROW_MAJOR
    : LAPACK_COL_MAJOR;

  ierr = LAPACKE_dgeqrf(  matrix_layout,
                          A_io->info.M,
                          A_io->info.N,
                          A_io->val,
                          A_io->info.lda,
                          tau.val);CPLM_CHKERR(ierr);

  if(H_io->val != NULL)//[Q] why test on val and not on H alone?
  {
    ierr = CPLM_MatDenseTriangularFillBlock( A_io,
                                        H_io,
                                        index_i,
                                        index_j,
                                        A_io->info.N);CPLM_CHKERR(ierr);
  }

  ierr = LAPACKE_dorgqr(  matrix_layout,
                          A_io->info.M,
                          A_io->info.N,
                          A_io->info.N,
                          A_io->val,
                          A_io->info.lda,
                          tau.val);CPLM_CHKERR(ierr);
  CPLM_DVectorFree(&tau);
CPLM_END_TIME
CPLM_POP
  return ierr;
}
