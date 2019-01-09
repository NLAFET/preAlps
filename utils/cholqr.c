/*
* This file contains functions used to compute variants of Cholesky QR factorization
*
* Authors : Sebastien Cayrols
*         : Olivier Tissot
* Email   : sebastien.cayrols@[(gmail.com) | (inria.fr)]
*         : olivier.tissot@inria.fr
*/

/******************************************************************************/
/*                                  INCLUDE                                   */
/******************************************************************************/

#include <stdio.h>
#include <math.h>
/* MKL */
#include <mkl.h>
/* CPaLAMeM */
//#include <cpalamem_macro.h>
#undef TIMERACTIVATE
//#include "cholqr.h"
#include <cplm_utils.h>
#include <cplm_timing.h>
#include <cplm_matdense.h>
#include <cplm_kernels.h>
//#include <matmult.h>
//#include <cpalamem_instrumentation.h>
/******************************************************************************/

/******************************************************************************/
/*                                    CODE                                    */
/******************************************************************************/


int CPLM_MatDenseANormalize(CPLM_Mat_Dense_t* P, CPLM_Mat_Dense_t* AP, CPLM_Mat_Dense_t* work, MPI_Comm comm)
{
CPLM_PUSH
CPLM_BEGIN_TIME
  int ierr = 0;
  // PAP = P^t*AP
  ierr = CPLM_MatDenseMatDotProd(AP, P, work, comm);

  // Renormalize P
  int loop_index_1, loop_index_2;
  if (P->info.stor_type == ROW_MAJOR)
  {
    loop_index_1 = P->info.n;
    loop_index_2 = 1;
  }
  else
  {
    loop_index_1 = 1;
    loop_index_2 = P->info.m;
  }
  for (int j = 0; j < work->info.n; ++j)
  {
    work->val[j*(work->info.lda) + j] = sqrt(work->val[j*(work->info.lda) + j]);
    for (int i = 0; i < P->info.m; ++i)
    {
      P->val[i*loop_index_1 + j*loop_index_2]  /= work->val[j*(work->info.lda) + j];
      AP->val[i*loop_index_1 + j*loop_index_2] /= work->val[j*(work->info.lda) + j];
    }
  }

CPLM_END_TIME
CPLM_POP
  return ierr;
}





/**
 * \fn int CPLM_MatDenseACholQR(CPLM_Mat_Dense_t* P,
 *                    CPLM_Mat_CSR_t* AP);
 */
int CPLM_MatDenseACholQR(CPLM_Mat_Dense_t* P, CPLM_Mat_Dense_t* AP, CPLM_Mat_Dense_t* work, MPI_Comm comm)
{
CPLM_PUSH
CPLM_BEGIN_TIME
CPLM_OPEN_TIMER
  int ierr;
  // C = P^t*AP
CPLM_TIC(step1, "dotprod")
  ierr = CPLM_MatDenseMatDotProd(AP, P, work, comm);
CPLM_TAC(step1)
  // Cholesky of C: R^tR = C
CPLM_TIC(step2, "cholesky")
  ierr = CPLM_MatDenseKernelCholesky(work);
CPLM_TAC(step2)
  // Solve triangular right system for P
CPLM_TIC(step3, "P trisol")
  ierr = CPLM_MatDenseKernelUpperTriangularRightSolve(work, P);
CPLM_TAC(step3)
  // Solve triangular right system for AP
CPLM_TIC(step4, "AP trisol")
  ierr = CPLM_MatDenseKernelUpperTriangularRightSolve(work, AP);
CPLM_TAC(step4)
  // A-normalize P and AP
CPLM_TIC(step5, "anorm")
  ierr = CPLM_MatDenseANormalize(P,AP,work,comm);
CPLM_TAC(step5)
CPLM_CLOSE_TIMER
CPLM_END_TIME
CPLM_POP
  return ierr;
}





/**
 * \fn int CPLM_MatDenseCholQR(CPLM_Mat_Dense_t* P);
 */
int CPLM_MatDenseCholQR(CPLM_Mat_Dense_t* P, CPLM_Mat_Dense_t *R, MPI_Comm comm)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  int ierr = 0;
  // C = P^t*P
  ierr = CPLM_MatDenseMatDotProd(P, P, R, comm);
  // Cholesky of C: R^tR = C
  ierr = CPLM_MatDenseKernelCholesky(R);
  // Solve triangular right system for P
  ierr = CPLM_MatDenseKernelUpperTriangularRightSolve(R, P);

  // Remove lower part of R
  ierr = CPLM_MatDenseGetRInplace(R);CPLM_CHKERR(ierr);

CPLM_END_TIME
CPLM_POP
  return ierr;
}




int MatDenseCholRQTest(int rank)
{
CPLM_PUSH
CPLM_BEGIN_TIME
  int ierr = 0;
  int size;
  CPLM_Mat_Dense_t P = CPLM_MatDenseNULL();
  CPLM_Mat_Dense_t R = CPLM_MatDenseNULL();

  MPI_Comm comm;
  MPI_Comm_dup(MPI_COMM_WORLD,&comm);

  if (rank == 0)
    printf("::: Testing MatDenseCholQR :::\n");

  // CholQR test
  MPI_Comm_size(comm, &size);
  printf("  [%d] MatDenseCholQR::Construction of P...\n", rank);

  CPLM_MatDenseSetInfo(&P, 8, 4, 4, 4, ROW_MAJOR);
  CPLM_MatDenseConstant(&P,1);

  printf("  [%d] MatDenseCholQR::P is constructed!\n", rank);
  CPLM_MatDensePrintf2D("P",&P);
  printf("  [%d] MatDenseCholQR::Orthonormalization of P...\n", rank);

  ierr = CPLM_MatDenseCholQR(&P, &R, comm);

  if (ierr != 0)
    printf("  [%d] MatDenseCholQR::Error!\n", rank);
  else
    printf("  [%d] MatDenseCholQR::Passed!\n", rank);
  printf("  [%d] MatDenseCholQR::Solution obtained:\n", rank);
  CPLM_MatDensePrint2D(&P);

  MPI_Barrier(comm);

  if (rank == 0)
  {
    printf("::: End testing A_Ortho :::\n");
  }

  CPLM_MatDenseFree(&P);
  CPLM_MatDenseFree(&R);
CPLM_END_TIME
CPLM_POP
  return ierr;
}

/******************************************************************************/
