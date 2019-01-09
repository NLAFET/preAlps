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





int CPLM_MatDenseKernelCholesky(CPLM_Mat_Dense_t* C_io)
{
CPLM_PUSH
CPLM_BEGIN_TIME
  int ierr          = 0;
  int matrix_layout = 0;
  int order         = 0;
  int lda           = 0;
  char uplo         = 0;

  CPLM_ASSERT(C_io       !=  NULL);
  CPLM_ASSERT(C_io->val  !=  NULL);

  matrix_layout = (C_io->info.stor_type == ROW_MAJOR) ? LAPACK_ROW_MAJOR
    : LAPACK_COL_MAJOR;
  uplo          = 'U';
  order         = C_io->info.n;
  lda           = C_io->info.lda;

  ierr = LAPACKE_dpotrf(  matrix_layout,
                          uplo,
                          order,
                          C_io->val,
                          lda);CPLM_CHKERR(ierr);
  if (ierr > 0)
  {
    CPLM_Abort("The matrix is not SPD!");
  }

CPLM_END_TIME
CPLM_POP
  return ierr;
}




int CPLM_MatDenseKernelLowerTriangularLeftSolve(CPLM_Mat_Dense_t* L, CPLM_Mat_Dense_t* B)
{
CPLM_PUSH
CPLM_BEGIN_TIME
  int ierr          = 0;
  int matrix_layout = 0;
  int m             = 0;
  int nrhs          = 0;
  char uplo  = 'L'; // L is lower triangular
  char trans = 'N'; // No transpose, no conjugacy
  char diag  = 'N'; // No unitary

  CPLM_ASSERT(L      !=  NULL);
  CPLM_ASSERT(L->val !=  NULL);
  CPLM_ASSERT(B      !=  NULL);
  CPLM_ASSERT(B->val !=  NULL);

  matrix_layout = (L->info.stor_type == ROW_MAJOR) ? LAPACK_ROW_MAJOR
    : LAPACK_COL_MAJOR;
  m             = B->info.m;
  nrhs          = B->info.n;

  ierr = LAPACKE_dtrtrs(matrix_layout,
                        uplo,
                        trans,
                        diag,
                        m,
                        nrhs,
                        L->val,
                        L->info.lda,
                        B->val,
                        B->info.lda);CPLM_CHKERR(ierr);
CPLM_END_TIME
CPLM_POP
  return ierr;
}





int CPLM_MatDenseKernelUpperTriangularLeftSolve(CPLM_Mat_Dense_t* R, CPLM_Mat_Dense_t* B)
{
CPLM_PUSH
CPLM_BEGIN_TIME
  int ierr          = 0;
  int matrix_layout = -1;
  int m             = 0;
  int nrhs          = 0;
  char uplo  = 'U'; // R is upper triangular
  char trans = 'N'; // No transpose, no conjugacy
  char diag  = 'N'; // No unitary

  CPLM_ASSERT(R      !=  NULL);
  CPLM_ASSERT(R->val !=  NULL);
  CPLM_ASSERT(B      !=  NULL);
  CPLM_ASSERT(B->val !=  NULL);

  matrix_layout = (R->info.stor_type == ROW_MAJOR) ? LAPACK_ROW_MAJOR
    : LAPACK_COL_MAJOR;
  m             = B->info.m;
  nrhs          = B->info.n;

  ierr = LAPACKE_dtrtrs(matrix_layout,
                        uplo,
                        trans,
                        diag,
                        m,
                        nrhs,
                        R->val,
                        R->info.lda,
                        B->val,
                        B->info.lda);CPLM_CHKERR(ierr);
CPLM_END_TIME
CPLM_POP
  return ierr;
}





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





//#ifdef MKLACTIVATE

//B is triangular
int CPLM_MatDenseDtrmm(CPLM_Mat_Dense_t *A_in ,
                  CPLM_Mat_Dense_t *B_io ,
                  char        side  ,
                  char        uplo  ,
                  char        trans ,
                  char        diag)
{
CPLM_PUSH
  int ierr          = 0;
  CBLAS_LAYOUT matrix_layout = -1;
  CBLAS_DIAG  blasDiag = CblasNonUnit;
  CBLAS_SIDE blasSide = CblasLeft;
  CBLAS_UPLO blasUPLO = CblasUpper;
  CBLAS_TRANSPOSE blasTrans = CblasTrans;

  CPLM_ASSERT(A_in->val != NULL);
  CPLM_ASSERT(B_io->val != NULL);

  blasDiag = (diag == 'U') ? CblasUnit : CblasNonUnit;
  blasSide = (side == 'R') ? CblasRight : CblasLeft;
  blasUPLO = (uplo == 'L') ? CblasLower : CblasUpper;
  blasTrans = (trans == 'T') ? CblasTrans : CblasNoTrans;

  matrix_layout = (A_in->info.stor_type == ROW_MAJOR) ? CblasRowMajor
    : CblasColMajor;

  cblas_dtrmm(matrix_layout,
      blasSide,
      blasUPLO,
      blasTrans,
      blasDiag,
      A_in->info.m,
      A_in->info.n,
      1.0,
      A_in->val,
      A_in->info.lda,
      B_io->val,
      B_io->info.lda);
CPLM_POP
  return ierr;
}
//#else
//  #ifdef LAPACKEACTIVATE
//  int CPLM_MatDenseDtrmm(CPLM_Mat_Dense_t *A_in ,
//                    CPLM_Mat_Dense_t *B_io ,
//                    char        side  ,
//                    char        uplo  ,
//                    char        trans ,
//                    char        diag)
//  {
//  CPLM_PUSH
//    int ierr = 0;
//
//    CPLM_ASSERT(A_in->val != NULL);
//    CPLM_ASSERT(B_io->val != NULL);
//
//    if(A_in->info.stor_type == ROW_MAJOR)
//    {
//      CPLM_Abort("Do not call trmm with row major");
//    }
//
//    dtrmm(
//        side,
//        uplo,
//        trans,
//        diag,
//        A_in->info.m,
//        A_in->info.n,
//        1.0,
//        A_in->val,
//        A_in->info.lda,
//        B_io->val,
//        B_io->info.lda);
//  CPLM_POP
//    return ierr;
//  }
//  #endif
//#endif




// C=alpha A^{transa} * B^{transb} + beta C
int CPLM_MatDenseKernelMatMult(CPLM_Mat_Dense_t*      A,
                          char              ptransa,
                          CPLM_Mat_Dense_t*      B,
                          char              ptransb,
                          CPLM_Mat_Dense_t*      C,
                          double            alpha,
                          double            beta)
{
CPLM_PUSH
CPLM_BEGIN_TIME
  int ierr = 0;
  CBLAS_TRANSPOSE transa;
  CBLAS_TRANSPOSE transb;
  // Allocate memory if needed
  if (C->val == NULL)
  {
    int Crow, Ccol, CrowGlob, CcolGlob;
    CrowGlob = (ptransa == 'N') ? A->info.M : A->info.N;
    Crow     = (ptransa == 'N') ? A->info.m : A->info.n;
    CcolGlob = (ptransb == 'N') ? B->info.N : B->info.M;
    Ccol     = (ptransb == 'N') ? B->info.n : B->info.m;
    ierr = CPLM_MatDenseSetInfo(C,
                           CrowGlob,
                           CcolGlob,
                           Crow,
                           Ccol,
                           A->info.stor_type); // We use A storage type but this
                                               // is arbitrary
    CPLM_CHKERR(ierr);
    ierr = CPLM_MatDenseMalloc(C);
    CPLM_CHKERR(ierr);
  }

  transa = (ptransa == 'N') ?  CblasNoTrans : CblasTrans;
  transb = (ptransb == 'N') ?  CblasNoTrans : CblasTrans;

  // BLAS parameters
  CBLAS_LAYOUT layout = (A->info.stor_type == ROW_MAJOR) ? CblasRowMajor
    : CblasColMajor;
  int nbColOpA = (transa == CblasNoTrans) ? A->info.n : A->info.m;

  cblas_dgemm (layout,
               transa,
               transb,
               C->info.m,
               C->info.n,
               nbColOpA,
               alpha,
               A->val,
               A->info.lda,
               B->val,
               B->info.lda,
               beta,
               C->val,
               C->info.lda);CPLM_CHKERR(ierr);

CPLM_END_TIME
CPLM_POP
  return ierr;

}





int CPLM_MatDenseKernelUpperTriangularRightSolve(CPLM_Mat_Dense_t* R, CPLM_Mat_Dense_t* B)
{
CPLM_PUSH
CPLM_BEGIN_TIME
  int ierr = 0;
  CBLAS_LAYOUT matrix_layout = -1;
#ifdef MKLACTIVATE
  MKL_INT m    = 0;
  MKL_INT nrhs = 0;
#elif defined(LAPACKEACTIVATE)
  int m    = 0;
  int nrhs = 0;
#endif
  double alpha = 1e0;

  CPLM_ASSERT(R      !=  NULL);
  CPLM_ASSERT(R->val !=  NULL);
  CPLM_ASSERT(B      !=  NULL);
  CPLM_ASSERT(B->val !=  NULL);

  matrix_layout = (R->info.stor_type == ROW_MAJOR) ? CblasRowMajor
    : CblasColMajor;
  m             = B->info.m;
  nrhs          = B->info.n;

  cblas_dtrsm(matrix_layout,
              CblasRight,
              CblasUpper,
              CblasNoTrans,
              CblasNonUnit,
              m,
              nrhs,
              alpha,
              R->val,
              R->info.lda,
              B->val,
              B->info.lda);
CPLM_END_TIME
CPLM_POP
  return ierr;
}





// y = beta*y + alpha*A*x
int CPLM_MatDenseKernelMatVec(CPLM_Mat_Dense_t* A_in,
                              double*           x_in,
                              double*           y_io,
                              double            alpha,
                              double            beta)
{
CPLM_PUSH
CPLM_BEGIN_TIME
  int ierr = 0;
  CBLAS_LAYOUT layout = -1;

  CPLM_ASSERT(A_in       !=  NULL);
  CPLM_ASSERT(x_in       !=  NULL);
  CPLM_ASSERT(y_io       !=  NULL);
  CPLM_ASSERT(A_in->val  !=  NULL);

  /* if (*y_io == NULL) */
  /* { */
  /*   y_io = malloc(A_in->info.m*sizeof(double)); */
  /* } */

  // BLAS parameters
  layout = (A_in->info.stor_type == ROW_MAJOR) ? CblasRowMajor : CblasColMajor;

  cblas_dgemv(layout,
              CblasNoTrans,
              A_in->info.m,
              A_in->info.n,
              alpha,
              A_in->val,
              A_in->info.lda,
              x_in,
              1,             // increment for the elements of x
              beta,
              y_io,
              1);            // increment of the elements of y
CPLM_END_TIME
CPLM_POP
  return ierr;
}





int CPLM_MatDenseKernelSumColumns(CPLM_Mat_Dense_t* A_in, double* sumCol)
{
CPLM_PUSH
CPLM_BEGIN_TIME
  int ierr = 0;
  CPLM_ASSERT(A_in->val != NULL);
  CPLM_ASSERT(sumCol != NULL);

  double* ones = (double*) malloc(A_in->info.n*sizeof(double));
  for (int i = 0; i < A_in->info.n; ++i)
    ones[i] = 1.E0;

  ierr = CPLM_MatDenseKernelMatVec(A_in, ones, sumCol, 1.0, 0.0);CPLM_CHKERR(ierr);

  if (ones != NULL) free(ones);
CPLM_END_TIME
CPLM_POP
  return ierr;
}





#ifdef MKLACTIVATE
// C = A + B using mkl
int CPLM_MatDenseKernelMatAdd(CPLM_Mat_Dense_t* A_in,
                         CPLM_Mat_Dense_t* B_in,
                         CPLM_Mat_Dense_t* C_out,
                         double       alpha,
                         double       beta)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  int ierr = 0;
  char ordering = 0;
  char transa   = 0;
  char transb   = 0;

  if(!CPLM_MatDenseIsSameLocalInfo(A_in,B_in))
  {
    CPLM_MatDensePrintfInfo("A info",A_in);
    CPLM_MatDensePrintfInfo("B info",B_in);
    CPLM_Abort("A and B do not have the same structure");
  }

  // Allocate memory if needed
  if(C_out->val == NULL)
  {
    ierr = CPLM_MatDenseInit(C_out, A_in->info);CPLM_CHKERR(ierr);
    ierr = CPLM_MatDenseMalloc(C_out);CPLM_CHKERR(ierr);
  }

  // BLAS parameters
  ordering = (A_in->info.stor_type == ROW_MAJOR) ? 'R' : 'C';
  transa = 'N';
  transb = 'N';

  mkl_domatadd (ordering,
                transa,
                transb,
                A_in->info.m,
                A_in->info.n,
                alpha,
                A_in->val,
                A_in->info.lda,
                beta,
                B_in->val,
                B_in->info.lda,
                C_out->val,
                C_out->info.lda);
CPLM_END_TIME
CPLM_POP
  return ierr;
}





int CPLM_MatCSRKernelMatDenseMult(CPLM_Mat_CSR_t    *A_in,
                             CPLM_Mat_Dense_t  *B_in,
                             CPLM_Mat_Dense_t  *C_io,
                             double       alpha,
                             double       beta)
{
CPLM_PUSH
CPLM_BEGIN_TIME
CPLM_OPEN_TIMER
  int ierr = 0;
  char matdescra[6];
  char trans  = 'N';

  CPLM_ASSERT(A_in       !=  NULL);
  CPLM_ASSERT(B_in       !=  NULL);
  CPLM_ASSERT(C_io       !=  NULL);
  CPLM_ASSERT(A_in->val  != NULL);
  CPLM_ASSERT(B_in->val  != NULL);

  matdescra[0]  = 'G';
  matdescra[1]  = 'L';//ignored if G
  matdescra[2]  = 'N';//ignored if G
  matdescra[3]  = 'C';

  if(C_io->val == NULL)
  {
    ierr = CPLM_MatDenseSetInfo(C_io,
                          A_in->info.m,
                          B_in->info.n,
                          A_in->info.m,
                          B_in->info.n,
                          B_in->info.stor_type);CPLM_CHKERR(ierr);
    ierr = CPLM_MatDenseMalloc(C_io);CPLM_CHKERR(ierr);
  }

#ifdef DEBUG
  CPLM_MatCSRPrintf2D("M1",A_in);
  CPLM_MatDensePrintf2D("* M2",B_in);
#endif
CPLM_TIC(step1, "ConvertTo1BasedIndexing")
  // If COL_MAJOR we need to use 1-base indexes (don't ask why...)
  if (B_in->info.stor_type == COL_MAJOR)
  {
    matdescra[3]  = 'F';
    CPLM_MatCSRConvertTo1BasedIndexing(A_in);
  }
CPLM_TAC(step1)
CPLM_TIC(step2, "Call SpBLAS")
  mkl_dcsrmm(&trans,
             &A_in->info.m,
             &B_in->info.n,
             &A_in->info.n,
             &alpha,
             matdescra,
             A_in->val,
             A_in->colInd,
             A_in->rowPtr,
             &(A_in->rowPtr[1]),
             B_in->val,
             &(B_in->info.lda),
             &beta,
             C_io->val,
             &C_io->info.lda);
CPLM_TAC(step2)
  // Back to 0-base indexes :)
CPLM_TIC(step3, "ConvertTo0BasedIndexing")
  if (B_in->info.stor_type == COL_MAJOR)
  {
    CPLM_MatCSRConvertTo0BasedIndexing(A_in);
  }
CPLM_TAC(step3)

#ifdef DEBUG
  CPLM_MatDensePrintf2D("= ",C_io);
#endif

CPLM_CLOSE_TIMER
CPLM_END_TIME
CPLM_POP
  return ierr;
}




int CPLM_MatCSRKernelGenMatDenseMult(double      *val_in,
                                int         *colInd_in,
                                int         *rowPtrB_in,
                                int         *rowPtrE_in,
                                int          nrowA_in,
                                int          ncolA_in,
                                CPLM_Mat_Dense_t *B_in,
                                CPLM_Mat_Dense_t *C_io,
                                double       alpha,
                                double       beta)
{
CPLM_PUSH
CPLM_BEGIN_TIME
CPLM_OPEN_TIMER
  int ierr = 0;
  char matdescra[6];

  matdescra[0]  = 'G';
  matdescra[1]  = 'L';//ignored if G
  matdescra[2]  = 'N';//ignored if G
  char trans  = 'N';

  CPLM_ASSERT(C_io->val != NULL);

  if (B_in->info.stor_type == COL_MAJOR)
    matdescra[3] = 'F';
  else
    matdescra[3]  = 'C';

CPLM_TIC(step1, "Call SpBLAS")
  mkl_dcsrmm(&trans,
             &nrowA_in,
             &B_in->info.n,
             &ncolA_in,
             &alpha,
             matdescra,
             val_in,
             colInd_in,
             rowPtrB_in,
             rowPtrE_in,
             B_in->val,
             &B_in->info.lda,
             &beta,
             C_io->val,
             &C_io->info.lda);
CPLM_TAC(step1)

CPLM_CLOSE_TIMER
CPLM_END_TIME
CPLM_POP
  return ierr;
}





void CPLM_MatCSRPARDISOSetParameters(MKL_INT* iparam_io)
{
CPLM_PUSH
  CPLM_ASSERT(iparam_io != NULL);

  memset(iparam_io,0,64*sizeof(MKL_INT));

  iparam_io[0]  = 1; // Non standard solver
  iparam_io[1]  = 2; // Metis permutation to reduce fill-in
  iparam_io[4]  = 0; // Return Metis permutation
  iparam_io[9] = 13; // Pivot perturbation
  iparam_io[24] = 0; // Sequential forward and backward solve
  iparam_io[26] = 0; // Check A
  iparam_io[34] = 1; // C-style array indexing (starts from 0)
  //  iparam_io[17] = -1;
  //  iparam_io[18] = -1;
CPLM_POP
}





int CPLM_MatCSRPARDISOFree(CPLM_Mat_CSR_t*  A_in,
                      _MKL_DSS_HANDLE_t       pardisoHandle_io,
                      MKL_INT*    iparam_in,
                      MKL_INT     mtype_in)
{
CPLM_PUSH
  MKL_INT maxfct  = 1;
  MKL_INT mnum    = 1;
  MKL_INT phase   = -1;
  MKL_INT n       = A_in->info.m;
  MKL_INT nrhs    = 0;
  MKL_INT msglvl  = 0;
  MKL_INT error   = 0;
  MKL_INT iDummy  = -1;
  double dDummy = -1e0;

  pardiso(pardisoHandle_io,
          &maxfct,
          &mnum,
          &mtype_in,
          &phase,
          &n,
          &dDummy,
          A_in->rowPtr,
          A_in->colInd,
          &iDummy,
          &nrhs,
          iparam_in,
          &msglvl,
          &dDummy, // b: not needed here
          &dDummy, // x: not needed here
          &error);

CPLM_POP
  return error;
}





int CPLM_MatCSRPARDISOFactorization( CPLM_Mat_CSR_t*  A_in,
                                _MKL_DSS_HANDLE_t       pardisoHandle_out,
                                MKL_INT*    iparam_in,
                                MKL_INT     mtype_in,
                                MKL_INT*    perm_out)
{
CPLM_PUSH
CPLM_BEGIN_TIME
  MKL_INT maxfct  = 1;
  MKL_INT mnum    = 1;
  MKL_INT phase   = 12;
  MKL_INT n       = A_in->info.m;
  MKL_INT nrhs    = 0;
  MKL_INT msglvl  = 0;
  MKL_INT error   = 0;
  double dDummy = -1e0;

  CPLM_ASSERT(A_in               !=  NULL);
  CPLM_ASSERT(A_in->val          !=  NULL);
  CPLM_ASSERT(pardisoHandle_out  !=  NULL);
  CPLM_ASSERT(iparam_in          !=  NULL);
  //  CPLM_ASSERT(perm_out           !=  NULL);

  pardiso(pardisoHandle_out,
          &maxfct,
          &mnum,
          &mtype_in,
          &phase,
          &n,
          A_in->val,
          A_in->rowPtr,
          A_in->colInd,
          perm_out, // the permutation returned by PARDISO
          &nrhs,
          iparam_in,
          &msglvl,
          &dDummy,  // b: not needed here
          &dDummy,  // x: not needed here
          &error);

CPLM_END_TIME
CPLM_POP
  return error;
}





int CPLM_MatCSRPARDISOGeneralSolve(CPLM_Mat_CSR_t*    A_in,
                              CPLM_Mat_Dense_t*  B_in,
                              CPLM_Mat_Dense_t*  X_out,
                              MKL_INT       phase,
                              _MKL_DSS_HANDLE_t         pardisoHandle_in,
                              MKL_INT*      iparam_in,
                              MKL_INT       mtype_in,
                              MKL_INT*      perm_in)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  int ierr = 0;
  MKL_INT maxfct  = 1;
  MKL_INT mnum    = 1;
  MKL_INT n       = 0;
  MKL_INT nrhs    = 0;
  MKL_INT msglvl  = 0;
  MKL_INT error   = 0;

  CPLM_ASSERT(A_in               !=  NULL);
  CPLM_ASSERT(B_in               !=  NULL);
  CPLM_ASSERT(X_out              !=  NULL);
  CPLM_ASSERT(A_in->val          !=  NULL);
  CPLM_ASSERT(B_in->val          !=  NULL);
  CPLM_ASSERT(pardisoHandle_in   !=  NULL);
  CPLM_ASSERT(iparam_in          !=  NULL);
  //  CPLM_ASSERT(perm_in            !=  NULL);

  if (X_out->val == NULL)
  {
    X_out->info = B_in->info;
    ierr = CPLM_MatDenseMalloc(X_out);CPLM_CHKERR(ierr);
  }
  else if (!CPLM_MatDenseIsSameLocalInfo(X_out,B_in))
  {
    X_out->info = B_in->info;
    ierr = CPLM_MatDenseRealloc(X_out);CPLM_CHKERR(ierr);
  }

  n       = (MKL_INT) A_in->info.m;
  nrhs    = (MKL_INT) B_in->info.n;

  pardiso(pardisoHandle_in,
          &maxfct,
          &mnum,
          &mtype_in,
          &phase,
          &n,
          A_in->val,
          A_in->rowPtr,
          A_in->colInd,
          perm_in,
          &nrhs,
          iparam_in,
          &msglvl,
          B_in->val,
          X_out->val,
          &error);

CPLM_END_TIME
CPLM_POP
  return error || ierr;
}





int CPLM_MatCSRPARDISOSolve( CPLM_Mat_CSR_t*    A_in,
                        CPLM_Mat_Dense_t*  B_in,
                        CPLM_Mat_Dense_t*  X_out,
                        _MKL_DSS_HANDLE_t         pardisoHandle_in,
                        MKL_INT*      iparam_in,
                        MKL_INT       mtype_in,
                        MKL_INT*      perm_in)
{
CPLM_PUSH
  MKL_INT phase = 33;
  int ierr = CPLM_MatCSRPARDISOGeneralSolve( A_in,
                                        B_in,
                                        X_out,
                                        phase,
                                        pardisoHandle_in,
                                        iparam_in,
                                        mtype_in,
                                        perm_in);
CPLM_POP
  return ierr;
}





int CPLM_MatCSRPARDISOSolveForward(CPLM_Mat_CSR_t*    A_in,
                              CPLM_Mat_Dense_t*  B_in,
                              CPLM_Mat_Dense_t*  X_out,
                              _MKL_DSS_HANDLE_t         pardisoHandle_in,
                              MKL_INT*      iparam_in,
                              MKL_INT       mtype_in,
                              MKL_INT*      perm_in)
{
CPLM_PUSH
  MKL_INT phase = 331;
  int ierr = CPLM_MatCSRPARDISOGeneralSolve( A_in,
                                        B_in,
                                        X_out,
                                        phase,
                                        pardisoHandle_in,
                                        iparam_in,
                                        mtype_in,
                                        perm_in);
  if (ierr != 0)
  {
    CPLM_Abort("PARDISO Solve forward");
  }
CPLM_POP
  return ierr;
}





int CPLM_MatCSRPARDISOSolveBackward( CPLM_Mat_CSR_t*    A_in,
                                CPLM_Mat_Dense_t*  B_in,
                                CPLM_Mat_Dense_t*  X_out,
                                _MKL_DSS_HANDLE_t         pardisoHandle_in,
                                MKL_INT*      iparam_in,
                                MKL_INT       mtype_in,
                                MKL_INT*      perm_in)
{
CPLM_PUSH
  MKL_INT phase = 333;
  int ierr = CPLM_MatCSRPARDISOGeneralSolve( A_in,
                                        B_in,
                                        X_out,
                                        phase,
                                        pardisoHandle_in,
                                        iparam_in,
                                        mtype_in,
                                        perm_in);
  if (ierr != 0)
  {
    CPLM_Abort("PARDISO Solve backward");
  }
CPLM_POP
  return ierr;
}
#endif





int CPLM_MatDenseNorm(CPLM_Mat_Dense_t *A_in, const char type, double *norm)
{
  CPLM_PUSH
  int ierr = 0;
  int major  = (A_in->info.stor_type == COL_MAJOR) ? LAPACK_COL_MAJOR
    : LAPACK_ROW_MAJOR;

  *norm = LAPACKE_dlange(major,
      type,
      A_in->info.m,
      A_in->info.n,
      A_in->val,
      A_in->info.lda);

  CPLM_POP
  return ierr;
}

int CPLM_MatDenseMatDotProd(CPLM_Mat_Dense_t* A, CPLM_Mat_Dense_t* B, CPLM_Mat_Dense_t* C, MPI_Comm comm)
{
CPLM_PUSH
CPLM_BEGIN_TIME
  int ierr = 0;
  // Do local dot product
  ierr = CPLM_MatDenseKernelMatDotProd(A, B, C);

  // Sum local dot products in place (no mem alloc needed)
  ierr = MPI_Allreduce(MPI_IN_PLACE, C->val, C->info.nval, MPI_DOUBLE, MPI_SUM, comm);CPLM_checkMPIERR(ierr,"MatDenseMatDotProd::MPI_Allreduce");
CPLM_END_TIME
CPLM_POP
  return ierr;
}

/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/*                                  HANDLER                                   */
/******************************************************************************/
/*
void CPLM_MatDenseSVDHandler(int ierr)
{

  if (ierr > 0)
  {
    CPLM_eprintf("the eigensolver did not converge!");
  }
  else if (ierr < 0)
  {
    CPLM_eprintf("parameter %d has an illegal value!", -ierr + 1);
  }

}
*/
