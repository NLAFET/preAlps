/*
* This file contains functions used to call sequential MKL/BLAS/LAPACK routines
*
* Authors : Sebastien Cayrols
*         : Olivier Tissot
* Email   : sebastien.cayrols@[(gmail.com) | (inria.fr)]
*         : olivier.tissot@inria.fr
*/

#ifndef PREALPS_CPLM_KERNELS_H
#define PREALPS_CPLM_KERNELS_H


/******************************************************************************/
/*                                  INCLUDE                                   */
/******************************************************************************/

#include "preAlps_cplm_matcsr.h"
#include "preAlps_cplm_matdense.h"
#include "preAlps_cplm_dvector.h"
/* MKL */
#ifdef MKLACTIVATE
  #include <mkl.h>
#endif
/******************************************************************************/

/******************************************************************************/
/*                                    CODE                                    */
/******************************************************************************/

/*************************** MAT DENSE KERNELS ********************************/
/*
int CPLM_MatDenseSVD(CPLM_Mat_Dense_t* A_in,
                CPLM_Mat_Dense_t* U_out,
                CPLM_DVector_t*   S_out,
                CPLM_Mat_Dense_t* Vt_out,
                double** work,
                size_t* workSize);
*/
// C = L^t L (using MKL Lapack)
int CPLM_MatDenseKernelCholesky(CPLM_Mat_Dense_t* C);

// B = L^-1 B with L lower triangular matrix
int CPLM_MatDenseKernelLowerTriangularLeftSolve(CPLM_Mat_Dense_t* L, CPLM_Mat_Dense_t* B);

// B = R^-1 B with R upper triangular matrix
int CPLM_MatDenseKernelUpperTriangularLeftSolve(CPLM_Mat_Dense_t* R, CPLM_Mat_Dense_t* B);

// C = alpha*A*B + beta*C

// C=alpha A^{transa} * B^{transb} + beta C
int CPLM_MatDenseKernelMatMult(CPLM_Mat_Dense_t*  A_in,
                          char          transa, //either T for transpose or N for no transpose
                          CPLM_Mat_Dense_t*  B_in,
                          char          transb,
                          CPLM_Mat_Dense_t*  C_io,
                          double        alpha,
                          double        beta);

// C = (A^t)*B
#define CPLM_MatDenseKernelMatDotProd(_A,_B,_C) CPLM_MatDenseKernelMatMult((_A),\
                                                                 'T',\
                                                                 (_B),\
                                                                 'N',\
                                                                 (_C),\
                                                                 1.0,\
                                                                 0.0)


//B is triangular
int CPLM_MatDenseDtrmm(CPLM_Mat_Dense_t *A_in ,
                  CPLM_Mat_Dense_t *B_io ,
                  char        side  ,
                  char        uplo  ,
                  char        trans ,
                  char        diag);

  // B = B R^-1 with R upper triangular matrix
  int CPLM_MatDenseKernelUpperTriangularRightSolve(CPLM_Mat_Dense_t* R, CPLM_Mat_Dense_t* B);

  // y = beta*y + alpha*A*x
  int CPLM_MatDenseKernelMatVec(CPLM_Mat_Dense_t* A,
                                double*   x,
                                double*   y,
                                double       alpha,
                                double       beta);
  // sumCol = sum_i A^(i)
  int CPLM_MatDenseKernelSumColumns(CPLM_Mat_Dense_t* A_in, double* sumCol_out);

  // Function computes Q factor of Matrix A and returns it in place, the factor
  // R can be copied to another matrix H starting from H(index_i,index_j) when
  // H->val is allocated;
  int CPLM_MatDenseKernelQR(CPLM_Mat_Dense_t* A_io,
                       CPLM_Mat_Dense_t* H_io,
                       int          index_i,
                       int          index_j); //to be verified

#ifdef MKLACTIVATE
  // C = A + B
  int CPLM_MatDenseKernelMatAdd(CPLM_Mat_Dense_t* A_in,
                           CPLM_Mat_Dense_t* B_in,
                           CPLM_Mat_Dense_t* C_out,
                           double       alpha,
                           double       beta);

/***************************** MAT CSR KERNELS ********************************/

  // C = alpha*A*B + beta*C
  int CPLM_MatCSRKernelMatDenseMult(CPLM_Mat_CSR_t    *A_in,
                               CPLM_Mat_Dense_t  *B_in,
                               CPLM_Mat_Dense_t  *C_io,
                               double       alpha,
                               double       beta);

// Same as previous, but more general
int CPLM_MatCSRKernelGenMatDenseMult(double      *val_in,
                                int         *colInd_in,
                                int         *rowPtrB_in,
                                int         *rowPtrE_in,
                                int          nrowA_in,
                                int          ncolA_in,
                                CPLM_Mat_Dense_t *B_in,
                                CPLM_Mat_Dense_t *C_io,
                                double       alpha,
                                double       beta);

/******************************** PARDISO *************************************/

  // PARDISO interface
  void CPLM_MatCSRPARDISOSetParameters(int* iparam_io);

  int CPLM_MatCSRPARDISOFactorization(CPLM_Mat_CSR_t* A_in,
                                 _MKL_DSS_HANDLE_t      pardisoHandle_out,
                                 MKL_INT*   iparam_in,
                                 MKL_INT    mtype_in,
                                 MKL_INT*   perm_out);

  int CPLM_MatCSRPARDISOGeneralSolve(CPLM_Mat_CSR_t*    A_in,
                                CPLM_Mat_Dense_t*  B_in,
                                CPLM_Mat_Dense_t*  X_out,
                                MKL_INT       phase,
                                _MKL_DSS_HANDLE_t         pardisoHandle_out,
                                MKL_INT*      iparam_in,
                                MKL_INT       mtype_in,
                                MKL_INT*      perm_in);

  int CPLM_MatCSRPARDISOSolve(CPLM_Mat_CSR_t*     A_in,
                         CPLM_Mat_Dense_t*   B_in,
                         CPLM_Mat_Dense_t*   X_out,
                         _MKL_DSS_HANDLE_t          pardisoHandle_in,
                         MKL_INT*       param_in,
                         MKL_INT        mtype_in,
                         MKL_INT*       perm_in);

  int CPLM_MatCSRPARDISOSolveForward(CPLM_Mat_CSR_t*    A_in,
                                CPLM_Mat_Dense_t*  B_in,
                                CPLM_Mat_Dense_t*  X_out,
                                _MKL_DSS_HANDLE_t         pardisoHandle_in,
                                MKL_INT*      param_in,
                                MKL_INT       mtype_in,
                                MKL_INT*      perm_in);

  int CPLM_MatCSRPARDISOSolveBackward(CPLM_Mat_CSR_t*   A_in,
                                 CPLM_Mat_Dense_t* B_in,
                                 CPLM_Mat_Dense_t* X_in,
                                 _MKL_DSS_HANDLE_t        pardisoHandle_out,
                                 MKL_INT*     param_in,
                                 MKL_INT      mtype_in,
                                 MKL_INT*     perm_in);

  int CPLM_MatCSRPARDISOFree(CPLM_Mat_CSR_t*  A_in,
                        _MKL_DSS_HANDLE_t       pardisoHandle_io,
                        MKL_INT*    iparam_in,
                        MKL_INT     mtype_in);
  // End of PARDISO interface

#endif

/******************************************************************************/

int CPLM_MatDenseNorm(CPLM_Mat_Dense_t *A_in, const char type, double *norm);

int CPLM_MatDenseMatDotProd(CPLM_Mat_Dense_t* A, CPLM_Mat_Dense_t* B, CPLM_Mat_Dense_t* C, MPI_Comm comm);

/******************************************************************************/
/**************************    HANDLER      ***********************************/
/******************************************************************************/
/*
void CPLM_MatDenseSVDHandler(int ierr);
*/
#endif
