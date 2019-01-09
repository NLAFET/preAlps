/*
* This file contains functions used to call sequential MKL/BLAS/LAPACK routines
*
* Authors : Sebastien Cayrols
*         : Olivier Tissot
* Email   : sebastien.cayrols@[(gmail.com) | (inria.fr)]
*         : olivier.tissot@inria.fr
*/

#ifndef CPLM_V0_KERNELS_H
#define CPLM_V0_KERNELS_H


/******************************************************************************/
/*                                  INCLUDE                                   */
/******************************************************************************/

#include "cplm_matcsr.h"
#include "cplm_matdense.h"
#include "cplm_v0_dvector.h"
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

// Function computes Q factor of Matrix A and returns it in place, the factor
// R can be copied to another matrix H starting from H(index_i,index_j) when
// H->val is allocated;
int CPLM_MatDenseKernelQR(CPLM_Mat_Dense_t* A_io,
                     CPLM_Mat_Dense_t* H_io,
                     int          index_i,
                     int          index_j); //to be verified
                     
#endif
