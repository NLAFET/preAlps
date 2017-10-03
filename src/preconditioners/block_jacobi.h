/******************************************************************************/
/* Author     : Olivier Tissot                                                */
/* Creation   : 2016/09/08                                                    */
/* Description: Block Jacobi preconditioner                                   */
/******************************************************************************/


/******************************************************************************/
/*                                  INCLUDE                                   */
/******************************************************************************/
#ifndef BLOCK_JACOBI_H
#define BLOCK_JACOBI_H

/* STD */
#include <stdio.h>
#include <stdlib.h>
/* MPI */
#include <mpi.h>
/* CPaLAMeM */
#include <cpalamem_macro.h>
#include <cpalamem_instrumentation.h>
#include <mat_csr.h>
#include <mat_dense.h>
#include <kernels.h>
#include <ivector.h>
/* MKL */
#include <mkl_pardiso.h>
/* ParBCG */
#include "operator.h"
/******************************************************************************/

/******************************************************************************/
/*                                    CODE                                    */
/******************************************************************************/

/*
 * Factorize the diagonal block associated to the local row panel owned locally
 * input: A         : a CSR matrix CPLM_Mat_CSR_t
 *        rowPos    : the vector returned by preAlps_OperatorGetRowPosPtr
 *        sizeRowPos: the size of rowPos
 *        colPos    : the vector returned by preAlps_OperatorGetColPosPtr
 *        sizeColPos: the size of colPos
 */
int preAlps_BlockJacobiCreate(CPLM_Mat_CSR_t* A,
                              int* rowPos,
                              int sizeRowPos,
                              int* colPos,
                              int sizeColPos);

/*
 * Solve Mx = rhs and put the result into rhs.
 * Internal usage only.
 */
int preAlps_BlockJacobiInitialize(CPLM_DVector_t* rhs);

/*
 * Solve M B_out = A_in with M a block Jacobi preconditioner.
 */
int preAlps_BlockJacobiApply(CPLM_Mat_Dense_t* A_in, CPLM_Mat_Dense_t* B_out);

/*
 * Free the memory allocated during the construction of the preconditioner.
 */
void preAlps_BlockJacobiFree();

/******************************************************************************/

#endif
