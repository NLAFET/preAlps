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

int preAlps_BlockJacobiCreate(CPLM_Mat_CSR_t* A,
                              int* rowPos,
                              int sizeRowPos,
                              int* colPos,
                              int sizeColPos,
                              int* dep,
                              int sizeDep);
int preAlps_BlockJacobiInitialize(CPLM_DVector_t* rhs);
int preAlps_BlockJacobiApply(CPLM_Mat_Dense_t* A_in, CPLM_Mat_Dense_t* B_out);
void preAlps_BlockJacobiFree();

/******************************************************************************/

#endif
