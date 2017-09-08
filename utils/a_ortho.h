/******************************************************************************/
/* Author     : Olivier Tissot                                                */
/* Creation   : 2016/06/24                                                    */
/* Description: A-orthonormalization methods                                  */
/******************************************************************************/

/******************************************************************************/
/*                                  INCLUDE                                   */
/******************************************************************************/
#ifndef AORTHO_H
#define AORTHO_H


/* STD */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
/* CPaLAMeM */
#include <cpalamem_macro.h>
#include <cpalamem_instrumentation.h>
#include <mat_csr.h>
#include <mat_load_mm.h>
#include <metis_interface.h>
#include <mat_dense.h>
#include <dvector.h>
/* TSQR */
//#include <tsqr.h>
/* MKL */
#include <mkl.h>
/* MPI */
#include <mpi.h>
/* ParBCG */
#include "solver.h"
#include "operator.h"
#include "parbcg_macro.h"
/******************************************************************************/

/******************************************************************************/
/*                                    CODE                                    */
/******************************************************************************/
// See B. Loweryand, J. Langou "Stability Analysis of QR factorization in an Oblique Inner Product"
int A_CholQR(Mat_Dense_t* P, Mat_Dense_t* AP);

// C = L^t L (using MKL Lapack)
int Cholesky(Mat_Dense_t* C);
// B = L^-1 B with L lower triangular matrix
int LowerTriangularLeftSolve(Mat_Dense_t* L, Mat_Dense_t* B);
// B = B R^-1 with R upper triangular matrix
int UpperTriangularRightSolve(Mat_Dense_t* R, Mat_Dense_t* B);
// A-normalize P vectors
int A_Normalize(Mat_Dense_t* P, Mat_Dense_t* AP);

// Test function
int A_OrthoTest(int rank);
/******************************************************************************/

#endif
