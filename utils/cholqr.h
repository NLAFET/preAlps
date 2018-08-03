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
#ifndef CHOLQR_H
#define CHOLQR_H

/* STD */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
/* MPI */
#include <mpi.h>
/* CPaLAMeM */
#include <cplm_matdense.h>
/******************************************************************************/

/******************************************************************************/
/*                                    CODE                                    */
/******************************************************************************/
// See B. Loweryand, J. Langou "Stability Analysis of QR factorization in an Oblique Inner Product"
int CPLM_MatDenseACholQR(CPLM_Mat_Dense_t* P, CPLM_Mat_Dense_t* AP, CPLM_Mat_Dense_t* work, MPI_Comm comm);
// A-normalize P vectors
int CPLM_MatDenseANormalize(CPLM_Mat_Dense_t* P, CPLM_Mat_Dense_t* AP, CPLM_Mat_Dense_t* work, MPI_Comm comm);

// Same as A_CholQR but with the usual scalar product
int CPLM_MatDenseCholQR(CPLM_Mat_Dense_t* P, CPLM_Mat_Dense_t *R, MPI_Comm comm);

// Test function
int CPLM_MatDenseCholQRTest(int rank);
/******************************************************************************/

#endif
