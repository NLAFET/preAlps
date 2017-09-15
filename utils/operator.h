/******************************************************************************/
/* Author     : Olivier Tissot                                                */
/* Creation   : 2016/08/05                                                    */
/* Description: Definition of the linear operator                             */
/******************************************************************************/


/******************************************************************************/
/*                                  INCLUDE                                   */
/******************************************************************************/
#ifndef OPERATOR_H
#define OPERATOR_H

/* STD */
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
/* MPI */
#include <mpi.h>
/* CPaLAMeM */
#include <cpalamem_macro.h>
#include <cpalamem_instrumentation.h>
#include <mat_csr.h>
#include <mat_dense.h>
#include <mat_load_mm.h>
#include <metis_interface.h>
#include <matmult.h>
/******************************************************************************/

/******************************************************************************/
/*                                    CODE                                    */
/******************************************************************************/

/* Mem management */
int OperatorBuild(const char* matrixFilename, MPI_Comm comm); // Read the global variables A and AStruct
void OperatorFree();
/* Utils */
void OperatorPrint(int rank);
int  OperatorGetSizes(int* M, int* m);
int  BlockOperator(Mat_Dense_t* X, Mat_Dense_t* AX);
/* With Block diagonal preconditioner */
/* int  BlockOperatorJacPrec(Mat_Dense_t* X, Mat_Dense_t* AX); */
int  Operator(DVector_t* x, DVector_t* ax); // TODO
int  OperatorGetA(Mat_CSR_t* A);
int  OperatorGetRowPosPtr(int** rowPos, int* sizeRowPos);
int  OperatorGetColPosPtr(int** colPos, int* sizeColPos);
int  OperatorGetDepPtr(int** dep, int* sizeDep);
/******************************************************************************/

#endif
