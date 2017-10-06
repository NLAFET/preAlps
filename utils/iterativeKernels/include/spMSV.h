/*
 ============================================================================
 Name        : spMSV.h
 Author      : Simplice Donfack
 Version     : 0.1
 Description : Sparse matrix matrix product
 Date        : Sept 27, 2016
 ============================================================================
 */
#ifndef SPMSV_H
#define SPMSV_H

#include <mpi.h>


/*
 * Purpose
 * =======
 *
 * Perform a matrix matrix product C = A*B, 
 * where A is a CSR matrix, B is a CSR matrix formed by a set of vectors, and C a CSR matrix.
 * The matrix is initially 1D row block distributed.
 *
 * Arguments
 * =========
 *
 * mloc:
 *		input: local number of rows of A owned by the processor calling this routine
 *
 * nloc:
 *		input: local number of columns of A
 *
 * rloc:
 *		input: local number of columns of B
 *
 * xa,asub,a:
 *		input: matrix A of size (mloc x nloc) stored using CSR.
 *
 * xb,bsub,b:
 *		input: matrix B of size (nloc x rloc) stored using CSC.  If B is stored using CSR , then see options below;
 *
 * a_nparts:
 *	 	input: the number of rows block for matrix A. Obviously, it is the number of processors in case of a 1D block distribution.
 *		The global block structure of A has size (a_nparts x a_nparts).
 *
 * a_nrowparts:
 * 		input: Array of size a_nparts to indice the number of rows in Block column of A.
 *
 * b_nparts:
 * 		input: number of block columns for matrix B
 *		The global sparse block struct size of B is (a_nparts x b_nparts).
 *
 * b_ncolparts:
 * 		input: Array of size b_nparts to indicate the number of columns in Block column of  B.
 *
 * ABlockStruct
 *		input: array of size (a_nparts x a_nparts) to indicate the sparse block structure of A. 
 *			   ABlockStruct(i,j) = 0 if the block does not contains any element.
 *			   This array must be precomputed by the user before calling this routine.
 *
 * xc,csub,c:
 *		output: matrix C stored as CSC
 *		if C is NULL, or  
 *		if C size is smaller than the size required for A*B, it will be reallocated with the enlarged size.
 *		Otherwise C will be reused.
 *		The sparse block struct size of C is (a_nparts x b_nparts)
 *
 * options: array of size 3 to control this routine. (Coming soon)
 * 		input:
 *			options[0] = 1 if the input matrix B is stored as CSR instead of CSC. An internal buffer will be required.
 *			options[1] = 1 if the result matrix must be stored as CSR instead of CSC.
 *			options[2] = 1 switch off the possibility to switch to dense matrix.
 *
 *
 * Return
 * ======
 * 0: the resulting matrix C is sparse, use (xc, asubc, c) to manipulate it.
 * 1: the resulting matrix C is converted to dense, use only (c) to manipulate it.
 * < 0: an error occured during the execution.
 */

int preAlps_spMSV(MPI_Comm comm, int mloc, int nloc, int rloc, int *xa, int *asub, double *a, int *xb, int *bsub, double *b,
	int a_nparts, int *a_nrowparts,
	int b_nparts, int *b_ncolparts,
	int *ABlockStruct, 
	int **xc, int **csub, double **c, int *options);



#endif