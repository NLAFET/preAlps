/*
 ============================================================================
 Name        : preAlps_matrix.h
 Author      : Simplice Donfack
 Version     : 0.1
 Description : Sequential matrix utilities
 Date        : Sept 27, 2016
 ============================================================================
 */
#ifndef PREALPS_MATRIX_UTILS_H
#define PREALPS_MATRIX_UTILS_H

#include <mpi.h>

#include "mmio.h"
#include "s_utils.h"

typedef enum {MATRIX_CSR, MATRIX_CSC} preAlps_matrix_storage_t;

typedef enum {MATRIX_ROW_MAJOR, MATRIX_COLUMN_MAJOR} preAlps_matrix_layout_t;

/*Sort the row index of a CSR matrix*/
void preAlps_matrix_colIndex_sort(int m, int *xa, int *asub, double *a);

/*
 * Create a sparse column block structure of an CSR matrix.
 * n:
 *		input: global number of columns
 * nparts:
 *		input: number of domain
 * ncounts:
 * 		input: number of columns in each subdomain
 * ABlockStruct
 *		output: array of size npart allocated on any process calling this routine.
 */
int preAlps_matrix_ColumnBlockStruct(preAlps_matrix_storage_t mtype, int mloc, int nloc, int *xa, int *asub, int nparts, int *ncounts, int *AcolBlockStruct);


/*Convert a matrix from csc to csr*/
int preAlps_matrix_convert_csc_to_csr(int m, int n, int **xa, int *asub, double *a);

/*Convert a matrix from csr to csc*/
int preAlps_matrix_convert_csr_to_csc(int m, int n, int **xa, int *asub, double *a);

/*
 * Convert a matrix from csr to dense
 */

int preAlps_matrix_convert_csr_to_dense(int m, int n, int *xa, int *asub, double *a, preAlps_matrix_layout_t mlayout, double *a1, int lda1);

/*
 * Convert a matrix from csr to dense
 */
/*
int preAlps_matrix_convert_csr_to_dense(int m, int n, int **xa, int *asub, double *a, double *a1, int lda1);
*/
/* copy matrix A to A1 */
void preAlps_matrix_copy(int m, int *xa, int *asub, double *a, int *xa1, int *asub1, double *a1);

/*print a dense matrix*/
void preAlps_matrix_dense_print(preAlps_matrix_layout_t mlayout, int m, int n, double *a, int lda, char *s);

#ifdef USE_HPARTITIONING
/*
 * Partition a matrix using and hypergraph to represent its structure
 * part_loc:
 *     output: part_loc[i]=k means rows i belongs to subdomain k
 */
int preAlps_matrix_hpartition_sequential(int m, int n, int *xa, int *asub, int nparts, int *part);
#else
	
/*
 * Partition a matrix using Metis
 * part_loc:
 * 		output: part_loc[i]=k means rows i belongs to subdomain k
 */
int preAlps_matrix_partition_sequential(int m, int *xa, int *asub, int nparts, int *part);
#endif


/* 
 * Compute A1 = A(pinv,q) where pinv and q are permutations of 0..m-1 and 0..n-1. 
 * if pinv or q is NULL it is considered as the identity
 */
void preAlps_matrix_permute (int n, int *xa, int *asub, double *a, int *pinv, int *q,int *xa1, int *asub1,double *a1);

/* Print a CSR matrix */
void preAlps_matrix_print(preAlps_matrix_storage_t mtype, int m, int *xa, int *asub, double *a, char *s);

/* Read a matrix market data file and stores the matrix using CSC format */
int preAlps_matrix_readmm_csc(char *filename, int *m, int *n, int *nnz, int **xa, int **asub, double **a);

/* Read a matrix market data file and stores the matrix using CSR format */
int preAlps_matrix_readmm_csr(char *filename, int *m, int *n, int *nnz, int **xa, int **asub, double **a);

/*
 * Perform a product of two matrices, A(i_begin:i_end, :) stores as CSR and B as dense 
 *
 * ptrRowBegin:
 *		input: ptrRowBegin[i] = j means the first non zeros element of row i is in column j
 * ptrRowEnd:
 *		input: ptrRowEnd[i] = j means the last non zeros element of row i is in column j
 */
int preAlps_matrix_subMatrix_CSRDense_Product(int m, int *ptrRowBegin, int *ptrRowEnd, int a_colOffset, int *asub, double *a, double *b, int ldb, int b_nbcol, double *c, int ldc);


/* Create a full symmetric from a lower triangular matrix stored in CSC format */
int preAlps_matrix_symmetrize(int m, int n, int nnz, int *xa, int *asub, double *a, int *nnz2, int **xa2, int **asub2, double **a2);



#endif