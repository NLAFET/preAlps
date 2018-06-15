/*
============================================================================
Name        : preAlps_utils.h
Author      : Simplice Donfack
Version     : 0.1
Description : Utils for preAlps
Date        : Mai 15, 2017
============================================================================
*/
#ifndef PREALPS_UTILS_H
#define PREALPS_UTILS_H

#include <stdlib.h>
#include <mpi.h>
#ifdef USE_MKL
#include <mkl.h>
#endif

#include "preAlps_cplm_ivector.h"
#include "preAlps_cplm_matcsr.h"
#include "preAlps_intvector.h"
#include "preAlps_doublevector.h"
#include "preAlps_cplm_matcsr.h"

#ifndef max
#define max(a,b) ((a) > (b) ? a : b)
#endif

#ifndef min
#define min(a,b) ((a) < (b) ? a : b)
#endif

/*
 * Macros
 */

#define preAlps_checkError(errnum) preAlps_checkError_srcLine((errnum), __LINE__, __FILE__)

/*
 * Functions
 */

 /* MPI custom function to sum the column of a matrix using MPI_REDUCE */
 void DtColumnSum(void *invec, void *inoutvec, int *len, MPI_Datatype *dtype);


/* Display a message and stop the execution of the program */
void preAlps_abort(char *s, ...);

/*
 * From an array, set one when the node number is a leave.
 * The array should be initialized with zeros before calling this routine
 * n:  total number of nodes in the tree.
*/
void preAlps_binaryTreeIsLeaves(int nparts, int *isLeave);

/*
 * From an array, set one when the node number is a node at the target Level.
 * The array should be initialized with zeros before calling this routine
*/
void preAlps_binaryTreeIsNodeAtLevel(int targetLevel, int twoPowerLevel, int part_root, int *isNodeLevel);

/*
 * Create a block arrow structure from a matrix A
 * comm:
 *     input: the communicator for all the processors calling the routine
 * m:
 *     input: the number of rows of the global matrix
 * A:
 *     input: the input matrix
 * AP:
 *     output: the matrix permuted into a block arrow structure (relevant only on proc 0)
 * perm:
 *     output: the permutation vector
 * nbparts:
 *     output: the number of the partition created
 * partCount:
 *     output: the number of rows in each part
 * partBegin:
 *     output: the begining rows of each part.
 */
int preAlps_blockArrowStructCreate(MPI_Comm comm, int m, CPLM_Mat_CSR_t *A, CPLM_Mat_CSR_t *AP, int *perm, int *nbparts, int **partCount, int **partBegin);



/*
 * Distribute the matrix which has a block Arrow structure to the processors.
 */

 int preAlps_blockArrowStructDistribute(MPI_Comm comm, int m, CPLM_Mat_CSR_t *AP, int *perm, int nparts, int *partCount, int *partBegin,
   CPLM_Mat_CSR_t *locAP, int *newPerm, CPLM_Mat_CSR_t *Aii, CPLM_Mat_CSR_t *Aig, CPLM_Mat_CSR_t *Agi, CPLM_Mat_CSR_t *locAgg, int *sep_mcounts, int *sep_moffsets);


/* Distribute the separator to each proc and permute the matrix such as their are contiguous in memory */
int preAlps_blockArrowStructSeparatorDistribute(MPI_Comm comm, int m, CPLM_Mat_CSR_t *AP, int *perm, int nparts, int *partCount, int *partBegin,
    CPLM_Mat_CSR_t *locAP, int *newPerm, CPLM_Mat_CSR_t *locAgg, int *sep_mcounts, int *sep_moffsets);


/*
 *
 * First permute the matrix using kway partitioning
 * Permute each block row such as any row with zeros outside the diagonal move
 * to the bottom on the matrix (ODB)
 *
 * comm:
 *     input: the communicator for all the processors calling the routine
 * A:
 *     input: the input matrix
 * locA:
 *     output: the matrix permuted into a block arrow structure on each procs
 * perm:
 *     output: the permutation vector
 * partBegin:
 *     output: the begining rows of each part.
 * nbDiagRows:
 *     output: the number of rows in the diag of each Row block
*/
int preAlps_blockDiagODBStructCreate(MPI_Comm comm, CPLM_Mat_CSR_t *A, CPLM_Mat_CSR_t *locA, int *perm, int **partBegin, int *nbDiagRows);

/*
 * Check errors
 * No need to call this function directly, use preAlps_checkError() instead.
*/
void preAlps_checkError_srcLine(int err, int line, char *src);



/* Display statistiques min, max and avg of a double*/
void preAlps_dstats_display(MPI_Comm comm, double d, char *str);



/*
 * Each processor print the value of type int that it has
 * Work only in debug (-DDEBUG) mode
 * a:
 *    The variable to print
 * s:
 *   The string to display before the variable
 */
void preAlps_int_printSynchronized(int a, char *s, MPI_Comm comm);


/*Sort the row index of a CSR matrix*/
void preAlps_matrix_colIndex_sort(int m, int *xa, int *asub, double *a);

/*
 * Compute A1 = A(pinv,q) where pinv and q are permutations of 0..m-1 and 0..n-1.
 * if pinv or q is NULL it is considered as the identity
 */
void preAlps_matrix_permute (int n, int *xa, int *asub, double *a, int *pinv, int *q,int *xa1, int *asub1,double *a1);

/*
 * We consider one binary tree A and two array part_in and part_out.
 * part_in stores the nodes of A as follows: first all the children at the last level n,
 * then all the children at the level n-1 from left to right, and so on,
 * while part_out stores the nodes of A in depth first search, so each parent node follows all its children.
 * The array part_in is compatible with the array sizes returned by ParMETIS_V3_NodeND.
 * part_out[i] = j means node i in part_in correspond to node j in part_in.
*/
void preAlps_NodeNDPostOrder(int nparts, int *part_in, int *part_out);


/*
 * Number the nodes at level targetLevel and decrease the value of pos.
*/
void preAlps_NodeNDPostOrder_targetLevel(int targetLevel, int twoPowerLevel, int part_root, int *part_in, int *part_out, int *pos);


/*
 * Split n in P parts.
 * Returns the number of element, and the data offset for the specified index.
 */
void preAlps_nsplit(int n, int P, int index, int *n_i, int *offset_i);

/* pinv = p', or p = pinv' */
int *preAlps_pinv (int const *p, int n);

/* pinv = p', or p = pinv' */
int preAlps_pinv_outplace (int const *p, int n, int *pinv);


/*
 * Permute the rows of the matrix with offDiag elements at the bottom
 * locA:
 *     input: the local part of the matrix owned by the processor calling this routine
 * idxRowBegin:
 *     input: the global array to indicate the partition of each column
 * nbDiagRows:
 *     output: the number of rows permuted on the diagonal
 * colPerm
 *     output: a preallocated vector of the size of the number of columns of A
 *            to return the global permutation vector
*/
int preAlps_permuteOffDiagRowsToBottom(CPLM_Mat_CSR_t *locA, int *idxColBegin, int *nbDiagRows, int *colPerm, MPI_Comm comm);


/*
 * Permute the matrix to create the global matrix structure where all the Block diag are ordered first
 * followed by the Schur complement.
 * The permuted local matrix will have the form locA = [... A_{i, Gamma};... A_{gamma,gamma}]
 *
 * nbDiagRowsloc:
       input: the number of diagonal block on the processor callinf this routine
 * locA:
 *     input: the local part of the matrix owned by the processor calling this routine
 * idxRowBegin:
 *     input: the global array to indicate the column partitioning
 * locAP:
 *     output: the permuted matrix
 * colPerm
 *     output: a preallocated vector of the size of the number of columns of A
 *            to return the global permutation vector
 * schur_ncols
 *    output: the number of column of the schur complement after the partitioning
 *
*/
int preAlps_permuteSchurComplementToBottom(CPLM_Mat_CSR_t *locA, int nbDiagRows, int *idxColBegin, CPLM_Mat_CSR_t *locAP, int *colPerm, int *schur_ncols, MPI_Comm comm);


/*
 * Check the permutation vector for consistency
 */
int preAlps_permVectorCheck(int *perm, int n);


/*
 * Extract the schur complement of a matrix A
 * A:
 *     input: the matrix from which we want to extract the schur complement
 * firstBlock_nrows:
 *     input: the number of rows of the top left block
 * firstBlock_ncols:
 *     input: the number of cols of the top left block
 * Agg
 *     output: the schur complement matrix
*/

int preAlps_schurComplementGet(CPLM_Mat_CSR_t *A, int firstBlock_nrows, int firstBlock_ncols, CPLM_Mat_CSR_t *Agg);

/*Force the current process to sleep few seconds for debugging purpose*/
void preAlps_sleep(int my_rank, int nbseconds);

/* Load a vector from a file and distribute the other procs */
int preAlps_loadDistributeFromFile(MPI_Comm comm, char *fileName, int *mcounts, double **x);

/* Create two MPI typeVector which can be use to assemble a local vector to a global one */
int preAlps_multiColumnTypeVectorCreate(int ncols, int local_nrows, int global_nrows, MPI_Datatype *localType, MPI_Datatype *globalType);
#endif
