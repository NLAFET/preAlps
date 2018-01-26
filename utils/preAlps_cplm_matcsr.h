/*
============================================================================
Name        : preAlps_cplm_matcsr.c
Author      : Simplice Donfack
Version     : 0.1
Description : Functions of preAlps which will be part of MatCSR.
Date        : Oct 13, 2017
============================================================================
*/

#ifndef PREALPS_CPLM_MATCSR_H
#define PREALPS_CPLM_MATCSR_H

#include <mat_csr.h>


/*
 * Split the matrix in block column and extract the selected block column number.
 * The input matrix is unchanged
 * A:
 *     input: the input matrix
 * nparts:
 *     input: the number of block columns
 * partBegin:
 *     input: the begining position of each blocks
 * numBlock:
 *     input: the number of the block to remove
 * B_out:
 *     out: the output block
 */

int CPLM_MatCSRBlockColumnExtract(CPLM_Mat_CSR_t *A, int nparts, int *partBegin, int numBlock, CPLM_Mat_CSR_t *B_out);


/*
 * Split the matrix in block column and fill the selected block column number with zeros,
 * Optimize the routine to avoid storing these zeros in the output matrix.
 * A_in:
 *     input: the input matrix
 * colCount:
 *     input: the global number of columns in each Block
 * numBlock:
 *     input: the number of the block to fill with zeros
 * B_out:
 *     output: the output matrix after removing the diag block
 */

int CPLM_MatCSRBlockColumnZerosFill(CPLM_Mat_CSR_t *A_in, int *colCount, int numBlock, CPLM_Mat_CSR_t *B_out);

/*
 * 1D block row distirbution of the matrix. At the end, each proc has approximatively the same number of rows.
 *
 */
int CPLM_MatCSRBlockRowDistribute(CPLM_Mat_CSR_t *Asend, CPLM_Mat_CSR_t *Arecv, int *mcounts, int *moffsets, int root, MPI_Comm comm);

  /*
   * 1D block rows gather of the matrix from all the processors in the communicator .
   * Asend:
   *     input: the matrix to send
   * Arecv
   *     output: the matrix to assemble the block matrix received from all (relevant only on the root)
   * idxRowBegin:
   *     input: the global row indices of the distribution
   */
 int CPLM_MatCSRBlockRowGatherv(CPLM_Mat_CSR_t *Asend, CPLM_Mat_CSR_t *Arecv, int *idxRowBegin, int root, MPI_Comm comm);

 /*
  * Gatherv a local matrix from each process and dump into a file
  *
  */
 int CPLM_MatCSRBlockRowGathervDump(CPLM_Mat_CSR_t *locA, char *filename, int *idxRowBegin, int root, MPI_Comm comm);


 /*
  * 1D block rows distribution of the matrix to all the processors in the communicator using ncounts and displs.
  * The data are originally stored on processor root. After this routine each processor i will have the row indexes from
  * idxRowBegin[i] to idxRowBegin[i+1] - 1 of the input matrix.
  *
  * Asend:
  *     input: the matrix to scatterv (relevant only on the root)
  * Arecv
  *     output: the matrix to store the block matrix received
  * idxRowBegin:
  *     input: the global row indices of the distribution
  */
int CPLM_MatCSRBlockRowScatterv(CPLM_Mat_CSR_t *Asend, CPLM_Mat_CSR_t *Arecv, int *idxRowBegin, int root, MPI_Comm comm);

/*Create a matrix from a dense vector of type double, the matrix is stored in column major format*/
int CPLM_MatCSRConvertFromDenseColumnMajorDVectorPtr(CPLM_Mat_CSR_t *m_out, double *v_in, int M, int N);

/*Create a matrix from a dense vector of type double*/
int CPLM_MatCSRConvertFromDenseDVectorPtr(CPLM_Mat_CSR_t *m_out, double *v_in, int M, int N);


/* Create a MatCSRNULL matrix, same as A = CPLM_MatCSRNULL() but for a matrix referenced as pointer. */
int CPLM_MatCSRCreateNULL(CPLM_Mat_CSR_t **A);


/*
 * Matrix matrix product, C := alpha*A*B + beta*C
 * where A is a CSR matrix, B and C is are dense Matrices stored in column major layout/
 */
int CPLM_MatCSRMatrixCSRDenseMult(CPLM_Mat_CSR_t *A, double alpha, double *B, int B_ncols, int ldB, double beta, double *C, int ldC);

/*
 * Matrix vector product, y := alpha*A*x + beta*y
 */
int CPLM_MatCSRMatrixVector(CPLM_Mat_CSR_t *A, double alpha, double *x, double beta, double *y);

/*
 * Perform an ordering of a matrix using parMetis
 *
 */

int CPLM_MatCSROrderingND(MPI_Comm comm, CPLM_Mat_CSR_t *A, int *vtdist, int *order, int *sizes);

/*
 * Partition a matrix using parMetis
 * part_loc:
 *     output: part_loc[i]=k means rows i belongs to subdomain k
 */

int CPLM_MatCSRPartitioningKway(MPI_Comm comm, CPLM_Mat_CSR_t *A, int *vtdist, int nparts, int *partloc);


/* Print a CSR matrix as coordinate triplet (i,j, val)*/
void CPLM_MatCSRPrintCoords(CPLM_Mat_CSR_t *A, char *s);

/* Only one process print its matrix, forces synchronisation between all the procs in the communicator*/
void CPLM_MatCSRPrintSingleCoords(CPLM_Mat_CSR_t *A, MPI_Comm comm, int root, char *varname, char *s);

/*
 * Each processor print the matrix it has as coordinate triplet (i,j, val)
 * Work only in debug (-DDEBUG) mode
 * A:
 *    The matrix to print
 */

void CPLM_MatCSRPrintSynchronizedCoords (CPLM_Mat_CSR_t *A, MPI_Comm comm, char *varname, char *s);


/*
 * Merge the rows of two matrices (one on top of another)
 */
int CPLM_MatCSRRowsMerge(CPLM_Mat_CSR_t *A_in, CPLM_Mat_CSR_t *B_in, CPLM_Mat_CSR_t *C_out);


/*
 *
 * Scale a scaling vectors R and C, and scale the matrix by computing A1 = R * A * C
 * A:
 *     input: the matrix to scale
 * R:
 *     output: a vector with the same size as the number of rows of the matrix
 * C:
 *     output: a vector with the same size as the number of columns of the matrix
 */

int CPLM_MatCSRSymRACScaling(CPLM_Mat_CSR_t *A, double *R, double *C);



/* Transpose a matrix */
int CPLM_MatCSRTranspose(CPLM_Mat_CSR_t *A_in, CPLM_Mat_CSR_t *B_out);

#endif
