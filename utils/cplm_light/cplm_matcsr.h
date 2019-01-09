/*
============================================================================
Name        : cplm_matcsr.h
Author      : Simplice Donfack, Sebastien Cayrols
Version     : 0.1
Description : Operations on sparse matrices stored on CSR format.
Date        : June 22, 2018
============================================================================
*/
#ifndef CPLM_MATCSR_H
#define CPLM_MATCSR_H


#include <stdlib.h>
#include <mpi.h>

#include <cplm_matcsr_struct.h>
#include <cplm_matcsr_core.h>

#include <cplm_v0_matcsr.h> // Backward compatibility

/*
 *
 * Author: Sebastion Cayrols
 *
 */



/* Constantes */
#define PRINT_PARTIAL_M 10
#define PRINT_PARTIAL_N 10
#define _PRECISION 4

/* Macros */

#define CPLM_MatCSRPrintf2D(_msg,_A) { printf("%s\n",(_msg));                                              \
                                  ( ((_A)->info.m<PRINT_PARTIAL_M && (_A)->info.n<PRINT_PARTIAL_N) ?  \
                                      CPLM_MatCSRPrint2D((_A))     :                                       \
                                      CPLM_MatCSRPrintPartial2D((_A))  ); }

/* MPI Utils */
MPI_Datatype initMPI_StructCSR();

/* MatrixMarket routines */
#define MM_MAX_LINE_LENGTH  1025
#define MM_MAX_TOKEN_LENGTH 64
#define MM_MATRIXMARKET_STR "%%MatrixMarket"
#define MM_MATRIX_STR       "matrix"
#define MM_DENSE_STR        "array"
#define MM_SPARSE_STR       "coordinate"
#define MM_COMPLEX_STR      "complex"
#define MM_REAL_STR         "real"
#define MM_INT_STR          "integer"
#define MM_GENERAL_STR      "general"
#define MM_SYMM_STR         "symmetric"
#define MM_HERM_STR         "hermitian"
#define MM_SKEW_STR         "skew-symmetric"
#define MM_PATTERN_STR      "pattern"

/*
 * Load a matrix from a file using MatrixMarket format
 *
 * TODO: full support of the MatrixMarket format?
 */
int CPLM_LoadMatrixMarket( const char* filename, CPLM_Mat_CSR_t* mat);




/* MatCSR routines */

/**
 * \fn int CPLM_MatCSRChkDiag(CPLM_Mat_CSR_t *m)
 * \brief This function looks for position of diagonal elements in m and assume the matrix does not have any zero on the diagonal
 * \param *m    The CSR matrix
 * \return      0 if the searching is done
 */
int CPLM_MatCSRChkDiag(CPLM_Mat_CSR_t *m);




/**
 * \fn void CPLM_MatCSRConvertTo0BasedIndexing(CPLM_Mat_CSR_t *m)
 * \brief Convert a matrix into C format index
 * \param *matCSR The matrix which has to be reformated
 */
void CPLM_MatCSRConvertTo0BasedIndexing(CPLM_Mat_CSR_t *m);

/**
 * \fn void CPLM_MatCSRConvertTo1BasedIndexing(CPLM_Mat_CSR_t *m)
 * \brief Convert a matrix into matlab format index
 * \param *matCSR The matrix which has to be reformated
 */
void CPLM_MatCSRConvertTo1BasedIndexing(CPLM_Mat_CSR_t *m);

/**
 * \fn int CPLM_MatCSRCopy(CPLM_Mat_CSR_t *m1, CPLM_Mat_CSR_t *m2)
 * \brief Function which copy all the CSR matrix to a new CSR matrix
 * \param *m1     The original matrix which has to be copied
 * \param *m2     The copy matrix
 * \return        0 if copy has succeed
 */
int CPLM_MatCSRCopy(CPLM_Mat_CSR_t *m1, CPLM_Mat_CSR_t *m2);

/**
 * \fn int CPLM_MatCSRCopyStruct(CPLM_Mat_CSR_t *m1, CPLM_Mat_CSR_t *m2)
 * \brief Function which copy the structure of the CSR matrix to a new CSR matrix
 * \param *m1     The original matrix which has to be copied
 * \param *m2     The copy matrix
 * \return        0 if copy has succeed
 */
int CPLM_MatCSRCopyStruct(CPLM_Mat_CSR_t *A_in, CPLM_Mat_CSR_t *B_out);



/**
 * \fn int CPLM_MatCSRRecv(CPLM_Mat_CSR_t *m, int source, MPI_Comm comm)
 * \brief This function sends a CSR matrix to id'th MPI process
 * \param *m      The CSR matrix
 * \param source  The MPI process id which sends the matrix
 * \param comm    The MPI communicator of the group
 * \return        0 if the reception succeed
 */
int CPLM_MatCSRRecv(CPLM_Mat_CSR_t *A_out, int source, MPI_Comm comm);

/**
 * \fn int CPLM_MatCSRSend(CPLM_Mat_CSR_t *m, int dest, MPI_Comm comm)
 * \brief This function sends a CSR matrix to id'th MPI process
 * \param *m    The CSR matrix
 * \param dest  The MPI process id which receives the matrix
 * \param comm  The MPI communicator of the group
 * \return      0 if the sending is done
 */
int CPLM_MatCSRSend(CPLM_Mat_CSR_t *m, int dest, MPI_Comm comm);




/**
 * \fn void CPLM_MatCSRPrint2D(CPLM_Mat_CSR_t *m)
 * \brief Method which prints the CSR matrix into standard format
 * \param *matCSR The matrix which has to be printed
 */
/*Print original matrix */
void CPLM_MatCSRPrintPartial2D(CPLM_Mat_CSR_t *m);


/**
 * \fn void CPLM_MatCSRPrint2D(CPLM_Mat_CSR_t *m)
 * \brief Method which prints the CSR matrix into standard format
 * \param *matCSR The matrix which has to be printed
 */
/*Print original matrix */
void CPLM_MatCSRPrint2D(CPLM_Mat_CSR_t *m);


/*
 *
 * Author: Simplice Donfack
 *
 */



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





/* Create a MatCSRNULL matrix, same as A = CPLM_MatCSRNULL() but for a matrix referenced as pointer. */
int CPLM_MatCSRCreateNULL(CPLM_Mat_CSR_t **A);

/* Broadcast the matrix dimension from the root to the other procs*/
int CPLM_MatCSRDimensions_Bcast(CPLM_Mat_CSR_t *A, int root, int *m, int *n, int *nnz, MPI_Comm comm);

/*
 * Load a matrix from a file using MatrixMarket or PETSc format depending on the file extension
 */
int CPLM_LoadMatrix( const char* matrix_filename, CPLM_Mat_CSR_t* A_out);

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


/**
 * \fn int CPLM_MatCSRSave(CPLM_Mat_CSR_t *m, const char *filename)
 * \brief Method which saves a CSR matrix into a file
 * \Note This function saves the matrix into Matrix market format
 * \param *m          The CSR matrix which has to be saved
 * \param *filename   The name of the file
 */
/*Function saves a CPLM_Mat_CSR_t matrix in a file*/
int CPLM_MatCSRSave(CPLM_Mat_CSR_t *m, const char *filename);

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
