/*
============================================================================
Name        : cplm_matcsr.c
Author      : Simplice Donfack, Sebastien Cayrols
Version     : 0.1
Description : Functions of preAlps which will be part of MatCSR.
Date        : Oct 13, 2017
============================================================================
*/

#ifndef PREALPS_CPLM_MATCSR_H
#define PREALPS_CPLM_MATCSR_H

#include <stdlib.h>
#include <mpi.h>
#include <cplm_v0_ivector.h>
#include <cplm_v0_dvector.h>
#include <cplm_v0_metis_utils.h>
/*
 *
 * Author: Sebastion Cayrols
 *
 */

/******************************************************************************/
/*                                  STRUCT                                    */
/******************************************************************************/
/**
 *  \enum
 *
 */
typedef enum {
  FORMAT_CSR,
  FORMAT_BCSR,
  FORMAT_BCSR_VAR // strange BCRS variant used by PETSc, the values aren't stored by block
} CPLM_Mat_CSR_format_t;

/**
 *  \enum Struct_Type
 *  \brief This enumeration contains information about structure
 */
typedef enum {
  UNSYMMETRIC,
  SYMMETRIC
} Struct_Type;

/**
 *  \enum Choice_permutation
 *
 */
typedef enum {
  AVOID_PERMUTE,
  PERMUTE
} Choice_permutation;

/**
 *  \struct CPLM_Info_t
 *  \brief Structure which represents the structure of the CSR matrix
 */
/*Structure represents main informations from a CPLM_Mat_CSR_t*/
typedef struct{
  int M;                  //Global num of rows
  int N;                  //Global num of cols
  int nnz;                //Global non-zeros entries
  int m;                  //Local num of rows
  int n;                  //Local num of cols
  int lnnz;               //Local non-zeros entries
  int blockSize;          //Local block size
  CPLM_Mat_CSR_format_t format;//Local storage format : block or not
  Struct_Type structure;  //Local symmetric or unsymmetric pattern
} CPLM_Info_t;

/**
 *  \struct CPLM_Mat_CSR_t
 *  \brief Structure which represents a CSR format to store a matrix with the smallest size in memory
 *
 */
typedef struct {
  CPLM_Info_t info;
  int* rowPtr; //A pointer to an array of size M+1 or m+1
  int* colInd; //A pointer to an array of size nnz or lnnz
  double* val; //A pointer to an array of size nnz or lnnz
} CPLM_Mat_CSR_t;


/* Constantes */
#define PRINT_PARTIAL_M 10
#define PRINT_PARTIAL_N 10
#define _PRECISION 4

/* Macros */
#define CPLM_MatCSRNULL() {\
  .info={ .M=0, .N=0, .nnz=0, .m=0, .n=0, .lnnz=0, .blockSize=0, .format=FORMAT_CSR, .structure=UNSYMMETRIC },\
  .rowPtr=NULL, .colInd=NULL, .val=NULL\
}\

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

/* Partitioning routines */

idx_t* callKway(CPLM_Mat_CSR_t *matCSR, idx_t nbparts);
int CPLM_metisKwayOrdering(CPLM_Mat_CSR_t *m1, CPLM_IVector_t *perm, int nblock, CPLM_IVector_t *posB);

/* MatCSR routines */

/**
 * \fn int CPLM_MatCSRChkDiag(CPLM_Mat_CSR_t *m)
 * \brief This function looks for position of diagonal elements in m and assume the matrix does not have any zero on the diagonal
 * \param *m    The CSR matrix
 * \param *v    The CPLM_IVector_t returned containing indices
 * \return      0 if the searching is done
 */
int CPLM_MatCSRChkDiag(CPLM_Mat_CSR_t *m);

int CPLM_MatCSRConvertFromDenseDVector(CPLM_Mat_CSR_t *m_out, CPLM_DVector_t *v_in, int M, int N);


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
 * \fn int CPLM_MatCSRDelDiag(CPLM_Mat_CSR_t *matCSR)
 * \brief Function which creates a CSR matrix with the same structure of the original CSR matrix and
 * deletes the diagonal values
 * \param *m1 The original CSR matrix
 * \param *m2 The CSR matrix created without diagonal values
 */
/*Function deletes diagonal element from a CPLM_Mat_CSR_t matrix*/
int CPLM_MatCSRDelDiag(CPLM_Mat_CSR_t *m1, CPLM_Mat_CSR_t *m2);
/**
 * \fn void CPLM_MatCSRFree(CPLM_Mat_CSR_t *A_io)
 * \brief This method frees the memory occuped by a matrix
 * \param *A_io The matrix which has to be freed
 */
void CPLM_MatCSRFree(CPLM_Mat_CSR_t *A_io);

/*
*
* This function returns colPos which is an index of the begin and end of each block in column point of view.
*
*/
int CPLM_MatCSRGetColBlockPos(CPLM_Mat_CSR_t *m, CPLM_IVector_t *pos, CPLM_IVector_t *colPos);

/**
 * \fn
 * \brief Function creates a CSR matrix without value which corresponds to a submatrix of the original CSR matrix and this submatrix is a part given by Metis_GraphPartKway
 * Note : this function does not need the matrix values
 * \param *A_in         The original CSR matrix
 * \param *B_out        The original CSR matrix
 * \param *pos          The array containing the begin of each part of the CSR matrix
 * \param numBlock      The number of the part which will be returned
 * \return            0 if succeed
 */
/*function which returns a submatrice at CSR format from an original matrix (in CSR format too) and filtered by parts*/
/*num_parts corresponds to the number which selects rows from original matrix and interval allows to select the specific rows*/
int CPLM_MatCSRGetDiagBlock(CPLM_Mat_CSR_t *A_in, CPLM_Mat_CSR_t *B_out, CPLM_IVector_t *pos, CPLM_IVector_t *colPos, int structure);

//colPos  is the starting index of each block in each row
//m       is the number of rows
//nblock  is the number of blocks
//bi      is the current block
//dep     is the dependency CPLM_IVector_t containing block id dependency of the block bi
int CPLM_MatCSRGetCommDep(CPLM_IVector_t *colPos, int m, int nblock, int bi, CPLM_IVector_t *dep);

/**
 * \fn int CPLM_MatCSRGetDiagInd(CPLM_Mat_CSR_t *A_in, CPLM_IVector_t *v)
 * \brief This function looks for position of diagonal elements in m and assume the matrix does not have any zero on the diagonal
 * \param *A_in   The CSR matrix
 * \param *v      The CPLM_IVector_t returned containing indices
 * \return      0 if the memory allocation is ok
 */
int CPLM_MatCSRGetDiagInd(CPLM_Mat_CSR_t *A_in, CPLM_IVector_t *v);

/**
 * \fn int CPLM_MatCSRGetDiagIndOfPanel(CPLM_Mat_CSR_t *A_in, CPLM_IVector_t *v)
 * \brief This function looks for position of diagonal elements in m and assume the matrix does not have any zero on the diagonal
 * \param *A_in   The CSR matrix
 * \param *v      The CPLM_IVector_t returned containing indices
 * \param offset  The offset to local the beginning of the diagonal block in the panel
 * \return      0 if the memory allocation is ok
 */
int CPLM_MatCSRGetDiagIndOfPanel(CPLM_Mat_CSR_t *A_in, CPLM_IVector_t *d_out, CPLM_IVector_t *pos, CPLM_IVector_t *colPos, int numBlock);

/*
*
* This function returns colPos which is an index of the begin and end of each block in column point of view.
*
*/
int CPLM_MatCSRGetPartialColBlockPos(CPLM_Mat_CSR_t *A_in,
                                CPLM_IVector_t *posR,
                                int       numBlock,
                                CPLM_IVector_t *posC,
                                CPLM_IVector_t *colPos);

int CPLM_MatCSRGetRowPanel(CPLM_Mat_CSR_t *m1, CPLM_Mat_CSR_t *m2, CPLM_IVector_t *interval, int num_parts);

int CPLM_MatCSRGetSubBlock ( CPLM_Mat_CSR_t *A_in,
                        CPLM_Mat_CSR_t *B_out,
                        CPLM_IVector_t *posR,
                        CPLM_IVector_t *posC,
                        int       numRBlock,
                        int       numCBlock,
                        int       **work,
                        size_t    *workSize);

int CPLM_MatCSRInit(CPLM_Mat_CSR_t *A_out, CPLM_Info_t *info);


int CPLM_MatCSRIsSym(CPLM_Mat_CSR_t *m);

/**
 * \fn int CPLM_MatCSRMalloc(CPLM_Mat_CSR_t *A_io)
 * \brief Allocate the memory following the info part.
 * More precisely, it allows m+1 INT for rowPtr, lnnz INT and lnnz DOUBLE for colInd and val arrays.
 * It checks weither the arrays are null or not.
 * \param *A_io   The matrix to free
 */
int CPLM_MatCSRMalloc(CPLM_Mat_CSR_t *A_io);

/*Function which permutes CPLM_Mat_CSR_t matrix with vec and ivec vector*/
int CPLM_MatCSRPermute(CPLM_Mat_CSR_t *A_in, CPLM_Mat_CSR_t *B_out, int *rowPerm, int *colPerm, Choice_permutation permute_values);

/**
 * \fn void CPLM_MatCSRPrintInfo(CPLM_Mat_CSR_t *m)
 * \brief Method which prints the data structure
 * \param *info The data structure of a CSR matrix
 */
/*Function prints informations about a CPLM_Info_t matrix */
void CPLM_MatCSRPrintInfo(CPLM_Mat_CSR_t *m);

/**
 * \fn int CPLM_MatCSRRealloc(CPLM_Mat_CSR_t *A_io)
 * \brief Reallocate the memory following the info part.
 * More precisely, it allows m+1 INT for rowPtr, lnnz INT and lnnz DOUBLE for colInd and val arrays.
 * It checks weither the arrays are null or not.
 * \param *A_io   The matrix to free
 */
int CPLM_MatCSRRealloc( CPLM_Mat_CSR_t *A_io);

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

int CPLM_MatCSRSetInfo(CPLM_Mat_CSR_t *A_out, int M, int N, int nnz, int m, int n, int lnnz, int blockSize);


/**
 * \fn int CPLM_MatCSRSymStruct(CPLM_Mat_CSR_t *m1, CPLM_Mat_CSR_t *m2)
 * \brief Function symmetrizes the structure of a matrix
 * \param *m1   The original CSR matrix
 * \param *m2   The symmetric CSR matrix
 */
/*Function symmetrizes a CPLM_Mat_CSR_t matrix and delete its diagonal elements if wondered*/
int CPLM_MatCSRSymStruct(CPLM_Mat_CSR_t *m1, CPLM_Mat_CSR_t *m2);

/**
 * \fn int CPLM_MatCSRUnsymStruct(CPLM_Mat_CSR_t *m1, CPLM_Mat_CSR_t *m2)
 * \brief Function which creates a CSR matrix and fills it from the original CSR matrix symmetrizing the structure
 * \param *m1 The input symmetric CSR matrix where the structure is for instancethe upper part
 * \param *m2 The output general CSR matrix
 */
/*Function symmetrizes a CPLM_Mat_CSR_t matrix*/
int CPLM_MatCSRUnsymStruct(CPLM_Mat_CSR_t *m1, CPLM_Mat_CSR_t *m2);

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

/* Broadcast the matrix dimension from the root to the other procs*/
int CPLM_MatCSRDimensions_Bcast(CPLM_Mat_CSR_t *A, int root, int *m, int *n, int *nnz, MPI_Comm comm);

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
