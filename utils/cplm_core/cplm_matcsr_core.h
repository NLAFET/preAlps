#ifndef CPLM_MATCSR_CORE_H
#define CPLM_MATCSR_CORE_H

#include <cplm_matcsr_struct.h>
#include <cplm_v0_metis_utils.h>

/* Macros */
#define CPLM_MatCSRNULL() {\
  .info={ .M=0, .N=0, .nnz=0, .m=0, .n=0, .lnnz=0, .blockSize=0, .format=FORMAT_CSR, .structure=UNSYMMETRIC },\
  .rowPtr=NULL, .colInd=NULL, .val=NULL\
}\

/**
 * \fn void CPLM_MatCSRFree(CPLM_Mat_CSR_t *A_io)
 * \brief This method frees the memory occuped by a matrix
 * \param *A_io The matrix which has to be freed
 */
void CPLM_MatCSRFree(CPLM_Mat_CSR_t *A_io);


int CPLM_MatCSRInit(CPLM_Mat_CSR_t *A_out, CPLM_Info_t *info);




/**
 * \fn int CPLM_MatCSRMalloc(CPLM_Mat_CSR_t *A_io)
 * \brief Allocate the memory following the info part.
 * More precisely, it allows m+1 INT for rowPtr, lnnz INT and lnnz DOUBLE for colInd and val arrays.
 * It checks weither the arrays are null or not.
 * \param *A_io   The matrix to free
 */
int CPLM_MatCSRMalloc(CPLM_Mat_CSR_t *A_io);


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
 * \fn int CPLM_MatCSRDelDiag(CPLM_Mat_CSR_t *matCSR)
 * \brief Function which creates a CSR matrix with the same structure of the original CSR matrix and
 * deletes the diagonal values
 * \param *m1 The original CSR matrix
 * \param *m2 The CSR matrix created without diagonal values
 */
/*Function deletes diagonal element from a CPLM_Mat_CSR_t matrix*/
int CPLM_MatCSRDelDiag(CPLM_Mat_CSR_t *m1, CPLM_Mat_CSR_t *m2);

int CPLM_MatCSRSetInfo(CPLM_Mat_CSR_t *A_out, int M, int N, int nnz, int m, int n, int lnnz, int blockSize);

/* Partitioning routines */

idx_t* callKway(CPLM_Mat_CSR_t *matCSR, idx_t nbparts);
#endif
