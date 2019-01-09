/*
============================================================================
Name        : cplm_matcsr.c
Author      : Simplice Donfack, Sebastien Cayrols
Version     : 0.1
Description : Functions of preAlps which will be part of MatCSR.
Date        : Oct 13, 2017
============================================================================
*/

#ifndef CPLM_V0_MATCSR_H
#define CPLM_V0_MATCSR_H

#include <stdlib.h>
#include <cplm_v0_metis_utils.h>
#include <cplm_v0_ivector.h>
#include <cplm_v0_dvector.h>
#include <cplm_v0_timing.h>
#include <cplm_matcsr_struct.h>
#include <cplm_matcsr_core.h>

int CPLM_MatCSRConvertFromDenseDVector(CPLM_Mat_CSR_t *m_out, CPLM_DVector_t *v_in, int M, int N);

/*Create a matrix from a dense vector of type double, the matrix is stored in column major format*/
int CPLM_MatCSRConvertFromDenseColumnMajorDVectorPtr(CPLM_Mat_CSR_t *m_out, double *v_in, int M, int N);

/*Create a matrix from a dense vector of type double*/
int CPLM_MatCSRConvertFromDenseDVectorPtr(CPLM_Mat_CSR_t *m_out, double *v_in, int M, int N);

int CPLM_metisKwayOrdering(CPLM_Mat_CSR_t *m1, CPLM_IVector_t *perm, int nblock, CPLM_IVector_t *posB);

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

int CPLM_MatCSRIsSym(CPLM_Mat_CSR_t *m);

/*Function which permutes CPLM_Mat_CSR_t matrix with vec and ivec vector*/
int CPLM_MatCSRPermute(CPLM_Mat_CSR_t *A_in, CPLM_Mat_CSR_t *B_out, int *rowPerm, int *colPerm, Choice_permutation permute_values);

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

#endif
