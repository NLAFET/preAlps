/*
============================================================================
Name        : cplm_matcsr_struct.h
Author      : Simplice Donfack, Sebastien Cayrols
Version     : 0.1
Description : Operations on sparse matrices stored on CSR format.
Date        : Jan 06, 2019
============================================================================
*/
#ifndef CPLM_MATCSR_STRUCT_H
#define CPLM_MATCSR_STRUCT_H

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

#endif
