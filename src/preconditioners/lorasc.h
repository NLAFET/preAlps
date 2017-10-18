/*
============================================================================
Name        : lorasc.h
Author      : Simplice Donfack
Version     : 0.1
Description : Preconditioner based on Schur complement
Date        : Sept 20, 2017
============================================================================
*/
#ifndef LORASC_H
#define LORASC_H

#include <mpi.h>
#include <mat_csr.h>
#include <mat_dense.h>

#include "preAlps_solver.h"

/*Structure of the preconditioner*/
typedef struct{
  /* User input*/
  double deflation_tol;    /* The deflation tolerance (default: 10^-2)*/
  int nrhs;                /* The number of rhs on which lorasc will be applied. This is required internally for the analysis. Default value: 1 */
  int OptPermuteOnly;         /* Option to only permute the matrix for LORASC, do not build the preconditioner */

  /* Computed during the build but accessible by the user */
  int *partCount;           /* array of size P (idxRowCount[i] indicates the number of rows for processor i in the permuted matrix)*/
  int *partBegin;          /* array of size P+1 (idxRowBegin[i] indicates the position of the first row of processor i in the permuted matrix)*/
  int *perm;               /* The permutation applied by lorasc to construct the block Arrow structure and distribute */
  double *eigvalues;       /* The eigenvalues from the eigenvalue problem of lorasc */
  double *eigvectors;      /* The eigenvectors from the eigenvalue problem of lorasc (allocated on the root only) */
  int eigvalues_deflation; /* Number of eigenvalues selected for the deflation*/

  /* infos about the separator  also accessible by the user */
  int *sep_mcounts;
  int *sep_moffsets;
  int sep_nrows;

  /* */
  MPI_Comm comm;            /* The MPI communicator */
  int nev;                  /* Number of eigenvalues computed */


  /* Lorasc required matrices */
  CPLM_Mat_CSR_t *Aii;
  CPLM_Mat_CSR_t *Aig;
  CPLM_Mat_CSR_t *Agi;
  CPLM_Mat_CSR_t *Aggloc;

  /* Lorasc required solvers object */
  preAlps_solver_t *Aii_sv;
  preAlps_solver_t *Agg_sv;



  /* workspace */
  double *vi; //buffer with the same size as the number of rows of Aii
  double *zi; //buffer with the same size as the number of rows of Aii
  double *dwork1;
  double *dwork2;
  double *eigWork; //a buffer with the same as the number of eigenvalues computed * nrhs
  double *sigma;   /* Array of the same size as the number of eigenvalues computed */
} preAlps_Lorasc_t;


/*Allocate workspace for the preconditioner*/
int preAlps_LorascAlloc(preAlps_Lorasc_t **lorasc);

/*
 * Build the preconditioner
 * lorasc:
 *     input: the preconditioner object to construct
 * A:
 *     input: the input matrix on processor 0
 * locAP:
 *     output: the local permuted matrix on each proc after the preconditioner is built
*/
int preAlps_LorascBuild(preAlps_Lorasc_t *lorasc, CPLM_Mat_CSR_t *A, CPLM_Mat_CSR_t *locAP, MPI_Comm comm);


/*Destroy the preconditioner*/
int preAlps_LorascDestroy(preAlps_Lorasc_t **lorasc);

/*
 * Apply Lorasc preconditioner on a dense matrice
 * i.e Compute  W = M_{lorasc}^{-1} * V
 */
int preAlps_LorascMatApply(preAlps_Lorasc_t *lorasc, CPLM_Mat_Dense_t *X, CPLM_Mat_Dense_t *Y);

/* Free internal allocated workspace */
int preAlps_LorascMatApplyWorkspaceFree(preAlps_Lorasc_t *lorasc);

/* Allocate workspace if required */
int preAlps_LorascMatApplyWorkspacePrepare(preAlps_Lorasc_t *lorasc, int Aii_m, int separator_m, int v_nrhs);
#endif
