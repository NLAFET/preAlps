/*
============================================================================
Name        : presc.h
Author      : Simplice Donfack
Version     : 0.1
Description : Preconditioner based on Schur complement
Date        : Mai 15, 2017
============================================================================
*/
#ifndef PRESC_H
#define PRESC_H

#include <mpi.h>
#include "preAlps_solver.h"


/* Which eigenvalue problem to solve */
typedef enum {
  PRESC_EIGS_SSLOC = 0, /* Solve the eigenproblem S*u = \lambda*Sloc*u */
  PRESC_EIGS_SALOC,    /* Solve the eigenproblem S*u = \lambda*Aloc*u */
} presc_eigs_kind_t;


/*Structure of the preconditioner*/
typedef struct{

  /* User input*/
  double deflation_tol;        /* The deflation tolerance (default: 10^-2)*/
  int nrhs;                    /* The number of rhs on which lorasc will be applied. This is required internally for the analysis. Default value: 1 */
  presc_eigs_kind_t eigs_kind; /* Which eigenvalue problem to solve */

  /* Computed during the build but accessible by the user */
  double *eigvalues;       /* The eigenvalues from the eigenvalue problem of the preconditioner*/
  double *eigvectors;      /* The eigenvectors from the eigenvalue problem  (allocated on the root only) */
  int eigvalues_deflation; /* Number of eigenvalues selected for the deflation*/



  int nev;                /* number of eigenvalues computed */
  MPI_Comm comm;          /* The MPI communicator used to build the preconditioner. Should be the same to apply it. */


  /* infos about the separator  also accessible by the user */
  int *sep_mcounts;
  int *sep_moffsets;
  int sep_nrows;

  /* Presc required matrices */
  CPLM_Mat_CSR_t *Aii;
  CPLM_Mat_CSR_t *Aig;
  CPLM_Mat_CSR_t *Agi;
  CPLM_Mat_CSR_t *Aloc;
  CPLM_Mat_CSR_t *locAgg;

  /* Presc required solvers object */
  preAlps_solver_t *Aii_sv;
  preAlps_solver_t *Agg_sv;



  /* workspace */
  double *vi; //buffer with the same size as the number of rows of Aii
  double *zi; //buffer with the same size as the number of rows of Aii
  double *dwork1;
  double *dwork2;
  double *eigWork; //a buffer with the same as the number of eigenvalues computed * nrhs
  double *sigma;   /* Array of the same size as the number of eigenvalues computed */

} preAlps_Presc_t;





/*Allocate workspace for the preconditioner*/
int preAlps_PrescAlloc(preAlps_Presc_t **presc);

/*
 * Build the preconditioner
 * presc:
 *     input/output: the preconditioner object to construct
 * locAP:
 *     input: the local permuted matrix on each proc after the partitioning
 * partBegin:
 *    input: the global array to indicate the row partitioning
 * locNbDiagRows:
      input: the number of row in the diagonal as returned by preAlps_blockDiagODBStructCreate();
*/
int preAlps_PrescBuild(preAlps_Presc_t *presc, CPLM_Mat_CSR_t *locAP, int *partBegin, int locNbDiagRows, MPI_Comm comm);


/*Destroy the preconditioner*/
int preAlps_PrescDestroy(preAlps_Presc_t **presc);

/*
 * Apply Presc preconditioner on a dense matrice
 * i.e Compute  W = M_{presc}^{-1} * V
 */

int preAlps_PrescMatApply(preAlps_Presc_t *presc, CPLM_Mat_Dense_t *V, CPLM_Mat_Dense_t *W);

/* Free internal allocated workspace */
int preAlps_PrescMatApplyWorkspaceFree(preAlps_Presc_t *presc);

//Always check if the buffer has been allocated before, if so then to nothing.
int preAlps_PrescMatApplyWorkspacePrepare(preAlps_Presc_t *presc, int Aii_m, int separator_m, int v_nrhs);

/* Extract the local matrices Agi, Aii, Aig, Aloc from the matrix A*/
int preAlps_PrescSubMatricesExtract(CPLM_Mat_CSR_t *locA, int locNbDiagRows, CPLM_Mat_CSR_t *Agi, CPLM_Mat_CSR_t *Aii, CPLM_Mat_CSR_t *Aig,
CPLM_Mat_CSR_t *Aloc);
#endif
