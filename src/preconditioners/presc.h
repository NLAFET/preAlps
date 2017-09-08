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

#include "preAlps_solver.h"


/* Which eigenvalue problem to solve */
typedef enum {
  PRESC_EIGS_SLOC = 0, /* Solve the eigenproblem S*u = \lambda*Sloc*u */
  PRESC_EIGS_ALOC,    /* Solve the eigenproblem S*u = \lambda*Aloc*u */
} presc_eigs_kind_t;


/*Structure of the preconditioner*/
typedef struct{
  //int m;
  int nev; /* number of eigenvalues computed */
  double *eigvalues; /* The eigenvalues computed during the build of the solver */
  int eigvalues_deflation; /*Number of eigenvalues selected for the deflation*/
  presc_eigs_kind_t eigs_kind; /* Which eigenvalue problem to solve */
} Presc_t;





/*Allocate workspace for the preconditioner*/
int Presc_alloc(Presc_t **presc);

/*
 * Build the preconditioner
 * presc:
 *     input: the preconditioner object to construct
 * A:
 *     input: the input matrix on processor 0
 * locAP:
 *     output: the local permuted matrix on each proc after the preconditioner is built
 *
*/
int Presc_build(Presc_t *presc, Mat_CSR_t *A, Mat_CSR_t *locAP, MPI_Comm comm);

/*Destroy the preconditioner*/
int Presc_destroy(Presc_t **presc);

/*
 * Solve the eigenvalues problem (I + AggP)S_{loc}^{-1}u = \lambda u using arpack.
 * Where AggP and S_{loc} are two sparse matrices.
*  AggP is formed by the offDiag elements of Agg, and S_{loc} = Block-Diag(S);
 * Check the paper for the structure of AggP and S_{loc}.
 *
 * m:
 *    the number of rows of the global matrice. Also correspond to the size of the eigenvectors.
 * mcounts:
 *    input: the local of rows of each processor.
 * Sloc_sv
 *    input: the solver object to apply to compute  Sloc^{-1}v
*/
int Presc_eigSolve(Presc_t *presc, int *mcounts, preAlps_solver_t *Sloc_sv, Mat_CSR_t *Sloc, Mat_CSR_t *AggP, MPI_Comm comm);




/*Initialize arpack with default parameters*/
int Presc_eigSolve_init(char *bmat, char *which, int *maxit, int *iparam, double *eigs_tolerance, double *deflation_tol);


/*
 * Solve the eigenvalues problem Su = \lambda Aloc u using arpack.
 * Where S and Aloc are two sparse matrices.
 *
 * S = Agg - Agi*Aii^{-1}*Agi
 * mloc:
 *    the local problem size.
*/
int Presc_eigSolve_SAloc(Presc_t *presc, int mloc, Mat_CSR_t *Aggloc, Mat_CSR_t *Agi, Mat_CSR_t *Aii, Mat_CSR_t *Aig, Mat_CSR_t *Aloc
                                       , preAlps_solver_t *Aii_sv, preAlps_solver_t *Aloc_sv, MPI_Comm comm);

/* Allocate workspace*/
int Presc_eigSolve_workspace_alloc(int m, int ncv, int ldv, int lworkl, double **resid, double **v, double **workd, double **workl);

/* Free workspace*/
int Presc_eigSolve_workspace_free(double **resid, double **v, double **workd, double **workl);

/*
 * Solve the eigenvalues problem SS_{loc}^{-1}u = \lambda u using arpack.
 * Where S is the schur complement, and S_{loc} = Block-Diag(S);
 * Check the paper for the structure of S and S_{loc}.
 *
 * m:
 *    the number of rows of the global matrice. Also correspond to the size of the eigenvectors.
 * mloc:
 *    input: the local number of rows owned by this processors.
 * Sloc_sv
 *    input: the solver object to apply to compute  Sloc^{-1}v
*/
//int Presc_eigSolve_1(Presc_t *presc, int m, int mloc, preAlps_solver_t *Sloc_sv, MPI_Comm comm);


#endif
