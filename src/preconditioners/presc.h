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


#endif
