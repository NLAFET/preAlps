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

#include <mat_csr.h>
#include <mat_dense.h>

/*Structure of the preconditioner*/
typedef struct{
  //int m;
  int nev;                /* number of eigenvalues computed */
  double deflation_tolerance; /* The deflation tolerance */
  double *eigvalues;      /* The eigenvalues computed during the build of the solver */
  int eigvalues_deflation; /*Number of eigenvalues selected for the deflation*/

} Lorasc_t;


/*Allocate workspace for the preconditioner*/
int Lorasc_alloc(Lorasc_t **lorasc);

/*
 * Build the preconditioner
 * lorasc:
 *     input: the preconditioner object to construct
 * A:
 *     input: the input matrix on processor 0
 * locAP:
 *     output: the local permuted matrix on each proc after the preconditioner is built
 *
*/
int Lorasc_build(Lorasc_t *lorasc, CPLM_Mat_CSR_t *A, CPLM_Mat_CSR_t *locAP, MPI_Comm comm);


/*Destroy the preconditioner*/
int Lorasc_destroy(Lorasc_t **lorasc);

#endif
