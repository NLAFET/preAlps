/*
============================================================================
Name        : mumps_solver.h
Author      : Simplice Donfack
Version     : 0.1
Description : Wrapper for mumps functions.
Date        : July 8, 2017
============================================================================
*/
#ifndef MUMPS_SOLVER_H
#define MUMPS_SOLVER_H

#include<mpi.h>

#include "dmumps_c.h"
#define JOB_INIT -1
#define JOB_END -2

#define ICNTL(I) icntl[(I)-1] /* macro s.t. indices match documentation */

typedef struct
{

  MPI_Comm comm;

  //buffer to store the matrix as COO
  int *irn;
  int *jcn;

  DMUMPS_STRUC_C id;

} mumps_solver_t;


/* Initialize mumps structure*/
int mumps_solver_init(mumps_solver_t *solver, MPI_Comm comm);

/* Perform the partial factorization of the matrix,
 * and compute S = A_{22} - A_{21}A_{11}^{-1}A_{12}
 * The factored part of the matrix can be use to solve the system A_{11}x= b1;
 * (S, iS,jS) is the returned schur complement
 * if S_n=0, the schur complement is not computed
*/
int mumps_solver_partial_factorize(mumps_solver_t *solver, int n, double *a, int *ia, int *ja, int S_n,
                                            double **S, int **iS, int **jS);


void mumps_solver_finalize(mumps_solver_t *solver, int n, int *ia, int *ja);

#endif
