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

#define MAX_ERROR_REPORTING_TRIANGULAR_SOLVE 5

typedef struct
{

  MPI_Comm comm;

  //buffer to store the matrix as COO
  int *irn;
  int *jcn;

  int m_glob;     //global problem size
  int idxRowPos;  //the starting position of the processor calling the initialization routine in the distributed case in the global matrix
  int nrhs;       //The number of rhs for the analysis and the solve phase

  DMUMPS_STRUC_C id;

  // The number of times a triangular solve error is reported,
  //this is useful to prevent huge log file when using triangular solve several times
  int error_reporting_triangular_solve;

} mumps_solver_t;


/* Initialize mumps structure*/
int mumps_solver_init(mumps_solver_t *solver, MPI_Comm comm);


/* Perform the factorization of the matrix,
*/
int mumps_solver_factorize(mumps_solver_t *solver, int n, double *a, int *ia, int *ja);


/* Perform the partial factorization of the matrix,
 * and compute S = A_{22} - A_{21}A_{11}^{-1}A_{12}
 * The factored part of the matrix can be use to solve the system A_{11}x= b1;
 * (S, iS,jS) is the returned schur complement
 * if S_n=0, the schur complement is not computed
*/
int mumps_solver_partial_factorize(mumps_solver_t *solver, int n, double *a, int *ia, int *ja, int S_n,
                                            double **S, int **iS, int **jS);


void mumps_solver_finalize(mumps_solver_t *solver, int n, int *ia, int *ja);

/*Solve Ax = b using mumps */
int mumps_solver_triangsolve(mumps_solver_t *solver, int n, double *a, int *ia, int *ja, int nrhs, double *x, double *b);

#endif
