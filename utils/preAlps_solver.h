/*
============================================================================
Name        : preAlps_solver.h
Author      : Simplice Donfack
Version     : 0.1
Description : Provide a wrapper for external solvers use by preAlps (Pardiso, mkl_pardiso, umfpack, petsc (seq), mumps, petsc (mpi),).
Each solver enumerated here should be able to solve Ax = b
Compile preAlps with USE_SOLVER_<name> to enable any of them.
Date        : Mai 31, 2017
============================================================================
*/

#ifndef PREALPS_SOLVER_H
#define PREALPS_SOLVER_H

/* Include all supported solvers */
#include <mpi.h>

#ifdef USE_SOLVER_MKL_PARDISO
#include "mkl_pardiso_solver.h"
#endif

#ifdef USE_SOLVER_PARDISO
#include "pardiso_solver.h"
#endif

#ifdef USE_SOLVER_MUMPS
#include "mumps_solver.h"
#endif





/* Which sequential solver to use for sparse matrices */
typedef enum {
  SOLVER_MKL_PARDISO = 0, /* PARDISO Solver from MKL */
  SOLVER_PARDISO,    /* PARDISO Solver from http://www.pardiso-project.org/ */
  SOLVER_MUMPS       /* Parallel version of Mumps but always called with MPI_COMM_SELF */
} preAlps_solver_type_t;

/* Which parallel solver to use for sparse matrices (NOT YET SUPPORTED) */
/*
typedef enum {
  SOLVER_MUMPS,       // MUMPS parallel solver
  SOLVER_PETSC        // PETSC parallel solver
} preAlps_parallel_solver_type_t;
*/

typedef enum {
  SOLVER_MATRIX_REAL_NONSYMMETRIC,
  SOLVER_MATRIX__REAL_SYMMETRIC,
} preAlps_solver_matrix_type_t;

typedef struct
{

  preAlps_solver_type_t type;

  MPI_Comm comm;

  #if defined (USE_SOLVER_MKL_PARDISO)
    mkl_pardiso_solver_t mkl_pardiso_ps;
  #endif

  #if defined (USE_SOLVER_PARDISO)
    pardiso_solver_t pardiso_ps;
  #endif

  #if defined (USE_SOLVER_MUMPS)
    mumps_solver_t mumps_ps;
  #endif

} preAlps_solver_t;

/*
 * Create a new solver
 * comm: use NULL for MPI_COMM_SELF for the sequential version
 */
int preAlps_solver_create(preAlps_solver_t **solver, preAlps_solver_type_t stype, MPI_Comm comm);

/*Destroy the solver*/
int preAlps_solver_destroy(preAlps_solver_t **solver);

/* Factorize a matrix A using the default solver */
int preAlps_solver_factorize(preAlps_solver_t *solver, int n, double *a, int *ia, int *ja);

/* Get the permutation vector used by the solver */
int* preAlps_solver_getPerm(preAlps_solver_t *solver);

/* Release internal memory */
int preAlps_solver_finalize(preAlps_solver_t *solver, int n, int *ia, int *ja);

/* Default initialisation of the selected solver */
int preAlps_solver_init(preAlps_solver_t *solver);

/* Factorize a matrix A using the default solver and return the schur complement*/
int preAlps_solver_partial_factorize(preAlps_solver_t *solver, int n, double *a, int *ia, int *ja, int S_n,
                                          double **S, int **iS, int **jS);

/* Set the type of the matrix to factorize */
int preAlps_solver_setMatrixType(preAlps_solver_t *solver, preAlps_solver_matrix_type_t matrix_type);

/* Solve A x = b with the previous factorized matrix*/
int preAlps_solver_triangsolve(preAlps_solver_t *solver, int n, double *a, int *ia, int *ja, double *x, double *b);
#endif
