/*
============================================================================
Name        : preAlps_solver.h
Author      : Simplice Donfack
Version     : 0.1
Description : Provide a wrapper for external solvers use by preAlps (Pardiso, mkl_pardiso, mumps (with mpi_comm_self), umfpack, petsc (seq), petsc (mpi)).
Each solver enumerated here should be able to solve Ax = b
Compile preAlps with USE_SOLVER_<name> to enable any of them.
Date        : Mai 31, 2017
============================================================================
*/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "preAlps_solver.h"
#include "preAlps_utils.h"


/*
 * Create a new solver
 * comm: use NULL for MPI_COMM_SELF for the sequential version
 */
int preAlps_solver_create(preAlps_solver_t **solver, preAlps_solver_type_t stype, MPI_Comm comm){

  *solver = (preAlps_solver_t*) malloc(sizeof(preAlps_solver_t));

  if(*solver==NULL) return 1;

  switch(stype){
    case SOLVER_MKL_PARDISO:
    break;
    case SOLVER_PARDISO:
    break;
    case SOLVER_MUMPS:
    break;
    default :
    preAlps_abort("*** Please select a valid sparse solver");
  }

  (*solver)->type = stype;
  (*solver)->comm = comm;

  return 0;
}

/*Destroy the solver*/
int preAlps_solver_destroy(preAlps_solver_t **solver){

  free(*solver);
  return 0;
}


/* Factorize a matrix A using the default solver */
int preAlps_solver_factorize(preAlps_solver_t *solver, int n, double *a, int *ia, int *ja){
  int ierr = 0;

  if(solver->type==SOLVER_MKL_PARDISO){
    #if defined (USE_SOLVER_MKL_PARDISO)
      ierr = mkl_pardiso_factorize(&solver->mkl_pardiso_ps, n, a, ia, ja);
    #endif
  } else if(solver->type==SOLVER_PARDISO){
    #if defined (USE_SOLVER_PARDISO)
      ierr = pardiso_solver_factorize(&solver->pardiso_ps, n, a, ia, ja);
    #endif
  } else{
      preAlps_abort("factorize() not yet supported for this solver");
  }

  return ierr;
}

/* Release internal memory */
int preAlps_solver_finalize(preAlps_solver_t *solver, int n, int *ia, int *ja){


  if(solver->type==SOLVER_MKL_PARDISO){
    #if defined (USE_SOLVER_MKL_PARDISO)
      mkl_pardiso_solver_finalize(&solver->mkl_pardiso_ps, n, ia, ja);
    #endif
  } else if(solver->type==SOLVER_PARDISO){
    #if defined (USE_SOLVER_PARDISO)
      pardiso_solver_finalize(&solver->pardiso_ps, n, ia, ja);
    #endif
  } else if(solver->type==SOLVER_MUMPS){

    #if defined (USE_SOLVER_MUMPS)
      mumps_solver_finalize(&solver->mumps_ps, n, ia, ja);
    #endif
  }

  return 0;
}


/* Get the permutation vector used by the solver */
int* preAlps_solver_getPerm(preAlps_solver_t *solver){

  if(solver->type==SOLVER_MKL_PARDISO){
    #if defined (USE_SOLVER_MKL_PARDISO)
      return solver->mkl_pardiso_ps.perm;
    #endif
  } else if(solver->type==SOLVER_PARDISO){
    #if defined (USE_SOLVER_PARDISO)
      return solver->pardiso_ps.perm;
    #endif
  } else {
      preAlps_abort("getPerm not yet supported for this solver");
  }

  return NULL;
}

/*
 * Default initialisation of the selected solver
 */
int preAlps_solver_init(preAlps_solver_t *solver){

  if(solver->type==SOLVER_MKL_PARDISO){
    #if defined (USE_SOLVER_MKL_PARDISO)
      mkl_pardiso_solver_init(&solver->mkl_pardiso_ps);
    #else
      preAlps_abort("*** Please compile preAlps with MKL");
    #endif
  }

  if(solver->type==SOLVER_PARDISO){
    #if defined (USE_SOLVER_PARDISO)
      pardiso_solver_init(&solver->pardiso_ps);
    #else
      preAlps_abort("*** Please compile preAlps with PARDISO");
    #endif
  }

  if(solver->type==SOLVER_MUMPS){
    #if defined (USE_SOLVER_MUMPS)
      mumps_solver_init(&solver->mumps_ps, solver->comm);
    #else
      preAlps_abort("*** Please compile preAlps with MUMPS");
    #endif
  }


    return 0;
}

/* Factorize a matrix A using the default solver and return the schur complement*/
int preAlps_solver_partial_factorize(preAlps_solver_t *solver, int n, double *a, int *ia, int *ja, int S_n,
                                          double **S, int **iS, int **jS){

  int ierr = 0;


  if(solver->type==SOLVER_MKL_PARDISO){
    #if defined (USE_SOLVER_MKL_PARDISO)

    mkl_pardiso_solver_partial_factorize(&solver->mkl_pardiso_ps, n,
                                         a, ia, ja,
                                         S_n, S, iS, jS);
    #endif
  } else if(solver->type==SOLVER_PARDISO){
    #if defined (USE_SOLVER_PARDISO)
    pardiso_solver_partial_factorize(&solver->pardiso_ps, n,
                                        a, ia, ja,
                                        S_n, S, iS, jS);
    #endif
  }
  else if(solver->type==SOLVER_MUMPS){
    #if defined (USE_SOLVER_MUMPS)
    mumps_solver_partial_factorize(&solver->mumps_ps, n,
                                        a, ia, ja,
                                        S_n, S, iS, jS);
    #endif
  }else { //if(solver->type==SOLVER_MUMPS){
      preAlps_abort("partial_factorize() not supported for this solver");
  }

  return ierr;
}


/* Set the type of the matrix to factorize */
int preAlps_solver_setMatrixType(preAlps_solver_t *solver, preAlps_solver_matrix_type_t matrix_type){


  if(solver->type==SOLVER_MKL_PARDISO){
    #if defined (USE_SOLVER_MKL_PARDISO)
      if(matrix_type==SOLVER_MATRIX_REAL_NONSYMMETRIC) solver->mkl_pardiso_ps.mtype = 11;
      else preAlps_abort("Matrix type not supported for this solver");
    #endif
  }
  else if(solver->type==SOLVER_PARDISO){
    #if defined (USE_SOLVER_PARDISO)
      if(matrix_type==SOLVER_MATRIX_REAL_NONSYMMETRIC) solver->pardiso_ps.mtype = 11;
      else preAlps_abort("Matrix type not supported for this solver");
    #endif
  } else { //if(solver->type==SOLVER_MUMPS){
      preAlps_abort("setMatrixType() not supported for this solver");
  }

  return 0;
}

/* Solve A x = b with the previous factorized matrix*/
int preAlps_solver_triangsolve(preAlps_solver_t *solver, int n, double *a, int *ia, int *ja, double *x, double *b){

  if(solver->type==SOLVER_MKL_PARDISO){
    #if defined (USE_SOLVER_MKL_PARDISO)
      mkl_pardiso_solver_triangsolve(&solver->mkl_pardiso_ps, n, a, ia, ja, x, b);
    #endif
  } else if(solver->type==SOLVER_PARDISO){
    #if defined (USE_SOLVER_PARDISO)
      pardiso_solver_triangsolve(&solver->pardiso_ps, n, a, ia, ja, x, b);
    #endif
  }
  else {
      preAlps_abort("triangsolve() not yet supported for this solver");
  }

    return 0;
}
