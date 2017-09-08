/******************************************************************************/
/* Author     : Olivier Tissot                                                */
/* Creation   : 2016/08/03                                                    */
/* Description: Generic iterative solver                                      */
/******************************************************************************/

/******************************************************************************/
/*                                  INCLUDE                                   */
/******************************************************************************/
#ifndef SOLVER_H
#define SOLVER_H

/* ParBBCG */
#include <usr_param.h>
/* STD */
#include <stdlib.h>
#include <stdio.h>
/* CPaLAMeM */
#include <cpalamem_macro.h>
#include <cpalamem_instrumentation.h>
#include <mat_csr.h>
#include <mat_dense.h>
#include <dvector.h>
/* MKL */
#include <mkl.h>
/* MPI */
#include <mpi.h>
/******************************************************************************/

/******************************************************************************/
/*                                    CODE                                    */
/******************************************************************************/
#define OFLEN 100
typedef struct {
  MPI_Comm     comm;      /* MPI Communicator */
  int          N;         /* Dimension of the global problem */
  int          n;         /* Dimension of the local problem */
  int          NNZ;       /* Global number of unknowns */
  int          nnz;       /* Local number of unknowns */
  const char*  name;      /* Method name */
  Usr_Param_t  param;     /* User parameters */
  double*      error;     /* Error */
  double       finalRes;  /* Final residual */
  char         oFileName[OFLEN]; /* Output file */
} Solver_t;

/* Utils */
int SolverInit(Solver_t* solver, int M, int m, Usr_Param_t param, const char* name);
void SolverFree(Solver_t* solver);
/******************************************************************************/

#endif
