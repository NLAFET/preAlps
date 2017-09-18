/*
============================================================================
Name        : eigsolve.h
Author      : Simplice Donfack
Version     : 0.1
Description : eigenvalues interface using ARPACK
Date        : Sept 13, 2017
============================================================================
*/
#ifndef EIGSOLVER_H
#define EIGSOLVER_H

#include <mpi.h>

typedef struct{

  /*Input*/
  char bmat;      /*Standard or generalized problem*/
  char which[2];  /*Larger or smaller eigenvalues*/
  int maxit;      /*maximum number of iterations*/

  int iparam[11];
  int ipntr[14]; /*Maximum size of ipntr for the routines pdnaupd and pdsaupd */

  //double deflation_tolerance; //the deflation tolerance, all eigenvalues lower than this will be selected for deflation
  double residual_tolerance; // The tolerance of the arnoldi iterative solver

  int nev;   /* The number of eigenvalues to compute */
  int nevComputed; /* Number of nev computed*/
  int issym; /*The problem to solve is symmetric*/

  /*Output*/

  int info;
  //int ido;
  int RCI_iter; /* Number of RCI_iterations */
  int OPX_iter; /* Number of matrix vector product (case 1 of ARPACK: Y = inv(A)*B*X )*/
  int BX_iter; /* Number of matrix vector product (case 2 of ARPACK: Y = B*X )*/
  double *eigvalues; /* The eigenvalues computed during the build of the solver */

  /*Internal*/
  int ncv;
  int lworkl;
  double *resid;
  double *v;
  int ldv;

  int m; /*the global problem size*/

  /* Workspace */
  //int *mdispls;
  double *workd;
  double *workl;

  /*Times*/
  double tEigValues;
  double tEigVectors;

} Eigsolver_t;


/* Create an eigensolver object */
int Eigsolver_create(Eigsolver_t **eigs);

/*Initialize the solver and allocate workspace*/
int Eigsolver_init(Eigsolver_t *eigs, int m, int mloc);

/* Terminate the solver and free the allocated workspace*/
int Eigsolver_finalize(Eigsolver_t **eigs);

/* Set the default parameters for the solver*/
int Eigsolver_setDefaultParameters(Eigsolver_t *eigs);

/* Perform one iteration of the eigensolver and return the hand to the RCI*/
int Eigsolver_iterate(Eigsolver_t *eigs, MPI_Comm comm, int mloc, double **X, double **Y, int *ido);
#endif
