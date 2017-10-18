/*
============================================================================
Name        : pardiso_solver.h
Author      : Simplice Donfack
Version     : 0.1
Description : Wrapper for pardiso functions. The following functions are based
on the pardiso lib from USI
Date        : Mai 16, 2017
============================================================================
*/
#ifndef PARDISO_SOLVER_H
#define PARDISO_SOLVER_H
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


typedef struct
{

	/* Internal solver memory pointer pt,                  */
	/* 32-bit: int pt[64]; 64-bit: long int pt[64]         */
	/* or void *pt[64] should be OK on both architectures  */
	void    *pt[64];

	int      mtype;

	/* Pardiso control parameters. */
  int      iparm[64];
  double   dparm[64];

	/* Maximum number of numerical factorizations.  */
	int maxfct;

	/* Which factorization to use. */
	int mnum;

	int error;

	/*Which solver to use*/
	int solver;

	int msglvl;

	/* Number of right hand sides. */
	int nrhs;

	/*permutation vector*/
	int *perm;

} pardiso_solver_t;



int pardiso_solver_init(pardiso_solver_t *ps);
int pardiso_solver_factorize(pardiso_solver_t *ps, int n, double *a, int *ia, int *ja);
/* Perform the partial factorization of the matrix,
 * and compute S = A_{22} - A_{21}A_{11}^{-1}A_{12}
 * The factored part of the matrix can be use to solve the system A_{11}x= b1;
*/
int pardiso_solver_partial_factorize(pardiso_solver_t *ps, int n, double *a, int *ia, int *ja, int S_n,
                                          double **S, int **iS, int **jS);
int pardiso_solver_triangsolve(pardiso_solver_t *ps, int n, double *a, int *ia, int *ja, int nrhs, double *x, double *b);
void pardiso_solver_finalize(pardiso_solver_t *ps, int n, int *ia, int *ja);
#endif
