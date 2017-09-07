/*
============================================================================
Name        : mkl_pardiso_solver.h
Author      : Simplice Donfack
Version     : 0.1
Description : Wrapper for pardiso functions. The following functions are based
on the pardiso lib from MKL
Date        : Mai 24, 2017
============================================================================
*/
#ifndef MKL_PARDISO_SOLVER_H
#define MKL_PARDISO_SOLVER_H
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

} mkl_pardiso_solver_t;

/* Initialize pardiso structure*/
int mkl_pardiso_solver_init(mkl_pardiso_solver_t *ps);

/* Factorize the matrix */
int mkl_pardiso_factorize(mkl_pardiso_solver_t *ps, int n, double *a, int *ia, int *ja);

void mkl_pardiso_solver_finalize(mkl_pardiso_solver_t *ps, int n, int *ia, int *ja);

/* Perform the partial factorization of the matrix,
 * and compute S = A_{22} - A_{21}A_{11}^{-1}A_{12}
 * The factored part of the matrix can be use to solve the system A_{11}x= b1;
 * (S, iS,jS) is the returned schur complement
 * if S_n=0, the schur complement is not computed
*/
int mkl_pardiso_solver_partial_factorize(mkl_pardiso_solver_t *ps, int n, double *a, int *ia, int *ja, int S_n,
                                            double **S, int **iS, int **jS);

/*Solve Ax = b using pardiso*/
int mkl_pardiso_solver_triangsolve(mkl_pardiso_solver_t *ps, int n, double *a, int *ia, int *ja, double *x, double *b);
#endif
