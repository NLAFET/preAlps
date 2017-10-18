/*
============================================================================
Name        : mkl_pardiso_solver.c
Author      : Simplice Donfack
Version     : 0.1
Description : Wrapper for pardiso functions. The following functions are based
on the pardiso lib from MKL. NOTE: this is only compatible with PARDISO FROM INTEL MKL
Date        : Mai 24, 2017
============================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include "mkl_pardiso_solver.h"

#ifdef DEBUG
#include<mpi.h>
#endif
/*
 * PARDISO functions prototype.
 */
//void pardisoinit (void   *, int    *,   int *, int *, double *, int *);

void pardiso (void *pt, const int *maxfct, const int *mnum,
              const int *mtype, const int *phase, const int *n,
              const void *a, const int *ia, const int *ja,
              int *perm, const int *nrhs, int *iparm,
              const int *msglvl, void *b, void *x, int *error);


/* Initialize pardiso structure*/
int mkl_pardiso_solver_init(mkl_pardiso_solver_t *ps){

/* -------------------------------------------------------------------- */
/* ..  Setup Pardiso control parameters.                                */
/* -------------------------------------------------------------------- */

    /* Auxiliary variables. */
    int i;

    ps->error = 0;	  /* Initialize error flag */

    ps->solver=0;	    /* use sparse direct solver */

    ps->msglvl = 0;   /* Print statistical information  */
    ps->mtype = 11;   /* Real and non symmetric Could be changed by the user*/
    ps->maxfct = 1;	  /* Maximum number of numerical factorizations.  */
    ps->mnum   = 1;   /* The matrix number to factorize. */
    ps->nrhs = 1;     /* Default number of nrhs, could be changed by the user*/
    /* -------------------------------------------------------------------- */
    /* .. Initialize the internal solver memory pointer. This is only */
    /* necessary for the FIRST call of the PARDISO solver. */
    /* -------------------------------------------------------------------- */
    	for (i = 0; i < 64; i++) {
    		ps->pt[i] = 0;
        ps->iparm[i] = 0;
    	}

    ps->iparm[0] = 1;  /* Not the default solver. We supply all values of iparm */
    ps->iparm[1] = 2;  /* Fill-in reordering from METIS */
    ps->iparm[4]  = 2; /*Return the ordering permutation*/
    ps->iparm[34] = 1; /* Zero-based indexing (C-style)*/
#ifdef DEBUG

    ps->iparm[17] = -1;//Report the number of non-zero elements in the factors
    ps->iparm[18] = -1;//Report number of floating point operations

    ps->iparm[26] = 1; printf("[MKL_Pardiso_solver] Check of the matrix activated in mkl_pardiso_solver_partial_factorization()\n");//debugg
#endif

    ps->perm = NULL;  /* allocated during the factorization*/
    return 0;
}

/* Factorize the matrix */
int mkl_pardiso_factorize(mkl_pardiso_solver_t *ps, int n, double *a, int *ia, int *ja){


int ierr=0, S_n = 0;
double **ddum = NULL; //schur complement not needed
int **idum = NULL;

ierr = mkl_pardiso_solver_partial_factorize(ps, n, a, ia, ja, S_n, ddum, idum, idum);

return ierr;

}

/* Perform the partial factorization of the matrix,
 * and compute S = A_{22} - A_{21}A_{11}^{-1}A_{12}
 * The factored part of the matrix can be use to solve the system A_{11}x= b1;
 * (S, iS,jS) is the returned schur complement
 * if S_n=0, the schur complement is not computed
*/
int mkl_pardiso_solver_partial_factorize(mkl_pardiso_solver_t *ps, int n, double *a, int *ia, int *ja, int S_n,
                                            double **S, int **iS, int **jS){

  //int  idum;              /* Integer dummy. */
  double   ddum;              /* Double dummy */

  // Reorder and factor a matrix sytem
  int phase = 12;

  ps->error  = 0;         /* Initialize error flag */

  //int      nnz = n>0?ia[n]:0;
  int i,j;

  double *Swork = NULL;

  /* Allocate workspace for the schur complement */
  if(S_n>0) Swork = (double*) malloc((S_n * S_n)*sizeof(double));

  //ps->iparm[4] = 1; /* use user supply perm*/
  //ps->iparm[30] = 0; /* Disable perm*/

  ps->perm = malloc(n * sizeof(int));


  if(!ps->perm) {
    printf("[MKL_Pardiso_solver] malloc fails for perm in mkl_pardiso_solver_partial_factorization()\n");
    exit(1);
  }

  for(i=0;i<n;i++) ps->perm[i] = 0;

  //use user perm
  //ps->iparm[4] = 1; /* use user supply perm*/
  ///ps->iparm[30] = 0; /* Disable perm*/
  //for(i = 0;i<n;i++) ps->perm[i] = i;


  if(S_n>0){

    ps->iparm[35] = 2; /* Compute the schur complement */
    //ps->iparm[35] = 1; /* Compute only the schur complement */

    /* Indicate that the last rows of the matrix are part of the Schur complement */
    //for(i = 0;i<n - S_n;i++) ps->perm[i] = 0;
    for(i = n - S_n;i<n;i++) ps->perm[i] = 1;

  }
  else
    ps->iparm[35] = 0; /* No need to compute the schur complement */






  /* Factorize the matrix */
  /*
  pardiso (ps->pt, &(ps->maxfct), &(ps->mnum), &(ps->mtype), &phase,
          &n, a, ia, ja, perm, &(ps->nrhs), ps->iparm, &(ps->msglvl), &ddum, S, &(ps->error));

  if (ps->error != 0) {
      printf("ERROR during the factorization: %d\n", ps->error);
      exit(2);
  }

  //printf("Factorization completed ... \n");
  printf("Number of nonzeros in factors  = %d\n", ps->iparm[17]);
  printf("Number of factorization MFLOPS = %d\n", ps->iparm[18]);
  */

#if 0
  /* -------------------------------------------------------------------- */
/* .. Reordering and Symbolic Factorization. This step also allocates   */
/* all memory that is necessary for the factorization.                  */
/* -------------------------------------------------------------------- */

    phase = 11;
    pardiso (ps->pt, &(ps->maxfct), &(ps->mnum), &(ps->mtype), &phase,
             &n, a, ia, ja, ps->perm, &(ps->nrhs), ps->iparm, &(ps->msglvl), &ddum, &ddum, &(ps->error));
    if ( ps->error != 0 )
    {
        printf ("ERROR during symbolic factorization: %d\n", ps->error);
        exit (1);
    }
#ifdef DEBUG
    printf ("Reordering completed ... \n");
#endif

    //for(i=0;i<n;i++) printf(" REORDER perm[%d]:%d\n", i, ps->perm[i]);
    preAlps_intVector_printSynchronized(ps->perm, n, "REORDER perm", "REORDER perm", MPI_COMM_WORLD);
   //use user perm
   //ps->iparm[4] = 1; /* use user supply perm*/
   //ps->iparm[30] = 0; /* Disable perm*/
   //for(i = 0;i<n;i++) ps->perm[i] = i;


/* -------------------------------------------------------------------- */
/* .. Numerical factorization. */
/* -------------------------------------------------------------------- */

    phase = 22;
    pardiso(ps->pt, &(ps->maxfct), &(ps->mnum), &(ps->mtype), &phase,
            &n, a, ia, ja, ps->perm, &(ps->nrhs),
            ps->iparm, &(ps->msglvl), &ddum, Swork, &(ps->error));


#else
  phase = 12;
  pardiso (ps->pt, &(ps->maxfct), &(ps->mnum), &(ps->mtype), &phase,
         &n, a, ia, ja, ps->perm, &(ps->nrhs), ps->iparm, &(ps->msglvl), &ddum, Swork, &(ps->error));
#endif

#ifdef DEBUG
  printf ("Factorization completed ... \n");
  //printf("Factorization completed ... \n");
  printf("Number of pertubed pivots: %d\n", ps->iparm[13]);
  printf("Number of nonzeros in factors: %d\n", ps->iparm[17]);
  printf("Number of factorization MFLOPS: %d\n", ps->iparm[18]);
  printf("Number of negatives or zeros pivot: %d\n", ps->iparm[29]);
#endif

  //hack test
  //for(i=0;i<n;i++) printf(" FACT perm[%d]:%d\n", i, ps->perm[i]);

  //preAlps_intVector_printSynchronized(ps->perm, n, "FACT perm", "FACT perm", MPI_COMM_WORLD);

  if ( ps->error != 0 )
  {
      printf ("[MKL_Pardiso_solver] ERROR during the factorization: %d\n", ps->error);
      exit (2);
  }


  if(S_n>0) {

    /*convert to an CSR matrix*/

    int count_nnz = 0;
    for(int i=0;i<S_n * S_n;i++){
      if(Swork[i] != 0.0 ) count_nnz++;
    }

    /* Convert the matrix from Dense to CSR */

    *iS = (int*) malloc((S_n+1)*sizeof(int));
    *jS = (int*) malloc((count_nnz)*sizeof(int));
    *S = (double*) malloc((count_nnz)*sizeof(double));

    if(!*iS || !*jS || !*S){
      printf("[MKL_Pardiso_solver] Malloc fails for the CSR matrix S in mkl_solver_partial_factorize\n");
      exit(1);
    }


    int count=0;
    (*iS)[0]=0;
    for(i=0;i<S_n;i++) {
      for(j=0;j<S_n;j++){
        if(Swork[j*S_n+i] != 0.0 ) {
          (*jS)[count] = j;
          (*S)[count]  = Swork[j*S_n+i];
          count++;
        }
      }
      (*iS)[i+1]=count;
    }

    free(Swork);
  }


  return 0;
}

/*Solve Ax = b using pardiso*/
int mkl_pardiso_solver_triangsolve(mkl_pardiso_solver_t *ps, int n, double *a, int *ia, int *ja, int nrhs, double *x, double *b){

  //int      idum;              /* Integer dummy. */


/* -------------------------------------------------------------------- */
/* ..  Back substitution and iterative refinement.                      */
/* -------------------------------------------------------------------- */
    int phase = 33;

    //ps->iparm[7] = 0; //1      /* Max numbers of iterative refinement steps. */

    /*pardiso (ps->pt, &(ps->maxfct), &(ps->mnum), &(ps->mtype), &phase,
             &n, a, ia, ja, &idum, &(ps->nrhs),
             ps->iparm, &(ps->msglvl), b, x, &(ps->error));*/
    ps->nrhs = nrhs;

    pardiso (ps->pt, &(ps->maxfct), &(ps->mnum), &(ps->mtype), &phase,
                      &n, a, ia, ja, ps->perm, &(ps->nrhs),
                      ps->iparm, &(ps->msglvl), b, x, &(ps->error));

    if (ps->error != 0) {
        printf("[MKL_Pardiso_solver] ERROR during solution: %d\n", ps->error);
        exit(3);
    }

return 0;
}

void mkl_pardiso_solver_finalize(mkl_pardiso_solver_t *ps, int n, int *ia, int *ja){

    double   ddum;              /* Double dummy */
    int      idum;              /* Integer dummy. */

/* -------------------------------------------------------------------- */
/* ..  Termination and release of memory.                               */
/* -------------------------------------------------------------------- */
    int phase = -1;                 /* Release internal memory. */

    pardiso (ps->pt, &(ps->maxfct), &(ps->mnum), &(ps->mtype), &phase,
             &n, &ddum, ia, ja, &idum, &(ps->nrhs),
             ps->iparm, &(ps->msglvl), &ddum, &ddum, &(ps->error));

   if(ps->perm!=NULL) free(ps->perm);
}
