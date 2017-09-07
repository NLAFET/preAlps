/*
============================================================================
Name        : pardiso_solver.c
Author      : Simplice Donfack
Version     : 0.1
Description : Wrapper for pardiso functions
Date        : Mai 16, 2017
============================================================================
*/
#include <stdio.h>
#include <stdlib.h>
#include "pardiso_solver.h"

/*
 * PARDISO functions prototype.
 */
void pardisoinit (void   *, int    *,   int *, int *, double *, int *);
void pardiso (void   *, int    *,   int *, int *,    int *, int *,
            double *, int    *,    int *, int *,   int *, int *,
               int *, double *, double *, int *, double *);
 void pardiso_chkvec     (int *, int *, double *, int *);
 void pardiso_printstats (int *, int *, double *, int *, int *, int *,
                            double *, int *);

 void pardiso_residual (int *mtype, int *n, double *a, int *ia, int *ja, double *b, double *x, double *y, double *normb, double *normr);
 void pardiso_get_schur(void *, int *, int *, int *, double *, int *, int *);
 void pardiso_chkmatrix  (int *, int *, double *, int *, int *, int *);
/*
 * Pardiso wrapper functions
 */

/* Initialize pardiso structure*/
int pardiso_solver_init(pardiso_solver_t *ps){
/* -------------------------------------------------------------------- */
/* ..  Setup Pardiso control parameters.                                */
/* -------------------------------------------------------------------- */


/* Number of processors. */
    int      num_procs;

    /* Auxiliary variables. */
    char    *var;
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
        ps->dparm[i] = 0;
    	}

    pardisoinit (ps->pt,  &(ps->mtype), &(ps->solver), ps->iparm, ps->dparm, &(ps->error));

    if (ps->error != 0)
    {
        if (ps->error == -10 )
           printf("No license file found \n");
        if (ps->error == -11 )
           printf("License is expired \n");
        if (ps->error == -12 )
           printf("Wrong username or hostname \n");
         return 1;
    }
    else{
 //       printf("[PARDISO]: License check was successful ... \n");
    }

    /* Numbers of processors, value of OMP_NUM_THREADS */
    var = getenv("OMP_NUM_THREADS");
    if(var != NULL)
        sscanf( var, "%d", &num_procs );
    else {
        printf("[Pardiso_solver] ***** Please set environment OMP_NUM_THREADS to 1 or greater\n");
        exit(1);
    }


    ps->iparm[2]  = num_procs;

    ps->perm = NULL; /*allocated during the factorization*/

    return 0;
}

/*Factorize a matrix A using pardiso*/
int pardiso_solver_factorize(pardiso_solver_t *ps, int n, double *a, int *ia, int *ja)
{


    int      nnz = ia[n];

    int      phase;




    double   ddum;              /* Double dummy */
    int      idum;              /* Integer dummy. */

    int shift_index = 1; /*0 or 1-based index, use 1 to convert matrix from 0-based C-notation to Fortran 1-based*/ /*TODO. Remove*/

    int      i;

    ps->error  = 0;         /* Initialize error flag */

    ps->iparm[2-1] = 0; /*Try minimun degree*/

    ps->perm = malloc(n * sizeof(int));

    if(!ps->perm) {
      printf("[Pardiso_solver] malloc fails for perm in mkl_pardiso_solver_partial_factorization()\n");
      exit(1);
    }

    for(i=0;i<n;i++) ps->perm[i] = 0; /*TODO: use pardiso options to set perm if required*/

/* -------------------------------------------------------------------- */
/* ..  Convert matrix from 0-based C-notation to Fortran 1-based        */
/*     notation.                                                        */
/* -------------------------------------------------------------------- */
if(shift_index!=0){
    for (i = 0; i < n+1; i++) {
        ia[i] += shift_index;
    }
    for (i = 0; i < nnz; i++) {
        ja[i] += shift_index;
    }
}


/* -------------------------------------------------------------------- */
/*  .. pardiso_chk_matrix(...)                                          */
/*     Checks the consistency of the given matrix.                      */
/*     Use this functionality only for debugging purposes               */
/* -------------------------------------------------------------------- */


    printf("[Pardiso_solver] Checking the consistency of the matrix activated.\n");
    pardiso_chkmatrix  (&(ps->mtype), &n, a, ia, ja, &(ps->error));
    if (ps->error != 0) {
        printf("[Pardiso_solver] ERROR %d in the consistency check of the matrix\n", ps->error);
        exit(1);
    }



/* -------------------------------------------------------------------- */
/* ..  Reordering and Symbolic Factorization.  This step also allocates */
/*     all memory that is necessary for the factorization.              */
/* -------------------------------------------------------------------- */
    phase = 11;

    pardiso (ps->pt, &(ps->maxfct), &(ps->mnum), &(ps->mtype), &phase,
	     &n, a, ia, ja, &idum, &(ps->nrhs),
             ps->iparm, &(ps->msglvl), &ddum, &ddum, &(ps->error), ps->dparm);

    if (ps->error != 0) {
        printf("[Pardiso_solver] ERROR during symbolic factorization: %d\n", ps->error);
        exit(1);
    }
/*
    printf("Reordering completed ... \n");
    printf("Number of nonzeros in factors  = %d\n", iparm[17]);
    printf("Number of factorization MFLOPS = %d\n", iparm[18]);
  */
/* -------------------------------------------------------------------- */
/* ..  Numerical factorization.                                         */
/* -------------------------------------------------------------------- */
    phase = 22;
    ps->iparm[32] = 0; //1 /* compute determinant */

    pardiso (ps->pt, &(ps->maxfct), &(ps->mnum), &(ps->mtype), &phase,
             &n, a, ia, ja, &idum, &(ps->nrhs),
             ps->iparm, &(ps->msglvl), &ddum, &ddum, &(ps->error),  ps->dparm);

    if (ps->error != 0) {
        printf("[Pardiso_solver] ERROR during numerical factorization: %d\n", ps->error);
        exit(2);
    }
//    printf("\nFactorization completed ...\n ");




/* -------------------------------------------------------------------- */
/* ..  Convert matrix back to 0-based C-notation.                       */
/* -------------------------------------------------------------------- */
if(shift_index!=0){
    for (i = 0; i < n+1; i++) {
        ia[i] -= shift_index;
    }
    for (i = 0; i < nnz; i++) {
        ja[i] -= shift_index;
    }
}

return 0;
}

/* Perform the partial factorization of the matrix,
 * and compute S = A_{22} - A_{21}A_{11}^{-1}A_{12}
 * The factored part of the matrix can be use to solve the system A_{11}x= b1;
*/
int pardiso_solver_partial_factorize(pardiso_solver_t *ps, int n, double *a, int *ia, int *ja, int S_n,
                                          double **S, int **iS, int **jS){

  int  idum;              /* Integer dummy. */
  double   ddum;              /* Double dummy */

  // Reorder and factor a matrix sytem
  int phase = 12;

  int shift_index = 1; /*0 or 1-based index, use 1 to convert matrix from 0-based C-notation to Fortran 1-based*/



  ps->error  = 0;         /* Initialize error flag */

  int nnz = n>0?ia[n]:0;
  int i, j, count;

/* -------------------------------------------------------------------- */
/* ..  Convert matrix from 0-based C-notation to Fortran 1-based        */
/*     notation.                                                        */
/* -------------------------------------------------------------------- */
if(shift_index!=0){
  for (i = 0; i < n+1; i++) {
      ia[i] += shift_index;
  }
  for (i = 0; i < nnz; i++) {
      ja[i] += shift_index;
  }
}


/* -------------------------------------------------------------------- */
/*  .. pardiso_chk_matrix(...)                                          */
/*     Checks the consistency of the given matrix.                      */
/*     Use this functionality only for debugging purposes               */
/* -------------------------------------------------------------------- */


  pardiso_chkmatrix  (&(ps->mtype), &n, a, ia, ja, &(ps->error));
  if (ps->error != 0) {
      printf("[Pardiso_solver] ERROR %d in the consistency check of the matrix\n", ps->error);
      exit(1);
  }


  ps->iparm[38-1] = S_n; /* Number of rows of the Schur complement */

  pardiso (ps->pt, &(ps->maxfct), &(ps->mnum), &(ps->mtype), &phase,
     &n, a, ia, ja, &idum, &(ps->nrhs),
           ps->iparm, &(ps->msglvl), &ddum, &ddum, &(ps->error), ps->dparm);

#ifdef DEBUG
  //printf("Factorization completed ... \n");
  printf("[Pardiso_solver] nonzeros in the Schur  = %d\n", ps->iparm[39-1]);
  printf("[Pardiso_solver] Number of nonzeros in factors  = %d\n", ps->iparm[17]);
  printf("[Pardiso_solver] Number of factorization MFLOPS = %d\n", ps->iparm[18]);
#endif

  // allocate memory for the Schur-complement and copy it there.
  int S_nnz = ps->iparm[39-1];

  *iS = (int*) malloc((S_n+1)*sizeof(int));
  if(!*iS){
    printf("[Pardiso_solver] Malloc fails for iS in pardiso_partial_factorization\n");
    exit(1);
  }

  if(S_n<=0){
      /*The matrix is empty*/
      *jS = NULL;
      *S = NULL;
  }else{

      *jS = (int*) malloc((S_nnz)*sizeof(int));
      *S = (double*) malloc((S_nnz)*sizeof(double));

      if(!*jS || !*S){
        printf("[Pardiso_solver] Malloc fails for the CSR matrix S in pardiso_partial_factorization\n");
        exit(1);
      }
  }

  pardiso_get_schur(ps->pt, &(ps->maxfct), &(ps->mnum), &(ps->mtype), *S, *iS, *jS);

  /* -------------------------------------------------------------------- */
  /* ..  Convert matrix back to 0-based C-notation.                       */
  /* -------------------------------------------------------------------- */
  if(shift_index!=0){

      for (i = 0; i < n+1; i++) {
          ia[i] -= shift_index;
      }
      for (i = 0; i < nnz; i++) {
          ja[i] -= shift_index;
      }

      for (i = 0; i < S_n+1; i++) {
        (*iS)[i] -= shift_index;
      }

      for (i = 0; i < S_nnz; i++) {
          (*jS)[i] -= shift_index;
      }

      #ifdef PARDISO_SCHUR_COMPLEMENT_PATCH
        //Try to fix the bug in the schur complement
        int max_n = S_n>0?1:0;
        for (i = 0; i < max_n; i++) {
          for(j=(*iS)[i] + 1 ;j<(*iS)[i+1];j++){
            (*jS)[j] += 1;
          }

        }
      #endif

  }


  /*
   * remove all implicit zeros in the Schur
   */

  count = 0;
  int *iwork = (int*) malloc((S_n+1)*sizeof(int)); //To avoid changing the index of the j loop (performance optimization)

  for (i = 0; i < S_n; i++) {
    for(j=(*iS)[i];j<(*iS)[i+1];j++){
      if((*S)[j]!= 0.0){
        (*jS)[count] = (*jS)[j];
        (*S)[count]  = (*S)[j];
        count++;
      }
    }

    iwork[i+1]=count;
  }

  for (i = 1; i < S_n+1; i++) (*iS)[i]=iwork[i];
  free(iwork);

  return 0;
}

/*Solve Ax = b using pardiso*/
int pardiso_solver_triangsolve(pardiso_solver_t *ps, int n, double *a, int *ia, int *ja, double *x, double *b){

  int      idum;              /* Integer dummy. */

/* -------------------------------------------------------------------- */
/* ..  pardiso_chkvec(...)                                              */
/*     Checks the given vectors for infinite and NaN values             */
/*     Input parameters (see PARDISO user manual for a description):    */
/*     Use this functionality only for debugging purposes               */
/* -------------------------------------------------------------------- */


    pardiso_chkvec (&n, &(ps->nrhs), b, &(ps->error));
    if (ps->error != 0) {
        printf("[Pardiso_solver] ERROR  in right hand side: %d\n", ps->error);
        exit(1);
    }

/* -------------------------------------------------------------------- */
/* .. pardiso_printstats(...)                                           */
/*    prints information on the matrix to STDOUT.                       */
/*    Use this functionality only for debugging purposes                */
/* -------------------------------------------------------------------- */

/*
    pardiso_printstats (&(ps->mtype), &n, a, ia, ja, &(ps->nrhs), b, &(ps->error));
    if (ps->error != 0) {
        printf("\nERROR right hand side: %d", ps->error);
        exit(1);
    }

*/

/* -------------------------------------------------------------------- */
/* ..  Back substitution and iterative refinement.                      */
/* -------------------------------------------------------------------- */
    int phase = 33;

    ps->iparm[7] = 0; //1      /* Max numbers of iterative refinement steps. */

    pardiso (ps->pt, &(ps->maxfct), &(ps->mnum), &(ps->mtype), &phase,
             &n, a, ia, ja, &idum, &(ps->nrhs),
             ps->iparm, &(ps->msglvl), b, x, &(ps->error),  ps->dparm);

    if (ps->error != 0) {
        printf("[Pardiso_solver] ERROR during solution: %d\n", ps->error);
        exit(3);
    }

    /*
    printf("\nSolve completed ... ");
    printf("\nThe solution of the system is: ");
    for (i = 0; i < n; i++) {
        printf("\n x [%d] = % f", i, x[i] );
    }
    printf ("\n\n");
    */

/* -------------------------------------------------------------------- */
/* ... Inverse factorization.                                           */
/* -------------------------------------------------------------------- */
#if 0
    if (solver == 0)
    {
    	printf("\nCompute Diagonal Elements of the inverse of A ... \n");
	phase = -22;
        iparm[35]  = 1; /*  no not overwrite internal factor L */

        pardiso (pt, &&(ps->maxfct), &mnum, &mtype, &phase, &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, b, x, &error,  dparm);

       /* print diagonal elements */
       for (k = 0; k < n; k++)
       {
            int j = ia[k]-1;
            printf ("Diagonal element of A^{-1} = %d %d %32.24e\n", k, ja[j]-1, a[j]);
       }

    }
#endif

return 0;
}

void pardiso_solver_finalize(pardiso_solver_t *ps, int n, int *ia, int *ja){

    double   ddum;              /* Double dummy */
    int      idum;              /* Integer dummy. */

/* -------------------------------------------------------------------- */
/* ..  Termination and release of memory.                               */
/* -------------------------------------------------------------------- */
    int phase = -1;                 /* Release internal memory. */

    pardiso (ps->pt, &(ps->maxfct), &(ps->mnum), &(ps->mtype), &phase,
             &n, &ddum, ia, ja, &idum, &(ps->nrhs),
             ps->iparm, &(ps->msglvl), &ddum, &ddum, &(ps->error),  ps->dparm);

    if(ps->perm!=NULL) free(ps->perm);
}
