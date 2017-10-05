/*
============================================================================
Name        : mumps_solver.c
Author      : Simplice Donfack
Version     : 0.1
Description : Wrapper for mumps functions.
Date        : July 8, 2017
============================================================================
*/
#ifdef USE_SOLVER_MUMPS

#include <stdio.h>
#include <stdlib.h>

#include "mumps_solver.h"

#ifdef DEBUG
#include<mpi.h>
#endif

/* Initialize mumps structure*/
int mumps_solver_init(mumps_solver_t *solver, MPI_Comm comm){

  solver->comm = comm;
  solver->id.comm_fortran = MPI_Comm_c2f(solver->comm);
  solver->id.par=1; //parallel
  solver->id.sym=0;
  solver->id.job=JOB_INIT;

  solver->id.ISOL_loc = NULL;

  solver->irn = NULL;
  solver->jcn = NULL;

  dmumps_c(&solver->id);

  if (solver->id.infog[0] < 0) {
    printf("MUMPS initialization error. infolog[0]:%d,  infolog[1]:%d\n ", solver->id.infog[0], solver->id.infog[1]);
    exit(1);
  }

  return 0;
}


/* Perform the factorization of the matrix,
*/
int mumps_solver_factorize(mumps_solver_t *solver, int n, double *a, int *ia, int *ja){

  int ierr;

  /* call the partial interface without the schur complement */
  int S_n =0;
  ierr = mumps_solver_partial_factorize(solver, n, a, ia, ja, S_n,
                                              NULL, NULL, NULL);

  return ierr;
}


/* Perform the partial factorization of the matrix,
 * and compute S = A_{22} - A_{21}A_{11}^{-1}A_{12}
 * The factored part of the matrix can be use to solve the system A_{11}x= b1;
 * (S, iS,jS) is the returned schur complement
 * if S_n=0, the schur complement is not computed
*/
int mumps_solver_partial_factorize(mumps_solver_t *solver, int n, double *a, int *ia, int *ja, int S_n,
                                            double **S, int **iS, int **jS){

  int i, j, nnz, myid, ierr = 0;
  int *listvar_schur=NULL;
  double *Swork = NULL;

  MPI_Comm_rank(solver->comm, &myid);


  if(S_n>0) listvar_schur = (int*) malloc((S_n)*sizeof(int));

  for(i=0;i<S_n;i++) listvar_schur[i] = i+(n-S_n)+1;

  nnz = n>0?ia[n]:0;

  /* Allocate workspace for the schur complement */
  if(S_n>0) Swork = (double*) malloc((S_n * S_n)*sizeof(double));

  /* Define the problem on the host */
  if (myid == 0) {

    /*convert to 1-based indexing*/
    /*
    for (i = 0; i < n+1; i++) {
        ia[i] += 1;
    }
    for (i = 0; i < nnz; i++) {
        ja[i] += 1;
    }
    */
    solver->irn = (int*) malloc(sizeof(int) * nnz);
    solver->jcn = (int*) malloc(sizeof(int) * nnz);

    /* Convert to COO storage and 1-based indexed*/
    for (i = 0; i < n; i++) {
      for (j = ia[i]; j < ia[i + 1]; j++) {
        solver->irn[j] = i + 1;
        //solver->jcn[j] = ja[j] + 1;
      }
    }

    for (i = 0; i < nnz; i++) {
     solver->jcn[i] = ja[i] + 1;
    }

    solver->id.n   = n;
    solver->id.nnz = nnz;
    solver->id.irn = solver->irn;
    solver->id.jcn = solver->jcn;
    solver->id.a   = a;

    if(S_n>0) {

      solver->id.schur = Swork;
      //solver->id.rhs = rhs;

      solver->id.size_schur = S_n;

      //solver->id.schur_lld = S_n;
      solver->id.listvar_schur = listvar_schur;
    }

  }


  /* ICNTL is a macro s.t. indices match documentation */

  //Debug mumps param
  //solver->id.ICNTL(1)=-1; //output stream
  //solver->id.ICNTL(2)=6; //output error
  //solver->id.ICNTL(3)=-1;
  //solver->id.ICNTL(4)=6; //output stream level

  solver->id.ICNTL(1)=-1; //output stream
  solver->id.ICNTL(2)=-1; //output error
  solver->id.ICNTL(3)=-1;
  solver->id.ICNTL(4)=6; //output stream level


  //solver->id.ICNTL(7)=3; //Try using pORD
  solver->id.ICNTL(18) = 0;//The input matrix is centralized on the host

  if(S_n>0) solver->id.ICNTL(19) = 1; //Computes the schur centralized by rows on the host

  /* Call the MUMPS package (analyse, factorization). */
  solver->id.job=4;
  dmumps_c(&solver->id);

  if (solver->id.infog[0] < 0) {
    printf("*** MUMPS factorization error. infog[0]:%d,  infog[1]:%d\n ", solver->id.infog[0], solver->id.infog[1]);
    //preAlps_abort("");
  }

  if (myid == 0) {
    printf("[mumps] INFO[23]:%d\n", solver->id.info[23]);
  }

  //solver->id.LSOL_loc = solver->id.info[23];
  //solver->id.ISOL_loc = (int*) malloc((solver->id.LSOL_loc)*sizeof(int));;

  if (myid == 0) {

    /*convert back to 0-based indexing*/
    /*

    for (i = 0; i < n+1; i++) {
        ia[i] -= 1;
    }

    for (i = 0; i < nnz; i++) {
        ja[i] -= 1;
    }

    */

    if(S_n>0){
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
        printf("Malloc fails for the CSR matrix S in preAlps_solver_partial_factorize\n");
        exit(1);
      }


      int count=0;
      (*iS)[0]=0;
      for(int i=0;i<S_n;i++) {
        for(int j=0;j<S_n;j++){
          if(Swork[i*S_n+j] != 0.0 ) {
           (*jS)[count] = j;
           (*S)[count]  = Swork[i*S_n+j];
           count++;
         }
        }
        (*iS)[i+1]=count;
      }
    }
  }

  if(S_n>0){
    free(Swork);
    free(listvar_schur);
  }

  return ierr;
}


void mumps_solver_finalize(mumps_solver_t *solver, int n, int *ia, int *ja){

  free(solver->irn);
  free(solver->jcn);

  if(solver->id.ISOL_loc != NULL) free(solver->id.ISOL_loc);

  solver->id.job=JOB_END;
  dmumps_c(&solver->id);

}

/*Solve Ax = b using mumps */
int mumps_solver_triangsolve(mumps_solver_t *ps, int n, double *a, int *ia, int *ja, double *x, double *b){


  if(x!= NULL ){
    printf("MUMPS triangular solve. Argument x is not used, the solution will be overwrite b, set x = NULL\n ");
    exit(1);
  }
  //solver->id.ICNTL(21) = 1;//distributed solution
  solver->id.ICNTL(21) = 0;//centralized solution

  //solver->id.SOL_loc = b;
  solver->id.rhs = b;

  /* Call the MUMPS package (solve). */
  solver->id.job=3;
  dmumps_c(&solver->id);

  if (solver->id.infog[0] < 0) {
    printf("*** MUMPS factorization error. infog[0]:%d,  infog[1]:%d\n ", solver->id.infog[0], solver->id.infog[1]);
    //preAlps_abort("");
  }

}
#endif
