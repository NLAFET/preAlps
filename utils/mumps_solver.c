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

#include "preAlps_intvector.h"
#include "preAlps_doublevector.h"

#ifdef DEBUG
#include <mpi.h>
#endif

/* Initialize mumps structure*/
int mumps_solver_init(mumps_solver_t *solver, MPI_Comm comm){

  solver->comm = comm;
  solver->id.comm_fortran = MPI_Comm_c2f(solver->comm);
  solver->id.par=1; //parallel
  //solver->id.sym=0; //must be set by the user
  solver->id.job=JOB_INIT;

  //solver->id.ISOL_loc = NULL;

  solver->irn = NULL;
  solver->jcn = NULL;
  solver->a   = NULL;

  solver->error_reporting_triangular_solve = 0;

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

  int i, j, nnz, myid, nprocs, ierr = 0, is_sym = 0;
  int *listvar_schur=NULL;
  double *Swork = NULL;

  MPI_Comm_rank(solver->comm, &myid);
  MPI_Comm_size(solver->comm, &nprocs);

  if(S_n>0) listvar_schur = (int*) malloc((S_n)*sizeof(int));

  for(i=0;i<S_n;i++) listvar_schur[i] = i+(n-S_n)+1;

  nnz = n>0?ia[n]:0;

  /* Allocate workspace for the schur complement */
  if(S_n>0) Swork = (double*) malloc((S_n * S_n)*sizeof(double));

  //set the matrix dimension on the host

  solver->id.n      = solver->m_glob;
  solver->id.nrhs   = solver->nrhs;
  if(solver->id.sym==1 || solver->id.sym==2) is_sym = 1;

  #ifdef DEBUG
      printf("[MUMPS] nprocs:%d, m_glob:%d, nnz_loc:%d, idxRowPos:%d\n", nprocs, solver->m_glob, nnz, solver->idxRowPos);
  #endif

  solver->irn = (int*) malloc(sizeof(int) * nnz);
  solver->jcn = (int*) malloc(sizeof(int) * nnz);

  if(!solver->irn || !solver->jcn) {printf("[MUMPS] Error: malloc failed for irc/jcn\n"); exit(1);}

  if(is_sym){
      solver->a = (double*) malloc(sizeof(double) * nnz);
      if(!solver->a ) {printf("[MUMPS] Error: malloc failed for a\n"); exit(1);}
  }else{
    solver->a = a;
  }


  // Convert to COO storage and 1-based indexed

  int compt = 0;
  for (i = 0; i < n; i++) {
    for (j = ia[i]; j < ia[i + 1]; j++) {

      if( (!is_sym) || (is_sym && (i + solver->idxRowPos>=ja[j]))){ //For the symmetric case, only provide the lower triangular part of the matrix
        solver->irn[compt] = i + 1 + solver->idxRowPos; //global indices of the matrix
        solver->jcn[compt] = ja[j] + 1;
        solver->a[compt] = a[j];
        compt++;
      }


    }
  }

  if(nprocs==1){
    //input matrix on the host
    solver->id.nnz = nnz;
    solver->id.irn = solver->irn;
    solver->id.jcn = solver->jcn;
    solver->id.a   = solver->a;
  }else{
    //input matrix is distributed
    solver->id.nz_loc   = nnz;
    solver->id.irn_loc = solver->irn;
    solver->id.jcn_loc = solver->jcn;
    solver->id.a_loc   = solver->a;
  }

  preAlps_intVector_printSynchronized(solver->irn, nnz, "irn", "irn in mumps", solver->comm);
  preAlps_intVector_printSynchronized(solver->jcn, nnz, "jcn", "jcn in mumps", solver->comm);
  preAlps_doubleVector_printSynchronized(a, nnz, "a", "a in mumps", solver->comm);

  if(S_n>0){
    if(nprocs==1){
      if (myid == 0) { //the host must have id 0 in the communicator
          solver->id.schur = Swork;
          //solver->id.rhs = rhs;
          solver->id.size_schur = S_n;
          //solver->id.schur_lld = S_n;
          solver->id.listvar_schur = listvar_schur;
      }
    }else{
      printf("Error: parallel schur complement computation using MUMPS is not yet supported in preAlps");
      exit(1);
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

  if(nprocs==1){
    solver->id.ICNTL(18) = 0;           //The input matrix is centralized on the host
    if(S_n>0) solver->id.ICNTL(19) = 1; //Computes the schur centralized by rows on the host. PreAlps does not support parallel schur yet
    solver->id.lrhs = n;                //the leading dimension of the rhs corresponds to the number of rows of the sequential problem
  }else{

    solver->id.ICNTL(18) = 3;           //The input matrix is distributed
    solver->id.lrhs = solver->m_glob;   //the leading dimension of the rhs corresponds to the global number of rows

    //Actually only the complete factorization can be done in parallel
    if(S_n>0) {printf("preAlps does not support parallel schur complement yet."); exit(1);}

  }

  //solver->id.ICNTL(21) = 1;//distributed solution
  solver->id.ICNTL(21) = 0;//centralized solution

  /* Call the MUMPS package (analyse, factorization). */
  solver->id.job=4;
  dmumps_c(&solver->id);

  if (solver->id.infog[0] == -9) {
    printf("[MUMPS] Factorization:  increasing internal required memory  by infog[1]:%d and restarting the factorization.\n ", solver->id.infog[1]);
    //Increase required memory

    solver->id.ICNTL(14) = solver->id.ICNTL(14) + solver->id.infog[1];

    //restart the numerical factorization
    solver->id.job=2;
    dmumps_c(&solver->id);
  }

  if (solver->id.infog[0] < 0) {
    printf("[MUMPS] Factorization error. infog[0]:%d,  infog[1]:%d\n ", solver->id.infog[0], solver->id.infog[1]);
    //preAlps_abort("");
  }

  //if (myid == 0) {
  //  printf("[mumps] INFO[23]:%d\n", solver->id.info[23]);
  //}
  //solver->id.LSOL_loc = solver->id.info[23];
  //solver->id.ISOL_loc = (int*) malloc((solver->id.LSOL_loc)*sizeof(int));;

  //Collect the schur complement
  if (myid == 0) {
    if(S_n>0){

      /* Convert to a CSR matrix*/

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
    if(Swork) free(Swork);
    if(listvar_schur) free(listvar_schur);
  }

  return ierr;
}


void mumps_solver_finalize(mumps_solver_t *solver, int n, int *ia, int *ja){

  if(solver->irn) free(solver->irn);
  if(solver->jcn) free(solver->jcn);
  //Memory allocated only for the symmetric case
  if(solver->id.sym==1 || solver->id.sym==2){
    if(solver->a) free(solver->a);
  }

  //if(solver->id.ISOL_loc != NULL) free(solver->id.ISOL_loc);

  solver->id.job=JOB_END;
  dmumps_c(&solver->id);

}

/*Solve Ax = b using mumps */
int mumps_solver_triangsolve(mumps_solver_t *solver, int n, double *a, int *ia, int *ja, int nrhs, double *x, double *b){


  if(x!= NULL ){
    printf("[MUMPS] *** Triangular solve error: Argument x is not used, the solution will overwrite b, set x = NULL\n ");
    exit(1);
  }


  //solver->id.SOL_loc = b;
  solver->id.rhs = b;
  solver->id.nrhs = nrhs;
  //solver->id.lrhs = n; //this is set during the factorization

  /*
  if(solver->id.nrhs != nrhs){
    printf("[MUMPS] *** Triangular solve error: mumps requires the same nrhs for the analysis and the solution. "
    "nrhs provided: analysis: %d, solve: %d\n", solver->id.nrhs, nrhs);
    exit(1);
  }
  */

  /* Call the MUMPS package (solve). */
  solver->id.job=3;
  dmumps_c(&solver->id);

  if (solver->id.infog[0] < 0) {

    solver->error_reporting_triangular_solve++;

    if(solver->error_reporting_triangular_solve<MAX_ERROR_REPORTING_TRIANGULAR_SOLVE)
      printf("[MUMPS] *** Triangular solve error: infog[0]:%d,  infog[1]:%d\n ", solver->id.infog[0], solver->id.infog[1]);
    else if(solver->error_reporting_triangular_solve==MAX_ERROR_REPORTING_TRIANGULAR_SOLVE)
      printf("[MUMPS] *** Triangular solve error: infog[0]:%d,  infog[1]:%d. (Max reporting of this error reached)\n ", solver->id.infog[0], solver->id.infog[1]);
    //preAlps_abort("");
  }

  return 0;
}
#endif
