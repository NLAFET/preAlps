/*
============================================================================
Name        : presc_eigsolve.c
Author      : Simplice Donfack
Version     : 0.1
Description : Solve eigenvalues problem using ARPACK
Date        : Mai 15, 2017
============================================================================
*/
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "preAlps_utils.h"
#include "presc.h"

#include "preAlps_solver.h"
#include "eigsolver.h"
#include "solverStats.h"
#include "matrixVectorOp.h"
#include "presc_eigsolve.h"

#define EIGVALUES_PRINT 0

#define USE_GENERALIZED_SYSTEM 1

#define EIGSOLVE_DISPLAY_STATS 1


/*
 * Solve the eigenvalues problem (I + AggP*S_{loc}^{-1})u = \lambda u using arpack.
 * Where AggP and S_{loc} are two sparse matrices.
 * AggP is formed by the offDiag elements of Agg, and S_{loc} = Block-Diag(S);
 * Check the paper for the structure of AggP and S_{loc}.
 *
 * presc:
 *    input/output: stores the computed eigenvalues at the end of this routine
 * comm:
 *    input: the communicator
 * mloc:
 *    input: the number of rows of the local matrice.
 * Sloc_sv
 *    input: the solver object to apply to compute  Sloc^{-1}v
 * Sloc
 *    input: the matrix Sloc
 * AggP
 *    input: the matrix AggP
*/
int Presc_eigSolve_SSloc(Presc_t *presc, MPI_Comm comm, int mloc, preAlps_solver_t *Sloc_sv, Mat_CSR_t *Sloc, Mat_CSR_t *AggP){

  int ierr;
  int root = 0, my_rank, nbprocs;
  double *Y, *X;
  double *dwork, *ywork;
  double dONE = 1.0, dZERO = 0.0;
  double t = 0.0, ttemp = 0.0;
  SolverStats_t tstats;
  int ido = 0, RCI_its = 0;
  int i, iterate = 0, m;
  int *mcounts, *mdispls;
  double deflation_tol = 1e-2; //the deflation tolerance, all eigenvalues lower than this will be selected for deflation
  Eigsolver_t *eigs;



  /* Create the eigensolver object*/
  Eigsolver_create(&eigs);

  /* Set the default parameters for the eigen solver*/
  Eigsolver_setDefaultParameters(eigs);

  /* Set the parameters for this specific problem */
  #if USE_GENERALIZED_SYSTEM
    eigs->bmat = 'G'; //Generalized eigenvalue problem
    eigs->issym = 1; //The problem is symmetric
  #else
    eigs->bmat = 'I'; //Standard eigenvalue problem
    eigs->issym = 0; //The problem is symmetric
  #endif

  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &nbprocs);

  SolverStats_init(&tstats);

  t = MPI_Wtime();

  if ( !(mcounts  = (int *) malloc(nbprocs * sizeof(int))) ) preAlps_abort("Malloc fails for mcounts[].");
  if ( !(mdispls  = (int *) malloc(nbprocs * sizeof(int))) ) preAlps_abort("Malloc fails for mdispls[].");

  //Gather the number of rows for each procs
  MPI_Allgather(&mloc, 1, MPI_INT, mcounts, 1, MPI_INT, comm);

  /* Compute the global problem size */
  m = 0;
  for(i=0;i<nbprocs;i++) m += mcounts[i];

  preAlps_int_printSynchronized(mloc, "mloc in PARPACK", comm);
  preAlps_int_printSynchronized(m, "m in PARPACK", comm);

  /* Set the number of eigenvalues to compute*/
  #ifdef NEV
   eigs->nev = NEV;
  #else
   //nev = (int) m*1e-2;
   eigs->nev = (int) m*2e-3;
   if(eigs->nev<=10) eigs->nev = 10;
  #endif

  Eigsolver_init(eigs, m, mloc);

  if(my_rank==0) printf("m:%d, mloc:%d, ncv:%d, nev:%d\n", m, mloc, eigs->ncv, eigs->nev);

  /* Allocate workspace*/

  if ( !(dwork  = (double *) malloc(mloc * sizeof(double))) ) preAlps_abort("Malloc fails for dwork[]."); //M
  if ( !(ywork  = (double *) malloc(m * sizeof(double))) ) preAlps_abort("Malloc fails for ywork[].");

  //compute displacements
  mdispls[0] = 0;
  for(i=1;i<nbprocs;i++) mdispls[i] = mdispls[i-1] + mcounts[i-1];

  preAlps_intVector_printSynchronized(mcounts, nbprocs, "mcounts", "mcounts", comm);
  preAlps_intVector_printSynchronized(mdispls, nbprocs, "mdispls", "mdispls", comm);


  MatCSRPrintSynchronizedCoords (AggP, comm, "AggP", "AggP");
  if(my_rank==root) printf("Agg size: %d\n", m);

  iterate = 1;
  while(iterate){

    RCI_its++;

    preAlps_int_printSynchronized(RCI_its, "Iteration", comm);

    //eigsolver_iterate(&eigs, X, Y);
    Eigsolver_iterate(eigs, comm, mloc, &X, &Y, &ido);

    /* Reverse communication */

    if(ido==-1||ido==1){
      ///if(RCI_its==1) {for(i=0;i<mloc;i++) X[i] = 1e-2; printf("dbgsimp1\n");}

      /* Compute the matrix vector product y = A*x
       * where A = S*S_{loc}^{-1}, S_{loc} = Block-Diag(S).
       * S*S_{loc}^{-1} = (I + AggP*S_{loc}^{-1})
      */

      if(eigs->bmat == 'G'){
        //Generalized eigenvalue problem, Compute inv(Sloc)*S*X
        ierr = matrixVectorOp_SlocInvxS(comm, mloc, m, mcounts, mdispls, Sloc_sv, Sloc, AggP, X, Y, dwork, ywork, &tstats);
      }
      else{
        //standard eigenvalue problem, compute S*Inv(Sloc)*X
        ierr = matrixVectorOp_SxSlocInv(comm, mloc, m, mcounts, mdispls, Sloc_sv, Sloc, AggP, X, Y, dwork, ywork, &tstats);
      }

    }else if(ido==2){

      /* Compute  Y = Sloc * X */
      ttemp = MPI_Wtime();
      MatCSRMatrixVector(Sloc, dONE, X, dZERO, Y);
      tstats.tAv += MPI_Wtime() - ttemp;

      preAlps_doubleVector_printSynchronized(Y, mloc, "Y", "Y after Sloc*v", comm);
    }else if(ido==99){
       iterate = 0;
    }else{
       preAlps_abort("[PARPACK] (Unhandled case) ido is not 99, current value:%d", ido);
    } //ido

    //if(RCI_its>=5) iterate = 0; //DEBUG: force break for debugging purpose
  }

  free(dwork);
  free(ywork);


  /* Select the eigenvalues to deflate */
  if(my_rank == root){
    for (i = 0; i < eigs->nevComputed; i++) {
      if (eigs->eigvalues[i] <= deflation_tol) {
        presc->eigvalues_deflation++;
      }
    }
  }

  #ifdef EIGVALUES_PRINT
    if(my_rank == root){
      printf("[%d] eigenvalues:\n", my_rank);
      for (i = 0; i < eigs->nevComputed; i++) {
        printf("\t%.16e", eigs->eigvalues[i]);
        if (eigs->eigvalues[i] <= deflation_tol) {
          printf(" (selected)\n");
        }else{
          printf(" (ignored)\n");
        }
      }
    }
  #endif

  if(my_rank==root) printf("Eigenvalues selected for deflation: %d/%d\n", presc->eigvalues_deflation, eigs->nev);

  /* Terminate the solver and free the allocated workspace*/
  ierr = Eigsolver_finalize(&eigs);

  tstats.tTotal = MPI_Wtime() - t;

  #if EIGSOLVE_DISPLAY_STATS
    preAlps_dstats_display(comm, tstats.tParpack, "Time Parpack");
    preAlps_dstats_display(comm, tstats.tAv, "Time Agg*v");
    preAlps_dstats_display(comm, tstats.tSolve, "Time Sloc^{-1}*v");
    preAlps_dstats_display(comm, tstats.tComm, "Time Comm");
    preAlps_dstats_display(comm, tstats.tTotal, "Time EigSolve");
  #endif

  /*Free memory*/
  free(mcounts);
  free(mdispls);

  return ierr;
}

/*
 * Version with ALOC
 */


 /*
  * Solve the eigenvalues problem Sloc*u = \lambda*Aloc*u using arpack.
  * Where  Sloc = Aggloc - Agi*inv(Aii)*Aig.
  *
  * presc:
  *     input/output: stores the computed eigenvalues at the end of this routine
  * comm:
  *    input: the communicator
  * mloc:
  *    input: the number of rows of the local matrice.
  * Aggloc, Agi, Aii, Aig
  *    input: the matrices required for Sloc
  * Aloc
  *    input: the matrix Aloc
  * Aii_sv
  *    input: the solver object to apply to compute  Aii^{-1}v
  * Aloc_sv
  *    input: the solver object to apply to compute  Aloc^{-1}v
 */

 int Presc_eigSolve_SAloc(Presc_t *presc, MPI_Comm comm, int mloc,
                          Mat_CSR_t *Aggloc, Mat_CSR_t *Agi, Mat_CSR_t *Aii, Mat_CSR_t *Aig,
                          Mat_CSR_t *Aloc, preAlps_solver_t *Aii_sv, preAlps_solver_t *Aloc_sv){

  int ierr = 0;
  int root = 0, my_rank, nbprocs;
  double *Y, *X;
  double *dwork1, *dwork2, *ywork;
  double dONE = 1.0, dZERO = 0.0;
  double t = 0.0, ttemp = 0.0;
  SolverStats_t tstats;
  int i, iterate = 0, m, ido = 0, RCI_its = 0;
  int *mcounts, *mdispls;
  double deflation_tol = 1e-2; //the deflation tolerance, all eigenvalues lower than this will be selected for deflation
  Eigsolver_t *eigs;


  /* Create the eigensolver object*/
  Eigsolver_create(&eigs);

  /* Set the default parameters for the eigen solver*/
  Eigsolver_setDefaultParameters(eigs);

  /* Set the parameters for this specific problem */
  eigs->bmat = 'G'; //Generalized eigenvalue problem
  eigs->issym = 1; //The problem is symmetric

  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &nbprocs);

  SolverStats_init(&tstats);
  t = MPI_Wtime();

  if ( !(mcounts  = (int *) malloc(nbprocs * sizeof(int))) ) preAlps_abort("Malloc fails for mcounts[].");
  if ( !(mdispls  = (int *) malloc(nbprocs * sizeof(int))) ) preAlps_abort("Malloc fails for mdispls[].");

  preAlps_int_printSynchronized(mloc, "mloc in PARPACK", comm);

  //Gather the number of rows for each procs
  MPI_Allgather(&mloc, 1, MPI_INT, mcounts, 1, MPI_INT, comm);

  /* Compute the global problem size */
  m = 0;
  for(i=0;i<nbprocs;i++) m += mcounts[i];

  preAlps_int_printSynchronized(mloc, "mloc in PARPACK", comm);
  preAlps_int_printSynchronized(m, "m in PARPACK", comm);

  /* Set the number of eigenvalues to compute*/
  #ifdef NEV
    eigs->nev = NEV;
  #else
    //nev = (int) m*1e-2;
    eigs->nev = (int) m*2e-3;
    if(eigs->nev<=10) eigs->nev = 10;
   #endif

  Eigsolver_init(eigs, m, mloc);

  #ifdef DEBUG
    printf("mloc:%d, m:%d, Aii: (%d x %d), Aig: (%d x %d)\n", mloc, m, Aii->info.m, Aii->info.n, Aig->info.m, Aig->info.n);
    if(my_rank==0) printf("m:%d, mloc:%d, ncv:%d, nev:%d\n", m, mloc, eigs->ncv, eigs->nev);
    MPI_Barrier(comm); //debug only
  #endif

  /* Allocate workspace*/

  if ( !(dwork1  = (double *) malloc(Aii->info.m * sizeof(double))) ) preAlps_abort("Malloc fails for dwork1[].");
  if ( !(dwork2  = (double *) malloc(Aii->info.m * sizeof(double))) ) preAlps_abort("Malloc fails for dwork2[].");

  if ( !(ywork  = (double *) malloc(m * sizeof(double))) ) preAlps_abort("Malloc fails for ywork[].");

  //compute displacements
  mdispls[0] = 0;
  for(i=1;i<nbprocs;i++) mdispls[i] = mdispls[i-1] + mcounts[i-1];

  preAlps_intVector_printSynchronized(mcounts, nbprocs, "mcounts", "mcounts", comm);
  preAlps_intVector_printSynchronized(mdispls, nbprocs, "mdispls", "mdispls", comm);

  MatCSRPrintSynchronizedCoords (Aggloc, comm, "Aggloc", "Aggloc");
  if(my_rank==root) printf("Agg size: %d\n", m);

  iterate = 1;
  while(iterate){

    RCI_its++;

    preAlps_int_printSynchronized(RCI_its, "Iteration", comm);

    //eigsolver_iterate(&eigs, X, Y);
    Eigsolver_iterate(eigs, comm, mloc, &X, &Y, &ido);

    /* Reverse communication */

    if(ido==-1||ido==1){
      ///if(RCI_its==1) {for(i=0;i<mloc;i++) X[i] = 1e-2; printf("dbgsimp1\n");}
      /*
       * Compute the matrix vector product Y = OP*X = Inv(Aloc)*S*X
       */

      ierr = matrixVectorOp_AlocInvxS(comm, mloc, m, mcounts, mdispls,
                                          Aggloc, Agi, Aii, Aig, Aloc,
                                          Aii_sv, Aloc_sv, X, Y,
                                          dwork1, dwork2, ywork, &tstats);
    }else if(ido==2){

      /* Compute  Y = Aloc * X */
      ttemp = MPI_Wtime();
      MatCSRMatrixVector(Aloc, dONE, X, dZERO, Y);
      tstats.tAv += MPI_Wtime() - ttemp;

      preAlps_doubleVector_printSynchronized(Y, mloc, "Y", "Y after Aloc*v", comm);

    }else if(ido==99){
      iterate = 0;

    }else{
      preAlps_abort("[PARPACK] (Unhandled case) ido is not 99, current value:%d", ido);
    } //ido
     //if(RCI_its>=5) iterate = 0; //DEBUG: force break for debugging purpose
  }


  preAlps_int_printSynchronized(1, "eigs free", comm);
  free(dwork1);
  free(dwork2);

  free(ywork);

  /* Select the eigenvalues to deflate */
  if(my_rank == root){
    for (i = 0; i < eigs->nevComputed; i++) {
      if (eigs->eigvalues[i] <= deflation_tol) {
        presc->eigvalues_deflation++;
      }
    }
  }

  #ifdef EIGVALUES_PRINT
    if(my_rank == root){
      printf("[%d] eigenvalues:\n", my_rank);
      for (i = 0; i < eigs->nevComputed; i++) {
        printf("\t%.16e", eigs->eigvalues[i]);
        if (eigs->eigvalues[i] <= deflation_tol) {
          printf(" (selected)\n");
        }else{
          printf(" (ignored)\n");
        }
      }
    }
  #endif

  if(my_rank==root) printf("Eigenvalues selected for deflation: %d/%d\n", presc->eigvalues_deflation, eigs->nev);

  /* Terminate the solver and free the allocated workspace*/
  ierr = Eigsolver_finalize(&eigs);

  tstats.tTotal = MPI_Wtime() - t;

  #if EIGSOLVE_DISPLAY_STATS
    preAlps_dstats_display(comm, tstats.tParpack, "Time Parpack");
    //preAlps_dstats_display(comm, tstats.tAv, "Time Agg*v");
    //preAlps_dstats_display(comm, tstats.tSolve, "Time Aii^{-1}*v");
    preAlps_dstats_display(comm, tstats.tSv, "Time S*v");
    preAlps_dstats_display(comm, tstats.tInvAv, "Time Inv(Agg)*v");
    preAlps_dstats_display(comm, tstats.tComm, "Time Comm");
    preAlps_dstats_display(comm, tstats.tTotal, "Time EigSolve");
  #endif

  /*Free memory*/
  free(mcounts);
  free(mdispls);

  return ierr;
 }
