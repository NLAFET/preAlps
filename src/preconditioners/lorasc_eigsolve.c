/*
============================================================================
Name        : lorasc_eigsolve.h
Author      : Simplice Donfack
Version     : 0.1
Description : Solve eigenvalues problem using ARPACK
Date        : oct 1, 2017
============================================================================
*/
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "preAlps_utils.h"
#include "lorasc.h"

#include "preAlps_solver.h"
#include "eigsolver.h"
#include "solverStats.h"
#include "matrixVectorOp.h"
#include "lorasc_eigsolve.h"

#define EIGVALUES_PRINT 0

#define USE_GENERALIZED_SYSTEM 1

#define EIGSOLVE_DISPLAY_STATS 1


 /*
  * Solve the eigenvalues problem S*u = \lambda*Agg*u using arpack.
  * Where  S = Agg - Agi*inv(Aii)*Aig.
  *
  * lorascA:
  *     input/output: stores the computed eigenvalues at the end of this routine
  * comm:
  *    input: the communicator
  * mloc:
  *    input: the number of rows of the local matrice.
  * Aggloc
  *    input: the matrix Agg distributed on all procs
  * Agi, Aii, Aig
  *    input: the matrices required for computing the second part of S
  * Agg
  *    input: the matrix Agg on processor 0
  * Aii_sv
  *    input: the solver object to apply to compute  Aii^{-1}v
  * Agg_sv
  *    input: the solver object to apply to compute  Agg^{-1}v
 */

 int preAlps_LorascEigSolve(preAlps_Lorasc_t *lorascA, MPI_Comm comm, int mloc, CPLM_Mat_CSR_t *Aggloc, CPLM_Mat_CSR_t *Agi, CPLM_Mat_CSR_t *Aii, CPLM_Mat_CSR_t *Aig,
                          CPLM_Mat_CSR_t *Agg, preAlps_solver_t *Aii_sv, preAlps_solver_t *Agg_sv){

  int ierr = 0;
  int root = 0, my_rank, nbprocs;
  double *Y, *X;
  double *dwork1, *dwork2, *ywork;
  double dONE = 1.0, dZERO = 0.0;
  double t = 0.0, ttemp = 0.0;
  SolverStats_t tstats;
  int i, iterate = 0, m, ido = 0, RCI_its = 0, max_m;
  int *mcounts, *mdispls;
  double deflation_tol = 1e-2; //the deflation tolerance, all eigenvalues lower than this will be selected for deflation
  Eigsolver_t *eigs;




  /* Create the eigensolver object*/
  Eigsolver_create(&eigs);
preAlps_int_printSynchronized(1, "dbg", comm);

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
  max_m = max(m, Aii->info.m); //Allow multiple usage of the same buffer
  if ( !(dwork1  = (double *) malloc(max_m * sizeof(double))) ) preAlps_abort("Malloc fails for dwork1[].");
  if ( !(dwork2  = (double *) malloc(max_m * sizeof(double))) ) preAlps_abort("Malloc fails for dwork2[].");

  if ( !(ywork  = (double *) malloc(m * sizeof(double))) ) preAlps_abort("Malloc fails for ywork[].");

  //compute displacements
  mdispls[0] = 0;
  for(i=1;i<nbprocs;i++) mdispls[i] = mdispls[i-1] + mcounts[i-1];

  preAlps_intVector_printSynchronized(mcounts, nbprocs, "mcounts", "mcounts", comm);
  preAlps_intVector_printSynchronized(mdispls, nbprocs, "mdispls", "mdispls", comm);

  CPLM_MatCSRPrintSynchronizedCoords (Agg, comm, "Agg", "Agg");
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
       * Compute the matrix vector product Y = OP*X = Inv(Agg)*S*X
       */

      ierr = matrixVectorOp_AggInvxS(comm, mloc, m, mcounts, mdispls,
                                          Aggloc, Agi, Aii, Aig, Agg,
                                          Aii_sv, Agg_sv, X, Y,
                                          dwork1, dwork2, ywork, &tstats);
    }else if(ido==2){

      //Gather the vector from each procs
      ttemp = MPI_Wtime();
      MPI_Allgatherv(X, mloc, MPI_DOUBLE, ywork, mcounts, mdispls, MPI_DOUBLE, comm);
      tstats.tComm+= MPI_Wtime() - ttemp;

      /* Compute  Y = Agg * X */
      ttemp = MPI_Wtime();
      CPLM_MatCSRMatrixVector(Aggloc, dONE, ywork, dZERO, Y);
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
        lorascA->eigvalues_deflation++;
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

  if(my_rank==root) printf("Eigenvalues selected for deflation: %d/%d\n", lorascA->eigvalues_deflation, eigs->nev);

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
