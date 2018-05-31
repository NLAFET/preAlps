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

#define EIGVALUES_PRINT 1

#define EIGSOLVE_DISPLAY_STATS 0

#define USE_GENERALIZED_SYSTEM 1




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
  * Agi, Aii, Aig
  *    input: the matrices required for computing the second part of S
  * Aggloc
  *    input: the matrix Agg distributed on all procs
  * Aii_sv
  *    input: the solver object to apply to compute  Aii^{-1}v
  * Agg_sv
  *    input: the solver object to apply to compute  Agg^{-1}v
 */

 int preAlps_LorascEigSolve(preAlps_Lorasc_t *lorascA, MPI_Comm comm, int mloc, CPLM_Mat_CSR_t *Agi, CPLM_Mat_CSR_t *Aii, CPLM_Mat_CSR_t *Aig,
                            CPLM_Mat_CSR_t *Aggloc, preAlps_solver_t *Aii_sv, preAlps_solver_t *Agg_sv){

  int ierr = 0;
  int root = 0, my_rank, nbprocs;
  double *Y, *X;
  double *dwork1, *dwork2, *ywork;
  double dONE = 1.0, dZERO = 0.0;
  double t = 0.0, ttemp = 0.0, ttemp1 = 0.0;
  SolverStats_t tstats;
  int i, iterate = 0, m, ido = 0, RCI_its = 0, max_m;
  int *mcounts = NULL, *mdispls = NULL;
  double deflation_tol = lorascA->deflation_tol; //1e-2//the deflation tolerance, all eigenvalues lower than this will be selected for deflation
  Eigsolver_t *eigs;

  int eigvalues_deflation = 0;
  double *eigvalues = NULL, *eigvectors = NULL, *sigma = NULL;
  int masterGroup_myrank, masterGroup_nbprocs, localGroup_myrank, localGroup_nbprocs, local_root =0;

  //Retrieve parameters
  MPI_Comm comm_masterGroup   = lorascA->comm_masterGroup;
  MPI_Comm comm_localGroup    = lorascA->comm_localGroup;

  int *Aii_mcounts   = lorascA->Aii_mcounts;
  int *Aii_moffsets  = lorascA->Aii_moffsets;
  int *Aig_mcounts   = lorascA->Aig_mcounts;
  int *Aig_moffsets  = lorascA->Aig_moffsets;
  int *Agi_mcounts   = lorascA->Agi_mcounts;
  int *Agi_moffsets  = lorascA->Agi_moffsets;


  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &nbprocs);

  if(comm_masterGroup!=MPI_COMM_NULL){
    MPI_Comm_rank(comm_masterGroup, &masterGroup_myrank);
    MPI_Comm_size(comm_masterGroup, &masterGroup_nbprocs);
  }
  MPI_Comm_rank(comm_localGroup, &localGroup_myrank);
  MPI_Comm_size(comm_localGroup, &localGroup_nbprocs);

  preAlps_int_printSynchronized(0, "dbg", comm);

  if(comm_masterGroup!=MPI_COMM_NULL){

    SolverStats_init(&tstats);
    t = MPI_Wtime();

    /* Create the eigensolver object*/
    Eigsolver_create(&eigs);

    /* Set the default parameters for the eigen solver*/
    Eigsolver_setDefaultParameters(eigs);

    /* Set the parameters for this specific problem */
    eigs->bmat = 'G'; //Generalized eigenvalue problem
    eigs->issym = 1; //The problem is symmetric
    eigs->residual_tolerance  = 1e-3;

    if ( !(mcounts  = (int *) malloc(masterGroup_nbprocs * sizeof(int))) ) preAlps_abort("Malloc fails for mcounts[].");
    if ( !(mdispls  = (int *) malloc(masterGroup_nbprocs * sizeof(int))) ) preAlps_abort("Malloc fails for mdispls[].");

    //Gather the number of rows for each procs
    MPI_Allgather(&mloc, 1, MPI_INT, mcounts, 1, MPI_INT, comm_masterGroup);

    /* Compute the global problem size */
    m = 0;
    for(i=0;i<masterGroup_nbprocs;i++) m += mcounts[i];

    /* Set the number of eigenvalues to compute*/
    #ifdef NEV
      eigs->nev = NEV;
    #else
      //eigs->nev = (int) m*1e-2;
      //eigs->nev = (int) m*2e-3;
      eigs->nev = (int) m*4e-3;
      if(eigs->nev<40) eigs->nev = eigs->nev*2; //small impact on the preconditionner time, huge impact on the solve
      if(eigs->nev<10) eigs->nev = 10;
      //if(eigs->nev<50) eigs->nev = 50; //small impact on the preconditionner time, huge impact on the solve
    #endif

    Eigsolver_init(eigs, m, mloc);

    #ifdef DEBUG
      printf("mloc:%d, m:%d, Aii: (%d x %d), Aig: (%d x %d)\n", mloc, m, Aii->info.m, Aii->info.n, Aig->info.m, Aig->info.n);
      if(masterGroup_myrank==0) printf("m:%d, mloc:%d, ncv:%d, nev:%d\n", m, mloc, eigs->ncv, eigs->nev);
    #endif



    //compute displacements
    mdispls[0] = 0;
    for(i=1;i<masterGroup_nbprocs;i++) mdispls[i] = mdispls[i-1] + mcounts[i-1];

    //if(my_rank==root) printf("Agg size: %d\n", m);

  }

  preAlps_int_printSynchronized(1, "dbg", comm);

  //broadcast the global matrix dimension to the local group
  if(localGroup_nbprocs>1){
    MPI_Bcast(&m, 1, MPI_INT, local_root, comm_localGroup);
  }

  preAlps_int_printSynchronized(m, "m in lorasc_eigsolve", comm);

  /* Allocate workspace*/

  max_m = max(m, Aii_moffsets[localGroup_nbprocs]); //max(m, Aii->info.m); //Allow multiple usage of the same buffer
  if ( !(dwork1  = (double *) malloc(max_m * sizeof(double))) ) preAlps_abort("Malloc fails for dwork1[].");
  if ( !(dwork2  = (double *) malloc(max_m * sizeof(double))) ) preAlps_abort("Malloc fails for dwork2[].");

  if ( !(ywork  = (double *) malloc(m * sizeof(double))) ) preAlps_abort("Malloc fails for ywork[].");


  iterate = 1;

  while(iterate){

    if(comm_masterGroup!=MPI_COMM_NULL){

      RCI_its++;

      #ifdef DEBUG
        preAlps_int_printSynchronized(RCI_its, "****************** Iteration", comm_masterGroup);
      #endif

      Eigsolver_iterate(eigs, comm_masterGroup, mloc, &X, &Y, &ido);
    }

    //broadcast ido to the local group
    if(localGroup_nbprocs>1){
      MPI_Bcast(&ido, 1, MPI_INT, local_root, comm_localGroup);
    }

    preAlps_int_printSynchronized(ido, "ido in lorasc_eigsolve", comm);

    #ifdef DEBUG
      if(comm_masterGroup!=MPI_COMM_NULL){
        MPI_Gatherv(X, mloc, MPI_DOUBLE, ywork, mcounts, mdispls, MPI_DOUBLE, root, comm_masterGroup);
        if(masterGroup_myrank==root) preAlps_doubleVector_printSynchronized(ywork, m, "X", " X before matvec", MPI_COMM_SELF);
      }
    #endif

    /* Reverse communication */

    if(ido==-1||ido==1){
      ttemp = MPI_Wtime();
      /*
       * Compute the matrix vector product Y = OP*X = Inv(Agg)*S*X
       */
      /*
      ierr = matrixVectorOp_AggInvxS(comm, mloc, m, mcounts, mdispls,
                                          Agi, Aii, Aig, Aggloc,
                                          Aii_sv, Agg_sv, X, Y,
                                          dwork1, dwork2, ywork, &tstats);
      */
      ierr = matrixVectorOp_AggInvxS_mlevel(mloc, m, mcounts, mdispls,
                                          Agi, Aii, Aig, Aggloc,
                                          Aii_sv, Agg_sv, X, Y,
                                          dwork1, dwork2, ywork,
                                          comm_masterGroup,
                                          comm_localGroup,
                                          Aii_mcounts, Aii_moffsets,
                                          Aig_mcounts, Aig_moffsets,
                                          Agi_mcounts, Agi_moffsets,
                                          &tstats);
      tstats.tOPv+= MPI_Wtime() - ttemp;

    }else if(ido==2){
      if(comm_masterGroup!=MPI_COMM_NULL){
        ttemp = MPI_Wtime();

        //Gather the vector from each procs
        ttemp1 = MPI_Wtime();
        MPI_Allgatherv(X, mloc, MPI_DOUBLE, ywork, mcounts, mdispls, MPI_DOUBLE, comm_masterGroup);
        tstats.tComm+= MPI_Wtime() - ttemp1;

        /* Compute  Y = Agg * X */
        ttemp1 = MPI_Wtime();
        CPLM_MatCSRMatrixVector(Aggloc, dONE, ywork, dZERO, Y);
        tstats.tAv += MPI_Wtime() - ttemp1;

        tstats.tBv+= MPI_Wtime() - ttemp;
      }
    }else if(ido==99){

      iterate = 0;

    }else{
      preAlps_abort("[PARPACK] (Unhandled case) ido is not 99, current value:%d", ido);
    } //ido
     //if(RCI_its>=5) iterate = 0; //DEBUG: force break for debugging purpose

     #ifdef DEBUG
       if(comm_masterGroup!=MPI_COMM_NULL){
         MPI_Gatherv(X, mloc, MPI_DOUBLE, ywork, mcounts, mdispls, MPI_DOUBLE, root, comm_masterGroup);
         if(masterGroup_myrank==root) preAlps_doubleVector_printSynchronized(ywork, m, "X", " X after matvec", MPI_COMM_SELF);
       }
     #endif
    #ifdef DEBUG
      if(comm_masterGroup!=MPI_COMM_NULL){
        MPI_Gatherv(Y, mloc, MPI_DOUBLE, ywork, mcounts, mdispls, MPI_DOUBLE, root, comm_masterGroup);
        if(masterGroup_myrank==root) preAlps_doubleVector_printSynchronized(ywork, m, "Y", " Y gathered after matvec", MPI_COMM_SELF);
      }
    #endif

  }

  free(dwork1);
  free(dwork2);

  free(ywork);

  if(comm_masterGroup!=MPI_COMM_NULL){

    /* Select the eigenvalues to deflate */
    //if(my_rank == root){
      if ( !(eigvalues  = (double *) malloc(eigs->nevComputed * sizeof(double))) ) preAlps_abort("Malloc fails for lorasc->eigenvalues[].");
      if ( !(sigma      = (double *) malloc(eigs->nevComputed * sizeof(double))) ) preAlps_abort("Malloc fails for lorasc->sigma[].");
      for (i = 0; i < eigs->nevComputed; i++) {
        if (eigs->eigvalues[i] <= deflation_tol) {
          eigvalues[i] = eigs->eigvalues[i];
          sigma[i] = (lorascA->deflation_tol - eigs->eigvalues[i])/eigs->eigvalues[i];
          eigvalues_deflation++;
        }
      }
    //}

    #if EIGVALUES_PRINT
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

    if(my_rank==root) printf("Eigenvalues selected for deflation: %d/%d\n", eigvalues_deflation, eigs->nev);

    // Gather the local computed eigenvectors on the root process
    Eigsolver_eigenvectorsGather(eigs, comm_masterGroup, mcounts, mdispls, &eigvectors);

    // Terminate the solver and free the allocated workspace
    ierr = Eigsolver_finalize(&eigs);

    tstats.tTotal = MPI_Wtime() - t;

    tstats.tParpack = eigs->tEigValues;
    tstats.teigvectors = eigs->tEigVectors;

    #if EIGSOLVE_DISPLAY_STATS
      preAlps_dstats_display(comm_masterGroup, tstats.tParpack, "Time Parpack");
      preAlps_dstats_display(comm_masterGroup, tstats.tOPv, "Time OP*v");
      preAlps_dstats_display(comm_masterGroup, tstats.tBv, "Time B*v");
      preAlps_dstats_display(comm_masterGroup, tstats.teigvectors, "Time eigensolvers");
      preAlps_dstats_display(comm_masterGroup, tstats.tSv, "Time S*v");
      preAlps_dstats_display(comm_masterGroup, tstats.tInvAv, "Time Inv(Agg)*v");
      preAlps_dstats_display(comm_masterGroup, tstats.tComm, "Time Comm");
      preAlps_dstats_display(comm_masterGroup, tstats.tTotal, "Time EigSolve");
    #endif

    // Save the eigenproblem data for the application of the preconditioner
    lorascA->eigvalues_deflation = eigvalues_deflation;
    lorascA->eigvalues           = eigvalues;
    lorascA->eigvectors          = eigvectors;
    lorascA->sigma               = sigma;


    /*Free memory*/
    free(mcounts);
    free(mdispls);
  }

  return ierr;

 }
