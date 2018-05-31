/*
============================================================================
Name        : matrixVectorOp.c
Author      : Simplice Donfack
Version     : 0.1
Description : Matrix vector product routines which can be used as operator for
an iterative methods or eigensolver
Date        : Sept 15, 2017
============================================================================
*/

#include <string.h>
#include "preAlps_utils.h"
#include "solverStats.h"
#include "matrixVectorOp.h"



/*
 * Compute the matrix vector product y = A*x
 * where A = Agg^{-1}*S, S = Agg - sum(Agi*Aii^{-1}*Aig).
*/

int matrixVectorOp_AggInvxS(MPI_Comm comm, int mloc, int m, int *mcounts, int *mdispls,
                             CPLM_Mat_CSR_t *Agi, CPLM_Mat_CSR_t *Aii, CPLM_Mat_CSR_t *Aig, CPLM_Mat_CSR_t *Aggloc,
                             preAlps_solver_t *Aii_sv, preAlps_solver_t *Agg_sv, double *X, double *Y,
                             double *dwork1, double *dwork2, double *ywork,
                             SolverStats_t *tstats){

  int ierr = 0; int nrhs = 1;

  int my_rank, nbprocs, root = 0;
  double dONE = 1.0, dZERO = 0.0, dMONE = -1.0;
  double ttemp;


  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &nbprocs);

  /*
   * Compute Y = OP x X = A_{gg}^{-1}*S*X
   * S = Agg - sum(Agi*Aii^{-1}*Aig);
  */

  //Gather the vector from each procs
  ttemp = MPI_Wtime();
  MPI_Allgatherv(X, mloc, MPI_DOUBLE, ywork, mcounts, mdispls, MPI_DOUBLE, comm);
  tstats->tComm+= MPI_Wtime() - ttemp;

  /* Compute S*v */

  ttemp = MPI_Wtime();
  //ttemp1 = MPI_Wtime();
  CPLM_MatCSRMatrixVector(Aig, dONE, ywork, dZERO, dwork1);
  //tstats->tAv += MPI_Wtime() - ttemp1;

  //ttemp1 = MPI_Wtime();
  preAlps_solver_triangsolve(Aii_sv, Aii->info.m, Aii->val, Aii->rowPtr, Aii->colInd, nrhs, dwork2, dwork1);
  //tstats->tSolve += MPI_Wtime() - ttemp1;


  //ttemp1 = MPI_Wtime();
  CPLM_MatCSRMatrixVector(Agi, dONE, dwork2, dZERO, dwork1);
  //tstats->tAv += MPI_Wtime() - ttemp1;


  /* Sum on proc O */
  MPI_Reduce(dwork1, dwork2, Agi->info.m, MPI_DOUBLE, MPI_SUM, root, comm);

  /* Scatter dwork2 to X */
  MPI_Scatterv(dwork2, mcounts, mdispls, MPI_DOUBLE, X, mloc, MPI_DOUBLE, root, comm);


  //Copy of Su must be in X
  //ttemp1 = MPI_Wtime();
  CPLM_MatCSRMatrixVector(Aggloc, dONE, ywork, dMONE, X);
  //tstats->tAv += MPI_Wtime() - ttemp1;
  tstats->tSv += MPI_Wtime() - ttemp;

  /* Compute Y = inv(Agg)*X => solve A Y = X with the previous factorized matrix*/

  ttemp = MPI_Wtime();
  //Centralized rhs, gather X on the host
  MPI_Gatherv(X, mloc, MPI_DOUBLE, dwork2, mcounts, mdispls, MPI_DOUBLE, root, comm);

  //solve the system

  preAlps_solver_triangsolve(Agg_sv, Aggloc->info.m, Aggloc->val, Aggloc->rowPtr, Aggloc->colInd, nrhs, NULL, dwork2);

  //centralized solution, scatter Y to each procs
  MPI_Scatterv(dwork2, mcounts, mdispls, MPI_DOUBLE, Y, mloc, MPI_DOUBLE, root, comm);

  tstats->tInvAv += MPI_Wtime() - ttemp;

  return ierr;
}


/*
 * Compute the matrix vector product y = A*x (multilevel version)
 * where A = Agg^{-1}*S, S = Agg - sum(Agi*Aii^{-1}*Aig).
*/

int matrixVectorOp_AggInvxS_mlevel(int mloc, int m, int *mcounts, int *mdispls,
                             CPLM_Mat_CSR_t *Agi, CPLM_Mat_CSR_t *Aii, CPLM_Mat_CSR_t *Aig, CPLM_Mat_CSR_t *Aggloc,
                             preAlps_solver_t *Aii_sv, preAlps_solver_t *Agg_sv, double *X, double *Y,
                             double *dwork1, double *dwork2, double *ywork,
                             MPI_Comm comm_masterGroup,
                             MPI_Comm comm_localGroup,
                             int *Aii_mcounts, int *Aii_moffsets,
                             int *Aig_mcounts, int *Aig_moffsets,
                             int *Agi_mcounts, int *Agi_moffsets,
                             SolverStats_t *tstats){

  int ierr = 0; int nrhs = 1;

  int my_rank, nbprocs, root = 0;
  int masterGroup_myrank, masterGroup_nbprocs, localGroup_myrank, localGroup_nbprocs, local_root = 0;

  double dONE = 1.0, dZERO = 0.0, dMONE = -1.0;
  double ttemp, ttemp1;

  int Aii_m, Agi_m;

  //MPI_Comm_rank(comm, &my_rank);
  //MPI_Comm_size(comm, &nbprocs);

  preAlps_int_printSynchronized(1, "matvec dbg", comm_localGroup);

  if(comm_masterGroup!=MPI_COMM_NULL){
    MPI_Comm_rank(comm_masterGroup, &masterGroup_myrank);
    MPI_Comm_size(comm_masterGroup, &masterGroup_nbprocs);
  }
  MPI_Comm_rank(comm_localGroup, &localGroup_myrank);
  MPI_Comm_size(comm_localGroup, &localGroup_nbprocs);


  //The global size of Aii
  Aii_m = Aii_moffsets[localGroup_nbprocs];
  Agi_m = Agi_moffsets[localGroup_nbprocs];
  /*
   * Compute Y = OP x X = A_{gg}^{-1}*S*X
   * S = Agg - sum(Agi*Aii^{-1}*Aig);
  */

  preAlps_int_printSynchronized(2, "matvec dbg", comm_localGroup);

  //Gather the vector from each procs
  if(comm_masterGroup!=MPI_COMM_NULL){
    ttemp1 = MPI_Wtime();
    MPI_Allgatherv(X, mloc, MPI_DOUBLE, ywork, mcounts, mdispls, MPI_DOUBLE, comm_masterGroup);
    tstats->tComm+= MPI_Wtime() - ttemp1;
  }

  //Broadcast the vector to the local group
  if(localGroup_nbprocs>1){
    MPI_Bcast(ywork, m, MPI_DOUBLE, local_root, comm_localGroup);
  }


  /*
   * Compute S*v
   */
  ttemp = MPI_Wtime();

  //ttemp1 = MPI_Wtime();
  CPLM_MatCSRMatrixVector(Aig, dONE, ywork, dZERO, dwork1);
  //tstats->tAv += MPI_Wtime() - ttemp1;
  if(localGroup_nbprocs>1){ //gather on the master proc
    ttemp1 = MPI_Wtime();
    MPI_Gatherv(localGroup_myrank==local_root?MPI_IN_PLACE:dwork1, Aig->info.m, MPI_DOUBLE,  dwork1, Aig_mcounts, Aig_moffsets, MPI_DOUBLE, local_root, comm_localGroup);
    tstats->tComm+= MPI_Wtime() - ttemp1;
  }


  //ttemp1 = MPI_Wtime();
  if(Aii_sv->type==SOLVER_MUMPS){
    preAlps_solver_triangsolve(Aii_sv, Aii->info.m, Aii->val, Aii->rowPtr, Aii->colInd, nrhs, NULL, dwork1);
    memcpy(dwork2, dwork1, Aii_m*nrhs*sizeof(double));
  }
  else{
    preAlps_solver_triangsolve(Aii_sv, Aii->info.m, Aii->val, Aii->rowPtr, Aii->colInd, nrhs, dwork2, dwork1);
  }
  //tstats->tSolve += MPI_Wtime() - ttemp1;

  //Broadcast the vector to the local group
  if(localGroup_nbprocs>1){
    ttemp1 = MPI_Wtime();
    MPI_Bcast(dwork2, Aii_m, MPI_DOUBLE, local_root, comm_localGroup);
    tstats->tComm+= MPI_Wtime() - ttemp1;
  }

  //dwork1 = Agi*dwork2
  //ttemp1 = MPI_Wtime();
  CPLM_MatCSRMatrixVector(Agi, dONE, dwork2, dZERO, dwork1);
  if(localGroup_nbprocs>1){ //gather on the master proc
    ttemp1 = MPI_Wtime();
    MPI_Gatherv(localGroup_myrank==local_root?MPI_IN_PLACE:dwork1, Agi->info.m, MPI_DOUBLE,  dwork1, Agi_mcounts, Agi_moffsets, MPI_DOUBLE, local_root, comm_localGroup);
    tstats->tComm+= MPI_Wtime() - ttemp1;
  }
  //tstats->tAv += MPI_Wtime() - ttemp1;


  if(comm_masterGroup!=MPI_COMM_NULL){

    /* Sum on proc O */
    ttemp1 = MPI_Wtime();
    MPI_Reduce(dwork1, dwork2, Agi_m, MPI_DOUBLE, MPI_SUM, root, comm_masterGroup);
    tstats->tComm+= MPI_Wtime() - ttemp1;



    /* Scatter dwork2 to X */
    ttemp1 = MPI_Wtime();
    MPI_Scatterv(dwork2, mcounts, mdispls, MPI_DOUBLE, X, mloc, MPI_DOUBLE, root, comm_masterGroup);
    tstats->tComm+= MPI_Wtime() - ttemp1;

    //Copy of Su must be in X
    //ttemp1 = MPI_Wtime();
    CPLM_MatCSRMatrixVector(Aggloc, dONE, ywork, dMONE, X);

  }

  //tstats->tAv += MPI_Wtime() - ttemp1;
  tstats->tSv += MPI_Wtime() - ttemp;

  /*
   * Compute Y = inv(Agg)*X => solve A Y = X with the previous factorized matrix
  */

  if(comm_masterGroup!=MPI_COMM_NULL){


    ttemp = MPI_Wtime();
    //Centralized rhs, gather X on the host
    MPI_Gatherv(X, mloc, MPI_DOUBLE, dwork2, mcounts, mdispls, MPI_DOUBLE, root, comm_masterGroup);

    //solve the system

    preAlps_solver_triangsolve(Agg_sv, Aggloc->info.m, Aggloc->val, Aggloc->rowPtr, Aggloc->colInd, nrhs, NULL, dwork2);

    #ifdef DEBUG
    if(masterGroup_myrank==root) preAlps_doubleVector_printSynchronized(dwork2, m, "Y", " Y proc 0 after matvec", MPI_COMM_SELF);
    #endif
    //centralized solution, scatter Y to each procs
    MPI_Scatterv(dwork2, mcounts, mdispls, MPI_DOUBLE, Y, mloc, MPI_DOUBLE, root, comm_masterGroup);

    tstats->tInvAv += MPI_Wtime() - ttemp;


    preAlps_doubleVector_printSynchronized(Y, mloc, "Y", "Y after matvec", comm_masterGroup);
  }

  return ierr;
}


/* Compute the matrix vector product y = A*x
 * where A = A_{loc}^{-1}*S, S = Aggloc - Agi*Aii^{-1}*Aig.
*/
int matrixVectorOp_AlocInvxS(MPI_Comm comm, int mloc, int m, int *mcounts, int *mdispls,
                             CPLM_Mat_CSR_t *Aggloc, CPLM_Mat_CSR_t *Agi, CPLM_Mat_CSR_t *Aii, CPLM_Mat_CSR_t *Aig, CPLM_Mat_CSR_t *Aloc,
                             preAlps_solver_t *Aii_sv, preAlps_solver_t *Aloc_sv, double *X, double *Y,
                             double *dwork1, double *dwork2, double *ywork,
                             SolverStats_t *tstats){

 int ierr = 0; int nrhs =1;

 int my_rank, nbprocs;
 double dONE = 1.0, dZERO = 0.0, dMONE = -1.0;
 double ttemp;


 MPI_Comm_rank(comm, &my_rank);
 MPI_Comm_size(comm, &nbprocs);

  /*
   * Compute Y = OP x X = A_{loc}^{-1}S*X
   * Si = Agg - Agi*Aii^{-1}*Aig
  */
  //Gather the vector from each procs
  ttemp = MPI_Wtime();
  MPI_Allgatherv(X, mloc, MPI_DOUBLE, ywork, mcounts, mdispls, MPI_DOUBLE, comm);
  tstats->tComm+= MPI_Wtime() - ttemp;

  /* Compute S*v */

  ttemp = MPI_Wtime();
  //ttemp1 = MPI_Wtime();
  CPLM_MatCSRMatrixVector(Aig, dONE, X, dZERO, dwork1);
  //tstats->tAv += MPI_Wtime() - ttemp1;

  //ttemp1 = MPI_Wtime();
  preAlps_solver_triangsolve(Aii_sv, Aii->info.m, Aii->val, Aii->rowPtr, Aii->colInd, nrhs, dwork2, dwork1);
  //tstats->tSolve += MPI_Wtime() - ttemp1;

  //ttemp1 = MPI_Wtime();
  CPLM_MatCSRMatrixVector(Agi, dONE, dwork2, dZERO, X);
  //tstats->tAv += MPI_Wtime() - ttemp1;

  //Copy of Su must be in X

  //ttemp1 = MPI_Wtime();
  CPLM_MatCSRMatrixVector(Aggloc, dONE, ywork, dMONE, X);
  //tstats->tAv += MPI_Wtime() - ttemp1;
  tstats->tSv += MPI_Wtime() - ttemp;

  /* Compute Y = inv(Aloc)*X => solve A Y = X with the previous factorized matrix*/
  ttemp = MPI_Wtime();
  preAlps_solver_triangsolve(Aloc_sv, Aloc->info.m, Aloc->val, Aloc->rowPtr, Aloc->colInd, nrhs, Y, X);
  tstats->tInvAv += MPI_Wtime() - ttemp;

  return ierr;
}



/* Compute the matrix vector product y = A*x
 * where A = S*S_{loc}^{-1}, S_{loc} = Block-Diag(S).
 * S*S_{loc}^{-1} = (I + AggP*S_{loc}^{-1})
*/
int matrixVectorOp_SxSlocInv(MPI_Comm comm, int mloc, int m, int *mcounts, int *mdispls,
                            preAlps_solver_t *Sloc_sv, CPLM_Mat_CSR_t *Sloc, CPLM_Mat_CSR_t *AggP,
                            double *X, double *Y, double *dwork, double *ywork,
                            SolverStats_t *tstats){
  int ierr = 0; int nrhs =1;
  double ttemp;
  int i;
  int my_rank, nbprocs;
  double dONE = 1.0, dZERO = 0.0;

  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &nbprocs);


  /*
   * Compute Y = OP x X = (I + AggP*S_{loc}^{-1}) x X
  */

  /* Compute  y = S_{loc}^{-1} x X => solve S_{loc} y = X */
  /* Solve A x = b with the previous factorized matrix*/
  ttemp = MPI_Wtime();
  preAlps_solver_triangsolve(Sloc_sv, Sloc->info.m, Sloc->val, Sloc->rowPtr, Sloc->colInd, nrhs, dwork, X);
  tstats->tSolve += MPI_Wtime() - ttemp;

  //Gather the vector from each
  ttemp = MPI_Wtime();
  //MPI_Allgather(dwork, mloc, MPI_INT, ywork, 1, MPI_INT, comm);
  MPI_Allgatherv(dwork, mloc, MPI_DOUBLE, ywork, mcounts, mdispls, MPI_DOUBLE, comm);
  tstats->tComm+= MPI_Wtime() - ttemp;

  /* Y = (I + AggP x S_{loc}^{-1})X = I*X +  AggP * S_{loc}^{-1}*X = I.X + AggP*ywork */

  ttemp = MPI_Wtime();
  CPLM_MatCSRMatrixVector(AggP, dONE, ywork, dZERO, Y);
  tstats->tAv += MPI_Wtime() - ttemp;

  for(i=0;i<mloc;i++) Y[i] = X[i]+Y[i];

  return ierr;
}

/* Compute the matrix vector product y = A*x
 * where A = S_{loc}^{-1}*S, S_{loc} = Block-Diag(S).
 * S_{loc}^{-1}*S = (I + S_{loc}^{-1}AggP) * X
*/
int matrixVectorOp_SlocInvxS(MPI_Comm comm, int mloc, int m, int *mcounts, int *mdispls,
                            preAlps_solver_t *Sloc_sv, CPLM_Mat_CSR_t *Sloc, CPLM_Mat_CSR_t *AggP,
                            double *X, double *Y, double *dwork, double *ywork,
                            SolverStats_t *tstats){
  int ierr = 0; int nrhs =1;
  double ttemp;

  int i;
  int my_rank, nbprocs;
  double dONE = 1.0, dZERO = 0.0;

  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &nbprocs);

  /*
   * Compute Y = OP x X = S_{loc}^{-1}S*X = (I + S_{loc}^{-1}AggP) * X
  */

  //Gather the vector X from each procs
  ttemp = MPI_Wtime();
  MPI_Allgatherv(X, mloc, MPI_DOUBLE, ywork, mcounts, mdispls, MPI_DOUBLE, comm);
  tstats->tComm+= MPI_Wtime() - ttemp;

  ttemp = MPI_Wtime();
  CPLM_MatCSRMatrixVector(AggP, dONE, ywork, dZERO, dwork);
  tstats->tAv += MPI_Wtime() - ttemp;

  /* Solve A x = b with the previous factorized matrix*/
  ttemp = MPI_Wtime();
  preAlps_solver_triangsolve(Sloc_sv, Sloc->info.m, Sloc->val, Sloc->rowPtr, Sloc->colInd, nrhs, Y, dwork);
  tstats->tSolve += MPI_Wtime() - ttemp;

  for(i=0;i<mloc;i++) Y[i] = X[i]+Y[i];

  /* Overwrite X with A*X (required mode=2)*/

  ttemp = MPI_Wtime();

  CPLM_MatCSRMatrixVector(Sloc, dONE, Y, dZERO, X);
  tstats->tAv += MPI_Wtime() - ttemp;

  return ierr;
}
