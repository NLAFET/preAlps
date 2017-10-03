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
#include "preAlps_utils.h"
#include "solverStats.h"
#include "matrixVectorOp.h"

/* Compute the matrix vector product y = A*x
 * where A = A_{loc}^{-1}*S, S = Aggloc - Agi*Aii^{-1}*Aig.
*/
int matrixVectorOp_AlocInvxS(MPI_Comm comm, int mloc, int m, int *mcounts, int *mdispls,
                             CPLM_Mat_CSR_t *Aggloc, CPLM_Mat_CSR_t *Agi, CPLM_Mat_CSR_t *Aii, CPLM_Mat_CSR_t *Aig, CPLM_Mat_CSR_t *Aloc,
                             preAlps_solver_t *Aii_sv, preAlps_solver_t *Aloc_sv, double *X, double *Y,
                             double *dwork1, double *dwork2, double *ywork,
                             SolverStats_t *tstats){

 int ierr = 0;

 int my_rank, nbprocs, root = 0;
 double dONE = 1.0, dZERO = 0.0, dMONE = -1.0;
 double ttemp;


 MPI_Comm_rank(comm, &my_rank);
 MPI_Comm_size(comm, &nbprocs);


  ///if(RCI_its==1) {for(i=0;i<mloc;i++) X[i] = 1e-2; printf("dbgsimp1\n");}

  /*
   * Compute Y = OP x X = A_{loc}^{-1}S*X
   * Si = Agg - Agi*Aii^{-1}*Aig
  */
  //Gather the vector from each procs
  ttemp = MPI_Wtime();
  MPI_Allgatherv(X, mloc, MPI_DOUBLE, ywork, mcounts, mdispls, MPI_DOUBLE, comm);
  tstats->tComm+= MPI_Wtime() - ttemp;

  if(my_rank==root) preAlps_doubleVector_printSynchronized(ywork, m, "ywork", "ywork", MPI_COMM_SELF);


  /* Compute S*v */

  ttemp = MPI_Wtime();
  //ttemp1 = MPI_Wtime();
  CPLM_MatCSRMatrixVector(Aig, dONE, X, dZERO, dwork1);
  //tstats->tAv += MPI_Wtime() - ttemp1;

  preAlps_doubleVector_printSynchronized(dwork1, Aig->info.m, "dwork1", "dwork1 = Aig*X", comm);

  //ttemp1 = MPI_Wtime();
  preAlps_solver_triangsolve(Aii_sv, Aii->info.m, Aii->val, Aii->rowPtr, Aii->colInd, dwork2, dwork1);
  //tstats->tSolve += MPI_Wtime() - ttemp1;

  preAlps_doubleVector_printSynchronized(dwork2, Aii->info.m, "dwork", "dwork2 = Aii^{-1}*dwork1", comm);

  //ttemp1 = MPI_Wtime();
  CPLM_MatCSRMatrixVector(Agi, dONE, dwork2, dZERO, X);
  //tstats->tAv += MPI_Wtime() - ttemp1;


  preAlps_doubleVector_printSynchronized(X, Agi->info.m, "X", "X = Agi*dwork2", comm);

  /* Copy dwork = Su in X */
  //for(i=0;i<mloc;i++) X[i] = dwork[i];

  //preAlps_doubleVector_printSynchronized(X, mloc, "X", "X = SX", comm);


  //Copy of Su must be in X

  //ttemp1 = MPI_Wtime();
  CPLM_MatCSRMatrixVector(Aggloc, dONE, ywork, dMONE, X);
  //tstats->tAv += MPI_Wtime() - ttemp1;
  tstats->tSv += MPI_Wtime() - ttemp;


  preAlps_doubleVector_printSynchronized(X, mloc, "X", "X = SX", comm);

  /* Compute Y = inv(Aloc)*X => solve A Y = X with the previous factorized matrix*/
  ttemp = MPI_Wtime();
  preAlps_solver_triangsolve(Aloc_sv, Aloc->info.m, Aloc->val, Aloc->rowPtr, Aloc->colInd, Y, X);
  tstats->tInvAv += MPI_Wtime() - ttemp;

  preAlps_doubleVector_printSynchronized(Y, mloc, "Y", "Y after matvec", comm);

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
  int ierr = 0;
  double ttemp;
  int i;
  int my_rank, nbprocs, root = 0;
  double dONE = 1.0, dZERO = 0.0;

  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &nbprocs);


  /*
   * Compute Y = OP x X = (I + AggP*S_{loc}^{-1}) x X
  */

  /* Compute  y = S_{loc}^{-1} x X => solve S_{loc} y = X */
  /* Solve A x = b with the previous factorized matrix*/
  ttemp = MPI_Wtime();
  preAlps_solver_triangsolve(Sloc_sv, Sloc->info.m, Sloc->val, Sloc->rowPtr, Sloc->colInd, dwork, X);
  tstats->tSolve += MPI_Wtime() - ttemp;

  preAlps_doubleVector_printSynchronized(dwork, mloc, "dwork", "dwork", comm);

  //Gather the vector from each
  ttemp = MPI_Wtime();
  //MPI_Allgather(dwork, mloc, MPI_INT, ywork, 1, MPI_INT, comm);
  MPI_Allgatherv(dwork, mloc, MPI_DOUBLE, ywork, mcounts, mdispls, MPI_DOUBLE, comm);
  tstats->tComm+= MPI_Wtime() - ttemp;

  //preAlps_doubleVector_printSynchronized(ywork, m, "ywork", "ywork", comm);

  if(my_rank==root) preAlps_doubleVector_printSynchronized(ywork, m, "ywork", "ywork", MPI_COMM_SELF);

  /* Y = (I + AggP x S_{loc}^{-1})X = I*X +  AggP * S_{loc}^{-1}*X = I.X + AggP*ywork */
  //if(RCI_its==1) {for(i=0;i<m;i++) ywork[i] = 1e-2; printf("dbgsimp2\n");}
  ttemp = MPI_Wtime();
  CPLM_MatCSRMatrixVector(AggP, dONE, ywork, dZERO, Y);
  tstats->tAv += MPI_Wtime() - ttemp;

  preAlps_doubleVector_printSynchronized(Y, mloc, "Y", "Y after AggP*ywork", comm);

  for(i=0;i<mloc;i++) Y[i] = X[i]+Y[i];

  preAlps_doubleVector_printSynchronized(Y, mloc, "Y", "Y after matvec", comm);

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
  int ierr = 0;
  double ttemp;

  int i;
  int my_rank, nbprocs, root = 0;
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

  if(my_rank==root) preAlps_doubleVector_printSynchronized(ywork, m, "ywork", "ywork", MPI_COMM_SELF);

  ttemp = MPI_Wtime();
  CPLM_MatCSRMatrixVector(AggP, dONE, ywork, dZERO, dwork);
  tstats->tAv += MPI_Wtime() - ttemp;

  preAlps_doubleVector_printSynchronized(dwork, mloc, "dwork", "dwork after AggP*ywork", comm);

  /* Solve A x = b with the previous factorized matrix*/
  ttemp = MPI_Wtime();
  preAlps_solver_triangsolve(Sloc_sv, Sloc->info.m, Sloc->val, Sloc->rowPtr, Sloc->colInd, Y, dwork);
  tstats->tSolve += MPI_Wtime() - ttemp;

  preAlps_doubleVector_printSynchronized(Y, mloc, "Y", "Y after Sloc-1*v", comm);

  for(i=0;i<mloc;i++) Y[i] = X[i]+Y[i];

  /* Overwrite X with A*X (required mode=2)*/

  ttemp = MPI_Wtime();
  //CPLM_MatCSRMatrixVector(Sloc, dONE, ywork, dONE, X);
  //CPLM_MatCSRMatrixVector(Sloc, dONE, X, dONE, X);
  CPLM_MatCSRMatrixVector(Sloc, dONE, Y, dZERO, X);
  tstats->tAv += MPI_Wtime() - ttemp;

  preAlps_doubleVector_printSynchronized(X, mloc, "X", "X = AX", comm);


  preAlps_doubleVector_printSynchronized(Y, mloc, "Y", "Y after matvec", comm);

  return ierr;
}
