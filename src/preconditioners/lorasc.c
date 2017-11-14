/*
============================================================================
Name        : lorasc.c
Author      : Simplice Donfack
Version     : 0.1
Description : Preconditioner based on Schur complement
Date        : Sept 20, 2017
============================================================================
*/
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <metis_interface.h>
#include "preAlps_utils.h"
#include "preAlps_solver.h"
#include "lorasc.h"
#include "lorasc_eigsolve.h"
/*Allocate workspace for the preconditioner*/
int preAlps_LorascAlloc(preAlps_Lorasc_t **lorasc){

  *lorasc = (preAlps_Lorasc_t*) malloc(sizeof(preAlps_Lorasc_t));

  if(*lorasc!=NULL){
    (*lorasc)->eigvalues=NULL;
    (*lorasc)->eigvectors=NULL;
    (*lorasc)->sigma=NULL;

    //partitioning and permutation vector
    (*lorasc)->partCount=NULL;
    (*lorasc)->partBegin=NULL;
    (*lorasc)->perm=NULL;

    //workspace
    (*lorasc)->vi=NULL;
    (*lorasc)->zi=NULL;
    (*lorasc)->dwork1=NULL;
    (*lorasc)->dwork2=NULL;
    (*lorasc)->eigWork=NULL;

    //default parameters
    (*lorasc)->deflation_tol = 1e-2;
    (*lorasc)->nrhs = 1;
    (*lorasc)->OptPermuteOnly = 0;

    //matrices
    (*lorasc)->Aii=NULL;
    (*lorasc)->Aig=NULL;
    (*lorasc)->Agi=NULL;
    (*lorasc)->Aggloc=NULL;
    //Solvers
    (*lorasc)->Aii_sv=NULL;
    (*lorasc)->Agg_sv=NULL;
  }

  return (*lorasc==NULL);
}

/*
 * Build the preconditioner
 * lorasc:
 *     input: the preconditioner object to construct
 * A:
 *     input: the input matrix on processor 0
 * locAP:
 *     output: the local permuted matrix on each proc after the preconditioner is built
*/
int preAlps_LorascBuild(preAlps_Lorasc_t *lorasc, CPLM_Mat_CSR_t *A, CPLM_Mat_CSR_t *locAP, MPI_Comm comm){

  int ierr = 0, nbprocs, my_rank, root = 0;
  int *perm = NULL, *perm1=NULL, m, n, nnz;
  int nparts, *partCount=NULL, *partBegin=NULL;
  int *sep_mcounts = NULL, *sep_moffsets=NULL, sep_nrows=0;
  CPLM_Mat_CSR_t AP = CPLM_MatCSRNULL();
  CPLM_Mat_CSR_t *Aii = NULL, *Aig =NULL, *Agi=NULL, *Aggloc=NULL;
  preAlps_solver_type_t stype;
  preAlps_solver_t *Aii_sv = NULL, *Agg_sv = NULL; //Solver to factorize Aii and Agg


  /*
   * 0. Let's begin
   */

  lorasc->eigvalues_deflation = 0;
  lorasc->comm = comm;

  //Let me know who I am
  MPI_Comm_size(comm, &nbprocs);
  MPI_Comm_rank(comm, &my_rank);

  // Create matrix objects
  CPLM_MatCSRCreateNULL(&Aii);
  CPLM_MatCSRCreateNULL(&Aig);
  CPLM_MatCSRCreateNULL(&Agi);
  CPLM_MatCSRCreateNULL(&Aggloc);

  // Broadcast the global matrix dimension from the root to the other procs
  preAlps_matrixDim_Bcast(comm, A, root, &m, &n, &nnz);


  // Allocate memory for the permutation array
  if ( !(perm  = (int *) malloc(m*sizeof(int))) ) preAlps_abort("Malloc fails for perm[].");
  if ( !(perm1  = (int *) malloc(m*sizeof(int))) ) preAlps_abort("Malloc fails for perm1[].");

  // Allocate memory for the distribution of the separator
  if ( !(sep_mcounts  = (int *) malloc((nbprocs) * sizeof(int))) ) preAlps_abort("Malloc fails for mcounts[].");
  if ( !(sep_moffsets = (int *) malloc((nbprocs+1) * sizeof(int))) ) preAlps_abort("Malloc fails for moffsets[].");

  /*
   * 1. Create a block arrow structure of the matrix
   */

  preAlps_blockArrowStructCreate(comm, m, A, &AP, perm1, &nparts, &partCount, &partBegin);


  // Check if each processor has at least one block
  if(nbprocs!=nparts-1){
    preAlps_abort("This number of process is not support yet. Please use a multiple of 2. nbprocs:%d, nparts created:%d\n", nbprocs, nparts-1);
  }

  /*
   * 2. Distribute the permuted matrix to all the processors
   */

  preAlps_blockArrowStructDistribute(comm, m, &AP, perm1, nparts, partCount, partBegin, locAP, perm,
                                     Aii, Aig, Agi, Aggloc, sep_mcounts, sep_moffsets);


  // Get the global number of rows of the separator
  sep_nrows = sep_moffsets[nbprocs];


  if(!lorasc->OptPermuteOnly){

    /*
     * 3. Factorize the blocks Aii and Agg
     */

    switch(SPARSE_SOLVER){
      case 0: stype = SOLVER_MKL_PARDISO; break;
      case 1: stype = SOLVER_PARDISO; break;
      case 2: stype = SOLVER_MUMPS; break;
      default: stype = 0;
    }

    // Factorize Aii
    preAlps_solver_create(&Aii_sv, stype, MPI_COMM_SELF);
    preAlps_solver_init(Aii_sv);
    preAlps_solver_setMatrixType(Aii_sv, SOLVER_MATRIX_REAL_NONSYMMETRIC);  //TODO: SOLVER_MATRIX_REAL_SYMMETRIC
    preAlps_solver_factorize(Aii_sv, Aii->info.m, Aii->val, Aii->rowPtr, Aii->colInd);


    // Factorize Agg in parallel
    stype = SOLVER_MUMPS; //only MUMPS is supported for the moment
    preAlps_solver_create(&Agg_sv, stype, comm);
    preAlps_solver_setMatrixType(Agg_sv, SOLVER_MATRIX_REAL_NONSYMMETRIC); //must be done before solver_init
    //set the global problem infos (required only for parallel solver)
    preAlps_solver_setGlobalMatrixParam(Agg_sv, sep_nrows, lorasc->nrhs, sep_moffsets[my_rank]);
    //initialize the solver
    preAlps_solver_init(Agg_sv);
    preAlps_solver_factorize(Agg_sv, Aggloc->info.m, Aggloc->val, Aggloc->rowPtr, Aggloc->colInd);


    /*
     * 4. Solve the eigenvalue problem
     */

    preAlps_LorascEigSolve(lorasc, comm, Aggloc->info.m, Agi, Aii, Aig, Aggloc, Aii_sv, Agg_sv);
  }

  /*
   * 5. Save infos for the application of the preconditioner
   */

  //Save partitioning infos
  lorasc->partCount    = partCount;
  lorasc->partBegin    = partBegin;
  lorasc->perm         = perm;

  //Matrices
  lorasc->Aii          = Aii;
  lorasc->Aig          = Aig;
  lorasc->Agi          = Agi;
  lorasc->Aggloc       = Aggloc;

  //Solvers
  lorasc->Aii_sv       = Aii_sv;
  lorasc->Agg_sv       = Agg_sv;

  //Separator infos
  lorasc->sep_mcounts  = sep_mcounts;
  lorasc->sep_moffsets = sep_moffsets;
  lorasc->sep_nrows    = sep_nrows;

  // Free memory
  free(perm1);
  CPLM_MatCSRFree(&AP);

  return ierr;
}

/*Destroy the preconditioner*/
int preAlps_LorascDestroy(preAlps_Lorasc_t **lorasc){

  preAlps_Lorasc_t *lorascA = *lorasc;

  if(!lorascA) return 0;

  //Free partitioning and permutation
  if(lorascA->partCount) free(lorascA->partCount);
  if(lorascA->partBegin) free(lorascA->partBegin);
  if(lorascA->perm)      free(lorascA->perm);

  //Free internal allocated workspace
  preAlps_LorascMatApplyWorkspaceFree(lorascA);

  //destroy the solvers
  if(lorascA->Aii_sv) {
    preAlps_solver_finalize(lorascA->Aii_sv, lorascA->Aii->info.m, lorascA->Aii->rowPtr, lorascA->Aii->colInd);
    free(lorascA->Aii_sv);
  }
  if(lorascA->Agg_sv) {
    preAlps_solver_finalize(lorascA->Agg_sv, lorascA->Aggloc->info.m, lorascA->Aggloc->rowPtr, lorascA->Aggloc->colInd);
    free(lorascA->Agg_sv);
  }

  //free matrix objects
  if(lorascA->Aii)    CPLM_MatCSRFree(lorascA->Aii);
  if(lorascA->Aig)    CPLM_MatCSRFree(lorascA->Aig);
  if(lorascA->Agi)    CPLM_MatCSRFree(lorascA->Agi);
  if(lorascA->Aggloc) CPLM_MatCSRFree(lorascA->Aggloc);

  //free separator infos
  if(lorascA->sep_mcounts)      free(lorascA->sep_mcounts);
  if(lorascA->sep_moffsets)     free(lorascA->sep_moffsets);

  if(lorascA->eigvalues!=NULL)  free(lorascA->eigvalues);
  if(lorascA->eigvectors!=NULL) free(lorascA->eigvectors);
  if(lorascA->sigma!=NULL)      free(lorascA->sigma);

  free(lorascA);

  return 0;
}



/*
 * Apply Lorasc preconditioner on a dense matrice
 * i.e Compute  W = M_{lorasc}^{-1} * V
 */

int preAlps_LorascMatApply(preAlps_Lorasc_t *lorasc, CPLM_Mat_Dense_t *V, CPLM_Mat_Dense_t *W){
  int ierr = 0, nbprocs, my_rank, root = 0;
  double *v, *vgloc, *w;
  int i, j, ldv, v_nrhs, ldw;
  int sep_mloc;
  double dMONE = -1.0, dONE=1.0, dZERO = 0.0;


  // Let's begin

  if(!lorasc) preAlps_abort("Please first use LorascBuild() to construct the preconditioner!");

  // Let me know who I am
  MPI_Comm comm  = lorasc->comm;
  MPI_Comm_size(comm, &nbprocs);
  MPI_Comm_rank(comm, &my_rank);

  // Retrieve  parameters from lorasc
  CPLM_Mat_CSR_t *Aii      = lorasc->Aii;
  CPLM_Mat_CSR_t *Aig      = lorasc->Aig;
  CPLM_Mat_CSR_t *Agi      = lorasc->Agi;
  CPLM_Mat_CSR_t *Aggloc   = lorasc->Aggloc;
  preAlps_solver_t *Aii_sv = lorasc->Aii_sv;
  preAlps_solver_t *Agg_sv = lorasc->Agg_sv;
  int *sep_mcounts         = lorasc->sep_mcounts;
  int *sep_moffsets        = lorasc->sep_moffsets;
  int sep_nrows            = lorasc->sep_nrows;

  // Get the vectorized notations of the matrices
  v = V->val; ldv = V->info.lda; v_nrhs = V->info.n;
  w = W->val; ldw = W->info.lda;
  int Aii_m = Aii->info.m;

#ifdef DEBUG
  printf("[%d] M:%d, N:%d, m:%d, n:%d, lda:%d, nval:%d\n", my_rank, V->info.M, V->info.N, V->info.m, V->info.n, V->info.lda, V->info.nval);
#endif


  //Prepare the reusable workspace
  preAlps_LorascMatApplyWorkspacePrepare(lorasc, Aii_m, sep_nrows, v_nrhs);

  //Get the buffers (after workspacePrepare())
  double *vi      = lorasc->vi;
  double *zi      = lorasc->zi;
  double *dwork1  = lorasc->dwork1;
  double *dwork2  = lorasc->dwork2;
  double *eigWork = lorasc->eigWork;

  //copy V to make it contiguous in memory as required by the solvers
  for(j=0;j<v_nrhs;j++){
    for(i=0;i<Aii_m;i++) vi[j*Aii_m+i] = v[j*ldv+i];
  }

  //Get the position and the local number of rows in the separator
  vgloc = &v[Aii_m];
  sep_mloc = sep_mcounts[my_rank];


  /*
   * 1. Compute dwork1 = M_L^{-1}*v; (Refer to the paper)
   */

  // Solve Aii*zi = vi

  preAlps_solver_triangsolve(Aii_sv, Aii->info.m, Aii->val, Aii->rowPtr, Aii->colInd, v_nrhs, zi, vi);


  // Create a vector with my set of rows from the separator and zero everywhere
  for(i=0;i<sep_nrows*v_nrhs;i++) dwork1[i] = 0;

  for(j=0;j<v_nrhs;j++){
    for(i=0;i<sep_mloc;i++) dwork1[j*sep_nrows+sep_moffsets[my_rank]+i] = vgloc[j*ldv+i]; //v[j*ldv+Aii_m+i]; //
  }

  // dwork = dwork - Agi*zi;
  if(v_nrhs==1) CPLM_MatCSRMatrixVector(Agi, dMONE, zi, dONE, dwork1); //faster than using matmult for nrhs = 1
  else CPLM_MatCSRMatrixCSRDenseMult(Agi, dMONE, zi, v_nrhs, Aii_m, dONE, dwork1, Agi->info.m);


  //Sum on proc O
  MPI_Reduce(my_rank==0?MPI_IN_PLACE:dwork1, dwork1, Agi->info.m*v_nrhs, MPI_DOUBLE, MPI_SUM, root, comm);


  // Compute y = Agg^{-1}*dwork1 + E * sigma * E^T * dwork1 on the root
  int E_r = lorasc->eigvalues_deflation;

  if(my_rank==root){
    // Compute eigWork = E^T * dwork1
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
           E_r, v_nrhs, sep_nrows, 1.0, lorasc->eigvectors, sep_nrows, dwork1, sep_nrows, 0.0, eigWork, E_r);
  }

  // Compute dwork1 = Agg^{-1}*dwork1 (overwrites dwork1)
  //the rhs is already centralized on the host, just solve the system
  preAlps_solver_triangsolve(Agg_sv, Aggloc->info.m, Aggloc->val, Aggloc->rowPtr, Aggloc->colInd, v_nrhs, NULL, dwork1);


  if(my_rank==root){

    // Finalize the computation y = Agg^{-1}*dwork1 + E * sigma * E^T * dwork1 on the root = dwork1 + E * sigma *eigWork
    // eigwork = sigma * eigwork
    for(j=0;j<v_nrhs;j++){
      for (i = 0; i < E_r; i++) {
        eigWork[j*E_r+i] = eigWork[j*E_r+i]*lorasc->sigma[i];
      }
    }

    // v = v + E * eigwork
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
           sep_nrows, v_nrhs, E_r, 1.0, lorasc->eigvectors, sep_nrows, eigWork, E_r, 1.0, dwork1, sep_nrows);
  }

  //Broadcast dwork to all processors
  MPI_Bcast(dwork1, sep_nrows*v_nrhs, MPI_DOUBLE, root, comm);


  /*
   * 2. Compute W = M_U^{-1}*v; (Refer to the paper)
   */

   // Wg = zg
   for(j=0;j<v_nrhs;j++){
     for(i=0;i<sep_mloc;i++) w[j*ldw+Aii_m+i] = dwork1[j*sep_nrows+sep_moffsets[my_rank]+i];
   }

  // Compute dwork2 = Aig*dwork1 (Aig*zg)
  if(v_nrhs==1) CPLM_MatCSRMatrixVector(Aig, dONE, dwork1, dZERO, dwork2); //faster than using matmult for nrhs = 1
  else CPLM_MatCSRMatrixCSRDenseMult(Aig, dONE, dwork1, v_nrhs, sep_nrows, dZERO, dwork2, Aig->info.m);

  // Compute vi = Aii^{-1}*dwork2  //use vi as buffer
  preAlps_solver_triangsolve(Aii_sv, Aii->info.m, Aii->val, Aii->rowPtr, Aii->colInd, v_nrhs, vi, dwork2);

  // w = zi - vi

  for(j=0;j<v_nrhs;j++){
    for(i=0;i<Aii_m;i++) w[j*ldw+i] = zi[j*Aii_m+i] - vi[j*Aii_m+i];
  }

#ifdef DEBUG
  preAlps_doubleVectorSet_printSynchronized(w, Aii_m, v_nrhs, ldw, "Wi", "Wi: Wi computed", comm);
#endif

  return ierr;
}

/* Free internal allocated workspace */
int preAlps_LorascMatApplyWorkspaceFree(preAlps_Lorasc_t *lorasc){
  int ierr = 0;
  if(lorasc->vi!=NULL)         free(lorasc->vi);
  if(lorasc->zi!=NULL)         free(lorasc->zi);
  if(lorasc->dwork1!=NULL)     free(lorasc->dwork1);
  if(lorasc->dwork2!=NULL)     free(lorasc->dwork2);
  if(lorasc->eigWork!=NULL)    free(lorasc->eigWork);
  return ierr;
}

/* Allocate workspace if required */
//Always check if the buffer has been allocated before, if so then to nothing.
int preAlps_LorascMatApplyWorkspacePrepare(preAlps_Lorasc_t *lorasc, int Aii_m, int separator_m, int v_nrhs){

  int ierr =0, max_m;

  if(lorasc->vi==NULL){
    if ( !(lorasc->vi  = (double *) malloc((Aii_m*v_nrhs) * sizeof(double))) ) preAlps_abort("Malloc fails for lorasc->zi[].");
  }

  if(lorasc->zi==NULL){
    if ( !(lorasc->zi  = (double *) malloc((Aii_m*v_nrhs) * sizeof(double))) ) preAlps_abort("Malloc fails for lorasc->zi[].");
  }

  // Determine the max workspace so we can use it for any operation
  max_m = Aii_m>separator_m ? Aii_m:separator_m;

  if(lorasc->dwork1==NULL){
    if ( !(lorasc->dwork1  = (double *) malloc((max_m*v_nrhs) * sizeof(double))) ) preAlps_abort("Malloc fails for lorasc->dwork1[].");
  }

  if(lorasc->dwork2==NULL){
    if ( !(lorasc->dwork2  = (double *) malloc((max_m*v_nrhs) * sizeof(double))) ) preAlps_abort("Malloc fails for lorasc->dwork2[].");
  }

  if(lorasc->eigWork==NULL){ //a buffer with the same as the number of eigenvalues computed
    if ( !(lorasc->eigWork  = (double *) malloc((lorasc->eigvalues_deflation*v_nrhs) * sizeof(double))) ) preAlps_abort("Malloc fails for lorasc->eigWork[].");
  }

  return ierr;
}
