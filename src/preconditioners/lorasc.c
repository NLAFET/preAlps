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
#include <string.h>
#include <mpi.h>
//#include <metis_interface.h>
#include "preAlps_utils.h"
#include "preAlps_solver.h"
#include "lorasc.h"
#include "lorasc_eigsolve.h"
/*Allocate workspace for the preconditioner*/
int preAlps_LorascAlloc(preAlps_Lorasc_t **lorasc){

  *lorasc = (preAlps_Lorasc_t*) malloc(sizeof(preAlps_Lorasc_t));

  if(*lorasc!=NULL){

    (*lorasc)->eigvalues  = NULL;
    (*lorasc)->eigvectors = NULL;

    // Global separator
    (*lorasc)->sep_mcounts  = NULL;
    (*lorasc)->sep_moffsets = NULL;
    (*lorasc)->sep_nrows    = 0;


    // Default parameters
    (*lorasc)->deflation_tol = 1e-2;
    (*lorasc)->nrhs          = 1;

    // Matrices
    (*lorasc)->Aii        = NULL;
    (*lorasc)->Aig        = NULL;
    (*lorasc)->Agi        = NULL;
    (*lorasc)->Aggloc     = NULL;

    // Solvers
    (*lorasc)->Aii_sv     = NULL;
    (*lorasc)->Agg_sv     = NULL;




    // Eigenvalues workspace
    (*lorasc)->sigma      = NULL;

    //multilevel var
    (*lorasc)->comm_masterLevel = MPI_COMM_NULL;
    (*lorasc)->comm_localLevel  = MPI_COMM_NULL;

    (*lorasc)->Aii_mcounts  = NULL;
    (*lorasc)->Aii_moffsets = NULL;
    (*lorasc)->Aig_mcounts  = NULL;
    (*lorasc)->Aig_moffsets = NULL;
    (*lorasc)->Agi_mcounts  = NULL;
    (*lorasc)->Agi_moffsets = NULL;

    // Workspace
    (*lorasc)->vi         = NULL;
    (*lorasc)->zi         = NULL;
    (*lorasc)->dwork1     = NULL;
    (*lorasc)->dwork2     = NULL;
    (*lorasc)->eigWork    = NULL;

  }

  return (*lorasc==NULL);
}

/*
 * Build the preconditioner
 * lorasc:
 *     input/output: the preconditioner object to construct
 * commMultilevel:
 *    input: a multilevel communicator built with preAlps_comm2LevelsSplit
 * A:
 *     input: the input matrix on processor 0
 * locAP:
 *     output: the local permuted matrix on each proc after the preconditioner is built
 * partCount:
 *     output: the number of rows in each part
 * partBegin:
 *     output: the begining rows of each part.
 * perm:
 *     output: the permutation vector
*/
int preAlps_LorascBuild(preAlps_Lorasc_t *lorasc, MPI_Comm *commMultilevel, CPLM_Mat_CSR_t *A, CPLM_Mat_CSR_t *locAP, int **partCount, int **partBegin, int *perm){

  int ierr = 0, nbprocs, my_rank, root = 0;
  MPI_Comm comm = MPI_COMM_NULL, comm_masterLevel = MPI_COMM_NULL, comm_localLevel = MPI_COMM_NULL;
  int masterLevel_myrank, masterGroup_nbprocs, localLevel_myrank, localGroup_nbprocs, local_root = 0;

  int *perm1=NULL, m, n, nnz;
  int nparts, *partCountWork=NULL, *partBeginWork=NULL;
  int *sep_mcounts = NULL, *sep_moffsets=NULL, sep_nrows=0;
  int *Aii_mcounts = NULL, *Aii_moffsets=NULL;
  int *Aig_mcounts = NULL, *Aig_moffsets=NULL;
  int *Agi_mcounts = NULL, *Agi_moffsets=NULL;

  CPLM_Mat_CSR_t AP = CPLM_MatCSRNULL(), Aiwork = CPLM_MatCSRNULL();
  CPLM_Mat_CSR_t *Aii = NULL, *Aig =NULL, *Agi=NULL, *Aggloc=NULL;

  preAlps_solver_type_t stype;
  preAlps_solver_t *Aii_sv = NULL, *Agg_sv = NULL; //Solver to factorize Aii and Agg

  double ttemp1 = 0.0;

  /*
   * 0. Let's begin
   */

  ttemp1 = MPI_Wtime();

  //initialization
  lorasc->eigvalues_deflation = 0;
  lorasc->tPartition = 0.0;

  // Get the multilevel communicators
  comm              = commMultilevel[0];
  comm_masterLevel  = commMultilevel[1];
  comm_localLevel   = commMultilevel[2];

  // Let me know who I am at each level
  MPI_Comm_size(comm, &nbprocs);
  MPI_Comm_rank(comm, &my_rank);

  if(comm_masterLevel!=MPI_COMM_NULL){
    MPI_Comm_rank(comm_masterLevel, &masterLevel_myrank);
	  MPI_Comm_size(comm_masterLevel, &masterGroup_nbprocs);
    //printf("myrank (world): %d, myrank (Level1): %d/%d\n", my_rank, masterLevel_myrank, masterGroup_nbprocs);
  }

  MPI_Comm_rank(comm_localLevel, &localLevel_myrank);
  MPI_Comm_size(comm_localLevel, &localGroup_nbprocs);

  // Create matrix objects
  CPLM_MatCSRCreateNULL(&Aii);
  CPLM_MatCSRCreateNULL(&Aig);
  CPLM_MatCSRCreateNULL(&Agi);
  CPLM_MatCSRCreateNULL(&Aggloc);

  // Broadcast the global matrix dimension from the root to the other procs
  if(comm_masterLevel!=MPI_COMM_NULL){
    CPLM_MatCSRDimensions_Bcast(A, root, &m, &n, &nnz, comm_masterLevel);
  }

  //Allocate memory
  if(comm_masterLevel!=MPI_COMM_NULL){
    // Allocate memory for the permutation array
    if ( !(perm1  = (int *) malloc(m*sizeof(int))) ) preAlps_abort("Malloc fails for perm1[].");

    // Allocate memory for the distribution of the separator
    if ( !(sep_mcounts  = (int *) malloc((masterGroup_nbprocs) * sizeof(int))) ) preAlps_abort("Malloc fails for sep_mcounts[].");
    if ( !(sep_moffsets = (int *) malloc((masterGroup_nbprocs+1) * sizeof(int))) ) preAlps_abort("Malloc fails for sep_moffsets[].");

  }

  /*
   * 1. Create a block arrow structure of the matrix
   */

  if(comm_masterLevel!=MPI_COMM_NULL){

    preAlps_blockArrowStructCreate(comm_masterLevel, m, A, &AP, perm1, &nparts, &partCountWork, &partBeginWork);

    // Check if each processor has at least one block
    if(masterGroup_nbprocs!=nparts-1){
      preAlps_abort("This number of process is not support yet. Please use a multiple of 2. nbprocs (level 1): %d, nparts created:%d\n", masterGroup_nbprocs, nparts-1);
    }
  }

  /*
   * 2. Distribute the permuted matrix to all the processors
   */

  if(comm_masterLevel!=MPI_COMM_NULL){

    preAlps_blockArrowStructDistribute(comm_masterLevel, m, &AP, perm1, nparts, partCountWork, partBeginWork, locAP, perm,
                                       Aii, Aig, Agi, Aggloc, sep_mcounts, sep_moffsets);

    // Get the global number of rows of the separator
    sep_nrows = sep_moffsets[masterGroup_nbprocs];

  }
  lorasc->tPartition = MPI_Wtime() - ttemp1;


  // Allocate memory for the distribution of the blocks of the Arrow Block Strcuture

  if ( !(Aii_mcounts  = (int *) malloc((localGroup_nbprocs) * sizeof(int))) ) preAlps_abort("Malloc fails for Aii_mcounts[].");
  if ( !(Aii_moffsets = (int *) malloc((localGroup_nbprocs+1) * sizeof(int))) ) preAlps_abort("Malloc fails for Aii_moffsets[].");
  if ( !(Agi_mcounts  = (int *) malloc((localGroup_nbprocs) * sizeof(int))) ) preAlps_abort("Malloc fails for Agi_mcounts[].");
  if ( !(Agi_moffsets = (int *) malloc((localGroup_nbprocs+1) * sizeof(int))) ) preAlps_abort("Malloc fails for Agi_moffsets[].");
  if ( !(Aig_mcounts  = (int *) malloc((localGroup_nbprocs) * sizeof(int))) ) preAlps_abort("Malloc fails for Aig_mcounts[].");
  if ( !(Aig_moffsets = (int *) malloc((localGroup_nbprocs+1) * sizeof(int))) ) preAlps_abort("Malloc fails for Aig_moffsets[].");

  //Distribute the blocks to the local group

  CPLM_MatCSRBlockRowDistribute(Aii, &Aiwork, Aii_mcounts, Aii_moffsets, local_root, comm_localLevel);
  CPLM_MatCSRCopy(&Aiwork, Aii);

  CPLM_MatCSRBlockRowDistribute(Agi, &Aiwork, Agi_mcounts, Agi_moffsets, local_root, comm_localLevel);
  CPLM_MatCSRCopy(&Aiwork, Agi);

  CPLM_MatCSRBlockRowDistribute(Aig, &Aiwork, Aig_mcounts, Aig_moffsets, local_root, comm_localLevel);
  CPLM_MatCSRCopy(&Aiwork, Aig);

  /*
   * 3. Factorize the blocks Aii and Agg
   */

  // Factorize Aii sequentially or in parallel depending on the number of procs within each blocks

  if(localGroup_nbprocs>1){
    stype = SOLVER_MUMPS; //only MUMPS is supported for the moment for the parallel case
  }else{
    switch(SPARSE_SOLVER){
      case 0: stype = SOLVER_MKL_PARDISO; break;
      case 1: stype = SOLVER_PARDISO; break;
      case 2: stype = SOLVER_MUMPS; break;
      default: stype = 0;
    }
  }

  if(stype==SOLVER_MUMPS){
    preAlps_solver_create(&Aii_sv, stype, comm_localLevel);
    //set the global problem infos (required only for parallel solver)
    preAlps_solver_setGlobalMatrixParam(Aii_sv, Aii_moffsets[localGroup_nbprocs], lorasc->nrhs, Aii_moffsets[localLevel_myrank]);
  }else{
    preAlps_solver_create(&Aii_sv, stype, MPI_COMM_SELF);
  }

  preAlps_solver_setMatrixType(Aii_sv, SOLVER_MATRIX_REAL_SPD);
  preAlps_solver_init(Aii_sv);
  preAlps_solver_factorize(Aii_sv, Aii->info.m, Aii->val, Aii->rowPtr, Aii->colInd);


  // Factorize Agg in parallel

  if(comm_masterLevel!=MPI_COMM_NULL){
    stype = SOLVER_MUMPS; //only MUMPS is supported for the moment
    preAlps_solver_create(&Agg_sv, stype, comm_masterLevel);
    preAlps_solver_setMatrixType(Agg_sv, SOLVER_MATRIX_REAL_SPD); //must be done before solver_init
    //set the global problem infos (required only for parallel solver)
    preAlps_solver_setGlobalMatrixParam(Agg_sv, sep_nrows, lorasc->nrhs, sep_moffsets[masterLevel_myrank]);
    //initialize the solver
    preAlps_solver_init(Agg_sv);
    preAlps_solver_factorize(Agg_sv, Aggloc->info.m, Aggloc->val, Aggloc->rowPtr, Aggloc->colInd);
  }

  // Restore output vectors
  *partCount    = partCountWork;
  *partBegin    = partBeginWork;

  /*
   * 4. Save infos for the application of the preconditioner
   */

  lorasc->comm             = comm;
  lorasc->comm_masterLevel = comm_masterLevel;
  lorasc->comm_localLevel  = comm_localLevel;





  // Matrices
  lorasc->Aii          = Aii;
  lorasc->Aig          = Aig;
  lorasc->Agi          = Agi;
  lorasc->Aggloc       = Aggloc;

  // Distribution of the matrices to the local processors
  lorasc->Aii_mcounts      = Aii_mcounts;
  lorasc->Aii_moffsets     = Aii_moffsets;
  lorasc->Aig_mcounts      = Aig_mcounts;
  lorasc->Aig_moffsets     = Aig_moffsets;
  lorasc->Agi_mcounts      = Agi_mcounts;
  lorasc->Agi_moffsets     = Agi_moffsets;

  // Solvers
  lorasc->Aii_sv       = Aii_sv;
  lorasc->Agg_sv       = Agg_sv;

  // Separator infos
  lorasc->sep_mcounts  = sep_mcounts;
  lorasc->sep_moffsets = sep_moffsets;
  lorasc->sep_nrows    = sep_nrows;

  /*
   * 5.  Solve the eigenvalue problem
   */

  preAlps_LorascEigSolve(lorasc, Aggloc->info.m);


  // Free memory
  free(perm1);
  CPLM_MatCSRFree(&AP);
  CPLM_MatCSRFree(&Aiwork);

  return ierr;
}

/*Destroy the preconditioner*/
int preAlps_LorascDestroy(preAlps_Lorasc_t **lorasc){

  preAlps_Lorasc_t *lorascA = *lorasc;

  if(!lorascA) return 0;

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
  if(lorascA->sep_mcounts)     free(lorascA->sep_mcounts);
  if(lorascA->sep_moffsets)    free(lorascA->sep_moffsets);

  if(lorascA->eigvalues)       free(lorascA->eigvalues);
  if(lorascA->eigvectors)      free(lorascA->eigvectors);

  if(lorascA->sigma)           free(lorascA->sigma);

  //multilevel
  if(lorascA->Aii_mcounts)     free(lorascA->Aii_mcounts);
  if(lorascA->Aii_moffsets)    free(lorascA->Aii_moffsets);
  if(lorascA->Aig_mcounts)     free(lorascA->Aig_mcounts);
  if(lorascA->Aig_moffsets)    free(lorascA->Aig_moffsets);
  if(lorascA->Agi_mcounts)     free(lorascA->Agi_mcounts);
  if(lorascA->Agi_moffsets)    free(lorascA->Agi_moffsets);

  free(lorascA);

  return 0;
}



/*
 * Apply Lorasc preconditioner on a dense matrice
 * i.e Compute  W = M_{lorasc}^{-1} * V
 */

int preAlps_LorascMatApply(preAlps_Lorasc_t *lorasc, CPLM_Mat_Dense_t *V, CPLM_Mat_Dense_t *W){
  int ierr = 0, nbprocs, my_rank, root = 0;
  double *v = NULL, *vgloc = NULL, *w = NULL;
  int i, j, ldv, v_nrhs, ldw;
  int sep_mloc;
  double dMONE = -1.0, dONE=1.0, dZERO = 0.0;
  int masterLevel_myrank, masterGroup_nbprocs, localLevel_myrank, localGroup_nbprocs, local_root =0;

  int Aii_m, Agi_m, sep_m, Aig_m;

  MPI_Datatype localType = MPI_DATATYPE_NULL, globalType = MPI_DATATYPE_NULL;


  // Let's begin

  // Check args
  if(!lorasc) preAlps_abort("Please first use LorascBuild() to construct the preconditioner!");

  // Retrieve  parameters from lorasc
  MPI_Comm comm             = lorasc->comm;
  MPI_Comm comm_masterLevel = lorasc->comm_masterLevel;
  MPI_Comm comm_localLevel  = lorasc->comm_localLevel;

  CPLM_Mat_CSR_t *Aii       = lorasc->Aii;
  CPLM_Mat_CSR_t *Aig       = lorasc->Aig;
  CPLM_Mat_CSR_t *Agi       = lorasc->Agi;
  CPLM_Mat_CSR_t *Aggloc    = lorasc->Aggloc;
  preAlps_solver_t *Aii_sv  = lorasc->Aii_sv;
  preAlps_solver_t *Agg_sv  = lorasc->Agg_sv;

  //int *Aii_mcounts        = lorasc->Aii_mcounts; //not used
  int *Aii_moffsets         = lorasc->Aii_moffsets;
  int *Aig_mcounts          = lorasc->Aig_mcounts;
  int *Aig_moffsets         = lorasc->Aig_moffsets;
  int *Agi_mcounts          = lorasc->Agi_mcounts;
  int *Agi_moffsets         = lorasc->Agi_moffsets;

  int *sep_mcounts          = lorasc->sep_mcounts;
  int *sep_moffsets         = lorasc->sep_moffsets;
  int sep_nrows             = lorasc->sep_nrows;

  // Let me know who I am at each level
  MPI_Comm_size(comm, &nbprocs);
  MPI_Comm_rank(comm, &my_rank);

  if(comm_masterLevel!=MPI_COMM_NULL){
    MPI_Comm_rank(comm_masterLevel, &masterLevel_myrank);
    MPI_Comm_size(comm_masterLevel, &masterGroup_nbprocs);
  }
  MPI_Comm_rank(comm_localLevel, &localLevel_myrank);
  MPI_Comm_size(comm_localLevel, &localGroup_nbprocs);


  //Get pointers to the local position in each matrices

  Aii_m = Aii_moffsets[localGroup_nbprocs]; //;Aii->info.m;
  Agi_m = sep_m = Agi_moffsets[localGroup_nbprocs];
  Aig_m = Aig_moffsets[localGroup_nbprocs];

  // Get the vectorized notations of the matrices
  if(comm_masterLevel!=MPI_COMM_NULL){
    v = V->val; ldv = V->info.lda; v_nrhs = V->info.n;
    w = W->val; ldw = W->info.lda;
  }

  //broadcast nrhs to the local group as it might change from one iteration to another with the block size reduction
  if(localGroup_nbprocs>1){
    MPI_Bcast(&v_nrhs, 1, MPI_INT, local_root, comm_localLevel);
  }

#ifdef DEBUG
  if(comm_masterLevel!=MPI_COMM_NULL){
    //printf("[%d]  m:%d, n:%d, lda:%d, nval:%d, nrhs:%d, Aii_m:%d, sep_m:%d, sep_nrows:%d\n", my_rank, V->info.m, V->info.n, V->info.lda, V->info.nval, v_nrhs, Aii_m, sep_m, sep_nrows);
    printf("[%d] M:%d, N:%d, m:%d, n:%d, lda:%d, nval:%d, nrhs:%d, Aii_m:%d, sep_m:%d, sep_nrows:%d\n", my_rank, V->info.M, V->info.N, V->info.m, V->info.n, V->info.lda, V->info.nval, v_nrhs, Aii_m, sep_m, sep_nrows);
  }
#endif

  //Prepare the reusable workspace
  preAlps_LorascMatApplyWorkspacePrepare(lorasc, Aii_m, sep_m, v_nrhs);

  //Get the buffers (after workspacePrepare())
  double *vi      = lorasc->vi;
  double *zi      = lorasc->zi;
  double *dwork1  = lorasc->dwork1;
  double *dwork2  = lorasc->dwork2;
  double *eigWork = lorasc->eigWork;

  //copy V to make it contiguous in memory as required by the solvers

  if(comm_masterLevel!=MPI_COMM_NULL){

    for(j=0;j<v_nrhs;j++){
      for(i=0;i<Aii_m;i++) vi[j*Aii_m+i] = v[j*ldv+i];
    }

    //Get the position and the local number of rows in the separator
    vgloc = &v[Aii_m];
    sep_mloc = sep_mcounts[masterLevel_myrank];
  }


  /*
   * 1. Compute dwork1 = M_L^{-1}*v; (Refer to the paper)
   */


  // Solve Aii*zi = vi

  if(Aii_sv->type==SOLVER_MUMPS){

    preAlps_solver_triangsolve(Aii_sv, Aii->info.m, Aii->val, Aii->rowPtr, Aii->colInd, v_nrhs, NULL, vi);

    if(localLevel_myrank==0) memcpy(zi, vi, Aii_m*v_nrhs*sizeof(double));
  }
  else{
    preAlps_solver_triangsolve(Aii_sv, Aii->info.m, Aii->val, Aii->rowPtr, Aii->colInd, v_nrhs, zi, vi);
  }

  // Compute dwork = dwork - Agi*zi

  for(i=0;i<sep_m*v_nrhs;i++) dwork1[i] = 0;

  //compute -Agi*zi

  //Broadcast the vector to the local group
  if(localGroup_nbprocs>1){
    MPI_Bcast(zi, Aii_m*v_nrhs, MPI_DOUBLE, local_root, comm_localLevel);
  }

  if(v_nrhs==1) CPLM_MatCSRMatrixVector(Agi, dMONE, zi, dZERO, dwork1); //faster than using matmult for nrhs = 1
  else CPLM_MatCSRMatrixCSRDenseMult(Agi, dMONE, zi, v_nrhs, Aii_m, dZERO, dwork1, Agi->info.m);

  //gather the result on the master proc
  if(localGroup_nbprocs>1){ //gather on the master proc

    // copy the root block in place
    if(localLevel_myrank==local_root) dlacpy("A", &Agi->info.m, &v_nrhs, dwork1, &Agi->info.m, dwork1, &Agi_m);

    preAlps_multiColumnTypeVectorCreate(v_nrhs, Agi->info.m, Agi_m, &localType, &globalType);

    MPI_Gatherv(localLevel_myrank==local_root?MPI_IN_PLACE:dwork1, Agi->info.m, localType, dwork1, Agi_mcounts, Agi_moffsets, globalType, local_root, comm_localLevel);
    //MPI_Gatherv(b, solver->mat->m, localType, rhs, counts, displs, globalType, 0, solver->comm);
  }

  // add a vector with my set of rows from the separator and zero everywhere
  if(comm_masterLevel!=MPI_COMM_NULL){
    for(j=0;j<v_nrhs;j++){
      //for(i=0;i<sep_mloc;i++) dwork1[j*sep_nrows+sep_moffsets[my_rank]+i] = vgloc[j*ldv+i]; //v[j*ldv+Aii_m+i]; //
      for(i=0;i<sep_mloc;i++) dwork1[j*sep_m+sep_moffsets[masterLevel_myrank]+i] += vgloc[j*ldv+i]; //v[j*ldv+Aii_m+i]; //
    }
  }

  //Sum on proc O
  if(comm_masterLevel!=MPI_COMM_NULL){
    MPI_Reduce(my_rank==0?MPI_IN_PLACE:dwork1, dwork1, Agi_m*v_nrhs, MPI_DOUBLE, MPI_SUM, root, comm_masterLevel);
  }


  if(comm_masterLevel!=MPI_COMM_NULL){

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
    MPI_Bcast(dwork1, sep_nrows*v_nrhs, MPI_DOUBLE, root, comm_masterLevel);

  }

  /*
   * 2. Compute W = M_U^{-1}*v; (Refer to the paper)
   */

  // Wg = zg
  if(comm_masterLevel!=MPI_COMM_NULL){

    for(j=0;j<v_nrhs;j++){
      for(i=0;i<sep_mloc;i++) w[j*ldw+Aii_m+i] = dwork1[j*sep_nrows+sep_moffsets[masterLevel_myrank]+i];
    }

  }

  // Compute dwork2 = Aig*dwork1 (Aig*zg)
  //Broadcast the vector to the local group
  if(localGroup_nbprocs>1){
    MPI_Bcast(dwork1, sep_m*v_nrhs, MPI_DOUBLE, local_root, comm_localLevel);
  }

  if(v_nrhs==1) CPLM_MatCSRMatrixVector(Aig, dONE, dwork1, dZERO, dwork2); //faster than using matmult for nrhs = 1
  else CPLM_MatCSRMatrixCSRDenseMult(Aig, dONE, dwork1, v_nrhs, sep_m, dZERO, dwork2, Aig->info.m);

  //gather the result on the master proc
  if(localGroup_nbprocs>1){ //gather on the master proc
    // copy the root block in place
    if(localLevel_myrank==local_root) dlacpy("A", &Aig->info.m, &v_nrhs, dwork2, &Aig->info.m, dwork2, &Aig_m);

    preAlps_multiColumnTypeVectorCreate(v_nrhs, Aig->info.m, Aig_m, &localType, &globalType);

    MPI_Gatherv(localLevel_myrank==local_root?MPI_IN_PLACE:dwork2, Aig->info.m, localType, dwork2, Aig_mcounts, Aig_moffsets, globalType, local_root, comm_localLevel);
  }

  // Compute vi = Aii^{-1}*dwork2  (use vi as buffer)

  if(Aii_sv->type==SOLVER_MUMPS){
    preAlps_solver_triangsolve(Aii_sv, Aii->info.m, Aii->val, Aii->rowPtr, Aii->colInd, v_nrhs, NULL, dwork2);
    if(localLevel_myrank==0) memcpy(vi, dwork2, Aii_m*v_nrhs*sizeof(double));
  }
  else{
    preAlps_solver_triangsolve(Aii_sv, Aii->info.m, Aii->val, Aii->rowPtr, Aii->colInd, v_nrhs, vi, dwork2);
  }

  // w = zi - vi

  if(comm_masterLevel!=MPI_COMM_NULL){

    for(j=0;j<v_nrhs;j++){
      for(i=0;i<Aii_m;i++) w[j*ldw+i] = zi[j*Aii_m+i] - vi[j*Aii_m+i];
    }

    #ifdef DEBUG
      preAlps_doubleVectorSet_printSynchronized(w, Aii_m, v_nrhs, ldw, "Wi", "Wi: Wi computed", comm_masterLevel);
    #endif
  }

  return ierr;
}

/* Free internal allocated workspace */
int preAlps_LorascMatApplyWorkspaceFree(preAlps_Lorasc_t *lorasc){
  int ierr = 0;
  if(lorasc->vi!=NULL)         free(lorasc->vi);
  if(lorasc->zi!=NULL)         free(lorasc->zi);
  if(lorasc->dwork1!=NULL)     free(lorasc->dwork1);
  if(lorasc->dwork2!=NULL)     free(lorasc->dwork2);
  if(lorasc->eigWork)          free(lorasc->eigWork);
  return ierr;
}

/* Allocate workspace if required */
//Always check if the buffer has been allocated before, if so then to nothing.
int preAlps_LorascMatApplyWorkspacePrepare(preAlps_Lorasc_t *lorasc, int Aii_m, int separator_m, int v_nrhs){

  int ierr =0, max_m;

  if(lorasc->vi==NULL){
    if ( !(lorasc->vi  = (double *) malloc((Aii_m*v_nrhs) * sizeof(double))) ) preAlps_abort("Malloc fails for lorasc->vi[].");
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
