/*
============================================================================
Name        : presc.c
Author      : Simplice Donfack
Version     : 0.1
Description : Preconditioner based on Schur complement
Date        : Mai 15, 2017
============================================================================
*/
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <metis_interface.h>
#include "preAlps_utils.h"
#include "presc.h"
#include "presc_eigsolve.h"
#include "preAlps_solver.h"

#ifdef USE_SOLVER_MKL_PARDISO
#include "mkl_pardiso_solver.h"
#endif
#ifdef USE_SOLVER_PARDISO
#include "pardiso_solver.h"
#endif
#ifdef USE_SOLVER_MUMPS
#include "mumps_solver.h"
#endif


//#define MAT_CUSTOM_PARTITIONING_FILE "ela12"




/*Allocate workspace for the preconditioner*/
int preAlps_PrescAlloc(preAlps_Presc_t **presc){

  *presc = (preAlps_Presc_t*) malloc(sizeof(preAlps_Presc_t));

  if(*presc!=NULL){
    (*presc)->eigvalues=NULL;
    (*presc)->eigvectors=NULL;

    //default parameters
    (*presc)->deflation_tol = 1e-2;
    (*presc)->nrhs = 1;

    //matrices
    (*presc)->Aii=NULL;
    (*presc)->Aig=NULL;
    (*presc)->Agi=NULL;
    (*presc)->Aloc=NULL;
    (*presc)->locAgg=NULL;


  }
  return (*presc==NULL);
}


/*
 * Build the preconditioner
 * presc:
 *     input/output: the preconditioner object to construct
 * locAP:
 *     input: the local permuted matrix on each proc after the partitioning
 * partBegin:
 *    input: the global array to indicate the row partitioning
 * locNbDiagRows:
      input: the number of row in the diagonal as returned by preAlps_blockDiagODBStructCreate();
*/
int preAlps_PrescBuild(preAlps_Presc_t *presc, CPLM_Mat_CSR_t *locAP, int *partBegin, int locNbDiagRows, MPI_Comm comm){

  int nbprocs, my_rank, ierr = 0;
  int i,  n;

  CPLM_Mat_CSR_t locABlockDiag = CPLM_MatCSRNULL(), locAP2 = CPLM_MatCSRNULL();

  CPLM_IVector_t idxColBegin = CPLM_IVectorNULL();

  CPLM_IVector_t colPos = CPLM_IVectorNULL();
  CPLM_Mat_CSR_t Sloc  = CPLM_MatCSRNULL();
  int Sloc_nrows, *Sloc_rowPtr =  NULL, *Sloc_colInd=NULL;
  double *Sloc_val=NULL;

  int *workP, *idxworkP; //a workspace of the size of the number of procs
  int *workColPerm;
  int *locRowPerm;//permutation applied on the local matrix


  CPLM_Mat_CSR_t *Aii = NULL, *Aig =NULL, *Agi=NULL, *Aloc=NULL;

  CPLM_Mat_CSR_t *locAgg=NULL;
  int locAgg_n;
  int *locAgg_mcounts; // a workspace of the size of the number of procs

  preAlps_solver_type_t stype;
  preAlps_solver_t *ABlockDiagloc_sv; //Solver to factorize Adiag
  preAlps_solver_t *Sloc_sv, *Aloc_sv, *Aii_sv; //Solver to factorize Adiag


  /*
   * Let's begin
   */

  presc->eigvalues_deflation = 0;
  presc->comm = comm;

  //Let me know who I am
  MPI_Comm_size(comm, &nbprocs);
  MPI_Comm_rank(comm, &my_rank);

  // Create matrix objects
  CPLM_MatCSRCreateNULL(&Aii);
  CPLM_MatCSRCreateNULL(&Aig);
  CPLM_MatCSRCreateNULL(&Agi);
  CPLM_MatCSRCreateNULL(&Aloc);
  CPLM_MatCSRCreateNULL(&locAgg);

  // Broadcast the global matrix dimension from the root to the other procs
  //preAlps_matrixDim_Bcast(comm, A, root, &m, &n, &nnz);
  n = locAP->info.n;

  //workspace of the size of the number of column of the global matrix
  if ( !(workColPerm  = (int *) malloc(n * sizeof(int))) ) preAlps_abort("Malloc fails for workColPerm[].");


  //Workspace of the size of the number of processors
  if ( !(workP  = (int *) malloc((nbprocs+1) * sizeof(int))) ) preAlps_abort("Malloc fails for workP[].");
  if ( !(locAgg_mcounts  = (int *) malloc((nbprocs) * sizeof(int))) ) preAlps_abort("Malloc fails for locAgg_mcounts[].");
  if ( !(idxworkP  = (int *) malloc((nbprocs+1) * sizeof(int))) ) preAlps_abort("Malloc fails for idxworkP[].");


  if ( !(locRowPerm  = (int *) malloc(locAP->info.m * sizeof(int))) ) preAlps_abort("Malloc fails for locRowPerm[].");

  /*
   * Extract the Block diagonal part of locA, locABlockDiag = Block-Diag(locA)
  */

  CPLM_IVectorCreateFromPtr(&idxColBegin, nbprocs+1, partBegin); //The matrix is symmetric


  /* Block columns indexation*/
  ierr = CPLM_MatCSRGetColBlockPos(locAP, &idxColBegin, &colPos);preAlps_checkError(ierr);

  ierr = CPLM_MatCSRGetColPanel(locAP,  &locABlockDiag, &idxColBegin, &colPos, my_rank); preAlps_checkError(ierr);


  preAlps_int_printSynchronized(locABlockDiag.info.m, "locABlockDiag.info.m", comm);
  CPLM_MatCSRPrintSynchronizedCoords (&locABlockDiag, comm, "locABlockDiag", "locABlockDiag");

  #ifdef BUILDING_MATRICES_DUMP
    if(my_rank==0) CPLM_MatCSRSave(&locABlockDiag, "dump/locABlockDiag_P0.mtx");
  #endif

  CPLM_IVectorFree(&colPos);

  /*
   * Permute the  matrix to form Agg at the end of the matrix
   * (order all part of each row block without offDiag first, followed by all the part with offDiag elements)
   */

  preAlps_permuteSchurComplementToBottom(locAP, locNbDiagRows, idxColBegin.val, &locAP2, workColPerm, &locAgg_n, comm);

  CPLM_MatCSRPrintSynchronizedCoords (&locAP2, comm, "locAP2", "locAP2 after permuteSchurToBottom");


  /*
   * Get the (2,2) block
   */

  /* Partition the matrix into 2 x 2, and get block A_{22}*/
  int firstBlock_nrows = locNbDiagRows, firstBlock_ncols = n - locAgg_n;

  preAlps_schurComplementGet(&locAP2, firstBlock_nrows, firstBlock_ncols, locAgg);

  preAlps_intVector_printSynchronized(locAgg->rowPtr, locAgg->info.m, "locAgg.rowPtr", "", comm);
  preAlps_intVector_printSynchronized(locAgg->colInd, locAgg->info.nnz, "locAgg.colInd", "", comm);

  CPLM_MatCSRPrintSynchronizedCoords (locAgg, comm, "locAgg", "locAgg extracted");

  CPLM_MatCSRFree(&locAP2);


  /* Extract blocks of the Matrix */

  preAlps_PrescSubMatricesExtract(&locABlockDiag, locNbDiagRows, Agi, Aii, Aig, Aloc);

  CPLM_MatCSRPrintSynchronizedCoords (Aii, comm, "Aii", "Aii extracted");
  CPLM_MatCSRPrintSynchronizedCoords (Agi, comm, "Agi", "Agi extracted");
  CPLM_MatCSRPrintSynchronizedCoords (Aig, comm, "Aig", "Aig extracted");
  CPLM_MatCSRPrintSynchronizedCoords (Aloc,comm, "Aloc", "Aloc extracted");
  #ifdef BUILDING_MATRICES_DUMP
     char logFile1[250];
     sprintf(logFile1, "dump/Aig_p%d.mtx", my_rank);
     CPLM_MatCSRSave(Aig, logFile1); //each proc dump its Sloc
  #endif


  if(presc->eigs_kind == PRESC_EIGS_SSLOC){

    /*
     * Compute the schur complement of the diagonal block.
     *
     * Following the partitioning locABlockDiag =[A11 A12;A21 A22], where A11 is of size locNbDiagRows x locNbDiagRows
     * Compute the schur complement of the diagonal block. Sloc = A22 - A_{21}A_{11}^{-1}A_{12}
     */

    //Number of rows of the schur complement
    Sloc_nrows = locABlockDiag.info.m - locNbDiagRows;

    preAlps_int_printSynchronized(Sloc_nrows, "Sloc_nrows", comm);

    switch(SCHUR_COMPLEMENT_SOLVER){
      case 0: stype = SOLVER_MKL_PARDISO; break;
      case 1: stype = SOLVER_PARDISO; break;
      case 2: stype = SOLVER_MUMPS; break;
      default: stype = 0;
    }



    printf("[%d] Slow_nrows:%d, locABlockDiag.info.m:%d\n", my_rank, Sloc_nrows, locABlockDiag.info.m);

    preAlps_solver_create(&ABlockDiagloc_sv, stype, MPI_COMM_SELF);
    /* Real unsymmetric matrix. TODO: need to switch to symmetric matrix */
    preAlps_solver_setMatrixType(ABlockDiagloc_sv, SOLVER_MATRIX_REAL_NONSYMMETRIC);
    //set the global problem infos (required only for parallel solver)
    if(stype == SOLVER_MUMPS) preAlps_solver_setGlobalMatrixParam(ABlockDiagloc_sv, Sloc_nrows, 1, 0);

    preAlps_solver_init(ABlockDiagloc_sv);


    /* Allocate workspace for the schur complement */
    //Swork = (double*) malloc((Sloc_nrows * Sloc_nrows)*sizeof(double));

    //Sloc_nrows = 0; //DEBUG

    if(locABlockDiag.info.m>0) preAlps_solver_partial_factorize(ABlockDiagloc_sv, locABlockDiag.info.m,
                                         locABlockDiag.val, locABlockDiag.rowPtr, locABlockDiag.colInd,
                                         Sloc_nrows, &Sloc_val, &Sloc_rowPtr, &Sloc_colInd);


    int S_nnz = (Sloc_nrows>0) ? Sloc_rowPtr[Sloc_nrows]: 0;

    //preAlps_intVector_printSynchronized(Sloc_rowPtr, Sloc_nrows+1, "Sloc_rowPtr", "Sloc_rowPtr", comm);
    //preAlps_intVector_printSynchronized(Sloc_colInd, S_nnz, "Sloc_colInd", "Sloc_colInd", comm);

    preAlps_doubleVector_printSynchronized(Sloc_val, S_nnz, "val", "Sloc_val", comm);

    CPLM_MatCSRSetInfo(&Sloc, Sloc_nrows, Sloc_nrows, S_nnz,
                Sloc_nrows,  Sloc_nrows, S_nnz, 1);
    CPLM_MatCSRCreateFromPtr(&Sloc, Sloc_rowPtr, Sloc_colInd, Sloc_val);

    #ifdef DEBUG
      if(stype == SOLVER_MKL_PARDISO || stype == SOLVER_PARDISO){
        int *perm1 = preAlps_solver_getPerm(ABlockDiagloc_sv);
        preAlps_intVector_printSynchronized(perm1, locABlockDiag.info.m, "FACT perm", "ABlockDiag FACT perm", comm);
      }
    #endif

    CPLM_MatCSRPrintSynchronizedCoords (&Sloc, comm, "Sloc", "Sloc");

    /*
     * Factorize Sloc
    */

    switch(SPARSE_SOLVER){
      case 0: stype = SOLVER_MKL_PARDISO; break;
      case 1: stype = SOLVER_PARDISO; break;
      case 2: stype = SOLVER_MUMPS; break;
      default: stype = 0;
    }

    preAlps_solver_create(&Sloc_sv, stype, MPI_COMM_SELF);
    preAlps_solver_init(Sloc_sv);
    preAlps_solver_setMatrixType(Sloc_sv, SOLVER_MATRIX_REAL_NONSYMMETRIC);
    if(Sloc_nrows>0) preAlps_solver_factorize(Sloc_sv, Sloc_nrows, Sloc.val, Sloc.rowPtr, Sloc.colInd);

    #ifdef DEBUG
      if(stype == SOLVER_MKL_PARDISO || stype == SOLVER_PARDISO){
        int *perm2 = preAlps_solver_getPerm(Sloc_sv);
        preAlps_intVector_printSynchronized(perm2, Sloc_nrows, "FACT perm", "Sloc FACT perm", comm);
      }
    #endif

    /*
     * Create OffDiag(Agg): Fill block diag of Agg with zeros in the place of Sloc (inplace)
     */

     //Gather the number of elements in the diag for each procs
    MPI_Allgather(&Sloc_nrows, 1, MPI_INT, locAgg_mcounts, 1, MPI_INT, comm);

    idxworkP[0] = 0;
    for(i=0;i<nbprocs;i++) idxworkP[i+1] = idxworkP[i] + locAgg_mcounts[i];

    #ifdef BUILDING_MATRICES_DUMP
       char logFile[250];
       sprintf(logFile, "dump/Sloc_p%d.mtx", my_rank);
       CPLM_MatCSRSave(&Sloc, logFile); //each proc dump its Sloc

       //CPLM_MatCSRBlockRowGathervDump(&Sloc, "dump_Sloc.mtx", idxworkP, root, comm);
       CPLM_MatCSRBlockRowGathervDump(locAgg, "dump_Agg.mtx", idxworkP, root, comm);
    #endif

    /* Remove the block column at the position my_rank (fill with zeros) */
    CPLM_MatCSRBlockColRemove(locAgg, locAgg_mcounts, my_rank);

    preAlps_intVector_printSynchronized(locAgg->rowPtr, locAgg->info.m+1, "Agg->rowPtr", "", comm);

    CPLM_MatCSRPrintSynchronizedCoords (locAgg, comm, "locAgg", "locAgg after zeros Diag");


    #ifdef BUILDING_MATRICES_DUMP
      CPLM_MatCSRBlockRowGathervDump(locAgg, "dump_AggZerosDiag.mtx", idxworkP, root, comm);
    #endif


   /*
    * Solve the eigenvalue problem  SS_{loc}^{-1}u = \lambda u
    *
    * => Solve the eigenvalue problem  (I + OffDiag(A_gg)S_{loc}^{-1}) u = \lambda u
    */

    Presc_eigSolve_SSloc(presc, comm, locAgg->info.m, Sloc_sv, &Sloc, locAgg);

    if(Sloc_nrows>0) preAlps_solver_finalize(Sloc_sv, Sloc_nrows, Sloc.rowPtr, Sloc.colInd);

    preAlps_solver_destroy(&Sloc_sv);

    CPLM_MatCSRFree(&Sloc);

    preAlps_solver_destroy(&ABlockDiagloc_sv);

  }else{


    /*
     * Factorize Aii and Aloc
     */


    switch(SPARSE_SOLVER){
      case 0: stype = SOLVER_MKL_PARDISO; break;
      case 1: stype = SOLVER_PARDISO; break;
      case 2: stype = SOLVER_MUMPS; break;
      default: stype = 0;
    }


    /* Factorize Aii */
    preAlps_solver_create(&Aii_sv, stype, MPI_COMM_SELF);
    preAlps_solver_init(Aii_sv);
    preAlps_solver_setMatrixType(Aii_sv, SOLVER_MATRIX_REAL_NONSYMMETRIC);
    preAlps_solver_factorize(Aii_sv, Aii->info.m, Aii->val, Aii->rowPtr, Aii->colInd);

    /* Factorize Aloc */
    preAlps_solver_create(&Aloc_sv, stype, MPI_COMM_SELF);
    preAlps_solver_init(Aloc_sv);
    preAlps_solver_setMatrixType(Aloc_sv, SOLVER_MATRIX_REAL_NONSYMMETRIC);
    preAlps_solver_factorize(Aloc_sv, Aloc->info.m, Aloc->val, Aloc->rowPtr, Aloc->colInd);

    #ifdef BUILDING_MATRICES_DUMP
       char logFile2[250];
       sprintf(logFile2, "dump/Aloc_p%d.mtx", my_rank);
       CPLM_MatCSRSave(&Aloc, logFile2); //each proc dump its Sloc
    #endif

    /*
     * Solve the eigenvalue problem  Sloc*u = \lambda*Aloc*u
     *
     */

    Presc_eigSolve_SAloc(presc, comm, locAgg->info.m, locAgg,
                         Agi, Aii, Aig, Aloc ,Aii_sv, Aloc_sv);

    if(Aii->info.m>0)  preAlps_solver_finalize(Aii_sv, Aii->info.m, Aii->rowPtr, Aii->colInd);
    if(Aloc->info.m>0) preAlps_solver_finalize(Aloc_sv, Aloc->info.m, Aloc->rowPtr, Aloc->colInd);

    preAlps_solver_destroy(&Aii_sv);
    preAlps_solver_destroy(&Aloc_sv);

  }

  /*
   * Save infos for the application of the preconditioner
   */

   //Matrices
   presc->Aii          = Aii;
   presc->Aig          = Aig;
   presc->Agi          = Agi;
   presc->Aloc         = Aloc;
   presc->locAgg       = locAgg;

  /*
   * Free memory
  */

  CPLM_IVectorFree(&colPos);
  //CPLM_IVectorFree(&idxRowBegin);


  free(workColPerm);
  free(locAgg_mcounts);
  free(workP);
  free(idxworkP);

  return 0;
}




/*Destroy the preconditioner*/
int preAlps_PrescDestroy(preAlps_Presc_t **presc){

  preAlps_Presc_t *prescA = *presc;
  if(!prescA) return 0;

  if(prescA->eigvalues!=NULL)  free(prescA->eigvalues);
  if(prescA->eigvectors!=NULL) free(prescA->eigvectors);

  //free matrix objects
  if(prescA->Aii)    CPLM_MatCSRFree(prescA->Aii);
  if(prescA->Aig)    CPLM_MatCSRFree(prescA->Aig);
  if(prescA->Agi)    CPLM_MatCSRFree(prescA->Agi);
  if(prescA->Aloc)   CPLM_MatCSRFree(prescA->Aloc);
  if(prescA->locAgg) CPLM_MatCSRFree(prescA->locAgg);

  free(*presc);
  return 0;
}


/*
 * Apply Presc preconditioner on a dense matrice
 * i.e Compute  W = M_{presc}^{-1} * V
 */

int preAlps_PrescMatApply(preAlps_Presc_t *presc, CPLM_Mat_Dense_t *V, CPLM_Mat_Dense_t *W){
  int ierr = 0, nbprocs, my_rank, root = 0;
  double *v, *vgloc, *w;
  int i, j, ldv, v_nrhs, ldw;
  int sep_mloc;
  double dMONE = -1.0, dONE=1.0, dZERO = 0.0;


  // Let's begin

  if(!presc) preAlps_abort("Please first use PrescBuild() to construct the preconditioner!");

  // Let me know who I am
  MPI_Comm comm  = presc->comm;
  MPI_Comm_size(comm, &nbprocs);
  MPI_Comm_rank(comm, &my_rank);

  // Retrieve  parameters from presc
  CPLM_Mat_CSR_t *Aii      = presc->Aii;
  CPLM_Mat_CSR_t *Aig      = presc->Aig;
  CPLM_Mat_CSR_t *Agi      = presc->Agi;
  CPLM_Mat_CSR_t *Aloc     = presc->Aloc;
  CPLM_Mat_CSR_t *locAgg   = presc->locAgg;
  preAlps_solver_t *Aii_sv = presc->Aii_sv;
  preAlps_solver_t *Agg_sv = presc->Agg_sv;
  int *sep_mcounts         = presc->sep_mcounts;
  int *sep_moffsets        = presc->sep_moffsets;
  int sep_nrows            = presc->sep_nrows;

  // Get the vectorized notations of the matrices
  v = V->val; ldv = V->info.lda; v_nrhs = V->info.n;
  w = W->val; ldw = W->info.lda;
  int Aii_m = Aii->info.m;

#ifdef DEBUG
  printf("[%d] M:%d, N:%d, m:%d, n:%d, lda:%d, nval:%d\n", my_rank, V->info.M, V->info.N, V->info.m, V->info.n, V->info.lda, V->info.nval);
#endif

  printf("[%d]PreAlps dbg 1, Aii_m:%d\n", my_rank, Aii_m);

  //Prepare the reusable workspace
  preAlps_PrescMatApplyWorkspacePrepare(presc, Aii_m, sep_nrows, v_nrhs);

  //Get the buffers (after workspacePrepare())
  double *vi      = presc->vi;
  double *zi      = presc->zi;
  double *dwork1  = presc->dwork1;
  double *dwork2  = presc->dwork2;
  double *eigWork = presc->eigWork;

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
  int E_r = presc->eigvalues_deflation;

  if(my_rank==root){
    // Compute eigWork = E^T * dwork1
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
           E_r, v_nrhs, sep_nrows, 1.0, presc->eigvectors, sep_nrows, dwork1, sep_nrows, 0.0, eigWork, E_r);
  }

  // Compute dwork1 = Agg^{-1}*dwork1 (overwrites dwork1)
  //the rhs is already centralized on the host, just solve the system
  preAlps_solver_triangsolve(Agg_sv, locAgg->info.m, locAgg->val, locAgg->rowPtr, locAgg->colInd, v_nrhs, NULL, dwork1);


  if(my_rank==root){

    // Finalize the computation y = Agg^{-1}*dwork1 + E * sigma * E^T * dwork1 on the root = dwork1 + E * sigma *eigWork
    // eigwork = sigma * eigwork
    for(j=0;j<v_nrhs;j++){
      for (i = 0; i < E_r; i++) {
        eigWork[j*E_r+i] = eigWork[j*E_r+i]*presc->sigma[i];
      }
    }

    // v = v + E * eigwork
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
           sep_nrows, v_nrhs, E_r, 1.0, presc->eigvectors, sep_nrows, eigWork, E_r, 1.0, dwork1, sep_nrows);
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
int preAlps_PrescMatApplyWorkspaceFree(preAlps_Presc_t *presc){
  int ierr = 0;
  if(presc->vi!=NULL)         free(presc->vi);
  if(presc->zi!=NULL)         free(presc->zi);
  if(presc->dwork1!=NULL)     free(presc->dwork1);
  if(presc->dwork2!=NULL)     free(presc->dwork2);
  if(presc->eigWork!=NULL)    free(presc->eigWork);
  return ierr;
}

/* Allocate workspace if required */
//Always check if the buffer has been allocated before, if so then to nothing.
int preAlps_PrescMatApplyWorkspacePrepare(preAlps_Presc_t *presc, int Aii_m, int separator_m, int v_nrhs){

  int ierr =0, max_m;

  if(presc->vi==NULL){
    if ( !(presc->vi  = (double *) malloc((Aii_m*v_nrhs) * sizeof(double))) ) preAlps_abort("Malloc fails for presc->zi[].");
  }

  if(presc->zi==NULL){
    if ( !(presc->zi  = (double *) malloc((Aii_m*v_nrhs) * sizeof(double))) ) preAlps_abort("Malloc fails for presc->zi[].");
  }

  // Determine the max workspace so we can use it for any operation
  max_m = Aii_m>separator_m ? Aii_m:separator_m;

  if(presc->dwork1==NULL){
    if ( !(presc->dwork1  = (double *) malloc((max_m*v_nrhs) * sizeof(double))) ) preAlps_abort("Malloc fails for presc->dwork1[].");
  }

  if(presc->dwork2==NULL){
    if ( !(presc->dwork2  = (double *) malloc((max_m*v_nrhs) * sizeof(double))) ) preAlps_abort("Malloc fails for presc->dwork2[].");
  }

  if(presc->eigWork==NULL){ //a buffer with the same as the number of eigenvalues computed
    if ( !(presc->eigWork  = (double *) malloc((presc->eigvalues_deflation*v_nrhs) * sizeof(double))) ) preAlps_abort("Malloc fails for presc->eigWork[].");
  }

  return ierr;
}

/* Extract the local matrices Agi, Aii, Aig, Aloc from the matrix A*/
int preAlps_PrescSubMatricesExtract(CPLM_Mat_CSR_t *locA, int locNbDiagRows,
  CPLM_Mat_CSR_t *Agi, CPLM_Mat_CSR_t *Aii, CPLM_Mat_CSR_t *Aig, CPLM_Mat_CSR_t *Aloc){

  int ierr = 0;
  CPLM_IVector_t colPos = CPLM_IVectorNULL();

  /* Partition the matrix into 2 x 2, and get block A_{22}*/

  CPLM_IVector_t rowPart = CPLM_IVectorNULL();
  CPLM_IVector_t colPart = CPLM_IVectorNULL();
  CPLM_IVector_t work    = CPLM_IVectorNULL();

  ierr = CPLM_IVectorMalloc(&rowPart, 3);preAlps_checkError(ierr);
  rowPart.val[0] = 0; rowPart.val[1] = locNbDiagRows; rowPart.val[2] = locA->info.m;

  ierr = CPLM_IVectorMalloc(&colPart, 3);preAlps_checkError(ierr);
  colPart.val[0] = 0; colPart.val[1] = locNbDiagRows; colPart.val[2] = locA->info.n;

  /* Indexation for block Columns*/
  ierr = CPLM_MatCSRGetColBlockPos(locA, &colPart, &colPos);preAlps_checkError(ierr);

  CPLM_MatCSRGetSubMatrix(locA, Aii, &rowPart, &colPos, 0, 0, &work);

  if(locA->info.m>locNbDiagRows) CPLM_MatCSRGetSubMatrix(locA, Agi, &rowPart, &colPos, 1, 0, &work);

  if(locA->info.n>locNbDiagRows) CPLM_MatCSRGetSubMatrix(locA, Aig, &rowPart, &colPos, 0, 1, &work);

  if(locA->info.m>locNbDiagRows && locA->info.n>locNbDiagRows) CPLM_MatCSRGetSubMatrix(locA, Aloc, &rowPart, &colPos, 1, 1, &work);

  CPLM_IVectorFree(&rowPart);
  CPLM_IVectorFree(&colPart);
  CPLM_IVectorFree(&work);

  CPLM_IVectorFree(&colPos);

  return ierr;
}
