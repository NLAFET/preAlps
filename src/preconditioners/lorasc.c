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
 *
*/
int preAlps_LorascBuild(preAlps_Lorasc_t *lorasc, CPLM_Mat_CSR_t *A, CPLM_Mat_CSR_t *locAP, MPI_Comm comm){

  int ierr = 0, nbprocs, my_rank, root = 0;
  int *perm = NULL, m, n, nnz, mloc;
  int nparts, *partCount, *partBegin;
  CPLM_Mat_CSR_t AP = CPLM_MatCSRNULL();
  CPLM_Mat_CSR_t Aii = CPLM_MatCSRNULL(), Aig = CPLM_MatCSRNULL(), Agi = CPLM_MatCSRNULL(), Agg = CPLM_MatCSRNULL(), locAgg = CPLM_MatCSRNULL();
  int i;
  int *Agg_mcounts, *Agg_moffsets, Agg_m, Agg_n, Agg_nnz;

  preAlps_solver_type_t stype;
  preAlps_solver_t *Aii_sv, *Agg_sv = NULL; //Solver to factorize Aii and Agg


  /*
   * Let's begin
   */

  lorasc->eigvalues = NULL;
  lorasc->eigvalues_deflation = 0;

  MPI_Comm_size(comm, &nbprocs);
  MPI_Comm_rank(comm, &my_rank);

  /* Broadcast the global matrix dimension from the root to the other procs */
  preAlps_matrixDim_Bcast(comm, A, root, &m, &n, &nnz);

  /* Allocate memory for the permutation array */
  if ( !(perm  = (int *) malloc(m*sizeof(int))) ) preAlps_abort("Malloc fails for perm[].");


  /*
   * Create a block arrow structure of the matrix
   */

  preAlps_blockArrowStructCreate(comm, m, A, &AP, perm, &nparts, &partCount, &partBegin);


  preAlps_intVector_printSynchronized(partCount, nparts, "partCount", "parcount after separator sort", comm);
  preAlps_intVector_printSynchronized(partBegin, nparts+1, "partBegin", "parbegin after separator sort", comm);


  /* Check if each proc has at least one block */
  if(nbprocs!=nparts-1){
    preAlps_abort("This number of process is not support yet. Please use a multiple of 2. nbprocs:%d, nparts created:%d\n", nbprocs, nparts-1);
  }

  /* Distribute the n-1 first blocks of the matrix to all procs */
  ierr = CPLM_MatCSRBlockRowScatterv(&AP, locAP, partBegin, root, comm); preAlps_checkError(ierr);

  CPLM_MatCSRPrintSynchronizedCoords (locAP, comm, "locAP", "RowScatterv locAP");
  mloc = locAP->info.m;

  preAlps_int_printSynchronized(mloc, "mloc", comm);




  /*
   * Extract the blocks Aii, Aig and Agi for each procs
   */

  CPLM_IVector_t rowPart = CPLM_IVectorNULL();
  CPLM_IVector_t colPart = CPLM_IVectorNULL();

  ierr = CPLM_IVectorMalloc(&rowPart, 2);preAlps_checkError(ierr);
  rowPart.val[0] = 0; rowPart.val[1] = mloc;

  CPLM_IVectorCreateFromPtr(&colPart, nparts+1, partBegin);

  int *work1 = NULL;
  size_t work1Size = 0;
  ierr = CPLM_MatCSRGetSubBlock (locAP, &Aii, &rowPart, &colPart,
                                  0, my_rank, &work1, &work1Size); preAlps_checkError(ierr);

  CPLM_MatCSRPrintSynchronizedCoords (&Aii, comm, "Aii", "Aii extracted");
  /*
  if(my_rank==0) {
    CPLM_MatCSRPrintfInfo("Info Aii", &Aii);
    CPLM_MatCSRPrintf2D("Aii", &Aii);
    CPLM_MatCSRPrintf("Aii", &Aii);
  }
  */

  ierr = CPLM_MatCSRGetSubBlock (locAP, &Aig, &rowPart, &colPart,
                                  0, nparts-1, &work1, &work1Size); preAlps_checkError(ierr);

  Aig.info.nnz = Aig.info.lnnz; //tmp bug fix in CPLM_MatCSRGetSubBlock();

#ifdef DEBUG
  for(int ip=0;ip<nbprocs;ip++){
    if(my_rank==ip){
      printf("[%d] ", ip);
      CPLM_MatCSRPrintfInfo("Info Aig", &Aig);
      printf("nnz:%d\n", Aig.info.nnz);
      if(Aig.info.lnnz>0)
      {
        CPLM_MatCSRPrintf2D("Aig", &Aig);
      }
      else printf("[]\n");
    }
    MPI_Barrier(comm);
  }
  //CPLM_MatCSRPrintSynchronizedCoords (&Aig, comm, "Aig", "Aig extracted");
#endif

  if(work1!=NULL) free(work1);
  CPLM_IVectorFree(&rowPart);

  /* Get the matrix Agi:  Transpose Aig to get Agi */

  CPLM_MatCSRTranspose(&Aig, &Agi);

#ifdef DEBUG
  for(int ip=0;ip<nbprocs;ip++){
    if(my_rank==ip){
      printf("[%d] ", ip);
      CPLM_MatCSRPrintfInfo("Info Agi", &Agi);
      printf("nnz:%d\n", Agi.info.nnz);
      if(Agi.info.lnnz>0)
      {
        CPLM_MatCSRPrintf2D("Agi", &Agi);
      }
      else printf("[]\n");
    }
    MPI_Barrier(comm);
  }
  //CPLM_MatCSRPrintSynchronizedCoords (&Aig, comm, "Aig", "Aig extracted");
#endif


  /*
   * Extract the schur complement matrix Agg
   */

  if(my_rank==root){
    /* Partition the matrix into 2 x 2, and get block A_{22}*/
    int firstBlock_nrows = partBegin[nparts-1], firstBlock_ncols = partBegin[nparts-1];
    preAlps_schurComplementGet(&AP, firstBlock_nrows, firstBlock_ncols, &Agg);

    CPLM_MatCSRPrintSynchronizedCoords (&Agg, MPI_COMM_SELF, "Agg", "Agg extracted");

#ifdef DEBUG

    CPLM_MatCSRPrintfInfo("Info Agg", &Agg);
    CPLM_MatCSRPrintf2D("Agg", &Agg);
    CPLM_MatCSRPrintf("Agg", &Agg);
#endif
  }

  /* Split and distribute Agg */
  if ( !(Agg_mcounts  = (int *) malloc((nbprocs) * sizeof(int))) ) preAlps_abort("Malloc fails for mcounts[].");
  if ( !(Agg_moffsets  = (int *) malloc((nbprocs+1) * sizeof(int))) ) preAlps_abort("Malloc fails for moffsets[].");

  /* Broadcast the global matrix dimension from the root to the other procs */
  preAlps_matrixDim_Bcast(comm, &Agg, root, &Agg_m, &Agg_n, &Agg_nnz);

  /* Split the number of rows of Agg among the processors */
  for(i=0;i<nbprocs;i++){
    preAlps_nsplit(Agg_m, nbprocs, i, &Agg_mcounts[i], &Agg_moffsets[i]);
  }
  Agg_moffsets[nbprocs] = Agg_m;

  preAlps_intVector_printSynchronized(Agg_mcounts, nbprocs, "Agg_mcounts", "Agg_mcounts", comm);

  /* Distribute the matrix Agg to all procs */
  ierr = CPLM_MatCSRBlockRowScatterv(&Agg, &locAgg, Agg_moffsets, root, comm); preAlps_checkError(ierr);
  CPLM_MatCSRPrintSynchronizedCoords (&locAgg, comm, "locAgg", "locAgg distributed");


  /*
   * Factorize Aii and Agg
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
  preAlps_solver_factorize(Aii_sv, Aii.info.m, Aii.val, Aii.rowPtr, Aii.colInd);

#if 1
  /* Factorize Agg */
#if DEBUG_AGG_SOLVE
  printf("[DEBUG] using SOLVER_MKL_PARDISO to factorize Agg for debugging purpose\n");
  if(my_rank==0){
   stype = SOLVER_MKL_PARDISO;
   preAlps_solver_create(&Agg_sv, stype, comm);
   preAlps_solver_init(Agg_sv);
   preAlps_solver_setMatrixType(Agg_sv, SOLVER_MATRIX_REAL_NONSYMMETRIC);
   preAlps_solver_factorize(Agg_sv, Agg.info.m, Agg.val, Agg.rowPtr, Agg.colInd);
  }
#else
  stype = SOLVER_MUMPS; //only MUMPS is supported for the moment
  preAlps_solver_create(&Agg_sv, stype, comm);
  preAlps_solver_init(Agg_sv);
  preAlps_solver_setMatrixType(Agg_sv, SOLVER_MATRIX_REAL_NONSYMMETRIC);
  preAlps_solver_factorize(Agg_sv, Agg.info.m, Agg.val, Agg.rowPtr, Agg.colInd);
#endif
#endif


  /*
   * Solve the eigenvalue problem
   */

   preAlps_Lorasc_eigSolve(lorasc, comm, mloc, &locAgg, &Agi, &Aii, &Aig, &Agg, Aii_sv, Agg_sv);


  /* Free memory*/
  CPLM_MatCSRFree(&Aii);
  CPLM_MatCSRFree(&Aig);
  CPLM_MatCSRFree(&Agi);
  if(my_rank==root) CPLM_MatCSRFree(&Agg);
  CPLM_MatCSRFree(&locAgg);

  free(partCount);
  free(partBegin);
  free(perm);
  CPLM_MatCSRFree(&AP);
  return ierr;
}

/*Destroy the preconditioner*/
int preAlps_LorascDestroy(preAlps_Lorasc_t **lorasc){

  if((*lorasc)->eigvalues!=NULL) free((*lorasc)->eigvalues);
  if(*lorasc!=NULL) free(*lorasc);

  return 0;
}
