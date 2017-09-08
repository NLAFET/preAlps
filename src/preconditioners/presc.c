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
int Presc_alloc(Presc_t **presc){

  *presc = (Presc_t*) malloc(sizeof(Presc_t));

  return (*presc==NULL);
}

/*
 * Build the preconditioner
 * presc:
 *     input: the preconditioner object to construct
 * A:
 *     input: the input matrix on processor 0
 * locAP:
 *     output: the local permuted matrix on each proc after the preconditioner is built
 *
*/
int Presc_build(Presc_t *presc, Mat_CSR_t *A, Mat_CSR_t *locAP, MPI_Comm comm){

  int nbprocs, my_rank, root = 0, ierr = 0;
  int mloc, n ;
  Mat_CSR_t locA = MatCSRNULL(), AP = MatCSRNULL(), locABlockDiag = MatCSRNULL();


  IVector_t idxRowBegin   = IVectorNULL(), idxColBegin = IVectorNULL();
  IVector_t perm   = IVectorNULL();
  IVector_t colPos = IVectorNULL();
  int i;



  Mat_CSR_t Sloc  = MatCSRNULL();
  int Sloc_nrows, *Sloc_rowPtr =  NULL, *Sloc_colInd=NULL;
  double *Sloc_val=NULL;
  int nbDiagRows;
  int *workP, *idxworkP; //a workspace of the size of the number of procs
  int *workColPerm;
  int *locRowPerm;//permutation applied on the local matrix
  Mat_CSR_t locAgg  = MatCSRNULL();
  int locAgg_n;
  int *locAgg_mcounts; /* a workspace of the size of the number of procs */

  Mat_CSR_t Agi = MatCSRNULL(), Aii = MatCSRNULL(), Aig  = MatCSRNULL(), Aloc  = MatCSRNULL();

  preAlps_solver_type_t stype;

  /* Solver to factorize Adiag */
  preAlps_solver_t *ABlockDiagloc_sv;

  /* Solver to factorize Adiag */
  preAlps_solver_t *Sloc_sv, *Aloc_sv, *Aii_sv;


  /*
   * Let's begin
   */


  presc->eigvalues = NULL;
  presc->eigvalues_deflation = 0;


  MPI_Comm_size(comm, &nbprocs);
  MPI_Comm_rank(comm, &my_rank);

  //Workspace of the size of the number of processors
  if ( !(workP  = (int *) malloc((nbprocs+1) * sizeof(int))) ) preAlps_abort("Malloc fails for workP[].");
  if ( !(locAgg_mcounts  = (int *) malloc((nbprocs) * sizeof(int))) ) preAlps_abort("Malloc fails for locAgg_mcounts[].");
  if ( !(idxworkP  = (int *) malloc((nbprocs+1) * sizeof(int))) ) preAlps_abort("Malloc fails for idxworkP[].");



  /*
   * Partition the matrix on proc 0 and distribute (TODO: later use parMETIS)
   */

  if(my_rank==root){

    #ifdef MAT_LFAT5
      /*DEBUG: reproductible permutation */
      ierr = IVectorMalloc(&perm, A->info.m);preAlps_checkError(ierr);
      ierr = IVectorMalloc(&idxRowBegin, nbprocs+1);preAlps_checkError(ierr);

      /*
      //Metis on MAC
      perm.val[0]=8; perm.val[1]=11; perm.val[2]=12; perm.val[3]=13; perm.val[4]=0; perm.val[5]=3; perm.val[6]=4;
      perm.val[7]=7; perm.val[8]=1; perm.val[9]=5; perm.val[10]=9; perm.val[11]=2; perm.val[12]=6; perm.val[13]=10;

      idxRowBegin.val[0]= 0;idxRowBegin.val[1]= 4;idxRowBegin.val[2]= 8;
      idxRowBegin.val[3]= 11; idxRowBegin.val[4]= 14;
      */

      //MEtis on Matlab
      perm.val[0]=1;perm.val[1]=5;perm.val[2]=9;perm.val[3]=2;perm.val[4]=6;perm.val[5]=10;perm.val[6]=0;
      perm.val[7]=3;perm.val[8]=4;perm.val[9]=7;perm.val[10]=8;perm.val[11]=11;perm.val[12]=12;perm.val[13]=13;

      idxRowBegin.val[0]= 0;idxRowBegin.val[1]= 3;idxRowBegin.val[2]= 6;
      idxRowBegin.val[3]= 10; idxRowBegin.val[4]= 14;
      IVectorPrintf("***** ATT: Permutation vector for reproductibility",&perm);
      IVectorPrintf("***** ATT: Row position for reproductibility",&idxRowBegin);
    #elif defined(MAT_CUSTOM_PARTITIONING_FILE)

      char permFile[250], rowPosFile[250];
      sprintf(permFile, "matrix/%s.perm.txt", MAT_CUSTOM_PARTITIONING_FILE);
      sprintf(rowPosFile, "matrix/%s.rowPos.txt", MAT_CUSTOM_PARTITIONING_FILE);

      printf("Loading partititioning details from files: Perm vector:%s, rowPos:%s ... \n", permFile, rowPosFile);
      /* This matrice provide its own permutation vector Get the partitioning externally */
      IVectorLoad(permFile, &perm, 0); //perm.nval
      IVectorLoad(rowPosFile, &idxRowBegin, 0); //idxRowBegin.nval

      //IVectorLoad("matrix/ela12.perm.txt", &perm, A->info.m); //perm.nval
      //IVectorLoad("matrix/ela12.rowPos.txt", &idxRowBegin, nbprocs+1); //idxRowBegin.nval

      /* Convert to zero based indexing */
      for(i=0;i<perm.nval;i++) perm.val[i]-=1;
      for(i=0;i<idxRowBegin.nval;i++) idxRowBegin.val[i]-=1;

      #ifdef DEBUG
        preAlps_permVectorCheck(perm.val, perm.nval);
      #endif

      //IVectorLoad(permFile, &perm, perm.nval);
      //IVectorLoad(rowPosFile, &idxRowBegin, idxRowBegin.nval);
      printf("Loading ... done\n");

      IVectorPrintf("***** ATT: ELA12 matrix Permutation vector for reproductibility",&perm);
      IVectorPrintf("***** ATT: ELA12 matrix Row position for reproductibility",&idxRowBegin);

      //Check the size
      int m_expected = 0;

      for(i=0;i<nbprocs;i++) m_expected+=(idxRowBegin.val[i+1] - idxRowBegin.val[i]);
      if(A->info.m!=m_expected){
        printf("Error: the sum of the rows in the provided partitioning: %d is different to the matrix size:%d\n", m_expected, A->info.m);
        preAlps_abort("Error in the provided partitioning");
      }


    #else
      ierr = metisKwayOrdering(A, &perm, nbprocs, &idxRowBegin);preAlps_checkError(ierr);
      IVectorPrintf("Permutation vector returned by Kway",&perm);
      IVectorPrintf("Row position",&idxRowBegin);
    #endif


//    IVector_t colPos  = IVectorNULL();
//    ierr = MatCSRGetColBlockPos(A,&idxRowBegin,&colPos);preAlps_checkError(ierr);

//    IVectorPrintf("pos of blocks",&idxRowBegin);
//    IVectorPrintf("colPos of blocks",&colPos);


    #ifdef MAT_CUSTOM_PARTITIONING_FILE

      //MatCSRCopyStruct(A, &AP);
      MatCSRCopy(A, &AP);

      MatCSRPrintfInfo("A Info", A);
      MatCSRPrintfInfo("AP info", &AP);
      //MatCSRConvertTo0BasedIndexing(&AP);
    #else
      #if 1
        // AP = P x A x  P^T
	      ierr  = MatCSRPermute(A, &AP, perm.val, perm.val, PERMUTE);preAlps_checkError(ierr);
      #else
        //AP = P x A x  P^T
        MatCSRCopyStruct(A, &AP);
        int *iperm;

        iperm = s_return_pinv(perm.val, perm.nval);
        preAlps_matrix_permute (A->info.m, A->rowPtr, A->colInd, A->val, iperm, perm.val,AP.rowPtr, AP.colInd,AP.val);

        free(iperm);
      #endif
    #endif

//    ierr  = MatCSRPermute(A, &AP, perm.val, perm.val, PERMUTE);preAlps_checkError(ierr);

    MatCSRPrintCoords(&AP, "Permuted matrix from Kway");

    #ifdef BUILDING_MATRICES_DUMP
      printf("Dumping the matrix ...\n");
      MatCSRSave(&AP, "dump_AP.mtx");
      printf("Dumping the matrix ... done\n");
    #endif

  }else{

    /*Allocate memory for the partitioning vector*/
    ierr = IVectorMalloc(&idxRowBegin, nbprocs+1);preAlps_checkError(ierr);
  }




  /*
   *  distribute the matrix using block row data distribution
   */





  /*Broadcast the Block row distribution of the global matrix*/
  MPI_Bcast(idxRowBegin.val, idxRowBegin.nval, MPI_INT, root, comm);

  IVectorPrintSynchronized (&idxRowBegin, comm, "idxRowBegin", "after dist.");


  ierr = MatCSRBlockRowScatterv(&AP, &locA, idxRowBegin.val, root, comm); preAlps_checkError(ierr);


  if(my_rank==root){
    MatCSRFree(&AP);
  }


  MatCSRPrintSynchronizedCoords (&locA, comm, "locA", "Recv locA");

  mloc = locA.info.m; n = locA.info.n;

  //workspace of the size of the number of column of the global matrix

  if ( !(workColPerm  = (int *) malloc(n * sizeof(int))) ) preAlps_abort("Malloc fails for workColPerm[].");
  if ( !(locRowPerm  = (int *) malloc(mloc * sizeof(int))) ) preAlps_abort("Malloc fails for locRowPerm[].");


#ifdef DEBUG
   //Mat_CSR_t Awork1 = MatCSRNULL();
   //MatCSRBlockRowGatherv(&locA, &Awork1, idxRowBegin.val, root, comm);
   //if(my_rank==root) MatCSRPrintCoords(&Awork1, "Assembled matrix for testing");
   //MatCSRFree(&Awork1);
#endif



  /*
   * Permute the off diag rows on each local matrix to the bottom (inplace)
   */



  idxColBegin = idxRowBegin; //The matrix is symmetric

  preAlps_permuteOffDiagRowsToBottom(&locA, idxColBegin.val, &nbDiagRows, workColPerm, comm);

  MatCSRPrintSynchronizedCoords (&locA, comm, "locA", "2.0 locA after permuteOffDiagrows");

  preAlps_int_printSynchronized(nbDiagRows, "nbDiagRows", comm);



  /*
   * Extract the Block diagonal part of locA, locABlockDiag = Block-Diag(locA)
  */

  //IVector_t rowPos = IVectorNULL();
  //ierr = IVectorMalloc(&rowPos, 2); preAlps_checkError(ierr);
  //rowPos.val[0] = 0; rowPos.val[1] = locA.info.m;

  //MatCSRGetSubMatrix(&locA, &locABlockDiag, &rowPos, &idxColBegin,
  //                   0, my_rank, &work);
  //MatCSRGetSubBlock(&locA, &locABlockDiag, &rowPos, &idxColBegin,
  //                                      0, my_rank, &work);

  //MatCSRGetColPanel(&locA,  &locABlockDiag, &rowPos, &idxColBegin, my_rank);


  /* Indexation for block Columns*/
  ierr = MatCSRGetColBlockPos(&locA, &idxColBegin, &colPos);preAlps_checkError(ierr);

  ierr = MatCSRGetColPanel(&locA,  &locABlockDiag, &idxColBegin, &colPos, my_rank); preAlps_checkError(ierr);


  preAlps_int_printSynchronized(locABlockDiag.info.m, "locABlockDiag.info.m", comm);
  MatCSRPrintSynchronizedCoords (&locABlockDiag, comm, "locABlockDiag", "locABlockDiag");

  #ifdef BUILDING_MATRICES_DUMP
    if(my_rank==0) MatCSRSave(&locABlockDiag, "dump/locABlockDiag_P0.mtx");
  #endif

  IVectorFree(&colPos);

  preAlps_int_printSynchronized(nbDiagRows, "nbDiagRows", comm);


  /*
   * Permute the  matrix to form Agg at the end of the matrix
   */

  preAlps_permuteSchurComplementToBottom(&locA, nbDiagRows, idxColBegin.val, workColPerm, &locAgg_n, comm);

  MatCSRPrintSynchronizedCoords (&locA, comm, "locA", "locA after permuteSchurToBottom");


  /*
   * Get (2,2) block
   */

  /* Sum of the element before the selected block */
  //sumDiag = 0;
  //for(i=0;i<numBlock;i++) sumDiag += workP[i];

  //preAlps_int_printSynchronized(sumDiag, "sumDiag", comm);

  #if 0
  /* Partition the matrix into 2 x 2, and get block A_{22}*/
  IVector_t rowPart1 = IVectorNULL();
  IVector_t colPart1 = IVectorNULL();
  IVector_t work1    = IVectorNULL();
  ierr = IVectorMalloc(&rowPart1, 3);preAlps_checkError(ierr);

  rowPart1.val[0] = 0; rowPart1.val[1] = nbDiagRows; rowPart1.val[2] = locA.info.m;

  ierr = IVectorMalloc(&colPart1, 3);preAlps_checkError(ierr);

  colPart1.val[0] = 0; colPart1.val[1] = nbDiagRows; colPart1.val[2] = locA.info.n;


  /* Indexation for block Columns*/
  ierr = MatCSRGetColBlockPos(&locA, &colPart1, &colPos);preAlps_checkError(ierr);

  //  IVectorPrintf("Colpos", &colPos);

  MatCSRGetSubMatrix(&locA, &locAgg, &rowPart1, &colPos, 1, 1, &work1);

  IVectorFree(&rowPart1);
  IVectorFree(&colPart1);
  IVectorFree(&work1);

  #elif 0

    IVector_t rowPart = IVectorNULL();
    //Mat_CSR_t rowPanel = MatCSRNULL();
    ierr = IVectorMalloc(&rowPart, 3);preAlps_checkError(ierr);
    rowPart.val[0] = 0; rowPart.val[1] = nbDiagRows;rowPart.val[2] = mloc;

    if(mloc -  nbDiagRows> 0){
      ierr = MatCSRGetRowPanel(&locA, &rowPanel, &rowPart, 1); preAlps_checkError(ierr);
    }

    MatCSRPrintSynchronizedCoords (&rowPanel, comm, "rowPanel", "rowPanel");

    if(mloc -  nbDiagRows> 0){
      IVector_t colPart = IVectorNULL();
      ierr = IVectorMalloc(&colPart, 3);preAlps_checkError(ierr);
      colPart.val[0] = 0; colPart.val[1] = n - locAgg_n; colPart.val[2] = n;


      /* Indexation for block Columns*/
      ierr = MatCSRGetColBlockPos(&rowPanel, &colPart, &colPos);preAlps_checkError(ierr);

      ierr = MatCSRGetColPanel(&rowPanel,  &locAgg, &colPart, &colPos, 1); preAlps_checkError(ierr);

      MatCSRPrintfInfo("LocAgg", &locAgg);
      MatCSRPrintf("LocAgg", &locAgg);

      IVectorFree(&colPos);
      IVectorFree(&colPart);
    }

    IVectorFree(&rowPart);
    MatCSRFree(&rowPanel);
  #elif 0
    IVector_t rowPart = IVectorNULL();
    IVector_t colPart = IVectorNULL();
    int *works    = NULL;
    size_t lworks  =0 ;
    //Mat_CSR_t rowPanel = MatCSRNULL();
    ierr = IVectorMalloc(&rowPart, 3);preAlps_checkError(ierr);
    rowPart.val[0] = 0; rowPart.val[1] = nbDiagRows;rowPart.val[2] = mloc;

    ierr = IVectorMalloc(&colPart, 3);preAlps_checkError(ierr);
    //colPart.nval = 3;
    colPart.val[0] = 0; colPart.val[1] = n - locAgg_n;colPart.val[2] = n;

    IVectorPrintf("rowPart", &rowPart);
    IVectorPrintf("colPart", &colPart);
    //MatCSRPrintfInfo("A", A);
    MatCSRPrintfInfo("Info local A", &locA);

    ierr = MatCSRGetSubBlock(&locA, &locAgg, &rowPart, &colPart,
      1, 1, &works, &lworks);

    MatCSRPrintfInfo("LocAgg", &locAgg);
    MatCSRPrintf("LocAgg", &locAgg);

    free(works);
  #else

    /* TODO: possible Optimization: Get the Schur complement and put zeros in the Block diagonal directly */

    /*Extract the Schur complement */
    /* Partition the matrix into 2 x 2, and get block A_{22}*/
    int firstBlock_nrows = nbDiagRows, firstBlock_ncols = n - locAgg_n;

    preAlps_schurComplementGet(&locA, firstBlock_nrows, firstBlock_ncols, &locAgg);

  #endif

  preAlps_intVector_printSynchronized(locAgg.rowPtr, locAgg.info.m, "locAgg.rowPtr", "", comm);
  preAlps_intVector_printSynchronized(locAgg.colInd, locAgg.info.nnz, "locAgg.colInd", "", comm);

  MatCSRPrintSynchronizedCoords (&locAgg, comm, "locAgg", "locAgg extracted");


  if(presc->eigs_kind == PRESC_EIGS_SLOC){


    /*
     * Compute the schur complement of the diagonal block.
     *
     * Following the partitioning locABlockDiag =[A11 A12;A21 A22], where A11 is of size nbDiagRows x nbDiagRows
     * Compute the schur complement of the diagonal block. Sloc = A22 - A_{21}A_{11}^{-1}A_{12}
     */

    //Number of rows of the schur complement
    Sloc_nrows = locABlockDiag.info.m - nbDiagRows;

    preAlps_int_printSynchronized(Sloc_nrows, "Sloc_nrows", comm);

    //locABlockDiag_ps.mtype = -2;   /* Real symmetric matrix, pardiso requires only the upper part of the matrix */
    //locABlockDiag_ps.mtype = 2;   /* Real symmetric positive definite matrix, pardiso requires only the upper part of the matrix */
    //locABlockDiag_ps.mtype = 11;
    //locABlockDiag_ps.nrhs = 1;	   /* Number of right hand sides. */


    switch(SCHUR_COMPLEMENT_SOLVER){
      case 0: stype = SOLVER_MKL_PARDISO; break;
      case 1: stype = SOLVER_PARDISO; break;
      case 2: stype = SOLVER_MUMPS; break;
      default: stype = 0;
    }

    preAlps_solver_create(&ABlockDiagloc_sv, stype, MPI_COMM_SELF);
    preAlps_solver_init(ABlockDiagloc_sv);

    /* Real unsymmetric matrix. TODO: need to switch to symmetric matrix */
    //preAlps_solver_setMatrixType(ABlockDiagloc_sv, SOLVER_MATRIX_REAL_NONSYMMETRIC);
    /* Allocate workspace for the schur complement */
    //Swork = (double*) malloc((Sloc_nrows * Sloc_nrows)*sizeof(double));

    //Sloc_nrows = 0; //DEBUG

    if(locABlockDiag.info.m>0) preAlps_solver_partial_factorize(ABlockDiagloc_sv, locABlockDiag.info.m,
                                         locABlockDiag.val, locABlockDiag.rowPtr, locABlockDiag.colInd,
                                         Sloc_nrows, &Sloc_val, &Sloc_rowPtr, &Sloc_colInd);

    preAlps_int_printSynchronized(1, "Synchro after partial_factorization", comm);

    int S_nnz = (Sloc_nrows>0) ? Sloc_rowPtr[Sloc_nrows]: 0;

    //preAlps_intVector_printSynchronized(Sloc_rowPtr, Sloc_nrows+1, "Sloc_rowPtr", "Sloc_rowPtr", comm);
    //preAlps_intVector_printSynchronized(Sloc_colInd, S_nnz, "Sloc_colInd", "Sloc_colInd", comm);

    preAlps_doubleVector_printSynchronized(Sloc_val, S_nnz, "val", "Sloc_val", comm);

    MatCSRSetInfo(&Sloc, Sloc_nrows, Sloc_nrows, S_nnz,
                Sloc_nrows,  Sloc_nrows, S_nnz, 1);
    MatCSRCreateFromPtr(&Sloc, Sloc_rowPtr, Sloc_colInd, Sloc_val);


    #ifdef DEBUG
      if(stype == SOLVER_MKL_PARDISO || stype == SOLVER_PARDISO){
        int *perm1 = preAlps_solver_getPerm(ABlockDiagloc_sv);
        preAlps_intVector_printSynchronized(perm1, locABlockDiag.info.m, "FACT perm", "ABlockDiag FACT perm", comm);
      }
    #endif

    MatCSRPrintSynchronizedCoords (&Sloc, comm, "Sloc", "Sloc");

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

    preAlps_int_printSynchronized(1, "Synchro after the factorization", comm);







    /*
     * Create OffDiag(Agg): Fill block diag of Agg with zeros in the place of Sloc (inplace)
     */

     //Gather the number of elements in the diag for each procs
    MPI_Allgather(&Sloc_nrows, 1, MPI_INT, locAgg_mcounts, 1, MPI_INT, comm);

    idxworkP[0] = 0;
    for(i=0;i<nbprocs;i++) idxworkP[i+1] = idxworkP[i] + locAgg_mcounts[i];
    preAlps_intVector_printSynchronized(locAgg_mcounts, nbprocs, "locAgg_mcounts = all Sloc_nrows", "", comm);
    preAlps_intVector_printSynchronized(idxworkP, nbprocs+1, "idxworkP", "", comm);

    #ifdef BUILDING_MATRICES_DUMP
       char logFile[250];
       sprintf(logFile, "dump/Sloc_p%d.mtx", my_rank);
       MatCSRSave(&Sloc, logFile); //each proc dump its Sloc

       //MatCSRBlockRowGathervDump(&Sloc, "dump_Sloc.mtx", idxworkP, root, comm);
       MatCSRBlockRowGathervDump(&locAgg, "dump_Agg.mtx", idxworkP, root, comm);
    #endif

    //preAlps_sleep(my_rank, my_rank*2);//DEBUG
    /* Remove the block column at the position my_rank (fill with zeros) */
    MatCSRBlockColRemove(&locAgg, locAgg_mcounts, my_rank);

    preAlps_intVector_printSynchronized(locAgg.rowPtr, locAgg.info.m+1, "Agg->rowPtr", "", comm);

    MatCSRPrintSynchronizedCoords (&locAgg, comm, "locAgg", "locAgg after zeros Diag");


    #ifdef BUILDING_MATRICES_DUMP
      MatCSRBlockRowGathervDump(&locAgg, "dump_AggZerosDiag.mtx", idxworkP, root, comm);
    #endif


   /*
    * Solve the eigenvalue problem  SS_{loc}^{-1}u = \lambda u
    *
    * => Solve the eigenvalue problem  (I + OffDiag(A_gg)S_{loc}^{-1}) u = \lambda u
    */

    Presc_eigSolve(presc, locAgg_mcounts, Sloc_sv, &Sloc, &locAgg, comm);

    if(Sloc_nrows>0) preAlps_solver_finalize(Sloc_sv, Sloc_nrows, Sloc.rowPtr, Sloc.colInd);

    preAlps_solver_destroy(&Sloc_sv);

    MatCSRFree(&Sloc);

    preAlps_solver_destroy(&ABlockDiagloc_sv);


  }else{

    /* Extract blocks of the Matrix */
    preAlps_int_printSynchronized(2, "Synchro before extracting AblockDiag blocks", comm);

    /* Partition the matrix into 2 x 2, and get block A_{22}*/
    IVector_t rowPart = IVectorNULL();
    IVector_t colPart = IVectorNULL();
    IVector_t work    = IVectorNULL();
    ierr = IVectorMalloc(&rowPart, 3);preAlps_checkError(ierr);

    rowPart.val[0] = 0; rowPart.val[1] = nbDiagRows; rowPart.val[2] = locABlockDiag.info.m;

    ierr = IVectorMalloc(&colPart, 3);preAlps_checkError(ierr);

    colPart.val[0] = 0; colPart.val[1] = nbDiagRows; colPart.val[2] = locABlockDiag.info.n;
  /*
    IVectorPrintf("rowPart", &rowPart);
    IVectorPrintf("colPart", &colPart);
    MatCSRPrintfInfo("A", A);
    MatCSRPrintfInfo("Info local A", &locA);
  */

    /* Indexation for block Columns*/
    ierr = MatCSRGetColBlockPos(&locABlockDiag, &colPart, &colPos);preAlps_checkError(ierr);

  //  IVectorPrintf("Colpos", &colPos);

    MatCSRGetSubMatrix(&locABlockDiag, &Aii, &rowPart, &colPos, 0, 0, &work);

#if DEBUG
    for(i=0;i<nbprocs;i++){
      if(my_rank==i){
        MatCSRPrintfInfo("Aii", &Aii);
      }
      MPI_Barrier(comm);
    }
#endif

    MatCSRPrintSynchronizedCoords (&Aii, comm, "Aii", "Aii extracted");

    preAlps_int_printSynchronized(3, "Synchro before extracting AblockDiag blocks", comm);

    if(locABlockDiag.info.m>nbDiagRows) MatCSRGetSubMatrix(&locABlockDiag, &Agi, &rowPart, &colPos, 1, 0, &work);

    preAlps_int_printSynchronized(4, "Synchro before extracting AblockDiag blocks", comm);
    #if DEBUG
        for(i=0;i<nbprocs;i++){
          if(my_rank==i){
            MatCSRPrintfInfo("Agi", &Agi);
          }
          MPI_Barrier(comm);
        }
    #endif



    MatCSRPrintSynchronizedCoords (&Agi, comm, "Agi", "Agi extracted");


    preAlps_int_printSynchronized(5, "Synchro before extracting AblockDiag blocks", comm);


    if(locABlockDiag.info.n>nbDiagRows) MatCSRGetSubMatrix(&locABlockDiag, &Aig, &rowPart, &colPos, 0, 1, &work);
    if(locABlockDiag.info.m>nbDiagRows && locABlockDiag.info.n>nbDiagRows) MatCSRGetSubMatrix(&locABlockDiag, &Aloc, &rowPart, &colPos, 1, 1, &work);

    //MatCSRPrintfInfo("LocAgg", &locAgg);

    #if DEBUG
        for(i=0;i<nbprocs;i++){
          if(my_rank==i){
            MatCSRPrintfInfo("Aig", &Aig);
          }
          MPI_Barrier(comm);
        }
    #endif

    #ifdef BUILDING_MATRICES_DUMP
       char logFile1[250];
       sprintf(logFile1, "dump/Aig_p%d.mtx", my_rank);
       MatCSRSave(&Aig, logFile1); //each proc dump its Sloc
    #endif

    MatCSRPrintSynchronizedCoords (&Aig, comm, "Aig", "Aig extracted");
    MatCSRPrintSynchronizedCoords (&Aloc,comm, "Aloc", "Aloc extracted");

    IVectorFree(&rowPart);
    IVectorFree(&colPart);
    IVectorFree(&work);

    //preAlps_abort("dbg ");

    /*
     * Factorize Aii and Aloc
     */


    switch(SPARSE_SOLVER){
      case 0: stype = SOLVER_MKL_PARDISO; break;
      case 1: stype = SOLVER_PARDISO; break;
      case 2: stype = SOLVER_MUMPS; break;
      default: stype = 0;
    }


    /*Aii*/
    preAlps_solver_create(&Aii_sv, stype, MPI_COMM_SELF);
    preAlps_solver_init(Aii_sv);
    preAlps_solver_setMatrixType(Aii_sv, SOLVER_MATRIX_REAL_NONSYMMETRIC);
    preAlps_solver_factorize(Aii_sv, Aii.info.m, Aii.val, Aii.rowPtr, Aii.colInd);

    /*Aloc*/
    preAlps_solver_create(&Aloc_sv, stype, MPI_COMM_SELF);
    preAlps_solver_init(Aloc_sv);
    preAlps_solver_setMatrixType(Aloc_sv, SOLVER_MATRIX_REAL_NONSYMMETRIC);
    preAlps_solver_factorize(Aloc_sv, Aloc.info.m, Aloc.val, Aloc.rowPtr, Aloc.colInd);



    #ifdef BUILDING_MATRICES_DUMP
       char logFile2[250];
       sprintf(logFile2, "dump/Aloc_p%d.mtx", my_rank);
       MatCSRSave(&Aloc, logFile2); //each proc dump its Sloc
    #endif


    preAlps_int_printSynchronized(2, "Synchro after the factorization", comm);




    Presc_eigSolve_SAloc(presc, locAgg.info.m, &locAgg, &Agi, &Aii, &Aig, &Aloc
                                           ,Aii_sv, Aloc_sv, comm);

    preAlps_int_printSynchronized(1, "finalize free", comm);
    ///preAlps_solver_finalize(Aii_sv, Aii.info.m, Aii.rowPtr, Aii.colInd);
    ///preAlps_solver_finalize(Aloc_sv, Aloc.info.m, Aloc.rowPtr, Aloc.colInd);
    preAlps_int_printSynchronized(2, "finalize free", comm);
    preAlps_solver_destroy(&Aii_sv);
    preAlps_solver_destroy(&Aloc_sv);

    preAlps_int_printSynchronized(3, "finalize free", comm);
    MatCSRFree(&Aii);
    MatCSRFree(&Agi);
    MatCSRFree(&Aig);
    MatCSRFree(&Aloc);

  }

  /*
   * Free memory
  */

  preAlps_int_printSynchronized(1, "free memory", comm);

  ///if(locABlockDiag.info.m>0) preAlps_solver_finalize(ABlockDiagloc_sv, locABlockDiag.info.m, locABlockDiag.rowPtr, locABlockDiag.colInd);

  //pardiso_solver_finalize(&locABlockDiag_ps, Sloc_nrows, Sloc_rowPtr, Sloc_colInd);
  preAlps_int_printSynchronized(2, "free memory", comm);


  preAlps_int_printSynchronized(3, "free memory", comm);
  if(my_rank==root){
    IVectorFree(&perm);
  }

  preAlps_int_printSynchronized(4, "free memory", comm);
  IVectorFree(&colPos);
  IVectorFree(&idxRowBegin);
  preAlps_int_printSynchronized(5, "free memory", comm);

  free(workColPerm);
  free(locAgg_mcounts);
  free(workP);
  free(idxworkP);
  preAlps_int_printSynchronized(1, "presc_build end", comm);

  return 0;
}




/*Destroy the preconditioner*/
int Presc_destroy(Presc_t **presc){

  if((*presc)->eigvalues!=NULL) free((*presc)->eigvalues);
  free(*presc);
  return 0;
}