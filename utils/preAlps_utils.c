/*
============================================================================
Name        : preAlps_utils.c
Author      : Simplice Donfack
Version     : 0.1
Description : Utils for preAlps
Date        : Mai 15, 2017
============================================================================
*/
#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdarg.h>

#include <cplm_iarray.h>

#include <cplm_v0_ivector.h>
#include "cplm_utils.h"
#include "cplm_matcsr.h"
#include "preAlps_intvector.h"
#include "preAlps_doublevector.h"
#include "preAlps_utils.h"

/*
 * utils
 */



/* Display a message and stop the execution of the program */
void preAlps_abort(char *s, ... ){
  char buff[1024];

  va_list arglist;

  va_start(arglist, s);

  vsprintf(buff, s, arglist);

  va_end(arglist);

  printf("===================\n");
  printf("%s\n", buff);
  printf("Aborting ...\n");
  printf("===================\n");
  exit(1);
}

/*
 * From an array, set one when the node number is a leave.
*/
void preAlps_binaryTreeIsLeaves(int nparts, int *isLeave){

  int i, twoPowerLevel = nparts+1;
  for(i=0;i<nparts;i++) isLeave[i] = 0;

  /* Call the recursive interface */
  preAlps_binaryTreeIsNodeAtLevel(2, twoPowerLevel, nparts - 1, isLeave);
}


/*
 * From an array, set one when the node number is a node at the target Level.
 * The array should be initialized with zeros before calling this routine
*/
void preAlps_binaryTreeIsNodeAtLevel(int targetLevel, int twoPowerLevel, int part_root, int *isNodeLevel){


    if(twoPowerLevel == targetLevel){

      /*We have reached a target node level */

      isNodeLevel[part_root] = 1;

    }else if(twoPowerLevel>targetLevel){

      if(twoPowerLevel>2){

        twoPowerLevel = (int) twoPowerLevel/2;
        /*right*/
        preAlps_binaryTreeIsNodeAtLevel(targetLevel, twoPowerLevel , part_root - 1, isNodeLevel);

        /*left*/
        preAlps_binaryTreeIsNodeAtLevel(targetLevel, twoPowerLevel, part_root - twoPowerLevel, isNodeLevel);
      }

    }
}

/*
 * Create a block arrow structure from a matrix A
 * comm:
 *     input: the communicator for all the processors calling the routine
 * m:
 *     input: the number of rows of the global matrix
 * A:
 *     input: the input matrix
 * AP:
 *     output: the matrix permuted into a block arrow structure (relevant only on proc 0)
 * perm:
 *     output: the permutation vector
 * nbparts:
 *     output: the number of the partition created
 * partCount:
 *     output: the number of rows in each part
 * partBegin:
 *     output: the begining rows of each part.
 */

int preAlps_blockArrowStructCreate(MPI_Comm comm, int m, CPLM_Mat_CSR_t *A, CPLM_Mat_CSR_t *AP, int *perm, int *nbparts, int **partCount, int **partBegin){


  int ierr = 0, nbprocs, my_rank, root = 0;

  int i, j, count = 0, mloc;
  int *order = NULL, *sizes = NULL, *partCountWork=NULL, *partBeginWork=NULL;

  int *partwork = NULL;
  int *mcounts, *moffsets;
  int *mwork = NULL; //vector array of size m
  int nparts, level;

  CPLM_Mat_CSR_t locAP = CPLM_MatCSRNULL();

  MPI_Comm_size(comm, &nbprocs);
  MPI_Comm_rank(comm, &my_rank);


  // Allocate Workspace
  if ( !(mcounts  = (int *) malloc((nbprocs) * sizeof(int))) ) preAlps_abort("Malloc fails for mcounts[].");
  if ( !(moffsets  = (int *) malloc((nbprocs+1) * sizeof(int))) ) preAlps_abort("Malloc fails for moffsets[].");

  // Split the number of rows among the processors
  for(i=0;i<nbprocs;i++){
    CPLM_nsplit(m, nbprocs, i, &mcounts[i], &moffsets[i]);
  }
  moffsets[nbprocs] = m;


  // Remove the diagonal in order to call ParMetis. ParMetis crashes if the diagonal is kept inplace. (use AP as workspace)
  if(my_rank==root) ierr = CPLM_MatCSRDelDiag(A, AP);preAlps_checkError(ierr);

  // Distribute the matrix to all procs
  ierr = CPLM_MatCSRBlockRowScatterv(AP, &locAP, moffsets, root, comm); preAlps_checkError(ierr);


  mloc = locAP.info.m;

  /*
   * Partition the matrix
   */

  nparts = 1;
  level = 0;
  while(nparts<nbprocs){
    nparts = 2*nparts+1;
    level++;
  }

  if ( !(order = (int *)  malloc((mloc*sizeof(int)))) ) preAlps_abort("Malloc fails for order[].");///mloc
  if ( !(sizes = (int *)  malloc((nparts*sizeof(int)))) ) preAlps_abort("Malloc fails for sizes[]."); //2*Ptilde

  CPLM_IArray_setValue(sizes, nparts, -1);

#ifdef MAT_CUSTOM_PARTITIONING
  /* Custom partitioning */
  int *permWork, perm_len, *sizesWork, sizes_len;

  if ( !(permWork  = (int *) malloc(m*sizeof(int))) ) preAlps_abort("Malloc fails for permWork[].");
  if ( !(sizes = (int *)  malloc((nparts*sizeof(int)))) ) preAlps_abort("Malloc fails for sizes[].");

  if(my_rank==0) printf("**** Reading partitioning data from Files\n");


  preAlps_intVector_load("dump/partition_perm.txt", &permWork, &perm_len);
  if(perm_len!=m) {
   preAlps_abort("Error in the number of rows. Nrows from file:%d, matrix nrows:%d\n", perm_len, m);
  }
  memcpy(perm, permWork, perm_len*sizeof(int));


  preAlps_intVector_load("dump/partition_sizes.txt", &sizesWork, &sizes_len);
  if(sizes_len!=nparts) {
    preAlps_abort("Error in the number of partitions. Nparts from file:%d, matrix nparts:%d\n", sizes_len, nparts);
  }
  memcpy(sizes, sizesWork, sizes_len*sizeof(int));


  free(permWork);
  free(sizesWork);

#else

  CPLM_MatCSROrderingND(comm, &locAP, moffsets, order, sizes);

  preAlps_intVector_printSynchronized(order, mloc, "order", "Order after ordering", comm);

  // Gather the ordering infos from all to all
  ierr = MPI_Allgatherv(order, mloc, MPI_INT, perm, mcounts, moffsets, MPI_INT, comm); preAlps_checkError(ierr);
#endif


  preAlps_intVector_printSynchronized(perm, m, "perm", "Permutation vector after ordering", comm);

  if ( !(mwork  = (int *) malloc(m*sizeof(int))) ) preAlps_abort("Malloc fails for mwork[].");
  if ( !(partwork = (int *)  malloc((nparts*sizeof(int)))) ) preAlps_abort("Malloc fails for partwork[].");
  if ( !(partCountWork = (int *)  malloc((nparts*sizeof(int)))) ) preAlps_abort("Malloc fails for partCountWork[].");
  if ( !(partBeginWork = (int *)  malloc((nparts+1)*sizeof(int))) ) preAlps_abort("Malloc fails for partBeginWork[].");

  // Permute the array order returned by ParMetis such as children are followed by their parent node (put each separator close to its children)
  preAlps_NodeNDPostOrder(nparts, sizes, partCountWork);

  // Compute the begining of each part
  partBeginWork[0] = 0;
  for(i=0;i<nparts;i++) partBeginWork[i+1] = partBeginWork[i] + partCountWork[i];


  // Get the permutation vector

  preAlps_pinv_outplace (perm, m, mwork);

  preAlps_intVector_printSynchronized(mwork, m, "mwork", "Permutation vector after ordering", comm);

  // Determine the leaves from the partitioning
  preAlps_binaryTreeIsLeaves(nparts, partwork);


  // Update the permutation vector in order to permute the separators at the end of the permutated matrix

  count = 0; int nbLeaves = 0;
  for(i=0;i<nparts;i++) partCountWork[i] = 0;

  for(i=0;i<nparts;i++) {

    if(partwork[i]==1) { //is a leave
      for(j=partBeginWork[i];j<partBeginWork[i+1];j++){
        perm[count++] = mwork[j];
      }
      /* update partCountWork */
      partCountWork[nbLeaves] = partBeginWork[i+1] - partBeginWork[i];
      nbLeaves++;
    }
  }

  partCountWork[nbLeaves] = 0;
  for(i=0;i<nparts;i++) {

    if(partwork[i]==0) { //is a separator
      for(j=partBeginWork[i];j<partBeginWork[i+1];j++){
        perm[count++] = mwork[j];
      }
      /* update partCountWork */
      partCountWork[nbLeaves] += partBeginWork[i+1] - partBeginWork[i]; //keep the separator at the end
    }
  }

  // Update the begining position of each partition
  partBeginWork[0] = 0;
  for(i=0;i<nparts;i++) partBeginWork[i+1] = partBeginWork[i] + partCountWork[i];

  // Permute the input matrix to move the separator to the end

  if(my_rank==0) {
    ierr  = CPLM_MatCSRPermute(A, AP, perm, perm, PERMUTE);preAlps_checkError(ierr);
    #ifdef SAVE_PERM
      preAlps_intVector_save(perm, m, "dump/perm.out.txt", "Permutation vector after NodeND");
      preAlps_intVector_save(sizes, nparts, "dump/sizes.out.txt", "Sizes vector after NodeND");
    #endif
  }


  // free memory
  free(mwork);
  if(partwork) free(partwork);
  if(sizes) free(sizes);
  free(mcounts);
  free(moffsets);

  // Set the output
  *nbparts = nbLeaves+1; //the number of nodes + the separator
  *partCount = partCountWork;
  *partBegin = partBeginWork;

  return ierr;
}


/*
 * Distribute the matrix which has a block Arrow structure to the processors.
 */

int preAlps_blockArrowStructDistribute(MPI_Comm comm, int m, CPLM_Mat_CSR_t *AP, int *perm, int nparts, int *partCount, int *partBegin,
  CPLM_Mat_CSR_t *locAP, int *newPerm, CPLM_Mat_CSR_t *Aii, CPLM_Mat_CSR_t *Aig, CPLM_Mat_CSR_t *Agi, CPLM_Mat_CSR_t *locAgg, int *sep_mcounts, int *sep_moffsets){

  int ierr = 0;

  int my_rank, nbprocs, root = 0;

  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &nbprocs);

  /*
   * Distribute the nparts-1 first blocks of the matrix to all procs
   */

  ierr = CPLM_MatCSRBlockRowScatterv(AP, locAP, partBegin, root, comm); preAlps_checkError(ierr);


  /*
   * Extract the block Aii, Agi and create Aig = Agi^T
   */

  //Aii
  CPLM_MatCSRBlockColumnExtract(locAP, nparts, partBegin, my_rank, Aii);
  //Aig
  CPLM_MatCSRBlockColumnExtract(locAP, nparts, partBegin, nparts-1, Aig);
  // Get the matrix Agi:  Transpose Aig to get Agi
  CPLM_MatCSRTranspose(Aig, Agi);


  /*
   * Permute and distribute the separator close to each procs
   */

  preAlps_blockArrowStructSeparatorDistribute(comm, m, AP, perm, nparts, partCount, partBegin, locAP, newPerm, locAgg, sep_mcounts, sep_moffsets);

  return ierr;
}


/* Distribute the separator to each proc and permute the matrix such as their are contiguous in memory */
int preAlps_blockArrowStructSeparatorDistribute(MPI_Comm comm, int m, CPLM_Mat_CSR_t *AP, int *perm, int nparts, int *partCount, int *partBegin,
  CPLM_Mat_CSR_t *locAP, int *newPerm, CPLM_Mat_CSR_t *locAgg, int *sep_mcounts, int *sep_moffsets){

  int ierr=0, my_rank, nbprocs, root = 0;
  int sep_nrows;
  int *mwork, *permIdentity, i, j, count;
  CPLM_Mat_CSR_t sepAP = CPLM_MatCSRNULL(), sepAPloc = CPLM_MatCSRNULL(), Awork = CPLM_MatCSRNULL();


  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &nbprocs);

  if(my_rank==root){
    // get the separator block (last block rows)
    CPLM_IVector_t rowPart = CPLM_IVectorNULL();
    CPLM_IVectorCreateFromPtr(&rowPart, nparts+1, partBegin);
    CPLM_MatCSRGetRowPanel(AP, &sepAP, &rowPart, nparts-1);
  }

  // Workspace
  if ( !(mwork        = (int *) malloc(m * sizeof(int))) ) preAlps_abort("Malloc fails for mwork[].");
  if ( !(permIdentity = (int *) malloc(m * sizeof(int))) ) preAlps_abort("Malloc fails for permIdentity[].");

  // Get the number of rows of the separator
  sep_nrows = partCount[nparts-1];

  // Split the separator among all the processors
  for(i=0;i<nbprocs;i++){
    CPLM_nsplit(sep_nrows, nbprocs, i, &sep_mcounts[i], &sep_moffsets[i]);
  }
  sep_moffsets[nbprocs] = sep_nrows;

  //Distribute the separator from the root to each procs
  ierr = CPLM_MatCSRBlockRowScatterv(&sepAP, &sepAPloc, sep_moffsets, root, comm); preAlps_checkError(ierr);

  //Extract the local part of the schur complement
  CPLM_MatCSRBlockColumnExtract(&sepAPloc, nparts, partBegin, nparts-1, locAgg);

  //merge each block received with the local matrix
  CPLM_MatCSRRowsMerge(locAP, &sepAPloc, &Awork);

  // Compute the global permutation vector which bring the rows of each process close to their initial block
  count = 0;
  for(i=0;i<nbprocs;i++){
    // The initial rows of the processor after the partitioning
    for(j=0;j<partCount[i];j++){
      mwork[count++] = partBegin[i]+j;
    }

    // The rows obtained from the separator
    for(j=0;j<sep_mcounts[i];j++){
      mwork[count++] = partBegin[nparts-1]+sep_moffsets[i]+j;
    }

  }

  // Update the global permutation vector
  preAlps_intVector_permute(mwork, perm, newPerm, m);

  // Update the number of rows for each procs
  for(i=0;i<nbprocs;i++) partCount[i] += sep_mcounts[i];

  // Update the begining positiong of each partition
  partBegin[0] = 0;
  for(i=0;i<nparts;i++) partBegin[i+1] = partBegin[i] + partCount[i];


  //Apply the same permutation on the columns of the matrix to preserve the symmetry

  for(i=0;i<m;i++) permIdentity[i] = i;
  ierr  = CPLM_MatCSRPermute(&Awork, locAP, permIdentity, mwork, PERMUTE);preAlps_checkError(ierr);


  free(permIdentity);
  free(mwork);
  CPLM_MatCSRFree(&Awork);
  CPLM_MatCSRFree(&sepAPloc);
  if(my_rank==root) CPLM_MatCSRFree(&sepAP);

  return ierr;
}

/*
 *
 * First permute the matrix using kway partitioning
 * Permute each block row such as any row with zeros outside the diagonal move
 * to the bottom on the matrix (ODB)
 *
 * comm:
 *     input: the communicator for all the processors calling the routine
 * A:
 *     input: the input matrix
 * locA:
 *     output: the matrix permuted into a block arrow structure on each procs
 * perm:
 *     output: the permutation vector
 * partBegin:
 *     output: the begining rows of each part.
 * nbDiagRows:
 *     output: the number of rows in the diag of each Row block
*/
int preAlps_blockDiagODBStructCreate(MPI_Comm comm, CPLM_Mat_CSR_t *A, CPLM_Mat_CSR_t *locA, int *perm, int **partBegin, int *nbDiagRows){

  int i, ierr=0, my_rank, nbprocs, root = 0;
  CPLM_IVector_t idxRowBegin   = CPLM_IVectorNULL(), idxColBegin   = CPLM_IVectorNULL();
  CPLM_IVector_t Iperm   = CPLM_IVectorNULL();
  CPLM_Mat_CSR_t AP = CPLM_MatCSRNULL();
  int *workColPerm;
  int n;

  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &nbprocs);


  /*
   * Partition the matrix on proc 0 and distribute (TODO: use parMETIS)
   */

  if(my_rank==root){

    #ifdef MAT_LFAT5 /* The smallest SPD matrix on matrix-market for debugging purpose */
      /*DEBUG: reproductible permutation */

      ierr = CPLM_IVectorMalloc(&idxRowBegin, nbprocs+1);preAlps_checkError(ierr);

      //MEtis on Matlab
      perm[0]=1;perm[1]=5;perm[2]=9;perm[3]=2;perm[4]=6;perm[5]=10;perm[6]=0;
      perm[7]=3;perm[8]=4;perm[9]=7;perm[10]=8;perm[11]=11;perm[12]=12;perm[13]=13;

      idxRowBegin.val[0]= 0;idxRowBegin.val[1]= 3;idxRowBegin.val[2]= 6;
      idxRowBegin.val[3]= 10; idxRowBegin.val[4]= 14;
      //CPLM_IVectorPrintf("***** ATT: Permutation vector for reproductibility",&perm);
      //CPLM_IVectorPrintf("***** ATT: Row position for reproductibility",&idxRowBegin);
    #elif defined(MAT_CUSTOM_PARTITIONING_FILE) /* The custom already permuted matrix and the corresponding permutation vector */

      char permFile[250], rowPosFile[250];
      sprintf(permFile, "matrix/%s.perm.txt", MAT_CUSTOM_PARTITIONING_FILE);
      sprintf(rowPosFile, "matrix/%s.rowPos.txt", MAT_CUSTOM_PARTITIONING_FILE);

      printf("Loading partititioning details from files: Perm vector:%s, rowPos:%s ... \n", permFile, rowPosFile);

      /* This matrice provides its own permutation vector */
      CPLM_IVectorLoad(permFile, &Iperm, 0); //perm.nval
      CPLM_IVectorLoad(rowPosFile, &idxRowBegin, 0); //idxRowBegin.nval

      //CPLM_IVectorLoad("matrix/ela12.perm.txt", &perm, A->info.m); //perm.nval
      //CPLM_IVectorLoad("matrix/ela12.rowPos.txt", &idxRowBegin, nbprocs+1); //idxRowBegin.nval

      /* Copy and Convert to zero based indexing */
      for(i=0;i<Iperm.nval;i++) perm[i]= Iperm[i] - 1;
      for(i=0;i<idxRowBegin.nval;i++) idxRowBegin.val[i]-=1;

      #ifdef DEBUG
        preAlps_permVectorCheck(perm, perm);
      #endif

      printf("Loading ... done\n");

      //CPLM_IVectorPrintf("***** ATT: CUSTOM matrix,  Permutation vector for reproductibility",&perm);
      //CPLM_IVectorPrintf("***** ATT: CUSTOM matrix, Row position for reproductibility",&idxRowBegin);

      //Check the size
      int m_expected = 0;

      for(i=0;i<nbprocs;i++) m_expected+=(idxRowBegin.val[i+1] - idxRowBegin.val[i]);
      if(A->info.m!=m_expected){
        preAlps_abort("Error: the sum of the rows in the provided partitioning: %d is different to the matrix size:%d\n", m_expected, A->info.m);
      }

      CPLM_IVectorFree(&Iperm);
    #else

      /* Use metis to partition the matrix */
      ierr = CPLM_metisKwayOrdering(A, &Iperm, nbprocs, &idxRowBegin);preAlps_checkError(ierr);
      //CPLM_IVectorPrintf("Permutation vector returned by Kway",&perm);
      //CPLM_IVectorPrintf("Row position",&idxRowBegin);
      for(i=0;i<Iperm.nval;i++) perm[i]= Iperm.val[i];

      CPLM_IVectorFree(&Iperm);
    #endif


    #ifdef MAT_CUSTOM_PARTITIONING_FILE

      CPLM_MatCSRCopy(A, &AP);

      CPLM_MatCSRPrintfInfo("A Info", A);
      CPLM_MatCSRPrintfInfo("AP info", &AP);

    #else
      ierr  = CPLM_MatCSRPermute(A, &AP, perm, perm, PERMUTE);preAlps_checkError(ierr);
    #endif

    CPLM_MatCSRPrintCoords(&AP, "Permuted matrix from Kway");

    #ifdef BUILDING_MATRICES_DUMP
      printf("Dumping the matrix ...\n");
      CPLM_MatCSRSave(&AP, "dump_AP.mtx");
      printf("Dumping the matrix ... done\n");
    #endif

  }else{

    /*Allocate memory for the partitioning vector*/
    ierr = CPLM_IVectorMalloc(&idxRowBegin, nbprocs+1);preAlps_checkError(ierr);
  }

  /*
   *  distribute the matrix using block row data distribution
   */

  /*Broadcast the Block row distribution of the global matrix*/
  MPI_Bcast(idxRowBegin.val, idxRowBegin.nval, MPI_INT, root, comm);


  preAlps_intVector_printSynchronized(idxRowBegin.val, idxRowBegin.nval, "idxRowBegin", "after dist.", comm);

  ierr = CPLM_MatCSRBlockRowScatterv(&AP, locA, idxRowBegin.val, root, comm); preAlps_checkError(ierr);


  if(my_rank==root){
    CPLM_MatCSRFree(&AP);
  }


  CPLM_MatCSRPrintSynchronizedCoords (locA, comm, "locA", "Recv locA");

  n = locA->info.n;

  //workspace of the size of the number of column of the global matrix

  if ( !(workColPerm  = (int *) malloc(n * sizeof(int))) ) preAlps_abort("Malloc fails for workColPerm[].");

  /*
   * Permute the off diag rows on each local matrix to the bottom (inplace)
   */

  idxColBegin = idxRowBegin; //The matrix is symmetric

  preAlps_permuteOffDiagRowsToBottom(locA, idxColBegin.val, nbDiagRows, workColPerm, comm);

  CPLM_MatCSRPrintSynchronizedCoords (locA, comm, "locA", "2.0 locA after permuteOffDiagrows");

  preAlps_int_printSynchronized(*nbDiagRows, "nbDiagRows", comm);


  *partBegin = idxRowBegin.val;

  free(workColPerm);

  return ierr;
}


/*
 * Check errors
 * No need to call this function directly, use preAlps_checkError() instead.
*/
void preAlps_checkError_srcLine(int err, int line, char *src){
  if(err){
      char str[250];
      sprintf(str, "Error %d, line %d in file %s", err, line, src);
      preAlps_abort(str);
  }
}

/* Create a multilevel communicator by spliting the communicator on two groups based on the number of processors provided*/
int preAlps_comm2LevelsSplit(MPI_Comm comm, int npLevel1, MPI_Comm *commMultilevel){

  int ierr = 0, nbprocs, my_rank, npLevel2, masterLevelMark, localLevelMark;
  MPI_Comm comm_masterLevel = MPI_COMM_NULL, comm_localLevel = MPI_COMM_NULL;

  //Get some infos about the communicator
  MPI_Comm_size(comm, &nbprocs);
  MPI_Comm_rank(comm, &my_rank);

  // Check args
  if((npLevel1<=0) || (npLevel1>nbprocs)) npLevel1 = nbprocs;

  // Number of processors within each blocks
  npLevel2 = nbprocs / npLevel1;
  //printf("npLevel1:%d, npLevel2:%d\n", npLevel1, npLevel2);

  // Create a communicator with only the master of each groups of procs
  masterLevelMark = my_rank%npLevel2;
  MPI_Comm_split(comm, masterLevelMark==0?0:MPI_UNDEFINED, my_rank, &comm_masterLevel);

  // Create a communicator with only the process of each local groups
  localLevelMark  = my_rank / npLevel2;
  MPI_Comm_split(comm, localLevelMark, my_rank, &comm_localLevel);

  //Save the communicator
  commMultilevel[0] = comm;
  commMultilevel[1] = comm_masterLevel;
  commMultilevel[2] = comm_localLevel;

  return ierr;
}

/* Display statistiques min, max and avg of a double*/
void preAlps_dstats_display(MPI_Comm comm, double d, char *str){

  int my_rank, nbprocs;
  int root = 0;
  double dMin, dMax, dSum;

  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &nbprocs);

  MPI_Reduce(&d, &dMin, 1, MPI_DOUBLE, MPI_MIN, root, comm);
  MPI_Reduce(&d, &dMax, 1, MPI_DOUBLE, MPI_MAX, root, comm);
  MPI_Reduce(&d, &dSum, 1, MPI_DOUBLE, MPI_SUM, root, comm);

  if(my_rank==0){
	  printf("%s:  min: %.2f , max: %.2f , avg: %.2f\n", str, dMin, dMax, (double) dSum/nbprocs);
  }
}


/* MPI custom function to sum the column of a matrix using MPI_REDUCE */
void preAlps_DtColumnSum(void *invec, void *inoutvec, int *len, MPI_Datatype *dtype)
{
    int i;
    double *invec_d = (double*) invec;
    double *inoutvec_d = (double*) inoutvec;

    for ( i=0; i<*len; i++ )
        inoutvec_d[i] += invec_d[i];
}

/*
 * Get the extension of a filename
 */
const char *preAlps_get_filename_extension(const char *filename) {
    const char *ext = strrchr(filename, '.');
    if(!ext) return filename;
    return ext + 1;
}

/*
 * Each processor print the value of type int that it has
 * Work only in debug (-DDEBUG) mode
 * a:
 *    The variable to print
 * s:
 *   The string to display before the variable
 */
void preAlps_int_printSynchronized(int a, char *s, MPI_Comm comm){
#ifdef DEBUG
   int i;
   int TAG_PRINT = 4;
   MPI_Status status;

   int b, my_rank, nbprocs;

   MPI_Comm_rank(comm, &my_rank);
   MPI_Comm_size(comm, &nbprocs);

   if(my_rank ==0){

     printf("[%d] %s: %d\n", my_rank, s, a);

     for(i = 1; i < nbprocs; i++) {

       MPI_Recv(&b, 1, MPI_INT, i, TAG_PRINT, comm, &status);
       printf("[%d] %s: %d\n", i, s, b);

     }
   }
   else{

     MPI_Send(&a, 1, MPI_INT, 0, TAG_PRINT, comm);
   }

   MPI_Barrier(comm);
#endif
}

/*Sort the row index of a CSR matrix*/
void preAlps_matrix_colIndex_sort(int m, int *xa, int *asub, double *a){

  int i,j,k;

  int *asub_ptr, row_nnz, itmp;
  double *a_ptr, dtmp;

  for (k=0; k<m; k++){

    asub_ptr = &asub[xa[k]];
    a_ptr = &a[xa[k]];
    row_nnz = xa[k+1] - xa[k];
    for(i=0;i<row_nnz;i++){
      for(j=0;j<i;j++){
        if(asub_ptr[i]<asub_ptr[j]){
          /*swap column index*/
            itmp=asub_ptr[i];
            asub_ptr[i]=asub_ptr[j];
            asub_ptr[j]=itmp;

          /*swap values */
            dtmp=a_ptr[i];
            a_ptr[i]=a_ptr[j];
            a_ptr[j]=dtmp;
        }
      }
    }
  }
}

/*
 * Compute A1 = A(pinv,q) where pinv and q are permutations of 0..m-1 and 0..n-1.
 * if pinv or q is NULL it is considered as the identity
 */
void preAlps_matrix_permute (int n, int *xa, int *asub, double *a, int *pinv, int *q,int *xa1, int *asub1,double *a1)
{
  int j, jp, i, nz = 0;

  for (i = 0 ; i < n ; i++){
    xa1 [i] = nz ;
    jp = q==NULL ? i: q [i];
    for (j = xa [jp] ; j < xa [jp+1] ; j++){
        asub1 [nz] = pinv==NULL ? asub [j]: pinv [asub [j]]  ;
        a1 [nz] = a [j] ;
        nz++;
    }
  }

  xa1 [n] = nz ;
  /*Sort the row index of the matrix*/
  preAlps_matrix_colIndex_sort(n, xa1, asub1, a1);
}

/*
 * We consider one binary tree A and two array part_in and part_out.
 * part_in stores the nodes of A as follows: first all the children at the last level n,
 * then all the children at the level n-1 from left to right, and so on,
 * while part_out stores the nodes of A in depth first search, so each parent node follows all its children.
 * The array part_in is compatible with the array sizes returned by ParMETIS_V3_NodeND.
 * part_out[i] = j means node i in part_in correspond to node j in part_in.
*/
void preAlps_NodeNDPostOrder(int nparts, int *part_in, int *part_out){

  int pos = nparts-1;
  int twoPowerLevel = nparts+1;
  int level = twoPowerLevel;
  while(level>1){

    preAlps_NodeNDPostOrder_targetLevel(level, twoPowerLevel, nparts-1, part_in, part_out, &pos);
    level = (int) level/2;
  }

}

/*
 * Number the nodes at level targetLevel and decrease the value of pos.
*/
void preAlps_NodeNDPostOrder_targetLevel(int targetLevel, int twoPowerLevel, int part_root, int *part_in, int *part_out, int *pos){

    if(twoPowerLevel == targetLevel){

      // We have reached the target level, number the node and decrease the value of pos
      part_out[part_root] = part_in[(*pos)--];

    }else if(twoPowerLevel>targetLevel){

      if(twoPowerLevel>2){

        twoPowerLevel = (int) twoPowerLevel/2;
        // Right
        preAlps_NodeNDPostOrder_targetLevel(targetLevel, twoPowerLevel , part_root - 1, part_in, part_out, pos);

        // Left
        preAlps_NodeNDPostOrder_targetLevel(targetLevel, twoPowerLevel, part_root - twoPowerLevel, part_in, part_out, pos);
      }

    }
}



/* pinv = p', or p = pinv' */
int *preAlps_pinv (int const *p, int n){
  int k, *pinv ;
  pinv = (int *) malloc (n *sizeof (int)) ;  /* allocate memory for the results */
  for (k = 0 ; k < n ; k++) pinv [p [k]] = k ;/* invert the permutation */
  return (pinv) ;        /* return result */
}

/* pinv = p', or p = pinv' */
int preAlps_pinv_outplace (int const *p, int n, int *pinv){
  int k ;
  for (k = 0 ; k < n ; k++) pinv [p [k]] = k ;/* invert the permutation */
  return 0;        /* return result */
}

/*
 * Permute the rows of the matrix with offDiag elements at the bottom
 * locA:
 *     input: the local part of the matrix owned by the processor calling this routine
 * idxRowBegin:
 *     input: the global array to indicate the partition of each column
 * nbDiagRows:
 *     output: the number of rows permuted on the diagonal
 * colPerm
 *     output: a preallocated vector of the size of the number of columns of A
 *            to return the global permutation vector
*/
int preAlps_permuteOffDiagRowsToBottom(CPLM_Mat_CSR_t *locA, int *idxColBegin, int *nbDiagRows, int *colPerm, MPI_Comm comm){
  int i,j, pos = 0, ierr   = 0, nbOffDiagRows = 0, my_rank, nbprocs;
  int *mark;
  int *locRowPerm;
  CPLM_Mat_CSR_t locAP = CPLM_MatCSRNULL();
  int *recvcounts;

  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &nbprocs);

  int mloc = locA->info.m;
  int n = locA->info.n;

  preAlps_int_printSynchronized(mloc, "mloc", comm);
  preAlps_int_printSynchronized(idxColBegin[my_rank], "mcolBegin", comm);
  preAlps_int_printSynchronized(idxColBegin[my_rank+1], "mcol End", comm);

#ifdef DEBUG
  printf("[%d] permuteOffDiagRowsToBottom mloc:%d, n:%d\n", my_rank, mloc, n);
#endif

  if ( !(mark  = (int *) malloc(mloc*sizeof(int))) ) preAlps_abort("Malloc fails for mark[].");
  if ( !(locRowPerm  = (int *) malloc(mloc*sizeof(int))) ) preAlps_abort("Malloc fails for locRowPerm[].");

  //ierr = CPLM_IVectorMalloc(&locRowPerm, locA->info.m); preAlps_checkError(ierr);

  //Compute the number of rows offDiag

  for (i=0; i< mloc; i++){
    mark[i] = -1;
    for (j=locA->rowPtr[i]; j<locA->rowPtr[i+1]; j++){
      if(locA->colInd[j] < idxColBegin[my_rank] || locA->colInd[j]>=idxColBegin[my_rank+1]){
         /* Off Diag element */
         mark[i] = pos++;
         nbOffDiagRows++;
         break; /* next row*/
      }

    }
  }

  //Construct the local row permutation vector
  pos = 0;

  /* Set the number of rows permuted on the diagonal*/
  *nbDiagRows = mloc - nbOffDiagRows;

  for (i=0; i<mloc; i++){
    if(mark[i]==-1) {
      locRowPerm[pos++] = i; //diag elements
    }
    else {
      locRowPerm[mark[i] + (*nbDiagRows)] = i;
      //nbOffDiagRows--;
    }
  }


  #ifdef DEBUG
      preAlps_intVector_printSynchronized(locRowPerm, mloc, "locRowPerm", "locRowPerm in permuteOffDiag", comm);
      preAlps_permVectorCheck(locRowPerm, mloc);
  #endif

  if ( !(recvcounts  = (int *) malloc(nbprocs*sizeof(int))) ) preAlps_abort("Malloc fails for recvcounts[].");

  for(i=0;i<nbprocs;i++) recvcounts[i] = idxColBegin[i+1] - idxColBegin[i];

  preAlps_intVector_printSynchronized(recvcounts, nbprocs, "recvcounts", "recvcounts in permuteOffDiag", comm);


  ierr = MPI_Allgatherv(locRowPerm, mloc, MPI_INT, colPerm, recvcounts, idxColBegin, MPI_INT, comm); preAlps_checkError(ierr);

#ifdef DEBUG
  preAlps_intVector_printSynchronized(colPerm, n, "colPerm", "colPerm in permuteOffDiag", comm);
#endif

  //Update global indexes of colPerm
  for(i=0;i<nbprocs;i++){
    for(j=idxColBegin[i];j<idxColBegin[i+1];j++){
      colPerm[j]+=idxColBegin[i];
    }
  }


  //CPLM_IVectorPrintSynchronized (colPerm, comm, "colPerm", "global colPerm in permuteOffDiag");
  if(my_rank==0) preAlps_intVector_printSynchronized(colPerm, n, "colPerm", "global colPerm in permuteOffDiag", MPI_COMM_SELF);


  #ifdef DEBUG
      printf("[permuteOffDiagRowsToBottom] Checking colPerm \n");
      preAlps_permVectorCheck(colPerm, n);
  #endif

  /* AP = P x A x  P^T */

  ierr  = CPLM_MatCSRPermute(locA, &locAP, locRowPerm, colPerm, PERMUTE); preAlps_checkError(ierr);

  /* Replace the matrix with the permuted one*/
  CPLM_MatCSRCopy(&locAP, locA);

  //CPLM_MatCSRFree(&locA);
  //locA = &locAP;
  CPLM_MatCSRFree(&locAP);

  free(locRowPerm);
  free(mark);
  free(recvcounts);
  return 0;
}

/*
 * Permute the matrix to create the global matrix structure where all the Block diag are ordered first
 * followed by the Schur complement.
 * The permuted local matrix will have the form locA = [... A_{i, Gamma};... A_{gamma,gamma}]
 *
 * nbDiagRowsloc:
       input: the number of diagonal block on the processor callinf this routine
 * locA:
 *     input: the local part of the matrix owned by the processor calling this routine
 * idxRowBegin:
 *     input: the global array to indicate the column partitioning
 * locAP:
 *     output: the permuted matrix
 * colPerm
 *     output: a preallocated vector of the size of the number of columns of A
 *            to return the global permutation vector
 * schur_ncols
 *    output: the number of column of the schur complement after the partitioning
 *
*/
int preAlps_permuteSchurComplementToBottom(CPLM_Mat_CSR_t *locA, int nbDiagRows, int *idxColBegin, CPLM_Mat_CSR_t *locAP, int *colPerm, int *schur_ncols, MPI_Comm comm){

  int nbprocs, my_rank;
  int *workP; //a workspace of the size of the number of procs
  int i, j, ierr = 0, sum = 0, count = 0;

  //CPLM_Mat_CSR_t locAP = CPLM_MatCSRNULL();
  int *locRowPerm;//permutation applied on the local matrix

  int mloc, n, r;

  mloc = locA->info.m; n = locA->info.n;

  MPI_Comm_size(comm, &nbprocs);
  MPI_Comm_rank(comm, &my_rank);

  //Workspace
  if ( !(workP  = (int *) malloc(nbprocs * sizeof(int))) ) preAlps_abort("Malloc fails for workP[].");
  if ( !(locRowPerm  = (int *) malloc(mloc * sizeof(int))) ) preAlps_abort("Malloc fails for locRowPerm[].");

  //Gather the number of elements in the diag for each procs
  MPI_Allgather(&nbDiagRows, 1, MPI_INT, workP, 1, MPI_INT, comm);


  sum = 0;
  for(i=0;i<nbprocs;i++) sum+=workP[i];

  count = 0; r = sum;
  for(i=0;i<nbprocs;i++){

    /* First part of the matrix */
    for(j=idxColBegin[i];j<idxColBegin[i]+workP[i];j++){
      colPerm[count++] = j;
    }
    /* Schur complement part of the matrix */
    for(j=idxColBegin[i]+workP[i];j<idxColBegin[i+1];j++){
      colPerm[r++] = j;
    }
  }

  if(my_rank==0) preAlps_intVector_printSynchronized(colPerm, n, "colPerm", "colPerm in permuteSchurToBottom", MPI_COMM_SELF);

  *schur_ncols = n - sum;

  //permute the matrix to form the schur complement
  for(i=0;i<mloc;i++) locRowPerm[i] = i; //no change in the rows
  ierr  = CPLM_MatCSRPermute(locA, locAP, locRowPerm, colPerm, PERMUTE); preAlps_checkError(ierr);

#ifdef DEBUG
  CPLM_MatCSRPrintSynchronizedCoords (locAP, comm, "locAP", "locAP after permuteSchurToBottom");
#endif

  /*Copy and free the workspace matrice*/
  //CPLM_MatCSRCopy(&locAP, locA);
  //CPLM_MatCSRFree(&locAP);

  free(workP);

  return ierr;
}


/*
 * Check the permutation vector for consistency
 */
int preAlps_permVectorCheck(int *perm, int n){
  int i,j, found;

  for(i=0;i<n;i++){
    found = 0;
    for(j=0;j<n;j++){
      if(perm[j]==i) {found = 1; break;}
    }
    if(!found) {printf("permVectorCheck: entry %d not found in the vector\n", i); preAlps_abort("Error in permVectorCheck()");}
  }

  return 0;
}

/*
 * Extract the schur complement of a matrix A
 * A:
 *     input: the matrix from which we want to extract the schur complement
 * firstBlock_nrows:
 *     input: the number of rows of the top left block
 * firstBlock_ncols:
 *     input: the number of cols of the top left block
 * Agg
 *     output: the schur complement matrix
*/

int preAlps_schurComplementGet(CPLM_Mat_CSR_t *A, int firstBlock_nrows, int firstBlock_ncols, CPLM_Mat_CSR_t *Agg){

  /*Count the element */
  int i, j, count, schur_nrows, schur_ncols, ierr = 0;

  count = 0;
  for(i=firstBlock_nrows;i<A->info.m;i++){
    for(j=A->rowPtr[i];j<A->rowPtr[i+1];j++){
      if(A->colInd[j]<firstBlock_ncols) continue;
      count ++;
    }
  }

 schur_nrows = A->info.m - firstBlock_nrows;
 schur_ncols = A->info.n - firstBlock_ncols;

  //preAlps_int_printSynchronized(count, "nnz in Agg", comm);

  CPLM_MatCSRSetInfo(Agg, schur_nrows, schur_ncols, count,
                schur_nrows, schur_ncols, count, 1);

  ierr = CPLM_MatCSRMalloc(Agg); preAlps_checkError(ierr);

  //Fill the matrix
  Agg->rowPtr[0] = 0; count = 0;
  for(i=firstBlock_nrows;i<A->info.m;i++){
    for(j=A->rowPtr[i];j<A->rowPtr[i+1];j++){

      if(A->colInd[j]<firstBlock_ncols) continue;

      /* Schur complement element , copy it with local indexing */
      Agg->colInd[count] = A->colInd[j]-firstBlock_ncols;

      Agg->val[count] = A->val[j];
      count ++;

    }

    Agg->rowPtr[i-firstBlock_nrows+1] = count;
  }

  return ierr;
}

/*Force the current process to sleep few seconds for debugging purpose*/
void preAlps_sleep(int my_rank, int nbseconds){
#ifdef DEBUG
  printf("[%d] Sleeping: %d (s)\n", my_rank, nbseconds);
  sleep(nbseconds);
#endif
}

/* Load a vector from a file and distribute the other procs */
int preAlps_loadDistributeFromFile(MPI_Comm comm, char *fileName, int *mcounts, double **x){
  int ierr = 0, nbprocs, my_rank, root = 0;
  int xlen, *moffsets;
  double *xtmp = NULL;
  double *xvals;
  int k, xnval;

  MPI_Comm_size(comm, &nbprocs);
  MPI_Comm_rank(comm, &my_rank);

  //load the vector
  if(my_rank==0) {
    preAlps_doubleVector_load(fileName, &xtmp, &xlen);
  }

  //distribute
  xnval = mcounts[my_rank];

  moffsets = (int*) malloc((nbprocs+1)*sizeof(int));
  xvals = (double*) malloc(xnval*sizeof(double));

  moffsets[0] = 0;
  for(k=0;k<nbprocs;k++) moffsets[k+1] = moffsets[k] + mcounts[k];

  MPI_Scatterv(xtmp, mcounts, moffsets, MPI_DOUBLE, xvals, xnval, MPI_DOUBLE, root, comm);

  free(moffsets);
  free(xtmp);

  *x = xvals;
  return ierr;
}

/* Create two MPI typeVector which can be use to assemble a local vector to a global one */
int preAlps_multiColumnTypeVectorCreate(int ncols, int local_nrows, int global_nrows, MPI_Datatype *localType, MPI_Datatype *globalType){
  int ierr = 0;

  MPI_Datatype tmpType;

  MPI_Type_vector(ncols, 1, local_nrows, MPI_DOUBLE, &tmpType);
  MPI_Type_create_resized(tmpType, 0, 1 * sizeof(double), localType);
  MPI_Type_commit(localType);

  MPI_Type_vector(ncols, 1, global_nrows, MPI_DOUBLE, &tmpType);
  MPI_Type_create_resized(tmpType, 0, 1 * sizeof(double), globalType);
  MPI_Type_commit(globalType);

  return ierr;
}
