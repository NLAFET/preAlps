/*
============================================================================
Name        : preAlps_cplm_matcsr.h
Author      : Simplice Donfack
Version     : 0.1
Description : Functions of preAlps which will be part of MatCSR.
Date        : Oct 13, 2017
============================================================================
*/

#include <stdlib.h>
#include <stdio.h>
#include <mat_csr.h>
#ifdef USE_PARMETIS
  #include <parmetis.h>
#endif

#include "preAlps_param.h"
#include "preAlps_cplm_matcsr.h"
#include "preAlps_utils.h"



/*
 * Split the matrix in block column and extract the selected block column number.
 * The input matrix is unchanged
 * A:
 *     input: the input matrix
 * nparts:
 *     input: the number of block columns
 * partBegin: Array of size nparts + 1
 *     input: the begining position of each blocks
 * numBlock:
 *     input: the number of the block to remove
 * B_out:
 *     out: the output block
 */

int CPLM_MatCSRBlockColumnExtract(CPLM_Mat_CSR_t *A, int nparts, int *partBegin, int numBlock, CPLM_Mat_CSR_t *B_out){

  int ierr=0, *work = NULL;
  size_t workSize = 0;

  CPLM_IVector_t rowPart = CPLM_IVectorNULL();
  CPLM_IVector_t colPart = CPLM_IVectorNULL();

  //We have only one block Row
  ierr = CPLM_IVectorMalloc(&rowPart, 2); preAlps_checkError(ierr);
  rowPart.val[0] = 0; rowPart.val[1] = A->info.m; //keep the same number of rows

  /* Create a column partitioning */
  CPLM_IVectorCreateFromPtr(&colPart, nparts+1, partBegin);

  ierr = CPLM_MatCSRGetSubBlock (A, B_out, &rowPart, &colPart,
                                  0, numBlock, &work, &workSize); preAlps_checkError(ierr);

  B_out->info.nnz = A->info.lnnz; //tmp bug fix in CPLM_MatCSRGetSubBlock(); which does not set nnz when the matrix is empty
  if(work!=NULL) free(work);

  return ierr;
}


/*
 * Split the matrix in block column and fill the selected block column number with zeros,
 * Optimize the routine to avoid storing these zeros in the output matrix.
 * A_in:
 *     input: the input matrix
 * colCount:
 *     input: the global number of columns in each Block
 * numBlock:
 *     input: the number of the block to fill with zeros
 * B_out:
 *     output: the output matrix after removing the diag block
 */

int CPLM_MatCSRBlockColumnZerosFill(CPLM_Mat_CSR_t *A_in, int *colCount, int numBlock, CPLM_Mat_CSR_t *B_out){

  int i,j, m, lpos = 0, count = 0, ierr = 0;
  int *mwork;

  m = A_in->info.m;

  if(m<=0) return 0;

  /* Sum of the element before the selected block */
  for(i=0;i<numBlock;i++) lpos += colCount[i];

  if ( !(mwork  = (int *) malloc((m+1) * sizeof(int))) ) preAlps_abort("Malloc fails for mwork[].");


  //First precompute the number of elements outside the colmun to remove
  count = 0;
  for(i=0;i<m;i++){
    for(j=A_in->rowPtr[i];j<A_in->rowPtr[i+1];j++){
      if(A_in->colInd[j]>=lpos && A_in->colInd[j]<lpos+colCount[numBlock]) continue;
      /* element outside the column to remove , count it */
      count ++;
    }
  }


  // Set the matrix infos
  CPLM_MatCSRSetInfo(B_out, A_in->info.m, A_in->info.n, count, A_in->info.m,  A_in->info.n, count, 1);
  ierr = CPLM_MatCSRMalloc(B_out); preAlps_checkError(ierr);

  // Fill the output matrix
  count = 0;
  for(i=0;i<m;i++){

    for(j=A_in->rowPtr[i];j<A_in->rowPtr[i+1];j++){

      if(A_in->colInd[j]>=lpos && A_in->colInd[j]<lpos+colCount[numBlock]) continue;

      /* element outside the column to remove , copy it */
      B_out->colInd[count] = A_in->colInd[j];
      B_out->val[count]    = A_in->val[j];
      count ++;

    }

    mwork[i+1] = count;
  }

  B_out->rowPtr[0] = 0;
  for(i=1;i<m+1;i++) B_out->rowPtr[i] = mwork[i];

  free(mwork);

  return ierr;
}

/*
 * 1D block row distirbution of the matrix. At the end, each proc has approximatively the same number of rows.
 *
 */
int CPLM_MatCSRBlockRowDistribute(CPLM_Mat_CSR_t *Asend, CPLM_Mat_CSR_t *Arecv, int *mcounts, int *moffsets, int root, MPI_Comm comm){

  int i, m, ierr = 0;
  int nbprocs, my_rank;

  MPI_Comm_size(comm, &nbprocs);
  MPI_Comm_rank(comm, &my_rank);

  //Broadcast the matrix size from the root to the other procs
  m = Asend->info.m;
  MPI_Bcast(&m, 1, MPI_INT, root, comm);
  preAlps_int_printSynchronized(m, "m in rowDistribute", comm);

  // Split the number of rows among the processors
  for(i=0;i<nbprocs;i++){
    preAlps_nsplit(m, nbprocs, i, &mcounts[i], &moffsets[i]);
  }
  moffsets[nbprocs] = m;

  //distributes the matrix
  ierr = CPLM_MatCSRBlockRowScatterv(Asend, Arecv, moffsets, root, comm); preAlps_checkError(ierr);

  return ierr;
}
/*
 * 1D block rows gatherv of the matrix from the processors in the communicator.
 * The result is stored on processor 0.
 * ncounts: ncounts[i] = k means processor i has k rows.
 */
 /*
  * 1D block rows gather of the matrix from all the processors in the communicator .
  * Asend:
  *     input: the matrix to send
  * Arecv
  *     output: the matrix to assemble the block matrix received from all (relevant only on the root)
  * idxRowBegin:
  *     input: the global row indices of the distribution
  */
int CPLM_MatCSRBlockRowGatherv(CPLM_Mat_CSR_t *Asend, CPLM_Mat_CSR_t *Arecv,  int *idxRowBegin, int root, MPI_Comm comm){

  int nbprocs, my_rank;

  int *nxacounts=NULL, *nzcounts=NULL, *nzoffsets=NULL;

  int i, m = 0, n=0, j, nz = 0, pos, mloc, nzloc ;


  MPI_Comm_size(comm, &nbprocs);

  MPI_Comm_rank(comm, &my_rank);


  /* Determine my local number of rows*/
  mloc = idxRowBegin[my_rank+1]-idxRowBegin[my_rank];

  /* The root prepare the receiving matrix*/
  if(my_rank==root){

    /* Determine the global number of rows*/
    m = 0;

    for(i=0;i<nbprocs;i++){
      m+= (idxRowBegin[i+1]-idxRowBegin[i]);
    }

    n = Asend->info.n;

    /* The receive matrix size is unknown, we reallocate it. TODO: check the Arecv size and realloc only if needed*/
    if(Arecv->rowPtr!=NULL)   free(Arecv->rowPtr);
    if(Arecv->colInd!=NULL)   free(Arecv->colInd);
    if(Arecv->val!=NULL)      free(Arecv->val);


    if ( !(Arecv->rowPtr = (int *)   malloc((m+1)*sizeof(int))) ) preAlps_abort("Malloc fails for xa[].");

    //buffer
    if ( !(nxacounts  = (int *) malloc((nbprocs+1)*sizeof(int))) ) preAlps_abort("Malloc fails for nxacounts[].");
    if ( !(nzcounts  = (int *) malloc(nbprocs*sizeof(int))) ) preAlps_abort("Malloc fails for nzcounts[].");

    /* Compute the number of elements to gather  */

    for(i=0;i<nbprocs;i++){
      nxacounts[i] = idxRowBegin[i+1]-idxRowBegin[i];
    }

  }


  /* Shift to take into account that the other processors will not send their first elements (which is rowPtr[0] = 0) */
  if(my_rank==root) {
    for(i=0;i<nbprocs+1;i++){
      idxRowBegin[i]++;
    }

    Arecv->rowPtr[0] = 0; //first element
  }

  /* Each process send mloc element to proc 0 (without the first element) */
  MPI_Gatherv(&Asend->rowPtr[1], mloc, MPI_INT, Arecv->rowPtr, nxacounts, idxRowBegin, MPI_INT, root, comm);



  /* Convert xa from local to global by adding the last element of each subset*/
  if(my_rank==root){

    for(i=1;i<nbprocs;i++){

      pos = idxRowBegin[i];

      for(j=0;j<nxacounts[i];j++){

        /*add the number of non zeros of the previous proc */
        Arecv->rowPtr[pos+j] = Arecv->rowPtr[pos+j] + Arecv->rowPtr[pos-1];

      }
    }

  }

  /* Restore idxRowBegin in the case the caller program needs it*/
  if(my_rank==root) {
    for(i=0;i<nbprocs+1;i++){

      idxRowBegin[i]--;
    }
  }

  /* Compute number of non zeros in each rows */

  if(my_rank==root){

    if ( !(nzoffsets = (int *) malloc((nbprocs+1)*sizeof(int))) ) preAlps_abort("Malloc fails for nzoffsets[].");

    nzoffsets[0] = 0; nz = 0;
    for(i=0;i<nbprocs;i++){

      nzcounts[i] = Arecv->rowPtr[idxRowBegin[i+1]] - Arecv->rowPtr[idxRowBegin[i]];

      nzoffsets[i+1] = nzoffsets[i] + nzcounts[i];

      nz+=nzcounts[i];
    }

    if ( !(Arecv->colInd = (int *)   malloc((nz*sizeof(int)))) ) preAlps_abort("Malloc fails for Arecv->colInd[].");
    if ( !(Arecv->val = (double *)   malloc((nz*sizeof(double)))) ) preAlps_abort("Malloc fails for Arecv->val[].");
  }


  /* Gather ja and a */
  nzloc = Asend->rowPtr[mloc];

  MPI_Gatherv(Asend->colInd, nzloc, MPI_INT, Arecv->colInd, nzcounts, nzoffsets, MPI_INT, root, comm);

  MPI_Gatherv(Asend->val, nzloc, MPI_DOUBLE, Arecv->val, nzcounts, nzoffsets, MPI_DOUBLE, root, comm);

  /* Set the matrix infos */
  if(my_rank==root){
    CPLM_MatCSRSetInfo(Arecv, m, n, nz, m,  n, nz, 1);
  }

  if(my_rank==root){

    free(nxacounts);

    free(nzcounts);
    free(nzoffsets);
  }

  return 0;
}

/*
 * Gatherv a local matrix from each process and dump into a file
 *
 */
int CPLM_MatCSRBlockRowGathervDump(CPLM_Mat_CSR_t *locA, char *filename, int *idxRowBegin, int root, MPI_Comm comm){
  int nbprocs, my_rank;
  MPI_Comm_size(comm, &nbprocs);
  MPI_Comm_rank(comm, &my_rank);

  CPLM_Mat_CSR_t Awork = CPLM_MatCSRNULL();
  CPLM_MatCSRBlockRowGatherv(locA, &Awork, idxRowBegin, root, comm);

  if(my_rank==root) {
    printf("Dumping the matrix ...\n");
    CPLM_MatCSRSave(&Awork, filename);
    printf("Dumping the matrix ... done\n");
  }

  CPLM_MatCSRFree(&Awork);

  return 0;
}

 /*
  * 1D block rows distribution of the matrix to all the processors in the communicator.
  * The data are originally stored on the processor root. After this routine each processor i will have the row indexes from
  * idxRowBegin[i] to idxRowBegin[i+1] - 1 of the input matrix.
  *
  * Asend:
  *     input: the matrix to scatterv (relevant only on the root)
  * Arecv
  *     output: the matrix to store the block matrix received
  * idxRowBegin:
  *     input: the global row indices of the distribution
  */
int CPLM_MatCSRBlockRowScatterv(CPLM_Mat_CSR_t *Asend, CPLM_Mat_CSR_t *Arecv, int *idxRowBegin, int root, MPI_Comm comm){

   int nbprocs, my_rank;

   int *nzcounts=NULL, *nzoffsets=NULL, *nxacounts = NULL;
   int mloc, nzloc;

   //int *xa_ptr, *asub_ptr;
   //double *a_ptr;
   int i, n;


   MPI_Comm_size(comm, &nbprocs);

   MPI_Comm_rank(comm, &my_rank);


   /* Compute the displacements for rowPtr */
   if(my_rank==root){

     if ( !(nxacounts  = (int *) malloc((nbprocs+1)*sizeof(int))) ) preAlps_abort("Malloc fails for nxacounts[].");
   }


   /* Determine my local number of rows*/
   mloc = idxRowBegin[my_rank+1]-idxRowBegin[my_rank];

   /* Broadcast the global number of rows. Only the root processos has it*/
   if(my_rank == root) n = Asend->info.n;
   MPI_Bcast(&n, 1, MPI_INT, root, comm);

   /* Compute the new number of columns per process*/

   if(my_rank==root){
     /* Compute the number of elements to send  */

     for(i=0;i<nbprocs;i++){

       nxacounts[i] = (idxRowBegin[i+1]-idxRowBegin[i])+1;  /* add the n+1-th element required for the CSR format */
     }

   }


   /* Allocate memory for rowPtr*/
   if(Arecv->rowPtr!=NULL) free(Arecv->rowPtr);

   if ( !(Arecv->rowPtr = (int *)   malloc((mloc+1)*sizeof(int))) ) preAlps_abort("Malloc fails for Arecv->rowPtr[].");

   /* Distribute xa to each procs. Each proc has mloc+1 elements */

   MPI_Scatterv(Asend->rowPtr, nxacounts, idxRowBegin, MPI_INT, Arecv->rowPtr, mloc+1, MPI_INT, root, comm);

   /* Convert xa from global to local */
   for(i=mloc;i>=0;i--){

     Arecv->rowPtr[i] = Arecv->rowPtr[i] - Arecv->rowPtr[0];

   }

   /*
    * Distribute asub and a to each procs
    */


  nzloc = Arecv->rowPtr[mloc]; // - xa_ptr[0]


  /* Allocate memory for colInd and val */
    if(Arecv->colInd!=NULL) free(Arecv->colInd);
    if(Arecv->val!=NULL) free(Arecv->val);
    if ( !(Arecv->colInd = (int *)   malloc((nzloc*sizeof(int)))) ) preAlps_abort("Malloc fails for Arecv->colInd[].");
    if ( !(Arecv->val = (double *)   malloc((nzloc*sizeof(double)))) ) preAlps_abort("Malloc fails for Arecv->val[].");


   /* Compute number of non zeros in each rows and the displacement for nnz*/

   if(my_rank==root){
     if ( !(nzcounts  = (int *) malloc(nbprocs*sizeof(int))) ) preAlps_abort("Malloc fails for nzcounts[].");
     if ( !(nzoffsets = (int *) malloc((nbprocs+1)*sizeof(int))) ) preAlps_abort("Malloc fails for nzoffsets[].");
     nzoffsets[0] = 0;
     for(i=0;i<nbprocs;i++){

       nzcounts[i] = Asend->rowPtr[idxRowBegin[i+1]] - Asend->rowPtr[idxRowBegin[i]];
       nzoffsets[i+1] = nzoffsets[i] + nzcounts[i];
     }
   }

   /* Distribute colInd and val */
   MPI_Scatterv(Asend->colInd, nzcounts, nzoffsets, MPI_INT, Arecv->colInd, nzloc, MPI_INT, root, comm);

   MPI_Scatterv(Asend->val, nzcounts, nzoffsets, MPI_DOUBLE, Arecv->val, nzloc, MPI_DOUBLE, root, comm);

   /* Set the matrix infos */
   CPLM_MatCSRSetInfo(Arecv, mloc, n, nzloc, mloc,  n, nzloc, 1);

   if(my_rank==root){
     free(nxacounts);
     free(nzcounts);
     free(nzoffsets);
   }

   return 0;
 }



 /*Create a matrix from a dense vector of type double, the matrix is stored in column major format*/
 int CPLM_MatCSRConvertFromDenseColumnMajorDVectorPtr(CPLM_Mat_CSR_t *m_out, double *v_in, int M, int N){

  int ierr = 0;
  int nnz=0;

  for(int i=0;i<M*N;i++){
    if(v_in[i] != 0.0 ) nnz++;
  }

  /* Set the matrix infos */
  CPLM_MatCSRSetInfo(m_out, M, N, nnz, M,  N, nnz, 1);

  CPLM_MatCSRMalloc(m_out);

  int count=0;
  m_out->rowPtr[0]=0;
  for(int i=0;i<M;i++) {
    for(int j=0;j<N;j++){
      if(v_in[j*N+i] != 0.0 ) {
        m_out->colInd[count] = j;
        m_out->val[count]  = v_in[j*N+i];
        count++;
      }
    }
    m_out->rowPtr[i+1]=count;
  }

  return ierr;
}

/*Create a matrix from a dense vector of type double*/
int CPLM_MatCSRConvertFromDenseDVectorPtr(CPLM_Mat_CSR_t *m_out, double *v_in, int M, int N){

  int ierr;
  CPLM_DVector_t Work1 = CPLM_DVectorNULL();
  CPLM_DVectorCreateFromPtr(&Work1, M*N, v_in);
  ierr = CPLM_MatCSRConvertFromDenseDVector(m_out, &Work1, M, N);

  return ierr;
}


/* Create a MatCSRNULL matrix, same as A = CPLM_MatCSRNULL() but for a matrix referenced as pointer. */
int CPLM_MatCSRCreateNULL(CPLM_Mat_CSR_t **A){

  int ierr = 0;

  CPLM_Info_t info = { .M=0, .N=0, .nnz=0, .m=0, .n=0, .lnnz=0, .blockSize=0, .format=FORMAT_CSR, .structure=UNSYMMETRIC };
  *A  = (CPLM_Mat_CSR_t *) malloc(sizeof(CPLM_Mat_CSR_t));

  if(!*A) return 1;
  (*A)->info = info;
  (*A)->rowPtr=NULL;
  (*A)->colInd=NULL;
  (*A)->val=NULL;

  return ierr;
}


/*
 * Matrix matrix product, C := alpha*A*B + beta*C
 * where A is a CSR matrix, B and C is are dense Matrices stored in column major layout/
 */
int CPLM_MatCSRMatrixCSRDenseMult(CPLM_Mat_CSR_t *A, double alpha, double *B, int B_ncols, int ldB, double beta, double *C, int ldC){
  int ierr = 0;

  #ifdef USE_MKL
    int i;
    char matdescra[6] ={'G',' ',' ', 'F', ' ', ' '};
    //char matdescra[6] ={'G',' ',' ', 'C', ' ', ' '}; //'X'


    // B and C are stored as Colum major layout, so MKL assumes that the input matrix is 1-based indexed
    for (i = 0; i < A->info.m+1; i++) {
        A->rowPtr[i] ++;
    }
    for (i = 0; i < A->info.lnnz; i++) {
        A->colInd[i] ++;
	  }

    //printf("m:%d, b_ncols:%d, n:%d, alpha:%f, beta:%f\n", A->info.m, B_ncols, A->info.n, alpha, beta);
    mkl_dcsrmm("N", &A->info.m, &B_ncols, &A->info.n, &alpha, matdescra,
              A->val, A->colInd, A->rowPtr, &A->rowPtr[1], B, &ldB, &beta, C, &ldC);

    //restore the indexing
    for (i = 0; i < A->info.m+1; i++) {
        A->rowPtr[i] --;
    }
    for (i = 0; i < A->info.lnnz; i++) {
        A->colInd[i] --;
	  }

  #else
   preAlps_abort("Only MKL is supported so far for the Matrix matrix product. Please compile with MKL\n");
  #endif

  return ierr;
}

 /*
  * Matrix vector product, y := alpha*A*x + beta*y
  */
int CPLM_MatCSRMatrixVector(CPLM_Mat_CSR_t *A, double alpha, double *x, double beta, double *y){


    #ifdef USE_MKL

     //char matdescra[6] = {'G', '\0', '\0', 'C', '\0', '\0'};
     //char matdescra[] = "G**C**";
     char matdescra[6] = {'G', ' ', ' ', 'C', ' ', ' '};
     mkl_dcsrmv("N", &A->info.m, &A->info.n, &alpha, matdescra, A->val, A->colInd, A->rowPtr, &A->rowPtr[1], x, &beta, y);
    #else
     //preAlps_abort("Only MKL is supported so far for the Matrix vector product\n");
     int i,j ;
     for (i=0; i<A->info.m; i++ ){
       //y[i]=0;
       for (j=A->rowPtr[i]; j<A->rowPtr[i+1]; j++) {
        y[i] = beta*y[i] + alpha * A->val[j]*x[A->colInd[j]];
       }
     }

     //return 1;
    #endif

    return 0;
}


/*
 * Perform an ordering of a matrix using parMetis
 *
 */

int CPLM_MatCSROrderingND(MPI_Comm comm, CPLM_Mat_CSR_t *A, int *vtdist, int *order, int *sizes){

  int err = 0;

#ifdef USE_PARMETIS

  idx_t options[METIS_NOPTIONS];
  idx_t numflag = 0; /*C-style numbering*/

  options[0] = 0;
  options[1] = 0;
  options[2] = 42; /* Fixed Seed for reproducibility */

#if 1

  //Silent parmetis compilation warning
  int i, nbprocs, nparts;
  idx_t *pmetis_vtdist, *pmetis_order, *pmetis_sizes;
  idx_t *pmetis_rowPtr, *pmetis_colInd;

  MPI_Comm_size(comm, &nbprocs);

  nparts = 1;
  while(nparts<nbprocs){
    nparts = 2*nparts+1;
  }

  if ( !(pmetis_vtdist = (idx_t *)   malloc(((nbprocs+1)*sizeof(idx_t)))) ) preAlps_abort("Malloc fails for pmetis_order[].");
  if ( !(pmetis_order  = (idx_t *)   malloc((A->info.m*sizeof(idx_t)))) ) preAlps_abort("Malloc fails for pmetis_order[].");
  if ( !(pmetis_sizes  = (idx_t *)   malloc((nparts*sizeof(idx_t)))) ) preAlps_abort("Malloc fails for pmetis_sizes[].");
  if ( !(pmetis_rowPtr = (idx_t *)   malloc(((A->info.m+1)*sizeof(idx_t)))) ) preAlps_abort("Malloc fails for pmetis_order[].");
  if ( !(pmetis_colInd = (idx_t *)   malloc((A->info.lnnz*sizeof(idx_t)))) ) preAlps_abort("Malloc fails for pmetis_sizes[].");

  //convert to idx_t
  if(sizeof(idx_t)!=sizeof(int)) printf("[preAlps parMetis] *** Type missmatch: sizeof(idx_t):%lu, sizeof(int):%lu. Please compile parMetis with appropriate idx_t 32;\n", sizeof(idx_t), sizeof(int));

  for(i=0;i<nbprocs+1;i++)    pmetis_vtdist[i] = (idx_t) vtdist[i];
  for(i=0;i<A->info.m+1;i++)  pmetis_rowPtr[i] = (idx_t) A->rowPtr[i];
  for(i=0;i<A->info.lnnz;i++) pmetis_colInd[i] = (idx_t) A->colInd[i];

  //call parMetis
  err = ParMETIS_V3_NodeND (pmetis_vtdist, pmetis_rowPtr, pmetis_colInd, &numflag, options, pmetis_order, pmetis_sizes, &comm);

  //copy back the result

  for(i=0;i<A->info.m;i++)    order[i]  = pmetis_order[i];
  for(i=0;i<nparts;i++)       sizes[i]  = pmetis_sizes[i];


  free(pmetis_rowPtr);
  free(pmetis_colInd);
  free(pmetis_vtdist);
  free(pmetis_order);
  free(pmetis_sizes);
#else
  err = ParMETIS_V3_NodeND (vtdist, A->rowPtr, A->colInd, &numflag, options, order, sizes, &comm);
#endif

  if(err!=METIS_OK) {printf("METIS returned error:%d\n", err); preAlps_abort("ParMetis Ordering Failed.");}

#else
  preAlps_abort("No other NodeND partitioning tool is supported at the moment. Please Rebuild with ParMetis !");
#endif

  return err;
}


/*
 * Partition a matrix using parMetis
 * part_loc:
 *     output: part_loc[i]=k means rows i belongs to subdomain k
 */

int CPLM_MatCSRPartitioningKway(MPI_Comm comm, CPLM_Mat_CSR_t *A, int *vtdist, int nparts, int *partloc){

  int i, err = 0;

#ifdef USE_PARMETIS

  idx_t options[METIS_NOPTIONS];
  idx_t numflag = 0; /*C-style numbering*/
  idx_t wgtflag = 0; /*No weights*/
  idx_t ncon = 1;
  idx_t edgecut = 0;
  idx_t pmetis_nparts = nparts;

  float *tpwgts;
  float *ubvec;

  if ( !(tpwgts = (float *)   malloc((nparts*ncon*sizeof(float)))) ) preAlps_abort("Malloc fails for tpwgts[].");
  if ( !(ubvec = (float *)    malloc((ncon*sizeof(float)))) ) preAlps_abort("Malloc fails for ubvec[].");

  options[0] = 0;
  options[1] = 0;
  options[2] = 42; /* Fixed Seed for reproducibility */

  for(i=0;i<nparts*ncon;i++) tpwgts[i] = 1.0/(real_t)nparts;
  for(i=0;i<ncon;i++) ubvec[i] =  1.05;

  //Silent parmetis compilation warning
  int nbprocs;
  MPI_Comm_size(comm, &nbprocs);
  idx_t *pmetis_vtdist, *pmetis_partloc;
  idx_t *pmetis_rowPtr, *pmetis_colInd;

  if ( !(pmetis_vtdist  = (idx_t *)   malloc(((nbprocs+1)*sizeof(idx_t)))) ) preAlps_abort("Malloc fails for pmetis_order[].");
  if ( !(pmetis_partloc = (idx_t *)   malloc((A->info.m*sizeof(idx_t)))) ) preAlps_abort("Malloc fails for pmetis_order[].");
  if ( !(pmetis_rowPtr  = (idx_t *)   malloc(((A->info.m+1)*sizeof(idx_t)))) ) preAlps_abort("Malloc fails for pmetis_order[].");
  if ( !(pmetis_colInd  = (idx_t *)   malloc((A->info.lnnz*sizeof(idx_t)))) ) preAlps_abort("Malloc fails for pmetis_sizes[].");

  //convert to idx_t
  for(i=0;i<nbprocs+1;i++)    pmetis_vtdist[i] = vtdist[i];
  for(i=0;i<A->info.m+1;i++)  pmetis_rowPtr[i] = A->rowPtr[i];
  for(i=0;i<A->info.lnnz;i++) pmetis_colInd[i] = A->colInd[i];

  //Call parmetis
  err = ParMETIS_V3_PartKway(pmetis_vtdist, pmetis_rowPtr, pmetis_colInd, NULL, NULL,
      &wgtflag, &numflag, &ncon, &pmetis_nparts, tpwgts, ubvec, options, &edgecut,
        pmetis_partloc, &comm);

  //copy back the result
  for(i=0;i<A->info.m;i++)    partloc[i]  = pmetis_partloc[i];

  free(pmetis_rowPtr);
  free(pmetis_colInd);
  free(pmetis_vtdist);
  free(pmetis_partloc);


  if(err!=METIS_OK) {printf("METIS returned error:%d\n", err); preAlps_abort("ParMetis Failed.");}

  free(tpwgts);
  free(ubvec);

#else
  preAlps_abort("No other Kway partitioning tool is supported at the moment. Please Rebuild with ParMetis !");
#endif
  return err;
}


/*
 * Print a CSR matrix as coordinate triplet (i,j, val)
 * Work only in debug mode
 */
void CPLM_MatCSRPrintCoords(CPLM_Mat_CSR_t *A, char *s){
#ifdef DEBUG
  int i,j;
  #ifdef PRINT_MOD
   int mark_i = 0, mark_j = 0;
  #endif
  if(s) printf("%s\n", s);

  for (i=0; i<A->info.m; i++){
    #ifdef PRINT_MOD
      //print only the borders and some values of the vector
      if((i>PRINT_DEFAULT_HEADCOUNT) && (i<A->info.m-1-PRINT_DEFAULT_HEADCOUNT) && (i%PRINT_MOD!=0)) {
        if(mark_i==0) {printf("... ... ...\n"); mark_i=1;} //prevent multiple print of "..."
        continue;
      }

      mark_i = 0;
      mark_j = 0;
    #endif

    for (j=A->rowPtr[i]; j<A->rowPtr[i+1]; j++){
      #ifdef PRINT_MOD
        //print only the borders and some values of the vector
        if((j>PRINT_DEFAULT_HEADCOUNT) && (j<A->rowPtr[i+1]-1-PRINT_DEFAULT_HEADCOUNT) && (A->colInd[j]%PRINT_MOD!=0)) {
          if(mark_j==0) {printf("... ... ...\n"); mark_j=1;} //prevent multiple print of "..."
          continue;
        }
        mark_j = 0;
      #endif
      printf("%d %d %20.19g\n", i, A->colInd[j], A->val[j]);
    }
  }
#endif
}



/* Only one process print its matrix, forces synchronisation between all the procs in the communicator*/
void CPLM_MatCSRPrintSingleCoords(CPLM_Mat_CSR_t *A, MPI_Comm comm, int root, char *varname, char *s){
#ifdef DEBUG
  int nbprocs, my_rank;

  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &nbprocs);

  if(my_rank==root) CPLM_MatCSRPrintCoords(A, s);

  MPI_Barrier(comm);
#endif
}

/*
 * Each processor print the matrix it has as coordinate triplet (i,j, val)
 * Work only in debug (-DDEBUG) mode
 * A:
 *    The matrix to print
 */

void CPLM_MatCSRPrintSynchronizedCoords (CPLM_Mat_CSR_t *A, MPI_Comm comm, char *varname, char *s){
#ifdef DEBUG
  int i;

  CPLM_Mat_CSR_t Abuffer = CPLM_MatCSRNULL();
  int my_rank, comm_size;

  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &comm_size);

  if(my_rank ==0){

    printf("[%d] %s\n", 0, s);

    CPLM_MatCSRPrintCoords(A, NULL);

    for(i = 1; i < comm_size; i++) {

      /*Receive a matrix*/
      CPLM_MatCSRRecv(&Abuffer, i, comm);

      printf("[%d] %s\n", i, s);
      CPLM_MatCSRPrintCoords(&Abuffer, NULL);
    }
    printf("\n");

    CPLM_MatCSRFree(&Abuffer);
  }
  else{
    CPLM_MatCSRSend(A, 0, comm);
  }

  MPI_Barrier(comm);

#endif
}

/*
 * Merge the rows of two matrices (one on top of another)
 */
int CPLM_MatCSRRowsMerge(CPLM_Mat_CSR_t *A_in, CPLM_Mat_CSR_t *B_in, CPLM_Mat_CSR_t *C_out){

  int ierr=0, i, j, C_m, C_nnz, count;

  //new number of rows and nnz
  C_m =  A_in->info.m + B_in->info.m;
  C_nnz = A_in->info.lnnz + B_in->info.lnnz;


  // Set the matrix infos
  CPLM_MatCSRSetInfo(C_out, C_m, A_in->info.n, C_nnz, C_m,  A_in->info.n, C_nnz, 1);

  ierr = CPLM_MatCSRMalloc(C_out); preAlps_checkError(ierr);

  //fill the matrix
  count = 0;
  C_out->rowPtr[0] = 0;

  //copy A
  for (i = 0; i < A_in->info.m; i++){
    for (j = A_in->rowPtr[i]; j < A_in->rowPtr[i+1]; j++) {
        C_out->colInd[count] =   A_in->colInd[j];
        C_out->val[count]    =   A_in->val[j];
        count++;
    }
    C_out->rowPtr[i+1] = count;
  }

  //Copy B
  for (i = 0; i < B_in->info.m; i++){
    //copy the columns of B
    for (j = B_in->rowPtr[i]; j < B_in->rowPtr[i+1]; j++) {
        C_out->colInd[count] =   B_in->colInd[j];
        C_out->val[count]    =   B_in->val[j];
        count++;
    }
    C_out->rowPtr[A_in->info.m+i+1] = count;
  }

  return ierr;
}
/*
 *
 * Scale a scaling vectors R and C, and scale the matrix by computing A1 = R * A * C
 * A:
 *     input: the matrix to scale
 * R:
 *     output: a vector with the same size as the number of rows of the matrix
 * C:
 *     output: a vector with the same size as the number of columns of the matrix
 */

int CPLM_MatCSRSymRACScaling(CPLM_Mat_CSR_t *A, double *R, double *C){

  int i, j;
  double rcmin, rcmax;

  /* Get machine constants. */
  //double smlnum = dlamch_("S");
  //double bignum = 1. / smlnum;


  /* Find the maximum element in each row. */

  //for (i = 0; i < A->info.m; ++i) R[i] = 0.;
  for (i = 0; i < A->info.m; i++){
      R[i] = 0.0;
      for (j = A->rowPtr[i]; j < A->rowPtr[i+1]; j++) {
          R[i] = max( R[i], fabs(A->val[j]) );
      }
  }

  /* Find the maximum and minimum scale factors. */
  rcmin = R[0];
  rcmax = R[0];
  for (i = 1; i < A->info.m; ++i) {
      rcmax = max(rcmax, R[i]);
      rcmin = min(rcmin, R[i]);
  }

#ifdef DEBUG
  printf("ROW: rcmin:%e, rcmax:%e\n", rcmin, rcmax);
#endif


  //*amax = rcmax;

  if (rcmin == 0.) {
    preAlps_abort("Impossible to scale the matrix, rcmin=0");

  } else {
      /* Invert the scale factors. */
      for (i = 0; i < A->info.m; i++){
          //R[i] = 1. / MIN( MAX( R[i], smlnum ), bignum );
          R[i] = sqrt(1.0 / R[i]);
      }
      /* Compute ROWCND = min(R(I)) / max(R(I)) */
      //*rowcnd = MAX( rcmin, smlnum ) / MIN( rcmax, bignum );
  }

#if 0

  /* Find the maximum element in each col. */
  for (j = 0; j < A->info.n; ++j) C[j] = 0.;

  /* Find the maximum element in each column, assuming the row
     scalings computed above. */
  for (j = 0; j < A->info.m; ++j){

      for (i = A->rowPtr[j]; i < A->rowPtr[j+1]; ++i) {
          C[j] = MAX( C[j], fabs(A->val[i]) * R[A->colInd[i]] );
      }
  }

  /* Find the maximum and minimum scale factors. */
  rcmin = C[0];
  rcmax = C[0];
  for (j = 1; j < A->info.n; ++j) {
      rcmax = MAX(rcmax, C[j]);
      rcmin = MIN(rcmin, C[j]);
  }

  if (rcmin == 0.) {
    preAlps_abort("Impossible to scale the matrix, rcmin=0");
  } else {
      /* Invert the scale factors. */
      for (j = 0; j < A->info.n; ++j)
          C[j] = 1. / MIN( MAX( C[j], smlnum ), bignum);
      /* Compute COLCND = min(C(J)) / max(C(J)) */
    //  *colcnd = MAX( rcmin, smlnum ) / MIN( rcmax, bignum );
  }
#else
  //Assume the matrix is symmetric
  for (i = 0; i < A->info.m; i++) C[i] = R[i];
#endif

  /* Row and column scaling */
  for (i = 0; i < A->info.m; i++) {
      //cj = C[i];
      for (j = A->rowPtr[i]; j < A->rowPtr[i+1]; j++) {
        A->val[j] = R[i] * A->val[j] * C[A->colInd[j]];
      }
  }

  return 0;
}


/* Transpose a matrix */
int CPLM_MatCSRTranspose(CPLM_Mat_CSR_t *A_in, CPLM_Mat_CSR_t *B_out){

  int ierr = 0;
  int irow, jcol, jpos;
  int *work;
  int *xa = A_in->rowPtr, *asub = A_in->colInd;
  double *a = A_in->val;


  B_out->info = A_in->info;
  B_out->info.m = A_in->info.n;
  B_out->info.n = A_in->info.m;

  if(A_in->info.lnnz==0) return ierr; //Quick return

  ierr  = CPLM_MatCSRMalloc(B_out); preAlps_checkError(ierr);

  /* Allocate workspace */

  work    = (int*) malloc( (A_in->info.n+1)   * sizeof(int));

  if(!work) preAlps_abort("Malloc failed for work");


  for(jcol=0;jcol<A_in->info.n+1;jcol++){
    work[jcol] = 0;
  }

  /* Compute the number of nnz per columns in A */
  for (irow=0; irow<A_in->info.m; irow++){
    for (jcol=xa[irow]; jcol<xa[irow+1]; jcol++) {
      work[asub[jcol]]++;
    }
  }

  /* Compute the index of each row of B*/

  B_out->rowPtr[0] = 0;
  for(irow=0;irow<B_out->info.m;irow++){
    B_out->rowPtr[irow+1] = B_out->rowPtr[irow] + work[irow];

    work[irow] = B_out->rowPtr[irow]; /* reused to store the first element of row irow. used for inserting the elements in the next step*/
  }

  /* Fill the matrix */

  for (irow=0; irow<A_in->info.m; irow++){
    for (jcol=xa[irow]; jcol<xa[irow+1]; jcol++){

      /* insert (irow, asub[jcol]) the element in column asub[jcol] */

      jpos = work[asub[jcol]];
      B_out->colInd[jpos] = irow;
      B_out->val[jpos] = a[jcol];
      work[asub[jcol]]++;

    }
  }

  /* Free memory*/
  free(work);

  return ierr;

}
