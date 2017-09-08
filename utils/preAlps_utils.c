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
#include <unistd.h>
#include <stdarg.h>
#include <mat_csr.h>
#include <ivector.h>

#include "preAlps_utils.h"


/*The default number of rows to print for larger vector */
#define PRINT_DEFAULT_HEADCOUNT 500

/* tmp functions*/

/*
 * Split n in P parts.
 * Returns the number of element, and the data offset for the specified index.
 */
void s_nsplit(int n, int P, int index, int *n_i, int *offset_i)
{

  int r;

  r = n % P;

  *n_i = (int)(n-r)/P;


  *offset_i = index*(*n_i);


  if(index<r) (*n_i)++;


  if(index < r) *offset_i+=index;
  else *offset_i+=r;
}

/* pinv = p', or p = pinv' */
int *s_return_pinv (int const *p, int n)
{
    int k, *pinv ;
    pinv = (int *) malloc (n *sizeof (int)) ;  /* allocate memory for the results */
    for (k = 0 ; k < n ; k++) pinv [p [k]] = k ;/* invert the permutation */
    return (pinv) ;        /* return result */
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
 * Move in Ivector.c
 */

/*
 * Each processor print a vector of integer
 * Work only in debug (-DDEBUG) mode
 * v:
 *    The vector to print
 */

void IVectorPrintSynchronized (IVector_t *v, MPI_Comm comm, char *varname, char *s){
#ifdef DEBUG
  int i,j,mark=0;

  int TAG_PRINT = 4;

  IVector_t vbuffer = IVectorNULL();
  int my_rank, comm_size;

  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &comm_size);

  if(my_rank ==0){

    printf("[%d] %s\n", 0, s);

    for(j=0;j<v->nval;j++) {
      #ifdef PRINT_MOD
        //print only the borders and some values of the vector
        if((j>PRINT_DEFAULT_HEADCOUNT) && (j<v->nval-1-PRINT_DEFAULT_HEADCOUNT) && (j%PRINT_MOD!=0)) {

          if(mark==0) {printf("%s[...]: ...\n", varname); mark=1;} //prevent multiple print of "..."

          continue;
        }
        mark = 0;
      #endif

      printf("%s[%d]: %d\n", varname, j, v->val[j]);

    }

    for(i = 1; i < comm_size; i++) {

      /*Receive a Vector*/

      IVectorRecv(&vbuffer, i, TAG_PRINT, comm);
      mark = 0;
      printf("[%d] %s\n", i, s);
      for(j=0;j<vbuffer.nval;j++) {
        #ifdef PRINT_MOD
          //print only the borders and some values of the vector
          if((j>PRINT_DEFAULT_HEADCOUNT) && (j<vbuffer.nval-1-PRINT_DEFAULT_HEADCOUNT) && (j%PRINT_MOD!=0)) {
            if(mark==0) {printf("%s[...]: ...\n", varname); mark=1;} //prevent multiple print of "..."
            continue;
          }

          mark = 0;
        #endif

        printf("%s[%d]: %d\n", varname, j, vbuffer.val[j]);

      }

    }
    printf("\n");

    IVectorFree(&vbuffer);
  }
  else{
    IVectorSend(v, 0, TAG_PRINT, comm);
  }

  MPI_Barrier(comm);

#endif
}


/*
 * Move in Dvector.c
 */

/*
 * Each processor print a vector of double
 * Work only in debug (-DDEBUG) mode
 * v:
 *    The vector to print
 */

void DVectorPrintSynchronized (DVector_t *v, MPI_Comm comm, char *varname, char *s){
#ifdef DEBUG
  int i,j,mark = 0;

  int TAG_PRINT = 4;

  DVector_t vbuffer = DVectorNULL();
  int my_rank, comm_size;

  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &comm_size);

  if(my_rank ==0){

    printf("[%d] %s\n", 0, s);

    for(j=0;j<v->nval;j++) {

      #ifdef PRINT_MOD
        //print only the borders and some values of the vector
        if((j>PRINT_DEFAULT_HEADCOUNT) && (j<v->nval-1-PRINT_DEFAULT_HEADCOUNT) && (j%PRINT_MOD!=0)) {
          if(mark==0) {printf("%s[...]: ...\n", varname); mark=1;} //prevent multiple print of "..."
          continue;
        }
        mark = 0;
      #endif

      printf("%s[%d]: %20.19g\n", varname, j, v->val[j]);
    }

    for(i = 1; i < comm_size; i++) {

      /*Receive a Vector*/
      DVectorRecv(&vbuffer, i, TAG_PRINT, comm);

      printf("[%d] %s\n", i, s);
      mark = 0;
      for(j=0;j<vbuffer.nval;j++) {

        #ifdef PRINT_MOD
          //print only the borders and some values of the vector
          if((j>PRINT_DEFAULT_HEADCOUNT) && (j<vbuffer.nval-1-PRINT_DEFAULT_HEADCOUNT) && (j%PRINT_MOD!=0)) {
            if(mark==0) {printf("%s[...]: ...\n", varname); mark=1;} //prevent multiple print of "..."
            continue;
          }
          mark = 0;
        #endif

        printf("%s[%d]: %20.19g\n", varname, j, vbuffer.val[j]);

      }
    }
    printf("\n");

    DVectorFree(&vbuffer);
  }
  else{
    DVectorSend(v, 0, TAG_PRINT, comm);
  }

  MPI_Barrier(comm);

#endif
}


/*
 * Move in MatCSR.c
 */

//#define MATCSR_UNKNOWN -1

/*
 * Split the matrix in block column and remove the selected block column number,
 * Which is the same as replacing all values of that block with zeros.
 * This reoutine can be used to fill the diag of a Block diag of global matrix with zeros when each proc
 * has a rowPanel as locA.
 * A:
 *     input: the input matrix to remove the diag block
 * colCount:
 *     input: the global number of columns in each Block
 * numBlock:
 *     input: the number of the block to remove
 */

int MatCSRBlockColRemove(Mat_CSR_t *A, int *colCount, int numBlock){

  int i,j, m, lpos = 0, count = 0;
  int *mwork;

  m = A->info.m;

  if(m<=0) return 0;

  /* Sum of the element before the selected block */
  for(i=0;i<numBlock;i++) lpos += colCount[i];

  //preAlps_int_printSynchronized(lpos, "lpos", comm);
  //printf("lpos:%d, m:%d\n", lpos, m);

  if ( !(mwork  = (int *) malloc((m+1) * sizeof(int))) ) preAlps_abort("Malloc fails for mwork[].");

  for(i=0;i<m;i++){
    //mwork[i+1] = 0;
    for(j=A->rowPtr[i];j<A->rowPtr[i+1];j++){

      if(A->colInd[j]>=lpos && A->colInd[j]<lpos+colCount[numBlock]) continue;

      /* OffDiag element , copy it */
      A->colInd[count] = A->colInd[j];
      A->val[count] = A->val[j];
      count ++;

    }

    mwork[i+1] = count;
  }

  for(i=1;i<m+1;i++) A->rowPtr[i] = mwork[i];

  free(mwork);

  return 0;
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
int MatCSRBlockRowGatherv(Mat_CSR_t *Asend, Mat_CSR_t *Arecv,  int *idxRowBegin, int root, MPI_Comm comm){

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

    /* The receive matrix is unknown, we reallocate it. TODO: check the Arecv size and realloc only if needed*/
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



  /*gather ja and a*/

  nzloc = Asend->rowPtr[mloc];

  //s_int_print_mp(comm, nzloc, "nzloc");


  MPI_Gatherv(Asend->colInd, nzloc, MPI_INT, Arecv->colInd, nzcounts, nzoffsets, MPI_INT, root, comm);

  MPI_Gatherv(Asend->val, nzloc, MPI_DOUBLE, Arecv->val, nzcounts, nzoffsets, MPI_DOUBLE, root, comm);

  /* Set the matrix infos */
  if(my_rank==root){
    MatCSRSetInfo(Arecv, m, n, nz, m,  n, nz, 1);
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
int MatCSRBlockRowGathervDump(Mat_CSR_t *locA, char *filename, int *idxRowBegin, int root, MPI_Comm comm){
  int nbprocs, my_rank;
  MPI_Comm_size(comm, &nbprocs);
  MPI_Comm_rank(comm, &my_rank);

  Mat_CSR_t Awork = MatCSRNULL();
  MatCSRBlockRowGatherv(locA, &Awork, idxRowBegin, root, comm);
  printf("Dumping the matrix ...\n");
  if(my_rank==root) MatCSRSave(&Awork, filename);
  printf("Dumping the matrix ... done\n");
  MatCSRFree(&Awork);

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
int MatCSRBlockRowScatterv(Mat_CSR_t *Asend, Mat_CSR_t *Arecv, int *idxRowBegin, int root, MPI_Comm comm){

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

   preAlps_int_printSynchronized(mloc, "mloc in MatCSRBlockRowScatterv", comm);

   /* Broadcast the global number of rows. Only the root processos has it*/
   if(my_rank == root) n = Asend->info.n;
   MPI_Bcast(&n, 1, MPI_INT, root, comm);
   preAlps_int_printSynchronized(n, "n in MatCSRBlockRowScatterv", comm);

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


   //xa_ptr = Arecv->rowPtr;


   /* Distribute xa to each procs. Each proc has mloc+1 elements */

   MPI_Scatterv(Asend->rowPtr, nxacounts, idxRowBegin, MPI_INT, Arecv->rowPtr, mloc+1, MPI_INT, root, comm);

   //s_ivector_print_mp (comm, xa_ptr, mloc+1, "xa", "");
   preAlps_intVector_printSynchronized(Arecv->rowPtr, mloc+1, "xa", "xa received", comm);

   /* Convert xa from global to local */
   for(i=mloc;i>=0;i--){

     Arecv->rowPtr[i] = Arecv->rowPtr[i] - Arecv->rowPtr[0];

   }

   //s_ivector_print_mp (comm, xa_ptr, mloc+1, "xa", "xa local");
   preAlps_intVector_printSynchronized(Arecv->rowPtr, mloc+1, "xa", "xa local", comm);


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
   MatCSRSetInfo(Arecv, mloc, n, nzloc, mloc,  n, nzloc, 1);

   if(my_rank==root){
     free(nxacounts);
     free(nzcounts);
     free(nzoffsets);
   }

   return 0;
 }



 /*Create a matrix from a dense vector of type double, the matrix is stored in column major format*/
 int MatCSRConvertFromDenseColumnMajorDVectorPtr(Mat_CSR_t *m_out, double *v_in, int M, int N){

  int ierr = 0;
  int nnz=0;

  for(int i=0;i<M*N;i++){
    if(v_in[i] != 0.0 ) nnz++;
  }

  /* Set the matrix infos */
  MatCSRSetInfo(m_out, M, N, nnz, M,  N, nnz, 1);

  MatCSRMalloc(m_out);

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
int MatCSRConvertFromDenseDVectorPtr(Mat_CSR_t *m_out, double *v_in, int M, int N){

  int ierr;
  DVector_t Work1 = DVectorNULL();
  DVectorCreateFromPtr(&Work1, M*N, v_in);
  ierr = MatCSRConvertFromDenseDVector(m_out, &Work1, M, N);

  return ierr;
 }



 /*
  * Matrix vector product, y := alpha*A*x + beta*y
  */
 int MatCSRMatrixVector(Mat_CSR_t *A, double alpha, double *x, double beta, double *y){


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
 * Print a CSR matrix as coordinate triplet (i,j, val)
 * Work only in debug mode
 */
void MatCSRPrintCoords(Mat_CSR_t *A, char *s){
#ifdef DEBUG
  int i,j, mark_i = 0, mark_j = 0;
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
void MatCSRPrintSingleCoords(Mat_CSR_t *A, MPI_Comm comm, int root, char *varname, char *s){
#ifdef DEBUG
  int nbprocs, my_rank;

  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &nbprocs);

  if(my_rank==root) MatCSRPrintCoords(A, s);

  MPI_Barrier(comm);
#endif
}

/*
 * Each processor print the matrix it has as coordinate triplet (i,j, val)
 * Work only in debug (-DDEBUG) mode
 * A:
 *    The matrix to print
 */

void MatCSRPrintSynchronizedCoords (Mat_CSR_t *A, MPI_Comm comm, char *varname, char *s){
#ifdef DEBUG
  int i;

  Mat_CSR_t Abuffer = MatCSRNULL();
  int my_rank, comm_size;

  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &comm_size);

  if(my_rank ==0){

    printf("[%d] %s\n", 0, s);

    MatCSRPrintCoords(A, NULL);

    for(i = 1; i < comm_size; i++) {

      /*Receive a matrix*/
      MatCSRRecv(&Abuffer, i, comm);

      printf("[%d] %s\n", i, s);
      MatCSRPrintCoords(&Abuffer, NULL);
    }
    printf("\n");

    MatCSRFree(&Abuffer);
  }
  else{
    MatCSRSend(A, 0, comm);
  }

  MPI_Barrier(comm);

#endif
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

int MatCSRSymRACScaling(Mat_CSR_t *A, double *R, double *C){
  double   *Aval;
  int i, j, irow;
  double rcmin, rcmax;
  double bignum, smlnum;


  /* Get machine constants. */
  smlnum = dlamch_("S");
  bignum = 1. / smlnum;


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

/*
 * Load a vector from a file.
 */
void preAlps_doubleVector_load(char *filename, double **v, int *vlen){

  DVector_t Work1 = DVectorNULL();

  DVectorLoad (filename, &Work1, 0);

  *v = Work1.val;
  *vlen = Work1.nval;
}


/*
 * Each processor print the vector of type double that it has.
 * Work only in debug (-DDEBUG) mode
 * v:
 *    input: The vector to print
 * vlen:
 *    input: The len of the vector to print
 * varname:
 *   The name of the vector
 * s:
 *   The string to display before the variable
 */
void preAlps_doubleVector_printSynchronized(double *v, int vlen, char *varname, char *s, MPI_Comm comm){

  DVector_t Work1 = DVectorNULL();
  if(v) DVectorCreateFromPtr(&Work1, vlen, v);
  DVectorPrintSynchronized (&Work1, comm, varname, s);
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
	  printf("%s:  min: %.6f , max: %.6f , avg: %.6f\n", str, dMin, dMax, (double) dSum/nbprocs);
  }
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

/*
 * Each processor print the vector of type int that it has.
 * Work only in debug (-DDEBUG) mode
 * v:
 *    input: The vector to print
 * vlen:
 *    input: The len of the vector to print
 * varname:
 *   The name of the vector
 * s:
 *   The string to display before the variable
 */
void preAlps_intVector_printSynchronized(int *v, int vlen, char *varname, char *s, MPI_Comm comm){
  IVector_t Work1 = IVectorNULL();
  if(v) IVectorCreateFromPtr(&Work1, vlen, v);
  IVectorPrintSynchronized (&Work1, comm, varname, s);
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
int preAlps_permuteOffDiagRowsToBottom(Mat_CSR_t *locA, int *idxColBegin, int *nbDiagRows, int *colPerm, MPI_Comm comm){
  int i,j, pos = 0, ierr   = 0, nbOffDiagRows = 0, my_rank, nbprocs;
  int *mark;
  int *locRowPerm;
  Mat_CSR_t locAP = MatCSRNULL();
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

  //ierr = IVectorMalloc(&locRowPerm, locA->info.m); preAlps_checkError(ierr);

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



  //IVectorCreateFromPtr(&itmp, locA->info.m, mark);
  //IVectorPrintSynchronized (&itmp, comm, "mark", "mark in permuteOffDiag");

  preAlps_intVector_printSynchronized(mark, mloc, "mark", "mark in permuteOffDiag", comm);


  //preAlps_sleep(my_rank, my_rank*2); //debug

  //Construct the local row permutation vector
  pos = 0;
  /*
  for (i=0; i<locA->info.m; i++){
    if(mark[i]==-1) locRowPerm.val[i] = pos++; //diag elements
    else {
      locRowPerm.val[i] = mark[i] + (locA->info.m - nbOffDiagRows);
      //nbOffDiagRows--;
    }
  }
  */

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

  //preAlps_intVector_printSynchronized (&locRowPerm, comm, "locRowPerm", "locRowPerm in permuteOffDiag");
  preAlps_intVector_printSynchronized(locRowPerm, mloc, "locRowPerm", "locRowPerm in permuteOffDiag", comm);

  #ifdef DEBUG
      printf("[permuteOffDiagRowsToBottom] Checking locRowPerm \n");
      preAlps_permVectorCheck(locRowPerm, mloc);
  #endif

  //MatCSRPrintSynchronizedCoords (locA, comm, "locA", "locA");

  /* Gather the global column permutation from all procs */
  //ierr = IVectorMalloc(&colPerm, locA->info.n); preAlps_checkError(ierr);

  if ( !(recvcounts  = (int *) malloc(nbprocs*sizeof(int))) ) preAlps_abort("Malloc fails for recvcounts[].");

  for(i=0;i<nbprocs;i++) recvcounts[i] = idxColBegin[i+1] - idxColBegin[i];

  preAlps_intVector_printSynchronized(recvcounts, nbprocs, "recvcounts", "recvcounts in permuteOffDiag", comm);


  ierr = MPI_Allgatherv(locRowPerm, mloc, MPI_INT,
                   colPerm, recvcounts, idxColBegin,
                   MPI_INT, comm);
  preAlps_checkError(ierr);

  //IVectorPrintf("colPerm in permuteOffDiag after gatherv",&colPerm);

  //IVectorPrintSynchronized (colPerm, comm, "colPerm", "colPerm in permuteOffDiag");
  preAlps_intVector_printSynchronized(colPerm, n, "colPerm", "colPerm in permuteOffDiag", comm);

  //Update global indexes of colPerm
  for(i=0;i<nbprocs;i++){
    for(j=idxColBegin[i];j<idxColBegin[i+1];j++){
      colPerm[j]+=idxColBegin[i];
    }
  }


  //IVectorPrintSynchronized (colPerm, comm, "colPerm", "global colPerm in permuteOffDiag");
  if(my_rank==0) preAlps_intVector_printSynchronized(colPerm, n, "colPerm", "global colPerm in permuteOffDiag", MPI_COMM_SELF);


  #ifdef DEBUG
      printf("[permuteOffDiagRowsToBottom] Checking colPerm \n");
      preAlps_permVectorCheck(colPerm, n);
  #endif

  #if 1
    /* AP = P x A x  P^T */

    ierr  = MatCSRPermute(locA, &locAP, locRowPerm, colPerm, PERMUTE); preAlps_checkError(ierr);
    MatCSRPrintSynchronizedCoords (&locAP, comm, "locAP", "locAP after permuteOffDiag");

    /* Replace the matrix with the permuted one*/
    MatCSRCopy(&locAP, locA);

    //MatCSRFree(&locA);
    //locA = &locAP;
    MatCSRFree(&locAP);
  #else
    printf("Disable MatcsrPermute !\n");
  #endif

  MatCSRPrintSynchronizedCoords (locA, comm, "locA", "1. locA  after permuteOffDiag");

  free(locRowPerm);
  free(mark);
  free(recvcounts);
  return 0;
}

/*
 * Permute the matrix to reflect the global matrix structure where all the Block diag are ordered first
 * followed by the Schur complement.
 * The permuted local matrix will have the form locA = [... A_{i, Gamma};... A_{gamma,gamma}]
 *
 * nbDiagRowsloc:
       input: the number of diagonal block on the processor callinf this routine
 * locA:
 *     input: the local part of the matrix owned by the processor calling this routine
 * idxRowBegin:
 *     input: the global array to indicate the column partitioning
 * colPerm
 *     output: a preallocated vector of the size of the number of columns of A
 *            to return the global permutation vector
 * schur_ncols
 *    output: the number of column of the schur complement after the partitioning
 *
*/
int preAlps_permuteSchurComplementToBottom(Mat_CSR_t *locA, int nbDiagRows, int *idxColBegin, int *colPerm, int *schur_ncols, MPI_Comm comm){

  int nbprocs, my_rank;
  int *workP; //a workspace of the size of the number of procs
  int i, j, ierr = 0, sum = 0, count = 0;

  Mat_CSR_t locAP = MatCSRNULL();
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

  //preAlps_int_printSynchronized(workP[1], "workP[1]", comm);
  preAlps_intVector_printSynchronized(workP, nbprocs, "workP", "workP", comm);


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
  ierr  = MatCSRPermute(locA, &locAP, locRowPerm, colPerm, PERMUTE); preAlps_checkError(ierr);

  MatCSRPrintSynchronizedCoords (&locAP, comm, "locAP", "locAP after permuteSchurToBottom");

  /*Copy and free the workspace matrice*/
  MatCSRCopy(&locAP, locA);

  MatCSRFree(&locAP);

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

int preAlps_schurComplementGet(Mat_CSR_t *A, int firstBlock_nrows, int firstBlock_ncols, Mat_CSR_t *Agg){

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

  MatCSRSetInfo(Agg, schur_nrows, schur_ncols, count,
                schur_nrows, schur_ncols, count, 1);

  ierr = MatCSRMalloc(Agg); preAlps_checkError(ierr);

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
