/*
 ============================================================================
 Name        : spMSV.c
 Author      : Simplice Donfack
 Version     : 0.1
 Description : Sparse matrix matrix product
 Date        : Sept 27, 2016
 ============================================================================
 */
#include <stdlib.h>
#include "s_utils.h"
#include "s_utils_mp.h"
#include "preAlps_matrix.h"
#include "preAlps_matrix_mp.h"
#include "spMSV.h"

#define A(I,J) A[(J)  + (I) * A_LD]
#define B(I,J) B[(J)  + (I) * B_LD]

#define idx(I, J) ((J)  + (I)* B_LD)

/*
 * Purpose
 * =======
 *
 * Perform a matrix matrix product C = A*B, 
 * where A is a CSR matrix, B is a CSR matrix formed by a set of vectors, and C is a CSR matrix.
 * The matrix is initially 1D row block distributed.
 *
 * Arguments
 * =========
 *
 * mloc:
 *    input: local number of rows of A owned by the processor calling this routine
 *
 * nloc:
 *    input: local number of columns of A
 *
 * rloc:
 *    input: local number of columns of B
 *
 * xa,asub,a:
 *    input: matrix A of size (mloc x nloc) stored using CSR.
 *
 * xb,bsub,b:
 *    input: matrix B of size (nloc x rloc) stored using CSC.  If B is stored using CSR , then see options below;
 *
 * a_nparts:
 *     input: the number of rows block for matrix A. Obviously, it is the number of processors in case of a 1D block distribution.
 *    The global block structure of A has size (a_nparts x a_nparts).
 *
 * a_nrowparts:
 *     input: Array of size a_nparts to indicate the number of rows in each Block column of A.
 *
 * b_nparts:
 *     input: number of block columns for matrix B
 *    The global sparse block struct size of B is (a_nparts x b_nparts).
 *
 * b_ncolparts:
 *     input: Array of size b_nparts to indicate the number of columns in Block column of B.
 *
 * ABlockStruct
 *    input: array of size (a_nparts x a_nparts) to indicate the sparse block structure of A. 
 *         ABlockStruct(i,j) = 0 if the block does not contains any element.
 *         This array must be precomputed by the user before calling this routine.
 *
 * xc,csub,c:
 *    output: matrix C stored as CSC
 *    if C is NULL, or  
 *    if C size is smaller than the size required for A*B, it will be reallocated with the enlarged size.
 *    Otherwise C will be reused.
 *    The sparse block struct size of C is (a_nparts x b_nparts)
 *
 * options: array of size 3 to control this routine. (Coming soon)
 *     input:
 *      options[0] = 1 if the input matrix B is stored as CSR instead of CSC. An internal buffer will be required.
 *      options[1] = 1 if the result matrix must be stored as CSR instead of CSC.
 *      options[2] = 1 switch off the possibility to switch to dense matrix.
 *
 *
 * Return
 * ======
 * 0: the resulting matrix C is sparse, use (xc, asubc, c) to manipulate it.
 * 1: the resulting matrix C is converted to dense, use only (c) to manipulate it.
 * < 0: an error occured during the execution.
 */

int preAlps_spMSV(MPI_Comm comm, int mloc, int nloc, int rloc, int *xa, int *asub, double *a, int *xb, int *bsub, double *b, 
  int a_nparts, int *a_nrowparts,
  int b_nparts, int *b_ncolparts,
  int *ABlockStruct, 
  int **xc, int **csub, double **c, int *options){
  
  
  /* local variables */

  int a_mparts; 
  int nbprocs, my_rank;
  
  int *A, A_LD; /*Block structure of the matrix A*/
  int *B, B_size, B_LD; //Block structure of the matrix B
  
  int I,J,K;
  int ds = 0;
  
  int *a_noffsets=NULL;
    
  double **buffer_recv;
  int *buffer_size;
  
  MPI_Status status;
  int tag, nrecv = 0, i, j, k, index, c_nnz= 0, c_expected_nnz;
  
  
  MPI_Request *requests_send;
  int requests_send_count =0;
  
  MPI_Request  *requests_recv;
  int requests_recv_count = 0;
  
  int *c_blockColumnStruct;
  
  int **ptrRowBegin, **ptrRowEnd;
  
  int *xc_ptr;
  double *c_ptr;
  
  int *partcol;  

  

  
  /* For statistics */
  double t1, tStart;
	
  int numDataRecv = 0, volDataRecv = 0;
  
  double tBStructComp = 0.0, tLocalComp = 0.0, tDepComp = 0.0, tWaitRecv = 0.0, tWaitSend = 0.0, tTime = 0.0;
	  
  /*Let's begin*/
  tStart = MPI_Wtime();
  
  
  MPI_Comm_size(comm, &nbprocs);
  MPI_Comm_rank(comm, &my_rank);
  
  //P = nbprocs;
  //N = b_nparts;
  
  if(options[0]!=0 || options[1]!=0 || options[2]!=0){
    s_abort("spMSV Options are not yet implemented, please set all options to 0.\n");
  }
  
  
  
  /*Number of block rows of A*/
  a_mparts = a_nparts;
  
  
  /*Allocate and initialize the array of receive buffer*/
  
  if ( !(a_noffsets  = (int *) malloc((nbprocs+1)*sizeof(int))) ) s_abort("Malloc fails for a_noffsets[].");
  
  if ( !(buffer_recv   = (double **)     malloc(a_nparts*b_nparts*sizeof(double*))) ) s_abort("Malloc fails for buffer_recv[].");
  if ( !(buffer_size   = (int *)         malloc(a_nparts*b_nparts*sizeof(int))) ) s_abort("Malloc fails for buffer_size[].");
  
  if ( !(requests_send  = (MPI_Request *) malloc(a_nparts*b_nparts*sizeof(MPI_Request))) ) s_abort("Malloc fails for request_send[].");
  
  
  if ( !(requests_recv  = (MPI_Request *) malloc(a_nparts*b_nparts*sizeof(MPI_Request))) ) s_abort("Malloc fails for requests_recv[].");
  
  if ( !(c_blockColumnStruct   = (int *)  malloc(b_nparts*sizeof(int))) ) s_abort("Malloc fails for c_blockColumnStruct[].");
  
  if ( !(ptrRowBegin   = (int **)  malloc(a_nparts*sizeof(int*))) ) s_abort("Malloc fails for ptrRowBegin[].");
  if ( !(ptrRowEnd     = (int **)  malloc(a_nparts*sizeof(int*))) ) s_abort("Malloc fails for ptrRowEnd[].");
  
  if ( !(partcol = (int *)   malloc(nloc*sizeof( int))) ) s_abort("Malloc fails for partcol[].");
  
  
  
  
  /*Compute the starting position of each block column*/
  a_noffsets[0] = 0;
  
  for(J=0;J<a_nparts;J++){
    a_noffsets[J+1] = a_noffsets[J] + a_nrowparts[J];
  }
  

  
  
  /* Setup the sparse block structure of the matrix A (nbprocs x nbprocs) */
  
  A = ABlockStruct;
  A_LD = a_nparts;
  
  /* Determine the sparse block struct of B (nbprocs x nparts) */
  
  B_size = a_nparts*b_nparts;
  B_LD = b_nparts;
  
  if ( !(B   = (int *) malloc(B_size*sizeof(int))) ) s_abort("Malloc fails for B[].");
  
  t1 = MPI_Wtime();
  preAlps_matrix_createBlockStruct(comm, MATRIX_CSC, mloc, rloc, xb, bsub, b_nparts, b_ncolparts, B);
  tBStructComp+=(MPI_Wtime() - t1);
  
  s_ivector_print_single_mp(comm, 0, B, B_size, "B", "global B blockStruct");
  
  
  /*Initialize the block column struct of C (my_rank, :)*/
  
  for(J=0;J<b_nparts;J++){      
    c_blockColumnStruct[J] = 0;      
  }
  
  /* 
   * Create a pointer to determine the beginning and the end of the rows in each block of A(my_rank, :). It follows the NIST sparse storage,  and it is also 
   * compatible with several MKL routines.
  */
  
  for(J=0;J<a_nparts;J++){
    if ( !(ptrRowBegin[J]   = (int *) malloc(mloc*sizeof(int))) ) s_abort("Malloc fails for ptrRowBegin[J]."); 
    if ( !(ptrRowEnd[J]     = (int *) malloc(mloc*sizeof(int))) ) s_abort("Malloc fails for ptrRowEnd[J].");
  }
  
  /*index table to indicate in which domain appears each rows*/
  k = 0;
  for(i=0;i<a_mparts;i++){
    for(j=0;j< a_nrowparts[i];j++){
      partcol[k] = i;
      k++;
    }
  }
  
  /*For each block column , for each rows, determine the pointer to the first and the last element */
  for(K=0;K<a_nparts;K++){
    for(j=0;j< mloc;j++){      
      ptrRowBegin[K][j] = ptrRowBegin[K][j] = -1;  
    }    
  }
  
  for (i=0; i<mloc; i++){
    for (j=xa[i]; j<xa[i+1]; j++)
    {
      J = partcol[asub[j]];
      
      if(ptrRowBegin[J][i] == -1) ptrRowBegin[J][i] = ptrRowEnd[J][i] = j; //ptrRowBegin[J][i] = asub[j];
      
      ptrRowEnd[J][i]++; //asub[j] + 1; //track the last indice of the row
    }
    }
  
  /*Finalize the rows*/
  for(K=0;K<a_nparts;K++){
    for(j=0;j< mloc;j++){      
      if(ptrRowBegin[K][j] == -1) ptrRowBegin[K][j] = ptrRowEnd[K][j] = 0;  
    }      
          
  }
  
  for(J=0;J<a_nparts;J++){
    s_int_print_mp(comm, J, "J");
    s_ivector_print_mp (comm, ptrRowBegin[J], mloc, "ptrRowBegin[J]", "ptrRowBegin[J]"); 
    s_ivector_print_mp (comm, ptrRowEnd[J], mloc, "ptrRowEnd[J]", "ptrRowEnd[J]"); 
  }
  
  
  
  /* Prepare the buffers for the irecv */
  
  for(I=0;I<a_nparts;I++){
    for(J=0;J<b_nparts;J++){      
      buffer_recv[idx(I,J)] = NULL;
      buffer_size[idx(I,J)] = a_nrowparts[I]*b_ncolparts[J];      
    }
  }
  
  /*
   * Matrix - product section
  */
  
  /* 
   * Initiate the reception of the block I need
   */
  
  I = my_rank;
  
  for(J=0;J<b_nparts;J++){
    
    /* Request block B(K,J) from all procs when required. */

    for(K=0;K<a_nparts;K++){
      
      if(A(I,K)!=0 && B(K, J)!=0 && K!=my_rank){
        
#ifdef DEBUG              
    //    printf("[%d] initiate ireceive B(%d,%d):%d from proc: %d\n", my_rank, K, J, B(K,J), K);
#endif
        
        /* Create a buffer to receive the block */
        
        if ( !(buffer_recv[idx(K,J)]   = (double *) malloc(buffer_size[idx(K,J)]*sizeof(double))) ) s_abort("Malloc fails for buffer[I,J].");
        
        //tag = idx(K,J); 
        tag = J;
		        
        MPI_Irecv(buffer_recv[idx(K,J)], buffer_size[idx(K,J)], MPI_DOUBLE, K,
                              tag, comm, &requests_recv[requests_recv_count]);
                      
        requests_recv_count++;
                      
        nrecv++;  
        
        c_blockColumnStruct[J] = 1;
        
      }
    }
  }
  
  
  /* 
   * Initiate the Send of the block the other proc need when required. 
   */
  
  
  for(J=0;J<b_nparts;J++){
        
    K = my_rank;
    /*B(K,J) where K==my_rank*/
    for(I=0;I<a_mparts;I++){
      if(A(I,K)!=0 && B(K, J)!=0 && I!=my_rank){
        
#ifdef DEBUG        
        printf("[%d] isend B(%d,%d):%d to proc: %d\n", my_rank, K, J, B(K,J), I);
#endif      
          
        //tag = idx(K,J);
        tag = J;
		
        /* send column J to proc I */
        /*TODO: fix case with multiple colums for b*/
        MPI_Isend(&b[xb[J]], buffer_size[idx(K,J)], MPI_DOUBLE, I, 
        tag, comm, &requests_send[requests_send_count]);
        
        requests_send_count++;
      }
    }

  }
  
  /*
   * Use my diagonal block A(I,I) to finalize the computation of the column block strcut of C(my_rank, :) 
   */
  
  I = my_rank;
  
  for(J=0;J<b_nparts;J++){

    K = my_rank;
    if(A(I,K)!=0 && B(K, J)!=0){
      c_blockColumnStruct[J] = 1;
    }
  }
  
  
  /*
   * Determine the required space and reserve memory for the matrix C
   */
  
  c_expected_nnz = 0;
  for(J=0;J<b_nparts;J++){
    //rloc+= b_ncolparts[J];
    if(c_blockColumnStruct[J]!=0){
      
      c_expected_nnz+= a_nrowparts[I]*b_ncolparts[J];
      
    }
  }
  
  c_nnz = 0;
  
  if(*xc!=NULL){
    
    c_nnz = (*xc)[rloc];
    
    /* The A matrix is provided, check the size */
    if( c_nnz < c_expected_nnz){
      if(*csub!=NULL) free(*csub); 
      *csub = NULL;
      if(*c!=NULL) free(*c); 
      *c = NULL;
    }
    
  }
  
  if(*xc==NULL){
    
    if ( !(*xc = (int *)   malloc((rloc+1)*sizeof( int))) ) s_abort("Malloc fails for xc[].");
  }
  
  if(*csub==NULL){
    if ( !(*csub = (int *)   malloc(c_expected_nnz*sizeof( int))) ) s_abort("Malloc fails for csub[].");
  }
  
  if(*c==NULL){
    if ( !(*c = (double *)   malloc(c_expected_nnz*sizeof( double))) ) s_abort("Malloc fails for c[].");
  }
  
  
  xc_ptr = *xc;
  c_ptr = *c;
  
  
  c_nnz = c_expected_nnz;
  
  k = 0; (*xc)[0] = 0;
  for(J=0;J<b_nparts;J++){
    for(i=0;i<b_ncolparts[J];i++){
      if(c_blockColumnStruct[J]!=0){
        (*xc)[k+1] = (*xc)[k] + a_nrowparts[I]; /* add the number of nnz for this column */
      }else{
        (*xc)[k+1] = (*xc)[k]; /* empty column */
      }
      
      k++;
    }
  }
  
  s_ivector_print_mp (comm, *xc, rloc+1, "xc", "Computed xc"); 
  
  /* 
   * Use my diagonal block A(I,I) to compute C(I,J) = C(I,J) + A(I,K)*B(K,J) when I == K ==my_rank 
   */
  
#ifdef DEBUG  
  //s_sleep(my_rank, (my_rank+1)*2);
#endif

  t1 = MPI_Wtime(); 
    
  for(J=0;J<b_nparts;J++){

    K = my_rank;
    if(A(I,K)!=0 && B(K, J)!=0){
      /* compute c(I,J) += A(I,K)*B(K,J) */    
#ifdef DEBUG
      printf("[%d] compute C(%d,%d) += A(%d,%d) x B(%d, %d)\n", my_rank, I, J, I,K,K,J);
      printf("J:%d, xc_ptr[J]:%d, c_ptr[xc_ptr[J]]\n", J, xc_ptr[J]); //(*c)[0]
#endif
      /*TODO: fix case with multiple colums for b,c and ja*/ //(*c)[xb[J]]
      
      preAlps_matrix_subMatrix_CSRDense_Product(mloc, ptrRowBegin[K], ptrRowEnd[K], a_noffsets[K], asub, a, &b[xb[J]], a_nrowparts[I], b_ncolparts[J], &c_ptr[xc_ptr[J]], a_nrowparts[I]);
      
      //fill ja
      for(i=0;i<a_nrowparts[I];i++){
        (*csub)[xc_ptr[J] + i] = i;
      }
    }
  }
  
  tLocalComp+=(MPI_Wtime() - t1);
  
  
  /* 
   * Check upcoming block and update the results
   */
  
  numDataRecv += nrecv;
  
  volDataRecv = 0;
  
  while(nrecv>0){
    
	t1 = MPI_Wtime();
	
    MPI_Waitany(requests_recv_count, requests_recv, &index, &status);
    
	tWaitRecv+=(MPI_Wtime() - t1);
	
    if(index == MPI_UNDEFINED) s_abort("MPI_Waitany error");

	K = status.MPI_SOURCE;
	 
	J = status.MPI_TAG;
	
#ifdef DEBUG    
    printf("[%d] Received index :%d, tag:%d, from :%d, K:%d, J:%d\n", my_rank, index, status.MPI_TAG, status.MPI_SOURCE, K, J);
	printf("buffer_size[K:%d,J:%d]:%d\n", K, J, buffer_size[idx(K,J)]);
    for(i=0;i<buffer_size[idx(K,J)];i++){  
      printf("[%d] buffer_recv(%d,%d)[0]:%e from proc: %d\n", my_rank, K, J, buffer_recv[idx(K,J)][i], K);
    }
    /* compute c(I,J) += A(I,K)*B(K,J) */  
    printf("[%d] compute C(%d,%d) += A(%d,%d) x B(%d, %d)\n", my_rank, I, J, I,K,K,J);
#endif

	volDataRecv+= buffer_size[idx(K,J)];
		
    t1=MPI_Wtime();    
    /*TODO: fix case with multiple columns for c*/ //(*c)[xb[J]]
    preAlps_matrix_subMatrix_CSRDense_Product(mloc, ptrRowBegin[K], ptrRowEnd[K], a_noffsets[K], asub, a, buffer_recv[idx(K,J)], a_nrowparts[I], b_ncolparts[J], &c_ptr[xc_ptr[J]], a_nrowparts[I]);
    
    //fill ja
    for(i=0;i<a_nrowparts[I];i++){
      (*csub)[xc_ptr[J] + i] = i;
    }
    
	tDepComp+=(MPI_Wtime() - t1);
	
    /* free the buffer */
    free(buffer_recv[idx(K,J)]);     
    
    buffer_recv[idx(K,J)] = NULL;
    
    nrecv--;  
  }       
  
  /* Check that all sending operations are completed*/
  t1=MPI_Wtime();
  MPI_Waitall(requests_send_count, requests_send, MPI_STATUS_IGNORE);
  tWaitSend+=(MPI_Wtime() - t1);
  
  /*Free buffer and ressources*/
  
  free(B);
  
  for(I=0;I<a_nparts;I++){
    for(J=0;J<b_nparts;J++){      
      if(buffer_recv[idx(I,J)] != NULL){
        printf("[%d] WARNING: Unhandled/freed block A(%d, %d)\n", my_rank, I, J);
        free(buffer_recv[idx(I,J)]);
      };
            
    }
  }
  
  free(a_noffsets);
  free(buffer_recv);
  free(buffer_size);
  
  free(requests_send);
  free(requests_recv);
  
  free(c_blockColumnStruct);
  
  
  for(J=0;J<a_nparts;J++){
    free(ptrRowBegin[J]);
    free(ptrRowEnd[J]);
  }
  
  free(ptrRowBegin);
  free(ptrRowEnd);
  
  free(partcol);

  tTime+=(MPI_Wtime() - tStart);
  
#ifdef SPMSV_STATS
  
  /*Display stats*/
  s_stats_display(comm, tBStructComp, "tBStructComp", tTime);
  s_stats_display(comm, tLocalComp, "tLocalComp", tTime);
  s_stats_display(comm, tDepComp, "tDepComp", tTime);
  s_stats_display(comm, tWaitRecv, "tWaitRecv", tTime);
  s_stats_display(comm, tWaitSend, "tWaitSend", tTime);
  s_stats_int_display(comm, volDataRecv, "volDataRecv");
  s_stats_int_display(comm, numDataRecv, "numDataRecv");
  
#endif
    
  /*Wait for other procs to finish their computations*/
  MPI_Barrier(comm);
 
  return ds;
}