
#include "tournamentPivoting.h"
#include "spTP_utils.h"
/*
 * Purpose
 * =======
 * Performs a sparse CUR factorization using tournament pivoting.
 *
 * Arguments
 * =========
 * Inputs:
 *  xa,ia,a: vectors that define the CSC matrix (column pointers, row indexes
 *           and matrix values respectively),
 *  m,n,nnz: dimensions of the matrix,
 *  col_offset: offset of local column indexes with respect to global indexes,
 *  k: rank of the approximation,
 *  Flags: printSVal (to print the singular values), checkFact (to print the factorization error),
 *         printFact (to print the vectors Jc and Jr and the matrix U) and ordering (to activate METIS).
 * Outputs:
 *  Jr: vector of indexes of selected rows,
 *  Jc: vector of indexes of selected columns,
 *  Sval: vector containing the approximated k first singular values.
 */

int preAlps_tournamentPivotingCUR(MPI_Comm comm, int *xa, int *ia, double *a, int m,  int n,  int nnz,
  long col_offset, int k, long *Jr, long *Jc, double *Sval, int printSVal, int checkFact, int printFact, int ordering){

  /* MPI initialization */
  int rank,size;
  MPI_Status stat;
  MPI_Comm_rank(comm,&rank);
  MPI_Comm_size(comm,&size);

  /* Global variables */
  double tol=DBL_EPSILON;
  int status=0;
  int tag=0;

 /* Start cholmod */
  cholmod_common *cc,Common;
  cc = &Common;
  cc->print=0;
  cholmod_l_start(cc);
  cholmod_sparse *A;
  cholmod_sparse *panel; // sparse panel of selected columns.
  cholmod_sparse *A_test; // Only used when error analysis is required.
  cholmod_sparse *U;
  cholmod_sparse *R;
  cholmod_sparse *S11;

if(ordering){
  ordering = SPQR_ORDERING_METIS;
}
  long *spqrOrd;


  /* Permutation vectors */
  long *kseq; // linear sequence from 0 to k
  kseq  = malloc(sizeof(long)*k);
  for(long i=0;i<k;i++)  kseq[i]=i;



/* Getting A_test for error analysis */
if(rank ==0){
  preAlps_CSC_to_cholmod_l_sparse(m, n, nnz, xa, ia, a, &A_test, cc);
  n = col_offset;
  nnz= xa[col_offset];
  col_offset = 0;

/* Local Matrix in master processor */
  int *colPtrLoc = NULL, *rowIndLoc = NULL;
  double *aLoc = NULL;
  colPtrLoc = malloc((n+1)*sizeof(int));
  rowIndLoc = malloc(nnz*sizeof(int));
  aLoc = malloc(nnz*sizeof(double));

for(int i =0;i<n+1;i++)
  colPtrLoc[i]= xa[i];

for(int i =0;i<nnz;i++){
  rowIndLoc[i]= ia[i];
  aLoc[i] = a[i];
}

/* Converting local matrix in master processor to cholmod */
  preAlps_CSC_to_cholmod_l_sparse(m, n, nnz, colPtrLoc, rowIndLoc, aLoc, &A, cc);

  free(colPtrLoc); free(rowIndLoc); free(aLoc);
}

/* Converting local matrices to cholmod */
if( rank > 0){
  preAlps_CSC_to_cholmod_l_sparse(m, n, nnz, xa, ia, a, &A, cc);
}

/* Let's begin*/

/* Section 1: Performing tournament pivoting */

double t_begin, t_tp;
t_begin=MPI_Wtime();
preAlps_tournamentPivoting(comm,xa,ia,a,m,n,nnz,col_offset,k,Jc,Sval,printSVal,ordering);
t_tp = MPI_Wtime()-t_begin;


/* Section 2. CUR factorization:
 *  Perform a CUR factorization of the global matrix in parallel,
 *  for details about this factorization see the the documentation.
 */

/* vector of selected rows */
 #ifdef MKL
   MKL_INT *JJr;
   JJr  = calloc(k,sizeof(MKL_INT));
 #elif defined(LAPACK)
   lapack_int *JJr;
   JJr  = calloc(k,sizeof(lapack_int));
 #endif

if(rank==0){
/*
 *  Assembling the CUR factors is done in processor 0 (master processor).
 *  Columns selected from the tournament pivoting are used as the columns selected for
 *  the CUR factorization, C = A_global(:,Jc) in matlab notation
 */
int info=0;
 cholmod_sparse *CT;
 cholmod_dense *Rdense;
 panel = cholmod_l_submatrix(A_test,NULL,-1,Jc,k,1,1,cc);  /* Get panel of selected columns */

/* Calculus of S11 and Q, see the documentation for details */
  status=SuiteSparseQR_C_QR (ordering, tol, panel->nrow, panel, NULL, &S11, NULL, cc) ;

/* sending matrix C to other processors */
for(int dest = 1; dest<size; dest++) {
    MPI_Send(&panel->nzmax,1,MPI_LONG,dest,tag,comm);
    MPI_Send(&panel->ncol,1,MPI_LONG,dest,tag,comm);
    MPI_Send(panel->p,panel->ncol+1,MPI_LONG,dest,tag,comm);
    MPI_Send(panel->i,panel->nzmax,MPI_LONG,dest,tag,comm);
    MPI_Send(panel->x,panel->nzmax,MPI_DOUBLE,dest,tag,comm);
}

 /* Getting the vector of selected rows, see the documentation for details  */
  CT = cholmod_l_transpose(panel,2,cc);
  Rdense = cholmod_l_sparse_to_dense(CT,cc);
  cholmod_l_free_sparse(&CT,cc);

  int pivSize = Rdense->ncol;
  double *tau_tmp;
  tau_tmp = malloc(pivSize*sizeof(double));

#ifdef MKL
  MKL_INT *e_tmp;
  e_tmp  = calloc(pivSize,sizeof(MKL_INT));
#elif defined(LAPACK)
  lapack_int *e_tmp;
  e_tmp  = calloc(pivSize,sizeof(lapack_int));
#endif

  info = LAPACKE_dgeqp3(LAPACK_COL_MAJOR,Rdense->nrow,Rdense->ncol,Rdense->x,Rdense->nrow,e_tmp,tau_tmp);

  cholmod_l_free_dense(&Rdense,cc);

#ifdef MKL
  memcpy(JJr,e_tmp,k*sizeof(MKL_INT));
#elif defined(LAPACK)
  memcpy(JJr,e_tmp,k*sizeof(lapack_int));
#endif

/* Sending Jrr to the other processors  */
for(int dest = 1; dest<size; dest++) {
  MPI_Send(JJr,k,MPI_INT,dest,tag,comm);
}

/* Vector Jr of selected rows, in C-format */
for(int i =0; i<k; i++) {
    Jr[i] = JJr[i]-1;
}

/* Getting R local matrix, R = A(Jr,:) in matlab notation */
  R = cholmod_l_submatrix(A,Jr,k,NULL,-1,1,1,cc);


/*
 * Multiplying QT*A, Q is the Q factor from the QR factorization of C,
 * for notation, a letter T after a variable means transpose, QT = Q'
 * in matlab notation.
 */
  status=SuiteSparseQR_C(ordering, tol, A->nrow , 0, panel,
  A, NULL, &A, NULL, NULL, NULL, NULL, NULL, NULL, cc) ;

/* Getting the local VT matrix, see the documentation for details */
  A = cholmod_l_submatrix(A,kseq,k,NULL,-1,1,1,cc);

  free(tau_tmp);
  free(e_tmp);
}

else {
/* Processor receiving from the master processor */
  int src = 0;
  cholmod_sparse *QT;
  long nzmaxRecv=0,ncolRecv;

/* Receiving QT matrix */
  MPI_Recv(&nzmaxRecv,1,MPI_LONG,src,tag,comm,&stat);
  MPI_Recv(&ncolRecv,1,MPI_LONG,src,tag,comm,&stat);
  QT = cholmod_l_allocate_sparse(A->nrow,ncolRecv,nzmaxRecv,1,1,0,CHOLMOD_REAL,cc);
  MPI_Recv(QT->p,QT->ncol+1,MPI_LONG,src,tag,comm,&stat);
  MPI_Recv(QT->i,QT->nzmax,MPI_LONG,src,tag,comm,&stat);
  MPI_Recv(QT->x,QT->nzmax,MPI_DOUBLE,src,tag,comm,&stat);

 /* Receiving Jr vector */
  MPI_Recv(JJr,k,MPI_INT,src,tag,comm,&stat);

 /* Getting the R local matrix */
 for(int i =0; i<k; i++) {
   Jr[i] = JJr[i]-1;
 }

 R = cholmod_l_submatrix(A,Jr,k,NULL,-1,1,1,cc);


 /* Multiplying QT*A */
  status=SuiteSparseQR_C(ordering, tol, A->nrow , 0, QT,
  A, NULL, &A, NULL, NULL, NULL, NULL, NULL, NULL, cc) ;

 /* Getting the local VT matrix, see the documentation for details */
  A = cholmod_l_submatrix(A,kseq,k,NULL,-1,1,1,cc);

  cholmod_l_free_sparse(&QT,cc);
}


if(rank>0) {
  int src = 0;
  /* Sending local VT matrix to master processor */
  MPI_Send(&A->nzmax,1,MPI_LONG,src,tag,comm);
  MPI_Send(&A->ncol,1,MPI_LONG,src,tag,comm);
  MPI_Send(A->p,A->ncol+1,MPI_LONG,src,tag,comm);
  MPI_Send(A->i,A->nzmax,MPI_LONG,src,tag,comm);
  MPI_Send(A->x,A->nzmax,MPI_DOUBLE,src,tag,comm);

  tag = 1;
  /* Sending local R matrix to master processor */
  MPI_Send(&R->nzmax,1,MPI_LONG,src,tag,comm);
  MPI_Send(&R->ncol,1,MPI_LONG,src,tag,comm);
  MPI_Send(R->p,R->ncol+1,MPI_LONG,src,tag,comm);
  MPI_Send(R->i,R->nzmax,MPI_LONG,src,tag,comm);
  MPI_Send(R->x,R->nzmax,MPI_DOUBLE,src,tag,comm);


}
 else { // Master processor
 cholmod_sparse *tempVT;
 cholmod_sparse *tempR;
 long nzmaxRecv=0,ncolRecv;

 /* Receiving local VT matrix from other processors */
  for(int dest = 1; dest<size;dest++) {
    MPI_Recv(&nzmaxRecv,1,MPI_LONG,dest,tag,comm,&stat);
    MPI_Recv(&ncolRecv,1,MPI_LONG,dest,tag,comm,&stat);
    tempVT = cholmod_l_allocate_sparse(k,ncolRecv,nzmaxRecv,1,1,0,CHOLMOD_REAL,cc);
    MPI_Recv(tempVT->p,tempVT->ncol+1,MPI_LONG,dest,tag,comm,&stat);
    MPI_Recv(tempVT->i,tempVT->nzmax,MPI_LONG,dest,tag,comm,&stat);
    MPI_Recv(tempVT->x,tempVT->nzmax,MPI_DOUBLE,dest,tag,comm,&stat);
    A = cholmod_l_horzcat(A,tempVT,1,cc);
  }

 /* Calculating the S11 matrix, see the documentation for details */
 S11 = cholmod_l_submatrix(S11,kseq,k,NULL,-1,1,1,cc);

 /* Calculating the global VT matrix, see the documentation for details */
 A = SuiteSparseQR_C_backslash_sparse(0,tol,S11,A,cc);

 tag = 1;
 /* Receiving local R matrix from other processors and asembling the global R matrix */

 for(int dest = 1; dest<size;dest++) {
    MPI_Recv(&nzmaxRecv,1,MPI_LONG,dest,tag,comm,&stat);
    MPI_Recv(&ncolRecv,1,MPI_LONG,dest,tag,comm,&stat);
    tempR = cholmod_l_allocate_sparse(k,ncolRecv,nzmaxRecv,1,1,0,CHOLMOD_REAL,cc);
    MPI_Recv(tempR->p,tempR->ncol+1,MPI_LONG,dest,tag,comm,&stat);
    MPI_Recv(tempR->i,tempR->nzmax,MPI_LONG,dest,tag,comm,&stat);
    MPI_Recv(tempR->x,tempR->nzmax,MPI_DOUBLE,dest,tag,comm,&stat);
    R = cholmod_l_horzcat(R,tempR,1,cc);
}

 /* Calculating the U matrix, see the documentation for details */
  R = cholmod_l_transpose(R,2,cc); // RT
  A = cholmod_l_transpose(A,2,cc); // V
  U = SuiteSparseQR_C_backslash_sparse(0,tol,R,A,cc);
  U = cholmod_l_transpose(U,2,cc);


  if(checkFact){
    cholmod_sparse *Aaprox;
    R = cholmod_l_transpose(R,2,cc);
    Aaprox = cholmod_l_ssmult(panel,U,0,1,0,cc);
    Aaprox = cholmod_l_ssmult(Aaprox,R,0,1,0,cc);
    double alpha[2] = {1.0,0};
    double beta[2] = {-1.0,0};
    Aaprox = cholmod_l_add (A_test, Aaprox, alpha, beta, 1,1, cc) ;
    printf ("norm(A-C*U*R) = %8.4e\n",  cholmod_l_norm_sparse (Aaprox, 0, cc)) ;
    cholmod_l_free_sparse(&Aaprox,cc);
  }

  if(printFact){
    FILE * matU;
    matU = fopen ("U.mtx", "w");
    printf("Matrix U written in U.mtx \n");
    cholmod_l_write_sparse(matU, U, NULL, NULL,cc);
    fclose(matU);
  }

  cholmod_l_free_sparse(&A_test,cc);
  cholmod_l_free_sparse(&tempVT,cc);
  cholmod_l_free_sparse(&tempR,cc);
  cholmod_l_free_sparse(&U,cc);
}


cholmod_l_free_sparse(&R,cc);
cholmod_l_free_sparse(&A,cc);
free(kseq);


  return 0;
}
