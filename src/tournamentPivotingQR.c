
#include "tournamentPivoting.h"
#include "spTP_utils.h"
/*
 * Purpose
 * =======
 * Performs a sparse QR factorization using tournament pivoting.
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
 *         printFact (to print the matrices Q and R) and ordering (to activate METIS).
 * Outputs:
 *  Jc: vector of indexes of selected columns,
 *  Sval: vector containing the approximated k first singular values.
 */

int preAlps_tournamentPivotingQR(MPI_Comm comm, int *xa, int *ia, double *a, int m,  int n,  int nnz,
  long col_offset, int k, long *Jc, double *Sval, int printSVal, int checkFact, int printFact, int ordering){

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
  cholmod_sparse *A_test; // Only used when error analysis is required.
  cholmod_sparse *R;
  cholmod_sparse *S11;
  cholmod_sparse *Q;


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


/* Section 2. QR factorization:
 *  Perform a parallel QR factorization of the global matrix,
 *  for details about this factorization see the documentation.
 */


if(rank==0){
/*
 *  Assembling the QR factors in processor 0 (master processor).
 */

  cholmod_sparse *panel;
  panel = cholmod_l_submatrix(A_test,NULL,-1,Jc,k,1,1,cc);  /* Get panel of selected columns */

/* Calculus of S11 and Q, see documentation for details */
  status=SuiteSparseQR_C_QR (ordering, tol, panel->nrow, panel, &Q, &S11, NULL, cc) ;

/* sending the k selected columns to other processors */
for(int dest = 1; dest<size; dest++) {
    MPI_Send(&panel->nzmax,1,MPI_LONG,dest,tag,comm);
    MPI_Send(&panel->ncol,1,MPI_LONG,dest,tag,comm);
    MPI_Send(panel->p,panel->ncol+1,MPI_LONG,dest,tag,comm);
    MPI_Send(panel->i,panel->nzmax,MPI_LONG,dest,tag,comm);
    MPI_Send(panel->x,panel->nzmax,MPI_DOUBLE,dest,tag,comm);
}

/*
 * Multiplying QT*A, for notation, a letter T after a variable means transpose, QT = Q'
 * in matlab notation.
 */
  status=SuiteSparseQR_C(ordering, tol, A->nrow , 0, panel,
  A, NULL, &A, NULL, NULL, NULL, NULL, NULL, NULL, cc) ;

/* Getting local part of the matrix [S_11, S_12]*P', see documentation for details */
  A = cholmod_l_submatrix(A,kseq,k,NULL,-1,1,1,cc);

  cholmod_l_free_sparse(&panel,cc);

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


 /* Multiplying QT*A */
  status=SuiteSparseQR_C(ordering, tol, A->nrow , 0, QT,
  A, NULL, &A, NULL, NULL, NULL, NULL, NULL, NULL, cc) ;

  /* Getting local part of the matrix [S_11, S_12]*P', see documentation for details */
  A = cholmod_l_submatrix(A,kseq,k,NULL,-1,1,1,cc);

  cholmod_l_free_sparse(&QT,cc);
}


if(rank>0) {
  int src = 0;
  /* Sending local part of the matrix [S_11, S_12]*P' to master processor */
  MPI_Send(&A->nzmax,1,MPI_LONG,src,tag,comm);
  MPI_Send(&A->ncol,1,MPI_LONG,src,tag,comm);
  MPI_Send(A->p,A->ncol+1,MPI_LONG,src,tag,comm);
  MPI_Send(A->i,A->nzmax,MPI_LONG,src,tag,comm);
  MPI_Send(A->x,A->nzmax,MPI_DOUBLE,src,tag,comm);
}
 else { // Master processor
 cholmod_sparse *tempS;
 long nzmaxRecv=0,ncolRecv;


 /* Assembling matrix [S_11, S_12]*P' */
  for(int dest = 1; dest<size;dest++) {
    MPI_Recv(&nzmaxRecv,1,MPI_LONG,dest,tag,comm,&stat);
    MPI_Recv(&ncolRecv,1,MPI_LONG,dest,tag,comm,&stat);
    tempS = cholmod_l_allocate_sparse(k,ncolRecv,nzmaxRecv,1,1,0,CHOLMOD_REAL,cc);
    MPI_Recv(tempS->p,tempS->ncol+1,MPI_LONG,dest,tag,comm,&stat);
    MPI_Recv(tempS->i,tempS->nzmax,MPI_LONG,dest,tag,comm,&stat);
    MPI_Recv(tempS->x,tempS->nzmax,MPI_DOUBLE,dest,tag,comm,&stat);
    A = cholmod_l_horzcat(A,tempS,1,cc);

    cholmod_l_free_sparse(&tempS,cc);
  }

 /* Getting Q1 matrix */
  Q = cholmod_l_submatrix(Q,NULL,-1,kseq,k,1,1,cc);


  if(checkFact){
    cholmod_sparse *Aaprox;
    Aaprox = cholmod_l_ssmult(Q,A,0,1,0,cc);
    double alpha[2] = {1.0,0};
    double beta[2] = {-1.0,0};
    Aaprox = cholmod_l_add (A_test, Aaprox, alpha, beta, 1,1, cc) ;
    printf ("norm(A-Q*R) = %8.4e\n",  cholmod_l_norm_sparse (Aaprox, 0, cc)) ;
    cholmod_l_free_sparse(&Aaprox,cc);
  }

  if(printFact){
    FILE * matQ, *matR;
    matQ = fopen ("Q.mtx", "w");
    cholmod_l_write_sparse(matQ, Q, NULL, NULL,cc);
    matR = fopen ("R.mtx", "w");
    cholmod_l_write_sparse(matR, A, NULL, NULL,cc);
    fclose(matQ);
    fclose(matR);
    printf("Matrix Q and R written in Q.mtx and R.mtx respectively \n");
  }

   cholmod_l_free_sparse(&Q,cc);
   cholmod_l_free_sparse(&A_test,cc);
}



cholmod_l_free_sparse(&A,cc);
free(kseq);


  return 0;
}
