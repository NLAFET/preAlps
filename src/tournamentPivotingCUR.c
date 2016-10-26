
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
 *  xa,ia,a: vectors that define the CSC matrix (Columns pointer, row indexes
 *           and matrix values respectively),
 *  m,n,nnz: dimensions of the matrix,
 *  k: rank of the approximation,
 *  Flags: printSVal (to print the singular values), checkFact (to print the factorization error),
 *         printFact (to print the vectors Jc and Jr and the matrix U) and ordering (to activate METIS).
 * Outputs:
 *  Jr: vector of indexes of selected rows,
 *  Jc: vector of indexes of selected columns,
 *  Sval: vector containing the approximated k first singular values.
 */

int preAlps_tournamentPivotingCUR(MPI_Comm comm, int *xa, int *ia, double *a, int m,  int n,  int nnz,
  long col_offset, int k, long *Jr, long *Jc, double **Sval, int printSVal, int checkFact, int printFact, int ordering){

  /* MPI initialization */
  int rank,size;
  MPI_Status stat;
  MPI_Comm_rank(comm,&rank);
  MPI_Comm_size(comm,&size);

  /* Global variables */
  double tol=DBL_EPSILON;
  int status=0;

 /* Start cholmod */
  cholmod_common *cc,Common;
  cc = &Common;
  cc->print=0;
  cholmod_l_start(cc);
  cholmod_sparse *A;
  cholmod_sparse *A_test; // Only used when error analysis is required
  cholmod_sparse *U;


if(ordering){
  ordering = SPQR_ORDERING_METIS;
}
  long *spqrOrd;


  cholmod_sparse *R;
  cholmod_dense *Rdense;
  cholmod_sparse *S11;

  /* Panel and their indexes */
  cholmod_sparse *tempPanel;
  long *panelInd,panelIndSize;
  long *bestCol,bestColSize;
  long *colPos, colPosSize;
  int npanel = 0;
  int remainCols = 0;
  long panelSize=0, ncol=0;

  /* Permutation vectors */
  long *kseq; // linear sequence from 0 to k
  kseq  = malloc(sizeof(long)*k);
  for(long i=0;i<k;i++)  kseq[i]=i;
  double *tau;


  /*  MKL-LAPACK variables */
#ifdef MKL
  MKL_INT info;
  MKL_INT *pivot, pivotSize;
  pivotSize = 2*k;
  pivot   = calloc(pivotSize,sizeof(MKL_INT));
#elif defined(LAPACK)
  lapack_int info;
  lapack_int *pivot, pivotSize;
  pivotSize = 2*k;
  pivot   = calloc(pivotSize,sizeof(lapack_int));
#endif


/* Getting A_test for error analysis */
if(checkFact==1 && rank ==0){
  preAlps_CSC_to_cholmod_l_sparse(m, n, nnz, xa, ia, a, &A_test, cc);
  n = col_offset;
  nnz= xa[col_offset];
  col_offset = 0;
}


/* Getting local matrix in cholmod */
preAlps_CSC_to_cholmod_l_sparse(m, n, nnz, xa, ia, a, &A, cc);




 /* Let's begin*/
 /* Section 1:  initialization of common variables */

  ncol = A->ncol;
  remainCols = ncol % k;
  npanel = ncol / k + (remainCols?1:0);

  tau = malloc(pivotSize*sizeof(double));
  colPosSize  = MAX(2*k,ncol);
  bestColSize = MAX(2*k,ncol);
  colPos  = malloc(sizeof(long)*colPosSize);
  bestCol = malloc(sizeof(long)*bestColSize);

  for(long i=0;i<colPosSize ;i++)
    colPos[i]=i;

 /* 2. Local Flat Tree:
  *     Inside each processor a flat tree is performed to select a local set of k pivot columns
  */

 /* Divide the local matrix in panels */
  panelIndSize = npanel+1;
  panelInd=malloc(sizeof(long)*panelIndSize);

  panelInd[0]=0;
  for(long i=1;i<panelIndSize-1;i++){
      panelInd[i]=panelInd[i-1]+k;
  }

  panelInd[panelIndSize-1]=ncol;

 /* Begin of the flat tree:
  * At each node of the tree, two panels are compared and the best k pivot columns are obtained
  */
  for(long i=0;i<panelIndSize-2;i++){

  panelSize=panelInd[i+2]-panelInd[i]; // Getting the size of the merged panel

  /* Get 2 panels and merge them into a temporal panel */
  tempPanel=cholmod_l_submatrix(A,NULL,-1,colPos+k*i,panelSize,1,1,cc);

  /* Get the QR-sparse factorization of the merged panel */
  if(ordering){
    status=SuiteSparseQR_C_QR (ordering, tol, panelSize, tempPanel, NULL, &R, &spqrOrd, cc) ;
  }else{
    status=SuiteSparseQR_C_QR (ordering, tol, panelSize, tempPanel, NULL, &R, NULL, cc) ;
  }

  cholmod_l_free_sparse(&tempPanel,cc);
  Rdense = cholmod_l_sparse_to_dense(R,cc);
  cholmod_l_free_sparse(&R,cc);

  /* Get the best k pivot columns using a QR factorization */
  info = LAPACKE_dgeqp3(LAPACK_COL_MAJOR,Rdense->nrow,Rdense->ncol,Rdense->x,Rdense->nrow,pivot,tau);
  cholmod_l_free_dense(&Rdense,cc);

  /* Copy permutation of columns into bestCol */

  for(long j=0;j<panelSize;j++){
      if(ordering){
      bestCol[j]=colPos[panelInd[i] + spqrOrd[pivot[j]-1]];
      }else{
      bestCol[j]=colPos[panelInd[i] + pivot[j]-1];
      }
  }

  /*colPos saves the pivot of this iteration to form the next tree node in the next iteration*/
  if(i+1<panelIndSize-2){
  memcpy(colPos+k*(i+1),bestCol,k*sizeof(long));
  }

  /* Setting to 0 for the next call of dgeqp3 */
#ifdef MKL
  memset(pivot,0,sizeof(MKL_INT)*panelSize);ASSERT(panelSize<=pivotSize);
#elif defined(LAPACK)
  memset(pivot,0,sizeof(lapack_int)*panelSize);ASSERT(panelSize<=pivotSize);
#endif

  } // End of the flat tree.

  /* colPos gets the final k pivot indexes selected in the current processor */
if(panelIndSize>2){
  memcpy(colPos,bestCol,k*sizeof(long));
}

/* 3. Global Binary Tree:
 *     Perform a binary tree between all processors to select the global set of k pivot columns
 */

  /* initialization of variables that participate in the binary tree */
  int tag = 0, Pcontrol = 0, complement = 0;
  cholmod_sparse *panel,*panelRecv;
  long *seq, seqSize, ncolNextIt, *perm, permSize;

  Pcontrol = size;
  seqSize = pivotSize;
  permSize = pivotSize;

  perm  = malloc(sizeof(long)*permSize);
  seq   = malloc(sizeof(long)*seqSize);
  for(long i=0;i<seqSize;i++) {
    seq[i]=i;
  }

  /* panel gets the k local pivot columns to send to other processors */
  if(k< A->ncol){
    panel = cholmod_l_submatrix(A,NULL,-1,colPos,k,1,1,cc);ASSERT(k<=colPosSize);
  }else
    panel = A;

  /* change colPos values from local to global */
  for(long i=0;i<k;i++){
    colPos[i]+=col_offset;
  }

  /* Begin of the binary tree:
   * At each node of the tree, two panels are compared and the best k pivot columns are obtained
   */
  while(Pcontrol>1){ // Pcontrol handles movement on the levels of the tree

    complement = Pcontrol - rank - 1;
    ASSERT(complement<size);

  if(rank>Pcontrol)
    break;

  /*  Processors that send the panel */
  if(complement<rank){
    int dest=complement;

  /* Send panel */
      MPI_Send(&panel->nzmax,1,MPI_LONG,dest,tag,comm);
      MPI_Send(&panel->ncol,1,MPI_LONG,dest,tag,comm);
      MPI_Send(panel->p,panel->ncol+1,MPI_LONG,dest,tag,comm);
      MPI_Send(panel->i,panel->nzmax,MPI_LONG,dest,tag,comm);
      MPI_Send(panel->x,panel->nzmax,MPI_DOUBLE,dest,tag,comm);

  /* Send colPos */
      MPI_Send(colPos,panel->ncol,MPI_LONG,dest,tag,comm);

  if(k < A->ncol)
    cholmod_l_free_sparse(&panel,cc);
  break;
    }
  /*  Processors that receive the panel */
  else if(complement>rank){

    int src=complement;
    long nzmaxRecv=0,ncolRecv;

  /* Receive panel */
      MPI_Recv(&nzmaxRecv,1,MPI_LONG,src,tag,comm,&stat);
      MPI_Recv(&ncolRecv,1,MPI_LONG,src,tag,comm,&stat);
      panelRecv = cholmod_l_allocate_sparse(panel->nrow,ncolRecv,nzmaxRecv,1,1,0,CHOLMOD_REAL,cc);
      MPI_Recv(panelRecv->p,panelRecv->ncol+1,MPI_LONG,src,tag,comm,&stat);
      MPI_Recv(panelRecv->i,panelRecv->nzmax,MPI_LONG,src,tag,comm,&stat);
      MPI_Recv(panelRecv->x,panelRecv->nzmax,MPI_DOUBLE,src,tag,comm,&stat);

  /* Receive colPos */
      MPI_Recv(colPos+panel->ncol,panelRecv->ncol,MPI_LONG,src,tag,comm,&stat);ASSERT(panel->ncol+panelRecv->ncol<=colPosSize);

  /* Create a tree node conformed by merging the 2 panels (analogous to the flat tree case) */
    tempPanel=cholmod_l_horzcat(panel,panelRecv,1,cc);
    panelSize=panel->ncol+panelRecv->ncol;

  if(panelSize>=k){
    if(ordering){
      status=SuiteSparseQR_C_QR (ordering, tol, panelSize, tempPanel, NULL, &R, &spqrOrd, cc) ;
    }else{
      status=SuiteSparseQR_C_QR (ordering, tol, panelSize, tempPanel, NULL, &R, NULL, cc) ;
    }

    Rdense = cholmod_l_sparse_to_dense(R,cc);
    cholmod_l_free_sparse(&R,cc);

    info = LAPACKE_dgeqp3(LAPACK_COL_MAJOR,Rdense->nrow,Rdense->ncol,Rdense->x,Rdense->nrow,pivot,tau);

  /*
   * If this is the final node of the tree we have the k pivot columns selected, we can get the svd
   * get the svd aproximation from the QR factorization that we have already performed on the node.
   */
    if(Pcontrol/2+Pcontrol%2<=1 && rank == 0 )
      if(printSVal){
        double *val = (double *)Rdense->x;
        for(int i=0;i<k;i++){
          ASSERT(i*(Rdense->nrow+1)<Rdense->nzmax);
          (*Sval)[i]=val[i*(Rdense->nrow+1)];
        }
      }

    cholmod_l_free_dense(&Rdense,cc);

  /* selected columns and permutation vector of the current node  */
  for(long j=0;j<panelSize;j++){
    if(ordering){
      bestCol[j]  = colPos[spqrOrd[pivot[j]-1]];
      perm[j]     = spqrOrd[pivot[j]-1];
    }else{
      bestCol[j]  = colPos[pivot[j]-1];
      perm[j]     = pivot[j]-1;
    }
  }

  /* Setting to 0 for the next call of dgeqp3 */
#ifdef MKL
        memset(pivot,0,sizeof(MKL_INT)*panelSize);ASSERT(panelSize<=pivotSize);
#elif defined(LAPACK)
        memset(pivot,0,sizeof(lapack_int)*panelSize);ASSERT(panelSize<=pivotSize);
#endif

  /* colPos gets the k pivot indexes selected in the current node */
  memcpy(colPos,bestCol,k*sizeof(long));ASSERT(k<=colPosSize);
  ncolNextIt=k; // prepare for the next level of the tree

}
else {
  ncolNextIt=panelSize;
}

if(panelSize<k){
  panel=cholmod_l_submatrix(tempPanel,NULL,-1,seq,ncolNextIt,1,1,cc);
}
else{
  panel=cholmod_l_submatrix(tempPanel,NULL,-1,perm,ncolNextIt,1,1,cc);
}
  cholmod_l_free_sparse(&tempPanel,cc);

  }

    Pcontrol=Pcontrol/2+Pcontrol%2;

} // End of the binary tree.


/* Free memory */

if(ordering){
  if(panelIndSize<=2){ /* When the flat tree is not performed (k=A->ncol) */
      if(rank<size/2)  free(spqrOrd);
   }
   else{
     free(spqrOrd);
   }
}

/* 4. CUR factorization:
 *  Perform a CUR factorization of the global matrix in parallel,
 *  for details about this factorization see the User Guide.
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
 *  The assembly of the CUR factors is done in processor 0 (master processor).
 *  Columns selected from the tournament pivoting are used as the columns selected for
 *  the CUR factorization, C = A_global(:,Jc) in matlab notation
 */
  memcpy(Jc,colPos,k*sizeof(long));   /* Get vector of selected columns */
  cholmod_sparse *CT;

/* Calculus of S11 and Q, see User Guide for details */
  status=SuiteSparseQR_C_QR (ordering, tol, panel->nrow, panel, NULL, &S11, NULL, cc) ;

/* sending matrix C to other processors */
for(int dest = 1; dest<size; dest++) {
    MPI_Send(&panel->nzmax,1,MPI_LONG,dest,tag,comm);
    MPI_Send(&panel->ncol,1,MPI_LONG,dest,tag,comm);
    MPI_Send(panel->p,panel->ncol+1,MPI_LONG,dest,tag,comm);
    MPI_Send(panel->i,panel->nzmax,MPI_LONG,dest,tag,comm);
    MPI_Send(panel->x,panel->nzmax,MPI_DOUBLE,dest,tag,comm);
}

 /* Getting the vector of selected rows, see User Guide for details  */
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

/* Getting the local VT matrix, see User Guide for details */
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

 /* Getting the local VT matrix, see User Guide for details */
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

 /* Calculating the S11 matrix, see User Guide for details */
 S11 = cholmod_l_submatrix(S11,kseq,k,NULL,-1,1,1,cc);

 /* Calculating the global VT matrix, see User Guide for details */
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

 /* Calculating the U matrix, see User Guide for details */
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
    cholmod_l_free_sparse(&A_test,cc);
    cholmod_l_free_sparse(&Aaprox,cc);
  }

  if(printFact){
    FILE * matU;
    matU = fopen ("U.mtx", "w");
    printf("Matrix U written in U.mtx \n");
    cholmod_l_write_sparse(matU, U, NULL, NULL,cc);
    fclose(matU);
  }
  cholmod_l_free_sparse(&tempVT,cc);
  cholmod_l_free_sparse(&tempR,cc);
  cholmod_l_free_sparse(&U,cc);
}


cholmod_l_free_sparse(&R,cc);
cholmod_l_free_sparse(&A,cc);
free(kseq);
free(tau);
free(colPos);
free(bestCol);
free(panelInd);
free(seq);
free(perm);

  return 0;
}
