#include <stdio.h>
#include <stdlib.h>
#include <spTP_utils.h>

void preAlps_TP_parameters_display( char **matrixName, int *k, int ordering, int *printSVal, int *checkFact,
  int *printFact, int argc, char **argv){
  /* Default parameters */
  int kD=2;
  *k=kD;
  char matrixNameD[]="cage4.mtx";
  int maxSizeChar = 128;
  *matrixName=malloc(maxSizeChar*sizeof(char));
  strcpy(*matrixName,matrixNameD);

  /* MPI parameters */
  int rank,size;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);


  /* User can change inputs from the terminal */

  if(argc>1){
    if(!strcmp("-h",argv[1]) || !strcmp("--help",argv[1])){
      printf("Usage | Default values: %s [-m <matrixName : %s>] [-k <value : %d>]  [--metis] [--printSVal] [--checkFact] [--printFact]\n",
              argv[0], matrixNameD, kD);
      MPI_Finalize();
      exit(0);
    }

      for(int i=1;i<argc;i++)
        if(!strcmp("-m",argv[i])){
          i++;
          strcpy(*matrixName,argv[i]);
        }else if(!strcmp("-k",argv[i])){
          i++;
          *k=atoi(argv[i]);
        }else if(!strcmp("--printSVal",argv[i])){
          *printSVal=1;
        }else if(!strcmp("--checkFact",argv[i])){
          *checkFact=1;
        }else if(!strcmp("--printFact",argv[i])){
          *printFact=1;
        }else if(!strcmp("--metis",argv[i])){
          ordering=1;
        }else{
          printf("Error in input : argv[%d]=%s\n",i,argv[i]);
          printf("Usage | Default values: %s [-m <matrixName : %s>] [-k <value : %d>]  [--metis] [--printSVal] [--checkFact] [--printFact]\n",
                  argv[0], matrixNameD, kD);
          MPI_Abort(MPI_COMM_WORLD,1);
        }
    }

    /* Print the workspace */
  if(rank ==0 && argc > 1){
    int mkl=0;
    int lapack=0;
    #ifdef MKL
        mkl= 1;
    #endif
    #ifdef LAPACK
        lapack = 1;
    #endif

      printf("Workspace: \t Matrix: %s\t k = %d\t%s %s %s \t %s \t %s \t %s \n",
        *matrixName, *k,
        (mkl == 1 ? "with-MKL" : ""),
        (lapack == 1 ? "with-LAPACK" : ""),
        (ordering == 1 ? "\twith-METIS-ORDERING" : "\twithout-ordering"),
        (*printSVal == 1 ? "printSVal: Yes" : "printSVal: No"),
        (*checkFact == 1 ? "checkFact: Yes" : "checkFact: No"),
        (*printFact == 1 ? "printFact: Yes" : "printFact: No")
      );
  }

}


/* Distribute the global matrix among all processors */
void preAlps_TP_matrix_distribute(int rank, int size, int *row_indx, int *col_indx, double *a, int m, int n, int nnz,
  long *col_offset, cholmod_sparse **A, int checkFact, cholmod_sparse **A_test, cholmod_common *cc){
    int tag = 0;
    MPI_Status stat;

    cholmod_sparse *A_global;
    long localStart = 0;
    cholmod_l_start(cc);

    if(rank==0) {
    cholmod_triplet *T;

    long ml = (long)m;
    long nl = (long)n;
    long nnzl = (long)nnz;

    long *ix;
    long *iy;
    double *ai;
    ix =  malloc(nnzl*sizeof(long));
    iy =  malloc(nnzl*sizeof(long));
    ai =  malloc(nnzl*sizeof(double));


    int count = 0;
      for (int i=0; i<m; i++){
        for (int j=row_indx[i]; j<row_indx[i+1]; j++){
            ix[count] = col_indx[j];
            iy[count] = i;
            ai[count] = a[j];
            count +=1;
        }
      }


    T=cholmod_l_allocate_triplet(ml,nl,nnzl,0,CHOLMOD_REAL,cc);
    memcpy(T->i,ix,sizeof(long)*nnzl);
    memcpy(T->j,iy,sizeof(long)*nnzl);
    memcpy(T->x,ai,sizeof(double)*nnz);
    T->nnz=nnzl;
    A_global=cholmod_l_triplet_to_sparse(T,nnzl,cc);

    if(checkFact){
      *A_test = cholmod_l_copy_sparse(A_global,cc);
    }

    /* Print global matrix */
    cc->print = 6;
    //cholmod_l_print_sparse(A_global,"A Global matrix",cc);
    cc->print = 0;

    /* Get localStart to update columns position */
    cholmod_sparse *A_temp;
    long localSize;
    long nbProcWithExtraSize= n % size;

    long *colInd;
    colInd = malloc(sizeof(long)*n);
    for(int i=0; i<n; i++)  colInd[i] = i;

    for(int dest = 1; dest<size; dest++) {
      localSize = n / size;

      if(dest<nbProcWithExtraSize){
        localSize++;
        localStart=dest*localSize;
      }else{
        localStart=nbProcWithExtraSize*(localSize+1)+(dest-nbProcWithExtraSize)*localSize;
      }

    /* Distribute the matrix over all the processors */
      A_temp =  cholmod_l_submatrix(A_global,NULL,-1,colInd + localStart,localSize,1,1,cc);

      MPI_Send(&localStart,1,MPI_LONG,dest,tag,MPI_COMM_WORLD);

      MPI_Send(&A_temp->nzmax,1,MPI_LONG,dest,tag,MPI_COMM_WORLD);
      MPI_Send(&A_temp->ncol,1,MPI_LONG,dest,tag,MPI_COMM_WORLD);
      MPI_Send(&A_temp->nrow,1,MPI_LONG,dest,tag,MPI_COMM_WORLD);
      MPI_Send(A_temp->p,A_temp->ncol+1,MPI_LONG,dest,tag,MPI_COMM_WORLD);
      MPI_Send(A_temp->i,A_temp->nzmax,MPI_LONG,dest,tag,MPI_COMM_WORLD);
      MPI_Send(A_temp->x,A_temp->nzmax,MPI_DOUBLE,dest,tag,MPI_COMM_WORLD);
    }

    *col_offset = 0;
    localSize = n / size;
    if(nbProcWithExtraSize>0) localSize++;
    (*A) =  cholmod_l_submatrix(A_global,NULL,-1,colInd,localSize,1,1,cc);
    free(colInd);
    cholmod_l_free_sparse(&A_temp,cc);
    cholmod_l_free_sparse(&A_global,cc); // to be used in CUR
    free(row_indx);
    free(col_indx);
    free(a);
    }
    else {
      int src = 0;
      long nzmaxRecv=0,ncolRecv,nrowRecv;
      MPI_Recv(&localStart,1,MPI_LONG,src,tag,MPI_COMM_WORLD,&stat);
      *col_offset = localStart;

      MPI_Recv(&nzmaxRecv,1,MPI_LONG,src,tag,MPI_COMM_WORLD,&stat);
      MPI_Recv(&ncolRecv,1,MPI_LONG,src,tag,MPI_COMM_WORLD,&stat);
      MPI_Recv(&nrowRecv,1,MPI_LONG,src,tag,MPI_COMM_WORLD,&stat);
      *A = cholmod_l_allocate_sparse(nrowRecv,ncolRecv,nzmaxRecv,1,1,0,CHOLMOD_REAL,cc);
      MPI_Recv((*A)->p,(*A)->ncol+1,MPI_LONG,src,tag,MPI_COMM_WORLD,&stat);
      MPI_Recv((*A)->i,(*A)->nzmax,MPI_LONG,src,tag,MPI_COMM_WORLD,&stat);
      MPI_Recv((*A)->x,(*A)->nzmax,MPI_DOUBLE,src,tag,MPI_COMM_WORLD,&stat);
    }

}
