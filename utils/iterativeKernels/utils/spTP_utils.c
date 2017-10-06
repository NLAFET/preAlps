#include <stdio.h>
#include <stdlib.h>
#include <spTP_utils.h>

void preAlps_TP_parameters_display(MPI_Comm comm, char **matrixName, int *k, int ordering, int *printSVal, int *checkFact,
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
  MPI_Comm_rank(comm,&rank);
  MPI_Comm_size(comm,&size);


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
          MPI_Abort(comm,1);
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


/* Change the matrix format to cholmod sparse long format */
void preAlps_CSC_to_cholmod_l_sparse(int m, int n, int nnz, int *colPtr, int *rowInd, double *a, cholmod_sparse **A, cholmod_common *cc){
  long *Arow, *Acol;

long alan = (long)nnz;

  (*A) = cholmod_l_allocate_sparse((long)m,(long)n,(long)nnz,1,1,0,CHOLMOD_REAL,cc);

  Acol=malloc((n+1)*sizeof(long));
  Arow=malloc(nnz*sizeof(long));

  for(int i=0;i< n+1;i++)
    Acol[i]=(long)colPtr[i];
  for(int i=0;i< nnz;i++)
    Arow[i]=(long)rowInd[i];

  memcpy((*A)->p,Acol,(n+1)*sizeof(long));
  memcpy((*A)->i,Arow,(nnz)*sizeof(long));
  memcpy((*A)->x,a,(nnz)*sizeof(double));

  (*A)->nzmax=(long)(nnz);

}


void preAlps_l_CSC_to_cholmod_l_sparse(long m, long n, long nnz, long *colPtr, long *rowInd, double *a, cholmod_sparse **A, cholmod_common *cc){

  (*A) = cholmod_l_allocate_sparse(m,n,nnz,1,1,0,CHOLMOD_REAL,cc);

  memcpy((*A)->p,colPtr,(n+1)*sizeof(long));
  memcpy((*A)->i,rowInd,(nnz)*sizeof(long));
  memcpy((*A)->x,a,(nnz)*sizeof(double));

  (*A)->nzmax=nnz;



}

/* Distribute the global matrix among all processors */
void preAlps_spTP_distribution(MPI_Comm comm, int *m, int *n, int *nnz, int **colPtr, int **rowInd,
  double **a, long *col_offset, int checkFact){

  long localStart = 0;
  int localNCol = 0;
  int neb; // Number of non-zero elements per in each processor
  int nrows;
  int rank,size;
  MPI_Comm_rank(comm,&rank);
  MPI_Comm_size(comm,&size);
  MPI_Status stat;
  int tag = 0;
  int *colPtrLoc = NULL, *rowIndLoc = NULL;
  double *aLoc = NULL;

/* Distribute among all processors */
if(rank ==0){

  nrows = *m; // same number of rows for all processors
  int nbProcWithExtraSize= *n % size;

  for(int dest = 1; dest<size; dest++) {
    localNCol = *n / size;
    if(dest<nbProcWithExtraSize){
      localNCol++;
      localStart=dest*localNCol;
    }else{
      localStart=nbProcWithExtraSize*(localNCol+1)+(dest-nbProcWithExtraSize)*localNCol;
     }

     neb = (*colPtr)[localStart+localNCol] - (*colPtr)[localStart];
     colPtrLoc = malloc((localNCol+1)*sizeof(int));
     rowIndLoc = malloc(neb*sizeof(int));
     aLoc = malloc(neb*sizeof(double));


     for(int i =0;i<localNCol+1;i++)
       colPtrLoc[i]=(*colPtr)[i+localStart]-(*colPtr)[localStart];

     for(int i =0;i<neb;i++){
         rowIndLoc[i]= (*rowInd)[i + (*colPtr)[localStart]];
         aLoc[i] = (*a)[i + (*colPtr)[localStart]];
     }

    MPI_Send(&nrows,1,MPI_INT,dest,tag,comm);
    MPI_Send(&localStart,1,MPI_LONG,dest,tag,comm);
    MPI_Send(&localNCol,1,MPI_INT,dest,tag,comm);
    MPI_Send(&neb,1,MPI_INT,dest,tag,comm);

    MPI_Send(colPtrLoc,localNCol+1,MPI_INT,dest,tag,comm);
    MPI_Send(rowIndLoc,neb,MPI_INT,dest,tag,comm);
    MPI_Send(aLoc,neb,MPI_DOUBLE,dest,tag,comm);

    free(colPtrLoc); free(rowIndLoc); free(aLoc);
  }

  /* Parameters of submatrix in processor 0 */

  localNCol = *n / size;
  if(nbProcWithExtraSize>0) localNCol = localNCol + 1;

  *col_offset = localNCol;


} else {
  /* Processor receiving local matrix from the master */
        int src = 0;
        MPI_Recv(&nrows,1,MPI_INT,src,tag,comm,&stat);
        *m = nrows;
        MPI_Recv(&localStart,1,MPI_LONG,src,tag,comm,&stat);
        *col_offset = localStart;
        MPI_Recv(&localNCol,1,MPI_INT,src,tag,comm,&stat);
        *n = localNCol;
        MPI_Recv(&neb,1,MPI_INT,src,tag,comm,&stat);
        *nnz = neb;

        colPtrLoc = malloc((localNCol+1)*sizeof(int));
        rowIndLoc = malloc(neb*sizeof(int));
        aLoc = malloc(neb*sizeof(double));

        *colPtr = NULL; *rowInd = NULL; *a = NULL;
        *colPtr = malloc((localNCol+1)*sizeof(int));
        *rowInd = malloc(neb*sizeof(int));
        *a = malloc(neb*sizeof(double));

        MPI_Recv(colPtrLoc,localNCol+1,MPI_INT,src,tag,comm,&stat);
        MPI_Recv(rowIndLoc,neb,MPI_INT,src,tag,comm,&stat);
        MPI_Recv(aLoc,neb,MPI_DOUBLE,src,tag,comm,&stat);

        memcpy((*colPtr),colPtrLoc,(localNCol+1)*sizeof(int));
        memcpy((*rowInd),rowIndLoc,neb*sizeof(int));
        memcpy((*a),aLoc,neb*sizeof(double));
    }
}
