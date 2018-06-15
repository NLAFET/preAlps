/*
* This file contains functions used to manipulate dense matrices
*
* Authors : Sebastien Cayrols
*         : Olivier Tissot
* Email   : sebastien.cayrols@[(gmail.com) | (inria.fr)]
*         : olivier.tissot@inria.fr
*/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <preAlps_cplm_utils.h>
#include <preAlps_cplm_timing.h>
#include <preAlps_cplm_matdense.h>
/* MPI Utils */
/*
//define the number of champ in the structure Partial_CPLM_Mat_Dense_t
#define _NB_CHAMPS_DENSE 7

MPI_Datatype initMPI_StructDenseInfo(){
  CPLM_Info_Dense_t info;

  MPI_Datatype MPI_DENSE_INFO;
  MPI_Datatype type[_NB_CHAMPS_DENSE] = {MPI_INT,//M
                                         MPI_INT,//N
                                         MPI_INT,//m
                                         MPI_INT,//n
                                         MPI_INT,//lda
                                         MPI_INT,//nval
                                         MPI_INT//type has to change and check with right struct called CPLM_storage_type_t
                                 };

  int blocklen[_NB_CHAMPS_DENSE];
  for(int i=0;i<_NB_CHAMPS_DENSE;i++)
    blocklen[i]=1;

  MPI_Aint disp[_NB_CHAMPS_DENSE];
  MPI_Aint addr[_NB_CHAMPS_DENSE+1];

  MPI_Get_address(&info,           &addr[0]);
  MPI_Get_address(&info.M,         &addr[1]);
  MPI_Get_address(&info.N,         &addr[2]);
  MPI_Get_address(&info.m,         &addr[3]);
  MPI_Get_address(&info.n,         &addr[4]);
  MPI_Get_address(&info.lda,       &addr[5]);
  MPI_Get_address(&info.nval,      &addr[6]);
  MPI_Get_address(&info.stor_type, &addr[7]);

  for(int i=0;i<_NB_CHAMPS_DENSE;i++)
    disp[i] = addr[i+1] - addr[0];

  MPI_Type_create_struct(_NB_CHAMPS_DENSE,blocklen,disp,type,&MPI_DENSE_INFO);
  MPI_Type_commit(&MPI_DENSE_INFO);

  return MPI_DENSE_INFO;
}
*/

int CPLM_MatDenseCalloc(CPLM_Mat_Dense_t *A_io)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  A_io->val = (double*) calloc(A_io->info.nval, sizeof(double));

CPLM_END_TIME
CPLM_POP
  return !(A_io->val != NULL);
}

void CPLM_MatDenseFree(CPLM_Mat_Dense_t  *A_io)
{
CPLM_PUSH
  int ierr = 0;

  if(A_io)
  {
    if(A_io->val)
    {
      free(A_io->val);
    }
    A_io->val = NULL;
    ierr = CPLM_MatDenseReset(A_io);
    CPLM_ASSERT( ierr == 0);
  }
CPLM_POP
}

int CPLM_MatDenseISendData(CPLM_Mat_Dense_t* A_in, int dest, int tag, MPI_Comm comm, MPI_Request **request)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  int ierr = 0;
  if(*request == NULL)
  {
    *request = (MPI_Request *)malloc(1*sizeof(MPI_Request));
    CPLM_ASSERT(*request != NULL);
  }

  // Send data
  //CPLM_debug("Will send %d data\n",A_in->info.m*A_in->info.n);
  ierr = MPI_Isend(A_in->val,(A_in->info.m)*(A_in->info.n),MPI_DOUBLE,dest,tag,comm,*request);
  CPLM_checkMPIERR(ierr,"send_vec");
CPLM_END_TIME
CPLM_POP
  return ierr;
}

int CPLM_MatDenseMalloc( CPLM_Mat_Dense_t *A_io)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  A_io->val = (double*) malloc(A_io->info.nval * sizeof(double));

CPLM_END_TIME
CPLM_POP
  return !(A_io->val != NULL);
}


int CPLM_MatDenseSetInfo(CPLM_Mat_Dense_t* A_out, int M, int N, int m, int n, CPLM_storage_type_t storage)
{
  A_out->info.M = M;
  A_out->info.N = N;
  A_out->info.m = m;
  A_out->info.n = n;
  A_out->info.stor_type = storage;
  A_out->info.lda = (storage == COL_MAJOR) ? m : n;
  A_out->info.nval = m * n;

  return 0;
}

// A function that takes a matrix A and a matrix B, it copies R the upper triangular of A in B with B(i,j) as a point of departure. we treat only the colmajor case
int CPLM_MatDenseTriangularFillBlock(CPLM_Mat_Dense_t* A_in, CPLM_Mat_Dense_t *B_io, int index_i, int index_j, int Size_R)
{
CPLM_PUSH
CPLM_BEGIN_TIME

#if 1
  int ierr      = 0;
  int dataSet   = 0;  //Number of values either in one line or one column
  int offset    = 0;  //Offset to be in first position in B_io
  int nset      = 0;  //Number of rows or columns to copy
  int minIter   = 0;
  CPLM_Mat_Dense_t A_remain_s  = CPLM_MatDenseNULL();

  CPLM_ASSERT(A_in->val  != NULL);
  CPLM_ASSERT(B_io->val  != NULL);
  CPLM_ASSERT(index_i    >= 0);
  CPLM_ASSERT(index_j    >= 0);
  CPLM_ASSERT(index_i    < B_io->info.m);
  CPLM_ASSERT(index_j    < B_io->info.n);


  if(B_io->info.stor_type != A_in->info.stor_type)
  {
    CPLM_Abort("Matrix storage type are not the same the storage type of "
        "the first matrix is %d and that of the second matrix is %d \n",
        B_io->info.stor_type,
        A_in->info.stor_type);
  }


  if(B_io->info.stor_type == COL_MAJOR)
  {
    //CPLM_ASSERT((index_j + ) <= B_io->info.n);

    offset  = index_j * B_io->info.lda + index_i;
    dataSet = 1;
    nset    = A_in->info.n;
  }
  else
  {
    CPLM_ASSERT((index_i + nset) <= B_io->info.m);

    offset  = index_i * B_io->info.lda + index_j;
    dataSet =  A_in->info.n;
    nset    = CPLM_MIN(A_in->info.m,A_in->info.n);
  }

  minIter = CPLM_MIN(nset,A_in->info.n);

  if(B_io->info.stor_type == COL_MAJOR)
  {
    for(int i = 0; i < minIter; i++)
    {
      CPLM_ASSERT((i * B_io->info.lda + offset + (dataSet + i)) <= B_io->info.nval);
      CPLM_ASSERT((i * A_in->info.lda + (dataSet + i)) <= A_in->info.nval);
      memcpy(B_io->val + i * B_io->info.lda + offset,
          A_in->val + i * A_in->info.lda,
          (dataSet + i )* sizeof(double));
    }
  }
  else
  {
    for(int i = 0; i < minIter; i++)
    {
      memcpy(B_io->val + i * B_io->info.lda + offset + i,
          A_in->val + i * A_in->info.lda + i,
          (dataSet - i )* sizeof(double));
    }
  }

  //Copy the remaining rectangular block if needed
  if(minIter != A_in->info.n)
  {
    CPLM_MatDenseSetInfo(&A_remain_s,
      A_in->info.M,
      A_in->info.N,
      A_in->info.m,
      A_in->info.n-minIter,
      A_in->info.stor_type);CPLM_CHKERR(ierr);
    A_remain_s.val = A_in->val + minIter *minIter;

    ierr = CPLM_MatDenseBlockFill( &A_remain_s,
        B_io,
        index_i,
        index_j + minIter);CPLM_CHKERR(ierr);
  }

//CPLM_MatDensePrintf2D("A_in",A_in);
//CPLM_MatDensePrintf2D("B_io",B_io);
#else
//*
  int mA = 0;
  int nA = 0;
  int mB = 0;
  int nB = 0;

  mA = B_io->info.m;
  nA = B_io->info.n;
  mB = A_in->info.m;
  nB = A_in->info.n;

  CPLM_ASSERT(A_in->val != NULL);
  CPLM_ASSERT(B_io->val != NULL);

  if(Size_R + index_i > mA || Size_R + index_j > nA )
  {
    CPLM_Abort("There is not enough space in the filling in matrix\n The triangular matrix is of size %dx%d and the block to be filled is of size %dx%d \n",Size_R, Size_R, mA - index_i , nA - index_j);
  }

  if( B_io->info.stor_type == ROW_MAJOR)
  {
    for(int i = 0; i < Size_R; i++)
    {
      for(int j = i; j < Size_R; j++)
      {
        CPLM_ASSERT((nA * (index_i + i) + index_j +j)  < B_io->info.nval);
        CPLM_ASSERT((nB * i + j)                       < A_in->info.nval);

        B_io->val[nA * (index_i + i) + index_j + j] = A_in->val[nB * i + j];
      }
    }
  }
  else
  {
    for(int j = 0; j < Size_R; j++)
    {
      for(int i = 0; i <= j; i++)
      {
        CPLM_ASSERT((mA * (index_j + j) + index_i + i) < B_io->info.nval);
        CPLM_ASSERT((mB * j + i)                       < A_in->info.nval);

        B_io->val[mA * (index_j + j) + index_i + i] = A_in->val[mB * j + i];
      }
    }
  }
//*/
#endif
CPLM_END_TIME
CPLM_POP
  return 0;
}

int CPLM_MatDenseInit(CPLM_Mat_Dense_t* A_out, CPLM_Info_Dense_t info)
{
CPLM_PUSH
  A_out->info = info;
  A_out->val  = NULL;
CPLM_POP
  return 0; // Success
}

int CPLM_MatDenseReset(CPLM_Mat_Dense_t *A)
{
  CPLM_ASSERT(A->val == NULL);

  A->info.M         = 0;
  A->info.N         = 0;
  A->info.m         = 0;
  A->info.n         = 0;
  A->info.stor_type = ROW_MAJOR;

  return 0;
}

int CPLM_MatDenseIRecvData(CPLM_Mat_Dense_t* A_in, int source, int tag, MPI_Comm comm, MPI_Request *request)
{
CPLM_PUSH
CPLM_BEGIN_TIME
  int ierr = 0;
  //recv values
  ierr = MPI_Irecv(A_in->val,A_in->info.nval,MPI_DOUBLE,source,tag,comm,request);
  CPLM_checkMPIERR(ierr,"recv_dense_data");
CPLM_END_TIME
CPLM_POP
  return ierr;
}

void CPLM_MatDensePrintInfo(CPLM_Mat_Dense_t* A)
{
  printf("Dense Matrix %dx%d\tLocal Data: %dx%d\t%s\tLeading Dimension Array %d\tNumber of elements of array %d\n",
          A->info.M,
          A->info.N,
          A->info.m,
          A->info.n,
          (A->info.stor_type==ROW_MAJOR) ? "Storage: Row major\n" : "Storage: Col major\n",
          A->info.lda,
          A->info.nval);
}



int CPLM_MatDenseGetRInplace(CPLM_Mat_Dense_t* A_io)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  int ierr  = 0;
  int m     = 0;
  int n     = 0;

  m = A_io->info.m;
  n = A_io->info.n;

  if(A_io->info.stor_type == ROW_MAJOR)
  {
    //Set lower part to 0
    for (int i = 1; i < CPLM_MIN(m,n); i++)
    {
      memset(A_io->val + i * A_io->info.lda, 0, i * sizeof(double));
    }

    CPLM_ASSERT(((m - 1) * A_io->info.lda + n) <= A_io->info.nval);
    //Remove extra lines
    for(int i = n; i < m; i++)
    {
      memset(A_io->val + i * A_io->info.lda, 0,  n * sizeof(double));
    }
  }
  else
  {
    //Set lower part to 0
    for (int i = 0; i < n; i++)
    {
      memset(A_io->val + i * A_io->info.lda + i + 1, 0, (A_io->info.lda - i - 1) * sizeof(double));
    }
  }
CPLM_END_TIME
CPLM_POP
  return ierr;
}

int CPLM_MatDenseConstant(CPLM_Mat_Dense_t* A, double value)
{
  int ierr = 0;

  if(A->val == NULL)
  {
    ierr = CPLM_MatDenseMalloc(A);CPLM_CHKERR(ierr);
  }
  // memset does not work here because val is an array of double AND memset uses char only
  for (int i = 0; i < (A->info.m)*(A->info.n); i++)
    A->val[i] = value;
  return ierr;
}

int CPLM_MatDenseRecvData(CPLM_Mat_Dense_t* A_io, int source, int tag, MPI_Comm comm)
{
CPLM_PUSH
CPLM_BEGIN_TIME
CPLM_OPEN_TIMER

  int ierr  = 0;
  MPI_Status status;

  CPLM_ASSERT(A_io->val != NULL);

  ierr = MPI_Recv(A_io->val,A_io->info.nval,MPI_DOUBLE,source,tag,comm,&status);
  CPLM_checkMPIERR(ierr,"recv_dense_val");

CPLM_CLOSE_TIMER
CPLM_END_TIME
CPLM_POP
  return ierr;
}

// Function that fills M2 in a block of M1 that starts on M1->val[index_i][index_j];
int CPLM_MatDenseBlockFill(CPLM_Mat_Dense_t* A_in, CPLM_Mat_Dense_t* B_io, int index_i, int index_j)
{
CPLM_PUSH
CPLM_BEGIN_TIME
CPLM_OPEN_TIMER
#if 1
  int ierr      = 0;
  int dataSet   = 0;  //Number of values either in one line or one column
  int offset    = 0;  //Offset to be in first position in B_io
  int nset      = 0;  //Number of rows or columns to copy

  CPLM_ASSERT(A_in->val  != NULL);
  CPLM_ASSERT(B_io->val  != NULL);
  CPLM_ASSERT(index_i    >= 0);
  CPLM_ASSERT(index_j    >= 0);
  CPLM_ASSERT(index_i    < B_io->info.m);
  CPLM_ASSERT(index_j    < B_io->info.n);
  CPLM_ASSERT((index_i + A_in->info.m) <= B_io->info.m);
  CPLM_ASSERT((index_j + A_in->info.n) <= B_io->info.n);

  if(B_io->info.stor_type != A_in->info.stor_type)
  {
      CPLM_Abort("Matrix storage type are not the same the storage type of the first matrix is %d and that of the second matrix is %d \n",B_io->info.stor_type, A_in->info.stor_type);
  }

  offset    = (B_io->info.stor_type == COL_MAJOR) ?
    index_j * B_io->info.lda + index_i:
    index_i * B_io->info.lda + index_j;

  dataSet  = (B_io->info.stor_type == COL_MAJOR) ?
    A_in->info.m:
    A_in->info.n;

  nset      = (B_io->info.stor_type == COL_MAJOR) ?
    A_in->info.n:
    A_in->info.m;

  for(int i = 0; i < nset; i++)
  {
    memcpy(B_io->val + i * B_io->info.lda + offset,
      A_in->val + i * A_in->info.lda,
      dataSet * sizeof(double));
  }

//CPLM_MatDensePrintf2D("A_in",A_in);
//CPLM_MatDensePrintf2D("B_io",B_io);
#else
//*
  int ierr  = 0;
  int i     = 0;
  int j     = 0;
  int m1    = 0;
  int m2    = 0;
  int n1    = 0;
  int n2    = 0;

  if(B_io->info.stor_type != A_in->info.stor_type)
  {
      CPLM_Abort("Matrix storage type are not the same the storage type of the first matrix is %d and that of the second matrix is %d \n",B_io->info.stor_type, A_in->info.stor_type);
  }

  m1  = B_io->info.m;
  m2  = A_in->info.m;
  if(A_in->info.stor_type == COL_MAJOR)
  {
      n2=A_in->info.n;
      for(j=0 ; j < n2 ; j++)
          for(i=0 ; i < m2 ; i++)
              B_io->val[m1 * (j + index_j) + i + index_i] = A_in->val[m2 * j + i];
  }
  else
  {
      n1=B_io->info.n;
      n2=A_in->info.n;
      for(i=0 ; i < m2 ; i++)
          for(j=0 ; j < n2 ; j++)
              B_io->val[n1 * (i + index_i) + j + index_j] = A_in->val[n2 * i + j];
  }
//*/
#endif
CPLM_CLOSE_TIMER
CPLM_END_TIME
CPLM_POP
  return ierr;
}

int CPLM_MatDenseRealloc(CPLM_Mat_Dense_t  *A_io)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  A_io->val = (double*) realloc(A_io->val, A_io->info.nval * sizeof(double));

CPLM_END_TIME
CPLM_POP
  return !(A_io->val != NULL);
}

int CPLM_MatDenseIsSameLocalInfo(CPLM_Mat_Dense_t *A_in, CPLM_Mat_Dense_t *B_in)
{
  int ok = CPLM_TRUE;

  CPLM_Info_Dense_t f1 = A_in->info;
  CPLM_Info_Dense_t f2 = B_in->info;

  return (f1.m          == f2.m     &&
          f1.n          == f2.n     &&
          f1.nval       == f2.nval  &&
          f1.lda        == f2.lda   &&
          f1.stor_type  == f2.stor_type) ? ok : !ok ;

}


int CPLM_MatDenseCopy(CPLM_Mat_Dense_t* A_in, CPLM_Mat_Dense_t* B_out)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  int ierr = 0;

	if(A_in->info.stor_type == COL_MAJOR)
	{
		CPLM_ASSERT(A_in->info.lda == A_in->info.m);
	}
	else
	{
		CPLM_ASSERT(A_in->info.lda == A_in->info.n);
	}
	if (B_out->val == NULL)
  {
    B_out->info = A_in->info;
    ierr = CPLM_MatDenseMalloc(B_out);CPLM_CHKERR(ierr);
  }
  else if (!CPLM_MatDenseIsSameLocalInfo(A_in,B_out))
  {
    B_out->info = A_in->info;
    ierr = CPLM_MatDenseRealloc(B_out);CPLM_CHKERR(ierr);
  }

  memcpy(B_out->val, A_in->val, B_out->info.m * B_out->info.n * sizeof(double));

CPLM_END_TIME
CPLM_POP
  return ierr;
}
