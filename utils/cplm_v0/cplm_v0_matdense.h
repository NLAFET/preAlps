/*
* This file contains functions used to manipulate dense matrices
*
* Authors : Sebastien Cayrols
*         : Olivier Tissot
* Email   : sebastien.cayrols@[(gmail.com) | (inria.fr)]
*         : olivier.tissot@inria.fr
*/

#ifndef CPLM_V0_MATDENSE_H
#define CPLM_V0_MATDENSE_H

#include <mpi.h>



/******************************************************************************/
/*                                  STRUCT                                    */
/******************************************************************************/
/* Storage type of a dense matrix */
typedef enum {
  ROW_MAJOR,
  COL_MAJOR
} CPLM_storage_type_t;
/* Information about a dense matrix */
typedef struct {
  int M; // Global num of rows
  int N; // Global num of cols
  int m; // Local num of rows
  int n; // Local num of cols
  int lda; // Leading dimension of the matrix
  int nval; //  m*n ;
  CPLM_storage_type_t stor_type; // Row Major or Col Major storage
} CPLM_Info_Dense_t;
/* Dense Matrix structure */
typedef struct {
  double* val;
  CPLM_Info_Dense_t info;
} CPLM_Mat_Dense_t;

/******************************************************************************/

#define PRINT_PARTIAL_M 10
#define PRINT_PARTIAL_N 10

/******************************************************************************/


#define CPLM_MatDenseNULL() { .val = NULL, .info = {.M=0, .N=0, .m=0, .n=0, .stor_type=ROW_MAJOR, .lda = 0, .nval = 0} }
#define CPLM_MatDensePrintfInfo(_msg,_A) { printf("%s\n",(_msg));    \
                                      CPLM_MatDensePrintInfo((_A)); }
// Printing
#define CPLM_MatDensePrintf2D(_msg,_A) { printf("%s\n",(_msg));                                              \
                                  ( ((_A)->info.m<PRINT_PARTIAL_M && (_A)->info.n<PRINT_PARTIAL_N) ?  \
                                      CPLM_MatDensePrint2D((_A))     :                                     \
                                      CPLM_MatDensePrintPartial2D((_A)) );}
//MPI_Datatype initMPI_StructDenseInfo();

int CPLM_MatDenseCalloc(CPLM_Mat_Dense_t *A_io);
void CPLM_MatDenseFree(CPLM_Mat_Dense_t  *A_io);
int CPLM_MatDenseISendData(CPLM_Mat_Dense_t* A_in, int dest, int tag, MPI_Comm comm, MPI_Request **request);
int CPLM_MatDenseMalloc( CPLM_Mat_Dense_t *A_io);
int CPLM_MatDenseSetInfo(CPLM_Mat_Dense_t* A_out, int M, int N, int m, int n, CPLM_storage_type_t storage);


// A function that takes a matrix A and a matrix B, it copies R the upper triangular of A in B with B(i,j) as a point of departure. we treat only the colmajor case
int CPLM_MatDenseTriangularFillBlock(CPLM_Mat_Dense_t* A_in, CPLM_Mat_Dense_t *B_io, int index_i, int index_j, int Size_R);

int CPLM_MatDenseInit(CPLM_Mat_Dense_t* A_out, CPLM_Info_Dense_t info);

int CPLM_MatDenseReset(CPLM_Mat_Dense_t *A);

int CPLM_MatDenseIRecvData(CPLM_Mat_Dense_t* A_in, int source, int tag, MPI_Comm comm, MPI_Request *request);

void CPLM_MatDensePrintInfo(CPLM_Mat_Dense_t* A);



int CPLM_MatDenseGetRInplace(CPLM_Mat_Dense_t* A_io);
int CPLM_MatDenseConstant(CPLM_Mat_Dense_t* A, double value);


//ADD a test to call it if nnz > cst i.e. here cst = limit*limit
//Cons : in that case, we can remove MINI() tests
void CPLM_MatDensePrintPartial2D(CPLM_Mat_Dense_t* A);

void CPLM_MatDensePrint2D(CPLM_Mat_Dense_t* A);

int CPLM_MatDenseRecvData(CPLM_Mat_Dense_t* A_io, int source, int tag, MPI_Comm comm);

// Function that fills M2 in a block of M1 that starts on M1->val[index_i][index_j];
int CPLM_MatDenseBlockFill(CPLM_Mat_Dense_t* A_in, CPLM_Mat_Dense_t* B_io, int index_i, int index_j);

int CPLM_MatDenseRealloc(CPLM_Mat_Dense_t  *A_io);

int CPLM_MatDenseIsSameLocalInfo(CPLM_Mat_Dense_t *A_in, CPLM_Mat_Dense_t *B_in);

int CPLM_MatDenseCopy(CPLM_Mat_Dense_t* A_in, CPLM_Mat_Dense_t* B_out);

//ADD a test to call it if nnz > cst i.e. here cst = limit*limit
//Cons : in that case, we can remove MINI() tests
void CPLM_MatDensePrintPartial2D(CPLM_Mat_Dense_t* A);

void CPLM_MatDensePrint2D(CPLM_Mat_Dense_t* A) ;

#endif
