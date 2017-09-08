/******************************************************************************/
/* Author     : Olivier Tissot                                                */
/* Creation   : 2016/08/05                                                    */
/* Description: Definition of the linear operator                             */
/******************************************************************************/


/******************************************************************************/
/*                                  INCLUDE                                   */
/******************************************************************************/
#ifdef USE_PETSC
#include <petsc_interface.h>
#endif

#include "operator.h"
/******************************************************************************/

/******************************************************************************/
/*                              GLOBAL VARIABLES                              */
/******************************************************************************/
static Mat_CSR_t A_g = MatCSRNULL();
static Operator_Struct_t AStruct_g;    // Partition of global A
/******************************************************************************/

/******************************************************************************/
/*                                    CODE                                    */
/******************************************************************************/
int OperatorBuild(Usr_Param_t* param) {
PUSH
  int rank, size, ierr = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int root = 0; // root or master processor
  AStruct_g.rowPos.val = NULL;
  AStruct_g.colPos.val = NULL;
  AStruct_g.rowPtr.val = NULL;
  AStruct_g.dep.val    = NULL;
  //Load on first process
  Mat_CSR_t matCSR = MatCSRNULL();
  const char* operator_type = param->matrixFilename + strlen(param->matrixFilename) - 3;
  // MatrixMarket format
  if (rank == root) {
    if (strcmp(operator_type,"mtx") == 0) {
      ierr = LoadMatrixMarket(param->matrixFilename, &matCSR);CHKERR(ierr);
    }
    else {
      #ifdef USE_PETSC
        Mat A_petsc;
        petscMatLoad(&A_petsc,param->matrixFilename,PETSC_COMM_SELF);
        petscCreateMatCSR(A_petsc,&matCSR);
        MatDestroy(&A_petsc);
      #else
        CPALAMEM_Abort("Please Compile with PETSC to read other matrix file type");
      #endif
    }
    IVector_t posB = IVectorNULL(), perm = IVectorNULL();
    ierr = metisKwayOrdering(&matCSR,
                             &perm,
                             param->nbBlockPart,
                             &posB);CHKERR(ierr);
    // Permute the matrix
    ierr = MatCSRPermute(&matCSR,
                         &A_g,
                         perm.val,
                         perm.val,
                         PERMUTE);CHKERR(ierr);
    // Change posB into rowPos because each proc might have several block Jacobi
    int inc = param->nbBlockPart / size;
    // For the moment we assume that each mpi process has one metis block
    if (inc != 1) {
	     CPALAMEM_Abort("Each MPI process must have one (and only one) metis"
        "block (nbMetis = %d != %d = nbProcesses)",param->nbBlockPart,size);
    }
    ierr = IVectorMalloc(&AStruct_g.rowPos, size+1);CHKERR(ierr);
    for (int i = 0; i < size; i++)
	     AStruct_g.rowPos.val[i] = posB.val[i*inc];
    AStruct_g.rowPos.val[size] = posB.val[param->nbBlockPart];
    // Send submatrices as row panel layout
    for (int dest = 1; dest < size; dest++) {
	      ierr = MatCSRGetRowPanel(&A_g,
                                &matCSR,
                                &AStruct_g.rowPos,
                                dest);CHKERR(ierr);
	      ierr = MatCSRSend(&matCSR, dest, MPI_COMM_WORLD);
    }
    MatCSRFree(&matCSR);
    // Just keep the row panel in master
    MatCSRCopy(&A_g,&matCSR);
    MatCSRFree(&A_g);
    ierr = MatCSRGetRowPanel(&matCSR,
                             &A_g,
                             &AStruct_g.rowPos,
                             0);CHKERR(ierr);
    // Free memory
    IVectorFree(&posB);
    IVectorFree(&perm);
    MatCSRFree(&matCSR);
  }
  else { //other MPI processes received their own submatrix
    ierr = MatCSRRecv(&A_g,0,MPI_COMM_WORLD);
  }
  ierr = IVectorBcast(&AStruct_g.rowPos,MPI_COMM_WORLD,root);
  ierr = MatCSRGetColBlockPos(&A_g,
                              &AStruct_g.rowPos,
                              &AStruct_g.colPos);CHKERR(ierr);
  ierr = MatCSRGetCommDep(&AStruct_g.colPos,
                          A_g.info.m,
                          size,
                          rank,
                          &AStruct_g.dep);CHKERR(ierr);

  AStruct_g.comm = MPI_COMM_WORLD;
  BlockJacobiCreate(&A_g,&AStruct_g);

POP
  return ierr;
}

void OperatorPrint(int rank) {
PUSH
  if (rank == 0) {
    IVectorPrintf("rowPos",&(AStruct_g.rowPos));
    IVectorPrintf("colPos",&(AStruct_g.colPos));
    IVectorPrintf("rowPtr",&(AStruct_g.rowPtr));
    IVectorPrintf("dep"   ,&(AStruct_g.dep));
    MatCSRPrintInfo(&A_g);
    // MatCSRPrintf2D(&A_g,"A_g");
  }
POP
}

void OperatorFree() {
PUSH
  BlockJacobiFree();
  IVectorFree(&(AStruct_g.rowPos));
  IVectorFree(&(AStruct_g.colPos));
  IVectorFree(&(AStruct_g.rowPtr));
  IVectorFree(&(AStruct_g.dep));
  MatCSRFree(&A_g);
POP
}

int BlockOperator(Mat_Dense_t* X, Mat_Dense_t* AX) {
PUSH
  int algMatMult = 2;
  int ierr = MatCSRMatMult(&A_g,
                           X,
                           &(AStruct_g.dep),
                           AStruct_g.dep.nval,
                           AX,
                           &(AStruct_g.rowPos),
                           &(AStruct_g.colPos),
                           &(AStruct_g.rowPtr),
                           AStruct_g.comm,
                           algMatMult);
POP
  return ierr;
}

int OperatorGetA(Mat_CSR_t* A) {
PUSH
  ASSERT(A != NULL);
  *A = A_g;
POP
  return 0;
}

int OperatorGetSizes(int* M, int* m) {
PUSH
  ASSERT(M != NULL);
  ASSERT(m != NULL);
  *M = A_g.info.M;
  *m = A_g.info.m;
POP
  return 0;
}

int OperatorGetRowPosPtr(IVector_t* rowPos) {
PUSH
  *rowPos = AStruct_g.rowPos;
POP
  return (rowPos == NULL);
}

int OperatorGetColPosPtr(IVector_t* colPos) {
PUSH
  *colPos = AStruct_g.colPos;
POP
  return (colPos == NULL);
}

int OperatorGetDepPtr(IVector_t* dep) {
PUSH
  *dep = AStruct_g.dep;
POP
  return (dep == NULL);
}

/******************************************************************************/
