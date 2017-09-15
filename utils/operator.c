/******************************************************************************/
/* Author     : Olivier Tissot                                                */
/* Creation   : 2016/08/05                                                    */
/* Description: Definition of the linear operator                             */
/******************************************************************************/


/******************************************************************************/
/*                                  INCLUDE                                   */
/******************************************************************************/
#ifdef PETSC
#include <petsc_interface.h>
#endif

#include "operator.h"
/******************************************************************************/

/******************************************************************************/
/*                              GLOBAL VARIABLES                              */
/******************************************************************************/
static Mat_CSR_t A_g = MatCSRNULL();
static IVector_t rowPos_g = DVectorNULL();
static IVector_t colPos_g = DVectorNULL();
static IVector_t rowPtr_g = DVectorNULL();
static IVector_t dep_g    = DVectorNULL();
static MPI_Comm comm_g;
/******************************************************************************/

/******************************************************************************/
/*                                    CODE                                    */
/******************************************************************************/
int OperatorBuild(const char* matrixFilename, MPI_Comm comm) {
PUSH
  int rank, size, ierr = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int root = 0; // root or master processor
  // Number of Metis domains: assumed to be size of comm for the moment
  int nbBlockPart = size;
  // Set the communicator
  comm_g = comm;

  //Load on first process
  Mat_CSR_t matCSR = MatCSRNULL();
  const char* operator_type = matrixFilename + strlen(matrixFilename) - 3;
  // MatrixMarket format
  if (rank == root) {
    if (strcmp(operator_type,"mtx") == 0) {
      ierr = LoadMatrixMarket(matrixFilename, &matCSR);CHKERR(ierr);
    }
    else {
      #ifdef PETSC
        Mat A_petsc;
        petscMatLoad(&A_petsc,matrixFilename,PETSC_COMM_SELF);
        petscCreateMatCSR(A_petsc,&matCSR);
        MatDestroy(&A_petsc);
      #else
        CPALAMEM_Abort("Please Compile with PETSC to read other matrix file type");
      #endif
    }
    IVector_t posB = IVectorNULL(), perm = IVectorNULL();
    ierr = metisKwayOrdering(&matCSR,
                             &perm,
                             nbBlockPart,
                             &posB);CHKERR(ierr);
    // Permute the matrix
    ierr = MatCSRPermute(&matCSR,
                         &A_g,
                         perm.val,
                         perm.val,
                         PERMUTE);CHKERR(ierr);
    // Change posB into rowPos because each proc might have several block Jacobi
    int inc = nbBlockPart / size;
    // For the moment we assume that each mpi process has one metis block
    if (inc != 1) {
       CPALAMEM_Abort("Each MPI process must have one (and only one) metis"
        "block (nbMetis = %d != %d = nbProcesses)",nbBlockPart,size);
    }
    ierr = IVectorMalloc(&rowPos_g, size+1);CHKERR(ierr);
    for (int i = 0; i < size; i++)
       rowPos_g.val[i] = posB.val[i*inc];
    rowPos_g.val[size] = posB.val[nbBlockPart];
    // Send submatrices as row panel layout
    for (int dest = 1; dest < size; dest++) {
        ierr = MatCSRGetRowPanel(&A_g,
                                &matCSR,
                                &rowPos_g,
                                dest);CHKERR(ierr);
        ierr = MatCSRSend(&matCSR, dest, MPI_COMM_WORLD);
    }
    MatCSRFree(&matCSR);
    // Just keep the row panel in master
    MatCSRCopy(&A_g,&matCSR);
    MatCSRFree(&A_g);
    ierr = MatCSRGetRowPanel(&matCSR,
                             &A_g,
                             &rowPos_g,
                             0);CHKERR(ierr);
    // Free memory
    IVectorFree(&posB);
    IVectorFree(&perm);
    MatCSRFree(&matCSR);
  }
  else { //other MPI processes received their own submatrix
    ierr = MatCSRRecv(&A_g,0,MPI_COMM_WORLD);
  }
  ierr = IVectorBcast(&rowPos_g,MPI_COMM_WORLD,root);
  ierr = MatCSRGetColBlockPos(&A_g,
                              &rowPos_g,
                              &colPos_g);CHKERR(ierr);
  ierr = MatCSRGetCommDep(&colPos_g,
                          A_g.info.m,
                          size,
                          rank,
                          &dep_g);CHKERR(ierr);

POP
  return ierr;
}

void OperatorPrint(int rank) {
PUSH
  if (rank == 0) {
    IVectorPrintf("rowPos",&rowPos_g);
    IVectorPrintf("colPos",&colPos_g);
    IVectorPrintf("rowPtr",&rowPtr_g);
    IVectorPrintf("dep"   ,&dep_g);
    MatCSRPrintInfo(&A_g);
    // MatCSRPrintf2D(&A_g,"A_g");
  }
POP
}

void OperatorFree() {
PUSH
  IVectorFree(&rowPos_g);
  IVectorFree(&colPos_g);
  IVectorFree(&rowPtr_g);
  IVectorFree(&dep_g);
  MatCSRFree(&A_g);
POP
}

int BlockOperator(Mat_Dense_t* X, Mat_Dense_t* AX) {
PUSH
  int size;
  MPI_Comm_size(comm_g,&size);
  int algMatMult = 2;
  int ierr = MatCSRMatMult(&A_g,
                           X,
                           &dep_g,
                           dep_g.nval,
                           AX,
                           &rowPos_g,
                           &colPos_g,
                           &rowPtr_g,
                           comm_g,
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

int OperatorGetRowPosPtr(int** rowPos, int* sizeRowPos) {
PUSH
  *sizeRowPos = rowPos_g.nval;
  *rowPos = rowPos_g.val;
POP
  return (rowPos == NULL);
}

int OperatorGetColPosPtr(int** colPos, int* sizeColPos) {
PUSH
  *sizeColPos = colPos_g.nval;
  *colPos = colPos_g.val;
POP
  return (colPos == NULL);
}

int OperatorGetDepPtr(int** dep, int* sizeDep) {
PUSH
  *sizeDep = dep_g.nval;
  *dep = dep_g.val;
POP
  return (dep == NULL);
}

/******************************************************************************/
