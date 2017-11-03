/******************************************************************************/
/* Author     : Olivier Tissot, Simplice Donfack                              */
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
static CPLM_Mat_CSR_t A_g = CPLM_MatCSRNULL();
static CPLM_IVector_t rowPos_g = CPLM_DVectorNULL();
static CPLM_IVector_t colPos_g = CPLM_DVectorNULL();
static CPLM_IVector_t rowPtr_g = CPLM_DVectorNULL();
static CPLM_IVector_t dep_g    = CPLM_DVectorNULL();
static MPI_Comm comm_g;
/******************************************************************************/

/******************************************************************************/
/*                                    CODE                                    */
/******************************************************************************/

int preAlps_OperatorBuild(const char* matrixFilename, MPI_Comm comm) {
CPLM_PUSH
  int rank, size, ierr = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int root = 0; // root or master processor
  // Number of Metis domains: assumed to be size of comm for the moment
  int nbBlockPart = size;
  // Set the communicator
  comm_g = comm;

  //Load on first process
  CPLM_Mat_CSR_t matCSR = CPLM_MatCSRNULL();
  const char* operator_type = matrixFilename + strlen(matrixFilename) - 3;
  // MatrixMarket format
  if (rank == root) {
    if (strcmp(operator_type,"mtx") == 0) {
      ierr = CPLM_LoadMatrixMarket(matrixFilename, &matCSR);CPLM_CHKERR(ierr);
    }
    else {
      #ifdef PETSC
        Mat A_petsc;
        petscMatLoad(&A_petsc,matrixFilename,PETSC_COMM_SELF);
        petscCreateMatCSR(A_petsc,&matCSR);
        MatDestroy(&A_petsc);
      #else
        CPLM_Abort("Please Compile with PETSC to read other matrix file type");
      #endif
    }
    // Scale the matrix
    double* R = NULL;
    double* C = NULL;
    R = (double*) malloc(matCSR.info.m*sizeof(double));
    C = (double*) malloc(matCSR.info.m*sizeof(double));
    CPLM_MatCSRSymRACScaling(&matCSR,R,C);
    if (R != NULL) free(R);
    if (C != NULL) free(C);
    CPLM_IVector_t posB = CPLM_IVectorNULL(), perm = CPLM_IVectorNULL();
    ierr = CPLM_metisKwayOrdering(&matCSR,
                                  &perm,
                                  nbBlockPart,
                                  &posB);CPLM_CHKERR(ierr);
    // Permute the matrix
    ierr = CPLM_MatCSRPermute(&matCSR,
                              &A_g,
                              perm.val,
                              perm.val,
                              PERMUTE);CPLM_CHKERR(ierr);
    // Change posB into rowPos because each proc might have several block Jacobi
    int inc = nbBlockPart / size;
    // For the moment we assume that each mpi process has one metis block
    if (inc != 1) {
       CPLM_Abort("Each MPI process must have one (and only one) metis"
        "block (nbMetis = %d != %d = nbProcesses)",nbBlockPart,size);
    }
    ierr = CPLM_IVectorMalloc(&rowPos_g, size+1);CPLM_CHKERR(ierr);
    for (int i = 0; i < size; i++)
       rowPos_g.val[i] = posB.val[i*inc];
    rowPos_g.val[size] = posB.val[nbBlockPart];
    // Send submatrices as row panel layout
    for (int dest = 1; dest < size; dest++) {
        ierr = CPLM_MatCSRGetRowPanel(&A_g,
                                      &matCSR,
                                      &rowPos_g,
                                      dest);CPLM_CHKERR(ierr);
        ierr = CPLM_MatCSRSend(&matCSR, dest, MPI_COMM_WORLD);
    }
    CPLM_MatCSRFree(&matCSR);
    // Just keep the row panel in master
    CPLM_MatCSRCopy(&A_g,&matCSR);
    CPLM_MatCSRFree(&A_g);
    ierr = CPLM_MatCSRGetRowPanel(&matCSR,
                                  &A_g,
                                  &rowPos_g,
                                  0);CPLM_CHKERR(ierr);
    // Free memory
    CPLM_IVectorFree(&posB);
    CPLM_IVectorFree(&perm);
    CPLM_MatCSRFree(&matCSR);
  }
  else { //other MPI processes received their own submatrix
    ierr = CPLM_MatCSRRecv(&A_g,0,MPI_COMM_WORLD);
  }
  ierr = CPLM_IVectorBcast(&rowPos_g,MPI_COMM_WORLD,root);
  ierr = CPLM_MatCSRGetColBlockPos(&A_g,
                                   &rowPos_g,
                                   &colPos_g);CPLM_CHKERR(ierr);
  ierr = CPLM_MatCSRGetCommDep(&colPos_g,
                               A_g.info.m,
                               size,
                               rank,
                               &dep_g);CPLM_CHKERR(ierr);

CPLM_POP
  return ierr;
}

/* Setup a matrix vector product without permuting the matrix. Useful for the cases where the matrix has already been partitioned */
int preAlps_OperatorBuildNoPerm(CPLM_Mat_CSR_t *locA, int *idxRowBegin, int nbBlockPerProcs, MPI_Comm comm) {
CPLM_PUSH
  int rank, size, ierr = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Set the communicator
  comm_g = comm;

  int inc = nbBlockPerProcs;
  // For the moment we assume that each mpi process has one metis block
  if (inc != 1) {
     CPLM_Abort("[OperatorBuildNoPerm] Each MPI process must have one (and only one) metis");
  }

  CPLM_MatCSRCopy(locA, &A_g);

  //Create a copy of idxRowBegin as it is from an external source
  int *rowPosTmp = (int*) malloc((size+1)*sizeof(int));
  CPLM_ASSERT(rowPosTmp != NULL);
  memcpy (rowPosTmp, idxRowBegin, (size+1)*sizeof(int) );

  CPLM_IVectorCreateFromPtr(&rowPos_g, size+1, rowPosTmp);



  ierr = CPLM_MatCSRGetColBlockPos(&A_g,
                                   &rowPos_g,
                                   &colPos_g);CPLM_CHKERR(ierr);
  ierr = CPLM_MatCSRGetCommDep(&colPos_g,
                               A_g.info.m,
                               size,
                               rank,
                               &dep_g);CPLM_CHKERR(ierr);
CPLM_POP
  return ierr;
}


void preAlps_OperatorPrint(int rank) {
CPLM_PUSH
  if (rank == 0) {
    CPLM_IVectorPrintf("rowPos",&rowPos_g);
    CPLM_IVectorPrintf("colPos",&colPos_g);
    CPLM_IVectorPrintf("rowPtr",&rowPtr_g);
    CPLM_IVectorPrintf("dep"   ,&dep_g);
    CPLM_MatCSRPrintInfo(&A_g);
    // CPLM_MatCSRPrintf2D(&A_g,"A_g");
  }
CPLM_POP
}

void preAlps_OperatorFree() {
CPLM_PUSH
  CPLM_IVectorFree(&rowPos_g);
  CPLM_IVectorFree(&colPos_g);
  CPLM_IVectorFree(&rowPtr_g);
  CPLM_IVectorFree(&dep_g);
  CPLM_MatCSRFree(&A_g);
CPLM_POP
}

int preAlps_BlockOperator(CPLM_Mat_Dense_t* X, CPLM_Mat_Dense_t* AX) {
CPLM_PUSH
  int size;
  MPI_Comm_size(comm_g,&size);
  int algMatMult = 2;
  int ierr = CPLM_MatCSRMatMult(&A_g,
                                X,
                                &dep_g,
                                dep_g.nval,
                                AX,
                                &rowPos_g,
                                &colPos_g,
                                &rowPtr_g,
                                comm_g,
                                algMatMult);
CPLM_POP
  return ierr;
}

int preAlps_OperatorGetA(CPLM_Mat_CSR_t* A) {
CPLM_PUSH
  CPLM_ASSERT(A != NULL);
  *A = A_g;
CPLM_POP
  return 0;
}

int preAlps_OperatorGetSizes(int* M, int* m) {
CPLM_PUSH
  CPLM_ASSERT(M != NULL);
  CPLM_ASSERT(m != NULL);
  *M = A_g.info.M;
  *m = A_g.info.m;
CPLM_POP
  return 0;
}

int preAlps_OperatorGetRowPosPtr(int** rowPos, int* sizeRowPos) {
CPLM_PUSH
  *sizeRowPos = rowPos_g.nval;
  *rowPos = rowPos_g.val;
CPLM_POP
  return (rowPos == NULL);
}

int preAlps_OperatorGetColPosPtr(int** colPos, int* sizeColPos) {
CPLM_PUSH
  *sizeColPos = colPos_g.nval;
  *colPos = colPos_g.val;
CPLM_POP
  return (colPos == NULL);
}

int preAlps_OperatorGetDepPtr(int** dep, int* sizeDep) {
CPLM_PUSH
  *sizeDep = dep_g.nval;
  *dep = dep_g.val;
CPLM_POP
  return (dep == NULL);
}

/******************************************************************************/
