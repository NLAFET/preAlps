/******************************************************************************/
/* Author     : Olivier Tissot , Simplice Donfack                             */
/* Creation   : 2016/06/23                                                    */
/* Description: Benchmark ECG vs. PETSc PCG                                   */
/******************************************************************************/

/******************************************************************************/
/*                                  INCLUDE                                   */
/******************************************************************************/
/* STD */
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
/* MPI */
#include <mpi.h>
/* MKL */
#include <mkl.h>

/* CPaLAMeM */
#include <cpalamem_macro.h>
#include <cpalamem_instrumentation.h>

#ifdef PETSC
#include <petsc_interface.h>
/* Petsc */
#include <petscksp.h>
#endif

/* preAlps */
#include "operator.h"
#include "block_jacobi.h"
#include "ecg.h"
/******************************************************************************/

/******************************************************************************/
/*                                    CODE                                    */
/******************************************************************************/
void mkl_sparse_set_double_sm_hint(sparse_matrix_t Aii, int nrhs, int ncall) {
  // Descriptor of main sparse matrix properties
  struct matrix_descr descr_Aii;
  // Analyze sparse matrix; choose proper kernels and workload balancing strategy
  descr_Aii.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
  descr_Aii.mode = SPARSE_FILL_MODE_LOWER;
  descr_Aii.diag = SPARSE_DIAG_UNIT;
  mkl_sparse_set_sm_hint(Aii,
                         SPARSE_OPERATION_NON_TRANSPOSE,
                         descr_Aii,
                         SPARSE_LAYOUT_COLUMN_MAJOR,
                         nrhs,
                         ncall);
  descr_Aii.mode = SPARSE_FILL_MODE_UPPER;
  descr_Aii.diag = SPARSE_DIAG_NON_UNIT;
  mkl_sparse_set_sm_hint(Aii,
                         SPARSE_OPERATION_NON_TRANSPOSE,
                         descr_Aii,
                         SPARSE_LAYOUT_COLUMN_MAJOR,
                         nrhs,
                         ncall);
}

void mkl_ilu0_apply(sparse_matrix_t Aii, double* x, double* y, int m, int n) {
  struct matrix_descr descr_Aii;
  descr_Aii.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
  descr_Aii.mode = SPARSE_FILL_MODE_LOWER;
  descr_Aii.diag = SPARSE_DIAG_UNIT;
  mkl_sparse_d_trsm(SPARSE_OPERATION_NON_TRANSPOSE,
                    1.0,Aii,descr_Aii,
                    SPARSE_LAYOUT_COLUMN_MAJOR,x,n,m,y,m);
  descr_Aii.mode = SPARSE_FILL_MODE_UPPER;
  descr_Aii.diag = SPARSE_DIAG_NON_UNIT;
  mkl_sparse_d_trsm(SPARSE_OPERATION_NON_TRANSPOSE,
                    1.0,Aii,descr_Aii,
                    SPARSE_LAYOUT_COLUMN_MAJOR,y,n,m,y,m);
}

int main(int argc, char** argv) {
#ifdef PETSC
  PetscInitialize(&argc, &argv,NULL,NULL);
  CPLM_SetEnv();

CPLM_OPEN_TIMER
  /*================ Initialize ================*/
  int rank, size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Force sequential execution on each MPI process
  // OT: I tested and it still works with OpenMP activated
  MKL_Set_Num_Threads(1);

  /*======== Construct the operator using a CSR matrix ========*/
  const char* matrixFilename = argv[1];
  CPLM_Mat_CSR_t A = CPLM_MatCSRNULL();
  int M, m;
  int* rowPos = NULL;
  int* colPos = NULL;
  int sizeRowPos, sizeColPos;
  // Read and partition the matrix
  preAlps_OperatorBuild(matrixFilename,MPI_COMM_WORLD);
  // Get the CSR structure of A
  preAlps_OperatorGetA(&A);
  // Get the sizes of A
  preAlps_OperatorGetSizes(&M,&m);
  // Get row partitioning of A
  preAlps_OperatorGetRowPosPtr(&rowPos,&sizeRowPos);
  // Get col partitioning induced by this row partitioning
  preAlps_OperatorGetColPosPtr(&colPos,&sizeColPos);

  /*======== Construct the preconditioner ========*/
  // Get the diagonal block corresponding to the local row panel
  CPLM_Mat_CSR_t Aii;
  CPLM_IVector_t rowPos_s = CPLM_IVectorNULL();
  CPLM_IVector_t colPos_s = CPLM_IVectorNULL();
  CPLM_IVectorCreateFromPtr(&rowPos_s,sizeRowPos,rowPos);
  CPLM_IVectorCreateFromPtr(&colPos_s,sizeRowPos,colPos);
  CPLM_MatCSRGetDiagBlock(&A,&Aii,&rowPos_s,&colPos_s,UNSYMMETRIC);
  CPLM_MatCSRConvertTo1BasedIndexing(&Aii);
  // Parameters (see MKL documentation)
  int ipar[128], ierr;
  double dpar[128];
  ipar[30] = 1;
  dpar[30] = 1e-16;
  dpar[31] = 1e-10;
  // Allocate memory
  double* ilu0 = NULL;
  ilu0 = (double*) malloc(Aii.info.nnz*sizeof(double));
  // Compute ilu0 factorization of Adiag
  dcsrilu0(&Aii.info.n,Aii.val,Aii.rowPtr,Aii.colInd,ilu0,ipar,dpar,&ierr);
  if (ierr < 0) {
    CPLM_Abort("Error in ilu0 factorization: %d!",ierr);
  }
  // Create MKL sparse handle
  sparse_matrix_t      mkl_Aii; // Structure with sparse matrix stored
  mkl_sparse_d_create_csr(&mkl_Aii,
                          SPARSE_INDEX_BASE_ONE,
                          Aii.info.m,
                          Aii.info.n,
                          Aii.rowPtr,
                          Aii.rowPtr+1,
                          Aii.colInd,
                          ilu0);

  /*============= Construct a random rhs =============*/
  double* rhs = (double*) malloc(A.info.m*sizeof(double));
  // Set the seed of the random generator
  srand(0);
  for (int i = 0; i < A.info.m; ++i)
    rhs[i] = ((double) rand() / (double) RAND_MAX);

  // Set global parameters for both PETSc and ECG
  double tol = 1e-5;
  int maxIter = 1000;

  /*================ Petsc solve ================*/
  Mat A_petsc;
  Vec X, B;
  KSP ksp;
  KSP *subksp;
  PC pc, subpc;
  int first,nlocal;

  // Set RHS
  VecCreateMPIWithArray(MPI_COMM_WORLD,1,A.info.m,A.info.M,rhs,&B);
  VecCreateMPI(MPI_COMM_WORLD,A.info.m,A.info.M,&X);
  VecAssemblyBegin(X);
  VecAssemblyBegin(X);
  VecAssemblyBegin(B);
  VecAssemblyBegin(B);
  // Set solver
  KSPCreate(MPI_COMM_WORLD,&ksp);
  CPLM_petscCreateMatFromMatCSR(&A,&A_petsc);
  KSPSetOperators(ksp,A_petsc,A_petsc);
  KSPSetType(ksp,KSPCG);
  KSPSetTolerances(ksp,
                   tol,
                   PETSC_DEFAULT,
                   PETSC_DEFAULT,
                   maxIter);
  KSPSetPCSide(ksp,PC_LEFT);
  KSPCGSetType(ksp,KSP_CG_SYMMETRIC);
  KSPSetNormType(ksp,KSP_NORM_UNPRECONDITIONED);
  KSPGetPC(ksp,&pc);
  PCSetType(pc,PCBJACOBI);
  KSPSetUp(ksp);
  PCBJacobiGetSubKSP(pc,&nlocal,&first,&subksp);

  // Loop over the local blocks, setting various KSP options
  // for each block.
  for (int i=0; i<nlocal; i++) {
    KSPGetPC(subksp[i],&subpc);
    PCSetType(subpc,PCILU);
    PCFactorSetLevels(subpc,0);
    /* PCFactorSetMatSolverPackage(subpc,MATSOLVERMKL_PARDISO); */
  }

CPLM_TIC(step1,"KSPSolve")
  KSPSolve(ksp,B,X);
CPLM_TAC(step1)

  int its = -1;
  double rnorm = 0e0;
  KSPGetIterationNumber(ksp,&its);
  KSPGetResidualNorm(ksp,&rnorm);
  if (rank == 0)
    printf("=== Petsc ===\n\titerations: %d\n\tnorm(res): %e\n",its,rnorm);
  KSPDestroy(&ksp);

  /*================ ECG solve ================*/
  preAlps_ECG_t ecg;
  // Set parameters
  ecg.comm = MPI_COMM_WORLD;  /* MPI Communicator */
  ecg.globPbSize = M;         /* Size of the global problem */
  ecg.locPbSize = m;          /* Size of the local problem */
  ecg.maxIter = maxIter;      /* Maximum number of iterations */
  ecg.enlFac = 4;             /* Enlarging factor */
  ecg.tol = tol;              /* Tolerance of the method */
  ecg.ortho_alg = ORTHODIR;   /* Orthogonalization algorithm */
  ecg.bs_red = ALPHA_RANK;     /* Only NO_BS_RED implemented !! */
  /* Restore the pointer */
  VecGetArray(B,&rhs);
  // Get local and global sizes of operator A
  int rci_request = 0;
  int stop = 0;
  double* sol = NULL;
  sol = (double*) malloc(m*sizeof(double));
CPLM_TIC(step2,"ECGSolve")
  // Analyze sparse matrix; choose proper kernels and workload balancing strategy
  mkl_sparse_set_double_sm_hint(mkl_Aii,maxIter,ecg.enlFac);
  mkl_sparse_optimize(mkl_Aii);
  // Allocate memory and initialize variables
  preAlps_ECGInitialize(&ecg,rhs,&rci_request);
  // Finish initialization
  mkl_ilu0_apply(mkl_Aii,ecg.R->val,ecg.P->val,m,ecg.enlFac);
  preAlps_BlockOperator(ecg.P,ecg.AP);
  // Main loop
  while (stop != 1) {
    preAlps_ECGIterate(&ecg,&rci_request);
    if (rci_request == 0) {
      preAlps_BlockOperator(ecg.P,ecg.AP);
    }
    else if (rci_request == 1) {
      preAlps_ECGStoppingCriterion(&ecg,&stop);
      if (stop == 1) break;
      if (ecg.ortho_alg == ORTHOMIN) {
        mkl_ilu0_apply(mkl_Aii,ecg.R->val,ecg.Z->val,m,ecg.enlFac);
      }
      else if (ecg.ortho_alg == ORTHODIR) {
        mkl_ilu0_apply(mkl_Aii,ecg.AP->val,ecg.Z->val,m,ecg.bs);
      }
    }
  }
  // Retrieve solution and free memory
  preAlps_ECGFinalize(&ecg,sol);
CPLM_TAC(step2)

  if (rank == 0)
    printf("=== ECG ===\n\titerations: %d\n\tnorm(res): %e\n\tbs: %d\n",
           ecg.iter,ecg.res,ecg.bs);

  /*================ Finalize ================*/

  // Free PETSc structure
  MatDestroy(&A_petsc);
  VecDestroy(&X);
  // Free MKL structure
  mkl_sparse_destroy(mkl_Aii);
  // Free arrays
  if (rhs != NULL) free(rhs);
  if (sol != NULL) free(sol);
  preAlps_OperatorFree();
CPLM_CLOSE_TIMER

  CPLM_printTimer(NULL);
  PetscFinalize();
#endif
  return 0;
}
/******************************************************************************/
