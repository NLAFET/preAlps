/******************************************************************************/
/* Author     : Olivier Tissot , Simplice Donfack                             */
/* Creation   : 2017/10/12                                                    */
/* Description: ECG calling example with PETSc MatMatMult instead of          */
/*              preAlps_BlockOPerator                                         */
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
  int* dep = NULL;
  int sizeRowPos, sizeColPos, sizeDep;
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
  // Get neighbours (dependencies)
  preAlps_OperatorGetDepPtr(&dep,&sizeDep);

  /*======== Construct the preconditioner ========*/
  preAlps_BlockJacobiCreate(&A,
                            rowPos,
                            sizeRowPos,
                            colPos,
                            sizeColPos);

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

  /*
    Loop over the local blocks, setting various KSP options
    for each block.
  */
  for (int i=0; i<nlocal; i++) {
    KSPGetPC(subksp[i],&subpc);
    PCSetType(subpc,PCCHOLESKY);
    //PCFactorSetMatSolverPackage(subpc,MATSOLVERMKL_PARDISO);
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
  ecg.ortho_alg = ORTHOMIN;   /* Orthogonalization algorithm */
  ecg.bs_red = NO_BS_RED;     /* Only NO_BS_RED implemented !! */

  // Restore the pointer
  VecGetArray(B,&rhs);
  // Get local and global sizes of operator A
  int rci_request = 0;
  int stop = 0;
  double* sol = NULL;
  sol = (double*) malloc(m*sizeof(double));
CPLM_TIC(step2,"ECGSolve")
  // Allocate memory and initialize variables
  preAlps_ECGInitialize(&ecg,rhs,&rci_request);
  // Finish initialization
  preAlps_BlockJacobiApply(ecg.R,ecg.P);
  Mat P_petsc, AP_petsc;
  MatCreateDense(PETSC_COMM_WORLD,
                 ecg.locPbSize,
                 PETSC_DECIDE,
                 ecg.globPbSize,
                 ecg.enlFac,
                 ecg.P->val,
                 &P_petsc);
  MatCreateDense(PETSC_COMM_WORLD,
                 ecg.locPbSize,
                 PETSC_DECIDE,
                 ecg.globPbSize,
                 ecg.enlFac,
                 ecg.AP->val,
                 &AP_petsc);
  MatMatMult(A_petsc, P_petsc, MAT_REUSE_MATRIX, PETSC_DEFAULT, &AP_petsc);
  MatDestroy(&P_petsc);
  MatDestroy(&AP_petsc);
  // Main loop
  while (stop != 1) {
    preAlps_ECGIterate(&ecg,&rci_request);
    if (rci_request == 0) {
      //preAlps_BlockOperator(ecg.P,ecg.AP);
      MatCreateDense(PETSC_COMM_WORLD,
                     ecg.locPbSize,
                     PETSC_DECIDE,
                     ecg.globPbSize,
                     ecg.enlFac,
                     ecg.P->val,
                     &P_petsc);
      MatCreateDense(PETSC_COMM_WORLD,
                     ecg.locPbSize,
                     PETSC_DECIDE,
                     ecg.globPbSize,
                     ecg.enlFac,
                     ecg.AP->val,
                     &AP_petsc);
      MatMatMult(A_petsc, P_petsc, MAT_REUSE_MATRIX, PETSC_DEFAULT, &AP_petsc);
      MatDestroy(&P_petsc);
      MatDestroy(&AP_petsc);
    }
    else if (rci_request == 1) {
      preAlps_ECGStoppingCriterion(&ecg,&stop);
      if (stop == 1) break;
      if (ecg.ortho_alg == ORTHOMIN)
        preAlps_BlockJacobiApply(ecg.R,ecg.Z);
      else if (ecg.ortho_alg == ORTHODIR)
        preAlps_BlockJacobiApply(ecg.AP,ecg.Z);
    }
  }
  // Retrieve solution and free memory
  preAlps_ECGFinalize(&ecg,sol);
CPLM_TAC(step2)

  if (rank == 0)
    printf("=== ECG ===\n\titerations: %d\n\tnorm(res): %e\n",ecg.iter,ecg.res);


  /*================ Finalize ================*/

  MatDestroy(&A_petsc);
  VecDestroy(&X);
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
