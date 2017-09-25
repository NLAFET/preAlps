/******************************************************************************/
/* Author     : Olivier Tissot , Simplice Donfack                             */
/* Creation   : 2016/06/23                                                    */
/* Description: Main file of Par(allel) B(lock) C(onjugate) G(gradient)       */
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
#include "ecg.h"
/******************************************************************************/

/******************************************************************************/
/*                                    CODE                                    */
/******************************************************************************/
int main(int argc, char** argv) {


  CPALAMEM_Init(&argc, &argv);
OPEN_TIMER

  /*================ Initialize ================*/
  int rank, size, ierr;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Force sequential execution on each MPI process
  // OT: I tested and it still works with OpenMP activated
  MKL_Set_Num_Threads(1);

  // Construct the operator using a CSR matrix
  const char* matrixFilename = argv[1];
  Mat_CSR_t A = MatCSRNULL();
  int M, m;
  int* rowPos = NULL;
  int* colPos = NULL;
  int* dep = NULL;
  int sizeRowPos, sizeColPos, sizeDep;
  OperatorBuild(matrixFilename,MPI_COMM_WORLD);
  OperatorGetA(&A);
  OperatorGetSizes(&M,&m);
  OperatorGetRowPosPtr(&rowPos,&sizeRowPos);
  OperatorGetColPosPtr(&colPos,&sizeColPos);
  OperatorGetDepPtr(&dep,&sizeDep);

  // Construct the preconditioner
  Prec_Type_t precond_type = PREALPS_BLOCKJACOBI;
  PrecondCreate(precond_type,
                &A,
                rowPos,
                sizeRowPos,
                colPos,
                sizeColPos,
                dep,
                sizeDep);

  // Construct the rhs
  DVector_t rhs = DVectorNULL();
  DVectorMalloc(&rhs,A.info.m);
  DVectorRandom(&rhs,0);
  CHKERR(ierr);

  double tol = 1e-5;
  int maxIter = 1000;
#ifdef PETSC
  /*================ Petsc solve ================*/
  Mat A_petsc;
  Vec X, B;
  KSP ksp;
  KSP *subksp;
  PC pc, subpc;
  int first,nlocal;

  // Set RHS
  VecCreateMPIWithArray(MPI_COMM_WORLD,1,A.info.m,A.info.M,rhs.val,&B);
  VecCreateMPI(MPI_COMM_WORLD,A.info.m,A.info.M,&X);
  VecAssemblyBegin(X);
  VecAssemblyBegin(X);
  VecAssemblyBegin(B);
  VecAssemblyBegin(B);
  // Set solver
  KSPCreate(MPI_COMM_WORLD,&ksp);
  petscCreateMatFromMatCSR(&A,&A_petsc);
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
  }

TIC(step1,"KSPSolve")
  KSPSolve(ksp,B,X);
TAC(step1)

  int its = -1;
  double rnorm = 0e0;
  KSPGetIterationNumber(ksp,&its);
  KSPGetResidualNorm(ksp,&rnorm);
  if (rank == 0)
    printf("=== Petsc ===\n\titerations: %d\n\tnorm(res): %e\n",its,rnorm);
  KSPDestroy(&ksp);
#endif

  /*================ BCG solve ================*/
  ECG_t ecg;
  // Set parameters
  ecg.comm = MPI_COMM_WORLD;  /* MPI Communicator */
  ecg.globPbSize = M;         /* Size of the global problem */
  ecg.locPbSize = m;          /* Size of the local problem */
  ecg.maxIter = maxIter;      /* Maximum number of iterations */
  ecg.enlFac = 2;             /* Enlarging factor */
  ecg.tol = tol;              /* Tolerance of the method */
  ecg.ortho_alg = ORTHODIR;   /* Orthogonalization algorithm */
  ecg.bs_red = NO_BS_RED;     /* Only NO_BS_RED implemented !! */

#ifdef PETSC
  /*Restore the pointer*/
  double* data = NULL;
  VecGetArray(B,&data);
  rhs.nval = A.info.m;
  rhs.val = data;
#endif
  // Get local and global sizes of operator A
  int rci_request = 0;
  int stop = 0;
  double* sol = NULL;
  sol = (double*) malloc(m*sizeof(double));
  // Allocate memory and initialize variables
  ierr = ECGInitialize(&ecg,rhs.val,&rci_request);CHKERR(ierr);
  // Finish initialization
  PrecondApply(precond_type,ecg.R,ecg.P);
  BlockOperator(ecg.P,ecg.AP);
  // Main loop
  while (stop != 1) {
    ierr = ECGIterate(&ecg,&rci_request);
    if (rci_request == 0) {
      BlockOperator(ecg.P,ecg.AP);
    }
    else if (rci_request == 1) {
      ierr = ECGStoppingCriterion(&ecg,&stop);
      if (stop == 1) break;
      if (ecg.ortho_alg == ORTHOMIN)
        PrecondApply(precond_type,ecg.R,ecg.Z);
      else if (ecg.ortho_alg == ORTHODIR)
        PrecondApply(precond_type,ecg.AP,ecg.Z);
    }
  }
  // Retrieve solution and free memory
  ECGFinalize(&ecg,sol);

  if (rank == 0)
    printf("=== ECG ===\n\titerations: %d\n\tnorm(res): %e\n",ecg.iter,ecg.res);


  /*================ Finalize ================*/

#ifdef PETSC
  MatDestroy(&A_petsc);
  VecDestroy(&X);
#endif

  DVectorFree(&rhs);
  OperatorFree();
CLOSE_TIMER
  CPALAMEM_Finalize();
  return ierr;
}
/******************************************************************************/
