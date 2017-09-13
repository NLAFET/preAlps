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

/* ParBCG */
#include "usr_param.h"
#include "operator.h"
#include "bcg.h"
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

  // Read parameter on first process
  Usr_Param_t param = UsrParamNULL();
  UsrParamReadFromCline(&param, argc, argv);

  // Construct the operator using a CSR matrix
  ierr = OperatorBuild(&param);CHKERR(ierr);
  Mat_CSR_t A = MatCSRNULL();
  OperatorGetA(&A);

  // Read the rhs
  DVector_t rhs = DVectorNULL();
  IVector_t rowPos = IVectorNULL();
  OperatorGetRowPosPtr(&rowPos);
  DVectorMalloc(&rhs,A.info.m);
  DVectorRandom(&rhs,0);
  //  const char* rhsFilename = "rhs.txt";
  // ierr = DVectorLoadAndDistribute(rhsFilename,&rhs,&rowPos,MPI_COMM_WORLD);
  CHKERR(ierr);

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
                   param.tolerance,
                   PETSC_DEFAULT,
                   PETSC_DEFAULT,
                   param.iterMax);
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
  BCG_t bcg_solver;
  const char* caseName = "debug";
  bcg_solver.comm = MPI_COMM_WORLD;
  BCGReadParamFromFile(&bcg_solver, param.solverFilename);

#ifdef PETSC
  /*Restore the pointer*/
  double* data = NULL;
  VecGetArray(B,&data);
  rhs.nval = A.info.m;
  rhs.val = data;
#endif
  // Get local and global sizes of operator A
  int M, m;
  int rci_request = 0;
  int stop = 0;
  double* sol = NULL;
  ierr = OperatorGetSizes(&M,&m);CHKERR(ierr);
  sol = (double*) malloc(m*sizeof(double));
  // Malloc memory
  BCGMalloc(&bcg_solver,M,m,&param,"TestECG");
  // Initialize variables
  ierr = BCGInitialize(&bcg_solver,data,&rci_request);CHKERR(ierr);
  // Finish initialization
  PrecondApply(bcg_solver.precond_type,bcg_solver.R,bcg_solver.P);
  BlockOperator(bcg_solver.P,bcg_solver.AP);
  // Main loop
  while (stop != 1) {
    ierr = BCGIterate(&bcg_solver,&rci_request);
    if (rci_request == 0) {
      BlockOperator(bcg_solver.P,bcg_solver.AP);
    }
    else if (rci_request == 1) {
      ierr = BCGStoppingCriterion(&bcg_solver,&stop);
      if (stop == 1) break;
      PrecondApply(bcg_solver.precond_type,bcg_solver.R,bcg_solver.Z);
    }
  }
  // Retrieve solution
  BCGFinalize(&bcg_solver,sol);
  // Release memory
  BCGFree(&bcg_solver);

  if (rank == 0)
    printf("=== ECG ===\n\titerations: %d\n\tnorm(res): %e\n",bcg_solver.iter,bcg_solver.res);


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
