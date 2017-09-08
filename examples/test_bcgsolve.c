/******************************************************************************/
/* Author     : Olivier Tissot                                                */
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

#ifdef USE_PETSC
//#include <petsc_interface.h>
/* Petsc */
#include <petscksp.h>
#endif

/* ParBCG */
#include "usr_param.h"
#include "solver.h"
#include "operator.h"
#include "bcg.h"
#include <cpalamem_macro.h>
#include <cpalamem_instrumentation.h>
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
  ierr = readMatCSRUsingKWay(&param);CHKERR(ierr);
  Mat_CSR_t A = MatCSRNULL();
  OperatorGetA(&A);

  // Read the rhs
  DVector_t rhs = DVectorNULL();
  IVector_t rowPos = IVectorNULL();
  IVectorPrintf("rowPosMain",&rowPos);
  OperatorGetRowPosPtr(&rowPos);
  const char* rhsFilename = "rhs.txt";
  DVectorMalloc(&rhs,A.info.m);
  DVectorRandom(&rhs,0);
  // ierr = DVectorLoadAndDistribute(rhsFilename,&rhs,&rowPos,MPI_COMM_WORLD);
  CHKERR(ierr);
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
  KSPSetTolerances(ksp,1e-6,PETSC_DEFAULT,PETSC_DEFAULT,100000);
  KSPSetPCSide(ksp,PC_LEFT);
  KSPCGSetType(ksp,KSP_CG_SYMMETRIC);
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

TIC
  KSPSolve(ksp,B,X);
TAC("KSPSolve")

  int its = -1;
  double rnorm = 0e0;
  KSPGetIterationNumber(ksp,&its);
  KSPGetResidualNorm(ksp,&rnorm);
  if (rank == 0)
    printf("=== Petsc ===\n\titerations: %d\n\tnorm(res): %e\n",its,rnorm);
  KSPDestroy(&ksp);

  /*================ BCG solve ================*/
  BCG_t bcg_solver;
  const char* caseName = "debug";
  bcg_solver.solver.comm = MPI_COMM_WORLD;
  BCGReadParamFromFile(&bcg_solver, param.solverFilename);
  // double* data = NULL;
  // VecGetArray(B,&data);
  // DVector_t rhs = DVectorNULL();
  // rhs.nval = A.info.m;
  // rhs.val = data;
TIC
  BCGSolve(&bcg_solver, &rhs, &param, caseName);
TAC("BCGSolve")
  if (rank == 0)
    printf("=== BCG ===\n\titerations: %d\n\tnorm(res): %e\n",bcg_solver.iter,bcg_solver.solver.finalRes);

  /*================ Finalize ================*/
  // VecRestoreArray(B,&data);
  MatDestroy(&A_petsc);
  // VecDestroy(&B);
  DVectorFree(&rhs);
  VecDestroy(&X);
  OperatorFree();
CLOSE_TIMER
  CPALAMEM_Finalize();
  return ierr;
}
/******************************************************************************/
