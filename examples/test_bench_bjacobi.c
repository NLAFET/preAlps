/**
 * \file    test_ecg_petsc_op.c
 * \author  Olivier Tissot
 * \date    2016/06/23
 * \brief   Benchmark of different routines for applying a bjacobi preconditioner
 */

/******************************************************************************/
/*                                  INCLUDE                                   */
/******************************************************************************/
/* STD */
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <math.h>
/* MPI */
#include <mpi.h>
/* MKL */
#include <mkl.h>

#ifdef PETSC
#include <preAlps_cplm_petsc_interface.h>
/* Petsc */
#include <petscksp.h>
#endif

/* preAlps */
#include "preAlps_cplm_matdense.h"
#include "operator.h"
#include "block_jacobi.h"
/******************************************************************************/

/******************************************************************************/
/*                            AUXILIARY FUNCTIONS                             */
/******************************************************************************/

#ifdef PETSC
/** \brief Simple wrapper for applying PETSc preconditoner */
void petsc_precond_apply(Mat P, double* V, double* W, int M, int m, int n) {
  Mat V_petsc, W_petsc;
  MatCreateSeqDense(PETSC_COMM_SELF,m,n,V,&V_petsc);
  MatCreateSeqDense(PETSC_COMM_SELF,m,n,W,&W_petsc);
  MatMatSolve(P,V_petsc,W_petsc);
  MatDestroy(&V_petsc);MatDestroy(&W_petsc);
}
#endif

/* Private function to print the help message */
void _print_help() {
  printf("DESCRIPTION\n");
  printf("\tBenchmark preAlps and PETSc bjacobi routines.\n");
  printf("USAGE\n");
  printf("\tmpirun test_bench_bjacobi -n nb_proc"
         " -m/--matrix file\n");
  printf("OPTIONS\n");
  printf("\t-m/--matrix           : the .mtx file containing the matrix A\n");

}
/******************************************************************************/

/******************************************************************************/
/*                                   MAIN                                     */
/******************************************************************************/
int main(int argc, char** argv) {
#ifdef PETSC
  PetscInitialize(&argc, &argv,NULL,NULL);

  /*================ Initialize ================*/
  int rank, size;
  double trash_t;
  int maxCol = 28;
  int nrepet = 10;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // Force sequential execution on each MPI process
  // MKL_Set_Num_Threads(1);
  /*================ Command line parser ================*/
  // Set default global parameters for both PETSc and ECG
  const char* matrixFilename = NULL;
  // A bit dirty but it allows PETSc to parse his own parameters
  for (size_t optind = 1; optind < argc; optind++) {
    if (argv[optind][0] == '-') {
      switch (argv[optind][1]) {
        case 'm': matrixFilename = argv[optind+1]; break;
      }
    }
  }
  // Small recap
  if (rank == 0) {
    printf("=== Parameters ===\n");
    printf("\tmatrix: %s\n",matrixFilename);
    printf("\tnrepet: %d\n",nrepet);
    printf("\tmaxCol: %d\n",maxCol);
  }

  /*======== Construct the operator using a CSR matrix ========*/
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

  if (rank == 0) {
    printf("=== Matrix informations ===\n\tsize: %d\n\tnnz : %d\n", A.info.M, A.info.nnz);
  }

  /*======== Construct the preconditioner ========*/
  preAlps_BlockJacobiCreate(&A,rowPos,sizeRowPos,colPos,sizeColPos);

  /*============= Construct a random rhs =============*/
  double* rhs = (double*) malloc(m*maxCol*sizeof(double));
  // Set the seed of the random generator
  srand(0);
  for (int i = 0; i < m*maxCol; ++i) {
    rhs[i] = ((double) rand() / (double) RAND_MAX);
  }
  double* res = (double*) malloc(m*maxCol*sizeof(double));

  /*================ Petsc set-up ================*/
  Mat A_petsc, M_petsc;
  KSP ksp;
  KSP *subksp;
  PC pc, subpc;
  int first,nlocal;

  // Set RHS
  Vec rhs_petsc;
  Vec res_petsc;
  VecCreateMPIWithArray(MPI_COMM_WORLD,1,m,M,rhs,&rhs_petsc);
  VecCreateMPIWithArray(MPI_COMM_WORLD,1,m,M,res,&res_petsc);
  VecAssemblyBegin(res_petsc);
  VecAssemblyBegin(res_petsc);
  VecAssemblyBegin(rhs_petsc);
  VecAssemblyBegin(rhs_petsc);

  // Set solver
  KSPCreate(MPI_COMM_WORLD,&ksp);
  CPLM_petscCreateMatFromMatCSR(&A,&A_petsc);
  KSPSetOperators(ksp,A_petsc,A_petsc);
  KSPSetType(ksp,KSPCG);
//  KSPSetTolerances(ksp,tol,PETSC_DEFAULT,PETSC_DEFAULT,maxIter);
  KSPSetPCSide(ksp,PC_LEFT);
  KSPCGSetType(ksp,KSP_CG_SYMMETRIC);
  KSPSetFromOptions(ksp);
  KSPSetUp(ksp);
  KSPGetPC(ksp,&pc);
  PCSetType(pc,PCBJACOBI);
  PCBJacobiGetSubKSP(pc,&nlocal,&first,&subksp);

  // Loop over the local blocks, setting various KSP options
  // for each block.
  // printf("nlocal: %d\n",nlocal);
  for (int i=0; i<nlocal; i++) {
    KSPGetPC(subksp[i],&subpc);
    PCSetUp(subpc);
    PCFactorGetMatrix(subpc,&M_petsc);
    /* PCFactorSetMatSolverPackage(subpc,MATSOLVERMKL_PARDISO); */
  }


  /*================ Petsc benchmark ================*/
  double* petsc_t = malloc(maxCol*sizeof(double)); // a bit oversized but it does not really matter
  // 1) built-in apply
  PCApply(pc,rhs_petsc,res_petsc);
  MPI_Barrier(MPI_COMM_WORLD);
  trash_t = MPI_Wtime();
  for (int i = 0; i < nrepet; ++i) {
    PCApply(pc,rhs_petsc,res_petsc);
    MPI_Barrier(MPI_COMM_WORLD);
  }
  petsc_t[maxCol - 1] = MPI_Wtime() - trash_t;
  // Retrieve the pointers
  VecGetArray(rhs_petsc,&rhs);
  VecGetArray(res_petsc,&res);

  // 2) several rhs
  petsc_precond_apply(M_petsc, rhs, res, M, m, 1);
  MPI_Barrier(MPI_COMM_WORLD);
  trash_t = MPI_Wtime();
  for (int i = 0; i < nrepet; ++i) {
    petsc_precond_apply(M_petsc, rhs, res, M, m, 1);
    MPI_Barrier(MPI_COMM_WORLD);
  }
  petsc_t[0] = MPI_Wtime() - trash_t;
  for (int t = 1; t <= (int) maxCol/2; t++) {
    petsc_precond_apply(M_petsc, rhs, res, M, m, 2*t);
    MPI_Barrier(MPI_COMM_WORLD);
    trash_t = MPI_Wtime();
    for (int i = 0; i < nrepet; ++i) {
      petsc_precond_apply(M_petsc, rhs, res, M, m, 2*t);
      MPI_Barrier(MPI_COMM_WORLD);
    }
    petsc_t[t] = MPI_Wtime() - trash_t;
  }

  if (rank == 0) {
    printf("=== Petsc timings ===\n");
    printf("\trhs\ttime\t\ttime/rhs\n");
    printf("trsv\t%2d\t%e\t%e\n",1,petsc_t[maxCol-1],petsc_t[maxCol-1]);
    printf("trsm\t%2d\t%e\t%e\n",1,petsc_t[0],petsc_t[0]);
    for (int t = 1; t <= (int) maxCol/2; t++) {
      printf("\t%2d\t%e\t%e\n",2*t,petsc_t[t],petsc_t[t]/(2*t));
    }
  }

  /*================ preAlps benchmark ================*/
  double* preAlps_t = malloc(maxCol*sizeof(double)); // a bit oversized but it does not really matter
  CPLM_Mat_Dense_t rhs_preAlps = CPLM_MatDenseNULL();
  CPLM_Mat_Dense_t res_preAlps = CPLM_MatDenseNULL();
  CPLM_MatDenseSetInfo(&rhs_preAlps, M, 1, m, 1, COL_MAJOR);
  rhs_preAlps.val = rhs;
  CPLM_MatDenseSetInfo(&res_preAlps, M, 1, m, 1, COL_MAJOR);
  res_preAlps.val = res;
  preAlps_BlockJacobiApply(&rhs_preAlps,&res_preAlps);
  MPI_Barrier(MPI_COMM_WORLD);
  trash_t = MPI_Wtime();
  for (int i = 0; i < nrepet; ++i) {
    preAlps_BlockJacobiApply(&rhs_preAlps,&res_preAlps);
    MPI_Barrier(MPI_COMM_WORLD);
  }
  preAlps_t[0] = MPI_Wtime() - trash_t;
  for (int t = 1; t <= (int) maxCol/2; t++) {
    CPLM_MatDenseSetInfo(&rhs_preAlps, M, 2*t, m, 2*t, COL_MAJOR);
    rhs_preAlps.val = rhs;
    CPLM_MatDenseSetInfo(&res_preAlps, M, 2*t, m, 2*t, COL_MAJOR);
    res_preAlps.val = res;
    preAlps_BlockJacobiApply(&rhs_preAlps,&res_preAlps);
    MPI_Barrier(MPI_COMM_WORLD);
    trash_t = MPI_Wtime();
    for (int i = 0; i < nrepet; ++i) {
      preAlps_BlockJacobiApply(&rhs_preAlps,&res_preAlps);
      MPI_Barrier(MPI_COMM_WORLD);
    }
    preAlps_t[t] = MPI_Wtime() - trash_t;
  }

  if (rank == 0) {
    printf("=== ECG timings ===\n");
    printf("\trhs\ttime\t\ttime/rhs\n");
    printf("trsm\t%2d\t%e\t%e\n",1,preAlps_t[0], preAlps_t[0]);
    for (int t = 1; t <= (int) maxCol/2; t++) {
      printf("\t%2d\t%e\t%e\n",2*t,preAlps_t[t],preAlps_t[t]/(2*t));
    }
  }
  /*================ Finalize ================*/

  // Free PETSc structure
  MatDestroy(&A_petsc);
  KSPDestroy(&ksp);

  // Free arrays
  if (rhs != NULL) free(rhs);
  if (res != NULL) free(res);
  preAlps_OperatorFree();

  PetscFinalize();
#endif
  return 0;
}
