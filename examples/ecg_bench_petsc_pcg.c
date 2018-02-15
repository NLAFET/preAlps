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
/*                            AUXILIARY FUNCTIONS                             */
/******************************************************************************/
/* Use PETSc MatMatMult */
void petsc_operator_apply(Mat A, double* V, double* AV, int M, int m, int n) {
  Mat V_petsc, AV_petsc;
  MatCreateDense(PETSC_COMM_WORLD,m,PETSC_DECIDE,M,n,V,&V_petsc);
  MatCreateDense(PETSC_COMM_WORLD,m,PETSC_DECIDE,M,n,AV,&AV_petsc);
  MatMatMult(A, V_petsc, MAT_REUSE_MATRIX, PETSC_DEFAULT, &AV_petsc);
  MatDestroy(&V_petsc);MatDestroy(&AV_petsc);
}
/* Apply PETSc preconditoner */
void petsc_precond_apply(Mat P, double* V, double* W, int M, int m, int n) {
  Mat V_petsc, W_petsc;
  MatCreateSeqDense(PETSC_COMM_SELF,m,n,V,&V_petsc);
  MatCreateSeqDense(PETSC_COMM_SELF,m,n,W,&W_petsc);
  // MatCreateDense(PETSC_COMM_WORLD,m,PETSC_DECIDE,M,n,V,&V_petsc);
  // MatCreateDense(PETSC_COMM_WORLD,m,PETSC_DECIDE,M,n,W,&W_petsc);
  MatMatSolve(P,V_petsc,W_petsc);
  MatDestroy(&V_petsc);MatDestroy(&W_petsc);
}

/* Private function to print the help message */
void _print_help() {
  printf("DESCRIPTION\n");
  printf("\tSolves Ax = b using a Parallel Enlarged Conjugate Gradient."
          " A must be symmetric positive definite.\n");
  printf("USAGE\n");
  printf("\tmpirun -n nb_proc"
         " ./ecg_bench_petsc_pcg"
         " -e/--enlarging-factor int"
         " [-h/--help]"
         " [-i/--iteration-maximum int]"
         " -m/--matrix file"
         " -o/--ortho-alg int"
         " -r/--search-dir-red int"
         " [-t/--tolerance double]\n");
  printf("OPTIONS\n");
  printf("\t-e/--enlarging-factor : enlarging factor"
                                  " (cannot exceed nprocs)\n");
  printf("\t-h/--help             : print this help message\n");
  printf("\t-i/--iteration-maximum: maximum of iteration count"
                                  " (default is 1000)\n");
  printf("\t-m/--matrix           : the .mtx file containing the matrix A\n");
  printf("\t-o/--ortho-alg        : orthogonalization scheme"
                                  " (0: odir, 1: omin)\n");
  printf("\t-r/--search-dir-red   : adaptive reduction of the search"
                                  " directions (0: no, 1: yes)\n");
  printf("\t-t/--tolerance        : tolerance of the method"
                                  " (default is 1e-5)\n");
}
/******************************************************************************/

/******************************************************************************/
/*                                   MAIN                                     */
/******************************************************************************/
int main(int argc, char** argv) {
#ifdef PETSC
  PetscInitialize(&argc, &argv,NULL,NULL);
  CPLM_SetEnv();

  /*================ Initialize ================*/
  int rank, size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  PetscViewerPushFormat(PETSC_VIEWER_STDOUT_SELF,	PETSC_VIEWER_ASCII_INFO);
  // Force sequential execution on each MPI process
  // OT: I tested and it still works with OpenMP activated
  // MKL_Set_Num_Threads(1);
  /*================ Command line parser ================*/
  // Set default global parameters for both PETSc and ECG
  double tol = 1e-5;
  int maxIter = 1000;
  int enlFac = 1, ortho_alg = 0, bs_red = 0;
  const char* matrixFilename = NULL;
  // A bit dirty but it allows PETSc to parse his own parameters
  for (size_t optind = 1; optind < argc; optind++) {
    if (argv[optind][0] == '-') {
      switch (argv[optind][1]) {
        case 'e': enlFac = atoi(argv[optind+1]);break;
        case 'h': _print_help();MPI_Abort(MPI_COMM_WORLD, 1);break;
        case 'i': maxIter = atoi(argv[optind+1]);break;
        case 'm': matrixFilename = argv[optind+1]; break;
        case 'o': ortho_alg = atoi(argv[optind]);break;
        case 'r': bs_red = atoi(argv[optind+1]);break;
        case 't': tol = atof(argv[optind+1]);break;
      }
    }
  }
  // Small recap
  if (rank == 0) {
    printf("=== Parameters ===\n");
    printf("\tnrhs     : %d\n",enlFac);
    printf("\titer max : %d\n",maxIter);
    printf("\tmatrix   : %s\n",matrixFilename);
    printf("\tortho alg: %d\n",ortho_alg);
    printf("\tbs red   : %d\n",bs_red);
    printf("\ttolerance: %f\n",tol);
  }

CPLM_OPEN_TIMER
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

  /*============= Construct a normalized random rhs =============*/
  double* rhs = (double*) malloc(m*sizeof(double));
  // Set the seed of the random generator
  srand(0);
  double normb = 0.0;
  for (int i = 0; i < m; ++i) {
    rhs[i] = ((double) rand() / (double) RAND_MAX);
    normb += pow(rhs[i],2);
  }
  // Compute the norm of rhs and scale it accordingly
  MPI_Allreduce(MPI_IN_PLACE,&normb,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  normb = sqrt(normb);
  for (int i = 1; i < m; ++i)
    rhs[i] /= normb;

  /*================ Petsc solve ================*/
  Mat A_petsc, M_petsc;
  Vec X, B;
  KSP ksp;
  KSP *subksp;
  PC pc, subpc;
  int first,nlocal;

  // Set RHS
  VecCreateMPIWithArray(MPI_COMM_WORLD,1,m,M,rhs,&B);
  VecCreateMPI(MPI_COMM_WORLD,m,M,&X);
  VecAssemblyBegin(X);
  VecAssemblyBegin(X);
  VecAssemblyBegin(B);
  VecAssemblyBegin(B);
  // Set solver
  KSPCreate(MPI_COMM_WORLD,&ksp);
  CPLM_petscCreateMatFromMatCSR(&A,&A_petsc);
  KSPSetOperators(ksp,A_petsc,A_petsc);
  KSPSetType(ksp,KSPCG);
  KSPSetTolerances(ksp,tol,PETSC_DEFAULT,PETSC_DEFAULT,maxIter);
  KSPSetPCSide(ksp,PC_LEFT);
  KSPCGSetType(ksp,KSP_CG_SYMMETRIC);
  KSPSetNormType(ksp,KSP_NORM_UNPRECONDITIONED);
  KSPSetFromOptions(ksp);
  KSPSetUp(ksp);
  KSPGetPC(ksp,&pc);
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

// CPLM_TIC(step1, "KSPSolve Warm-up")
//   KSPSolve(ksp,B,X);
// CPLM_TAC(step1)
  VecSet(X,0e0);
CPLM_TIC(step2, "KSPSolve")
  KSPSolve(ksp,B,X);
CPLM_TAC(step2)

  int its = -1;
  double rnorm = 0e0;
  KSPGetIterationNumber(ksp,&its);
  KSPGetResidualNorm(ksp,&rnorm);
  if (rank == 0)
    printf("=== Petsc ===\n\titerations: %d\n\tnorm(res): %e\n",its,rnorm);

  /*================ ECG solve ================*/
  preAlps_ECG_t ecg;
  // Set parameters
  ecg.comm = MPI_COMM_WORLD;
  ecg.globPbSize = M;
  ecg.locPbSize = m;
  ecg.maxIter = maxIter;
  ecg.enlFac = enlFac;
  ecg.tol = tol;
  ecg.ortho_alg = (ortho_alg == 0 ? ORTHODIR : ORTHOMIN);
  ecg.bs_red = (bs_red == 0 ? NO_BS_RED : ADAPT_BS);
  // Petsc variables
  Mat P_petsc, AP_petsc, R_petsc, Z_petsc;
  /* Restore the pointer */
  VecGetArray(B,&rhs);
  // Get local and global sizes of operator A
  int rci_request = 0;
  int stop = 0;
  double* sol = NULL;
  int* bs = NULL; // block size
  double* res = NULL; // residual
  sol = (double*) malloc(m*sizeof(double));
  bs = (int*) calloc(maxIter,sizeof(int));
  res = (double*) calloc(maxIter,sizeof(double));
//CPLM_TIC(step3,"ECGSolve Warm-up")
  // Allocate memory and initialize variables
  preAlps_ECGInitialize(&ecg,rhs,&rci_request);
  // Finish initialization
  preAlps_BlockJacobiApply(ecg.R,ecg.P);
  preAlps_BlockOperator(ecg.P,ecg.AV);
  // Main loop
  while (stop != 1) {
    preAlps_ECGIterate(&ecg,&rci_request);
    if (rci_request == 0) {
      preAlps_BlockOperator(ecg.P,ecg.AP);
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
  // Just get the solution but do not free internal memory
  _preAlps_ECGWrapUp(&ecg,sol);
//CPLM_TAC(step3)
  int mem_pool_size;
  if (ortho_alg == 0)
    mem_pool_size = 7*m*enlFac + 3*enlFac*enlFac;
  else
    mem_pool_size = 5*m*enlFac + 2*enlFac*enlFac;
  memset(ecg.work,0,mem_pool_size*sizeof(double));

  // In case ksp_monitor is set we also print ECG Residual
  char trash[1024];
  PetscBool verb;
  PetscOptionsGetString(NULL,NULL,"-ksp_monitor",
                        trash,PETSC_MAX_PATH_LEN,&verb);

CPLM_TIC(step4,"ECGSolve")
  // Initialize variables
  CPLM_TIC(step5, "        initialization")
  rci_request = 0; stop = 0;
  CPLM_resetTimer();
  // Reset internal parameters but do not reallocate memory
  _preAlps_ECGReset(&ecg,rhs,&rci_request);
  if (rank == 0 && verb == PETSC_TRUE)
    printf("%3d ECG Residual norm %.12e\n",ecg.iter, ecg.res);
  // Finish initialization
  petsc_precond_apply(M_petsc,ecg.R_p,ecg.P_p, M, m, enlFac);
  //preAlps_BlockJacobiApply(ecg.R,ecg.P);
  petsc_operator_apply(A_petsc, ecg.P_p, ecg.AP_p, M, m, enlFac);
  //preAlps_BlockOperator(ecg.P,ecg.AV);
  CPLM_TAC(step5)
  // Main loop
  while (stop != 1) {
    CPLM_TIC(step6, "        iteration")
    preAlps_ECGIterate(&ecg,&rci_request);
    CPLM_TAC(step6)
    if (rci_request == 0) {
      CPLM_TIC(step7, "        operator")
      petsc_operator_apply(A_petsc, ecg.P_p, ecg.AP_p, M, m, ecg.bs);
      //preAlps_BlockOperator(ecg.P,ecg.AP);
      CPLM_TAC(step7)
    }
    else if (rci_request == 1) {
      CPLM_TIC(step8, "        convergence test")
      preAlps_ECGStoppingCriterion(&ecg,&stop);
      bs[ecg.iter] = ecg.bs;
      res[ecg.iter] = ecg.res/ecg.normb;
      if (rank == 0 && verb == PETSC_TRUE)
        printf("%3d ECG Residual norm %.12e\n",ecg.iter, ecg.res);
      CPLM_TAC(step8)
      if (stop == 1) break;
      CPLM_TIC(step9, "        precond")
      if (ecg.ortho_alg == ORTHOMIN) {
        petsc_precond_apply(M_petsc, ecg.R_p, ecg.Z_p, M, m, enlFac);
        //preAlps_BlockJacobiApply(ecg.R,ecg.Z);
      }
      else if (ecg.ortho_alg == ORTHODIR) {
        petsc_precond_apply(M_petsc, ecg.AP_p, ecg.Z_p, M, m, ecg.bs);
        //preAlps_BlockJacobiApply(ecg.AP,ecg.Z);
      }
      CPLM_TAC(step9)
    }
  }
  // Retrieve solution and free memory
  CPLM_TIC(step10, "        finalize")
  preAlps_ECGFinalize(&ecg,sol);
  CPLM_TAC(step10)
CPLM_TAC(step4)

  if (rank == 0) {
    printf("=== ECG ===\n");
    printf("\titerations: %d\n",ecg.iter);
    printf("\tnorm(res) : %e\n",ecg.res);
    printf("\tblock size: %d\n",ecg.bs);
  }

  /*================ Finalize ================*/

  // Free PETSc structure
  MatDestroy(&A_petsc);
  VecDestroy(&X);
  KSPDestroy(&ksp);

  // Free arrays
  if (rhs != NULL) free(rhs);
  if (sol != NULL) free(sol);
  preAlps_OperatorFree();
CPLM_CLOSE_TIMER

  PetscOptionsGetString(NULL,NULL,"-print_timer",trash,
                        PETSC_MAX_PATH_LEN,&verb);
  if (verb == PETSC_TRUE) CPLM_printTimer(NULL);
  PetscFinalize();
#endif
  return 0;
}
/******************************************************************************/
