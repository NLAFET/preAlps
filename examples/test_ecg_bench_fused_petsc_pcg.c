/**
 * \file    test_ecg_bench_petsc_pcg.c
 * \author  Olivier Tissot
 * \date    2016/06/23
 * \brief   Comparison between ECG and PETSc PCG
 *
 * \details Be carefull to set MATSOLVERPACKAGE to MATSOLVERMKL_PARDISO
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

/* CPaLAMeM */
#include <preAlps_cplm_timing.h>

#ifdef PETSC
#include <preAlps_cplm_petsc_interface.h>
/* Petsc */
#include <petscksp.h>
#endif

/* preAlps */
#include "operator.h"
#include "block_jacobi.h"
#include "ecg.h"

/* Command line parser */
#include <getopt.h>
#include <ctype.h>
/******************************************************************************/

/******************************************************************************/
/*                                    CODE                                    */
/******************************************************************************/
/* Private function to print the help message */
void _print_help() {
  printf("DESCRIPTION\n");
  printf("\tSolves Ax = b using a Parallel Enlarged Conjugate Gradient."
          " A must be symmetric positive definite.\n");
  printf("USAGE\n");
  printf("\tmpirun -n nb_proc"
         " ./test_ecg_bench_petsc_pcg"
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

#ifdef PETSC
/** \brief Simple wrapper to PETSc MatMatMult */
void petsc_operator_apply(Mat A, double* V, double* AV, int M, int m, int n) {
  Mat V_petsc, AV_petsc;
  MatCreateDense(PETSC_COMM_WORLD,m,PETSC_DECIDE,M,n,V,&V_petsc);
  MatCreateDense(PETSC_COMM_WORLD,m,PETSC_DECIDE,M,n,AV,&AV_petsc);
  MatMatMult(A, V_petsc, MAT_REUSE_MATRIX, PETSC_DEFAULT, &AV_petsc);
  MatDestroy(&V_petsc);MatDestroy(&AV_petsc);
}
#endif


int main(int argc, char** argv) {
#ifdef PETSC
  PetscInitialize(&argc, &argv,NULL,NULL);

  /*================ Command line parser ================*/
  int c;
  static struct option long_options[] = {
    {"enlarging-factor" , required_argument, NULL, 'e'},
    {"help"             , no_argument      , NULL, 'h'},
    {"iteration-maximum", optional_argument, NULL, 'i'},
    {"matrix"           , required_argument, NULL, 'm'},
    {"ortho-alg"        , required_argument, NULL, 'o'},
    {"search-dir-red"   , required_argument, NULL, 'r'},
    {"tolerance"        , optional_argument, NULL, 't'},
    {NULL               , 0                , NULL, 0}
  };

  int opterr = 0;
  int option_index = 0;

  // Set global parameters for both PETSc and ECG
  double tol = 1e-5;
  int maxIter = 1000;
  int enlFac = 1, ortho_alg = 0, bs_red = 0;
  const char* matrixFilename = NULL;
  while ((c = getopt_long(argc, argv, "e:hi:m:o:r:t:", long_options, &option_index)) != -1)
    switch (c) {
      case 'e':
        enlFac = atoi(optarg);
        break;
      case 'h':
        _print_help();
        MPI_Abort(MPI_COMM_WORLD, opterr);
      case 'i':
        if (optarg != NULL)
          maxIter = atoi(optarg);
        break;
      case 'm':
        if (optarg == NULL) {
          _print_help();
          MPI_Abort(MPI_COMM_WORLD, opterr);
        }
        else
          matrixFilename = optarg;
        break;
      case 'o':
        ortho_alg = atoi(optarg);
        break;
      case 'r':
        bs_red = atoi(optarg);
        break;
      case 't':
        if (optarg != NULL)
          tol = atof(optarg);
        break;
      case '?':
        if (optopt == 'e'
            || optopt == 'i'
            || optopt == 'm'
            || optopt == 'o'
            || optopt == 'r'
            || optopt == 't')
          fprintf (stderr, "Option -%c requires an argument.\n", optopt);
        else if (isprint (optopt))
          fprintf (stderr, "Unknown option `-%c'.\n", optopt);
        else
          fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
        _print_help();
        MPI_Abort(MPI_COMM_WORLD, opterr);
      default:
        MPI_Abort(MPI_COMM_WORLD, opterr);
    }

  /*================ Initialize ================*/
  int rank, size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // Timings
  double trash_t, trash_tg;
  double petsc_t = 0.E0;
  double buildop_t = 0.E0, buildprec_t = 0.E0;
  double tot_t = 0.E0, op_t = 0.E0, prec_t = 0.E0;

  // Force sequential execution on each MPI process
  // OT: I tested and it still works with OpenMP activated
  //MKL_Set_Num_Threads(1);

  /*======== Construct the operator using a CSR matrix ========*/
  CPLM_Mat_CSR_t A = CPLM_MatCSRNULL();
  int M, m;
  int* rowPos = NULL;
  int* colPos = NULL;
  int sizeRowPos, sizeColPos;
  // Read and partition the matrix
  trash_t = MPI_Wtime();
  preAlps_OperatorBuild(matrixFilename,MPI_COMM_WORLD);
  buildop_t += MPI_Wtime() - trash_t;
  // Get the CSR structure of A
  preAlps_OperatorGetA(&A);
  // Get the sizes of A
  preAlps_OperatorGetSizes(&M,&m);
  // Get row partitioning of A
  preAlps_OperatorGetRowPosPtr(&rowPos,&sizeRowPos);
  // Get col partitioning induced by this row partitioning
  preAlps_OperatorGetColPosPtr(&colPos,&sizeColPos);

  /*======== Construct the preconditioner ========*/
  trash_t = MPI_Wtime();
  preAlps_BlockJacobiCreate(&A,rowPos,sizeRowPos,colPos,sizeColPos);
  buildprec_t += MPI_Wtime() - trash_t;

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
  Mat A_petsc;
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
  KSPGetPC(ksp,&pc);
  PCSetType(pc,PCBJACOBI);
  KSPSetUp(ksp);
  PCBJacobiGetSubKSP(pc,&nlocal,&first,&subksp);

  // Loop over the local blocks, setting various KSP options
  // for each block.
  for (int i=0; i<nlocal; i++) {
    KSPGetPC(subksp[i],&subpc);
    PCSetType(subpc,PCCHOLESKY);
    /* PCFactorSetMatSolverPackage(subpc,MATSOLVERMKL_PARDISO); */
  }

  VecSet(X,0e0);
  trash_t = MPI_Wtime();
  KSPSolve(ksp,B,X);
  MPI_Barrier(MPI_COMM_WORLD);
  petsc_t += MPI_Wtime() - trash_t;

  int its = -1;
  double rnorm = 0e0;
  KSPGetIterationNumber(ksp,&its);
  KSPGetResidualNorm(ksp,&rnorm);
  KSPDestroy(&ksp);

  /*================ ECG solve ================*/
  preAlps_ECG_t ecg;
  // Set parameters
  ecg.comm = MPI_COMM_WORLD;
  ecg.globPbSize = M;
  ecg.locPbSize = m;
  ecg.maxIter = maxIter;
  ecg.enlFac = enlFac;
  ecg.tol = tol;
  ecg.ortho_alg = ORTHODIR_FUSED;
  ecg.bs_red = (bs_red == 0 ? NO_BS_RED : ADAPT_BS);
  /* Restore the pointer */
  VecGetArray(B,&rhs);
  // Get local and global sizes of operator A
  int rci_request = 0;
  int stop = 0;
  double* sol = NULL;
  sol = (double*) malloc(m*sizeof(double));

  // Allocate memory and initialize variables
  trash_tg = MPI_Wtime();
  preAlps_ECGInitialize(&ecg,rhs,&rci_request);
  // Finish initialization
  trash_t = MPI_Wtime();
  preAlps_BlockJacobiApply(ecg.R,ecg.P);
  prec_t += MPI_Wtime() - trash_t;
  // Main loop
  while (rci_request != 1) {
    trash_t = MPI_Wtime();
    petsc_operator_apply(A_petsc, ecg.P_p, ecg.AP_p, M, m, enlFac);
    op_t += MPI_Wtime() - trash_t;
    trash_t = MPI_Wtime();
    preAlps_BlockJacobiApply(ecg.AP,ecg.Z);
    prec_t += MPI_Wtime() - trash_t;
    preAlps_ECGIterate(&ecg,&rci_request);
  }
  // Retrieve solution and free memory
  preAlps_ECGFinalize(&ecg,sol);
  tot_t += MPI_Wtime() - trash_tg;


  /*================== Post-processing ==================*/
  /* Global timings */
  MPI_Allreduce(MPI_IN_PLACE, &buildop_t, 1, MPI_DOUBLE, MPI_MAX, ecg.comm);
  MPI_Allreduce(MPI_IN_PLACE, &buildprec_t, 1, MPI_DOUBLE, MPI_MAX, ecg.comm);
  /* ECG timing and the corresponding rank */
  struct {
    double value;
    int rank;
  } pair_tot_t_rank;
  pair_tot_t_rank.value = tot_t;
  pair_tot_t_rank.rank = rank;
  MPI_Allreduce(MPI_IN_PLACE, &pair_tot_t_rank, 1, MPI_DOUBLE_INT, MPI_MAXLOC, ecg.comm);

  if (rank == pair_tot_t_rank.rank) {
    printf("=== SUMMARY ===\n");
    printf("\t# mpi        : %d\n",size);
    printf("\tmax iter     : %d\n",maxIter);
    printf("\ttolerance    : %e\n",tol);
    printf("== matrix informations ==\n");
    printf("\tmatrix       : %s\n",matrixFilename);
    printf("\tsize         : %d\n", A.info.M);
    printf("\tnnz          : %d\n", A.info.nnz);
    printf("Timing:\n");
    printf("\tbuild op     : %e s\n",buildop_t);
    printf("\tbuild precond: %e s\n\n",buildprec_t);
    printf("=== PETSC ===\n");
    printf("\titerations: %d\n",its);
    printf("\tnorm(res): %e\n",rnorm);
    printf("Timing:\n");
    printf("\ttotal   : %e s\n\n",petsc_t);
    printf("=== ECG-F ===\n");
    printf("\titerations: %d\n",ecg.iter);
    printf("\tnorm(res) : %e\n",ecg.res);
    printf("\tenl factor: %d\n",ecg.enlFac);
    printf("\tortho alg : %d\n",ecg.ortho_alg);
    printf("\treduction : %d\n",ecg.bs_red);
    printf("\tfinal bs  : %d\n",ecg.bs);
    printf("Timing:\n");
    printf("\ttotal   : %e s\n",tot_t);
    printf("\toperator: %e s\n",op_t);
    printf("\tprecond : %e s\n",prec_t);
    printf("\ttot_iter: %e s\n",   ecg.tot_t);
    printf("\tcomm    : %e s\n",   ecg.comm_t);
    printf("\ttrsm    : %e s\n",   ecg.trsm_t);
    printf("\tgemm    : %e s\n",   ecg.gemm_t);
    printf("\tpotrf   : %e s\n",   ecg.potrf_t);
    printf("\tpstrf   : %e s\n",   ecg.pstrf_t);
    printf("\tlapmt   : %e s\n",   ecg.lapmt_t);
    printf("\tgesvd   : %e s\n",   ecg.gesvd_t);
    printf("\tgeqrf   : %e s\n",   ecg.geqrf_t);
    printf("\tormqr   : %e s\n",   ecg.ormqr_t);
    printf("\tcopy    : %e s\n\n", ecg.copy_t);
  }

  /*================ Finalize ================*/

  // Free PETSc structure
  MatDestroy(&A_petsc);
  VecDestroy(&X);
  // Free arrays
  if (rhs != NULL) free(rhs);
  if (sol != NULL) free(sol);
  preAlps_OperatorFree();

  CPLM_printTimer(NULL);
  PetscFinalize();
#endif
  return 0;
}
/******************************************************************************/
