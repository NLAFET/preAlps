/**
 * \file    test_ecg_prealps_op.c
 * \author  Olivier Tissot
 * \date    2018/07/13
 * \brief   Example of usage of fused E(nlarged) C(onjugate) G(radient)
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
//#include <cpalamem_macro.h>
//#include <cpalamem_instrumentation.h>

/* preAlps */
#include "operator.h"
#include "block_jacobi.h"
#include "ecg.h"

/* Command line parser */
#include <ctype.h>
#include <getopt.h>
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
         " ./test_ecg_prealps_op"
         " -e/--enlarging-factor int"
         " [-h/--help]"
         " [-i/--iteration-maximum int]"
         " -m/--matrix file"
         " -r/--search-dir-red int"
         " [-t/--tolerance double]\n");
  printf("OPTIONS\n");
  printf("\t-e/--enlarging-factor : enlarging factor"
                                  " (cannot exceed nprocs)\n");
  printf("\t-h/--help             : print this help message\n");
  printf("\t-i/--iteration-maximum: maximum of iteration count"
                                  " (default is 1000)\n");
  printf("\t-m/--matrix           : the .mtx file containing the matrix A\n");
  printf("\t-r/--search-dir-red   : adaptive reduction of the search"
                                  " directions (0: no, 1: yes)\n");
  printf("\t-t/--tolerance        : tolerance of the method"
                                  " (default is 1e-5)\n");
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  CPLM_SetEnv();

  /*================ Command line parser ================*/
  int c;
  static struct option long_options[] = {
    {"enlarging-factor" , required_argument, NULL, 'e'},
    {"help"             , no_argument      , NULL, 'h'},
    {"iteration-maximum", optional_argument, NULL, 'i'},
    {"matrix"           , required_argument, NULL, 'm'},
    {"search-dir-red"   , required_argument, NULL, 'r'},
    {"tolerance"        , optional_argument, NULL, 't'},
    {NULL               , 0                , NULL, 0}
  };

  int opterr = 0;
  int option_index = 0;

  // Set global parameters for both PETSc and ECG
  double tol = 1e-5;
  int maxIter = 1000;
  int enlFac = 1, bs_red = 0;
  const char* matrixFilename = NULL;
  while ((c = getopt_long(argc, argv, "e:hi:m:r:t:", long_options, &option_index)) != -1)
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
  double trash_t, trash_tg;
  double buildop_t = 0.E0, buildprec_t = 0.E0;
  double tot_t = 0.E0, op_t = 0.E0, prec_t = 0.E0;
  double totf_t = 0.E0, opf_t = 0.E0, precf_t = 0.E0;

  // Force sequential execution on each MPI process
  MKL_Set_Num_Threads(1);

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

  /*================ ECG solve ================*/
  preAlps_ECG_t ecg;
  // Set parameters
  ecg.comm = MPI_COMM_WORLD;
  ecg.globPbSize = M;
  ecg.locPbSize = m;
  ecg.maxIter = maxIter;
  ecg.enlFac = enlFac;
  ecg.tol = tol;
  ecg.ortho_alg = ORTHODIR;
  ecg.bs_red = (bs_red == 0 ? NO_BS_RED : ADAPT_BS);
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
  trash_t = MPI_Wtime();
  preAlps_BlockOperator(ecg.P,ecg.AP);
  op_t += MPI_Wtime() - trash_t;
  // Main loop
  while (stop != 1) {
    preAlps_ECGIterate(&ecg,&rci_request);
    if (rci_request == 0) {
      trash_t = MPI_Wtime();
      preAlps_BlockOperator(ecg.P,ecg.AP);
      op_t += MPI_Wtime() - trash_t;
    }
    else if (rci_request == 1) {
      preAlps_ECGStoppingCriterion(&ecg,&stop);
      if (stop == 1) break;
      trash_t = MPI_Wtime();
      preAlps_BlockJacobiApply(ecg.AP,ecg.Z);
      prec_t += MPI_Wtime() - trash_t;
    }
  }
  // Retrieve solution and free memory
  preAlps_ECGFinalize(&ecg,sol);
  tot_t += MPI_Wtime() - trash_tg;

  /*================== Fused-ECG solve ==================*/
  preAlps_ECG_t ecg_f;
  // Set parameters
  ecg_f.comm = MPI_COMM_WORLD;
  ecg_f.globPbSize = M;
  ecg_f.locPbSize = m;
  ecg_f.maxIter = maxIter;
  ecg_f.enlFac = enlFac;
  ecg_f.tol = tol;
  ecg_f.ortho_alg = ORTHODIR_FUSED;
  ecg_f.bs_red = (bs_red == 0 ? NO_BS_RED : ADAPT_BS);
  stop = 0; rci_request = 0;
  if (sol != NULL) free(sol);
  sol = (double*) malloc(m*sizeof(double));
  // Allocate memory and initialize variables
  trash_tg = MPI_Wtime();
  preAlps_ECGInitialize(&ecg_f,rhs,&rci_request);
  // Finish initialization
  trash_t = MPI_Wtime();
  preAlps_BlockJacobiApply(ecg_f.R,ecg_f.P);
  precf_t += MPI_Wtime() - trash_t;
  // Main loop
  while (rci_request != 1) {
    trash_t = MPI_Wtime();
    preAlps_BlockOperator(ecg_f.P,ecg_f.AP);
    opf_t += MPI_Wtime() - trash_t;
    trash_t = MPI_Wtime();
    preAlps_BlockJacobiApply(ecg_f.AP,ecg_f.Z);
    precf_t += MPI_Wtime() - trash_t;
    preAlps_ECGIterate(&ecg_f,&rci_request);
  }
  // Retrieve solution and free memory
  preAlps_ECGFinalize(&ecg_f,sol);
  totf_t += MPI_Wtime() - trash_tg;

  // Post-processing
  /* Global timings */
  MPI_Allreduce(MPI_IN_PLACE, &buildop_t, 1, MPI_DOUBLE, MPI_MAX, ecg.comm);
  MPI_Allreduce(MPI_IN_PLACE, &buildprec_t, 1, MPI_DOUBLE, MPI_MAX, ecg.comm);
  MPI_Allreduce(MPI_IN_PLACE, &totf_t,  1, MPI_DOUBLE, MPI_MAX, ecg.comm);
  MPI_Allreduce(MPI_IN_PLACE, &tot_t,   1, MPI_DOUBLE, MPI_MAX, ecg.comm);
  MPI_Allreduce(MPI_IN_PLACE, &totf_t,  1, MPI_DOUBLE, MPI_MAX, ecg.comm);
  MPI_Allreduce(MPI_IN_PLACE, &op_t,    1, MPI_DOUBLE, MPI_MAX, ecg.comm);
  MPI_Allreduce(MPI_IN_PLACE, &opf_t,   1, MPI_DOUBLE, MPI_MAX, ecg.comm);
  MPI_Allreduce(MPI_IN_PLACE, &prec_t,  1, MPI_DOUBLE, MPI_MAX, ecg.comm);
  MPI_Allreduce(MPI_IN_PLACE, &precf_t, 1, MPI_DOUBLE, MPI_MAX, ecg.comm);
  /* standard odir */
  MPI_Allreduce(MPI_IN_PLACE, &ecg.tot_t  , 1, MPI_DOUBLE, MPI_MAX, ecg.comm);
  MPI_Allreduce(MPI_IN_PLACE, &ecg.comm_t , 1, MPI_DOUBLE, MPI_MAX, ecg.comm);
  MPI_Allreduce(MPI_IN_PLACE, &ecg.trsm_t , 1, MPI_DOUBLE, MPI_MAX, ecg.comm);
  MPI_Allreduce(MPI_IN_PLACE, &ecg.gemm_t , 1, MPI_DOUBLE, MPI_MAX, ecg.comm);
  MPI_Allreduce(MPI_IN_PLACE, &ecg.potrf_t, 1, MPI_DOUBLE, MPI_MAX, ecg.comm);
  MPI_Allreduce(MPI_IN_PLACE, &ecg.pstrf_t, 1, MPI_DOUBLE, MPI_MAX, ecg.comm);
  MPI_Allreduce(MPI_IN_PLACE, &ecg.lapmt_t, 1, MPI_DOUBLE, MPI_MAX, ecg.comm);
  MPI_Allreduce(MPI_IN_PLACE, &ecg.gesvd_t, 1, MPI_DOUBLE, MPI_MAX, ecg.comm);
  MPI_Allreduce(MPI_IN_PLACE, &ecg.geqrf_t, 1, MPI_DOUBLE, MPI_MAX, ecg.comm);
  MPI_Allreduce(MPI_IN_PLACE, &ecg.ormqr_t, 1, MPI_DOUBLE, MPI_MAX, ecg.comm);
  MPI_Allreduce(MPI_IN_PLACE, &ecg.copy_t , 1, MPI_DOUBLE, MPI_MAX, ecg.comm);
  /* fused odir */
  MPI_Allreduce(MPI_IN_PLACE, &ecg_f.tot_t  , 1, MPI_DOUBLE, MPI_MAX, ecg.comm);
  MPI_Allreduce(MPI_IN_PLACE, &ecg_f.comm_t , 1, MPI_DOUBLE, MPI_MAX, ecg.comm);
  MPI_Allreduce(MPI_IN_PLACE, &ecg_f.trsm_t , 1, MPI_DOUBLE, MPI_MAX, ecg.comm);
  MPI_Allreduce(MPI_IN_PLACE, &ecg_f.gemm_t , 1, MPI_DOUBLE, MPI_MAX, ecg.comm);
  MPI_Allreduce(MPI_IN_PLACE, &ecg_f.potrf_t, 1, MPI_DOUBLE, MPI_MAX, ecg.comm);
  MPI_Allreduce(MPI_IN_PLACE, &ecg_f.pstrf_t, 1, MPI_DOUBLE, MPI_MAX, ecg.comm);
  MPI_Allreduce(MPI_IN_PLACE, &ecg_f.lapmt_t, 1, MPI_DOUBLE, MPI_MAX, ecg.comm);
  MPI_Allreduce(MPI_IN_PLACE, &ecg_f.gesvd_t, 1, MPI_DOUBLE, MPI_MAX, ecg.comm);
  MPI_Allreduce(MPI_IN_PLACE, &ecg_f.geqrf_t, 1, MPI_DOUBLE, MPI_MAX, ecg.comm);
  MPI_Allreduce(MPI_IN_PLACE, &ecg_f.ormqr_t, 1, MPI_DOUBLE, MPI_MAX, ecg.comm);
  MPI_Allreduce(MPI_IN_PLACE, &ecg_f.copy_t , 1, MPI_DOUBLE, MPI_MAX, ecg.comm);

  if (rank == 0) {
    printf("=== SUMMARY ===\n");
    printf("\t# mpi        : %d\n",size);
    printf("\tbuild op     : %e s\n",buildop_t);
    printf("\tbuild precond: %e s\n\n",buildprec_t);
    printf("=== ODIR ===\n");
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
    printf("=== F-ODIR ===\n");
    printf("\ttotal   : %e s\n",totf_t);
    printf("\toperator: %e s\n",opf_t);
    printf("\tprecond : %e s\n",precf_t);
    printf("\ttot_iter: %e s\n",   ecg_f.tot_t);
    printf("\tcomm    : %e s\n",   ecg_f.comm_t);
    printf("\ttrsm    : %e s\n",   ecg_f.trsm_t);
    printf("\tgemm    : %e s\n",   ecg_f.gemm_t);
    printf("\tpotrf   : %e s\n",   ecg_f.potrf_t);
    printf("\tpstrf   : %e s\n",   ecg_f.pstrf_t);
    printf("\tlapmt   : %e s\n",   ecg_f.lapmt_t);
    printf("\tgesvd   : %e s\n",   ecg_f.gesvd_t);
    printf("\tgeqrf   : %e s\n",   ecg_f.geqrf_t);
    printf("\tormqr   : %e s\n",   ecg_f.ormqr_t);
    printf("\tcopy    : %e s\n\n", ecg_f.copy_t);

  }

  /*================ Finalize ================*/
  // Free arrays
  if (rhs != NULL) free(rhs);
  if (sol != NULL) free(sol);
  preAlps_OperatorFree();
  MPI_Finalize();
  return 0;
}
/******************************************************************************/
