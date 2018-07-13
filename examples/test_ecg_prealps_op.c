/**
 * \file    test_ecg_prealps_op.c
 * \author  Olivier Tissot
 * \date    2016/06/23
 * \brief   Example of usage of E(nlarged) C(onjugate) G(radient)
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

CPLM_OPEN_TIMER
  /*================ Initialize ================*/
  int rank, size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Force sequential execution on each MPI process
  // OT: I tested and it still works with OpenMP activated
  MKL_Set_Num_Threads(1);

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
  int rci_request = 0;
  int stop = 0;
  double* sol = NULL;
  sol = (double*) malloc(m*sizeof(double));
CPLM_TIC(step1,"ECGSolve")
  // Allocate memory and initialize variables
  preAlps_ECGInitialize(&ecg,rhs,&rci_request);
  // Finish initialization
  preAlps_BlockJacobiApply(ecg.R,ecg.P);
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
      if (ecg.ortho_alg == ORTHOMIN)
        preAlps_BlockJacobiApply(ecg.R,ecg.Z);
      else if (ecg.ortho_alg == ORTHODIR)
        preAlps_BlockJacobiApply(ecg.AP,ecg.Z);
    }
  }
  // Retrieve solution and free memory
  preAlps_ECGFinalize(&ecg,sol);
CPLM_TAC(step1)

  if (rank == 0) {
    preAlps_ECGPrint(&ecg,0);
  }

  /*================ Finalize ================*/

  // Free arrays
  if (rhs != NULL) free(rhs);
  if (sol != NULL) free(sol);
  preAlps_OperatorFree();
CPLM_CLOSE_TIMER

  CPLM_printTimer(NULL);
  MPI_Finalize();
  return 0;
}
/******************************************************************************/
