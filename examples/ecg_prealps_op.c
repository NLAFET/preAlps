/******************************************************************************/
/* Author     : Olivier Tissot , Simplice Donfack                             */
/* Creation   : 2016/06/23                                                    */
/* Description: Example of usage of E(nlarged) C(onjugate) G(radient)         */
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

/* preAlps */
#include "operator.h"
#include "block_jacobi.h"
#include "ecg.h"
/******************************************************************************/

/******************************************************************************/
/*                                    CODE                                    */
/******************************************************************************/
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
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

  // Set global parameters for ECG
  double tol = 1e-5;
  int maxIter = 1000;

  /*================ ECG solve ================*/
  preAlps_ECG_t ecg;
  // Set parameters
  ecg.comm = MPI_COMM_WORLD;  /* MPI Communicator */
  ecg.globPbSize = M;         /* Size of the global problem */
  ecg.locPbSize = m;          /* Size of the local problem */
  ecg.maxIter = maxIter;      /* Maximum number of iterations */
  ecg.enlFac = atoi(argv[2]); /* Enlarging factor */
  ecg.tol = tol;              /* Tolerance of the method */
  ecg.ortho_alg = ORTHOMIN;   /* Orthogonalization algorithm */
  ecg.bs_red = ADAPT_BS;      /* Adaptive reduction of the search directions */
  // Get local and global sizes of operator A
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
