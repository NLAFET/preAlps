/******************************************************************************/
/* Author     : Olivier Tissot                                                */
/* Creation   : 2016/06/24                                                    */
/* Description: Enlarged Preconditioned C(onjugate) G(radient)                */
/******************************************************************************/

/******************************************************************************/
/*                                  INCLUDE                                   */
/******************************************************************************/
#include "ecg.h"
/******************************************************************************/

/******************************************************************************/
/*                                    CODE                                    */
/******************************************************************************/
int ECGInitialize(ECG_t* ecg, double* rhs, int* rci_request) {
PUSH
BEGIN_TIME
  int rank, ierr = 0;
  int nCol = 0;

  // Allocate Memory
  ECGMalloc(ecg);

  // Simplify notations
  Mat_Dense_t* P        = ecg->P;
  Mat_Dense_t* R        = ecg->R;
  double*      ptrNormb = &(ecg->normb);

  // TODO remove this
  DVector_t b           = DVectorNULL();
  b.nval = P->info.m;
  b.val = rhs;
  // End TODO

  MPI_Comm_rank(ecg->comm, &rank);
  ecg->iter = 0;
  // Compute normb
  ierr = DVector2NormSquared(&b, ptrNormb);
  // Sum over all processes
  MPI_Allreduce(MPI_IN_PLACE,
                ptrNormb,
                1,
                MPI_DOUBLE,
                MPI_SUM,
                ecg->comm);
  *ptrNormb = sqrt(*ptrNormb);

  // First we construct R_0 by splitting b
  nCol = rank % (ecg->enlFac);
  ierr = ECGSplit(rhs, R, nCol);CHKERR(ierr);

  // No block size reduction for the moment
  ecg->bs_red = NO_BS_RED;

  // Then we need to construct R_0 and P_0
  *rci_request = 0;
END_TIME
POP
  return ierr;
}

int ECGMalloc(ECG_t* ecg) {
PUSH
BEGIN_TIME
  int ierr = 0;
  // Malloc the pointers
  ecg->X        = (Mat_Dense_t*) malloc(sizeof(Mat_Dense_t));
  ecg->R        = (Mat_Dense_t*) malloc(sizeof(Mat_Dense_t));
  ecg->P        = (Mat_Dense_t*) malloc(sizeof(Mat_Dense_t));
  ecg->AP       = (Mat_Dense_t*) malloc(sizeof(Mat_Dense_t));
  ecg->alpha    = (Mat_Dense_t*) malloc(sizeof(Mat_Dense_t));
  ecg->beta     = (Mat_Dense_t*) malloc(sizeof(Mat_Dense_t));
  ecg->Z        = (Mat_Dense_t*) malloc(sizeof(Mat_Dense_t));
  Info_Dense_t info = { .M = ecg->globPbSize,
                        .N = ecg->enlFac,
                        .m = ecg->locPbSize,
                        .n = ecg->enlFac,
                        .lda = ecg->locPbSize,
                        .nval = ecg->locPbSize*(ecg->enlFac),
                        .stor_type = COL_MAJOR };
  ierr = MatDenseCreateZero(ecg->X,info);CHKERR(ierr);
  ierr = MatDenseCreateZero(ecg->R,info);CHKERR(ierr);
  ierr = MatDenseCreate(ecg->P,info);CHKERR(ierr);
  ierr = MatDenseCreate(ecg->AP,info);CHKERR(ierr);
  ierr = MatDenseCreateZero(ecg->Z,info);CHKERR(ierr);
  if (ecg->ortho_alg == ORTHODIR) {
    ecg->P_prev  = (Mat_Dense_t*) malloc(sizeof(Mat_Dense_t));
    ecg->AP_prev = (Mat_Dense_t*) malloc(sizeof(Mat_Dense_t));
    ierr = MatDenseCreateZero(ecg->P_prev,info);CHKERR(ierr);
    ierr = MatDenseCreateZero(ecg->AP_prev,info);CHKERR(ierr);
  }
  // H has nbBlockCG-1 columns maximum
  if (ecg->bs_red == ALPHA_RANK) {
    info.N--;
    info.n--;
    ecg->H = (Mat_Dense_t*) malloc(sizeof(Mat_Dense_t));
    ecg->AH = (Mat_Dense_t*) malloc(sizeof(Mat_Dense_t));
    ierr = MatDenseCreateZero(ecg->H,info);CHKERR(ierr);
    ierr = MatDenseCreateZero(ecg->AH,info);CHKERR(ierr);
  }
  Info_Dense_t info_step = { .M    = ecg->enlFac,
                             .N    = ecg->enlFac,
                             .m    = ecg->enlFac,
                             .n    = ecg->enlFac,
                             .lda  = ecg->enlFac,
                             .nval = (ecg->enlFac)*(ecg->enlFac),
                             .stor_type = COL_MAJOR};
  ierr = MatDenseCreate(ecg->alpha,info_step);CHKERR(ierr);
  ierr = MatDenseCreate(ecg->beta,info_step);CHKERR(ierr);
  if (ecg->ortho_alg == ORTHODIR) {
    ecg->gamma  = (Mat_Dense_t*) malloc(sizeof(Mat_Dense_t));
    ierr = MatDenseCreate(ecg->gamma,info_step);CHKERR(ierr);
  }
  // Malloc the working arrays
  ecg->work = (double*) malloc(ecg->P->info.nval*sizeof(double));
  ecg->iwork = (int*) malloc(ecg->enlFac*sizeof(int));

END_TIME
POP
  return ierr;
}

int ECGSplit(double* x, Mat_Dense_t* XSplit, int colIndex) {
PUSH
BEGIN_TIME
  ASSERT(XSplit->val != NULL);
  ASSERT(x != NULL);
  int ierr = 0;
  int loop_index_1, loop_index_2;
  if (XSplit->info.stor_type == ROW_MAJOR) {
    loop_index_1 = XSplit->info.n;
    loop_index_2 = 1;
  }
  else {
    loop_index_1 = 1;
    loop_index_2 = XSplit->info.m;
  }
  // Assume that XSplit is zero everywhere
  for (int i = 0; i < XSplit->info.m; i++) {
    XSplit->val[i*loop_index_1 + colIndex*loop_index_2] = x[i];
  }
END_TIME
POP
  return ierr;
}

int ECGStoppingCriterion(ECG_t* ecg, int* stop) {
PUSH
BEGIN_TIME
int ierr = 0;
  // Simplify notations
  MPI_Comm comm    = ecg->comm;
  Mat_Dense_t* R   = ecg->R;
  double tolerance = ecg->tol;
  double normb     = ecg->normb;
  double* ptrRes   = &(ecg->res);
  double* work     = ecg->work;
  int iterMax      = ecg->maxIter;

  DVector_t r = DVectorNULL();
  r.nval = R->info.m;
  r.val  = work;

  ASSERT(stop != NULL);

  // Sum the columns of the block residual
  ierr = MatDenseKernelSumColumns(R, &r);CHKERR(ierr);
  // Sum over the line of the reduced residual
  *ptrRes = 0.0;
  ierr = DVector2NormSquared(&r, ptrRes);CHKERR(ierr);
  // Sum over all processes
  MPI_Allreduce(MPI_IN_PLACE,ptrRes,1,MPI_DOUBLE,MPI_SUM,comm);

  *ptrRes = sqrt(*ptrRes);

  // Stopping criterion
  if (*ptrRes > normb*tolerance && ecg->iter < iterMax )
    *stop = 0; // we continue
  else
    *stop = 1; // we stop

END_TIME
POP
  return ierr;
}

int ECGIterate(ECG_t* ecg, int* rci_request) {
PUSH
BEGIN_TIME
OPEN_TIMER
  int ierr = -1;
  /* int nrhs = ecg->param.nbRHS; */
  // Simplify notations
  MPI_Comm     comm    = ecg->comm;
  Mat_Dense_t* P       = ecg->P;
  Mat_Dense_t* AP      = ecg->AP;
  Mat_Dense_t* AP_prev = ecg->AP_prev;
  Mat_Dense_t* P_prev  = ecg->P_prev;
  Mat_Dense_t* X       = ecg->X;
  Mat_Dense_t* R       = ecg->R;
  /* Mat_Dense_t* H       = ecg->H; */
  /* Mat_Dense_t* AH      = ecg->AH; */
  Mat_Dense_t* alpha   = ecg->alpha;
  Mat_Dense_t* beta    = ecg->beta;
  Mat_Dense_t* gamma   = ecg->gamma;
  Mat_Dense_t* Z       = ecg->Z;
  /* int M = P->info.M; */
  /* int m = P->info.m; */
  /* int t = P->info.n; */

  if (*rci_request == 0) {
    // We have A*P here !!
    TIC(step1,"ACholQR")
    ierr = MatDenseACholQR(P,AP,beta,comm);
    TAC(step1)
    TIC(step2,"alpha = P^t*R")
    ierr = MatDenseMatDotProd(P,R,alpha,comm);
    TAC(step2)
    TIC(step3,"X = X + P*alpha")
    ierr = MatDenseKernelMatMult(P,'N',alpha,'N',X,1.0,1.0);
    TAC(step3)
    TIC(step4,"R = R - AP*alpha")
    ierr = MatDenseKernelMatMult(AP,'N',alpha,'N',R,-1.0,1.0);
    TAC(step4)
    // Iteration finished
    ecg->iter++;
    // Now we need the preconditioner
    *rci_request = 1;
  }
  else if (*rci_request == 1) {
    // We have the preconditioner here !!!
    TIC(step5,"beta = (AP)^t*Z")
    ierr = MatDenseMatDotProd(AP,Z,beta,comm);
    TAC(step5)
    TIC(step6,"Z = Z - P*beta")
    ierr = MatDenseKernelMatMult(P,'N',beta,'N',Z,-1.0,1.0);
    TAC(step6)
    if (ecg->ortho_alg == ORTHODIR) {
      TIC(step7,"gamma = (AP_prev)^t*Z")
      ierr = MatDenseMatDotProd(AP_prev,Z,gamma,comm);
      TAC(step7)
      TIC(step8,"Z = Z - P_prev*gamma")
      ierr = MatDenseKernelMatMult(P_prev,'N',gamma,'N',Z,-1.0,1.0);
      TAC(step8)
    }
    // Swapping time
    MatDenseSwap(P,Z);
    if (ecg->ortho_alg == ORTHODIR) {
      MatDenseSwap(AP,AP_prev);
      MatDenseSwap(P_prev,Z);
    }
    // Now we need A*P to continue
    *rci_request = 0;
  }
  else {
    CPALAMEM_Abort("Internal error: wrong rci_request value: %d",*rci_request);
  }
CLOSE_TIMER
END_TIME
POP
  return ierr;
}

int ECGFinalize(ECG_t* ecg, double* solution) {
PUSH
BEGIN_TIME
  int ierr = 0;
  // Simplify notations
  Mat_Dense_t* X = ecg->X;
  DVector_t sol = DVectorNULL();
  sol.nval = X->info.m;
  sol.val  = solution;
  // Get the solution
  ierr = MatDenseKernelSumColumns(X, &sol);CHKERR(ierr);
  ECGFree(ecg);
END_TIME
POP
  return ierr;
}

void ECGPrint(ECG_t* ecg) {
PUSH
BEGIN_TIME
  int rank;
  MPI_Comm_rank(ecg->comm,&rank);
  printf("[%d] prints ECG_t...\n", rank);
  MatDensePrintfInfo("X",    ecg->X);
  MatDensePrintfInfo("R",    ecg->R);
  MatDensePrintfInfo("P",    ecg->P);
  MatDensePrintfInfo("AP",   ecg->AP);
  MatDensePrintfInfo("Z",    ecg->Z);
  MatDensePrintfInfo("alpha",ecg->alpha);
  MatDensePrintfInfo("beta", ecg->beta);
  if (ecg->ortho_alg == ORTHODIR) {
    MatDensePrintfInfo("P_prev",ecg->P_prev);
    MatDensePrintfInfo("AP_prev",ecg->AP_prev);
    MatDensePrintfInfo("gamma",  ecg->gamma);
  }
  if (ecg->bs_red == ALPHA_RANK) {
    MatDensePrintfInfo("H",ecg->H);
    MatDensePrintfInfo("AH",  ecg->AH);
  }
  printf("\n");
  printf("iter: %d\n",ecg->iter);
  printf("[%d] ends printing ECG_t!\n", rank);
END_TIME
POP
}

void ECGFree(ECG_t* ecg) {
PUSH
BEGIN_TIME
  MatDenseFree(ecg->X);
  if (ecg->X != NULL)
    free(ecg->X);
  MatDenseFree(ecg->R);
  if (ecg->R != NULL)
    free(ecg->R);
  MatDenseFree(ecg->P);
  if (ecg->P != NULL)
    free(ecg->P);
  MatDenseFree(ecg->AP);
  if (ecg->AP != NULL)
    free(ecg->AP);
  MatDenseFree(ecg->alpha);
  if (ecg->alpha != NULL)
    free(ecg->alpha);
  MatDenseFree(ecg->beta);
  if (ecg->beta != NULL)
    free(ecg->beta);
  MatDenseFree(ecg->Z);
  if (ecg->Z != NULL)
    free(ecg->Z);
  if (ecg->ortho_alg == ORTHODIR) {
    MatDenseFree(ecg->P_prev);
    if (ecg->P_prev != NULL)
      free(ecg->P_prev);
    MatDenseFree(ecg->AP_prev);
    if (ecg->AP_prev != NULL)
      free(ecg->AP_prev);
    MatDenseFree(ecg->gamma);
    if (ecg->gamma != NULL)
      free(ecg->gamma);
  }
  if (ecg->bs_red == ALPHA_RANK) {
    MatDenseFree(ecg->H);
    if (ecg->H != NULL)
      free(ecg->H);
    MatDenseFree(ecg->AH);
    if (ecg->AH != NULL)
      free(ecg->AH);
  }
  if (ecg->work != NULL)
    free(ecg->work);
END_TIME
POP
}

/******************************************************************************/
