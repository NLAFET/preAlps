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
CPLM_PUSH
  int rank, ierr = 0;
  int nCol = 0;

  // Allocate Memory
  ECGMalloc(ecg);

  // Simplify notations
  CPLM_Mat_Dense_t* P        = ecg->P;
  CPLM_Mat_Dense_t* R        = ecg->R;
  double*      ptrNormb = &(ecg->normb);

  // TODO remove this
  CPLM_DVector_t b           = CPLM_DVectorNULL();
  b.nval = P->info.m;
  b.val = rhs;
  // End TODO

  MPI_Comm_rank(ecg->comm, &rank);
  ecg->iter = 0;
  // Compute normb
  ierr = CPLM_DVector2NormSquared(&b, ptrNormb);
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
CPLM_POP
  return ierr;
}

int ECGMalloc(ECG_t* ecg) {
CPLM_PUSH
  int ierr = 0;
  // Malloc the pointers
  ecg->X        = (CPLM_Mat_Dense_t*) malloc(sizeof(CPLM_Mat_Dense_t));
  ecg->R        = (CPLM_Mat_Dense_t*) malloc(sizeof(CPLM_Mat_Dense_t));
  ecg->P        = (CPLM_Mat_Dense_t*) malloc(sizeof(CPLM_Mat_Dense_t));
  ecg->AP       = (CPLM_Mat_Dense_t*) malloc(sizeof(CPLM_Mat_Dense_t));
  ecg->alpha    = (CPLM_Mat_Dense_t*) malloc(sizeof(CPLM_Mat_Dense_t));
  ecg->beta     = (CPLM_Mat_Dense_t*) malloc(sizeof(CPLM_Mat_Dense_t));
  ecg->Z        = (CPLM_Mat_Dense_t*) malloc(sizeof(CPLM_Mat_Dense_t));
  CPLM_Info_Dense_t info = { .M = ecg->globPbSize,
                             .N = ecg->enlFac,
                             .m = ecg->locPbSize,
                             .n = ecg->enlFac,
                             .lda = ecg->locPbSize,
                             .nval = ecg->locPbSize*(ecg->enlFac),
                             .stor_type = CPLM_COL_MAJOR };
  ierr = CPLM_MatDenseCreateZero(ecg->X,info);CHKERR(ierr);
  ierr = CPLM_MatDenseCreateZero(ecg->R,info);CHKERR(ierr);
  ierr = CPLM_MatDenseCreate(ecg->P,info);CHKERR(ierr);
  ierr = CPLM_MatDenseCreate(ecg->AP,info);CHKERR(ierr);
  ierr = CPLM_MatDenseCreateZero(ecg->Z,info);CHKERR(ierr);
  if (ecg->ortho_alg == ORTHODIR) {
    ecg->P_prev  = (CPLM_Mat_Dense_t*) malloc(sizeof(CPLM_Mat_Dense_t));
    ecg->AP_prev = (CPLM_Mat_Dense_t*) malloc(sizeof(CPLM_Mat_Dense_t));
    ierr = CPLM_MatDenseCreateZero(ecg->P_prev,info);CHKERR(ierr);
    ierr = CPLM_MatDenseCreateZero(ecg->AP_prev,info);CHKERR(ierr);
  }
  // H has nbBlockCG-1 columns maximum
  if (ecg->bs_red == ALPHA_RANK) {
    info.N--;
    info.n--;
    ecg->H = (CPLM_Mat_Dense_t*) malloc(sizeof(CPLM_Mat_Dense_t));
    ecg->AH = (CPLM_Mat_Dense_t*) malloc(sizeof(CPLM_Mat_Dense_t));
    ierr = CPLM_MatDenseCreateZero(ecg->H,info);CHKERR(ierr);
    ierr = CPLM_MatDenseCreateZero(ecg->AH,info);CHKERR(ierr);
  }
  CPLM_Info_Dense_t info_step = { .M    = ecg->enlFac,
                                  .N    = ecg->enlFac,
                                  .m    = ecg->enlFac,
                                  .n    = ecg->enlFac,
                                  .lda  = ecg->enlFac,
                                  .nval = (ecg->enlFac)*(ecg->enlFac),
                                  .stor_type = CPLM_COL_MAJOR};
  ierr = CPLM_MatDenseCreate(ecg->alpha,info_step);CHKERR(ierr);
  ierr = CPLM_MatDenseCreate(ecg->beta,info_step);CHKERR(ierr);
  if (ecg->ortho_alg == ORTHODIR) {
    ecg->gamma  = (CPLM_Mat_Dense_t*) malloc(sizeof(CPLM_Mat_Dense_t));
    ierr = CPLM_MatDenseCreate(ecg->gamma,info_step);CHKERR(ierr);
  }
  // Malloc the working arrays
  ecg->work = (double*) malloc(ecg->P->info.nval*sizeof(double));
  ecg->iwork = (int*) malloc(ecg->enlFac*sizeof(int));
CPLM_POP
  return ierr;
}

int ECGSplit(double* x, CPLM_Mat_Dense_t* XSplit, int colIndex) {
CPLM_PUSH
  CPLM_ASSERT(XSplit->val != NULL);
  CPLM_ASSERT(x != NULL);
  int ierr = 0;
  int loop_index_1, loop_index_2;
  if (XSplit->info.stor_type == CPLM_ROW_MAJOR) {
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
CPLM_POP
  return ierr;
}

int ECGStoppingCriterion(ECG_t* ecg, int* stop) {
CPLM_PUSH
int ierr = 0;
  // Simplify notations
  MPI_Comm comm       = ecg->comm;
  CPLM_Mat_Dense_t* R = ecg->R;
  double tolerance    = ecg->tol;
  double normb        = ecg->normb;
  double* ptrRes      = &(ecg->res);
  double* work        = ecg->work;
  int iterMax         = ecg->maxIter;

  CPLM_DVector_t r = DVectorNULL();
  r.nval = R->info.m;
  r.val  = work;

  CPLM_ASSERT(stop != NULL);

  // Sum the columns of the block residual
  ierr = CPLM_MatDenseKernelSumColumns(R, &r);CHKERR(ierr);
  // Sum over the line of the reduced residual
  *ptrRes = 0.0;
  ierr = CPLM_DVector2NormSquared(&r, ptrRes);CHKERR(ierr);
  // Sum over all processes
  MPI_Allreduce(MPI_IN_PLACE,ptrRes,1,MPI_DOUBLE,MPI_SUM,comm);

  *ptrRes = sqrt(*ptrRes);

  // Stopping criterion
  if (*ptrRes > normb*tolerance && ecg->iter < iterMax )
    *stop = 0; // we continue
  else
    *stop = 1; // we stop
CPLM_POP
  return ierr;
}

int ECGIterate(ECG_t* ecg, int* rci_request) {
CPLM_PUSH
  int ierr = -1;
  /* int nrhs = ecg->param.nbRHS; */
  // Simplify notations
  MPI_Comm     comm    = ecg->comm;
  CPLM_Mat_Dense_t* P       = ecg->P;
  CPLM_Mat_Dense_t* AP      = ecg->AP;
  CPLM_Mat_Dense_t* AP_prev = ecg->AP_prev;
  CPLM_Mat_Dense_t* P_prev  = ecg->P_prev;
  CPLM_Mat_Dense_t* X       = ecg->X;
  CPLM_Mat_Dense_t* R       = ecg->R;
  /* CPLM_Mat_Dense_t* H       = ecg->H; */
  /* CPLM_Mat_Dense_t* AH      = ecg->AH; */
  CPLM_Mat_Dense_t* alpha   = ecg->alpha;
  CPLM_Mat_Dense_t* beta    = ecg->beta;
  CPLM_Mat_Dense_t* gamma   = ecg->gamma;
  CPLM_Mat_Dense_t* Z       = ecg->Z;
  /* int M = P->info.M; */
  /* int m = P->info.m; */
  /* int t = P->info.n; */

  if (*rci_request == 0) {
    // We have A*P here !!
    TIC(step1,"ACholQR")
    ierr = CPLM_MatDenseACholQR(P,AP,beta,comm);
    TAC(step1)
    TIC(step2,"alpha = P^t*R")
    ierr = CPLM_MatDenseMatDotProd(P,R,alpha,comm);
    TAC(step2)
    TIC(step3,"X = X + P*alpha")
    ierr = CPLM_MatDenseKernelMatMult(P,'N',alpha,'N',X,1.0,1.0);
    TAC(step3)
    TIC(step4,"R = R - AP*alpha")
    ierr = CPLM_MatDenseKernelMatMult(AP,'N',alpha,'N',R,-1.0,1.0);
    TAC(step4)
    // Iteration finished
    ecg->iter++;
    // Now we need the preconditioner
    *rci_request = 1;
  }
  else if (*rci_request == 1) {
    // We have the preconditioner here !!!
    TIC(step5,"beta = (AP)^t*Z")
    ierr = CPLM_MatDenseMatDotProd(AP,Z,beta,comm);
    TAC(step5)
    TIC(step6,"Z = Z - P*beta")
    ierr = CPLM_MatDenseKernelMatMult(P,'N',beta,'N',Z,-1.0,1.0);
    TAC(step6)
    if (ecg->ortho_alg == ORTHODIR) {
      TIC(step7,"gamma = (AP_prev)^t*Z")
      ierr = CPLM_MatDenseMatDotProd(AP_prev,Z,gamma,comm);
      TAC(step7)
      TIC(step8,"Z = Z - P_prev*gamma")
      ierr = CPLM_MatDenseKernelMatMult(P_prev,'N',gamma,'N',Z,-1.0,1.0);
      TAC(step8)
    }
    // Swapping time
    CPLM_MatDenseSwap(P,Z);
    if (ecg->ortho_alg == ORTHODIR) {
      CPLM_MatDenseSwap(AP,AP_prev);
      CPLM_MatDenseSwap(P_prev,Z);
    }
    // Now we need A*P to continue
    *rci_request = 0;
  }
  else {
    CPLM_Abort("Internal error: wrong rci_request value: %d",*rci_request);
  }
CPLM_POP
  return ierr;
}

int ECGFinalize(ECG_t* ecg, double* solution) {
CPLM_PUSH
  int ierr = 0;
  // Simplify notations
  CPLM_Mat_Dense_t* X = ecg->X;
  CPLM_DVector_t sol = CPLM_DVectorNULL();
  sol.nval = X->info.m;
  sol.val  = solution;
  // Get the solution
  ierr = CPLM_MatDenseKernelSumColumns(X, &sol);CHKERR(ierr);
  ECGFree(ecg);
CPLM_POP
  return ierr;
}

void ECGPrint(ECG_t* ecg) {
CPLM_PUSH
  int rank;
  MPI_Comm_rank(ecg->comm,&rank);
  printf("[%d] prints ECG_t...\n", rank);
  CPLM_MatDensePrintfInfo("X",    ecg->X);
  CPLM_MatDensePrintfInfo("R",    ecg->R);
  CPLM_MatDensePrintfInfo("P",    ecg->P);
  CPLM_MatDensePrintfInfo("AP",   ecg->AP);
  CPLM_MatDensePrintfInfo("Z",    ecg->Z);
  CPLM_MatDensePrintfInfo("alpha",ecg->alpha);
  CPLM_MatDensePrintfInfo("beta", ecg->beta);
  if (ecg->ortho_alg == ORTHODIR) {
    CPLM_MatDensePrintfInfo("P_prev",ecg->P_prev);
    CPLM_MatDensePrintfInfo("AP_prev",ecg->AP_prev);
    CPLM_MatDensePrintfInfo("gamma",  ecg->gamma);
  }
  if (ecg->bs_red == ALPHA_RANK) {
    CPLM_MatDensePrintfInfo("H",ecg->H);
    CPLM_MatDensePrintfInfo("AH",  ecg->AH);
  }
  printf("\n");
  printf("iter: %d\n",ecg->iter);
  printf("[%d] ends printing ECG_t!\n", rank);
CPLM_POP
}

void ECGFree(ECG_t* ecg) {
CPLM_PUSH
  CPLM_MatDenseFree(ecg->X);
  if (ecg->X != NULL)
    free(ecg->X);
  CPLM_MatDenseFree(ecg->R);
  if (ecg->R != NULL)
    free(ecg->R);
  CPLM_MatDenseFree(ecg->P);
  if (ecg->P != NULL)
    free(ecg->P);
  CPLM_MatDenseFree(ecg->AP);
  if (ecg->AP != NULL)
    free(ecg->AP);
  CPLM_MatDenseFree(ecg->alpha);
  if (ecg->alpha != NULL)
    free(ecg->alpha);
  CPLM_MatDenseFree(ecg->beta);
  if (ecg->beta != NULL)
    free(ecg->beta);
  CPLM_MatDenseFree(ecg->Z);
  if (ecg->Z != NULL)
    free(ecg->Z);
  if (ecg->ortho_alg == ORTHODIR) {
    CPLM_MatDenseFree(ecg->P_prev);
    if (ecg->P_prev != NULL)
      free(ecg->P_prev);
    CPLM_MatDenseFree(ecg->AP_prev);
    if (ecg->AP_prev != NULL)
      free(ecg->AP_prev);
    CPLM_MatDenseFree(ecg->gamma);
    if (ecg->gamma != NULL)
      free(ecg->gamma);
  }
  if (ecg->bs_red == ALPHA_RANK) {
    CPLM_MatDenseFree(ecg->H);
    if (ecg->H != NULL)
      free(ecg->H);
    CPLM_MatDenseFree(ecg->AH);
    if (ecg->AH != NULL)
      free(ecg->AH);
  }
  if (ecg->work != NULL)
    free(ecg->work);
CPLM_POP
}

/******************************************************************************/
