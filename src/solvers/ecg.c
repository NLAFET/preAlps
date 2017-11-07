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
int preAlps_ECGInitialize(preAlps_ECG_t* ecg, double* rhs, int* rci_request) {
CPLM_PUSH
  int rank, size, ierr = 0;
  int nCol = 0;

  // Firstly check that nrhs < size
  MPI_Comm_size(ecg->comm,&size);
  if (size < ecg->enlFac) {
    CPLM_Abort("Enlarging factor must be lower than the number of processors"
               " in the MPI communicator! size: %d ; enlarging factor: %d",
	       size,ecg->enlFac);
  }

  // Allocate Memory
  ierr = _preAlps_ECGMalloc(ecg);

  // Simplify notations
  CPLM_Mat_Dense_t* P = ecg->P;
  CPLM_Mat_Dense_t* R = ecg->R;
  double* ptrNormb = &(ecg->normb);

  // TODO remove this
  CPLM_DVector_t b = CPLM_DVectorNULL();
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
  // Initialize res, iter and bs
  ecg->res  = -1.0;
  ecg->iter = -1;
  ecg->bs   = ecg->enlFac;
  // First we construct R_0 by splitting b
  nCol = rank % (ecg->enlFac);
  ierr = _preAlps_ECGSplit(rhs, R, nCol);

  // Then we need to construct R_0 and P_0
  *rci_request = 0;
CPLM_POP
  return ierr;
}

int _preAlps_ECGMalloc(preAlps_ECG_t* ecg) {
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
                             .stor_type = COL_MAJOR };
  ierr = CPLM_MatDenseCreateZero(ecg->X,info);CPLM_CHKERR(ierr);
  ierr = CPLM_MatDenseCreateZero(ecg->R,info);CPLM_CHKERR(ierr);
  ierr = CPLM_MatDenseCreate(ecg->P,info);CPLM_CHKERR(ierr);
  ierr = CPLM_MatDenseCreate(ecg->AP,info);CPLM_CHKERR(ierr);
  ierr = CPLM_MatDenseCreateZero(ecg->Z,info);CPLM_CHKERR(ierr);
  if (ecg->ortho_alg == ORTHODIR) {
    ecg->P_prev  = (CPLM_Mat_Dense_t*) malloc(sizeof(CPLM_Mat_Dense_t));
    ecg->AP_prev = (CPLM_Mat_Dense_t*) malloc(sizeof(CPLM_Mat_Dense_t));
    ierr = CPLM_MatDenseCreateZero(ecg->P_prev,info);CPLM_CHKERR(ierr);
    ierr = CPLM_MatDenseCreateZero(ecg->AP_prev,info);CPLM_CHKERR(ierr);
  }
  // H has nbBlockCG-1 columns maximum
  if (ecg->bs_red == ALPHA_RANK) {
    info.N--;
    info.n--;
    ecg->H = (CPLM_Mat_Dense_t*) malloc(sizeof(CPLM_Mat_Dense_t));
    ecg->AH = (CPLM_Mat_Dense_t*) malloc(sizeof(CPLM_Mat_Dense_t));
    ierr = CPLM_MatDenseCreateZero(ecg->H,info);CPLM_CHKERR(ierr);
    ierr = CPLM_MatDenseCreateZero(ecg->AH,info);CPLM_CHKERR(ierr);
    ierr = CPLM_MatDenseSetInfo(ecg->H, info.M,0,info.m,0,COL_MAJOR);
    ierr = CPLM_MatDenseSetInfo(ecg->AH,info.M,0,info.m,0,COL_MAJOR);
  }
  CPLM_Info_Dense_t info_step = { .M    = ecg->enlFac,
                                  .N    = ecg->enlFac,
                                  .m    = ecg->enlFac,
                                  .n    = ecg->enlFac,
                                  .lda  = ecg->enlFac,
                                  .nval = (ecg->enlFac)*(ecg->enlFac),
                                  .stor_type = COL_MAJOR};
  ierr = CPLM_MatDenseCreate(ecg->alpha,info_step);CPLM_CHKERR(ierr);
  ierr = CPLM_MatDenseCreate(ecg->beta,info_step);CPLM_CHKERR(ierr);
  if (ecg->ortho_alg == ORTHODIR) {
    ecg->gamma  = (CPLM_Mat_Dense_t*) malloc(sizeof(CPLM_Mat_Dense_t));
    ierr = CPLM_MatDenseCreate(ecg->gamma,info_step);CPLM_CHKERR(ierr);
  }
  if (ecg->bs_red == ALPHA_RANK) {
    info_step.M--;
    info_step.m--;
    ecg->delta = (CPLM_Mat_Dense_t*) malloc(sizeof(CPLM_Mat_Dense_t));
    ierr = CPLM_MatDenseCreateZero(ecg->delta,info_step);CPLM_CHKERR(ierr);
    ierr = CPLM_MatDenseSetInfo(ecg->delta,0,info.N,0,info.n,COL_MAJOR);
  }
  // Malloc the working arrays
  ecg->work = (double*) malloc(ecg->P->info.nval*sizeof(double));
  ecg->iwork = (int*) malloc(ecg->enlFac*sizeof(int));
CPLM_POP
  return ierr;
}

int _preAlps_ECGSplit(double* x, CPLM_Mat_Dense_t* XSplit, int colIndex) {
CPLM_PUSH
  CPLM_ASSERT(XSplit->val != NULL);
  CPLM_ASSERT(x != NULL);
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
CPLM_POP
  return ierr;
}

int preAlps_ECGStoppingCriterion(preAlps_ECG_t* ecg, int* stop) {
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

  CPLM_ASSERT(stop != NULL);

  // Sum the columns of the block residual
  /* CPLM_DVector_t r = CPLM_DVectorNULL(); */
  /* r.nval = R->info.m; */
  /* r.val  = work; */
  /* ierr = CPLM_MatDenseKernelSumColumns(R, &r);CPLM_CHKERR(ierr); */
  /* // Sum over the line of the reduced residual */
  /* *ptrRes = 0.0; */
  /* ierr = CPLM_DVector2NormSquared(&r, ptrRes);CPLM_CHKERR(ierr); */
  /* // Sum over all processes */
  /* MPI_Allreduce(MPI_IN_PLACE,ptrRes,1,MPI_DOUBLE,MPI_SUM,comm); */
  /* *ptrRes = sqrt(*ptrRes); */

  // Frobenius norm of R
  CPLM_Mat_Dense_t RtR_s = CPLM_MatDenseNULL();
  CPLM_MatDenseSetInfo(&RtR_s,
                  R->info.N,
                  R->info.N,
                  R->info.n,
                  R->info.n,
                  COL_MAJOR);
  RtR_s.val = work;
  CPLM_MatDenseMatDotProd(R,R,&RtR_s,comm);
  // Compute trace of RtR
  *ptrRes = 0.0;
  for (int i = 0; i < RtR_s.info.n; ++i)
    *ptrRes += RtR_s.val[i + RtR_s.info.lda*i];
  *ptrRes = sqrt(*ptrRes);

  // Stopping criterion
  if (*ptrRes > normb*tolerance && ecg->iter < iterMax && ecg->bs > 0 )
    *stop = 0; // we continue
  else
    *stop = 1; // we stop
CPLM_POP
  return ierr;
}

int preAlps_ECGIterate(preAlps_ECG_t* ecg, int* rci_request) {
CPLM_PUSH
CPLM_OPEN_TIMER
  int ierr = -1;
  if (*rci_request == 0) {
    //CPLM_TIC(step1,"RRQR(P_k)")
      //ierr = _preAlps_ECGIterateRRQRSearchDirections(ecg);
    //CPLM_TAC(step1)
    CPLM_TIC(step2,"Build X_k")
    ierr = _preAlps_ECGIterateBuildSolution(ecg);
    CPLM_TAC(step2)
    // Iteration finished
    ecg->iter++;
    // Now we need the preconditioner
    *rci_request = 1;
  }
  else if (*rci_request == 1) {
    CPLM_TIC(step3,"Build P_k")
    ierr = _preAlps_ECGIterateBuildSearchDirections(ecg);
    CPLM_TAC(step3)
    // Now we need A*P to continue
    *rci_request = 0;
  }
  else {
    CPLM_Abort("Internal error: wrong rci_request value: %d",*rci_request);
  }
CPLM_CLOSE_TIMER
CPLM_POP
  return ierr;
}

int _preAlps_ECGIterateBuildSolution(preAlps_ECG_t* ecg) {
CPLM_PUSH
  // Simplify notations
  int ierr;
  MPI_Comm          comm    = ecg->comm;
  CPLM_Mat_Dense_t* P       = ecg->P;
  CPLM_Mat_Dense_t* AP      = ecg->AP;
  CPLM_Mat_Dense_t* X       = ecg->X;
  CPLM_Mat_Dense_t* R       = ecg->R;
  CPLM_Mat_Dense_t* alpha   = ecg->alpha;
  CPLM_Mat_Dense_t work_s = CPLM_MatDenseNULL();
  double*  work = ecg->work;
  int t = P->info.n;
  ierr = CPLM_MatDenseSetInfo(&work_s,t,t,t,t,COL_MAJOR);
  work_s.val = work;

  ierr = CPLM_MatDenseACholQR(P,AP,&work_s,comm);
  ierr = CPLM_MatDenseMatDotProd(P,R,alpha,comm);
  if (ecg->bs_red == ALPHA_RANK) {
    ierr = _preAlps_ECGIterateRRQRAlpha(ecg);
  }
  ierr = CPLM_MatDenseKernelMatMult(P,'N',alpha,'N',X,1.0,1.0);
  ierr = CPLM_MatDenseKernelMatMult(AP,'N',alpha,'N',R,-1.0,1.0);
CPLM_POP
  return ierr;
}

int _preAlps_ECGIterateBuildSearchDirections(preAlps_ECG_t* ecg) {
CPLM_PUSH
  // Simplify notations
  int ierr;
  MPI_Comm          comm    = ecg->comm;
  CPLM_Mat_Dense_t* P       = ecg->P;
  CPLM_Mat_Dense_t* P_prev  = ecg->P_prev;
  CPLM_Mat_Dense_t* AP      = ecg->AP;
  CPLM_Mat_Dense_t* AP_prev = ecg->AP_prev;
  CPLM_Mat_Dense_t* beta    = ecg->beta;
  CPLM_Mat_Dense_t* gamma   = ecg->gamma;
  CPLM_Mat_Dense_t* Z       = ecg->Z;
  CPLM_Mat_Dense_t* H      = ecg->H;
  CPLM_Mat_Dense_t* AH     = ecg->AH;
  CPLM_Mat_Dense_t* delta  = ecg->delta;

  ierr = CPLM_MatDenseMatDotProd(AP,Z,beta,comm);
  ierr = CPLM_MatDenseKernelMatMult(P,'N',beta,'N',Z,-1.0,1.0);
  if (ecg->ortho_alg == ORTHODIR) {
    ierr = CPLM_MatDenseMatDotProd(AP_prev,Z,gamma,comm);
    ierr = CPLM_MatDenseKernelMatMult(P_prev,'N',gamma,'N',Z,-1.0,1.0);
  }
  if (ecg->bs_red == ALPHA_RANK && P->info.n < ecg->enlFac) {
    ierr = CPLM_MatDenseMatDotProd(AH,Z,delta,comm);
    ierr = CPLM_MatDenseKernelMatMult(H,'N',delta,'N',Z,-1.0,1.0);
  }

  // Swapping time
  CPLM_MatDenseSwap(P,Z);
  if (ecg->ortho_alg == ORTHODIR) {
    CPLM_MatDenseSwap(AP,AP_prev);
    CPLM_MatDenseSwap(P_prev,Z);
    AP->info = P->info;
    AP_prev->info = P_prev->info;
  }

CPLM_POP
  return ierr;
}

int _preAlps_ECGIterateRRQRSearchDirections(preAlps_ECG_t* ecg) {
CPLM_PUSH
  int ierr = -1;
  int nrhs = ecg->enlFac;
  // Simplify notations
  MPI_Comm     comm        = ecg->comm;
  CPLM_Mat_Dense_t* P      = ecg->P;
  CPLM_Mat_Dense_t* P_prev = ecg->P_prev;
  CPLM_Mat_Dense_t* AP     = ecg->AP;
  CPLM_Mat_Dense_t* alpha  = ecg->alpha;
  CPLM_Mat_Dense_t* beta   = ecg->beta;
  CPLM_Mat_Dense_t* gamma  = ecg->gamma;
  CPLM_Mat_Dense_t* Z      = ecg->Z;
  CPLM_Mat_Dense_t work_s = CPLM_MatDenseNULL();
  double*  work = ecg->work;
  int*    iwork = ecg->iwork;
  double tol = CPLM_EPSILON;
  int M  = P->info.M;
  int m  = P->info.m;
  int t  = P->info.n;
  ierr = CPLM_MatDenseSetInfo(&work_s,t,t,t,t,COL_MAJOR);
  work_s.val = work;

  // RR-QR with Cholesky-like algorithm
  //ierr = CPLM_MatDenseACholRRQR(P,AP,&work_s,tol,iwork,comm);
  ierr = CPLM_MatDenseMatDotProd(AP, P, &work_s, comm);
  // Cholesky of C: R^tR = C
  int nrank;
  ierr = LAPACKE_dpstrf(LAPACK_COL_MAJOR,'U',t,work,t,iwork,&nrank,tol);
  // Permute P and AP
  LAPACKE_dlapmt(LAPACK_COL_MAJOR,1,m,t,P->val,m,iwork);
  LAPACKE_dlapmt(LAPACK_COL_MAJOR,1,m,t,AP->val,m,iwork);
  // Update Sizes of work, P and AP
  CPLM_MatDenseSetInfo(&work_s,nrank,nrank,nrank,nrank,COL_MAJOR);
  CPLM_MatDenseSetInfo(P,P->info.M,P->info.N,P->info.m,nrank,COL_MAJOR);
  CPLM_MatDenseSetInfo(AP,AP->info.M,AP->info.N,AP->info.m,nrank,COL_MAJOR);
  // Solve triangular right system for P
  ierr = CPLM_MatDenseKernelUpperTriangularRightSolve(&work_s, P);
  // Solve triangular right system for AP
  ierr = CPLM_MatDenseKernelUpperTriangularRightSolve(&work_s, AP);

  t  = P->info.n; // Update the value of t!
  // Update the sizes of the other variables
  ierr = CPLM_MatDenseSetInfo(alpha,t,nrhs,t,nrhs,COL_MAJOR);
  if (ecg->ortho_alg == ORTHODIR) {
    int tp = P_prev->info.n;
    ierr = CPLM_MatDenseSetInfo(Z,M,t,m,t,COL_MAJOR);
    ierr = CPLM_MatDenseSetInfo(beta,t,t,t,t,COL_MAJOR);
    ierr = CPLM_MatDenseSetInfo(gamma,tp,t,tp,t,COL_MAJOR);
  }
  else if (ecg->ortho_alg == ORTHOMIN) {
    ierr = CPLM_MatDenseSetInfo(Z,M,nrhs,m,nrhs,COL_MAJOR);
    ierr = CPLM_MatDenseSetInfo(beta,t,nrhs,t,nrhs,COL_MAJOR);
  }
  // Update block size
  ecg->bs = t;

CPLM_POP
  return ierr;
}

int _preAlps_ECGIterateRRQRAlpha(preAlps_ECG_t* ecg) {
CPLM_PUSH
  int ierr = -1;
  // Simplify notations
  /* MPI_Comm     comm        = ecg->comm; */
  CPLM_Mat_Dense_t* P      = ecg->P;
  /* CPLM_Mat_Dense_t* P_prev = ecg->P_prev; */
  CPLM_Mat_Dense_t* AP     = ecg->AP;
  CPLM_Mat_Dense_t* alpha  = ecg->alpha;
  CPLM_Mat_Dense_t* beta   = ecg->beta;
  CPLM_Mat_Dense_t* gamma  = ecg->gamma;
  CPLM_Mat_Dense_t* Z      = ecg->Z;
  CPLM_Mat_Dense_t* H      = ecg->H;
  CPLM_Mat_Dense_t* AH     = ecg->AH;
  CPLM_Mat_Dense_t* delta  = ecg->delta;
  double*  work = ecg->work;
  int*    iwork = ecg->iwork;
  double* tau_s = NULL;   // Householder reflectors
  int M    = P->info.M;
  int m    = P->info.m;
  int nrhs = ecg->enlFac; // Initial size
  int t    = P->info.n;   // Unreduced size
  int t1   = 0;           // Reduced size
  double tol = ecg->tol*ecg->normb/sqrt(nrhs);

  memcpy(work,alpha->val,sizeof(double)*alpha->info.nval);
  // # RRQR
  // Very important: memset iwork to 0
  memset(iwork,0,nrhs*sizeof(int));
  // Reuse work for storing Householder reflectors
  tau_s = work+nrhs*t;
  ierr = LAPACKE_dgeqp3(LAPACK_COL_MAJOR,t,nrhs,work,nrhs,iwork,tau_s);
  for (int i = 0; i < nrhs; i++) {
    if (fabs(work[i + t * i]) > tol) {
      t1++;
    }
    else break;
  }

  //  Reduction of the search directions
  if (t1 > 0 && t1 < nrhs && t1 < t) {
    // Update alpha, P, AP
    LAPACKE_dormqr(LAPACK_COL_MAJOR,'L','T',t,nrhs,t,work,nrhs,tau_s,alpha->val,nrhs);
    LAPACKE_dormqr(LAPACK_COL_MAJOR,'R','N',m,t,t,work,nrhs,tau_s,P->val,m);
    LAPACKE_dormqr(LAPACK_COL_MAJOR,'R','N',m,t,t,work,nrhs,tau_s,AP->val,m);

    // Reduce sizes
    mkl_dimatcopy('C','N',t,nrhs,1.0,alpha->val,t,t1);
    CPLM_MatDenseSetInfo(alpha,t1,nrhs,t1,nrhs,COL_MAJOR);
    CPLM_MatDenseSetInfo(P ,M,t1,m,t1,COL_MAJOR);
    CPLM_MatDenseSetInfo(AP,M,t1,m,t1,COL_MAJOR);

    // Update H and AH
    mkl_domatcopy('C','N',H->info.m, t-t1, 1.0,
                  P->val + t1*m,
                  P->info.lda,
                  H->val + H->info.nval,
                  H->info.lda);
    mkl_domatcopy('C','N',AH->info.m, t-t1, 1.0,
                  AP->val + t1*m,
                  AP->info.lda,
                  AH->val + H->info.nval,
                  AH->info.lda);
    CPLM_MatDenseSetInfo( H, H->info.M, nrhs-t1,  H->info.m, nrhs-t1, COL_MAJOR);
    CPLM_MatDenseSetInfo(AH,AH->info.M, nrhs-t1, AH->info.m, nrhs-t1, COL_MAJOR);

    // Update the other variables
    if (ecg->ortho_alg == ORTHOMIN) {
      CPLM_MatDenseSetInfo(Z, M, nrhs, m, nrhs, COL_MAJOR);
      CPLM_MatDenseSetInfo(beta, t1, nrhs, t1, nrhs, COL_MAJOR);
      CPLM_MatDenseSetInfo(delta, nrhs-t1, nrhs, nrhs-t1, nrhs, COL_MAJOR);
    }
    else if (ecg->ortho_alg == ORTHODIR) {
      CPLM_MatDenseSetInfo(Z, M, t1, m, t1, COL_MAJOR);
      CPLM_MatDenseSetInfo(beta, t1, t1, t1, t1, COL_MAJOR);
      CPLM_MatDenseSetInfo(gamma, gamma->info.M, t1, gamma->info.m, t1, COL_MAJOR);
      CPLM_MatDenseSetInfo(delta, nrhs-t1, t1, nrhs-t1, t1, COL_MAJOR);
    }
  }

  // Update block size
  ecg->bs = t1;

CPLM_POP
  return ierr;
}

int preAlps_ECGFinalize(preAlps_ECG_t* ecg, double* solution) {
CPLM_PUSH
  int ierr = 0;
  // Simplify notations
  CPLM_Mat_Dense_t* X = ecg->X;
  CPLM_DVector_t sol = CPLM_DVectorNULL();
  sol.nval = X->info.m;
  sol.val  = solution;
  // Get the solution
  ierr = CPLM_MatDenseKernelSumColumns(X, &sol);CPLM_CHKERR(ierr);
  _preAlps_ECGFree(ecg);
CPLM_POP
  return ierr;
}

void preAlps_ECGPrint(preAlps_ECG_t* ecg, int verbosity) {
CPLM_PUSH
  int rank;
  MPI_Comm_rank(ecg->comm,&rank);
  printf("[%d] prints ECG_t...\n", rank);
  printf("=== Summary ===\n");
  printf("\titer: %d\n\tres : %e\n\tbs  : %1d\n",ecg->iter,ecg->res,ecg->bs);
  if (verbosity > 1) {
    printf("=== Memory consumption ===\n");
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
      CPLM_MatDensePrintfInfo("delta",  ecg->delta);
    }
    printf("\n");
  }
  printf("[%d] ends printing ECG_t!\n", rank);
CPLM_POP
}

void _preAlps_ECGFree(preAlps_ECG_t* ecg) {
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
    CPLM_MatDenseFree(ecg->delta);
    if (ecg->delta != NULL)
      free(ecg->delta);
  }
  if (ecg->work != NULL)
    free(ecg->work);
CPLM_POP
}

/******************************************************************************/
