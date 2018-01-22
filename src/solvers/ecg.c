/******************************************************************************/
/* Author     : Olivier Tissot                                                */
/* Creation   : 2016/06/24                                                    */
/* Description: Enlarged Preconditioned C(onjugate) G(radient)                */
/******************************************************************************/

/******************************************************************************/
/*                                  INCLUDE                                   */
/******************************************************************************/
/* STD */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
/* MPI */
#include <mpi.h>
/* MKL */
#include<mkl.h>
/* CPaLAMeM */
#include <cpalamem_macro.h>
#include <mat_csr.h>
#include <mat_dense.h>
#include <ivector.h>
#include <dvector.h>
#include <cholqr.h>
#include <matmult.h>
#include <kernels.h>
#include <cpalamem_instrumentation.h>
/* preAlps */
#include "ecg.h"
/******************************************************************************/

/******************************************************************************/
/*                                    CODE                                    */
/******************************************************************************/

int _preAlps_ECGMalloc(preAlps_ECG_t* ecg) {
CPLM_PUSH
  int ierr = 0;
  int allocatedSize = 0;
  int M = ecg->globPbSize, m = ecg->locPbSize, t = ecg->enlFac;
  // Malloc the pointers
  ecg->X     = (CPLM_Mat_Dense_t*) malloc(sizeof(CPLM_Mat_Dense_t));
  ecg->R     = (CPLM_Mat_Dense_t*) malloc(sizeof(CPLM_Mat_Dense_t));
  ecg->V     = (CPLM_Mat_Dense_t*) malloc(sizeof(CPLM_Mat_Dense_t));
  ecg->AV    = (CPLM_Mat_Dense_t*) malloc(sizeof(CPLM_Mat_Dense_t));
  ecg->Z     = (CPLM_Mat_Dense_t*) malloc(sizeof(CPLM_Mat_Dense_t));
  ecg->alpha = (CPLM_Mat_Dense_t*) malloc(sizeof(CPLM_Mat_Dense_t));
  ecg->beta  = (CPLM_Mat_Dense_t*) malloc(sizeof(CPLM_Mat_Dense_t));
  // Set sizes
  if (ecg->ortho_alg == ORTHOMIN) {
    allocatedSize = 5*m*t + 2*t*t;
    CPLM_MatDenseSetInfo(ecg->X    , M, t, m, t, COL_MAJOR);
    CPLM_MatDenseSetInfo(ecg->R    , M, t, m, t, COL_MAJOR);
    CPLM_MatDenseSetInfo(ecg->V    , M, t, m, t, COL_MAJOR);
    CPLM_MatDenseSetInfo(ecg->AV   , M, t, m, t, COL_MAJOR);
    CPLM_MatDenseSetInfo(ecg->Z    , M, t, m, t, COL_MAJOR);
    CPLM_MatDenseSetInfo(ecg->alpha, t, t, t, t, COL_MAJOR);
    CPLM_MatDenseSetInfo(ecg->beta , t, t, t, t, COL_MAJOR);
  }
  else if (ecg->ortho_alg == ORTHODIR) {
    allocatedSize = 7*m*t + 3*t*t;
    CPLM_MatDenseSetInfo(ecg->X    ,   M,   t,   m,   t, COL_MAJOR);
    CPLM_MatDenseSetInfo(ecg->R    ,   M,   t,   m,   t, COL_MAJOR);
    CPLM_MatDenseSetInfo(ecg->V    ,   M, 2*t,   m, 2*t, COL_MAJOR);
    CPLM_MatDenseSetInfo(ecg->AV   ,   M, 2*t,   m, 2*t, COL_MAJOR);
    CPLM_MatDenseSetInfo(ecg->Z    ,   M,   t,   m,   t, COL_MAJOR);
    CPLM_MatDenseSetInfo(ecg->alpha,   t,   t,   t,   t, COL_MAJOR);
    CPLM_MatDenseSetInfo(ecg->beta , 2*t,   t, 2*t,   t, COL_MAJOR);
  }
  // Allocate the whole memory
  ecg->work  = (double*) mkl_calloc(allocatedSize,sizeof(double),64);
  ecg->iwork = (int*) mkl_calloc(ecg->enlFac,sizeof(int),32);
  // Distribute it among variables
  if (ecg->ortho_alg == ORTHOMIN) {
    ecg->V->val     = ecg->work;
    ecg->AV->val    = ecg->work +   m*t;
    ecg->Z->val     = ecg->work + 2*m*t;
    ecg->R->val     = ecg->work + 3*m*t;
    ecg->X->val     = ecg->work + 4*m*t;
    ecg->alpha->val = ecg->work + 5*m*t;
    ecg->beta->val  = ecg->work + 5*m*t + t*t;
  }
  else if (ecg->ortho_alg == ORTHODIR) {
    ecg->V->val     = ecg->work;
    ecg->AV->val    = ecg->work + 2*m*t;
    ecg->Z->val     = ecg->work + 4*m*t;
    ecg->R->val     = ecg->work + 5*m*t;
    ecg->X->val     = ecg->work + 6*m*t;
    ecg->alpha->val = ecg->work + 7*m*t;
    ecg->beta->val  = ecg->work + 7*m*t + t*t;
  }
  // User interface variables
  ecg->P  = (CPLM_Mat_Dense_t*) malloc(sizeof(CPLM_Mat_Dense_t));
  ecg->AP = (CPLM_Mat_Dense_t*) malloc(sizeof(CPLM_Mat_Dense_t));
  CPLM_MatDenseSetInfo(ecg->P , M, t, m ,t, COL_MAJOR);
  CPLM_MatDenseSetInfo(ecg->AP, M, t, m ,t, COL_MAJOR);
  ecg->P->val  = ecg->V->val;
  ecg->AP->val = ecg->AV->val;
  ecg->P_p     = ecg->V->val;
  ecg->AP_p    = ecg->AV->val;
  ecg->R_p     = ecg->R->val;
  ecg->Z_p     = ecg->Z->val;
CPLM_POP
  return ierr;
}

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
  CPLM_Mat_Dense_t* R = ecg->R;
  double* normb_p = &(ecg->normb);
  MPI_Comm_rank(ecg->comm, &rank);
  ecg->iter = 0;
  // Compute normb
  for (int i = 0; i < ecg->locPbSize; ++i)
    *normb_p += pow(rhs[i],2);
  // Sum over all processes
  MPI_Allreduce(MPI_IN_PLACE,
                normb_p,
                1,
                MPI_DOUBLE,
                MPI_SUM,
                ecg->comm);
  *normb_p = sqrt(*normb_p);
  // Initialize res, iter and bs
  ecg->res  = -1.E0;
  ecg->iter = -1;
  ecg->bs   = ecg->enlFac;
  ecg->kbs  = ecg->V->info.n;
  // Warm-up some MKL kernels in order to speed-up the iterations
  if (ecg->ortho_alg == ORTHODIR) {
    int nrhs = ecg->enlFac;
    ierr = LAPACKE_dgeqrf(LAPACK_COL_MAJOR,nrhs,nrhs,ecg->beta->val,
                          nrhs,ecg->beta->val + nrhs*nrhs);
    LAPACKE_dormqr(LAPACK_COL_MAJOR,'L','T',nrhs,nrhs,nrhs,ecg->beta->val,
                   nrhs,ecg->beta->val + nrhs*nrhs,ecg->alpha->val,nrhs);
  }
  // First we construct R_0 by splitting b
  //nCol = rank % (ecg->enlFac);
  nCol = (int) (rank * (ecg->enlFac) / size);
  ierr = _preAlps_ECGSplit(rhs, R, nCol);
  // Then we need to construct R_0 and P_0
  *rci_request = 0;
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
  double* work        = ecg->work;
  double* res_p       = &(ecg->res);
  double tolerance    = ecg->tol;
  double normb        = ecg->normb;
  int iterMax         = ecg->maxIter;
  int m = ecg->locPbSize, t = ecg->enlFac;

  CPLM_ASSERT(stop != NULL);

  // Frobenius norm of R
  CPLM_Mat_Dense_t RtR_s = CPLM_MatDenseNULL();
  CPLM_MatDenseSetInfo(&RtR_s, t, t, t, t, COL_MAJOR);
  if (ecg->ortho_alg == ORTHOMIN) {
    RtR_s.val = work + 5*m*t + t*t;
  }
  else if (ecg->ortho_alg == ORTHODIR) {
    RtR_s.val = work + 7*m*t + 2*t*t;
  }
  // Do local dot product
  CPLM_MatDenseKernelMatDotProd(R,R,&RtR_s);
  // Sum local dot products in place (no mem alloc needed)
  ierr = MPI_Allreduce(MPI_IN_PLACE, RtR_s.val, RtR_s.info.nval,
                       MPI_DOUBLE, MPI_SUM, comm);
  // Compute trace of RtR
  *res_p = 0.E0;
  for (int i = 0; i < RtR_s.info.n; ++i)
    *res_p += RtR_s.val[i + RtR_s.info.lda*i];
  *res_p = sqrt(*res_p);

  // Stopping criterion
  if (*res_p > normb*tolerance && ecg->iter < iterMax && ecg->bs > 0 )
    *stop = 0; // we continue
  else
    *stop = 1; // we stop
CPLM_POP
  return ierr;
}

int preAlps_ECGIterate(preAlps_ECG_t* ecg, int* rci_request) {
CPLM_PUSH
  int ierr = 0;
  if (ecg->ortho_alg == ORTHOMIN)
    _preAlps_ECGIterateOmin(ecg, rci_request);
  else if (ecg->ortho_alg == ORTHODIR)
    _preAlps_ECGIterateOdir(ecg, rci_request);
CPLM_POP
  return ierr;
}

int _preAlps_ECGIterateOmin(preAlps_ECG_t* ecg, int* rci_request) {
CPLM_PUSH
CPLM_OPEN_TIMER
  int ierr = 0;
  // Simplify notations
  MPI_Comm          comm  = ecg->comm;
  CPLM_Mat_Dense_t* P     = ecg->P;
  CPLM_Mat_Dense_t* AP    = ecg->AP;
  CPLM_Mat_Dense_t* X     = ecg->X;
  CPLM_Mat_Dense_t* R     = ecg->R;
  CPLM_Mat_Dense_t* Z     = ecg->Z;
  CPLM_Mat_Dense_t* alpha = ecg->alpha;
  CPLM_Mat_Dense_t* beta  = ecg->beta;
  CPLM_Mat_Dense_t work_s = CPLM_MatDenseNULL();
  double*  work = ecg->work;
  int*    iwork = ecg->iwork;
  double tol = 1e-15;
  int m = ecg->locPbSize, M = ecg->globPbSize, nrhs = ecg->enlFac;
  int t = P->info.n;
  if (*rci_request == 0) {
    ierr = CPLM_MatDenseSetInfo(&work_s,t,t,t,t,COL_MAJOR);
    work_s.val = work + 5*m*nrhs + nrhs*nrhs;
    CPLM_TIC(step1, "ACHQR: AP^t P")
    ierr = CPLM_MatDenseKernelMatDotProd(AP, P, &work_s);
    CPLM_TAC(step1)
    CPLM_TIC(step2, "ACHQR: Allreduce")
    ierr = MPI_Allreduce(MPI_IN_PLACE, work_s.val, work_s.info.nval,
                         MPI_DOUBLE, MPI_SUM, comm);
    CPLM_TAC(step2)
    CPLM_TIC(step3, "ACHQR: dpotrf")
    ierr = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', t, work_s.val, t);
    CPLM_TAC(step3)
    if ( ierr != 0) {
      CPLM_Abort("ACHQR: dpotrf:\n ERROR: P^tAP is not spd!");
    }
    CPLM_TIC(step4, "ACHQR: P / (P^t AP)^(-1/2)")
    cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans,
                CblasNonUnit, m, t, 1.E0, work_s.val, t, P->val, m);
    CPLM_TAC(step4)
    CPLM_TIC(step5, "ACHQR: AP / (P^t AP)^(-1/2)")
    cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans,
                CblasNonUnit, m, t, 1.E0, work_s.val, t, AP->val, m);
    CPLM_TAC(step5)
    CPLM_TIC(step6, "alpha: P^t R")
    ierr = CPLM_MatDenseKernelMatDotProd(P,R,alpha);
    CPLM_TAC(step6)
    CPLM_TIC(step7,"alpha: Allreduce")
    ierr = MPI_Allreduce(MPI_IN_PLACE, alpha->val, alpha->info.nval,
                         MPI_DOUBLE, MPI_SUM, comm);
    CPLM_TAC(step7)
    CPLM_TIC(step8,"X = X + P*alpha")
    ierr = CPLM_MatDenseKernelMatMult(P,'N',alpha,'N',X,1.E0,1.E0);
    CPLM_TAC(step8)
    CPLM_TIC(step9,"R = R - AP*alpha")
    ierr = CPLM_MatDenseKernelMatMult(AP,'N',alpha,'N',R,-1.E0,1.E0);
    CPLM_TAC(step9)
    // Iteration finished
    ecg->iter++;
    // Now we need the preconditioner
    *rci_request = 1;
  }
  else if (*rci_request == 1) {
    CPLM_TIC(step10, "beta : AP^t Z")
    ierr = CPLM_MatDenseKernelMatDotProd(AP,Z,beta);
    CPLM_TAC(step10)
    CPLM_TIC(step11, "beta : Allreduce")
    ierr = MPI_Allreduce(MPI_IN_PLACE, beta->val, beta->info.nval,
                         MPI_DOUBLE, MPI_SUM, comm);
    CPLM_TAC(step11)
    CPLM_TIC(step12, "Z = Z - P*beta")
    ierr = CPLM_MatDenseKernelMatMult(P,'N',beta,'N',Z,-1.E0,1.E0);
    CPLM_TAC(step12)
    // Swapping time
    CPLM_TIC(step13, "P = Z")
    mkl_domatcopy('C','N',m,nrhs,1.E0,Z->val,m,P->val,m);
    CPLM_TAC(step13)
    /**************** RR-QR with Cholesky-like algorithm **********************/
    if (ecg->bs_red == ADAPT_BS) {
      ierr = CPLM_MatDenseSetInfo(&work_s,nrhs,nrhs,nrhs,nrhs,COL_MAJOR);
      work_s.val = work + 5*m*nrhs + nrhs*nrhs;
      // Do local dot product
      CPLM_TIC(step14, "BF   : P^t P")
      ierr = CPLM_MatDenseKernelMatDotProd(P, P, &work_s);
      CPLM_TAC(step14)
      // Sum local dot products in place (no mem alloc needed)
      CPLM_TIC(step15, "BF   : Allreduce")
      ierr = MPI_Allreduce(MPI_IN_PLACE, work_s.val, work_s.info.nval,
                           MPI_DOUBLE, MPI_SUM, comm);
      CPLM_TAC(step15)
      // Cholesky with pivoting of C := P^tP
      CPLM_TIC(step16, "BF   : dpstrf")
      ierr = LAPACKE_dpstrf(LAPACK_COL_MAJOR,'U',nrhs,work_s.val,nrhs,
                            iwork,&t,tol);
      CPLM_TAC(step16)
      // Permute P
      CPLM_TIC(step17, "BF   : dlapmt(P)")
      LAPACKE_dlapmt(LAPACK_COL_MAJOR,1,m,nrhs,P->val,m,iwork);
      CPLM_TAC(step17)
      // Solve triangular right system for P
      CPLM_TIC(step18, "BF   : P / (P^t P)^(-1/2)")
      cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans,
                  CblasNonUnit, m, t, 1.E0, work_s.val, nrhs, P->val, m);
      CPLM_TAC(step18)
      // Update the sizes
      CPLM_MatDenseSetInfo(P,M,t,m,t,COL_MAJOR);
      CPLM_MatDenseSetInfo(AP,M,t,m,t,COL_MAJOR);
      CPLM_MatDenseSetInfo(alpha,t,nrhs,t,nrhs,COL_MAJOR);
      CPLM_MatDenseSetInfo(beta,t,nrhs,t,nrhs,COL_MAJOR);
      // Update block size
      ecg->bs = t;
    }
    /**************************************************************************/// Now we need A*P to continue
    *rci_request = 0;
  }
CPLM_CLOSE_TIMER
CPLM_POP
  return ierr;
}

int _preAlps_ECGIterateOdir(preAlps_ECG_t* ecg, int* rci_request) {
CPLM_PUSH
CPLM_OPEN_TIMER
  int ierr = 0;
  // Simplify notations
  MPI_Comm          comm  = ecg->comm;
  CPLM_Mat_Dense_t* V     = ecg->V;
  CPLM_Mat_Dense_t* AV    = ecg->AV;
  CPLM_Mat_Dense_t* P     = ecg->P;
  CPLM_Mat_Dense_t* AP    = ecg->AP;
  CPLM_Mat_Dense_t* X     = ecg->X;
  CPLM_Mat_Dense_t* R     = ecg->R;
  CPLM_Mat_Dense_t* Z     = ecg->Z;
  CPLM_Mat_Dense_t* alpha = ecg->alpha;
  CPLM_Mat_Dense_t* beta  = ecg->beta;
  CPLM_Mat_Dense_t work_s = CPLM_MatDenseNULL();
  double*  work = ecg->work;
  int*    iwork = ecg->iwork;
  int m = ecg->locPbSize, M = ecg->globPbSize, nrhs = ecg->enlFac;
  int t = P->info.n, t1 = 0; // Reduced size
  double tol = ecg->tol*ecg->normb/sqrt(nrhs);
  if (*rci_request == 0) {
    ierr = CPLM_MatDenseSetInfo(&work_s,t,t,t,t,COL_MAJOR);
    // work_s.val = work + 7*m*nrhs + 2*nrhs*nrhs;
    work_s.val = work + 7*m*nrhs + nrhs*nrhs;
    CPLM_TIC(step1, "ACHQR   : AP^t P")
    ierr = CPLM_MatDenseKernelMatDotProd(AP, P, &work_s);
    CPLM_TAC(step1)
    CPLM_TIC(step2, "ACHQR   : Allreduce")
    ierr = MPI_Allreduce(MPI_IN_PLACE, work_s.val, work_s.info.nval,
                         MPI_DOUBLE, MPI_SUM, comm);
    CPLM_TAC(step2)
    CPLM_TIC(step3, "ACHQR   : dpotrf")
    ierr = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', t, work_s.val, t);
    CPLM_TAC(step3)
    CPLM_TIC(step4, "ACHQR   : P / (P^t AP)^(-1/2)")
    ierr = CPLM_MatDenseKernelUpperTriangularRightSolve(&work_s, P);
    CPLM_TAC(step4)
    CPLM_TIC(step5, "ACHQR   : AP / (P^t AP)^(-1/2)")
    ierr = CPLM_MatDenseKernelUpperTriangularRightSolve(&work_s, AP);
    CPLM_TAC(step5)
    CPLM_TIC(step6, "alpha   : P^t R")
    ierr = CPLM_MatDenseKernelMatDotProd(P,R,alpha);
    CPLM_TAC(step6)
    CPLM_TIC(step7,"alpha   : Allreduce")
    ierr = MPI_Allreduce(MPI_IN_PLACE, alpha->val, alpha->info.nval,
                         MPI_DOUBLE, MPI_SUM, comm);
    CPLM_TAC(step7)
    /**************** Reduction of the search directions **********************/
    if (ecg->bs_red == ADAPT_BS) {
      double* tau_p  = NULL; // Householder reflectors
      CPLM_TIC(step8, "ADAPT_BS: preparation")
      memcpy(work_s.val,alpha->val,sizeof(double)*alpha->info.nval);
      // 1) SVD on alpha
//      memset(iwork,0,nrhs*sizeof(int)); // Very important: memset iwork to 0
      CPLM_TAC(step8)
      // Reuse work for storing Householder reflectors
      tau_p = work_s.val + nrhs*t;
      CPLM_TIC(step9, "ADAPT_BS: dgesvd")
      ierr = LAPACKE_dgesvd(LAPACK_COL_MAJOR,'O','N',t,nrhs,work_s.val,alpha->info.lda,
                            tau_p,NULL,1,NULL,1,tau_p+t);
//      ierr = LAPACKE_dgeqp3(LAPACK_COL_MAJOR,t,nrhs,work_s.val,
//                            nrhs,iwork,tau_p);
      for (int i = 0; i < t; i++) {
        // if (fabs(work_s.val[i + t * i]) > tol) t1++;
        if (tau_p[i] > tol) t1++;
        else break;
      }
      CPLM_TAC(step9)

      // 2) reduction of the search directions
      if (t1 > 0 && t1 < nrhs && t1 < t) {
        // For in-place update
        CPLM_TIC(step10, "ADAPT_BS: dgeqrf")
        ierr = LAPACKE_dgeqrf(LAPACK_COL_MAJOR,t,t,work_s.val,t,tau_p);
        CPLM_TAC(step10)
        // Update alpha, P, AP
        CPLM_TIC(step11, "ADAPT_BS: dormqr(alpha)")
        LAPACKE_dormqr(LAPACK_COL_MAJOR,'L','T',t,nrhs,t,work_s.val,
                      t,tau_p,alpha->val,t);
        CPLM_TAC(step11)
        CPLM_TIC(step12, "ADAPT_BS: dormqr(P)")
        LAPACKE_dormqr(LAPACK_COL_MAJOR,'R','N',m,t,t,work_s.val,
                      t,tau_p,P->val,m);
        CPLM_TAC(step12)
        CPLM_TIC(step13, "ADAPT_BS: dormqr(AP)")
        LAPACKE_dormqr(LAPACK_COL_MAJOR,'R','N',m,t,t,work_s.val,
                      t,tau_p,AP->val,m);
        CPLM_TAC(step13)
        // Reduce sizes
        CPLM_TIC(step14, "ADAPT_BS: dimatcopy(alpha)")
        mkl_dimatcopy('C','N',t,nrhs,1.E0,alpha->val,t,t1);
        CPLM_TAC(step14)
        CPLM_MatDenseSetInfo(alpha,t1,nrhs,t1,nrhs,COL_MAJOR);
        CPLM_MatDenseSetInfo(P ,M,t1,m,t1,COL_MAJOR);
        CPLM_MatDenseSetInfo(AP,M,t1,m,t1,COL_MAJOR);
        // Update the other variables
        CPLM_MatDenseSetInfo(Z, M, t1, m, t1, COL_MAJOR);
        // Update sizes
        ecg->bs   = t1;
        ecg->kbs  = t + nrhs;
      }
      CPLM_MatDenseSetInfo(beta, ecg->kbs, nrhs, ecg->kbs, nrhs, COL_MAJOR);
      CPLM_MatDenseSetInfo(V, M, ecg->kbs, m, ecg->kbs, COL_MAJOR);
      CPLM_MatDenseSetInfo(AV, M, ecg->kbs, m, ecg->kbs, COL_MAJOR);
    }
    /**************************************************************************/

    CPLM_TIC(step15,"X = X + P*alpha")
    ierr = CPLM_MatDenseKernelMatMult(P,'N',alpha,'N',X,1.E0,1.E0);
    CPLM_TAC(step15)
    CPLM_TIC(step16,"R = R - AP*alpha")
    ierr = CPLM_MatDenseKernelMatMult(AP,'N',alpha,'N',R,-1.E0,1.E0);
    CPLM_TAC(step16)
    // Iteration finished
    ecg->iter++;
    // Now we need the preconditioner
    *rci_request = 1;
  }
  else if (*rci_request == 1) {
    CPLM_TIC(step17, "beta    : AV^t Z")
    ierr = CPLM_MatDenseKernelMatDotProd(AV,Z,beta);
    CPLM_TAC(step17)
    CPLM_TIC(step18, "beta    : Allreduce")
    ierr = MPI_Allreduce(MPI_IN_PLACE, beta->val, beta->info.nval,
                         MPI_DOUBLE, MPI_SUM, comm);
    CPLM_TAC(step18)
    CPLM_TIC(step19, "Z = Z - V*beta")
    ierr = CPLM_MatDenseKernelMatMult(V,'N',beta,'N',Z,-1.E0,1.E0);
    CPLM_TAC(step19)
    // Swapping time
    CPLM_TIC(step20, "domatcopy")
    mkl_domatcopy('C','N',m,t,1.E0,V->val,m,V->val+m*nrhs,m);
    mkl_domatcopy('C','N',m,t,1.E0,AV->val,m,AV->val+m*nrhs,m);
    mkl_domatcopy('C','N',m,t,1.E0,Z->val,m,V->val,m);
    CPLM_TAC(step20)
    // Now we need A*P to continue
    *rci_request = 0;
  }
CPLM_CLOSE_TIMER
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
    CPLM_MatDensePrintfInfo("V",   ecg->V);
    CPLM_MatDensePrintfInfo("AV",  ecg->AV);
    CPLM_MatDensePrintfInfo("P",   ecg->P);
    CPLM_MatDensePrintfInfo("AP",  ecg->AP);
    CPLM_MatDensePrintfInfo("Z",    ecg->Z);
    CPLM_MatDensePrintfInfo("alpha",ecg->alpha);
    CPLM_MatDensePrintfInfo("beta", ecg->beta);
    printf("\n");
  }
  printf("[%d] ends printing ECG_t!\n", rank);
CPLM_POP
}

void _preAlps_ECGFree(preAlps_ECG_t* ecg) {
CPLM_PUSH
  if (ecg->X     != NULL) free(ecg->X);
  if (ecg->R     != NULL) free(ecg->R);
  if (ecg->V     != NULL) free(ecg->V);
  if (ecg->AV    != NULL) free(ecg->AV);
  if (ecg->alpha != NULL) free(ecg->alpha);
  if (ecg->beta  != NULL) free(ecg->beta);
  if (ecg->Z     != NULL) free(ecg->Z);
  if (ecg->P     != NULL) free(ecg->P);
  if (ecg->AP    != NULL) free(ecg->AP);
  mkl_free(ecg->work);
CPLM_POP
}

/******************************************************************************/
