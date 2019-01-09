/**
 * \file    ecg.c
 * \author  Olivier Tissot
 * \date    2016/06/24
 * \brief   Enlarged Preconditioned C(onjugate) G(radient) solver
 *
 * \details Implements Orthomin, Orthodir as well as their dynamic
 *          counterparts (BF-Omin and D-Odir).
 */

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
//#include <cpalamem_macro.h>
#include <preAlps_cplm_utils.h>
#include <preAlps_cplm_timing.h>
#include <preAlps_cplm_matcsr.h>
#include <preAlps_cplm_matdense.h>
#include <preAlps_cplm_ivector.h>
#include <preAlps_cplm_dvector.h>
#include <cholqr.h>
//#include <matmult.h>
#include <preAlps_cplm_kernels.h>
//#include <cpalamem_instrumentation.h>
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
  int m = ecg->locPbSize, t = ecg->enlFac;
  // Malloc the pointers
  ecg->X     = (CPLM_Mat_Dense_t*) malloc(sizeof(CPLM_Mat_Dense_t));
  ecg->R     = (CPLM_Mat_Dense_t*) malloc(sizeof(CPLM_Mat_Dense_t));
  ecg->V     = (CPLM_Mat_Dense_t*) malloc(sizeof(CPLM_Mat_Dense_t));
  ecg->AV    = (CPLM_Mat_Dense_t*) malloc(sizeof(CPLM_Mat_Dense_t));
  ecg->Z     = (CPLM_Mat_Dense_t*) malloc(sizeof(CPLM_Mat_Dense_t));
  ecg->alpha = (CPLM_Mat_Dense_t*) malloc(sizeof(CPLM_Mat_Dense_t));
  ecg->beta  = (CPLM_Mat_Dense_t*) malloc(sizeof(CPLM_Mat_Dense_t));
  // Set sizes
  if (ecg->ortho_alg == ORTHOMIN)
    allocatedSize = 5*m*t + 2*t*t;
  else if (ecg->ortho_alg == ORTHODIR)
    allocatedSize = 7*m*t + 3*t*t;
  else if (ecg->ortho_alg == ORTHODIR_FUSED)
    allocatedSize = 7*m*t + 5*t*t + 2*t;
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
  else if (ecg->ortho_alg == ORTHODIR || ecg->ortho_alg == ORTHODIR_FUSED) {
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
  ecg->P->val  = ecg->V->val;
  ecg->AP->val = ecg->AV->val;
  ecg->P_p     = ecg->V->val;
  ecg->AP_p    = ecg->AV->val;
  ecg->R_p     = ecg->R->val;
  ecg->Z_p     = ecg->Z->val;
  // Be careful the sizes are not set in the CPLM_Mat_Dense_t structures!
  // You have to call ECGReset after in order to do this!
CPLM_POP
  return ierr;
}

int _preAlps_ECGReset(preAlps_ECG_t* ecg, double* rhs, int* rci_request) {
CPLM_PUSH
  int rank, ierr = 0;
  int nCol = 0;
  double trash_t;
  // Reset timings
  ecg->tot_t  = 0.E0;
  ecg->comm_t  = 0.E0;
  ecg->trsm_t  = 0.E0;
  ecg->gemm_t  = 0.E0;
  ecg->potrf_t = 0.E0;
  ecg->pstrf_t = 0.E0;
  ecg->lapmt_t = 0.E0;
  ecg->gesvd_t = 0.E0;
  ecg->geqrf_t = 0.E0;
  ecg->ormqr_t = 0.E0;
  ecg->copy_t  = 0.E0;
  // Set sizes
  int M = ecg->globPbSize, m = ecg->locPbSize, t = ecg->enlFac;
  if (ecg->ortho_alg == ORTHOMIN) {
    CPLM_MatDenseSetInfo(ecg->X    , M, t, m, t, COL_MAJOR);
    CPLM_MatDenseSetInfo(ecg->R    , M, t, m, t, COL_MAJOR);
    CPLM_MatDenseSetInfo(ecg->V    , M, t, m, t, COL_MAJOR);
    CPLM_MatDenseSetInfo(ecg->AV   , M, t, m, t, COL_MAJOR);
    CPLM_MatDenseSetInfo(ecg->Z    , M, t, m, t, COL_MAJOR);
    CPLM_MatDenseSetInfo(ecg->alpha, t, t, t, t, COL_MAJOR);
    CPLM_MatDenseSetInfo(ecg->beta , t, t, t, t, COL_MAJOR);
  }
  else if (ecg->ortho_alg == ORTHODIR || ecg->ortho_alg == ORTHODIR_FUSED) {
    CPLM_MatDenseSetInfo(ecg->X    ,   M,   t,   m,   t, COL_MAJOR);
    CPLM_MatDenseSetInfo(ecg->R    ,   M,   t,   m,   t, COL_MAJOR);
    CPLM_MatDenseSetInfo(ecg->V    ,   M, 2*t,   m, 2*t, COL_MAJOR);
    CPLM_MatDenseSetInfo(ecg->AV   ,   M, 2*t,   m, 2*t, COL_MAJOR);
    CPLM_MatDenseSetInfo(ecg->Z    ,   M,   t,   m,   t, COL_MAJOR);
    CPLM_MatDenseSetInfo(ecg->alpha,   t,   t,   t,   t, COL_MAJOR);
    CPLM_MatDenseSetInfo(ecg->beta , 2*t,   t, 2*t,   t, COL_MAJOR);
  }
  CPLM_MatDenseSetInfo(ecg->P , M, t, m ,t, COL_MAJOR);
  CPLM_MatDenseSetInfo(ecg->AP, M, t, m ,t, COL_MAJOR);
  // Simplify notations
  CPLM_Mat_Dense_t* R = ecg->R;
  double* normb_p = &(ecg->normb);
  MPI_Comm_rank(ecg->comm, &rank);
  ecg->iter = 0;
  // Compute normb
  *normb_p =0.E0;
  for (int i = 0; i < ecg->locPbSize; ++i)
    *normb_p += pow(rhs[i],2);
  // Sum over all processes
  trash_t = MPI_Wtime();
  MPI_Allreduce(MPI_IN_PLACE,
                  normb_p,
                  1,
                  MPI_DOUBLE,
                  MPI_SUM,
                  ecg->comm);
  ecg->comm_t += MPI_Wtime() - trash_t;
  *normb_p = sqrt(*normb_p);
  // Initialize res, iter and bs
  ecg->res  = 1.E0;
  ecg->iter = 0;
  ecg->bs   = t;
  ecg->kbs  = ecg->V->info.n;
  // First we construct R_0 by splitting b
  nCol = rank % t;
  // int size;
  // MPI_Comm_size(ecg->comm,&size);
  // nCol = (int) (rank * t / size);
  ierr = _preAlps_ECGSplit(rhs, R, nCol);
  // Then we need to construct R_0 and P_0
  *rci_request = 0;
CPLM_POP
    return ierr;
}

int preAlps_ECGInitialize(preAlps_ECG_t* ecg, double* rhs, int* rci_request) {
CPLM_PUSH
  int size, ierr = 0;

  // Firstly check that nrhs < size
  MPI_Comm_size(ecg->comm,&size);
  if (size < ecg->enlFac) {
    CPLM_Abort("Enlarging factor must be lower than the number of processors"
               " in the MPI communicator! size: %d ; enlarging factor: %d",
               size,ecg->enlFac);
  }

  // Allocate Memory
  ierr = _preAlps_ECGMalloc(ecg);
  // Set to zero the parameters and split rhs
  ierr = _preAlps_ECGReset(ecg,rhs,rci_request);
  // Warm-up some MKL kernels in order to speed-up the iterations
  if ((ecg->ortho_alg == ORTHODIR || ecg->ortho_alg == ORTHODIR_FUSED) && ecg->bs_red == ADAPT_BS) {
    int nrhs = ecg->enlFac;
    ierr = LAPACKE_dgeqrf(LAPACK_COL_MAJOR,nrhs,nrhs,ecg->beta->val,
                          nrhs,ecg->beta->val + nrhs*nrhs);
    LAPACKE_dormqr(LAPACK_COL_MAJOR,'L','T',nrhs,nrhs,nrhs,ecg->beta->val,
                   nrhs,ecg->beta->val + nrhs*nrhs,ecg->alpha->val,nrhs);
  }
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
  double trash_t, trash_tg = MPI_Wtime();
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
  trash_t = MPI_Wtime();
  CPLM_MatDenseKernelMatDotProd(R,R,&RtR_s);
  ecg->gemm_t += MPI_Wtime() - trash_t;
  // Sum local dot products in place (no mem alloc needed)
  trash_t = MPI_Wtime();
  ierr = MPI_Allreduce(MPI_IN_PLACE, RtR_s.val, RtR_s.info.nval,
                       MPI_DOUBLE, MPI_SUM, comm);
  ecg->comm_t += MPI_Wtime() - trash_t;
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
  ecg->tot_t += MPI_Wtime() - trash_tg;
CPLM_POP
  return ierr;
}

int preAlps_ECGIterate(preAlps_ECG_t* ecg, int* rci_request) {
CPLM_PUSH
  double trash_t = MPI_Wtime();
  int ierr = 0;
  if (ecg->ortho_alg == ORTHOMIN)
    _preAlps_ECGIterateOmin(ecg, rci_request);
  else if (ecg->ortho_alg == ORTHODIR)
    _preAlps_ECGIterateOdir(ecg, rci_request);
  else if (ecg->ortho_alg == ORTHODIR_FUSED)
    _preAlps_ECGIterateOdirFused(ecg, rci_request);
  ecg->tot_t += MPI_Wtime() - trash_t;
CPLM_POP
  return ierr;
}

// TODO finish timing
int _preAlps_ECGIterateOmin(preAlps_ECG_t* ecg, int* rci_request) {
CPLM_PUSH
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
  double tol = -1.E0, trash_t;
  int m = ecg->locPbSize, M = ecg->globPbSize, nrhs = ecg->enlFac;
  int t = P->info.n;
  if (*rci_request == 0) {
    ierr = CPLM_MatDenseSetInfo(&work_s,t,t,t,t,COL_MAJOR);
    work_s.val = work + 5*m*nrhs + nrhs*nrhs;
    trash_t = MPI_Wtime();
    ierr = CPLM_MatDenseKernelMatDotProd(AP, P, &work_s);
    ecg->gemm_t += MPI_Wtime() - trash_t;
    trash_t = MPI_Wtime();
    ierr = MPI_Allreduce(MPI_IN_PLACE, work_s.val, work_s.info.nval,
                         MPI_DOUBLE, MPI_SUM, comm);
    ecg->comm_t += MPI_Wtime() - trash_t;
    trash_t = MPI_Wtime();
    ierr = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', t, work_s.val, t);
    ecg->potrf_t += MPI_Wtime() - trash_t;
    if ( ierr != 0) {
      CPLM_Abort("ACHQR: dpotrf:\n ERROR: P^tAP is not spd!");
    }
    trash_t = MPI_Wtime();
    cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans,
                CblasNonUnit, m, t, 1.E0, work_s.val, t, P->val, m);
    cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans,
                CblasNonUnit, m, t, 1.E0, work_s.val, t, AP->val, m);
    ecg->trsm_t += MPI_Wtime() - trash_t;
    trash_t = MPI_Wtime();
    ierr = CPLM_MatDenseKernelMatDotProd(P,R,alpha);
    ecg->gemm_t += MPI_Wtime() - trash_t;
    trash_t = MPI_Wtime();
    ierr = MPI_Allreduce(MPI_IN_PLACE, alpha->val, alpha->info.nval,
                         MPI_DOUBLE, MPI_SUM, comm);
    ecg->comm_t += MPI_Wtime() - trash_t;
    trash_t = MPI_Wtime();
    ierr = CPLM_MatDenseKernelMatMult(P,'N',alpha,'N',X,1.E0,1.E0);
    ierr = CPLM_MatDenseKernelMatMult(AP,'N',alpha,'N',R,-1.E0,1.E0);
    ecg->gemm_t += MPI_Wtime() - trash_t;
    // Iteration finished
    ecg->iter++;
    // Now we need the preconditioner
    *rci_request = 1;
  }
  else if (*rci_request == 1) {
    trash_t = MPI_Wtime();
    ierr = CPLM_MatDenseKernelMatDotProd(AP,Z,beta);
    ecg->gemm_t += MPI_Wtime() - trash_t;
    trash_t = MPI_Wtime();
    ierr = MPI_Allreduce(MPI_IN_PLACE, beta->val, beta->info.nval,
                         MPI_DOUBLE, MPI_SUM, comm);
    ecg->comm_t += MPI_Wtime() - trash_t;
    trash_t = MPI_Wtime();
    ierr = CPLM_MatDenseKernelMatMult(P,'N',beta,'N',Z,-1.E0,1.E0);
    ecg->gemm_t += MPI_Wtime() - trash_t;
    // Swapping time
    trash_t = MPI_Wtime();
    mkl_domatcopy('C','N',m,nrhs,1.E0,Z->val,m,P->val,m);
    ecg->copy_t += MPI_Wtime() - trash_t;
    /**************** RR-QR with Cholesky-like algorithm **********************/
    if (ecg->bs_red == ADAPT_BS) {
      ierr = CPLM_MatDenseSetInfo(&work_s,nrhs,nrhs,nrhs,nrhs,COL_MAJOR);
      work_s.val = work + 5*m*nrhs + nrhs*nrhs;
      // Do local dot product
      trash_t = MPI_Wtime();
      ierr = CPLM_MatDenseKernelMatDotProd(P, P, &work_s);
      ecg->gemm_t += MPI_Wtime() - trash_t;
      // Sum local dot products in place (no mem alloc needed)
      trash_t = MPI_Wtime();
      ierr = MPI_Allreduce(MPI_IN_PLACE, work_s.val, work_s.info.nval,
                           MPI_DOUBLE, MPI_SUM, comm);
      ecg->comm_t += MPI_Wtime() - trash_t;
      // Cholesky with pivoting of C := P^tP
      trash_t = MPI_Wtime();
      ierr = LAPACKE_dpstrf(LAPACK_COL_MAJOR,'U',nrhs,work_s.val,nrhs,
                            iwork,&t,tol);
      ecg->pstrf_t += MPI_Wtime() - trash_t;
      // Permute P
      trash_t = MPI_Wtime();
      LAPACKE_dlapmt(LAPACK_COL_MAJOR,1,m,nrhs,P->val,m,iwork);
      ecg->lapmt_t += MPI_Wtime() - trash_t;
      // Solve triangular right system for P
      cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans,
                  CblasNonUnit, m, t, 1.E0, work_s.val, nrhs, P->val, m);
      ecg->trsm_t += MPI_Wtime() - trash_t;
      // Update the sizes
      CPLM_MatDenseSetInfo(P,M,t,m,t,COL_MAJOR);
      CPLM_MatDenseSetInfo(AP,M,t,m,t,COL_MAJOR);
      CPLM_MatDenseSetInfo(alpha,t,nrhs,t,nrhs,COL_MAJOR);
      CPLM_MatDenseSetInfo(beta,t,nrhs,t,nrhs,COL_MAJOR);
      // Update block size
      ecg->bs = t;
    }
    /**************************************************************************/
    // Now we need A*P to continue
    *rci_request = 0;
  }
CPLM_POP
  return ierr;
}

int _preAlps_ECGIterateOdir(preAlps_ECG_t* ecg, int* rci_request) {
CPLM_PUSH
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
  int m = ecg->locPbSize, M = ecg->globPbSize, nrhs = ecg->enlFac;
  int t = P->info.n, t1 = 0; // Reduced size
  double tol = ecg->tol*ecg->normb/sqrt(nrhs), trash_t;
  if (*rci_request == 0) {
    ierr = CPLM_MatDenseSetInfo(&work_s,t,t,t,t,COL_MAJOR);
    // work_s.val = work + 7*m*nrhs + 2*nrhs*nrhs;
    work_s.val = work + 7*m*nrhs + nrhs*nrhs;
    ierr = CPLM_MatDenseKernelMatDotProd(AP, P, &work_s);
    trash_t = MPI_Wtime();
    ierr = MPI_Allreduce(MPI_IN_PLACE, work_s.val, work_s.info.nval,
                         MPI_DOUBLE, MPI_SUM, comm);
    ecg->comm_t += MPI_Wtime() - trash_t;
    trash_t = MPI_Wtime();
    ierr = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', t, work_s.val, t);
    ecg->potrf_t += MPI_Wtime() - trash_t;
    trash_t = MPI_Wtime();
    ierr = CPLM_MatDenseKernelUpperTriangularRightSolve(&work_s, P);
    ierr = CPLM_MatDenseKernelUpperTriangularRightSolve(&work_s, AP);
    ecg->trsm_t += MPI_Wtime() - trash_t;
    trash_t = MPI_Wtime();
    ierr = CPLM_MatDenseKernelMatDotProd(P,R,alpha);
    ecg->gemm_t += MPI_Wtime() - trash_t;
    trash_t = MPI_Wtime();
    ierr = MPI_Allreduce(MPI_IN_PLACE, alpha->val, alpha->info.nval,
                         MPI_DOUBLE, MPI_SUM, comm);
    ecg->comm_t += MPI_Wtime() - trash_t;
    /**************** Reduction of the search directions **********************/
    if (ecg->bs_red == ADAPT_BS) {
      double* tau_p  = NULL; // Householder reflectors
      trash_t = MPI_Wtime();
      memcpy(work_s.val,alpha->val,sizeof(double)*alpha->info.nval);
      ecg->copy_t += MPI_Wtime() - trash_t;
      // 1) SVD on alpha
//      memset(iwork,0,nrhs*sizeof(int)); // Very important: memset iwork to 0
      // Reuse work for storing Householder reflectors
      tau_p = work_s.val + nrhs*t;
      trash_t = MPI_Wtime();
      ierr = LAPACKE_dgesvd(LAPACK_COL_MAJOR,'O','N',t,nrhs,work_s.val,alpha->info.lda,
                            tau_p,NULL,1,NULL,1,tau_p+t);
      ecg->gesvd_t += MPI_Wtime() - trash_t;
//      ierr = LAPACKE_dgeqp3(LAPACK_COL_MAJOR,t,nrhs,work_s.val,
//                            nrhs,iwork,tau_p);
      for (int i = 0; i < t; i++) {
        // if (fabs(work_s.val[i + t * i]) > tol) t1++;
        if (tau_p[i] > tol) t1++;
        else break;
      }

      // 2) reduction of the search directions
      if (t1 > 0 && t1 < nrhs && t1 < t) {
        // For in-place update
        trash_t = MPI_Wtime();
        ierr = LAPACKE_dgeqrf(LAPACK_COL_MAJOR,t,t,work_s.val,t,tau_p);
        ecg->geqrf_t += MPI_Wtime() - trash_t;
        // Update alpha, P, AP
        trash_t = MPI_Wtime();
        LAPACKE_dormqr(LAPACK_COL_MAJOR,'L','T',t,nrhs,t,work_s.val,
                      t,tau_p,alpha->val,t);
        LAPACKE_dormqr(LAPACK_COL_MAJOR,'R','N',m,t,t,work_s.val,
                      t,tau_p,P->val,m);
        LAPACKE_dormqr(LAPACK_COL_MAJOR,'R','N',m,t,t,work_s.val,
                      t,tau_p,AP->val,m);
        ecg->ormqr_t += MPI_Wtime() - trash_t;
        // Reduce sizes
        trash_t = MPI_Wtime();
        mkl_dimatcopy('C','N',t,nrhs,1.E0,alpha->val,t,t1);
        ecg->copy_t += MPI_Wtime() - trash_t;
        CPLM_MatDenseSetInfo(alpha,t1,nrhs,t1,nrhs,COL_MAJOR);
        CPLM_MatDenseSetInfo(P ,M,t1,m,t1,COL_MAJOR);
        CPLM_MatDenseSetInfo(AP,M,t1,m,t1,COL_MAJOR);
        // Update the other variables
        CPLM_MatDenseSetInfo(Z, M, t1, m, t1, COL_MAJOR);
        // Update sizes
        ecg->bs   = t1;
        ecg->kbs  = t + nrhs;
      }
      CPLM_MatDenseSetInfo(beta, ecg->kbs, t1, ecg->kbs, t1, COL_MAJOR);
      CPLM_MatDenseSetInfo(V, M, ecg->kbs, m, ecg->kbs, COL_MAJOR);
      CPLM_MatDenseSetInfo(AV, M, ecg->kbs, m, ecg->kbs, COL_MAJOR);
    }
    /**************************************************************************/
    trash_t = MPI_Wtime();
    ierr = CPLM_MatDenseKernelMatMult(P,'N',alpha,'N',X,1.E0,1.E0);
    ierr = CPLM_MatDenseKernelMatMult(AP,'N',alpha,'N',R,-1.E0,1.E0);
    ecg->gemm_t += MPI_Wtime() - trash_t;
    // Iteration finished
    ecg->iter++;
    // Now we need the preconditioner
    *rci_request = 1;
  }
  else if (*rci_request == 1) {
    trash_t = MPI_Wtime();
    ierr = CPLM_MatDenseKernelMatDotProd(AV,Z,beta);
    ecg->gemm_t += MPI_Wtime() - trash_t;
    trash_t = MPI_Wtime();
    ierr = MPI_Allreduce(MPI_IN_PLACE, beta->val, beta->info.nval,
                         MPI_DOUBLE, MPI_SUM, comm);
    ecg->comm_t += MPI_Wtime() - trash_t;
    trash_t = MPI_Wtime();
    ierr = CPLM_MatDenseKernelMatMult(V,'N',beta,'N',Z,-1.E0,1.E0);
    ecg->gemm_t += MPI_Wtime() - trash_t;
    // Swapping time
    trash_t = MPI_Wtime();
    mkl_domatcopy('C','N',m,t,1.E0,V->val,m,V->val+m*nrhs,m);
    mkl_domatcopy('C','N',m,t,1.E0,AV->val,m,AV->val+m*nrhs,m);
    mkl_domatcopy('C','N',m,t,1.E0,Z->val,m,V->val,m);
    ecg->copy_t += MPI_Wtime() - trash_t;
    // Now we need A*P to continue
    *rci_request = 0;
  }
CPLM_POP
  return ierr;
}

int _preAlps_ECGIterateOdirFused(preAlps_ECG_t* ecg, int* rci_request) {
CPLM_PUSH
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
  CPLM_Mat_Dense_t mu_s   = CPLM_MatDenseNULL();
  CPLM_Mat_Dense_t rtr_s  = CPLM_MatDenseNULL();
  int m = ecg->locPbSize, M = ecg->globPbSize, nrhs = ecg->enlFac;
  int t = P->info.n, t1 = 0; // Reduced size
  double tol = ecg->tol*ecg->normb/sqrt(nrhs), trash_t;
  int rank; MPI_Comm_rank(comm,&rank);
  ierr = CPLM_MatDenseSetInfo(&mu_s,t,t,t,t,COL_MAJOR);
  ierr = CPLM_MatDenseSetInfo(&rtr_s,nrhs,nrhs,nrhs,nrhs,COL_MAJOR);
  mu_s.val  = alpha->val + 3*nrhs*nrhs;
  rtr_s.val = alpha->val + 4*nrhs*nrhs;
  trash_t = MPI_Wtime();
  ierr = CPLM_MatDenseKernelMatDotProd(P, R, alpha);
  ierr = CPLM_MatDenseKernelMatDotProd(AV, Z, beta);
  ierr = CPLM_MatDenseKernelMatDotProd(AP, P, &mu_s);
  ierr = CPLM_MatDenseKernelMatDotProd(R, R, &rtr_s);
  ecg->gemm_t += MPI_Wtime() - trash_t;
  trash_t = MPI_Wtime();
  ierr = MPI_Allreduce(MPI_IN_PLACE, alpha->val, 5*nrhs*nrhs,
                       MPI_DOUBLE, MPI_SUM, comm);
  ecg->comm_t += MPI_Wtime() - trash_t;
  ecg->res = 0.E0;
  for (int i = 0; i < nrhs; ++i)
    ecg->res += rtr_s.val[i + rtr_s.info.lda*i];
  ecg->res = sqrt(ecg->res);

  if (ecg->res < ecg->tol*ecg->normb || ecg->iter > ecg->maxIter)
    *rci_request = 1; // The method has converged
  else
    *rci_request = 0; // We need to continue

  trash_t = MPI_Wtime();
  ierr = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', t, mu_s.val, t);
  ecg->potrf_t += MPI_Wtime() - trash_t;
  trash_t = MPI_Wtime();
  ierr = CPLM_MatDenseKernelUpperTriangularRightSolve(&mu_s, P);
  ierr = CPLM_MatDenseKernelUpperTriangularRightSolve(&mu_s, AP);
  ierr = CPLM_MatDenseKernelUpperTriangularRightSolve(&mu_s, beta);
  ierr = CPLM_MatDenseKernelUpperTriangularRightSolve(&mu_s, Z);
  cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
              t, nrhs, 1.E0, mu_s.val, mu_s.info.lda, alpha->val, t);
  cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
              t, t, 1.E0, mu_s.val, mu_s.info.lda, beta->val, ecg->kbs);
  ecg->trsm_t += MPI_Wtime() - trash_t;
  trash_t = MPI_Wtime();
  ierr = CPLM_MatDenseKernelMatMult(V,'N',beta,'N',Z,-1.E0,1.E0);
  ecg->gemm_t += MPI_Wtime() - trash_t;
  /**************** Reduction of the search directions **********************/
  if (ecg->bs_red == ADAPT_BS) {
    double* tau_p  = alpha->val + 5*nrhs*nrhs; // Householder
    trash_t = MPI_Wtime();
    memcpy(mu_s.val,alpha->val,sizeof(double)*alpha->info.nval);
    ecg->copy_t += MPI_Wtime() - trash_t;
    // 1) SVD on alpha
    trash_t = MPI_Wtime();
    ierr = LAPACKE_dgesvd(LAPACK_COL_MAJOR,'O','N',t,nrhs,mu_s.val,
                          alpha->info.lda,tau_p,NULL,1,NULL,1,tau_p+t);
    ecg->gesvd_t += MPI_Wtime() - trash_t;
    for (int i = 0; i < t; i++) {
      if (tau_p[i] > tol) t1++;
      else break;
    }

    // 2) reduction of the search directions
    if (t1 > 0 && t1 < nrhs && t1 < t) {
      // For in-place update
      trash_t = MPI_Wtime();
      ierr = LAPACKE_dgeqrf(LAPACK_COL_MAJOR,t,t,mu_s.val,t,tau_p);
      ecg->geqrf_t += MPI_Wtime() - trash_t;
      // Update alpha, P, AP
      trash_t = MPI_Wtime();
      LAPACKE_dormqr(LAPACK_COL_MAJOR,'L','T',t,nrhs,t,mu_s.val,
                    t,tau_p,alpha->val,t);
      LAPACKE_dormqr(LAPACK_COL_MAJOR,'R','N',m,t,t,mu_s.val,
                    t,tau_p,P->val,m);
      LAPACKE_dormqr(LAPACK_COL_MAJOR,'R','N',m,t,t,mu_s.val,
                    t,tau_p,AP->val,m);
      LAPACKE_dormqr(LAPACK_COL_MAJOR,'R','N',m,t,t,mu_s.val,
                    t,tau_p,Z->val,m);
      ecg->ormqr_t += MPI_Wtime() - trash_t;
      // Reduce sizes
      trash_t = MPI_Wtime();
      mkl_dimatcopy('C','N',t,nrhs,1.E0,alpha->val,t,t1);
      ecg->copy_t += MPI_Wtime() - trash_t;
      CPLM_MatDenseSetInfo(alpha,t1,nrhs,t1,nrhs,COL_MAJOR);
      CPLM_MatDenseSetInfo(P ,M,t1,m,t1,COL_MAJOR);
      CPLM_MatDenseSetInfo(AP,M,t1,m,t1,COL_MAJOR);
      // Update the other variables
      CPLM_MatDenseSetInfo(Z, M, t1, m, t1, COL_MAJOR);
      // Update sizes
      ecg->bs   = t1;
      ecg->kbs  = t + nrhs;
    }
    CPLM_MatDenseSetInfo(beta, ecg->kbs, t1, ecg->kbs, t1, COL_MAJOR);
    CPLM_MatDenseSetInfo(V, M, ecg->kbs, m, ecg->kbs, COL_MAJOR);
    CPLM_MatDenseSetInfo(AV, M, ecg->kbs, m, ecg->kbs, COL_MAJOR);
  }
  /**************************************************************************/

  ierr = CPLM_MatDenseKernelMatMult(P,'N',alpha,'N',X,1.E0,1.E0);
  ierr = CPLM_MatDenseKernelMatMult(AP,'N',alpha,'N',R,-1.E0,1.E0);
  // Iteration finished
  ecg->iter++;

  // Swapping time
  trash_t = MPI_Wtime();
  mkl_domatcopy('C','N',m,ecg->bs,1.E0,V->val,m,V->val+m*nrhs,m);
  mkl_domatcopy('C','N',m,ecg->bs,1.E0,AV->val,m,AV->val+m*nrhs,m);
  mkl_domatcopy('C','N',m,ecg->bs,1.E0,Z->val,m,V->val,m);
  ecg->copy_t += MPI_Wtime() - trash_t;
CPLM_CLOSE_TIMER
CPLM_POP
  return ierr;
}

int preAlps_ECGFinalize(preAlps_ECG_t* ecg, double* solution) {
CPLM_PUSH
  int ierr = _preAlps_ECGWrapUp(ecg, solution);
  _preAlps_ECGFree(ecg);
CPLM_POP
  return ierr;
}

int _preAlps_ECGWrapUp(preAlps_ECG_t* ecg, double* solution) {
CPLM_PUSH
  int ierr = 0;
  // Simplify notations
  CPLM_Mat_Dense_t* X = ecg->X;
  // Get the solution
  ierr = CPLM_MatDenseKernelSumColumns(X, solution);CPLM_CHKERR(ierr);
CPLM_POP
  return ierr;
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
  mkl_free(ecg->work);mkl_free(ecg->iwork);
CPLM_POP
}

void preAlps_ECGPrint(preAlps_ECG_t* ecg, int verbosity) {
CPLM_PUSH
  int rank;
  MPI_Comm_rank(ecg->comm,&rank);
  printf("[%d] prints ECG_t...\n", rank);
  printf("=== Summary ===\n");
  printf("\titer: %d\n\tres : %e\n\tbs  : %1d\n",ecg->iter,ecg->res,ecg->bs);
  printf("=== Timings ===\n");
  printf("\ttot_t  : %e s\n", ecg->tot_t);
  printf("\tcomm_t : %e s\n", ecg->comm_t);
  printf("\ttrsm_t : %e s\n", ecg->trsm_t);
  printf("\tgemm_t : %e s\n", ecg->gemm_t);
  printf("\tpotrf_t: %e s\n",ecg->potrf_t);
  printf("\tpstrf_t: %e s\n",ecg->pstrf_t);
  printf("\tlapmt_t: %e s\n",ecg->lapmt_t);
  printf("\tgesvd_t: %e s\n",ecg->gesvd_t);
  printf("\tgeqrf_t: %e s\n",ecg->geqrf_t);
  printf("\tormqr_t: %e s\n", ecg->ormqr_t);
  printf("\tcopy_t : %e s\n", ecg->copy_t);
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
/******************************************************************************/
