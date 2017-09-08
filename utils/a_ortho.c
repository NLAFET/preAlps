/******************************************************************************/
/* Author     : Olivier Tissot                                                */
/* Creation   : 2016/06/24                                                    */
/* Description: A-orthonormalization methods                                  */
/******************************************************************************/

/******************************************************************************/
/*                                  INCLUDE                                   */
/******************************************************************************/
#include "a_ortho.h"
/******************************************************************************/

/******************************************************************************/
/*                                    CODE                                    */
/******************************************************************************/

/**
 * \fn int A_CholQR(Mat_Dense_t* P,
 *                    Mat_CSR_t* AP);
 */
int A_CholQR(Mat_Dense_t* P, Mat_Dense_t* AP) {
PUSH
BEGIN_TIME
  int ierr;
  Mat_Dense_t C = MatDenseNULL();
  // C = P^t*AP
  ierr = MatDenseMatDotProd(AP, P, &C, MPI_COMM_WORLD);

  if (PRINT >= 10) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
      MatDensePrintfInfo("A_CholQR::AP init infos",AP);
      MatDensePrintf2D("A_CholQR::AP init",AP);
      MatDensePrintfInfo("A_CholQR::PAP0 init infos",&C);
      MatDensePrintf2D("A_CholQR::PAP0 init",&C);
    }
  }

  // Cholesky of C: R^tR = C
  ierr = Cholesky(&C);
  // Solve triangular right system for P
  ierr = UpperTriangularRightSolve(&C, P);
  // Solve triangular right system for AP
  ierr = UpperTriangularRightSolve(&C, AP);

  // A-normalize P and AP
  ierr = A_Normalize(P,AP);

  if (PRINT >= 10) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Mat_Dense_t PAP = MatDenseNULL(), APl = MatDenseNULL();
    BlockOperator(P,&APl);
    MatDenseMatDotProd(&APl,P,&PAP,MPI_COMM_WORLD);
    if (rank == 0) {
      MatDensePrintfInfo("A_CholQR::Check PAP0",&PAP);
      MatDensePrintf2D("A_CholQR::Check PAP0",&PAP);
    }
    MatDenseFree(&PAP);
    MatDenseFree(&APl);
  }

  MatDenseFree(&C);
END_TIME
POP
  return ierr;
}

int Cholesky(Mat_Dense_t* C) {
PUSH
BEGIN_TIME
  int ierr = 0;
  int matrix_layout = (C->info.stor_type == ROW_MAJOR) ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR;
  char uplo = 'U';
  int order = C->info.n;
  int lda = C->info.n;
  ierr = LAPACKE_dpotrf(matrix_layout,
                        uplo,
                        order,
                        C->val,
                        lda);
  if (ierr > 0) {
    eprintf("Cholesky::The matrix A is not spd!\n");
    MPI_Abort(MPI_COMM_WORLD, ierr);
  }
END_TIME
POP
  return ierr;
}

int LowerTriangularLeftSolve(Mat_Dense_t* L, Mat_Dense_t* B) {
PUSH
BEGIN_TIME
  int ierr = 0;
  int matrix_layout = (L->info.stor_type == ROW_MAJOR) ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR;
  char uplo  = 'L'; // L is lower triangular
  char trans = 'N'; // No transpose, no conjugacy
  char diag  = 'N'; // No unitary
  int m      = B->info.m;
  int nrhs   = B->info.n;
  int lda    = (L->info.stor_type == ROW_MAJOR) ? L->info.n : L->info.m;
  int ldb    = (B->info.stor_type == ROW_MAJOR) ? B->info.n : B->info.m;
  ierr = LAPACKE_dtrtrs(matrix_layout,
                        uplo,
                        trans,
                        diag,
                        m,
                        nrhs,
                        L->val,
                        lda,
                        B->val,
                        ldb);
END_TIME
POP
  return ierr;
}

int UpperTriangularRightSolve(Mat_Dense_t* R, Mat_Dense_t* B) {
PUSH
BEGIN_TIME
  int ierr = 0;
  CBLAS_LAYOUT matrix_layout = (R->info.stor_type == ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
  MKL_INT m    = B->info.m;
  MKL_INT nrhs = B->info.n;
  double alpha = 1e0;
  MKL_INT lda  = (R->info.stor_type == ROW_MAJOR) ? R->info.n : R->info.m;
  MKL_INT ldb  = (B->info.stor_type == ROW_MAJOR) ? B->info.n : B->info.m;
  cblas_dtrsm(matrix_layout,
              CblasRight,
              CblasUpper,
              CblasNoTrans,
              CblasNonUnit,
              m,
              nrhs,
              alpha,
              R->val,
              lda,
              B->val,
              ldb);
END_TIME
POP
  return ierr;
}

/* TODO Optimize this part */
int A_Normalize(Mat_Dense_t* P, Mat_Dense_t* AP) {
PUSH
BEGIN_TIME
  int ierr = 0;
  Mat_Dense_t PAP = MatDenseNULL();
  // // AP = A*P
  // ierr = BlockOperator(P, AP);
  // PAP = P^t*A*P
  ierr = MatDenseMatDotProd(AP, P, &PAP, MPI_COMM_WORLD);

  DVector_t a_norms = DVectorNULL();
  DVectorCalloc(&a_norms, PAP.info.n);
  // Calculate A-norm squared
  // for (int j = 0; j < a_norms.nv; ++j)
  //   a_norms.v[j] = PAP.val[j*(PAP.info.lda) + j];
  //
  // if (PRINT >= 15) {
  //   int rank;
  //   MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  //   if (rank == 0) {
  //   MatDensePrintf2D("A_Normalize::PAP0",&PAP);
  //     for (int j = 0; j < a_norms.nv; ++j)
  //       printf("A_Normalize::a_norm[%d] = %.14f\n",j,sqrt(PAP.val[j*(PAP.info.lda) + j]));
  //   }
  // }

  // Renormalize P
  int loop_index_1, loop_index_2;
  if (P->info.stor_type == ROW_MAJOR) {
    loop_index_1 = P->info.n;
    loop_index_2 = 1;
  }
  else {
    loop_index_1 = 1;
    loop_index_2 = P->info.m;
  }
  for (int j = 0; j < PAP.info.n; ++j) {
    PAP.val[j*(PAP.info.lda) + j] = sqrt(PAP.val[j*(PAP.info.lda) + j]);
    for (int i = 0; i < P->info.m; ++i) {
      P->val[i*loop_index_1 + j*loop_index_2]  /= PAP.val[j*(PAP.info.lda) + j];
      AP->val[i*loop_index_1 + j*loop_index_2] /= PAP.val[j*(PAP.info.lda) + j];
    }
  }

  DVectorFree(&a_norms);
  MatDenseFree(&PAP);
END_TIME
POP
  return ierr;
}

int A_OrthoTest(int rank) {
PUSH
BEGIN_TIME
  int ierr = 0;

  if (rank == 0) {
    printf("::: Testing A_Ortho :::\n");
    Mat_Dense_t A = MatDenseNULL();
    MatDenseSetInfo(&A, 4, 4, 4, 4, ROW_MAJOR);
    MatDenseConstant(&A, 1.0);
    for (int i = 0; i < 4; ++i)
      A.val[i+4*i] = 4.0;
    MatDensePrint2D(&A);
    ierr = Cholesky(&A);
    if (ierr != 0)
      printf("  Cholesky::Error!\n");
    else
      printf("  Cholesky::Passed!\n");
    MatDensePrint2D(&A);
    Mat_Dense_t B = MatDenseNULL();
    MatDenseSetInfo(&B, 4, 2, 4, 2, ROW_MAJOR);
    MatDenseConstant(&B, 1.0);
    printf("\n");
    MatDensePrint2D(&B);
    ierr = LowerTriangularLeftSolve(&A, &B);
    if (ierr != 0)
      printf("  LowerTriangularLeftSolve::Error!\n");
    else
      printf("  LowerTriangularLeftSolve::Passed!\n");
    MatDensePrint2D(&B);
    MatDenseSetInfo(&B, 4, 4, 4, 4, ROW_MAJOR);
    MatDenseIdentity(&B);

    MatDenseCalloc(&A);
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j <= i; ++j)
        A.val[i*A.info.n + j] = 1;
    printf("\n");
    MatDensePrint2D(&A);
    MatDensePrint2D(&B);
    ierr = UpperTriangularRightSolve(&A, &B);
    if (ierr != 0)
      printf("  UpperTriangularRightSolve::Error!\n");
    else
      printf("  UpperTriangularRightSolve::Passed!\n");
    MatDensePrint2D(&B);
    MatDenseFree(&B);
    MatDenseFree(&A);
  }

  // A_CholQR test
  int size, M, m;
  Mat_Dense_t matRhs = MatDenseNULL(), AmatRhs = MatDenseNULL();
  OperatorGetSizes(&M,&m);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  ierr = MatDenseSetInfo(&matRhs, M, size, m, size, ROW_MAJOR);
  ierr = MatDenseCalloc(&matRhs);
  printf("  [%d] A_CholQR::Construction of P...\n", rank);
  if (rank%2 == 0)
    matRhs.val[0] = 1.0;
  else
    matRhs.val[1] = 1.0;
  printf("  [%d] A_CholQR::P is constructed!\n", rank);
  printf("  [%d] A_CholQR::A-orthonormalization of P...\n", rank);
  ierr = A_CholQR(&matRhs,&AmatRhs);
  if (ierr != 0)
    printf("  [%d] A_CholQR::Error!\n", rank);
  else
    printf("  [%d] A_CholQR::Passed!\n", rank);
  printf("  [%d] A_CholQR::Solution obtained:\n", rank);
  MatDensePrint2D(&matRhs);
  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0)
    printf("::: End testing A_Ortho :::\n");
END_TIME
POP
  return ierr;
}

/******************************************************************************/
