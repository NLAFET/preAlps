/*
* This file contains functions used for interfacing with Petsc
*
* Authors : Sebastien Cayrols
*         : Olivier Tissot
* Email   : sebastien.cayrols@[(gmail.com) | (inria.fr)]
*         : olivier.tissot@inria.fr
*/
#include <petscdraw.h>
#include <petscviewer.h>
#include <petscsys.h>
#include <petscmat.h>

#include <preAlps_cplm_timing.h>
#include <preAlps_cplm_petsc_interface.h>
#include <preAlps_cplm_utils.h>


int CPLM_petscCreateSeqMatFromMatCSR(CPLM_Mat_CSR_t *m, Mat *M)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  int ierr = 0;

  ierr = MatCreate(PETSC_COMM_SELF, M);CHKERRQ(ierr);
  ierr = MatSetType(*M, MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatSetSizes(*M,
              m->info.m,
              m->info.m,
              m->info.m,
              m->info.n);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocationCSR(*M, m->rowPtr, m->colInd, m->val);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*M, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*M, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

CPLM_END_TIME
CPLM_POP
  return ierr;
}





int CPLM_petscCreateMatFromMatCSR(CPLM_Mat_CSR_t *m, Mat *M)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  int ierr = 0;

  ierr = MatCreate(PETSC_COMM_WORLD,M);CPLM_CHKERR(ierr);
  ierr = MatSetType(*M,MATMPIAIJ);CPLM_CHKERR(ierr);
  // ! Petsc needs the size of the local DIAGONAL block !
  ierr = MatSetSizes(*M,
                     m->info.m,
                     m->info.m,
                     m->info.M,
                     m->info.N);CPLM_CHKERR(ierr);
  ierr = MatMPIAIJSetPreallocationCSR(*M,
                                      m->rowPtr,
                                      m->colInd,
                                      m->val);CPLM_CHKERR(ierr);
  ierr = MatAssemblyBegin(*M,MAT_FINAL_ASSEMBLY);CPLM_CHKERR(ierr);
  ierr = MatAssemblyEnd(*M,MAT_FINAL_ASSEMBLY);CPLM_CHKERR(ierr);

CPLM_END_TIME
CPLM_POP
  return ierr;
}





// Create a Petsc Mat dense which shares val with the original MatDense
int CPLM_petscCreateMatFromMatDense(CPLM_Mat_Dense_t *m, Mat *M)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  int ierr = 0;

  ierr = MatCreateDense(PETSC_COMM_WORLD,
                        m->info.m,
                        PETSC_DECIDE,
                        m->info.M,
                        m->info.N,
                        m->val,
                        M);CPLM_CHKERR(ierr);
  ierr = MatAssemblyBegin(*M,MAT_FINAL_ASSEMBLY);CPLM_CHKERR(ierr);
  ierr = MatAssemblyEnd(*M,MAT_FINAL_ASSEMBLY);CPLM_CHKERR(ierr);

CPLM_END_TIME
CPLM_POP
  return ierr;
}





//Remark : This function does not take into account the fill-in. So It is not optimized for k>0
PetscErrorCode petscGetILUFactor(Mat *M, PetscReal k, Mat *F)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  int ierr = 0;
  MatFactorInfo  info;
  IS             row,col;

  ierr = MatGetFactor(*M,MATSOLVERPETSC,MAT_FACTOR_ILU,F);CHKERRQ(ierr);

  //reorder A matrix following one-Way Dissection
  ierr = MatGetOrdering(*M,MATORDERINGNATURAL,&row,&col);CHKERRQ(ierr);

  ierr = MatFactorInfoInitialize(&info);CHKERRQ(ierr);

  info.diagonal_fill = 1.0;
  info.fill          = 1.0;
  info.levels        = k;
/*  info.zeropivot     = 0.0;*/

  ierr = MatILUFactorSymbolic(*F,*M,row,col,&info);CHKERRQ(ierr);
  ierr = MatLUFactorNumeric(*F,*M,&info);CHKERRQ(ierr);

  ISDestroy(&row);
  ISDestroy(&col);

CPLM_END_TIME
CPLM_POP
  return ierr;
}





/**
 * \fn PetscErrorCode MatCSRGetFromPetsc(Mat mat, CPLM_Mat_CSR_t *m)
 * \brief Function returns a CSR matrix from a Petsc matrix
 * \param A_in   The Petsc matrix
 * \param B_out  The CSR matrix returned
 */
/*Function which return a CPLM_Mat_CSR_t matrix and takes in parameter a Petsc Mat matrix */
PetscErrorCode petscCreateMatCSR(Mat A_in, CPLM_Mat_CSR_t *B_out)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  PetscInt M      = 0;
  PetscInt N      = 0;
  PetscInt nnz    = 0;
  PetscInt m      = 0;
  PetscInt n      = 0;
  PetscInt lnnz   = 0;
  PetscInt rstart = 0;
  PetscInt rend   = 0;
  PetscErrorCode ierr = 0;
  MatInfo info;
  PetscInt nbCol_tmp = 0;
  const PetscInt *colInd_tmp = NULL;
  const PetscScalar *values_tmp = NULL;
  int ind  = 0;
  int lrow = 0;

  ierr = MatGetSize(A_in,&M,&N);
  ierr = MatGetLocalSize(A_in,&m,&n);
  ierr = MatGetInfo(A_in,MAT_GLOBAL_SUM,&info);
  nnz  = (int)info.nz_allocated;
  ierr = MatGetInfo(A_in,MAT_LOCAL,&info);
  lnnz = (int)info.nz_allocated;
  ierr = CPLM_MatCSRSetInfo(B_out,M,N,nnz,m,N,lnnz,1);CPLM_CHKERR(ierr);
  ierr = CPLM_MatCSRMalloc(B_out);CPLM_CHKERR(ierr);

  B_out->rowPtr[0]=0;
  ierr = MatGetOwnershipRange(A_in,&rstart,&rend);
  //Get a CSR matrix from a Petsc matrix format
  for(PetscInt row = rstart; row < rend; row++,lrow++)
  {

    ierr = MatGetRow(A_in,row,&nbCol_tmp,&colInd_tmp,&values_tmp);
    B_out->rowPtr[lrow+1] = B_out->rowPtr[lrow] + nbCol_tmp;
    for(int i = 0; i < nbCol_tmp; i++)
    {
      B_out->colInd[ind]  = (int)colInd_tmp[i];
      B_out->val[ind++]   = values_tmp[i];
    }
    ierr = MatRestoreRow(A_in,row,&nbCol_tmp,&colInd_tmp,&values_tmp);
  }

CPLM_END_TIME
CPLM_POP
  return ierr;
}





/**
 * \fn PetscErrorCode MatCSRPetscPrintMatrix(CPLM_Mat_CSR_t *m, const char *name)
 * \brief Method which prints a graphical representation of a CSR matrix
 * \brief Note : This method is enable only if PRINT_MATRIX is defined
 * \param *matCSR The CSR matrix
 * \param *name The name of the matrix
 */
/*Function prints a matrix into graphical window*/
PetscErrorCode petscPrintMatCSR(const char *name, CPLM_Mat_CSR_t *A_in)
{
  PetscErrorCode ierr = 0;
  PetscViewer viewer;
  PetscDraw draw;
  Mat B;

  //Declare the permuted matrix B
  ierr = MatCreate(PETSC_COMM_SELF, &B);CHKERRQ(ierr);
  //Set the type of B matrix
  ierr = MatSetType(B, MATMPIAIJ);CHKERRQ(ierr);
  //Set the size of B matrix
  ierr = MatSetSizes(B, A_in->info.m, A_in->info.n, A_in->info.M, A_in->info.N);CHKERRQ(ierr);
  //convert a CSR matrix into Petsc matrix format
  ierr = MatMPIAIJSetPreallocationCSR(B, A_in->rowPtr, A_in->colInd, A_in->val);CHKERRQ(ierr);

  MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscViewerDrawOpen(PETSC_COMM_SELF,0,name,PETSC_DECIDE,PETSC_DECIDE,
                            PETSC_DRAW_HALF_SIZE,PETSC_DRAW_HALF_SIZE,
                            &viewer);CHKERRQ(ierr);

  ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetPause(draw,-1);CHKERRQ(ierr);
  ierr = MatView(B,viewer);CHKERRQ(ierr);
  ierr = PetscDrawPause(draw);CHKERRQ(ierr);

  PetscDrawDestroy(&draw);CHKERRQ(ierr);
/*  PetscViewerDestroy(&viewer);CHKERRQ(ierr);//[TODO]*/
  MatDestroy(&B);CHKERRQ(ierr);

  return ierr;
}





PetscErrorCode petscMatCSRMatDenseMult(Mat*         A_in,
                                       CPLM_Mat_Dense_t* B_in,
                                       CPLM_Mat_Dense_t* C_out)
{
CPLM_PUSH
CPLM_BEGIN_TIME
  int ierr = 0;
  CPLM_ASSERT(A_in  != NULL);
  CPLM_ASSERT(B_in  != NULL);
  CPLM_ASSERT(C_out != NULL);
  // First create Petsc objects
  Mat B, C;
  ierr = CPLM_petscCreateMatFromMatDense(B_in, &B);CPLM_CHKERR(ierr);
  ierr = CPLM_petscCreateMatFromMatDense(C_out, &C);CPLM_CHKERR(ierr);
  // Use MatMatMult
  ierr = MatMatMult(*A_in, B, MAT_REUSE_MATRIX, PETSC_DEFAULT, &C);CPLM_CHKERR(ierr);
CPLM_END_TIME
CPLM_POP
  return ierr;
}





PetscErrorCode petscMatGetScaling(Mat A, Vec scale)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  PetscErrorCode ierr;
  Vec max;
  PetscReal threshold = 0.1;

  ierr = VecDuplicate(scale, &max);CHKERRQ(ierr);

  ierr = VecSet(scale, 1.0);CHKERRQ(ierr);

  ierr = MatGetRowMaxAbs(A,max,NULL);CHKERRQ(ierr);

  ierr = VecSqrtAbs(max);CHKERRQ(ierr);

  ierr = VecPointwiseDivide(scale, scale, max);CHKERRQ(ierr);

  VecDestroy(&max);CHKERRQ(ierr);

CPLM_END_TIME
CPLM_POP
  return ierr;
}





PetscErrorCode petscMatLoad(Mat *A, const char *fileName, MPI_Comm comm)
{
CPLM_PUSH

  PetscErrorCode  ierr = 0;
  PetscInt  M     = 0;
  PetscInt  N     = 0;
  PetscInt  m     = 0;
  PetscInt  n     = 0;
  PetscViewer     fd;
  int rank = 0;

  PetscViewerBinaryOpen(comm, fileName, FILE_MODE_READ, &fd);

  ierr = MatCreate(comm, A);CHKERRQ(ierr);
  if(comm == PETSC_COMM_SELF)
  {
    ierr = MatSetType(*A,MATSEQAIJ);CHKERRQ(ierr);
  }
  else
  {
    ierr = MatSetType(*A,MATMPIAIJ);CHKERRQ(ierr);
  }

  ierr = MatLoad(*A, fd); CHKERRQ(ierr);

  ierr = MatGetSize(*A, &M, &N); CHKERRQ(ierr);
  ierr = MatGetLocalSize(*A, &m, &n); CHKERRQ(ierr);

  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (rank == 0)
    printf("{%s} Matrix size %d x %d : local %d x %d\n", __FUNCTION__, M, N, m, n);

CPLM_POP
  return ierr;
}





int CPLM_petscCreateSeqMatDenseFromMatCSR(CPLM_Mat_CSR_t *m, Mat *M)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  int ierr = 0;
  double *val=NULL;
  size_t nval = 0;
  int cpt = 0;
  MPI_Comm comm;

  comm = PETSC_COMM_SELF;

  nval = m->info.m * m->info.m;
  val = calloc(nval, sizeof(double));
  CPLM_ASSERT(val != NULL);

  for(int i = 0; i < m->info.m; i++)
  {
    for(int j = m->rowPtr[i]; j < m->rowPtr[i + 1]; j++)
    {
      val[i + m->colInd[j] * m->info.m] = m->val[cpt++];
    }
  }

  ierr = MatCreate(comm, M);CHKERRQ(ierr);
  ierr = MatSetSizes(*M,
      m->info.m,
      m->info.n,
      m->info.m,
      m->info.n);CHKERRQ(ierr);
  ierr = MatSetType(*M, MATSEQDENSE);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(*M, val);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*M, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*M, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

CPLM_END_TIME
CPLM_POP
  return ierr;
}





//Packed variable is a flag to get after the L and U factors. To do it, PETSC needs to
//  allocate the factor in dense. Thus packed=1 allocates so much more i.e. m*m
PetscErrorCode petscLUFromMatCSR( CPLM_Mat_CSR_t   *m,
                                  Mat         *F,
                                  int         packed,
                                  const char  *solverPackage,
                                  MatFactorType typeLU,
                                  float tau,
                                  float k,
                                  float fillEstimator,
                                  float zeroPivot)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  PetscErrorCode ierr = 0;
  Mat CC;
  MatFactorInfo info;
  MatInfo info_F;
  MatInfo info_C;
  IS  row;
  IS  col;

  if(packed)
  {
    ierr = CPLM_petscCreateSeqMatDenseFromMatCSR(m, &CC);CPLM_CHKERR(ierr);
  }
  else
  {
    ierr = CPLM_petscCreateSeqMatFromMatCSR(m, &CC);CPLM_CHKERR(ierr);
  }

  ierr = petscLUFactorization(CC,
      F,
      solverPackage,
      typeLU,
      tau,
      k,
      fillEstimator,
      zeroPivot); CPLM_CHKERR(ierr);

CPLM_END_TIME
CPLM_POP
  PetscFunctionReturn(0);
}





PetscErrorCode petscLUFactorization(Mat A,
                                    Mat *F,
                                    const char *solverPackage,
                                    MatFactorType typeLU,
                                    float tau,
                                    float k,
                                    float fillEstimator,
                                    float zeroPivot)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  PetscErrorCode ierr = 0;
  MatFactorInfo info;
  MatInfo info_A;
  MatInfo info_F;
  IS  row;
  IS  col;
  PetscInt ma = 0;
  PetscInt na = 0;
  PetscInt mf = 0;
  PetscInt nf = 0;

  ierr = MatGetFactor(A, solverPackage, typeLU, F);CHKERRQ(ierr);

  //MATORDERINGNATURAL MATORDERINGND MATORDERING1WD MATORDERINGRCM MATORDERINGQMD
  ierr = MatGetOrdering(A, MATORDERINGND, &row, &col);CHKERRQ(ierr);

  ierr = MatFactorInfoInitialize(&info);CHKERRQ(ierr);

  info.diagonal_fill  = 1.0;
  info.usedt          = (tau > 0);
  info.dt             = (tau < 0) ? 0.0 : tau;
  info.fill           = (fillEstimator < 0) ? 5.0 : fillEstimator;
  info.levels         = k;
  info.zeropivot      = (zeroPivot < 0) ? 0.0 : zeroPivot;

  if(typeLU == MAT_FACTOR_LU)
  {
    ierr = MatLUFactorSymbolic(*F, A, row, col, &info);CHKERRQ(ierr);
  }
  else
  {
    ierr = MatILUFactorSymbolic(*F, A, row, col, &info);CHKERRQ(ierr);
  }
  ierr = MatLUFactorNumeric(*F, A, &info);CHKERRQ(ierr);

  ierr = MatGetInfo(*F, MAT_LOCAL, &info_F);CHKERRQ(ierr);
  ierr = MatGetInfo(A,  MAT_LOCAL, &info_A);CHKERRQ(ierr);
  ierr = MatGetSize(A, &ma, &na);CHKERRQ(ierr);
  ierr = MatGetSize(*F, &mf, &nf);CHKERRQ(ierr);
  printf("{%s}A\n\tsize : %d %d\n\tnnzA : %.0e\n\tF\n\tsize : %d %d\n\tnnzF : %.0e\n\t ratio_nnzF/nnzA : %.2f\n",
    __FUNCTION__,
    ma,
    na,
    info_A.nz_used,
    mf,
    nf,
    info_F.nz_used,
    info_F.nz_used/info_A.nz_used);

  ierr = ISDestroy(&row);CHKERRQ(ierr);
  ierr = ISDestroy(&col);CHKERRQ(ierr);

CPLM_END_TIME
CPLM_POP
  PetscFunctionReturn(0);
}




/*
PetscErrorCode petscConvertFactorToMatCSR(Mat *F, CPLM_Mat_CSR_t *L, CPLM_Mat_CSR_t *U)
{
CPLM_PUSH

  CPLM_Mat_CSR_t A = CPLM_MatCSRNULL();
  MatType type;
  PetscErrorCode ierr = 0;
  CPLM_IVector_t diag = CPLM_IVectorNULL();

  ierr = MatGetType(*F, &type);CHKERRQ(ierr);

  if(type != MATDENSE)
  {
    CPLM_Abort("Can not unfactored if the factor is not a MatDense");
  }

  ierr = MatSetUnfactored(*F);CHKERRQ(ierr);

  ierr = petscCreateMatCSR(*F, &A);CPLM_CHKERR(ierr);

  ierr = CPLM_MatCSRGetDiagInd(&A, &diag);CPLM_CHKERR(ierr);

  ierr = CPLM_MatCSRGetLUFactors(&A, L, U, &diag);CPLM_CHKERR(ierr);

  CPLM_MatCSRFree(&A);
  CPLM_IVectorFree(&diag);

CPLM_POP
  return ierr;
}
*/
