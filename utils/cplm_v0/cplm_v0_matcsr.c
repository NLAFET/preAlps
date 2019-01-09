/*
============================================================================
Name        : cplm_matcsr.h
Author      : Simplice Donfack, Sebastien Cayrols
Version     : 0.1
Description : Functions of preAlps which will be part of MatCSR.
Date        : Oct 13, 2017
============================================================================
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifdef USE_MKL
#include <mkl.h>
#endif
#include "cplm_utils.h"
#include "cplm_QS.h"
#include "cplm_v0_ivector.h"
#include "cplm_v0_dvector.h"
#include <cplm_v0_matcsr.h>

int CPLM_MatCSRConvertFromDenseDVector(CPLM_Mat_CSR_t *m_out, CPLM_DVector_t *v_in, int M, int N)
{

  int ierr = 0;
  int nnz=0;
  CPLM_Info_t info;
  info.M=info.m=M;
  info.N=info.n=N;

  for(int i=0;i<v_in->nval;i++)
    //if(v->val[i] > EPSILON )
    if(v_in->val[i] != 0.0 )
      nnz++;

  info.nnz=info.lnnz=nnz;
  info.blockSize=1;
  info.format=FORMAT_CSR;
  info.structure=UNSYMMETRIC;

  CPLM_MatCSRInit(m_out,&info);
  CPLM_MatCSRMalloc(m_out);

  int cpt=0;
  m_out->rowPtr[0]=cpt;
  for(int i=0;i<M;i++)
  {
    for(int j=0;j<N;j++)
      //if(v->val[i*N+j] > EPSILON){
      if(v_in->val[i*N+j] != 0.0 )
      {
        m_out->colInd[cpt]=j;
        m_out->val[cpt++]=v_in->val[i*N+j];
      }
    m_out->rowPtr[i+1]=cpt;
  }

  return ierr;

}

/*Create a matrix from a dense vector of type double, the matrix is stored in column major format*/
 int CPLM_MatCSRConvertFromDenseColumnMajorDVectorPtr(CPLM_Mat_CSR_t *m_out, double *v_in, int M, int N){

  int ierr = 0;
  int nnz=0;

  for(int i=0;i<M*N;i++){
    if(v_in[i] != 0.0 ) nnz++;
  }

  /* Set the matrix infos */
  CPLM_MatCSRSetInfo(m_out, M, N, nnz, M,  N, nnz, 1);

  CPLM_MatCSRMalloc(m_out);

  int count=0;
  m_out->rowPtr[0]=0;
  for(int i=0;i<M;i++) {
    for(int j=0;j<N;j++){
      if(v_in[j*N+i] != 0.0 ) {
        m_out->colInd[count] = j;
        m_out->val[count]  = v_in[j*N+i];
        count++;
      }
    }
    m_out->rowPtr[i+1]=count;
  }

  return ierr;
}

/*Create a matrix from a dense vector of type double*/
int CPLM_MatCSRConvertFromDenseDVectorPtr(CPLM_Mat_CSR_t *m_out, double *v_in, int M, int N){

  int ierr;
  CPLM_DVector_t Work1 = CPLM_DVectorNULL();
  CPLM_DVectorCreateFromPtr(&Work1, M*N, v_in);
  ierr = CPLM_MatCSRConvertFromDenseDVector(m_out, &Work1, M, N);

  return ierr;
}

/**
 * \fn int CPLM_metisKwayOrdering(CPLM_Mat_CSR_t *matCSR, CPLM_IVector_t *perm, int _nbparts, CPLM_IVector_t *interval_vec)
 * \brief Function calls Kway partitionning algorithm and return a new matrix permuted
 * \param *m1 The original matrix
 * \param *perm This vector contains data of permutation
 * \param nbparts Number of partition for Kway
 * \param *posB This vector contains first index of each block
 * \return TODO
 */
int CPLM_metisKwayOrdering(CPLM_Mat_CSR_t *m1,
    CPLM_IVector_t *perm,
    int nblock,
    CPLM_IVector_t *posB)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  CPLM_Mat_CSR_t m2 = CPLM_MatCSRNULL();
  int ierr  = 0;
  int M     = 0;
  idx_t nvtxs   = 0;
  idx_t *parts  = NULL;

  M = m1->info.M;

  // If nothing to do
  if(nblock == 1)
  {
    ierr = CPLM_IVectorMalloc(posB, nblock + 1);CPLM_CHKERR(ierr);

    posB->val[0]  = 0;
    posB->val[1]  = M;//[WARNING] It could be with -1

CPLM_END_TIME
CPLM_POP
    return ierr;
  }

  // The matrix has to be symmetric and without diagonal elements
  if(m1->info.structure==UNSYMMETRIC)
  {
    ierr = CPLM_MatCSRSymStruct(m1, &m2);CPLM_CHKERR(ierr);
	}
  else
  {
	  ierr = CPLM_MatCSRDelDiag(m1, &m2);CPLM_CHKERR(ierr);
	}

  nvtxs = (idx_t)M;
  parts = callKway(&m2,nblock);

  ierr = CPLM_getBlockPosition(M, parts, nblock, posB);CPLM_CHKERR(ierr);
	ierr = CPLM_getIntPermArray(nblock, nvtxs, parts, perm);CPLM_CHKERR(ierr);

  if (parts != NULL)
    free(parts);

  CPLM_MatCSRFree(&m2);

CPLM_END_TIME
CPLM_POP
  return ierr;
}

/*
*
* This function returns colPos which is an index of the begin and end of each block in column point of view.
*
*/
//#define IFCOND
int CPLM_MatCSRGetColBlockPos(CPLM_Mat_CSR_t *m, CPLM_IVector_t *pos, CPLM_IVector_t *colPos)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  int nblock      = pos->nval - 1;
  int ierr        = 0;
  int numBlock    = 0;//By default, we consider the first block
  int newNumBlock = 0;
  int c           = 0;

  ierr = CPLM_IVectorMalloc(colPos, m->info.m * nblock + 1);CPLM_CHKERR(ierr);

  colPos->val[0]             = 0;
  colPos->val[pos->nval - 1] = m->rowPtr[m->info.m];

  for(int i = 0; i < m->info.m; i++)
  {
    numBlock = 0;
    for(int j = m->rowPtr[i]; j < m->rowPtr[i+1]; j++)
    {
      c = m->colInd[j];
#ifdef IFCOND
      if (c >= pos->val[numBlock + 1])
      {
        //Save ending position
        colPos->val[i * nblock + numBlock + 1] = j;
        newNumBlock  = c / nblock;
        for(int k = numBlock + 1; k < newNumBlock; k++)
        {
          //Save starting position
          colPos->val[i * nblock + k + 1] = j;
        }
        numBlock = newNumBlock;
      }
#else
      while(c >= pos->val[numBlock + 1])
      {
        numBlock++;
        colPos->val[i * nblock + numBlock] = j;
      }
#endif
    }
    for(int k = numBlock + 1; k <= nblock; k++)
    {
      colPos->val[i * nblock + k] = m->rowPtr[i + 1];
    }
  }

CPLM_END_TIME
CPLM_POP
  return ierr;
}

//colPos  is the starting index of each block in each row
//m       is the number of rows
//nblock  is the number of blocks
//bi      is the current block
//dep     is the dependency CPLM_IVector_t containing block id dependency of the block bi
int CPLM_MatCSRGetCommDep(CPLM_IVector_t *colPos, int m, int nblock, int bi, CPLM_IVector_t *dep)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  int ierr  = 0;
  int cpt   = 0;
  CPLM_IVector_t tmp = CPLM_IVectorNULL();

  ierr = CPLM_IVectorCalloc(&tmp,nblock);CPLM_CHKERR(ierr);
  ierr = CPLM_IVectorMalloc(dep,nblock);CPLM_CHKERR(ierr);

  for(int i = 0; i < m; i++)
  {
    CPLM_ASSERT(i*nblock+nblock<colPos->nval);
    for(int j = 0; j < nblock; j++)
    {
      tmp.val[j] += colPos->val[i*nblock+j+1] - colPos->val[i*nblock+j];
    }
  }

#ifdef DEBUG
  CPLM_IVectorPrintf("[RAW] Dep vector", &tmp);
#endif

  for(int i = 0; i < nblock; i++)
    if(tmp.val[i] && i != bi)
      dep->val[cpt++] = i;

  if(cpt == 0)
    CPLM_Abort("There is no dependencies between some blocks of A...");

  ierr = CPLM_IVectorRealloc(dep,cpt);CPLM_CHKERR(ierr);

  CPLM_IVectorFree(&tmp);
CPLM_END_TIME
CPLM_POP
  return ierr;

}

/**
 * \fn
 * \brief Function creates a CSR matrix without value which corresponds to a submatrix of the original CSR matrix and this submatrix is a part given by Metis_GraphPartKway
 * Note : this function does not need the matrix values
 * \param *A_in         The original CSR matrix
 * \param *B_out        The original CSR matrix
 * \param *pos          The array containing the begin of each part of the CSR matrix
 * \param numBlock      The number of the part which will be returned
 * \return            0 if succeed
 */
/*function which returns a submatrice at CSR format from an original matrix (in CSR format too) and filtered by parts*/
/*num_parts corresponds to the number which selects rows from original matrix and interval allows to select the specific rows*/
int CPLM_MatCSRGetDiagBlock(CPLM_Mat_CSR_t *A_in, CPLM_Mat_CSR_t *B_out, CPLM_IVector_t *pos, CPLM_IVector_t *colPos, int structure)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  int ierr      = 0;
  int lm        = 0;
  int lnnz      = 0;
	int offset    = 0;//variable which adapts the values in rowPtr for local colInd
  int numBlock  = 0;
  int nblock    = 0;
  CPLM_IVector_t diagPos = CPLM_IVectorNULL();

  MPI_Comm_rank(MPI_COMM_WORLD, &numBlock);
  MPI_Comm_size(MPI_COMM_WORLD, &nblock);

  if(A_in->info.m < pos->val[ pos->nval - 1])//i.e. rowLayout
  {

    CPLM_ASSERT(colPos->val != NULL);

    //Count how many lnnz there will be
    int sum         = 0;
    CPLM_IVector_t rowSize  = CPLM_IVectorNULL();

    ierr = CPLM_IVectorMalloc(&rowSize,A_in->info.m);CPLM_CHKERR(ierr);

    if(structure == SYMMETRIC)
    {
      ierr = CPLM_MatCSRGetDiagIndOfPanel(A_in,&diagPos,pos,colPos,numBlock);CPLM_CHKERR(ierr);
      for(int i = 0; i < A_in->info.m; i++)
      {
        rowSize.val[i]  = colPos->val[i*nblock+numBlock+1] - diagPos.val[i];
        sum            += rowSize.val[i];
      }
    }
    else
    {
      for(int i = 0; i < A_in->info.m; i++)
      {
        rowSize.val[i]  = colPos->val[i*nblock+numBlock+1] - colPos->val[i * nblock + numBlock];
        sum            += rowSize.val[i];
      }
    }
	  /*
    * ====================
    *    copy of arrays
    * =====================
    */
    int ind   = 0;
    int ptr   = 0;
    int nvAdd = 0;

    ierr = CPLM_MatCSRSetInfo(B_out,
                            A_in->info.m,
                            A_in->info.n,
                            A_in->info.lnnz,
                            A_in->info.m,
                            pos->val[numBlock+1]-pos->val[numBlock],
                            sum,
                            1);CPLM_CHKERR(ierr);

    ierr = CPLM_MatCSRMalloc(B_out);CPLM_CHKERR(ierr);

    B_out->rowPtr[0] = 0;
    if(structure == SYMMETRIC)
    {
      for(int i = 0; i < A_in->info.m; i++)
      {
        ptr   = diagPos.val[i];
        nvAdd = rowSize.val[i];

        memcpy(&(B_out->colInd[ind]),&(A_in->colInd[ptr]),nvAdd*sizeof(int));
        memcpy(&(B_out->val[ind])   ,&(A_in->val[ptr]),   nvAdd*sizeof(double));

        B_out->rowPtr[i+1]  =   B_out->rowPtr[i] + nvAdd;
        ind                 +=  nvAdd;
      }
    }
    else
    {
      for(int i = 0; i < A_in->info.m; i++)
      {
        ptr   = colPos->val[i * nblock + numBlock];
        nvAdd = rowSize.val[i];

        memcpy(&(B_out->colInd[ind]),&(A_in->colInd[ptr]),nvAdd*sizeof(int));
        memcpy(&(B_out->val[ind])   ,&(A_in->val[ptr]),   nvAdd*sizeof(double));

        B_out->rowPtr[i+1]  =   B_out->rowPtr[i] + nvAdd;
        ind                 +=  nvAdd;
      }
    }

    int offset = pos->val[numBlock];

    for(int i = 0; i < B_out->info.lnnz; i++)
      B_out->colInd[i] -= offset;

    CPLM_IVectorFree(&rowSize);
    CPLM_IVectorFree(&diagPos);

  }
  else if(colPos == NULL) //A_in is a colPanel
  {

    CPLM_Abort("Case of column panel never tested. So => ABORT");

    CPLM_IVector_t lrowSize  = CPLM_IVectorNULL();
    CPLM_IVector_t endRow    = CPLM_IVectorNULL();
    int nvAdd = 0;

    lm = pos->val[numBlock+1] - pos->val[numBlock];

    if(structure == SYMMETRIC)
    {

      ierr = CPLM_IVectorCreateFromPtr ( &endRow,
                                    lm,
                                    &(A_in->rowPtr[pos->val[numBlock+1]])
                                  );CPLM_CHKERR(ierr);
      ierr = CPLM_IVectorSum(&endRow,&diagPos,&lrowSize,1);CPLM_CHKERR(ierr);
      ierr = CPLM_IVectorReduce(&lrowSize,&lnnz);CPLM_CHKERR(ierr);
    }
    else
    {
	    offset  = A_in->rowPtr[pos->val[numBlock]];
      lnnz    = A_in->rowPtr[pos->val[numBlock+1]] - A_in->rowPtr[pos->val[numBlock]];
    }

    ierr = CPLM_MatCSRSetInfo(B_out,
                            A_in->info.m,
                            A_in->info.n,
                            A_in->info.lnnz,
                            lm,
                            A_in->info.n,
                            lnnz,
                            1);CPLM_CHKERR(ierr);

    ierr = CPLM_MatCSRMalloc(B_out);CPLM_CHKERR(ierr);
	  /*
    * ====================
    *    copy of arrays
    * =====================
    */
    if(structure == SYMMETRIC)
    {
      B_out->rowPtr[0]=0;
      for(int i = 0; i < lm; i++)
      {
        int begin = diagPos.val[pos->val[numBlock]+i];
	      //copy of arrays
        memcpy(&(B_out->colInd[nvAdd]), &(A_in->colInd[begin]),            lrowSize.val[i]*sizeof(int));
        memcpy(&(B_out->val[nvAdd]),    &(A_in->val[begin]),               lrowSize.val[i]*sizeof(double));
        B_out->rowPtr[i+1]  = lrowSize.val[i] + B_out->rowPtr[i];
        nvAdd += lrowSize.val[i];
      }
    }
    else
    {
	    //copy of arrays
      memcpy(B_out->colInd, &(A_in->colInd[offset]),            B_out->info.lnnz*sizeof(int));
      memcpy(B_out->val,    &(A_in->val[offset]),               B_out->info.lnnz*sizeof(double));
      memcpy(B_out->rowPtr, &(A_in->rowPtr[pos->val[numBlock]]),  (B_out->info.m+1)*sizeof(int));

      for(int i = 0; i < B_out->info.m + 1; i++)
	      B_out->rowPtr[i]  -=  offset;
    }
  }
  else//i.e. not a panel
  {
    CPLM_Abort("The non-panel shape of A is not implemented yet");
  }
CPLM_END_TIME
CPLM_POP
	return ierr;
}


/**
 * \fn int CPLM_MatCSRGetDiagInd(CPLM_Mat_CSR_t *A_in, CPLM_IVector_t *v)
 * \brief This function looks for position of diagonal elements in m and assume the matrix does not have any zero on the diagonal
 * \param *A_in   The CSR matrix
 * \param *v      The CPLM_IVector_t returned containing indices
 * \return      0 if the memory allocation is ok
 */
int CPLM_MatCSRGetDiagInd(CPLM_Mat_CSR_t *A_in, CPLM_IVector_t *v)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  int ierr=0;

  if(v->val == NULL)
  {
    ierr = CPLM_IVectorMalloc(v,A_in->info.m);CPLM_CHKERR(ierr);
  }

  for(int i = 0; i < A_in->info.m; i++)
  {
    for(int col = A_in->rowPtr[i]; col < A_in->rowPtr[i+1]; col++)
    {
      if(A_in->colInd[col] == i)
      {
        v->val[i] = col;
        break;
      }
    }
  }

CPLM_END_TIME
CPLM_POP
  return ierr;
}

/**
 * \fn int CPLM_MatCSRGetDiagIndOfPanel(CPLM_Mat_CSR_t *A_in, CPLM_IVector_t *v)
 * \brief This function looks for position of diagonal elements in m and assume the matrix does not have any zero on the diagonal
 * \param *A_in   The CSR matrix
 * \param *v      The CPLM_IVector_t returned containing indices
 * \param offset  The offset to local the beginning of the diagonal block in the panel
 * \return      0 if the memory allocation is ok
 */
int CPLM_MatCSRGetDiagIndOfPanel(CPLM_Mat_CSR_t *A_in, CPLM_IVector_t *d_out, CPLM_IVector_t *pos, CPLM_IVector_t *colPos, int numBlock)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  int ierr      = 0;
  int cpt       = 0;
  int rowLayout = 0;
  int rowB      = 0; //The beginning index of the block line
  int rowE      = 0; //The beginning index of the next block line
  int colB      = 0;
  int nblock    = 0;
  int *colPtr = NULL;

  nblock = pos->nval-1;

  //Check if it is either a row layout or a column layout
  if(A_in->info.m > A_in->info.n)//i.e. Column layout
  {
    rowB  = pos->val[numBlock];
    rowE  = pos->val[numBlock+1];
    colPtr = A_in->rowPtr;
  }
  else//i.e. Row layout
  {
    rowB = 0;
    rowE = A_in->info.m;
    colPtr = colPos->val;
    rowLayout = 1;
    colB = pos->val[numBlock];
    //CPLM_debug("colB %d\n",colB);
  }

  //Allocate the output vector
  if(d_out->val == NULL)
  {
    ierr = CPLM_IVectorMalloc(d_out, rowE-rowB);CPLM_CHKERR(ierr);
  }

  //Search position of each diagonal element
  for(int i = rowB; i < rowE; i++)
  {
    //CPLM_debug("Row %d\n",i);
    int j = ((rowLayout == 0) ? i : (i * nblock + numBlock));
    //CPLM_debug("Corrected Row %d\n",j);
    for(int col = colPtr[j]; col < colPtr[j+1]; col++)
    {
      //CPLM_debug("Colptr %d\tof col %d =?= %d\n",col,A_in->colInd[col]-colB,i);
      if(A_in->colInd[col] - colB == i )
      {
        d_out->val[cpt++] = col;
        break;
      }
    }
  }

CPLM_END_TIME
CPLM_POP
  return ierr;
}


/*
*
* This function returns colPos which is an index of the begin and end of each block in column point of view.
*
*/
int CPLM_MatCSRGetPartialColBlockPos(CPLM_Mat_CSR_t *A_in,
                                CPLM_IVector_t *posR,
                                int       numBlock,
                                CPLM_IVector_t *posC,
                                CPLM_IVector_t *colPos)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  int nblockR     = posR->nval - 1;
  int nblockC     = posC->nval - 1;
  int ierr        = 0;
  int newNumBlock = 0;
  int c           = 0;
  int curRow      = 0;
  int nrow        = 0;
  int offsetB     = 0;
  int offsetE     = 0;


  offsetB = posR->val[numBlock];
  offsetE = posR->val[numBlock + 1];
  nrow    = offsetE - offsetB;

  if(colPos->val == NULL)
  {
    ierr = CPLM_IVectorMalloc(colPos, nrow * nblockC + 1);CPLM_CHKERR(ierr);
  }
  else if((nrow * nblockC + 1) > colPos->nval)
  {
  //ierr = CPLM_IVectorRealloc(colPos, nrow * nblock + 1);CPLM_CHKERR(ierr);
    CPLM_Abort("Try to realloc a pointer that could be symbolic");
  }

  colPos->val[0]                = A_in->rowPtr[offsetB];
  colPos->val[colPos->nval - 1] = A_in->rowPtr[offsetE];

  for(int i = 0; i < nrow; i++)
  {
    numBlock = 0;
    curRow = offsetB + i;
    colPos->val[i * nblockC] = A_in->rowPtr[curRow];
    for(int j = A_in->rowPtr[curRow]; j < A_in->rowPtr[curRow + 1]; j++)
    {
      c = A_in->colInd[j];
      while(c >= posC->val[numBlock + 1])
      {
        numBlock++;
        CPLM_ASSERT((i * nblockC + numBlock) < colPos->nval);
        colPos->val[i * nblockC + numBlock] = j;
      }
    }
    for(int k = numBlock + 1; k <= nblockC; k++)
    {
      CPLM_ASSERT((i * nblockC + k) < colPos->nval);
      CPLM_ASSERT((i + 1) < A_in->info.m + 1);
      colPos->val[i * nblockC + k] = A_in->rowPtr[curRow + 1];
    }
  }

CPLM_END_TIME
CPLM_POP
  return ierr;
}


/**
 * \fn int CPLM_MatCSRGetRowPanel(CPLM_Mat_CSR_t *A_in, CPLM_Mat_CSR_t *B_out, CPLM_IVector_t *pos, int numBlock)
 * \brief Function creates a CSR matrix without value which corresponds to a submatrix of the original CSR matrix and this submatrix is a part given by Metis_GraphPartKway
 * Note : this function does not need the matrix values
 * \param *A_in         The original CSR matrix
 * \param *B_out        The original CSR matrix
 * \param *pos          The array containing the begin of each part of the CSR matrix
 * \param numBlock      The number of the part which will be returned
 * \return            0 if succeed
 */
/*function which returns a submatrice at CSR format from an original matrix (in CSR format too) and filtered by parts*/
/*num_parts corresponds to the number which selects rows from original matrix and interval allows to select the specific rows*/
int CPLM_MatCSRGetRowPanel(CPLM_Mat_CSR_t *A_in, CPLM_Mat_CSR_t *B_out, CPLM_IVector_t *pos, int numBlock)
{

  int ierr    = 0;
  int lm      = 0;
  int lnnz    = 0;
	//variable which adapts the values in rowPtr for local colInd
	int offset  = 0;

  CPLM_ASSERT(numBlock < pos->nval);
  CPLM_ASSERT(pos->val[numBlock+1] < A_in->info.m+1);

  lm      = pos->val[numBlock+1]-pos->val[numBlock];
  lnnz    = A_in->rowPtr[pos->val[numBlock+1]]-A_in->rowPtr[pos->val[numBlock]];
	offset  = A_in->rowPtr[pos->val[numBlock]];

  if(B_out->val == NULL || (lm > B_out->info.m || lnnz > B_out->info.lnnz) )
  {

    B_out->info = A_in->info;

	  B_out->info.M    = A_in->info.m;
	  B_out->info.nnz  = A_in->info.lnnz;
    B_out->info.m    = lm;
    B_out->info.lnnz = lnnz;

    if(B_out->val == NULL)
    {
      ierr  = CPLM_MatCSRMalloc(B_out);CPLM_CHKERR(ierr);
    }
    else
    {
      ierr  = CPLM_MatCSRRealloc(B_out);CPLM_CHKERR(ierr);
    }
  }
  //Since it could be a work allocated before with a bigger size, we need to set the new real number of nnz in the row panel.
  //It involves an extra memory size if lnnz is lower than the previous one.
  //For instance, if lnnz = [ 12, 11, 11], at first, lnnz = 12 and after the memory size allocated is still 12 but only 11 values are in.
  B_out->info.lnnz  = lnnz;
  B_out->info.m     = lm;

	//CPLM_debug("Block n°%d nb_nnz = %d\tnb_vertices = %d\n",numBlock, B_out->info.lnnz, B_out->info.m);

  if(lnnz > 0)
  {
    CPLM_ASSERT(offset < A_in->info.lnnz);
    CPLM_ASSERT((offset + B_out->info.lnnz - 1) < A_in->info.lnnz);

    //copy of arrays
    memcpy(B_out->colInd, &(A_in->colInd[offset]),            B_out->info.lnnz*sizeof(int));
    memcpy(B_out->val,    &(A_in->val[offset]),               B_out->info.lnnz*sizeof(double));
    memcpy(B_out->rowPtr, &(A_in->rowPtr[pos->val[numBlock]]),  (B_out->info.m+1)*sizeof(int));

    for(int i = 0; i < B_out->info.m + 1; i++)
    {
      B_out->rowPtr[i]  -=  offset;
    }

  }
  else
  {
    memset(B_out->rowPtr, 0, (B_out->info.m + 1) * sizeof(int));
  }

	return ierr;
}


int CPLM_MatCSRGetSubBlock ( CPLM_Mat_CSR_t *A_in,
                        CPLM_Mat_CSR_t *B_out,
                        CPLM_IVector_t *posR,
                        CPLM_IVector_t *posC,
                        int       numRBlock,
                        int       numCBlock,
                        int       **work,
                        size_t    *workSize)
{
CPLM_PUSH
CPLM_BEGIN_TIME
  int ierr    = 0;
  int nblockR = posR->nval - 1;
  int nblockC = posC->nval - 1;
  int sum     = 0;
  int fl      = 0;  //First line of the block
  int flnb    = 0;  //First line of the next block
  int nrow    = 0;
  int fc      = 0;  //First col of the block
  int fcnb    = 0;  //First col of the next block
  int ncol    = 0;
  int ind     = 0;  //
  int ptr     = 0;  //Position of the first column of the block
  int ptrC    = 0;  //Pointer to the first column of the block
  int lnvAdd  = 0;  //Number of values on the current line to add
  int lwork   = 0;  //Computed ideal size of the work
  int ncolPos = 0;  //Size of colPos
  int offset  = 0;  //Offset applied to the colInd array to shift the column
                    // values such that the first column of the block in A
                    // is numbered as 0 in B
  CPLM_IVector_t rowSize_s = CPLM_IVectorNULL();
  CPLM_IVector_t colPos_s  = CPLM_IVectorNULL();

  CPLM_ASSERT(A_in       != NULL);
  CPLM_ASSERT(B_out      != NULL);
  CPLM_ASSERT(A_in->val  != NULL);
  CPLM_ASSERT(posR       != NULL);
  CPLM_ASSERT(posC       != NULL);
  CPLM_ASSERT(work       != NULL);

  fl    = posR->val[numRBlock];
  flnb  = posR->val[numRBlock + 1];
  nrow  = flnb - fl;

  fc    = posC->val[numCBlock];
  fcnb  = posC->val[numCBlock + 1];
  ncol  = fcnb - fc;

  //If non-allocated memory or number of columns are greater than previous call
  if(B_out->val == NULL || B_out->info.n < nrow )
  {
    B_out->info      = A_in->info;
    B_out->info.M    = A_in->info.m;
    B_out->info.N    = A_in->info.n;
    B_out->info.m    = nrow;
    B_out->info.n    = ncol;
	  B_out->info.nnz  = A_in->info.lnnz;
    B_out->info.lnnz = 0;
  }

  ncolPos = nrow * nblockC + 1;
  lwork   = nrow + ncolPos;

  if (*work == NULL)
  {
    *workSize = (size_t)lwork;
    *work     = (int*) malloc(*workSize * sizeof(int));
    CPLM_ASSERT(*work != NULL);
  }
  else if(*workSize < lwork)
  {
    *workSize = (size_t)lwork;
    *work     = (int*) realloc(*work, *workSize * sizeof(int));
    CPLM_ASSERT(*work != NULL);
  }


  ierr = CPLM_IVectorCreateFromPtr(&rowSize_s, nrow, *work);CPLM_CHKERR(ierr);
  ierr = CPLM_IVectorCreateFromPtr(&colPos_s,
      ncolPos,
      rowSize_s.val + rowSize_s.nval);CPLM_CHKERR(ierr);



  ierr = CPLM_MatCSRGetPartialColBlockPos(A_in,
      posR,
      numRBlock,
      posC,
      &colPos_s);CPLM_CHKERR(ierr);
//CPLM_IVectorPrintf("Partial colPos", &colPos_s);


  //Count how many lnnz there will be
  for(int i = 0; i < nrow; i++)
  {
    ptr = i * nblockC + numCBlock;

    rowSize_s.val[i]  = colPos_s.val[ptr + 1] - colPos_s.val[ptr];
  //CPLM_debug("%d elements in row %d\n", rowSize_s.val[i], i);
    sum               += rowSize_s.val[i];
  }

  if(!sum)
  {
    return ierr;
  }
//CPLM_debug("sum %d\n", sum);

  if(sum > B_out->info.lnnz)
  {
      B_out->info.lnnz = sum;

      if(B_out->val == NULL)
      {
        ierr = CPLM_MatCSRMalloc(B_out);CPLM_CHKERR(ierr);
      }
      else
      {
        ierr = CPLM_MatCSRRealloc(B_out);CPLM_CHKERR(ierr);
      }
  }
  else
  {
    //Here we can loose some space and imply a realloc elsewhere
    B_out->info.lnnz = sum;
  }

  /*
  * ====================
  *    copy of arrays
  * =====================
  */
  B_out->rowPtr[0] = 0;
  for(int i = 0; i < nrow; i++)
  {
    ptrC   = colPos_s.val[i * nblockC + numCBlock];
    lnvAdd = rowSize_s.val[i];

    memcpy(B_out->colInd  + ind,  A_in->colInd + ptrC, lnvAdd * sizeof(int));
    memcpy(B_out->val + ind,      A_in->val + ptrC,    lnvAdd * sizeof(double));

    B_out->rowPtr[i + 1]  =   B_out->rowPtr[i] + lnvAdd;
    ind                   +=  lnvAdd;
  }

  offset = posC->val[numCBlock];
  for(int i = 0; i < B_out->info.lnnz; i++)
  {
    B_out->colInd[i] -= offset;
  }


CPLM_END_TIME
CPLM_POP
	return ierr;
}

int CPLM_MatCSRIsSym(CPLM_Mat_CSR_t *m)
{
  int ok      = CPLM_TRUE;
  int ierr    = 0;
  int sym     = 1;
  int column  = 0;
  int length  = 0;
  double val = 0.0;
  CPLM_IVector_t ptr_diag = CPLM_IVectorNULL();

  ierr = CPLM_MatCSRGetDiagInd(m,&ptr_diag);CPLM_CHKERR(ierr);

  for(int line = 0; line < m->info.m; line++)
  {
    sym = 1;
    for (int idx_column = ptr_diag.val[line] + 1; idx_column < m->rowPtr[line+1]; idx_column++)
    {
      column  = m->colInd[idx_column];
      val     = m->val[idx_column];
      for(int ii = m->rowPtr[column]; ii < ptr_diag.val[column]; ii++)
      {
        if(m->colInd[ii] == line && m->val[ii] == val)
        {
          sym++;
          break;
        }
      }

    }
    length = (m->rowPtr[line+1] - ptr_diag.val[line]);
    if(sym != length)
    {
      fprintf(stderr,"Error, Matrix is not symmetric\nInfo : Line %d => %d / %d values have its symmetric value\n",line,sym,length);
      return !ok;
    }
  }

  CPLM_IVectorFree(&ptr_diag);

  return ok;
}

/**
 * \fn int CPLM_MatCSRPermute(CPLM_Mat_CSR_t *m1,
 * CPLM_Mat_CSR_t *m2,
 * int *colPerm,
 * int *rowPerm,
 * Choice_permutation permute_values)
 * \brief Function creates a CSR matrix and fills in with the original
 *        CSR matrix which has been permuted
 * \param *m1               The original CSR matrix
 * \param *m2               The permuted matrix CSR matrix
 * \param *colPerm          The array of permutation where the row of the
 *                          returned CSR matrix is equal to perm[i] where i
 *                          is the i'th row of the original CSR matrix
 * Note : i -> perm[i]
 * \param *rowPerm          The array of invert permutation
 * \param permute_values    Indicate if values have to be permuted or ignored
 * \return                  0 if permutation succeed
 */
/*Function which permutes CPLM_Mat_CSR_t matrix with colPerm and rowPerm vector*/
int CPLM_MatCSRPermute(CPLM_Mat_CSR_t           *A_in,
                  CPLM_Mat_CSR_t           *B_out,
                  int                 *rowPerm,
                  int                 *colPerm,
                  Choice_permutation  permute_values)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  int ierr  = 0;
	int lnval = 0;
  CPLM_IVector_t tmp_s     = CPLM_IVectorNULL();
  CPLM_IVector_t iColPerm  = CPLM_IVectorNULL();

  B_out->info = A_in->info;

	if(permute_values ==  PERMUTE)
  {
		ierr  = CPLM_MatCSRMalloc(B_out);CPLM_CHKERR(ierr);
  }
  else
  {
	  B_out->rowPtr = (int*)malloc((B_out->info.m+1)  * sizeof(int));
	  B_out->colInd = (int*)malloc(B_out->info.lnnz   * sizeof(int));
    CPLM_ASSERT(B_out->rowPtr != NULL);
    CPLM_ASSERT(B_out->colInd != NULL);
  }

  ierr = CPLM_IVectorCreateFromPtr(&tmp_s,A_in->info.n,colPerm);CPLM_CHKERR(ierr);
  ierr = CPLM_IVectorInvert(&tmp_s,&iColPerm);CPLM_CHKERR(ierr);

	B_out->rowPtr[0]=0;

  //Copy data where rows are permuted too
  for(int i = 0; i < B_out->info.m; i++)
  {
    lnval = A_in->rowPtr[ rowPerm[i] + 1] - A_in->rowPtr[ rowPerm[i] ];
    memcpy( B_out->colInd + B_out->rowPtr[i],
            A_in->colInd + A_in->rowPtr[rowPerm[i]],
            lnval * sizeof(int));
  	B_out->rowPtr[i+1] = B_out->rowPtr[i] + lnval;
  }

	if(permute_values == PERMUTE)
  {
  	for(int i = 0; i < B_out->info.m; i++)
    {
      lnval = A_in->rowPtr[ rowPerm[i] + 1] - A_in->rowPtr[ rowPerm[i] ];
      memcpy( B_out->val + B_out->rowPtr[i],
              A_in->val + A_in->rowPtr[rowPerm[i]],
              lnval * sizeof(double));
  	}
  }

  //Permute columns
  for(int i = 0; i < B_out->info.lnnz; i++)
  	B_out->colInd[i] = iColPerm.val[B_out->colInd[i]];

  //Sort columns
  if(permute_values == PERMUTE)
  {
	  for(int i = 0; i < B_out->info.m; i++)
	  	CPLM_quickSortWithValues(  B_out->colInd,
                            B_out->rowPtr[i],
                            B_out->rowPtr[i+1] - 1,
                            B_out->val);
  }
  else
  {
	  for(int i = 0; i < B_out->info.m;i++)
	  	CPLM_quickSort(  B_out->colInd,
                  B_out->rowPtr[i],
                  B_out->rowPtr[i+1] - 1);
  }

  CPLM_IVectorFree(&iColPerm);

CPLM_END_TIME
CPLM_POP
	return ierr;

}


/*
 * Split the matrix in block column and extract the selected block column number.
 * The input matrix is unchanged
 * A:
 *     input: the input matrix
 * nparts:
 *     input: the number of block columns
 * partBegin: Array of size nparts + 1
 *     input: the begining position of each blocks
 * numBlock:
 *     input: the number of the block to remove
 * B_out:
 *     out: the output block
 */

int CPLM_MatCSRBlockColumnExtract(CPLM_Mat_CSR_t *A, int nparts, int *partBegin, int numBlock, CPLM_Mat_CSR_t *B_out){

  int ierr=0, *work = NULL;
  size_t workSize = 0;

  CPLM_IVector_t rowPart = CPLM_IVectorNULL();
  CPLM_IVector_t colPart = CPLM_IVectorNULL();

  //We have only one block Row
  ierr = CPLM_IVectorMalloc(&rowPart, 2); CPLM_CHKERR(ierr);
  rowPart.val[0] = 0; rowPart.val[1] = A->info.m; //keep the same number of rows

  /* Create a column partitioning */
  CPLM_IVectorCreateFromPtr(&colPart, nparts+1, partBegin);

  ierr = CPLM_MatCSRGetSubBlock (A, B_out, &rowPart, &colPart,
                                  0, numBlock, &work, &workSize); CPLM_CHKERR(ierr);

  B_out->info.nnz = A->info.lnnz; //tmp bug fix in CPLM_MatCSRGetSubBlock(); which does not set nnz when the matrix is empty
  if(work!=NULL) free(work);

  return ierr;
}
