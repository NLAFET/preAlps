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
#include <strings.h>
#include <math.h>
#ifdef USE_PARMETIS
  #include <parmetis.h>
#endif

#include "preAlps_param.h"
//#include "preAlps_utils.h"
#include "cplm_utils.h"
#include "cplm_v0_timing.h"
#include "cplm_ijval.h"
#include "cplm_QS.h"
#include "cplm_v0_ivector.h"
#include "cplm_v0_dvector.h"
#include "cplm_matcsr.h"


/* For the matmult*/
//#include <cpalamem_macro.h>
//#include <cpalamem_instrumentation.h>

/*
 *
 * Author: Sebastion Cayrols
 *
 */

/* MPI Utils */
//define the number of champ in the structure Partial_CPLM_Mat_CSR_t
#define _NB_CHAMPS 9
MPI_Datatype initMPI_StructCSR(){

  CPLM_Mat_CSR_t m;

  MPI_Datatype MPI_MAT_CSR;
	MPI_Datatype type[_NB_CHAMPS]={MPI_INT,//M
											 MPI_INT,//N
											 MPI_INT,//nnz
											 MPI_INT,//m
											 MPI_INT,//n
											 MPI_INT,//lnnz
											 MPI_INT,//blockSize
											 MPI_INT,//type has to change and check with right struct called CPLM_Mat_CSR_format_t
											 MPI_INT
											 };

	int blocklen[_NB_CHAMPS];
	for(int i=0;i<_NB_CHAMPS;i++)
		blocklen[i]=1;

	MPI_Aint disp[_NB_CHAMPS];
	MPI_Aint addr[_NB_CHAMPS+1];

	MPI_Get_address(&m,               &addr[0]);
	MPI_Get_address(&m.info.M,        &addr[1]);
	MPI_Get_address(&m.info.N,        &addr[2]);
	MPI_Get_address(&m.info.nnz,      &addr[3]);
	MPI_Get_address(&m.info.m,        &addr[4]);
	MPI_Get_address(&m.info.n,        &addr[5]);
	MPI_Get_address(&m.info.lnnz,     &addr[6]);
	MPI_Get_address(&m.info.blockSize,&addr[7]);
	MPI_Get_address(&m.info.format,   &addr[8]);
	MPI_Get_address(&m.info.structure,&addr[9]);

  for(int i=0;i<_NB_CHAMPS;i++)
    disp[i] = addr[i+1] - addr[0];

	MPI_Type_create_struct(_NB_CHAMPS,blocklen,disp,type,&MPI_MAT_CSR);
  MPI_Type_commit(&MPI_MAT_CSR);

  return MPI_MAT_CSR;
}

/* MatrixMarket routines */

 /*
  * Load a matrix from a file using MatrixMarket format
  *
  * TODO: full support of the MatrixMarket format?
  */
 int CPLM_LoadMatrixMarket( const char* filename, CPLM_Mat_CSR_t* mat)
 {
 CPLM_PUSH
 CPLM_BEGIN_TIME
 #ifndef SILENCE
 	printf("Load of %s ...\n",filename);
 #endif

   int i         = 0;
   int n         = 0;
   int errorCode = 0;
   int nread     = 0;
   FILE *fd  = NULL;
   char buf[MM_MAX_LINE_LENGTH];
   char banner[MM_MAX_TOKEN_LENGTH];
   char matrix[MM_MAX_TOKEN_LENGTH];
   char coord[MM_MAX_TOKEN_LENGTH];
   char dataType[MM_MAX_TOKEN_LENGTH];
   char storageScheme[MM_MAX_TOKEN_LENGTH];
   CPLM_IJVal_t *ijvalsTmp  = NULL;
   char *ptr = NULL;
   //Avoiding warning of unused nread during compilation
   nread++;
   ptr++;

   //open the file
   fd = fopen(filename, "r");
   if (!fd)
   {
     CPLM_Abort("Impossible to open the file %s",filename);
   }

   // Get the header line
   ptr = fgets(buf, MM_MAX_LINE_LENGTH, fd);
   sscanf(buf, "%s %s %s %s %s", banner, matrix, coord, dataType, storageScheme);
   if (strcasecmp(banner, MM_MATRIXMARKET_STR)
         || strcasecmp(matrix, MM_MATRIX_STR)
         || strcasecmp(coord, MM_SPARSE_STR)
         || strcasecmp(dataType, MM_REAL_STR)
         || (strcasecmp(storageScheme, MM_GENERAL_STR) && strcasecmp(storageScheme, MM_SYMM_STR)))
         {
     fclose(fd);
     CPLM_Abort("Only sparse real < symmetric | general > matrix are currently supported.\nHere is %s",storageScheme);
   }

   // Skip comment lines
   do {
     ptr = fgets(buf, MM_MAX_LINE_LENGTH, fd);
   } while (!feof(fd) && buf[0] == '%');

   // Get matrix size
   sscanf(buf, "%d%d%d", &mat->info.M, &mat->info.N, &mat->info.nnz);
   if (mat->info.M < 1 || mat->info.N < 1 || mat->info.nnz < 1 || mat->info.nnz > (long long int)mat->info.M * mat->info.N) {
     fprintf(stderr,"[LoadMatrixMarket] Error: Invalid matrix dimensions.\n");
     fclose(fd);
 #ifdef MPIACTIVATE
     MPI_Abort(MPI_COMM_WORLD,1);
 #else
     exit(1);
 #endif
   }

   mat->info.m = mat->info.M;
   mat->info.n = mat->info.N;
   mat->info.lnnz = mat->info.nnz;
   mat->info.blockSize = 1;
   mat->info.format = FORMAT_CSR;
   if(!strcasecmp(storageScheme, MM_SYMM_STR))
     mat->info.structure = SYMMETRIC;
   else
     mat->info.structure = UNSYMMETRIC;

   ijvalsTmp = (CPLM_IJVal_t*)malloc(mat->info.nnz * sizeof(CPLM_IJVal_t));

   int base0=0;
   nread = fscanf(fd, "%d%d%lf", &ijvalsTmp[0].i, &ijvalsTmp[0].j, &ijvalsTmp[0].val); // TODO error checking
   if(ijvalsTmp[0].i==0 || ijvalsTmp[0].j==0){
     printf("0-based detected\n");
     base0=1;
   }else{
     ijvalsTmp[0].i--;
     ijvalsTmp[0].j--;
   }
   if(base0){
     for (n = 1; n < mat->info.nnz; n++)
       nread = fscanf(fd, "%d%d%lf", &ijvalsTmp[n].i, &ijvalsTmp[n].j, &ijvalsTmp[n].val); // TODO error checking

   }else{
     for (n = 1; n < mat->info.nnz; n++) {
       nread = fscanf(fd, "%d%d%lf", &ijvalsTmp[n].i, &ijvalsTmp[n].j, &ijvalsTmp[n].val); // TODO error checking

       // Use 0-based indexing
       ijvalsTmp[n].i--;
       ijvalsTmp[n].j--;
     }
   }
   fclose(fd);

   // Sort the local chunk
   qsort(ijvalsTmp, mat->info.nnz, sizeof(CPLM_IJVal_t), CPLM_CompareIJVal);

   // Convert to CSR
   mat->rowPtr = (int*)malloc((mat->info.M + 1) * sizeof(int));
   mat->colInd = (int*)malloc(mat->info.nnz * sizeof(int));
   mat->val = (double*)malloc(mat->info.nnz * sizeof(double));

   mat->rowPtr[0] = 0;
   for (n = i = 0; n < mat->info.nnz; n++) {
     while (ijvalsTmp[n].i > i) {
       i++;
       mat->rowPtr[i] = n;
     }

     mat->colInd[n] = ijvalsTmp[n].j;
     mat->val[n] = ijvalsTmp[n].val;
   }
   while (mat->info.M > i) {
     i++;
     mat->rowPtr[i] = n;
   }

   // Memory clean-up
   free(ijvalsTmp);

   if(mat->info.structure==SYMMETRIC)
   {
     CPLM_Mat_CSR_t tmp = CPLM_MatCSRNULL();

     //symmetrizeMatrix(mat,&tmp);
     errorCode = CPLM_MatCSRUnsymStruct(mat, &tmp);CPLM_CHKERR(errorCode);

     if(!CPLM_MatCSRIsSym(&tmp))
     {
       CPLM_Abort("The matrix is not symmetric as specified in the structure");
     }
     CPLM_MatCSRFree(mat);
     *mat = tmp;
   }

   if(CPLM_MatCSRChkDiag(mat))
   {
     CPLM_Abort("Diagonal is not set correctly");
   }

 CPLM_END_TIME
 CPLM_POP
   return 0;
}

/* Partitioning routines */

idx_t* callKway(CPLM_Mat_CSR_t *matCSR, idx_t nbparts)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  int ierr  = 0;
  idx_t nvtxs   = 0;
	idx_t ncon    = 1;
  idx_t *parts  = NULL;
	idx_t *xadj   = NULL;
  idx_t *adjncy = NULL;
  idx_t objval  = 0;

  //variable utilisée pour Kway donc recopie au bon type
	nvtxs = (idx_t)matCSR->info.M;

	//test if malloc has succeed
	parts   = (idx_t*)malloc (nvtxs             * sizeof(idx_t));
	xadj    = (idx_t*)malloc((nvtxs + 1)        * sizeof(idx_t));
	adjncy  = (idx_t*)malloc((matCSR->info.nnz) * sizeof(idx_t));

  CPLM_ASSERT(parts  != NULL);
  CPLM_ASSERT(xadj   != NULL);
  CPLM_ASSERT(adjncy !=  NULL);

	//copy the value of the matrix which is in CSR format int to CSR format idx_t
	for(int i = 0; i < nvtxs + 1; i++)
		xadj[i] = matCSR->rowPtr[i];

	for(int i = 0; i < matCSR->info.nnz; i++)
		adjncy[i] = matCSR->colInd[i];

	//call Metis function to get _NBPARTS blocks from the matrix represented by xadj and adjncy array
	ierr = METIS_PartGraphKway(&nvtxs,&ncon,xadj,adjncy,NULL,NULL,NULL,&nbparts,NULL,NULL,NULL,&objval,parts);

	//Match the return value of METIS_PartGraphKway function
	switch(ierr)
  {
	  case METIS_ERROR:
	  	fprintf(stderr,"Error\n");
	  	exit(1);
	  	break;
	  case METIS_ERROR_INPUT:
	  	fprintf(stderr,"Error INPUT\n");
	  	exit(1);
	  	break;
	  case METIS_ERROR_MEMORY:
	  	fprintf(stderr,"Error MEMORY\n");
	  	exit(1);
	  case METIS_OK:
	  	break;
	  default:
	  	fprintf(stderr,"Unknown value returned by METIS_PartGraphKway\n");
	  	exit(1);
	}

	free(xadj);
	free(adjncy);

CPLM_END_TIME
CPLM_POP
	return parts;

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


/* MatCSR functions */

/**
 * \fn int CPLM_MatCSRChkDiag(CPLM_Mat_CSR_t *m)
 * \brief This function looks for position of diagonal elements in m and assume the matrix does not have any zero on the diagonal
 * \param *m    The CSR matrix
 * \param *v    The CPLM_IVector_t returned containing indices
 * \return      0 if the searching is done
 */
int CPLM_MatCSRChkDiag(CPLM_Mat_CSR_t *m){
CPLM_PUSH
CPLM_BEGIN_TIME
  int err=0;
  int min_mn = (m->info.n < m->info.m) ? m->info.n : m->info.m;
  for(int i=0;i<min_mn;i++){
    int found=0;
    for(int col=m->rowPtr[i];col<m->rowPtr[i+1];col++){
      if(m->colInd[col]==i){
        found=1;
        break;
      }
    }
    if(!found){
      fprintf(stderr,"Error, no diagonal value on the row %d\n",i);
      return 1;
    }
  }
CPLM_END_TIME
CPLM_POP
  return err;
}

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


/**
 * \fn void CPLM_MatCSRConvertTo0BasedIndexing(CPLM_Mat_CSR_t *m)
 * \brief Convert a matrix into C format index
 * \param *matCSR The matrix which has to be reformated
 */
void CPLM_MatCSRConvertTo0BasedIndexing(CPLM_Mat_CSR_t *m)
{
  if (m) {
    int i;
    for (i = 0; i <= m->info.m; i++) {
      m->rowPtr[i]--;
    }
    for (i = 0; i < m->info.lnnz; i++) {
      m->colInd[i]--;
    }
  }
}

/**
 * \fn void CPLM_MatCSRConvertTo1BasedIndexing(CPLM_Mat_CSR_t *m)
 * \brief Convert a matrix into matlab format index
 * \param *matCSR The matrix which has to be reformated
 */
void CPLM_MatCSRConvertTo1BasedIndexing(CPLM_Mat_CSR_t *m)
{
  if (m) {
    int i;
    for (i = 0; i <= m->info.m; i++) {
      m->rowPtr[i]++;
    }
    for (i = 0; i < m->info.lnnz; i++) {
      m->colInd[i]++;
    }
  }
}

/**
 * \fn int CPLM_MatCSRCopy(CPLM_Mat_CSR_t *m1, CPLM_Mat_CSR_t *m2)
 * \brief Function which copy all the CSR matrix to a new CSR matrix
 * \param *m1     The original matrix which has to be copied
 * \param *m2     The copy matrix
 * \return        0 if copy has succeed
 */
int CPLM_MatCSRCopy(CPLM_Mat_CSR_t *m1, CPLM_Mat_CSR_t *m2){
CPLM_PUSH
CPLM_BEGIN_TIME

  int err=0;

  err = CPLM_MatCSRCopyStruct(m1,m2);

	memcpy(m2->val,m1->val,m2->info.lnnz*sizeof(double));

CPLM_END_TIME
CPLM_POP
	return err;

}

/**
 * \fn int CPLM_MatCSRCopyStruct(CPLM_Mat_CSR_t *m1, CPLM_Mat_CSR_t *m2)
 * \brief Function which copy the structure of the CSR matrix to a new CSR matrix
 * \param *m1     The original matrix which has to be copied
 * \param *m2     The copy matrix
 * \return        0 if copy has succeed
 */
int CPLM_MatCSRCopyStruct(CPLM_Mat_CSR_t *A_in, CPLM_Mat_CSR_t *B_out)
{
CPLM_PUSH
CPLM_BEGIN_TIME

	int ierr = 0;

  B_out->info = A_in->info;

  ierr = CPLM_MatCSRMalloc(B_out);CPLM_CHKERR(ierr);

  memcpy(B_out->rowPtr, A_in->rowPtr, (B_out->info.m+1) * sizeof(int));
  memcpy(B_out->colInd, A_in->colInd, B_out->info.lnnz  * sizeof(int));

CPLM_END_TIME
CPLM_POP
	return ierr;
}


/**
 * \fn int CPLM_MatCSRDelDiag(CPLM_Mat_CSR_t *matCSR)
 * \brief Function which creates a CSR matrix with the same structure of the original CSR matrix and
 * deletes the diagonal values
 * \param *m1 The original CSR matrix
 * \param *m2 The CSR matrix created without diagonal values
 */
/*Function deletes diagonal element from a CPLM_Mat_CSR_t matrix*/
int CPLM_MatCSRDelDiag(CPLM_Mat_CSR_t *m1, CPLM_Mat_CSR_t *m2){
CPLM_PUSH
CPLM_BEGIN_TIME

	int del    = 0;
  int err       = 0;
	int lnAdd     = 0;
  int nAdd      = 0;

  err = CPLM_MatCSRInit(m2,&(m1->info));CPLM_CHKERR(err);

  m2->info.lnnz -= m1->info.m;//Assumption : the diagonal elements are all non zero
  m2->info.M    = m2->info.m;
  m2->info.N    = m2->info.n;
  m2->info.nnz  = m2->info.lnnz;

  err = CPLM_MatCSRMalloc(m2);CPLM_CHKERR(err);

	m2->rowPtr[0] = 0;

	for(int i=0;i<m1->info.m;i++){
		lnAdd   = 0;
		del  = 0;
		for(int j=m1->rowPtr[i];j<m1->rowPtr[i+1];j++){
			if(m1->colInd[j]==i){
				del  = 1;
				continue;
			}

			m2->colInd[nAdd]  = m1->colInd[j];
			m2->val[nAdd]     = m1->val[j];
			nAdd++;
			lnAdd++;
		}

		if(del==0){
			fprintf(stderr,"Error, no diagonal value on row %d\n",i);
			return 1;
		}

		m2->rowPtr[i+1] = m2->rowPtr[i]+lnAdd;
	}

  if (m2->info.lnnz!=nAdd){
			fprintf(stderr,"Error during diagonal elimination\t Malloc of %d elements not equal to %d values added\n",m2->info.nnz,nAdd);
			return 1;
  }
CPLM_END_TIME
CPLM_POP
	return err;
}


/**
 * \fn void CPLM_MatCSRFree(CPLM_Mat_CSR_t *A_io)
 * \brief This method frees the memory occuped by a matrix
 * \param *A_io The matrix which has to be freed
 */
void CPLM_MatCSRFree(CPLM_Mat_CSR_t *A_io)
{
  if(A_io)
  {
    if(A_io->rowPtr)
    {
      free(A_io->rowPtr);
    }
    if(A_io->colInd)
    {
      free(A_io->colInd);
    }
    if(A_io->val)
    {
      free(A_io->val);
    }
    A_io->rowPtr  = NULL;
    A_io->colInd  = NULL;
    A_io->val     = NULL;
  }
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


int CPLM_MatCSRInit(CPLM_Mat_CSR_t *A_out, CPLM_Info_t *info)
{

  A_out->info      = *info;

	A_out->rowPtr    = NULL;
	A_out->colInd    = NULL;
	A_out->val       = NULL;

	return 0;
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
 * \fn int CPLM_MatCSRMalloc(CPLM_Mat_CSR_t *A_io)
 * \brief Allocate the memory following the info part.
 * More precisely, it allows m+1 INT for rowPtr, lnnz INT and lnnz DOUBLE for colInd and val arrays.
 * It checks weither the arrays are null or not.
 * \param *A_io   The matrix to free
 */
int CPLM_MatCSRMalloc(CPLM_Mat_CSR_t *A_io)
{
CPLM_PUSH

	A_io->rowPtr = (int*)   malloc( ( A_io->info.m+1) * sizeof(int));
	A_io->colInd = (int*)   malloc( A_io->info.lnnz   * sizeof(int));
	A_io->val    = (double*)malloc( A_io->info.lnnz   * sizeof(double));

  CPLM_ASSERT(A_io->colInd != NULL);
  CPLM_ASSERT(A_io->rowPtr != NULL);
  CPLM_ASSERT(A_io->val != NULL);

CPLM_POP
	return !((A_io->rowPtr != NULL) && (A_io->colInd != NULL) && (A_io->val != NULL));
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

/**
 * \fn void CPLM_MatCSRPrintInfo(CPLM_Mat_CSR_t *m)
 * \brief Method which prints the data structure
 * \param *info The data structure of a CSR matrix
 */
/*Function prints informations about a CPLM_Info_t matrix */
void CPLM_MatCSRPrintInfo(CPLM_Mat_CSR_t *m){
  CPLM_Info_t p = m->info;
	printf("Matrix %dx%d with %d nnz\tlocal data : %dx%d with %d lnnz %s",
					p.M,p.N,p.nnz,p.m,p.n,p.lnnz,
					(p.structure==SYMMETRIC) ? "Structure = Symmetric\n" : "Structure = Unsymmetric\n");
}

/**
 * \fn int CPLM_MatCSRRealloc(CPLM_Mat_CSR_t *A_io)
 * \brief Reallocate the memory following the info part.
 * More precisely, it allows m+1 INT for rowPtr, lnnz INT and lnnz DOUBLE for colInd and val arrays.
 * It checks weither the arrays are null or not.
 * \param *A_io   The matrix to free
 */
int CPLM_MatCSRRealloc( CPLM_Mat_CSR_t *A_io)
{
CPLM_PUSH

	A_io->rowPtr = (int*)   realloc(A_io->rowPtr, (A_io->info.m+1)  * sizeof(int));
	A_io->colInd = (int*)   realloc(A_io->colInd, A_io->info.lnnz   * sizeof(int));
	A_io->val    = (double*)realloc(A_io->val,    A_io->info.lnnz   * sizeof(double));

  CPLM_ASSERT(A_io->rowPtr != NULL);
  CPLM_ASSERT(A_io->colInd != NULL);
  CPLM_ASSERT(A_io->val != NULL);

CPLM_POP
	return !((A_io->rowPtr != NULL) && (A_io->colInd != NULL) && (A_io->val != NULL));
}

/**
 * \fn int CPLM_MatCSRRecv(CPLM_Mat_CSR_t *m, int source, MPI_Comm comm)
 * \brief This function sends a CSR matrix to id'th MPI process
 * \param *m      The CSR matrix
 * \param source  The MPI process id which sends the matrix
 * \param comm    The MPI communicator of the group
 * \return        0 if the reception succeed
 */
int CPLM_MatCSRRecv(CPLM_Mat_CSR_t *A_out, int source, MPI_Comm comm)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  int ierr  = 0;
  int tag   = 0;
  MPI_Status status;

  MPI_Datatype MPI_MAT_CSR = initMPI_StructCSR();

  //broadcast the matrix over all processes included into MPI comm group
  if( source == -1)
  {
    fprintf(stderr,"Broadcast is not managed for the moment\n");
    ierr = 1;
  }
  else
  {
    //recv information of the matrix
    ierr = MPI_Recv(&(A_out->info),1,MPI_MAT_CSR,source,tag,comm,&status);CPLM_checkMPIERR(ierr,"recv_info");

    ierr = CPLM_MatCSRMalloc(A_out);CPLM_CHKERR(ierr);

    if(A_out->info.m>0){ //The Data will be sent only if the number of rows is greater than 0
      //recv rowPtr
      ierr = MPI_Recv(A_out->rowPtr,A_out->info.m+1,  MPI_INT,source,tag,comm,&status);CPLM_checkMPIERR(ierr,"recv_rowPrt");

      //recv colInd
      ierr = MPI_Recv(A_out->colInd,A_out->info.lnnz, MPI_INT,source,tag,comm,&status);CPLM_checkMPIERR(ierr,"recv_colInd");

      //recv values
      ierr = MPI_Recv(A_out->val,   A_out->info.lnnz, MPI_DOUBLE,source,tag,comm,&status);CPLM_checkMPIERR(ierr,"recv_val");
   }

  }

CPLM_END_TIME
CPLM_POP
  return ierr;
}

/**
 * \fn int CPLM_MatCSRSave(CPLM_Mat_CSR_t *m, const char *filename)
 * \brief Method which saves a CSR matrix into a file
 * \Note This function saves the matrix into Matrix market format
 * \param *m          The CSR matrix which has to be saved
 * \param *filename   The name of the file
 */
/*Function saves a CPLM_Mat_CSR_t matrix in a file*/
int CPLM_MatCSRSave(CPLM_Mat_CSR_t *m, const char *filename){

	FILE *ofd;
	int err = 0;

	//WARNING : Always save in general structure
	const char *first_line = "%%MatrixMarket matrix coordinate real general\n";

	if((ofd = fopen(filename,"w"))==NULL){
		fprintf(stderr,"Error during fopen of %s\n",filename);
		return 1;
	}

	fputs(first_line,ofd);
	fprintf(ofd,"%d %d %d\n",m->info.m,m->info.n,m->info.lnnz);

	for(int line=0;line<m->info.m;line++)
	  for(int pos=m->rowPtr[line];pos<m->rowPtr[line+1];pos++)
	    fprintf(ofd,"%d %d %.16e\n",line+1,m->colInd[pos]+1,m->val[pos]);

	fclose(ofd);

	return err;

}

/**
 * \fn int CPLM_MatCSRSend(CPLM_Mat_CSR_t *m, int dest, MPI_Comm comm)
 * \brief This function sends a CSR matrix to id'th MPI process
 * \param *m    The CSR matrix
 * \param dest  The MPI process id which receives the matrix
 * \param comm  The MPI communicator of the group
 * \return      0 if the sending is done
 */
int CPLM_MatCSRSend(CPLM_Mat_CSR_t *m, int dest, MPI_Comm comm){
CPLM_PUSH
CPLM_BEGIN_TIME

  int err   = 0;
  int tag   = 0;

  MPI_Datatype MPI_MAT_CSR = initMPI_StructCSR();

  //broadcast the matrix to all processes included into MPI comm group
  if( dest == -1){
    fprintf(stderr,"Broadcast is not managed for the moment\n");
    return 1;
  }else{
    //send information of the matrix
    err=MPI_Send(&(m->info),1,MPI_MAT_CSR,dest,tag,comm);CPLM_checkMPIERR(err,"send_info");

    if(m->info.m>0){ //Send Data only if the number of rows is greater than 0
      //send rowPtr
      err=MPI_Send(m->rowPtr,m->info.m+1,   MPI_INT,dest,tag,comm);CPLM_checkMPIERR(err,"send_rowPrt");

      //send colInd
      err=MPI_Send(m->colInd,m->info.lnnz,   MPI_INT,dest,tag,comm);CPLM_checkMPIERR(err,"send_colInd");

      //send values
      err=MPI_Send(m->val,m->info.lnnz,      MPI_DOUBLE,dest,tag,comm);CPLM_checkMPIERR(err,"send_val");
    }
  }
CPLM_END_TIME
CPLM_POP
  return err;
}


int CPLM_MatCSRSetInfo(CPLM_Mat_CSR_t *A_out, int M, int N, int nnz, int m, int n, int lnnz, int blockSize)
{
  int ierr = 0;

  A_out->info.M         = M;
  A_out->info.N         = N;
  A_out->info.nnz       = nnz;
  A_out->info.m         = m;
  A_out->info.n         = n;
  A_out->info.lnnz      = lnnz;
  A_out->info.blockSize = blockSize;
  A_out->info.format    = FORMAT_CSR;
  A_out->info.structure = UNSYMMETRIC;

  return ierr;
}

/**
 * \fn int CPLM_MatCSRSymStruct(CPLM_Mat_CSR_t *m1, CPLM_Mat_CSR_t *m2)
 * \brief Function symmetrizes the structure of a matrix
 * \param *m1   The original CSR matrix
 * \param *m2   The symmetric CSR matrix
 */
/*Function symmetrizes a CPLM_Mat_CSR_t matrix and delete its diagonal elements if wondered*/
int CPLM_MatCSRSymStruct(CPLM_Mat_CSR_t *m1, CPLM_Mat_CSR_t *m2){
CPLM_PUSH
CPLM_BEGIN_TIME

  int ierr = 0;

	if(m1->info.m != m1->info.n){
		fprintf(stderr,"This matrix is not square\n");
		exit(1);
	}

  ierr = CPLM_MatCSRInit(m2,&(m1->info));CPLM_CHKERR(ierr);
  m2->info.M=m2->info.m;
  m2->info.N=m2->info.n;

	m2->rowPtr = (int*) calloc((m1->info.m+1),sizeof(int));
  //m2->rowPtr[0]=0;[TODO change this calloc and the algo maybe ! if time to do that]

	//nb val added into original matrix
	int nbVal=0,valueAdded=0;
  for(int i=0;i<m1->info.m;i++){
	  for(int j=m1->rowPtr[i];j<m1->rowPtr[i+1];j++){
		  int tmp_col=m1->colInd[j];
		  //if not a diagonal element
		  if(tmp_col!=i){
			  valueAdded=0;
			  //For each nnz of the column
			  for(int k=m1->rowPtr[tmp_col];k<m1->rowPtr[tmp_col+1];k++)
				  if(m1->colInd[k]==i){
					  valueAdded=1;
					  break;
				  }
				  else if(m1->colInd[k]>i){
					  valueAdded=1;
					  m2->rowPtr[tmp_col+1]++;
					  nbVal++;
					  break;
				  }
			  if(valueAdded==0){
				  m2->rowPtr[tmp_col+1]++;
				  nbVal++;
			  }
		  }
	  }
  }


	m2->info.lnnz=m1->info.lnnz+nbVal-m1->info.m;
  m2->info.nnz=m2->info.lnnz;

  m2->colInd = (int*)malloc((m2->info.lnnz)*sizeof(int));

	int valueIgnored=0;
	//init by -1
	memset(m2->colInd,-1,m2->info.lnnz*sizeof(int));

	int sum=0;

	for(int i=0;i<m1->info.m;i++){
		//compute the real number of values for the row i
		sum+=m2->rowPtr[i+1]-1;
		m2->rowPtr[i+1]=m1->rowPtr[i+1]+sum;
	}
  for(int i=0;i<m1->info.m;i++){
	  //route the row i
	  for(int j=m1->rowPtr[i];j<m1->rowPtr[i+1];j++){
		  int tmp_col=m1->colInd[j];
		  //if value is not the diagonal
		  if(tmp_col!=i){
			  //copy values from original matrix
			  for(int ii=m2->rowPtr[i];ii<m2->rowPtr[i+1];ii++){
				  if(m2->colInd[ii]==-1){
					  m2->colInd[ii]=tmp_col;
					  break;
				  }
			  }
			  valueIgnored=0;
			  //looking for if i ( the row ) has to be added into tmp_col row
			  for(int k=m1->rowPtr[tmp_col];k<m1->rowPtr[tmp_col+1];k++){
				  //if exists so continue
				  if(i==m1->colInd[k]){
					  valueIgnored=1;
					  break;
				  }
				  else if(i<m1->colInd[k]){
					  break;
				  }
			  }
			  //if all values are less than the row index
			  if(valueIgnored==0){
				  valueAdded=0;
				  for(int ii=m2->rowPtr[tmp_col];ii<m2->rowPtr[tmp_col+1];ii++){
						  if(m2->colInd[ii]==-1){
							  valueAdded=1;
							  m2->colInd[ii]=i;
							  break;
						  }
					  }
			  }
		  }
	  }
  }

	for(int i=0;i<m2->info.m;i++){
		CPLM_quickSort(m2->colInd,m2->rowPtr[i],m2->rowPtr[i+1]-1);
	}
	m2->info.structure=SYMMETRIC;
	m2->val=NULL;

CPLM_END_TIME
CPLM_POP
	return ierr;

}


/**
 * \fn int CPLM_MatCSRUnsymStruct(CPLM_Mat_CSR_t *m1, CPLM_Mat_CSR_t *m2)
 * \brief Function which creates a CSR matrix and fills it from the original CSR matrix symmetrizing the structure
 * \param *m1 The input symmetric CSR matrix where the structure is for instancethe upper part
 * \param *m2 The output general CSR matrix
 */
/*Function symmetrizes a CPLM_Mat_CSR_t matrix*/
int CPLM_MatCSRUnsymStruct(CPLM_Mat_CSR_t *m1, CPLM_Mat_CSR_t *m2){
CPLM_PUSH
CPLM_BEGIN_TIME

  int err=0;

	if(m1->info.m != m1->info.n){
		fprintf(stderr,"matrix is not square\n");
		exit(1);
	}

  err = CPLM_MatCSRInit(m2,&(m1->info));CPLM_CHKERR(err);
  m2->info.M=m2->info.m;
  m2->info.N=m2->info.n;

	m2->rowPtr = (int*)calloc((m1->info.m+1),sizeof(int));

	int *ptr;
	ptr=(int*)calloc(m1->info.m,sizeof(int));

  //For each line
  for(int i=0;i<m1->info.m;i++){
    //For each column
	  for(int j=m1->rowPtr[i];j<m1->rowPtr[i+1];j++){
		  int tmp_col=m1->colInd[j];
		  if(tmp_col==i) continue;
	    ptr[tmp_col]++;
	  }
  }

  m2->info.lnnz=m2->info.nnz=m1->info.lnnz*2-m1->info.m;

  m2->colInd = (int*)malloc(m2->info.lnnz*sizeof(int));

  m2->val = (double*)malloc(m2->info.lnnz*sizeof(double));

	int offset=0;

	//compute the real number of values for the row i
	for(int i=0;i<m1->info.m;i++){
		m2->rowPtr[i+1]=m1->rowPtr[i+1]+ptr[i]+offset;
		offset+=ptr[i];
	}

  memset(ptr,0,m1->info.m*sizeof(int));

  for(int i=0;i<m1->info.m;i++){
	  //route the i'th row
	  for(int j=m1->rowPtr[i];j<m1->rowPtr[i+1];j++){
		  int tmp_col=m1->colInd[j];
		  if(tmp_col==i) continue;
		  m2->colInd[m2->rowPtr[tmp_col]+ptr[tmp_col]]=i;
		  m2->val[m2->rowPtr[tmp_col]+ptr[tmp_col]]=m1->val[j];

		  ptr[tmp_col]++;
	  }
	  memcpy(&m2->colInd[m2->rowPtr[i]+ptr[i]],&m1->colInd[m1->rowPtr[i]],(m1->rowPtr[i+1]-m1->rowPtr[i])*sizeof(int));
	  memcpy(&m2->val[m2->rowPtr[i]+ptr[i]],&m1->val[m1->rowPtr[i]],(m1->rowPtr[i+1]-m1->rowPtr[i])*sizeof(double));

	  ptr[i]+=(m1->rowPtr[i+1]-m1->rowPtr[i]);
  }

  free(ptr);

  for(int i=0;i<m2->info.m;i++){
		CPLM_quickSortWithValues(m2->colInd,m2->rowPtr[i],m2->rowPtr[i+1]-1,m2->val);
	}

CPLM_END_TIME
CPLM_POP
	return err;

}



/**
 * \fn void CPLM_MatCSRPrint2D(CPLM_Mat_CSR_t *m)
 * \brief Method which prints the CSR matrix into standard format
 * \param *matCSR The matrix which has to be printed
 */
/*Print original matrix */
void CPLM_MatCSRPrintPartial2D(CPLM_Mat_CSR_t *m)
{
	int zero=0;
	if(m->val==NULL)
  {
		for(int i=0;i<m->info.m;i++){
			printf("[ ");
			for(int j=m->rowPtr[i], indCol=0; indCol < m->info.n ;indCol++){
				if(j<m->rowPtr[i+1] && indCol==m->colInd[j]){
					printf("X ");
					j++;
				}
				else
					printf(". ");
			}
			printf("]\n");
		}
	}
  else
  {
		for(int i = 0; i < m->info.m; i++)
    {
			printf("[ ");
			//for each case of the line
			for(int j = m->rowPtr[i], indCol = 0; indCol < m->info.n ;indCol++)
      {
				//if this case is in colInd
				if(j < m->rowPtr[i+1] && indCol == m->colInd[j])
        {
					printf("%2.*f\t",_PRECISION,m->val[j]);
					j++;
				}
				else
					printf("%7d\t",zero);

        //Jump if needed to the end of the line
        if(indCol == PRINT_PARTIAL_N)
        {
          indCol = CPLM_MAX(m->info.n - PRINT_PARTIAL_N - 1, indCol);
          printf("...");
        }
			}
			printf("]\n");
      if(i == PRINT_PARTIAL_M)
      {
        i = CPLM_MAX(m->info.m - PRINT_PARTIAL_M - 1, i);
        printf("...\n");
      }
		}
  }
}


/**
 * \fn void CPLM_MatCSRPrint2D(CPLM_Mat_CSR_t *m)
 * \brief Method which prints the CSR matrix into standard format
 * \param *matCSR The matrix which has to be printed
 */
/*Print original matrix */
void CPLM_MatCSRPrint2D(CPLM_Mat_CSR_t *m)
{
	int zero=0;
	if(m->val==NULL)
  {
		for(int i=0;i<m->info.m;i++){
			printf("[ ");
			for(int j=m->rowPtr[i], indCol=0; indCol < m->info.n ;indCol++){
				if(j<m->rowPtr[i+1] && indCol==m->colInd[j]){
					printf("X ");
					j++;
				}
				else
					printf(". ");
			}
			printf("]\n");
		}
	}
  else
  {
		for(int i=0;i<m->info.m;i++){
			printf("[ ");
			//for each case of the line
			for(int j=m->rowPtr[i], indCol=0; indCol < m->info.n ;indCol++){
				//if this case is in colInd
				if(j<m->rowPtr[i+1] && indCol==m->colInd[j]){
					printf("%2.*f\t",_PRECISION,m->val[j]);
					j++;
				}
				else
					printf("%7d\t",zero);
			}
			printf("]\n");
		}
  }
}

















/*
 *
 * Author: Simplice Donfack
 *
 */

/**/
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


/*
 * Split the matrix in block column and fill the selected block column number with zeros,
 * Optimize the routine to avoid storing these zeros in the output matrix.
 * A_in:
 *     input: the input matrix
 * colCount:
 *     input: the global number of columns in each Block
 * numBlock:
 *     input: the number of the block to fill with zeros
 * B_out:
 *     output: the output matrix after removing the diag block
 */

int CPLM_MatCSRBlockColumnZerosFill(CPLM_Mat_CSR_t *A_in, int *colCount, int numBlock, CPLM_Mat_CSR_t *B_out){

  int i,j, m, lpos = 0, count = 0, ierr = 0;
  int *mwork;

  m = A_in->info.m;

  if(m<=0) return 0;

  /* Sum of the element before the selected block */
  for(i=0;i<numBlock;i++) lpos += colCount[i];

  if ( !(mwork  = (int *) malloc((m+1) * sizeof(int))) ) CPLM_Abort("Malloc fails for mwork[].");


  //First precompute the number of elements outside the colmun to remove
  count = 0;
  for(i=0;i<m;i++){
    for(j=A_in->rowPtr[i];j<A_in->rowPtr[i+1];j++){
      if(A_in->colInd[j]>=lpos && A_in->colInd[j]<lpos+colCount[numBlock]) continue;
      /* element outside the column to remove , count it */
      count ++;
    }
  }


  // Set the matrix infos
  CPLM_MatCSRSetInfo(B_out, A_in->info.m, A_in->info.n, count, A_in->info.m,  A_in->info.n, count, 1);
  ierr = CPLM_MatCSRMalloc(B_out); CPLM_CHKERR(ierr);

  // Fill the output matrix
  count = 0;
  for(i=0;i<m;i++){

    for(j=A_in->rowPtr[i];j<A_in->rowPtr[i+1];j++){

      if(A_in->colInd[j]>=lpos && A_in->colInd[j]<lpos+colCount[numBlock]) continue;

      /* element outside the column to remove , copy it */
      B_out->colInd[count] = A_in->colInd[j];
      B_out->val[count]    = A_in->val[j];
      count ++;

    }

    mwork[i+1] = count;
  }

  B_out->rowPtr[0] = 0;
  for(i=1;i<m+1;i++) B_out->rowPtr[i] = mwork[i];

  free(mwork);

  return ierr;
}

/*
 * 1D block row distirbution of the matrix. At the end, each proc has approximatively the same number of rows.
 *
 */
int CPLM_MatCSRBlockRowDistribute(CPLM_Mat_CSR_t *Asend, CPLM_Mat_CSR_t *Arecv, int *mcounts, int *moffsets, int root, MPI_Comm comm){

  int i, m, ierr = 0;
  int nbprocs, my_rank;

  MPI_Comm_size(comm, &nbprocs);
  MPI_Comm_rank(comm, &my_rank);

  //Broadcast the matrix size from the root to the other procs
  m = Asend->info.m;
  MPI_Bcast(&m, 1, MPI_INT, root, comm);
  //preAlps_int_printSynchronized(m, "m in rowDistribute", comm);

  // Split the number of rows among the processors
  for(i=0;i<nbprocs;i++){
    CPLM_nsplit(m, nbprocs, i, &mcounts[i], &moffsets[i]);
  }
  moffsets[nbprocs] = m;

  //distributes the matrix
  ierr = CPLM_MatCSRBlockRowScatterv(Asend, Arecv, moffsets, root, comm); CPLM_CHKERR(ierr);

  return ierr;
}
/*
 * 1D block rows gatherv of the matrix from the processors in the communicator.
 * The result is stored on processor 0.
 * ncounts: ncounts[i] = k means processor i has k rows.
 */
 /*
  * 1D block rows gather of the matrix from all the processors in the communicator .
  * Asend:
  *     input: the matrix to send
  * Arecv
  *     output: the matrix to assemble the block matrix received from all (relevant only on the root)
  * idxRowBegin:
  *     input: the global row indices of the distribution
  */
int CPLM_MatCSRBlockRowGatherv(CPLM_Mat_CSR_t *Asend, CPLM_Mat_CSR_t *Arecv,  int *idxRowBegin, int root, MPI_Comm comm){

  int nbprocs, my_rank;

  int *nxacounts=NULL, *nzcounts=NULL, *nzoffsets=NULL;

  int i, m = 0, n=0, j, nz = 0, pos, mloc, nzloc ;


  MPI_Comm_size(comm, &nbprocs);

  MPI_Comm_rank(comm, &my_rank);


  /* Determine my local number of rows*/
  mloc = idxRowBegin[my_rank+1]-idxRowBegin[my_rank];

  /* The root prepare the receiving matrix*/
  if(my_rank==root){

    /* Determine the global number of rows*/
    m = 0;

    for(i=0;i<nbprocs;i++){
      m+= (idxRowBegin[i+1]-idxRowBegin[i]);
    }

    n = Asend->info.n;

    /* The receive matrix size is unknown, we reallocate it. TODO: check the Arecv size and realloc only if needed*/
    if(Arecv->rowPtr!=NULL)   free(Arecv->rowPtr);
    if(Arecv->colInd!=NULL)   free(Arecv->colInd);
    if(Arecv->val!=NULL)      free(Arecv->val);


    if ( !(Arecv->rowPtr = (int *)   malloc((m+1)*sizeof(int))) ) CPLM_Abort("Malloc fails for xa[].");

    //buffer
    if ( !(nxacounts  = (int *) malloc((nbprocs+1)*sizeof(int))) ) CPLM_Abort("Malloc fails for nxacounts[].");
    if ( !(nzcounts  = (int *) malloc(nbprocs*sizeof(int))) ) CPLM_Abort("Malloc fails for nzcounts[].");

    /* Compute the number of elements to gather  */

    for(i=0;i<nbprocs;i++){
      nxacounts[i] = idxRowBegin[i+1]-idxRowBegin[i];
    }

  }


  /* Shift to take into account that the other processors will not send their first elements (which is rowPtr[0] = 0) */
  if(my_rank==root) {
    for(i=0;i<nbprocs+1;i++){
      idxRowBegin[i]++;
    }

    Arecv->rowPtr[0] = 0; //first element
  }

  /* Each process send mloc element to proc 0 (without the first element) */
  MPI_Gatherv(&Asend->rowPtr[1], mloc, MPI_INT, Arecv->rowPtr, nxacounts, idxRowBegin, MPI_INT, root, comm);



  /* Convert xa from local to global by adding the last element of each subset*/
  if(my_rank==root){

    for(i=1;i<nbprocs;i++){

      pos = idxRowBegin[i];

      for(j=0;j<nxacounts[i];j++){

        /*add the number of non zeros of the previous proc */
        Arecv->rowPtr[pos+j] = Arecv->rowPtr[pos+j] + Arecv->rowPtr[pos-1];

      }
    }

  }

  /* Restore idxRowBegin in the case the caller program needs it*/
  if(my_rank==root) {
    for(i=0;i<nbprocs+1;i++){

      idxRowBegin[i]--;
    }
  }

  /* Compute number of non zeros in each rows */

  if(my_rank==root){

    if ( !(nzoffsets = (int *) malloc((nbprocs+1)*sizeof(int))) ) CPLM_Abort("Malloc fails for nzoffsets[].");

    nzoffsets[0] = 0; nz = 0;
    for(i=0;i<nbprocs;i++){

      nzcounts[i] = Arecv->rowPtr[idxRowBegin[i+1]] - Arecv->rowPtr[idxRowBegin[i]];

      nzoffsets[i+1] = nzoffsets[i] + nzcounts[i];

      nz+=nzcounts[i];
    }

    if ( !(Arecv->colInd = (int *)   malloc((nz*sizeof(int)))) ) CPLM_Abort("Malloc fails for Arecv->colInd[].");
    if ( !(Arecv->val = (double *)   malloc((nz*sizeof(double)))) ) CPLM_Abort("Malloc fails for Arecv->val[].");
  }


  /* Gather ja and a */
  nzloc = Asend->rowPtr[mloc];

  MPI_Gatherv(Asend->colInd, nzloc, MPI_INT, Arecv->colInd, nzcounts, nzoffsets, MPI_INT, root, comm);

  MPI_Gatherv(Asend->val, nzloc, MPI_DOUBLE, Arecv->val, nzcounts, nzoffsets, MPI_DOUBLE, root, comm);

  /* Set the matrix infos */
  if(my_rank==root){
    CPLM_MatCSRSetInfo(Arecv, m, n, nz, m,  n, nz, 1);
  }

  if(my_rank==root){

    free(nxacounts);

    free(nzcounts);
    free(nzoffsets);
  }

  return 0;
}

/*
 * Gatherv a local matrix from each process and dump into a file
 *
 */
int CPLM_MatCSRBlockRowGathervDump(CPLM_Mat_CSR_t *locA, char *filename, int *idxRowBegin, int root, MPI_Comm comm){
  int nbprocs, my_rank;
  MPI_Comm_size(comm, &nbprocs);
  MPI_Comm_rank(comm, &my_rank);

  CPLM_Mat_CSR_t Awork = CPLM_MatCSRNULL();
  CPLM_MatCSRBlockRowGatherv(locA, &Awork, idxRowBegin, root, comm);

  if(my_rank==root) {
    printf("Dumping the matrix ...\n");
    CPLM_MatCSRSave(&Awork, filename);
    printf("Dumping the matrix ... done\n");
  }

  CPLM_MatCSRFree(&Awork);

  return 0;
}

 /*
  * 1D block rows distribution of the matrix to all the processors in the communicator.
  * The data are originally stored on the processor root. After this routine each processor i will have the row indexes from
  * idxRowBegin[i] to idxRowBegin[i+1] - 1 of the input matrix.
  *
  * Asend:
  *     input: the matrix to scatterv (relevant only on the root)
  * Arecv
  *     output: the matrix to store the block matrix received
  * idxRowBegin:
  *     input: the global row indices of the distribution
  */
int CPLM_MatCSRBlockRowScatterv(CPLM_Mat_CSR_t *Asend, CPLM_Mat_CSR_t *Arecv, int *idxRowBegin, int root, MPI_Comm comm){

   int nbprocs, my_rank;

   int *nzcounts=NULL, *nzoffsets=NULL, *nxacounts = NULL;
   int mloc, nzloc;

   //int *xa_ptr, *asub_ptr;
   //double *a_ptr;
   int i, n;


   MPI_Comm_size(comm, &nbprocs);

   MPI_Comm_rank(comm, &my_rank);


   /* Compute the displacements for rowPtr */
   if(my_rank==root){

     if ( !(nxacounts  = (int *) malloc((nbprocs+1)*sizeof(int))) ) CPLM_Abort("Malloc fails for nxacounts[].");
   }


   /* Determine my local number of rows*/
   mloc = idxRowBegin[my_rank+1]-idxRowBegin[my_rank];

   /* Broadcast the global number of rows. Only the root processos has it*/
   if(my_rank == root) n = Asend->info.n;
   MPI_Bcast(&n, 1, MPI_INT, root, comm);

   /* Compute the new number of columns per process*/

   if(my_rank==root){
     /* Compute the number of elements to send  */

     for(i=0;i<nbprocs;i++){

       nxacounts[i] = (idxRowBegin[i+1]-idxRowBegin[i])+1;  /* add the n+1-th element required for the CSR format */
     }

   }


   /* Allocate memory for rowPtr*/
   if(Arecv->rowPtr!=NULL) free(Arecv->rowPtr);

   if ( !(Arecv->rowPtr = (int *)   malloc((mloc+1)*sizeof(int))) ) CPLM_Abort("Malloc fails for Arecv->rowPtr[].");

   /* Distribute xa to each procs. Each proc has mloc+1 elements */

   MPI_Scatterv(Asend->rowPtr, nxacounts, idxRowBegin, MPI_INT, Arecv->rowPtr, mloc+1, MPI_INT, root, comm);

   /* Convert xa from global to local */
   for(i=mloc;i>=0;i--){

     Arecv->rowPtr[i] = Arecv->rowPtr[i] - Arecv->rowPtr[0];

   }

   /*
    * Distribute asub and a to each procs
    */


  nzloc = Arecv->rowPtr[mloc]; // - xa_ptr[0]


  /* Allocate memory for colInd and val */
    if(Arecv->colInd!=NULL) free(Arecv->colInd);
    if(Arecv->val!=NULL) free(Arecv->val);
    if ( !(Arecv->colInd = (int *)   malloc((nzloc*sizeof(int)))) ) CPLM_Abort("Malloc fails for Arecv->colInd[].");
    if ( !(Arecv->val = (double *)   malloc((nzloc*sizeof(double)))) ) CPLM_Abort("Malloc fails for Arecv->val[].");


   /* Compute number of non zeros in each rows and the displacement for nnz*/

   if(my_rank==root){
     if ( !(nzcounts  = (int *) malloc(nbprocs*sizeof(int))) ) CPLM_Abort("Malloc fails for nzcounts[].");
     if ( !(nzoffsets = (int *) malloc((nbprocs+1)*sizeof(int))) ) CPLM_Abort("Malloc fails for nzoffsets[].");
     nzoffsets[0] = 0;
     for(i=0;i<nbprocs;i++){

       nzcounts[i] = Asend->rowPtr[idxRowBegin[i+1]] - Asend->rowPtr[idxRowBegin[i]];
       nzoffsets[i+1] = nzoffsets[i] + nzcounts[i];
     }
   }

   /* Distribute colInd and val */
   MPI_Scatterv(Asend->colInd, nzcounts, nzoffsets, MPI_INT, Arecv->colInd, nzloc, MPI_INT, root, comm);

   MPI_Scatterv(Asend->val, nzcounts, nzoffsets, MPI_DOUBLE, Arecv->val, nzloc, MPI_DOUBLE, root, comm);

   /* Set the matrix infos */
   CPLM_MatCSRSetInfo(Arecv, mloc, n, nzloc, mloc,  n, nzloc, 1);

   if(my_rank==root){
     free(nxacounts);
     free(nzcounts);
     free(nzoffsets);
   }

   return 0;
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


/* Create a MatCSRNULL matrix, same as A = CPLM_MatCSRNULL() but for a matrix referenced as pointer. */
int CPLM_MatCSRCreateNULL(CPLM_Mat_CSR_t **A){

  int ierr = 0;

  CPLM_Info_t info = { .M=0, .N=0, .nnz=0, .m=0, .n=0, .lnnz=0, .blockSize=0, .format=FORMAT_CSR, .structure=UNSYMMETRIC };
  *A  = (CPLM_Mat_CSR_t *) malloc(sizeof(CPLM_Mat_CSR_t));

  if(!*A) return 1;
  (*A)->info = info;
  (*A)->rowPtr=NULL;
  (*A)->colInd=NULL;
  (*A)->val=NULL;

  return ierr;
}


/* Broadcast the matrix dimension from the root to the other procs*/
int CPLM_MatCSRDimensions_Bcast(CPLM_Mat_CSR_t *A, int root, int *m, int *n, int *nnz, MPI_Comm comm){

  int ierr = 0, nbprocs, my_rank, matrixDim[3];

  MPI_Comm_size(comm, &nbprocs);
  MPI_Comm_rank(comm, &my_rank);

  /* Prepare the matrix dimensions for the broadcast */
  if(my_rank==root){
    matrixDim[0] = A->info.m;
    matrixDim[1] = A->info.n;
    matrixDim[2] = A->info.nnz;
  }

  /* Broadcast the global matrix dimension among all procs */
  ierr = MPI_Bcast(&matrixDim, 3, MPI_INT, root, comm);
  *m   = matrixDim[0];
  *n   = matrixDim[1];
  *nnz = matrixDim[2];

  return ierr;
}

/*
 * Matrix matrix product, C := alpha*A*B + beta*C
 * where A is a CSR matrix, B and C is are dense Matrices stored in column major layout/
 */
int CPLM_MatCSRMatrixCSRDenseMult(CPLM_Mat_CSR_t *A, double alpha, double *B, int B_ncols, int ldB, double beta, double *C, int ldC){
  int ierr = 0;

  #ifdef USE_MKL
    int i;
    char matdescra[6] ={'G',' ',' ', 'F', ' ', ' '};
    //char matdescra[6] ={'G',' ',' ', 'C', ' ', ' '}; //'X'


    // B and C are stored as Colum major layout, so MKL assumes that the input matrix is 1-based indexed
    for (i = 0; i < A->info.m+1; i++) {
        A->rowPtr[i] ++;
    }
    for (i = 0; i < A->info.lnnz; i++) {
        A->colInd[i] ++;
	  }

    //printf("m:%d, b_ncols:%d, n:%d, alpha:%f, beta:%f\n", A->info.m, B_ncols, A->info.n, alpha, beta);
    mkl_dcsrmm("N", &A->info.m, &B_ncols, &A->info.n, &alpha, matdescra,
              A->val, A->colInd, A->rowPtr, &A->rowPtr[1], B, &ldB, &beta, C, &ldC);

    //restore the indexing
    for (i = 0; i < A->info.m+1; i++) {
        A->rowPtr[i] --;
    }
    for (i = 0; i < A->info.lnnz; i++) {
        A->colInd[i] --;
	  }

  #else
   CPLM_Abort("Only MKL is supported so far for the Matrix matrix product. Please compile with MKL\n");
  #endif

  return ierr;
}

 /*
  * Matrix vector product, y := alpha*A*x + beta*y
  */
int CPLM_MatCSRMatrixVector(CPLM_Mat_CSR_t *A, double alpha, double *x, double beta, double *y){


    #ifdef USE_MKL

     //char matdescra[6] = {'G', '\0', '\0', 'C', '\0', '\0'};
     //char matdescra[] = "G**C**";
     char matdescra[6] = {'G', ' ', ' ', 'C', ' ', ' '};
     mkl_dcsrmv("N", &A->info.m, &A->info.n, &alpha, matdescra, A->val, A->colInd, A->rowPtr, &A->rowPtr[1], x, &beta, y);
    #else
     //CPLM_Abort("Only MKL is supported so far for the Matrix vector product\n");
     int i,j ;
     for (i=0; i<A->info.m; i++ ){
       //y[i]=0;
       for (j=A->rowPtr[i]; j<A->rowPtr[i+1]; j++) {
        y[i] = beta*y[i] + alpha * A->val[j]*x[A->colInd[j]];
       }
     }

     //return 1;
    #endif

    return 0;
}


/*
 * Perform an ordering of a matrix using parMetis
 *
 */

int CPLM_MatCSROrderingND(MPI_Comm comm, CPLM_Mat_CSR_t *A, int *vtdist, int *order, int *sizes){

  int err = 0;

#ifdef USE_PARMETIS

  idx_t options[METIS_NOPTIONS];
  idx_t numflag = 0; /*C-style numbering*/

  options[0] = 0;
  options[1] = 0;
  options[2] = 42; /* Fixed Seed for reproducibility */

#if 1

  //Silent parmetis compilation warning
  int i, nbprocs, nparts;
  idx_t *pmetis_vtdist, *pmetis_order, *pmetis_sizes;
  idx_t *pmetis_rowPtr, *pmetis_colInd;

  MPI_Comm_size(comm, &nbprocs);

  nparts = 1;
  while(nparts<nbprocs){
    nparts = 2*nparts+1;
  }

  if ( !(pmetis_vtdist = (idx_t *)   malloc(((nbprocs+1)*sizeof(idx_t)))) ) CPLM_Abort("Malloc fails for pmetis_order[].");
  if ( !(pmetis_order  = (idx_t *)   malloc((A->info.m*sizeof(idx_t)))) ) CPLM_Abort("Malloc fails for pmetis_order[].");
  if ( !(pmetis_sizes  = (idx_t *)   malloc((nparts*sizeof(idx_t)))) ) CPLM_Abort("Malloc fails for pmetis_sizes[].");
  if ( !(pmetis_rowPtr = (idx_t *)   malloc(((A->info.m+1)*sizeof(idx_t)))) ) CPLM_Abort("Malloc fails for pmetis_order[].");
  if ( !(pmetis_colInd = (idx_t *)   malloc((A->info.lnnz*sizeof(idx_t)))) ) CPLM_Abort("Malloc fails for pmetis_sizes[].");

  //convert to idx_t
  if(sizeof(idx_t)!=sizeof(int)) printf("[preAlps parMetis] *** Type missmatch: sizeof(idx_t):%lu, sizeof(int):%lu. Please compile parMetis with appropriate idx_t 32;\n", sizeof(idx_t), sizeof(int));

  for(i=0;i<nbprocs+1;i++)    pmetis_vtdist[i] = (idx_t) vtdist[i];
  for(i=0;i<A->info.m+1;i++)  pmetis_rowPtr[i] = (idx_t) A->rowPtr[i];
  for(i=0;i<A->info.lnnz;i++) pmetis_colInd[i] = (idx_t) A->colInd[i];

  //call parMetis
  err = ParMETIS_V3_NodeND (pmetis_vtdist, pmetis_rowPtr, pmetis_colInd, &numflag, options, pmetis_order, pmetis_sizes, &comm);

  //copy back the result

  for(i=0;i<A->info.m;i++)    order[i]  = pmetis_order[i];
  for(i=0;i<nparts;i++)       sizes[i]  = pmetis_sizes[i];


  free(pmetis_rowPtr);
  free(pmetis_colInd);
  free(pmetis_vtdist);
  free(pmetis_order);
  free(pmetis_sizes);
#else
  err = ParMETIS_V3_NodeND (vtdist, A->rowPtr, A->colInd, &numflag, options, order, sizes, &comm);
#endif

  if(err!=METIS_OK) {printf("METIS returned error:%d\n", err); CPLM_Abort("ParMetis Ordering Failed.");}

#else
  CPLM_Abort("No other NodeND partitioning tool is supported at the moment. Please Rebuild with ParMetis !");
#endif

  return err;
}


/*
 * Partition a matrix using parMetis
 * part_loc:
 *     output: part_loc[i]=k means rows i belongs to subdomain k
 */

int CPLM_MatCSRPartitioningKway(MPI_Comm comm, CPLM_Mat_CSR_t *A, int *vtdist, int nparts, int *partloc){

  int i, err = 0;

#ifdef USE_PARMETIS

  idx_t options[METIS_NOPTIONS];
  idx_t numflag = 0; /*C-style numbering*/
  idx_t wgtflag = 0; /*No weights*/
  idx_t ncon = 1;
  idx_t edgecut = 0;
  idx_t pmetis_nparts = nparts;

  float *tpwgts;
  float *ubvec;

  if ( !(tpwgts = (float *)   malloc((nparts*ncon*sizeof(float)))) ) CPLM_Abort("Malloc fails for tpwgts[].");
  if ( !(ubvec = (float *)    malloc((ncon*sizeof(float)))) ) CPLM_Abort("Malloc fails for ubvec[].");

  options[0] = 0;
  options[1] = 0;
  options[2] = 42; /* Fixed Seed for reproducibility */

  for(i=0;i<nparts*ncon;i++) tpwgts[i] = 1.0/(real_t)nparts;
  for(i=0;i<ncon;i++) ubvec[i] =  1.05;

  //Silent parmetis compilation warning
  int nbprocs;
  MPI_Comm_size(comm, &nbprocs);
  idx_t *pmetis_vtdist, *pmetis_partloc;
  idx_t *pmetis_rowPtr, *pmetis_colInd;

  if ( !(pmetis_vtdist  = (idx_t *)   malloc(((nbprocs+1)*sizeof(idx_t)))) ) CPLM_Abort("Malloc fails for pmetis_order[].");
  if ( !(pmetis_partloc = (idx_t *)   malloc((A->info.m*sizeof(idx_t)))) ) CPLM_Abort("Malloc fails for pmetis_order[].");
  if ( !(pmetis_rowPtr  = (idx_t *)   malloc(((A->info.m+1)*sizeof(idx_t)))) ) CPLM_Abort("Malloc fails for pmetis_order[].");
  if ( !(pmetis_colInd  = (idx_t *)   malloc((A->info.lnnz*sizeof(idx_t)))) ) CPLM_Abort("Malloc fails for pmetis_sizes[].");

  //convert to idx_t
  for(i=0;i<nbprocs+1;i++)    pmetis_vtdist[i] = vtdist[i];
  for(i=0;i<A->info.m+1;i++)  pmetis_rowPtr[i] = A->rowPtr[i];
  for(i=0;i<A->info.lnnz;i++) pmetis_colInd[i] = A->colInd[i];

  //Call parmetis
  err = ParMETIS_V3_PartKway(pmetis_vtdist, pmetis_rowPtr, pmetis_colInd, NULL, NULL,
      &wgtflag, &numflag, &ncon, &pmetis_nparts, tpwgts, ubvec, options, &edgecut,
        pmetis_partloc, &comm);

  //copy back the result
  for(i=0;i<A->info.m;i++)    partloc[i]  = pmetis_partloc[i];

  free(pmetis_rowPtr);
  free(pmetis_colInd);
  free(pmetis_vtdist);
  free(pmetis_partloc);


  if(err!=METIS_OK) {printf("METIS returned error:%d\n", err); CPLM_Abort("ParMetis Failed.");}

  free(tpwgts);
  free(ubvec);

#else
  CPLM_Abort("No other Kway partitioning tool is supported at the moment. Please Rebuild with ParMetis !");
#endif
  return err;
}


/*
 * Print a CSR matrix as coordinate triplet (i,j, val)
 * Work only in debug mode
 */
void CPLM_MatCSRPrintCoords(CPLM_Mat_CSR_t *A, char *s){
#ifdef DEBUG
  int i,j;
  #ifdef PRINT_MOD
   int mark_i = 0, mark_j = 0;
  #endif
  if(s) printf("%s\n", s);

  for (i=0; i<A->info.m; i++){
    #ifdef PRINT_MOD
      //print only the borders and some values of the vector
      if((i>PRINT_DEFAULT_HEADCOUNT) && (i<A->info.m-1-PRINT_DEFAULT_HEADCOUNT) && (i%PRINT_MOD!=0)) {
        if(mark_i==0) {printf("... ... ...\n"); mark_i=1;} //prevent multiple print of "..."
        continue;
      }

      mark_i = 0;
      mark_j = 0;
    #endif

    for (j=A->rowPtr[i]; j<A->rowPtr[i+1]; j++){
      #ifdef PRINT_MOD
        //print only the borders and some values of the vector
        if((j>PRINT_DEFAULT_HEADCOUNT) && (j<A->rowPtr[i+1]-1-PRINT_DEFAULT_HEADCOUNT) && (A->colInd[j]%PRINT_MOD!=0)) {
          if(mark_j==0) {printf("... ... ...\n"); mark_j=1;} //prevent multiple print of "..."
          continue;
        }
        mark_j = 0;
      #endif
      printf("%d %d %20.19g\n", i, A->colInd[j], A->val[j]);
    }
  }
#endif
}



/* Only one process print its matrix, forces synchronisation between all the procs in the communicator*/
void CPLM_MatCSRPrintSingleCoords(CPLM_Mat_CSR_t *A, MPI_Comm comm, int root, char *varname, char *s){
#ifdef DEBUG
  int nbprocs, my_rank;

  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &nbprocs);

  if(my_rank==root) CPLM_MatCSRPrintCoords(A, s);

  MPI_Barrier(comm);
#endif
}

/*
 * Each processor print the matrix it has as coordinate triplet (i,j, val)
 * Work only in debug (-DDEBUG) mode
 * A:
 *    The matrix to print
 */

void CPLM_MatCSRPrintSynchronizedCoords (CPLM_Mat_CSR_t *A, MPI_Comm comm, char *varname, char *s){
#ifdef DEBUG
  int i;

  CPLM_Mat_CSR_t Abuffer = CPLM_MatCSRNULL();
  int my_rank, comm_size;

  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &comm_size);

  if(my_rank ==0){

    printf("[%d] %s\n", 0, s);

    CPLM_MatCSRPrintCoords(A, NULL);

    for(i = 1; i < comm_size; i++) {

      /*Receive a matrix*/
      CPLM_MatCSRRecv(&Abuffer, i, comm);

      printf("[%d] %s\n", i, s);
      CPLM_MatCSRPrintCoords(&Abuffer, NULL);
    }
    printf("\n");

    CPLM_MatCSRFree(&Abuffer);
  }
  else{
    CPLM_MatCSRSend(A, 0, comm);
  }

  MPI_Barrier(comm);

#endif
}

/*
 * Merge the rows of two matrices (one on top of another)
 */
int CPLM_MatCSRRowsMerge(CPLM_Mat_CSR_t *A_in, CPLM_Mat_CSR_t *B_in, CPLM_Mat_CSR_t *C_out){

  int ierr=0, i, j, C_m, C_nnz, count;

  //new number of rows and nnz
  C_m =  A_in->info.m + B_in->info.m;
  C_nnz = A_in->info.lnnz + B_in->info.lnnz;


  // Set the matrix infos
  CPLM_MatCSRSetInfo(C_out, C_m, A_in->info.n, C_nnz, C_m,  A_in->info.n, C_nnz, 1);

  ierr = CPLM_MatCSRMalloc(C_out); CPLM_CHKERR(ierr);

  //fill the matrix
  count = 0;
  C_out->rowPtr[0] = 0;

  //copy A
  for (i = 0; i < A_in->info.m; i++){
    for (j = A_in->rowPtr[i]; j < A_in->rowPtr[i+1]; j++) {
        C_out->colInd[count] =   A_in->colInd[j];
        C_out->val[count]    =   A_in->val[j];
        count++;
    }
    C_out->rowPtr[i+1] = count;
  }

  //Copy B
  for (i = 0; i < B_in->info.m; i++){
    //copy the columns of B
    for (j = B_in->rowPtr[i]; j < B_in->rowPtr[i+1]; j++) {
        C_out->colInd[count] =   B_in->colInd[j];
        C_out->val[count]    =   B_in->val[j];
        count++;
    }
    C_out->rowPtr[A_in->info.m+i+1] = count;
  }

  return ierr;
}
/*
 *
 * Scale a scaling vectors R and C, and scale the matrix by computing A1 = R * A * C
 * A:
 *     input: the matrix to scale
 * R:
 *     output: a vector with the same size as the number of rows of the matrix
 * C:
 *     output: a vector with the same size as the number of columns of the matrix
 */

int CPLM_MatCSRSymRACScaling(CPLM_Mat_CSR_t *A, double *R, double *C){

  int i, j;
  double rcmin, rcmax;

  /* Get machine constants. */
  //double smlnum = dlamch_("S");
  //double bignum = 1. / smlnum;


  /* Find the maximum element in each row. */

  //for (i = 0; i < A->info.m; ++i) R[i] = 0.;
  for (i = 0; i < A->info.m; i++){
      R[i] = 0.0;
      for (j = A->rowPtr[i]; j < A->rowPtr[i+1]; j++) {
          R[i] = CPLM_MAX( R[i], fabs(A->val[j]) );
      }
  }

  /* Find the maximum and minimum scale factors. */
  rcmin = R[0];
  rcmax = R[0];
  for (i = 1; i < A->info.m; ++i) {
      rcmax = CPLM_MAX(rcmax, R[i]);
      rcmin = CPLM_MIN(rcmin, R[i]);
  }

#ifdef DEBUG
  printf("ROW: rcmin:%e, rcmax:%e\n", rcmin, rcmax);
#endif


  //*amax = rcmax;

  if (rcmin == 0.) {
    CPLM_Abort("Impossible to scale the matrix, rcmin=0");

  } else {
      /* Invert the scale factors. */
      for (i = 0; i < A->info.m; i++){
          //R[i] = 1. / MIN( MAX( R[i], smlnum ), bignum );
          R[i] = sqrt(1.0 / R[i]);
      }
      /* Compute ROWCND = min(R(I)) / max(R(I)) */
      //*rowcnd = MAX( rcmin, smlnum ) / MIN( rcmax, bignum );
  }

#if 0

  /* Find the maximum element in each col. */
  for (j = 0; j < A->info.n; ++j) C[j] = 0.;

  /* Find the maximum element in each column, assuming the row
     scalings computed above. */
  for (j = 0; j < A->info.m; ++j){

      for (i = A->rowPtr[j]; i < A->rowPtr[j+1]; ++i) {
          C[j] = MAX( C[j], fabs(A->val[i]) * R[A->colInd[i]] );
      }
  }

  /* Find the maximum and minimum scale factors. */
  rcmin = C[0];
  rcmax = C[0];
  for (j = 1; j < A->info.n; ++j) {
      rcmax = CPLM_MAX(rcmax, C[j]);
      rcmin = CPLM_MIN(rcmin, C[j]);
  }

  if (rcmin == 0.) {
    CPLM_Abort("Impossible to scale the matrix, rcmin=0");
  } else {
      /* Invert the scale factors. */
      for (j = 0; j < A->info.n; ++j)
          C[j] = 1. / MIN( MAX( C[j], smlnum ), bignum);
      /* Compute COLCND = min(C(J)) / max(C(J)) */
    //  *colcnd = MAX( rcmin, smlnum ) / MIN( rcmax, bignum );
  }
#else
  //Assume the matrix is symmetric
  for (i = 0; i < A->info.m; i++) C[i] = R[i];
#endif

  /* Row and column scaling */
  for (i = 0; i < A->info.m; i++) {
      //cj = C[i];
      for (j = A->rowPtr[i]; j < A->rowPtr[i+1]; j++) {
        A->val[j] = R[i] * A->val[j] * C[A->colInd[j]];
      }
  }

  return 0;
}


/* Transpose a matrix */
int CPLM_MatCSRTranspose(CPLM_Mat_CSR_t *A_in, CPLM_Mat_CSR_t *B_out){

  int ierr = 0;
  int irow, jcol, jpos;
  int *work;
  int *xa = A_in->rowPtr, *asub = A_in->colInd;
  double *a = A_in->val;


  B_out->info = A_in->info;
  B_out->info.m = A_in->info.n;
  B_out->info.n = A_in->info.m;

  if(A_in->info.lnnz==0) return ierr; //Quick return

  ierr  = CPLM_MatCSRMalloc(B_out); CPLM_CHKERR(ierr);

  /* Allocate workspace */

  work    = (int*) malloc( (A_in->info.n+1)   * sizeof(int));

  if(!work) CPLM_Abort("Malloc failed for work");


  for(jcol=0;jcol<A_in->info.n+1;jcol++){
    work[jcol] = 0;
  }

  /* Compute the number of nnz per columns in A */
  for (irow=0; irow<A_in->info.m; irow++){
    for (jcol=xa[irow]; jcol<xa[irow+1]; jcol++) {
      work[asub[jcol]]++;
    }
  }

  /* Compute the index of each row of B*/

  B_out->rowPtr[0] = 0;
  for(irow=0;irow<B_out->info.m;irow++){
    B_out->rowPtr[irow+1] = B_out->rowPtr[irow] + work[irow];

    work[irow] = B_out->rowPtr[irow]; /* reused to store the first element of row irow. used for inserting the elements in the next step*/
  }

  /* Fill the matrix */

  for (irow=0; irow<A_in->info.m; irow++){
    for (jcol=xa[irow]; jcol<xa[irow+1]; jcol++){

      /* insert (irow, asub[jcol]) the element in column asub[jcol] */

      jpos = work[asub[jcol]];
      B_out->colInd[jpos] = irow;
      B_out->val[jpos] = a[jcol];
      work[asub[jcol]]++;

    }
  }

  /* Free memory*/
  free(work);

  return ierr;

}
