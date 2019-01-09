
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cplm_matcsr_core.h>
#include <cplm_timing.h>
#include <cplm_utils.h>
#include <cplm_QS.h>

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




int CPLM_MatCSRInit(CPLM_Mat_CSR_t *A_out, CPLM_Info_t *info)
{

  A_out->info      = *info;

	A_out->rowPtr    = NULL;
	A_out->colInd    = NULL;
	A_out->val       = NULL;

	return 0;
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
