/*
* This file contains functions used for tracking symbolic pointer in a workspace
*
* Authors : Sebastien Cayrols
* Email   : sebastien.cayrols@[(gmail.com) | (inria.fr)]
*/
#include <stdlib.h>
#include <cplm_utils.h>
#include <cplm_v0_timing.h>
#include <cplm_v0_metis_utils.h>
#include <cplm_v0_ivector.h>

/**
 * \fn int CPLM_getIntPermArray(idx_t nb_parts, idx_t size_parts, idx_t *parts, CPLM_IVector_t *perm)
 * \brief Function which returns the integer perm array from parts array given by METIS_PartGraphKway
 * \param nb_parts The number of parts for METIS_PartGraphKway
 * \param size_parts The size of the array parts
 * \param *parts The array given by METIS_PartGraphKway
 * \return The permutation array
 */
/*Function which returns the integer perm array from parts array given by METIS_PartGraphKway*/
int CPLM_getIntPermArray(idx_t npart, idx_t size_parts, idx_t *parts, CPLM_IVector_t *perm)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  int ierr = 0;
  ierr = CPLM_IVectorMalloc(perm,(int)size_parts);CPLM_CHKERR(ierr);

	int	current_row = 0;
	//[caution], if int 64 loop has to be changed
	for(int i = 0; i < npart; i++)
  {
		for(int j = 0; j < size_parts; j++)
    {
			if(parts[j] == i)
				perm->val[current_row++] = j;
    }
  }
CPLM_END_TIME
CPLM_POP
	return ierr;
}





/**
 * \fn int CPLM_getIntIPermArray(idx_t size_perm, idx_t *perm)
 * \brief Function which returns the inverted integer perm array from perm array
 * \param size_perm The size of the array parts
 * \param *perm The integer array given by METIS_PartGraphKway
 * \return The inverted permutation array
 */
/*Function which returns the inverted integer perm array from perm array */
int CPLM_getIntIPermArray(idx_t size_perm, CPLM_IVector_t *perm, CPLM_IVector_t *iperm)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  int ierr = 0;

  ierr = CPLM_IVectorMalloc(iperm,size_perm);CPLM_CHKERR(ierr);
	//[caution], if int 64 loop has to be changed
	for(int j = 0; j < perm->nval; j++)
  {
		iperm->val[perm->val[j]] = j;
  }

CPLM_END_TIME
CPLM_POP
	return ierr;
}





/**
 * \fn idx_t* getPermArray(idx_t nb_parts, idx_t size_parts, idx_t *parts)
 * \brief Function which returns the perm array from parts array given by METIS_PartGraphKway
 * \param nb_parts The number of parts for METIS_PartGraphKway
 * \param size_parts The size of the array parts
 * \param *parts The array given by METIS_PartGraphKway
 * \return The permutation array
 */
/*Function which returns the perm array from parts array given by METIS_PartGraphKway*/
idx_t* getPermArray(idx_t nb_parts, idx_t size_parts, idx_t *parts)
{
CPLM_PUSH
CPLM_BEGIN_TIME

	idx_t *perm = NULL;
	idx_t	current_row = 0;

  perm = (idx_t*) malloc(size_parts*sizeof(idx_t));

  CPLM_ASSERT(perm != NULL);

	//[caution], if int 64 loop has to be changed
	for(int i = 0; i < nb_parts; i++)
  {
		for(int j = 0; j < size_parts; j++)
    {
			if(parts[j] == i)
				perm[current_row++] = j;
    }
  }

CPLM_END_TIME
CPLM_POP
	return perm;
}





/**
 * \fn idx_t* getPermArray(idx_t nb_parts, idx_t size_parts, idx_t *parts)
 * \brief Function which returns the inverted perm array from perm array
 * \param size_perm The size of the array parts
 * \param *perm The array given by METIS_PartGraphKway
 * \return The inverted permutation array
 */
/*Function which returns the inverted perm array from perm array */
int CPLM_getIPermArray(idx_t size_perm, idx_t *perm, idx_t **iperm)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  CPLM_ASSERT(*iperm == NULL);

	*iperm = (idx_t*) malloc(size_perm*sizeof(idx_t));
  CPLM_ASSERT(*iperm != NULL);

	//[caution], if int 64 loop has to be changed
	for(int j = 0; j < size_perm; j++)
  {
		(*iperm)[perm[j]] = j;
  }

CPLM_END_TIME
CPLM_POP
	return (*iperm != NULL);
}





/**
 * \fn int CPLM_getNumberOfRowByParts(int size_parts, idx_t *parts, int npart, CPLM_IVector_t *rowCount)
 * \brief Function which returns an array of number of row for each num_block belong to _NBPARTS
 * \param size_parts    The size of the array parts
 * \param *parts        The array given by METIS_PartGraphKway
 * \param npart         The number of parts for METIS_PartGraphKway
 * \param *rowCount     The returned vector of the number of rows by part
 * \return error if necessary
 */
//function which returns an array of number of row for each num_block belong to _NBPARTS
int CPLM_getNumberOfRowByParts(int size_parts,
    idx_t *parts,
    int npart,
    CPLM_IVector_t *rowCount)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  int ierr = 0;
  //allocation with one more cell to store last interval
  ierr = CPLM_IVectorCalloc(rowCount, npart + 1);CPLM_CHKERR(ierr);
	for(int i = 0; i < size_parts; i++)
  {
		rowCount->val[parts[i]]++;
  }

CPLM_END_TIME
CPLM_POP
	return ierr;
}





/**
 * \fn int* getBlockPositions(int size_parts, idx_t *parts)
 * \brief Function which returns an array of internal for each num_block belong to _NBPARTS
 * \param size_parts The size of the array parts
 * \param *parts The array given by METIS_PartGraphKway
 * \param nbparts The number of parts for METIS_PartGraphKway
 * \return The internal array
 */
//function which returns an array of interval for each num_block belong to _NBPARTS
int CPLM_getBlockPosition(int size_parts, idx_t *parts, int npart, CPLM_IVector_t *pos)
{
CPLM_PUSH
CPLM_BEGIN_TIME

	CPLM_IVector_t tmp = CPLM_IVectorNULL();
  int sum   = 0;
  int ierr  = 0;

	ierr = CPLM_getNumberOfRowByParts(size_parts,parts,npart,&tmp);CPLM_CHKERR(ierr);

  //allocation with one more cell to store last interval
  ierr = CPLM_IVectorCalloc(pos,npart+1);CPLM_CHKERR(ierr);

	for(int i = 0; i < pos->nval; i++)
  {
		pos->val[i] =   sum;
		sum         +=  tmp.val[i];
	}

	CPLM_IVectorFree(&tmp);

CPLM_END_TIME
CPLM_POP
	return ierr;
}
