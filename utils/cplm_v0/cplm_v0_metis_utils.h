/*
* This file contains functions used for tracking symbolic pointer in a workspace
*
* Authors : Sebastien Cayrols
* Email   : sebastien.cayrols@[(gmail.com) | (inria.fr)]
*/

#ifndef CPLM_V0_METIS_H
#define CPLM_V0_METIS_H

#include <metis.h>
#include <cplm_v0_ivector.h>

/*Function which returns the integer perm array from parts array given by METIS_PartGraphKway*/
int CPLM_getIntPermArray(idx_t nb_parts, idx_t size_parts, idx_t *parts, CPLM_IVector_t* perm);

/*Function which returns the inverted integer perm array from perm array */
int CPLM_getIntIPermArray(idx_t size_perm, CPLM_IVector_t *perm, CPLM_IVector_t *iperm);

/*Function which returns the perm array from parts array given by METIS_PartGraphKway*/
idx_t* getPermArray(idx_t nb_parts, idx_t size_parts, idx_t *parts);

/*Function which returns the inverse perm array from perm array */
int CPLM_getIPermArray(idx_t size_perm, idx_t *perm, idx_t **iperm);

//function which returns an array of number of row for each num_block belong to _NBPARTS
int CPLM_getNumberOfRowByParts(int size_parts, idx_t *parts, int npart, CPLM_IVector_t *rowCount);

//function which returns an array of interval for each num_block belong to _NBPARTS
int CPLM_getBlockPosition(int size_parts, idx_t *parts, int npart, CPLM_IVector_t *pos);


#endif
