/*
============================================================================
Name        : preAlps_ivector.c
Author      : Simplice Donfack, Sebastien Cayrols, Olivier Tissot,
Version     : 0.1
Description : Ivector class from CPaLAMeM. This file is provided for compatibility
and debugging purpose only, it will be removed later.
Date        : jun 13, 2018
============================================================================
*/
#ifndef PREALPS_IVECTOR_H
#define PREALPS_IVECTOR_H

#include<mpi.h>
/**
 *	\struct CPLM_IVector_t
 *	\brief Structure which represents a CPLM_IVector_t with its values and its size
 *
 */
/*Structure utilized to store an array of values and a length*/
typedef struct{
	int *val;
	int nval;
  int size;
} CPLM_IVector_t;

#define IVECTOR_MAXVAL_PRINT 10

#define CPLM_IVectorNULL() { .nval=0, .size=0, .val=NULL }

#define CPLM_IVectorPrintf(_msg,_v)  { printf("%s of %d/%d : ",\
    (_msg),(_v)->nval,(_v)->size);   \
    CPLM_IVectorPrintPartial((_v)); }

#define handleMRealloc(v_,l_) ((v_)->val == NULL) ?\
    CPLM_IVectorMalloc((v_), (l_)) : \
      ((v_)->size < (l_)) ? CPLM_IVectorRealloc((v_), (l_)) : ierr

/**
 * \fn bCast_Vect(int *vec, int length, MPI_Comm comm, int root)
 * \brief Method which sends a CPLM_IVector_t to all processes
 * \param *vec The value array of the IVector
 * \param length The size of the IVector
 * \param comm The communicator for MPI
 * \param root The rank of the process which wants to send the IVector
 */
/*Function sends a CPLM_IVector_t to send_to*/
int CPLM_IVectorBcast(CPLM_IVector_t *v, MPI_Comm comm, int root);
int CPLM_IVectorCalloc(CPLM_IVector_t *v_out, int length);
int CPLM_IVectorCreateFromPtr(CPLM_IVector_t *v, int length, int *val);
void CPLM_IVectorFree(CPLM_IVector_t *v_io);
//ASSUMING v is sorted
int CPLM_IVectorGetPos(CPLM_IVector_t *v, int a, int *pos);
/**
 * \fn CPLM_IVector_t invertIVector(CPLM_IVector_t *v,int size)
 * \brief Function which inverts a IVector
 * \param *v The IVector
 * \return The inverted IVector
 */
/*Function which inverts a IVector*/
int CPLM_IVectorInvert(CPLM_IVector_t *v_in, CPLM_IVector_t *v_out);
int CPLM_IVectorMalloc(CPLM_IVector_t   *v, int length);
/**
 * \fn void CPLM_IVectorPrintPartial(CPLM_IVector_t *v)
 * \brief This method prints a CPLM_IVector_t
 * \param *v The CPLM_IVector_t to print
 */
/*Function prints a CPLM_IVector_t*/
void CPLM_IVectorPrint(CPLM_IVector_t *v);
/**
 * \fn void CPLM_IVectorPrintPartial(CPLM_IVector_t *v)
 * \brief Method which prints partially a CPLM_IVector_t when the number of
 * values is greater than IVECTOR_MAXVAL_PRINT macro
 * and otherwise the IVectorPrint routine is called instead
 * \param *v The CPLM_IVector_t to print
 */
/*Function prints a CPLM_IVector_t*/
void CPLM_IVectorPrintPartial(CPLM_IVector_t *v);

int CPLM_IVectorRealloc( CPLM_IVector_t *v_io, int length);

/**
  *
  *
  * Computation complexity : 2 * number of lines in fileName
  * Memory complexity      : length of tmp + size of CPLM_IVector_t * size of int
**/
int CPLM_IVectorLoad(const char *fileName, CPLM_IVector_t *buf, int size);


//compute the sum of all elements of v
int CPLM_IVectorReduce(CPLM_IVector_t *v, int *sum);

/**
  *
  *
  * Computation complexity : size of the IVector
**/
int CPLM_IVectorSave(CPLM_IVector_t *v,const char *fileName, const char *header);



int CPLM_IVectorSum(CPLM_IVector_t *u_in, CPLM_IVector_t *v_in, CPLM_IVector_t *w_out, int op);

#endif
