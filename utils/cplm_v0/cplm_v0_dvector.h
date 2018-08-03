/*
============================================================================
Name        : preAlps_ivector.c
Author      : Simplice Donfack, Sebastien Cayrols, Olivier Tissot,
Version     : 0.1
Description : Dvector class from CPaLAMeM. This file is provided for compatibility
and debugging purpose only, it will be removed later.
Date        : jun 13, 2018
============================================================================
*/
#ifndef PREALPS_CPLM_DVECTOR_H
#define PREALPS_CPLM_DVECTOR_H

#include <mpi.h>

typedef struct{
	double *val;
	int nval;
} CPLM_DVector_t;


#define CPLM_DVectorNULL() { .nval=0, .val=NULL }
int CPLM_DVectorMalloc(CPLM_DVector_t   *v_out,
                  int         length);
void CPLM_DVectorFree(CPLM_DVector_t   *v_io);

int CPLM_DVectorConstant(CPLM_DVector_t* v, double value);
int CPLM_DVectorCreateFromPtr(CPLM_DVector_t *v, int length, double *val);

/**
  *
  *
  * Computation complexity : size of the vector
  * Memory complexity      : None
**/
int CPLM_DVectorSave(CPLM_DVector_t *v,const char *fileName, const char *header);

/**
  *
  * \param size The size of the vector
  * Computation complexity : 2 * number of lines in fileName
  * Memory complexity      : length of tmp + size of CPLM_IVector_t * size of double
**/
int CPLM_DVectorLoad(const char *fileName, CPLM_DVector_t *buf, int size);

int CPLM_DVectorRealloc( CPLM_DVector_t   *v_io,
                    int         length);

int CPLM_DVectorAddSpace(CPLM_DVector_t *v_out, int length);

/**
 * \fn void send_DVect(double *vec, int length, int send_to, int tag, MPI_Comm comm)
 * \brief Method which sends a CPLM_IVector_t to a process
 * \param *vec The value array of the vector
 * \param length The size of the vector
 * \param send_to The number of the process which will receive the vector
 * \param tag The tag of the communication
 * \param comm The communicator for MPI
 */
/*Function sends a CPLM_IVector_t to send_to*/

int CPLM_DVectorSend(CPLM_DVector_t *v, int dest, int tag, MPI_Comm comm);

/**
 * \fn CPLM_DVector_t recv_DVect(int recv_from, int tag, MPI_Comm comm)
 * \brief Function which manage the reception of a CPLM_IVector_t and return it
 * \param recv_from The number of the process sending the vector
 * \param tag The tag of the communication
 * \param comm The communicator for MPI
 * \return The CPLM_IVector_t received
 */
/*Function returns a CPLM_IVector_t received from recv_from*/
int CPLM_DVectorRecv(CPLM_DVector_t *v, int source, int tag, MPI_Comm comm);

#endif
