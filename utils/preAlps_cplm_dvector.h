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

#endif
