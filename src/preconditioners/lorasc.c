/*
============================================================================
Name        : lorasc.c
Author      : Simplice Donfack
Version     : 0.1
Description : Preconditioner based on Schur complement
Date        : Sept 20, 2017
============================================================================
*/
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <metis_interface.h>
#include "preAlps_utils.h"
#include "lorasc.h"

/*Allocate workspace for the preconditioner*/
int Lorasc_alloc(Lorasc_t **lorasc){

  *lorasc = (Lorasc_t*) malloc(sizeof(Lorasc_t));

  if(*lorasc!=NULL){
    (*lorasc)->eigvalues=NULL;
  }

  return (*lorasc==NULL);
}

/*
 * Build the preconditioner
 * lorasc:
 *     input: the preconditioner object to construct
 * A:
 *     input: the input matrix on processor 0
 * locAP:
 *     output: the local permuted matrix on each proc after the preconditioner is built
 *
*/
int Lorasc_build(Lorasc_t *lorasc, Mat_CSR_t *A, Mat_CSR_t *locAP, MPI_Comm comm){

  /* Lorasc preconditioner build placeholder */
}

/*Destroy the preconditioner*/
int Lorasc_destroy(Lorasc_t **lorasc){

  if((*lorasc)->eigvalues!=NULL) free((*lorasc)->eigvalues);
  if(*lorasc!=NULL) free(*lorasc);

  return 0;
}
