/*
============================================================================
Name        : preAlps_preconditioner.c
Author      : Simplice Donfack
Version     : 0.1
Description : A generic preconditioner class that can be used by the solver
Date        : Sept 27, 2017
============================================================================
*/
#include <stdio.h>
#include <stdlib.h>
#include "preAlps_preconditioner.h"
#include "preAlps_utils.h"

#include "lorasc.h"
#include "presc.h"
#include <mat_csr.h>

/* Create a generic preconditioner object compatible with EcgSolver*/
int preAlps_PreconditionerCreate(PreAlps_preconditioner_t **precond, Prec_Type_t precond_type, void *data){

  if ( !(*precond  = (PreAlps_preconditioner_t *) malloc(sizeof(PreAlps_preconditioner_t))) ) preAlps_abort("Malloc fails for precond[].");

  (*precond)->type = precond_type;

  (*precond)->data = data;

  return 0;
}

/*Apply the preconditioner to a matrix A_in*/
int preAlps_PreconditionerMatApply(PreAlps_preconditioner_t *precond, CPLM_Mat_Dense_t* A_in, CPLM_Mat_Dense_t* B_out){

  int ierr = 0;

  if(precond->type==PREALPS_NOPREC){
    //Just copy the matrix
    CPLM_MatDenseCopy(A_in, B_out);
  } else if(precond->type==PREALPS_LORASC){

    /* Lorasc preconditioner */
    preAlps_Lorasc_t *lorascA = NULL;

    /* Get the preconditioner data */
    lorascA = (preAlps_Lorasc_t*) precond->data;

    /* Apply Lorasc preconditioner on the matrix A_in */
    preAlps_LorascMatApply(lorascA, A_in, B_out);

  }else if(precond->type==PREALPS_PRESC){

    /* Presc preconditioner */
    preAlps_Presc_t *prescA = NULL;

    /* Get the preconditioner data */
    prescA = (preAlps_Presc_t*) precond->data;

    /* Apply Lorasc preconditioner on the matrix A_in */
    preAlps_PrescMatApply(prescA, A_in, B_out);

  }else{
    preAlps_abort("Unknown preconditioner: %d", precond->type);
  }

  return ierr;
}

/* Destroy the generic preconditioner object */
int preAlps_PreconditionerDestroy(PreAlps_preconditioner_t **precond){

  if(*precond!=NULL) free(*precond);

  return 0;
}
