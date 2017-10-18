/*
============================================================================
Name        : preAlps_preconditioner.h
Author      : Simplice Donfack
Version     : 0.1
Description : A generic preconditioner class that can be used by the solver
Date        : Sept 27, 2017
============================================================================
*/
#ifndef PREALPS_PRECONDITIONER_H
#define PREALPS_PRECONDITIONER_H

#include <mat_dense.h>
#include "preAlps_preconditioner_struct.h"


/* Create a generic preconditioner object compatible with EcgSolver*/
int preAlps_PreconditionerCreate(PreAlps_preconditioner_t **precond, Prec_Type_t precond_type, void *data);


/* Destroy the generic preconditioner object */
int preAlps_PreconditionerDestroy(PreAlps_preconditioner_t **precond);

/*Apply the preconditioner to a matrix A_in*/
int preAlps_PreconditionerMatApply(PreAlps_preconditioner_t *precond, CPLM_Mat_Dense_t* A_in, CPLM_Mat_Dense_t* B_out);

#endif
