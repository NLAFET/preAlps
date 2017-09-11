/******************************************************************************/
/* Author     : Olivier Tissot                                                */
/* Creation   : 2016/09/05                                                    */
/* Description: Definition of the preconditioner                              */
/******************************************************************************/


/******************************************************************************/
/*                                  INCLUDE                                   */
/******************************************************************************/
#ifndef PRECONDITIONER_H
#define PRECONDITIONER_H

/* STD */
#include <stdio.h>
#include <stdlib.h>
/* MPI */
#include <mpi.h>
/* MatCSR */
#include <cpalamem_macro.h>
#include <cpalamem_instrumentation.h>
#include <mat_csr.h>
#include <mat_dense.h>

/* Preconditioner */

/* From which side the preconditioner needs to be applied: NOPREC, LEFT or RIGHT */
typedef enum {
  LEFT_PREC,
  NO_PREC
} Precond_side_t;

/* Preconditioner type*/
typedef enum {
  PREALPS_NOPREC,       /* No preconditioner*/
  PREALPS_BLOCKJACOBI,  /* Block Jacobi preconditioner*/
  PREALPS_LORASC,       /* Lorasc */
  PREALPS_PRESC         /* Preconditioner based on the Schur-Complement */
} Precond_t;

/******************************************************************************/

/******************************************************************************/
/*                                    CODE                                    */
/******************************************************************************/
int PrecondBlockOperator(Precond_t precond_type, Mat_Dense_t* A_in, Mat_Dense_t* B_out);

/* Right preconditioner */
// User functions
int RightPrecondCreate(); // TODO
int RightPrecondApply();  // TODO
int RightPrecondFree();   // TODO
// ParBCG internals
int RightPrecondInitialize(); // TODO
int RightPrecondFinalize();   // TODO
/* Left preconditioner */
// User functions
int LeftPrecondCreate(); // TODO
int LeftPrecondApply();  // TODO
int LeftPrecondFree();   // TODO
// ParBCG internals
int LeftPrecondInitialize(); // TODO
int LeftPrecondFinalize();   // TODO

/******************************************************************************/

#endif
