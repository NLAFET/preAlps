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
/******************************************************************************/

/******************************************************************************/
/*                                    CODE                                    */
/******************************************************************************/

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
