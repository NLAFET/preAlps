/******************************************************************************/
/* Author     : Olivier Tissot, Simplice Donfack                              */
/* Creation   : 2016/09/05                                                    */
/* Description: Definition of the preconditioner                              */
/******************************************************************************/


/******************************************************************************/
/*                                  INCLUDE                                   */
/******************************************************************************/
#ifndef PRECOND_H
#define PRECOND_H

/* STD */
#include <stdio.h>
#include <stdlib.h>
/* MPI */
#include <mpi.h>
/* MatCSR */
//#include <cpalamem_macro.h>
//#include <cpalamem_instrumentation.h>
#include <preAlps_cplm_matcsr.h>
#include <preAlps_cplm_matdense.h>

#include <preAlps_preconditioner_struct.h>

/******************************************************************************/

/******************************************************************************/
/*                                    CODE                                    */
/******************************************************************************/
int  PrecondCreate(Prec_Type_t precond_type,
                   CPLM_Mat_CSR_t* A,
                   int* rowPos,
                   int sizeRowPos,
                   int* colPos,
                   int sizeColPos,
                   int* dep,
                   int sizeDep);
int  PrecondApply(Prec_Type_t precond_type, CPLM_Mat_Dense_t* A_in, CPLM_Mat_Dense_t* B_out);
void PrecondFree(Prec_Type_t precond_type);
/* /\* Right preconditioner *\/o */
/* // User functions */
/* int RightPrecondCreate(); // TODO */
/* int RightPrecondApply();  // TODO */
/* int RightPrecondFree();   // TODO */
/* // ParBCG internals */
/* int RightPrecondInitialize(); // TODO */
/* int RightPrecondFinalize();   // TODO */
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
