/******************************************************************************/
/* Author     : Olivier Tissot                                                */
/* Creation   : 2016/06/23                                                    */
/* Description: Basic struct for user parameters                              */
/******************************************************************************/


/******************************************************************************/
/*                                  INCLUDE                                   */
/******************************************************************************/
#ifndef USR_PARAM_H
#define USR_PARAM_H

/* STD */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
/* CPaLAMeM */
#include <cpalamem_macro.h>
#include <MPIutils.h>
#include <cpalamem_instrumentation.h>
/* MPI */
#include <mpi.h>
/******************************************************************************/

/******************************************************************************/
/*                                    CODE                                    */
/******************************************************************************/

/* The parameters that we will get from command line */
typedef struct {
  char* matrixFilename;     /* .mtx filename */
  char* solverFilename;     /* .dat filename containing infos on the solver */
  unsigned int nbBlockPart; /* Number of block in partitioning */
  unsigned int nbRHS;       /* Number of rhs in Block CG */
  double tolerance;         /* Solver tolerance */
  unsigned int iterMax;     /* Iteration maximum */
} Usr_Param_t;

/* Return a struct initialized but not allocated */
Usr_Param_t UsrParamNULL();
/* Fill the structure from command line arguments */
int UsrParamReadFromCline(Usr_Param_t* p, int argc, char** argv);

/******************************************************************************/

#endif
