/******************************************************************************/
/* Author     : Olivier Tissot                                                */
/* Creation   : 2016/06/24                                                    */
/* Description: Block Preconditioned C(onjugate) G(radient)                   */
/******************************************************************************/

/******************************************************************************/
/*                                  INCLUDE                                   */
/******************************************************************************/
#ifndef BCG_H
#define BCG_H

/* STD */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
/* MPI */
#include <mpi.h>
/* CPaLAMeM */
#include <mat_csr.h>
#include <mat_dense.h>
#include <ivector.h>
#include <dvector.h>
#include <cholqr.h>
#include <matmult.h>
/* Preconditioner */
#include <precond.h>
/* ParBCG */
#include "usr_param.h"
#include <cpalamem_macro.h>
#include <cpalamem_instrumentation.h>

#include<mkl.h>


/* A-orthonormalization algorithm */
typedef enum {
  ORTHOMIN,
  ORTHODIR
} Ortho_Alg_t;
/* Block size reduction */
typedef enum {
  ALPHA_RANK,
  NO_BS_RED
} Block_Size_Red_t;

typedef struct {
    /* Array type variables */
  DVector_t*        b;         /* Right hand side */
  Mat_Dense_t*      X;         /* Approximated solution */
  Mat_Dense_t*      R;         /* Residual */
  Mat_Dense_t*      P;         /* Descent direction */
  Mat_Dense_t*      AP;        /* A*P */
  Mat_Dense_t*      P_prev;    /* Previous descent direction */
  Mat_Dense_t*      AP_prev;   /* A*P_prev */
  Mat_Dense_t*      alpha;     /* Descent step */
  Mat_Dense_t*      beta;      /* Step to construt search directions */
  Mat_Dense_t*      gamma;     /* Step to construct odir search directions */
  Mat_Dense_t*      Z;         /* Extra memory */
  Mat_Dense_t*      H;         /* Descent directions needed to reduce block size */
  Mat_Dense_t*      AH;        /* A*H */
  double*           work;      /* working array */

  /* Single value variables */
  double            normb;     /* norm_2(b) */
  double            res;       /* norm_2 of the residual */
  int               iter;      /* Iteration */

  /* Options and parameters */
  const char*       name;      /* Method name */
  char*             oFileName; /* Output file name */
  Usr_Param_t       param;     /* User parameters */
  Precond_side_t    precond_side;
  Precond_t         precond_type;  /* Block diagonal preconditioner */
  Ortho_Alg_t       ortho_alg; /* A-orthonormalization algorithm */
  Block_Size_Red_t  bs_red;    /* Block size reduction */
  MPI_Comm          comm;      /* MPI communicator */
} BCG_t;


#include "operator.h"
#include "parbcg_getline.h"
#include "precond.h"
#include "block_jacobi.h"
/******************************************************************************/

/******************************************************************************/
/*                                    CODE                                    */
/******************************************************************************/

int BCGReadParamFromFile(BCG_t* bcg, const char* filename);
int BCGMalloc(BCG_t* bcg, int M, int m, Usr_Param_t* param, const char* name);
int BCGInitializeOutput(BCG_t* bcg);
int BCGInitialize(BCG_t* bcg);
int BCGSplit(DVector_t* x, Mat_Dense_t* XSplit, int colIndex);
int BCGIterate(BCG_t* bcg);
int BCGReduceSearchDirections(BCG_t* bcg);
int BCGStoppingCriterion(BCG_t* bcg, int* stop);
void BCGFree(BCG_t* bcg);
int BCGFinalize(BCG_t* bcg);
int BCGDump(BCG_t* bcg);
void BCGPrint(BCG_t* bcg);

int BCGSolve(BCG_t* bcg, DVector_t* rhs, Usr_Param_t* param, const char* name);
/******************************************************************************/

#endif
