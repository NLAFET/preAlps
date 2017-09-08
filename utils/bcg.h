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
#include <cpalamem_macro.h>
#include <cpalamem_instrumentation.h>
#include <mat_csr.h>
#include <mat_dense.h>
#include <ivector.h>
#include <dvector.h>
#include <cholqr.h>
#include <matmult.h>
/* ParBCG */
#include "parbcg_macro.h"
#include "solver.h"


/* Block CG algorithm */
typedef enum {
  EK,    /* Enlarged Krylov */
  COOP,  /* Cooperative */
  RRHS  /* Random Right Hand side */
} Block_CG_Alg_t;
/* Preconditioner */
typedef enum {
  LEFT_PREC,
  NO_PREC
} Prec_t;
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
/* Dubrulle's trick: QR on R (or eventually P) */
typedef enum {
  R_QR,
  NO_QR_STAB
} QR_Stab_t;

typedef struct {
  Solver_t          solver;    /* Generic solver structure */
  DVector_t*        b;         /* Right hand side */
  double            normb;     /* norm_2(b) */
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
  double*           work;      /* working array */
  int               iter;      /* Iteration */

  Prec_t            prec_type; /* Block diagonal preconditioner */
  Block_CG_Alg_t    bcg_alg;   /* Block CG algorithm */
  Ortho_Alg_t       ortho_alg; /* A-orthonormalization algorithm */
  Block_Size_Red_t  bs_red;    /* Block size reduction */
  QR_Stab_t         rqr_stab;  /* Stabilization by doing a QR on R */
} BCG_t;


#include "operator.h"
// #include "a_ortho.h"
#include "parbcg_getline.h"
#include "precond.h"
#include "block_jacobi.h"
/******************************************************************************/

/******************************************************************************/
/*                                    CODE                                    */
/******************************************************************************/

int BCGReadParamFromFile(BCG_t* bcg_solver, const char* filename);
int BCGMalloc(BCG_t* bcg_solver, int M, int m, Usr_Param_t* param, const char* name);
int BCGCreateRandomRhs(BCG_t* bcg_solver, int generatorSeed);
int BCGReadRhsFromFile(BCG_t* bcg_solver, const char* filename);
int BCGInitializeOutput(BCG_t* bcg_solver);
int BCGInitialize(BCG_t* bcg_solver);
int BCGSplit(DVector_t* x, Mat_Dense_t* XSplit, int colIndex);
int BCGIterate(BCG_t* bcg_solver);
int BCGOrthodir(BCG_t* bcg_solver); // P = AP - PP^t AA P - P_prevp P_prev^t AA P_prev (two times)
int BCGOrthomin(BCG_t* bcg_solver); // P = R - P^t A R (two times)
int BCGStoppingCriterion(BCG_t* bcg_solver, int* stop, int* min_index);
void BCGFree(BCG_t* bcg_solver);
int BCGFinalize(BCG_t* bcg_solver, int min_index);
int BCGDump(BCG_t* bcg_solver);
void BCGPrint(BCG_t* bcg_solver);

int BCGSolve(BCG_t* bcg_solver, DVector_t* rhs, Usr_Param_t* param, const char* name);
/******************************************************************************/

#endif
