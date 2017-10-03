/******************************************************************************/
/* Author     : Olivier Tissot                                                */
/* Creation   : 2016/06/24                                                    */
/* Description: Enlarged Preconditioned C(onjugate) G(radient)                */
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
/* MKL */
#include<mkl.h>
/* CPaLAMeM */
#include <cpalamem_macro.h>
#include <mat_csr.h>
#include <mat_dense.h>
#include <ivector.h>
#include <dvector.h>
#include <cholqr.h>
#include <matmult.h>
#include <kernels.h>
#include <cpalamem_instrumentation.h>

/* A-orthonormalization algorithm */
typedef enum {
  ORTHOMIN,
  ORTHODIR
} ECG_Ortho_Alg_t;
/* Block size reduction */
typedef enum {
  ALPHA_RANK,
  NO_BS_RED
} ECG_Block_Size_Red_t;

typedef struct {
  /* Array type variables */
  double*                b;         /* Right hand side */
  CPLM_Mat_Dense_t*      X;         /* Approximated solution */
  CPLM_Mat_Dense_t*      R;         /* Residual */
  CPLM_Mat_Dense_t*      P;         /* Descent direction */
  CPLM_Mat_Dense_t*      AP;        /* A*P */
  CPLM_Mat_Dense_t*      P_prev;    /* Previous descent direction */
  CPLM_Mat_Dense_t*      AP_prev;   /* A*P_prev */
  CPLM_Mat_Dense_t*      alpha;     /* Descent step */
  CPLM_Mat_Dense_t*      beta;      /* Step to construt search directions */
  CPLM_Mat_Dense_t*      gamma;     /* Step to construct odir search directions */
  CPLM_Mat_Dense_t*      Z;         /* Extra memory */
  CPLM_Mat_Dense_t*      H;         /* Descent directions needed to reduce block size */
  CPLM_Mat_Dense_t*      AH;        /* A*H */
  double*           work;           /* working array */
  int*              iwork;          /* working array */

  /* Single value variables */
  double            normb;     /* norm_2(b) */
  double            res;       /* norm_2 of the residual */
  int               iter;      /* Iteration */

  /* Options and parameters */
  int                  globPbSize;   /* Size of the global problem */
  int                  locPbSize;    /* Size of the local problem */
  int                  maxIter;      /* Maximum number of iterations */
  int                  enlFac;       /* Enlarging factor */
  double               tol;          /* Tolerance */
  ECG_Ortho_Alg_t      ortho_alg;    /* A-orthonormalization algorithm */
  ECG_Block_Size_Red_t bs_red;       /* Block size reduction */
  MPI_Comm             comm;         /* MPI communicator */
} ECG_t;

/******************************************************************************/

/******************************************************************************/
/*                                    CODE                                    */
/******************************************************************************/

int preAlps_ECGMalloc(ECG_t* ecg);
int preAlps_ECGInitialize(ECG_t* ecg, double* rhs, int* rci_request);
int preAlps_ECGSplit(double* x, CPLM_Mat_Dense_t* XSplit, int colIndex);
int preAlps_ECGIterate(ECG_t* ecg, int* rci_request);
int preAlps_ECGStoppingCriterion(ECG_t* ecg, int* stop);
void preAlps_ECGFree(ECG_t* ecg);
int preAlps_ECGFinalize(ECG_t* ecg, double* solution);
void preAlps_ECGPrint(ECG_t* ecg);

/******************************************************************************/

#endif
