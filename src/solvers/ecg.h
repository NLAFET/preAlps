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

/* A-orthonormalization algorithm */
typedef enum {
  ORTHOMIN,
  ORTHODIR
} preAlps_ECG_Ortho_Alg_t;
/* Block size reduction */
typedef enum {
  ADAPT_BS,
  NO_BS_RED
} preAlps_ECG_Block_Size_Red_t;

typedef struct {
  /* Input variable */
  double* b;                    /* Right hand side */

  /* Internal symbolic variables */
  CPLM_Mat_Dense_t* X;          /* Approximated solution */
  CPLM_Mat_Dense_t* R;          /* Residual */
  CPLM_Mat_Dense_t* Kp;         /* Descent directions ([P,P_prev] or P) */
  CPLM_Mat_Dense_t* AKp;        /* A*Kp */
  CPLM_Mat_Dense_t* Z;          /* Preconditioned residual or AP */
  CPLM_Mat_Dense_t* alpha;      /* Descent step */
  CPLM_Mat_Dense_t* beta;       /* Step to construt search directions */

  /* User interface variables */
  CPLM_Mat_Dense_t* P;          /* For retro compatibility */
  CPLM_Mat_Dense_t* AP;         /* For retro compatibility */
  // TODO
  // double* P_p;
  // double* AP_p;

  /* Working arrays */
  double*           work;
  int*              iwork;

  /* Single value variables */
  double            normb;     /* norm_2(b) */
  double            res;       /* norm_2 of the residual */
  int               iter;      /* Iteration */
  int               bs;        /* Block size */

  /* Options and parameters */
  int                          globPbSize; /* Size of the global problem */
  int                          locPbSize;  /* Size of the local problem */
  int                          maxIter;    /* Maximum number of iterations */
  int                          enlFac;     /* Enlarging factor */
  double                       tol;        /* Tolerance */
  preAlps_ECG_Ortho_Alg_t      ortho_alg;  /* A-orthonormalization algorithm */
  preAlps_ECG_Block_Size_Red_t bs_red;     /* Block size reduction */
  MPI_Comm             comm;               /* MPI communicator */
} preAlps_ECG_t;
/******************************************************************************/

/******************************************************************************/
/*                                    CODE                                    */
/******************************************************************************/

int  preAlps_ECGInitialize(preAlps_ECG_t* ecg, double* rhs, int* rci_request);
int  preAlps_ECGIterate(preAlps_ECG_t* ecg, int* rci_request);
int  preAlps_ECGStoppingCriterion(preAlps_ECG_t* ecg, int* stop);
int  preAlps_ECGFinalize(preAlps_ECG_t* ecg, double* solution);
void preAlps_ECGPrint(preAlps_ECG_t* ecg, int verbosity);
// "Private" functions
int  _preAlps_ECGMalloc(preAlps_ECG_t* ecg);
void _preAlps_ECGFree(preAlps_ECG_t* ecg);
int  _preAlps_ECGSplit(double* x, CPLM_Mat_Dense_t* XSplit, int colIndex);
int  _preAlps_ECGIterateOmin(preAlps_ECG_t* ecg, int* rci_request);
int  _preAlps_ECGIterateOdir(preAlps_ECG_t* ecg, int* rci_request);

/******************************************************************************/

#endif
