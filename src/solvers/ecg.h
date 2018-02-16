/**
 * \file    ecg.c
 * \author  Olivier Tissot
 * \date    2016/06/24
 * \brief   Enlarged Preconditioned C(onjugate) G(radient) solver
 *
 * \details Implements Orthomin, Orthodir as well as their dynamic
 *          counterparts (BF-Omin and D-Odir).
 */

/******************************************************************************/
/*                                  INCLUDE                                   */
/******************************************************************************/
#ifndef ECG_H
#define ECG_H

/**
 * \enum preAlps_ECG_Ortho_Alg_t
 * \brief A-orthonormalization algorithm
 * \author Olivier Tissot
 * \date 2016/06/24
 */
typedef enum {
  ORTHOMIN,
  ORTHODIR
} preAlps_ECG_Ortho_Alg_t;
/**
 * \enum preAlps_ECG_Block_Size_Red_t
 * \brief Block size reduction
 * \author Olivier Tissot
 * \date 2016/06/24
 */
typedef enum {
  ADAPT_BS,
  NO_BS_RED
} preAlps_ECG_Block_Size_Red_t;

/**
* \struct preAlps_ECG_t
* \brief Enlarged Conjugate Gradient solver
* \author Olivier Tissot
* \date 2016/06/24
*/
typedef struct {
  /* Input variable */
  double* b;                    /**< Right hand side */

  /* Internal symbolic variables */
  CPLM_Mat_Dense_t* X;     /**< Approximated solution */
  CPLM_Mat_Dense_t* R;     /**< Residual */
  CPLM_Mat_Dense_t* V;     /**< Descent directions ([P,P_prev] or P) */
  CPLM_Mat_Dense_t* AV;    /**< A*V */
  CPLM_Mat_Dense_t* Z;     /**< Preconditioned residual (Omin) or AP (Odir) */
  CPLM_Mat_Dense_t* alpha; /**< Descent step */
  CPLM_Mat_Dense_t* beta;  /**< Step to construt search directions */

  /** User interface variables */
  CPLM_Mat_Dense_t* P;      /**< Search directions */
  CPLM_Mat_Dense_t* AP;     /**< A*P */
  double* R_p;              /**< Residual */
  double* P_p;              /**< Search directions */
  double* AP_p;             /**< A*P_p */
  double* Z_p;              /**< Preconditioned residual (Omin) or AP (Odir) */

  /** Working arrays */
  double*           work;
  int*              iwork;

  /* Single value variables */
  double            normb;     /**< norm_2(b) */
  double            res;       /**< norm_2 of the residual */
  int               iter;      /**< Iteration */
  int               bs;        /**< Block size */
  int               kbs;       /**< Krylov basis size */

  /* Options and parameters */
  int                          globPbSize; /**< Size of the global problem */
  int                          locPbSize;  /**< Size of the local problem */
  int                          maxIter;    /**< Maximum number of iterations */
  int                          enlFac;     /**< Enlarging factor */
  double                       tol;        /**< Tolerance */
  preAlps_ECG_Ortho_Alg_t      ortho_alg;  /**< A-orthonormalization algorithm */
  preAlps_ECG_Block_Size_Red_t bs_red;     /**< Block size reduction */
  MPI_Comm             comm;               /**< MPI communicator */
} preAlps_ECG_t;
/******************************************************************************/

/******************************************************************************/
/*                                    CODE                                    */
/******************************************************************************/

/**
 * \brief Create the solver and allocate memory
 */
int  preAlps_ECGInitialize(preAlps_ECG_t* ecg, double* rhs, int* rci_request);
/**
 * \brief Performs different steps in ECG iteration according to the value of
 * rci_request
 */
int  preAlps_ECGIterate(preAlps_ECG_t* ecg, int* rci_request);
/**
 * \brief Check for the residual norm and return a boolean that is true if the
 * normalized residual is lower than the specified tolerance
 */
int  preAlps_ECGStoppingCriterion(preAlps_ECG_t* ecg, int* stop);
/**
 * \brief Releases the internal memory and returns the solution
 */
int  preAlps_ECGFinalize(preAlps_ECG_t* ecg, double* solution);
/**
 * \brief Print informations on the solver
 */
void preAlps_ECGPrint(preAlps_ECG_t* ecg, int verbosity);
/* "Private" functions */
/**
 * \brief Allocate memory for the solver
 */
int  _preAlps_ECGMalloc(preAlps_ECG_t* ecg);
/**
 * \brief Initialize the solver assuming that the memory has been allocated
 */
int  _preAlps_ECGReset(preAlps_ECG_t* ecg, double* rhs, int* rci_request);
/**
 * \brief Returns the solution
 */
int  _preAlps_ECGWrapUp(preAlps_ECG_t* ecg, double* solution);
/**
 * \brief Release memory of the solver
 */
void _preAlps_ECGFree(preAlps_ECG_t* ecg);
/**
 * \brief Enlarge the vector x
 */
int  _preAlps_ECGSplit(double* x, CPLM_Mat_Dense_t* XSplit, int colIndex);
/**
 * \brief Orthomin iteration
 */
int  _preAlps_ECGIterateOmin(preAlps_ECG_t* ecg, int* rci_request);
/**
 * \brief Orthodir iteration
 */
int  _preAlps_ECGIterateOdir(preAlps_ECG_t* ecg, int* rci_request);

/******************************************************************************/

#endif
