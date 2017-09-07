#ifndef PRESC_EIGSOLVE_H
#define PRESC_EIGSOLVE_H


typedef struct{

  double deflation_tolerance; //the deflation tolerance, all eigenvalues lower than this will be selected for deflation
  char bmat; /*Standard or generalized problem*/
  char* which;
  double residual_tolerance; // The tolerance of the arnoldi iterative solver

  int ido; /*Which operation to perform in the reverse communication*/
  int max_iterations; /*maximum number of iterations*/

  int iparam[11];
  int ipntr[11];
} presc_eigsolver_t;



#endif
