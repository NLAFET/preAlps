/*
* This file contains functions used for interfacing with Petsc
*
* Authors : Sebastien Cayrols
*         : Olivier Tissot
* Email   : sebastien.cayrols@[(gmail.com) | (inria.fr)]
*         : olivier.tissot@inria.fr
*/
#ifndef PETSC_INTERTACE_H
#define PETSC_INTERTACE_H

#include <petscmat.h>
#include <petscsys.h>
#include <cplm_light/cplm_matcsr.h>
#include <cplm_light/cplm_matdense.h>

//This function creates a sequential matrix from a MatCSR format
int CPLM_petscCreateSeqMatFromMatCSR(CPLM_Mat_CSR_t *m, Mat *M);
//This function creates a parallel matrix from a MatCSR format
int CPLM_petscCreateMatFromMatCSR(CPLM_Mat_CSR_t *m, Mat *M);
//This function creates a parallel matrix from a MatDense format
int CPLM_petscCreateMatFromMatDense(CPLM_Mat_Dense_t *m, Mat *M);

int CPLM_petscCreateSeqMatDenseFromMatCSR(CPLM_Mat_CSR_t *m, Mat *M);

PetscErrorCode petscGetILUFactor(Mat *M, PetscReal k, Mat *F);

PetscErrorCode petscCreateMatCSR(Mat A_in, CPLM_Mat_CSR_t *B_out);

PetscErrorCode petscPrintMatCSR(const char *name, CPLM_Mat_CSR_t *A_in);

PetscErrorCode petscMatCSRMatDenseMult(Mat*         A_in,
                                       CPLM_Mat_Dense_t* B_in,
                                       CPLM_Mat_Dense_t* C_out);

PetscErrorCode petscMatGetScaling(Mat A, Vec scale);

PetscErrorCode petscMatLoad(Mat *A, const char *fileName, MPI_Comm comm);

//PetscErrorCode petscConvertFactorToMatCSR(Mat *F, CPLM_Mat_CSR_t *matL, CPLM_Mat_CSR_t *matU);

PetscErrorCode petscLUFromMatCSR( CPLM_Mat_CSR_t   *m,
                                  Mat         *F,
                                  int         packed,
                                  const char  *solverPackage,
                                  MatFactorType typeLU,
                                  float tau,
                                  float k,
                                  float fillEstimator,
                                  float zeroPivot);

PetscErrorCode petscLUFactorization(Mat           A,
                                    Mat           *F,
                                    const char    *solverPackage,
                                    MatFactorType typeLU,
                                    float         tau,
                                    float         k,
                                    float         fillEstimator,
                                    float         zeroPivot);

#endif
