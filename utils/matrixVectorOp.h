/*
============================================================================
Name        : matrixVectorOp.h
Author      : Simplice Donfack
Version     : 0.1
Description : Matrix vector product routines which can be used as operator for
an iterative methods or eigensolver
Date        : Sept 15, 2017
============================================================================
*/
#ifndef MATRIXVECTOROP_H
#define MATRIXVECTOROP_H

#include "mat_csr.h"
#include "solverStats.h"
#include "preAlps_solver.h"



/*
 * Compute the matrix vector product y = A*x
 * where A = A_{loc}^{-1}*S, S = Agg - sum(Agi*Aii^{-1}*Aig).
*/

int matrixVectorOp_AggInvxS(MPI_Comm comm, int mloc, int m, int *mcounts, int *mdispls,
                             CPLM_Mat_CSR_t *Agi, CPLM_Mat_CSR_t *Aii, CPLM_Mat_CSR_t *Aig, CPLM_Mat_CSR_t *Aggloc,
                             preAlps_solver_t *Aii_sv, preAlps_solver_t *Agg_sv, double *X, double *Y,
                             double *dwork1, double *dwork2, double *ywork,
                             SolverStats_t *tstats);


 /*
  * Compute the matrix vector product y = A*x (multilevel version)
  * where A = Agg^{-1}*S, S = Agg - sum(Agi*Aii^{-1}*Aig).
 */

 int matrixVectorOp_AggInvxS_mlevel(int mloc, int m, int *mcounts, int *mdispls,
                              CPLM_Mat_CSR_t *Agi, CPLM_Mat_CSR_t *Aii, CPLM_Mat_CSR_t *Aig, CPLM_Mat_CSR_t *Aggloc,
                              preAlps_solver_t *Aii_sv, preAlps_solver_t *Agg_sv, double *X, double *Y,
                              double *dwork1, double *dwork2, double *ywork,
                              MPI_Comm comm_masterGroup,
                              MPI_Comm comm_localGroup,
                              int *Aii_mcounts, int *Aii_moffsets,
                              int *Aig_mcounts, int *Aig_moffsets,
                              int *Agi_mcounts, int *Agi_moffsets,
                              SolverStats_t *tstats);
                              
/* Compute the matrix vector product y = A*x
 * where A = A_{loc}^{-1}*S, S = Aggloc - Agi*Aii^{-1}*Aig.
*/
int matrixVectorOp_AlocInvxS(MPI_Comm comm, int mloc, int m, int *mcounts, int *mdispls,
                             CPLM_Mat_CSR_t *Aggloc, CPLM_Mat_CSR_t *Agi, CPLM_Mat_CSR_t *Aii, CPLM_Mat_CSR_t *Aig, CPLM_Mat_CSR_t *Aloc,
                             preAlps_solver_t *Aii_sv, preAlps_solver_t *Aloc_sv, double *X, double *Y,
                             double *dwork1, double *dwork2, double *ywork,
                             SolverStats_t *tstats);


/* Compute the matrix vector product y = A*x
 * where A = S*S_{loc}^{-1}, S_{loc} = Block-Diag(S).
 * S*S_{loc}^{-1} = (I + AggP*S_{loc}^{-1})
*/
int matrixVectorOp_SxSlocInv(MPI_Comm comm, int mloc, int m, int *mcounts, int *mdispls,
                            preAlps_solver_t *Sloc_sv, CPLM_Mat_CSR_t *Sloc, CPLM_Mat_CSR_t *AggP,
                            double *X, double *Y, double *dwork, double *ywork,
                            SolverStats_t *tstats);


int matrixVectorOp_SlocInvxS(MPI_Comm comm, int mloc, int m, int *mcounts, int *mdispls,
                            preAlps_solver_t *Sloc_sv, CPLM_Mat_CSR_t *Sloc, CPLM_Mat_CSR_t *AggP,
                            double *X, double *Y, double *dwork, double *ywork,
                            SolverStats_t *tstats);
#endif
