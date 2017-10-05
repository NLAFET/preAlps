/*
============================================================================
Name        : lorasc_eigsolve.h
Author      : Simplice Donfack
Version     : 0.1
Description : Solve eigenvalues problem using ARPACK
Date        : oct 1, 2017
============================================================================
*/
#ifndef LORASC_EIGSOLVE_H
#define LORASC_EIGSOLVE_H

/*
 * Solve the eigenvalues problem S*u = \lambda*Agg*u using arpack.
 * Where  S = Agg - Agi*inv(Aii)*Aig.
 *
 * lorascA:
 *     input/output: stores the computed eigenvalues at the end of this routine
 * comm:
 *    input: the communicator
 * mloc:
 *    input: the number of rows of the local matrice.
 * Aggloc
 *    input: the matrix Agg distributed on all procs
 * Agi, Aii, Aig
 *    input: the matrices required for computing the second part of S
 * Agg
 *    input: the matrix Agg on processor 0
 * Aii_sv
 *    input: the solver object to apply to compute  Aii^{-1}v
 * Agg_sv
 *    input: the solver object to apply to compute  Agg^{-1}v
*/

int preAlps_LorascEigSolve(preAlps_Lorasc_t *lorascA, MPI_Comm comm, int mloc, CPLM_Mat_CSR_t *Aggloc, CPLM_Mat_CSR_t *Agi, CPLM_Mat_CSR_t *Aii, CPLM_Mat_CSR_t *Aig,
                         CPLM_Mat_CSR_t *Agg, preAlps_solver_t *Aii_sv, preAlps_solver_t *Agg_sv);
#endif
