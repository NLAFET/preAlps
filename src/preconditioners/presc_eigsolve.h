/*
============================================================================
Name        : presc_eigsolve.c
Author      : Simplice Donfack
Version     : 0.1
Description : Solve eigenvalues problem using ARPACK
Date        : Mai 15, 2017
============================================================================
*/
#ifndef PRESC_EIGSOLVE_H
#define PRESC_EIGSOLVE_H

 /*
  * Solve the eigenvalues problem Sloc*u = \lambda*Aloc*u using arpack.
  * Where  Sloc = Aggloc - Agi*inv(Aii)*Aig.
  *
  * presc:
  *     input/output: stores the computed eigenvalues at the end of this routine
  * comm:
  *    input: the communicator
  * mloc:
  *    input: the number of rows of the local matrice.
  * Aggloc, Agi, Aii, Aig
  *    input: the matrices required for Sloc
  * Aloc
  *    input: the matrix Aloc
  * Aii_sv
  *    input: the solver object to apply to compute  Aii^{-1}v
  * Aloc_sv
  *    input: the solver object to apply to compute  Aloc^{-1}v
 */

 int Presc_eigSolve_SAloc(Presc_t *presc, MPI_Comm comm, int mloc,
                          CPLM_Mat_CSR_t *Aggloc, CPLM_Mat_CSR_t *Agi, CPLM_Mat_CSR_t *Aii, CPLM_Mat_CSR_t *Aig,
                          CPLM_Mat_CSR_t *Aloc, preAlps_solver_t *Aii_sv, preAlps_solver_t *Aloc_sv);
 /*
  * Solve the eigenvalues problem (I + AggP*S_{loc}^{-1})u = \lambda u using arpack.
  * Where AggP and S_{loc} are two sparse matrices.
 *  AggP is formed by the offDiag elements of Agg, and S_{loc} = Block-Diag(S);
  * Check the paper for the structure of AggP and S_{loc}.
  *
  * presc:
       input/output: stores the computed eigenvalues at the end of this routine
  * comm:
  *    input: the communicator
  * mloc:
  *    input: the number of rows of the local matrice.
  * Sloc_sv
  *    input: the solver object to apply to compute  Sloc^{-1}v
  * Sloc
  *    input: the matrix Sloc
  * AggP
  *    input: the matrix AggP
 */
 int Presc_eigSolve_SSloc(Presc_t *presc, MPI_Comm comm, int mloc, preAlps_solver_t *Sloc_sv, CPLM_Mat_CSR_t *Sloc, CPLM_Mat_CSR_t *AggP);

#endif
