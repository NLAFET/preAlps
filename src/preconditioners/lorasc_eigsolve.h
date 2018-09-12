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

#include <mpi.h>
#include <cplm_matcsr.h>
#include "lorasc.h"
#include "preAlps_solver.h"

/*
 * Solve the eigenvalues problem S*u = \lambda*Agg*u using arpack.
 * Where  S = Agg - Agi*inv(Aii)*Aig.
 * lorascA:
 *     input/output: stores the computed eigenvalues at the end of this routine
 * mloc:
 *    input: the number of rows of the local matrice.
*/

int preAlps_LorascEigSolve(preAlps_Lorasc_t *lorascA, int mloc);
#endif
