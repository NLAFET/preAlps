/******************************************************************************/
/* Author     : Olivier Tissot                                                */
/* Creation   : 2016/08/03                                                    */
/* Description: Generic iterative solver                                      */
/******************************************************************************/

/******************************************************************************/
/*                                  INCLUDE                                   */
/******************************************************************************/
#include <solver.h>
/******************************************************************************/

/******************************************************************************/
/*                                    CODE                                    */
/******************************************************************************/

int SolverInit(Solver_t* solver, int M, int m, Usr_Param_t param, const char* name) {
//  solver->comm     = MPI_COMM_WORLD;
  solver->N        = M,
  solver->n        = m;
  solver->NNZ      = 0;
  solver->nnz      = 0;
  solver->name     = name;
  solver->param    = param;
  solver->error    = NULL;
  solver->finalRes = -1.0;
  return 0;
}

void SolverFree(Solver_t* solver) {
  if (solver->error != NULL)
    free(solver->error);
}

/******************************************************************************/
