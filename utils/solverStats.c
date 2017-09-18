/*
============================================================================
Name        : solverStats.c
Author      : Simplice Donfack
Version     : 0.1
Description : Save solver statistics
Date        : Sept 15, 2017
============================================================================
*/

#include "solverStats.h"
/*Initialize the statistics structure*/
void SolverStats_init(SolverStats_t *tstats){
  tstats->tParpack = 0.0;
  tstats->tSolve   = 0.0;
  tstats->tAv      = 0.0;
  tstats->tSv      = 0.0;
  tstats->tInvAv   = 0.0;
  tstats->tComm    = 0.0;
  tstats->tTotal   = 0.0;
}
