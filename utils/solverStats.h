/*
============================================================================
Name        : solverStats.h
Author      : Simplice Donfack
Version     : 0.1
Description : Save solver statistics
Date        : Sept 15, 2017
============================================================================
*/

#ifndef SOLVERSTATS_H
#define SOLVERSTATS_H

/*Struct for statistics*/
typedef struct{

 double tParpack;
 double tSolve;
 double tAv;
 double tSv;
 double tInvAv;
 double tComm;
 double tTotal;

} SolverStats_t;

/*Initialize the statistics structure*/
void SolverStats_init(SolverStats_t *tstats);

#endif
