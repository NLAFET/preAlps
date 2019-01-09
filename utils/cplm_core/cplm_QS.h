/*
* This file contains functions used for sorting data
*
* Authors : Sebastien Cayrols
* Email   : sebastien.cayrols@[(gmail.com) | (inria.fr)]
*/
#ifndef PREALPS_CPLM_QS_H
#define PREALPS_CPLM_QS_H

void CPLM_echanger(int tableau[], int a, int b);

void CPLM_echangerWithValues(int tableau[], int a, int b, double values[]);

void CPLM_quickSort(int tableau[], int debut, int fin);

void CPLM_quickSortWithValues(int tableau[], int debut, int fin, double values[]);

void CPLM_quickSortGetPerm(int **value, int debut, int fin, int **perm);

void CPLM_quickSortDoubleWithPerm(double **value, int begin, int end, int **perm);

#endif
