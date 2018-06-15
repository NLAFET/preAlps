/*
* This file contains functions used for sorting data
*
* Authors : Sebastien Cayrols
* Email   : sebastien.cayrols@[(gmail.com) | (inria.fr)]
*/

/**
 * \file QS.c
 * \brief Function of sort
 * \author Sebastien Cayrols
 * \version 0.1
 * \date 25 june 2013
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <preAlps_cplm_QS.h>
#include <math.h>

typedef struct{
	int col;
	double val;
}ColVal;

typedef struct{
	int pos;
	int val;
}ValPos;

/**
 * \fn static int compare (void const *a, void const *b)
 * \brief Method which compares two values
 * \param a The first value
 * \param b The second value
 */
static int compare (void const *a, void const *b)
{
   /* definir des pointeurs type's et initialise's
      avec les parametres */
   int const *pa = (const int*)a;
   int const *pb = (const int*)b;

   /* evaluer et retourner l'etat de l'evaluation (tri croissant) */
   return *pa - *pb;
}

/**
 * \fn static int compare_col (void const *a, void const *b)
 * \brief Method which two labels of a ColVal struct
 * \param a The first value
 * \param b The second value
 */
static int compare_col (void const *a, void const *b)
{
   /* definir des pointeurs type's et initialise's
      avec les parametres */
   ColVal const *pa = (ColVal const*)a;
   ColVal const *pb = (ColVal const*)b;

   /* evaluer et retourner l'etat de l'evaluation (tri croissant) */
   return pa->col - pb->col;
}

static int compare_absval_colval (void const *a, void const *b)
{
   /* definir des pointeurs type's et initialise's
      avec les parametres */
   ColVal const *pa = (ColVal const*)a;
   ColVal const *pb = (ColVal const*)b;

   /* evaluer et retourner l'etat de l'evaluation (tri croissant) */
   return (fabs(pa->val) - fabs(pb->val) >= 0) ? 1 : -1;
}

/**
 * \fn void CPLM_quickSort(int tableau[], int debut, int fin)
 * \brief Method which sort fin - debut consecutives values in tableau
 * \param *tableau The array of values
 * \param debut The index of the first value
 * \param fin The index of the last value
 */
void CPLM_quickSort(int tableau[], int debut, int fin)
{
	qsort(&tableau[debut],fin+1-debut,sizeof(int),compare);
}

/**
 * \fn void CPLM_quickSortWithValues(int tableau[], int debut, int fin, double values[])
 * \brief Method which sort fin - debut consecutives values in tableau
 * \param *tableau The array of values
 * \param debut The index of the first value
 * \param fin The index of the last value
 * \param *values The array of values
 */
void CPLM_quickSortWithValues(int tableau[], int debut, int fin, double values[])
{
	ColVal *tab;
	int size=fin+1-debut;
	if((tab=(ColVal*)malloc(size*sizeof(ColVal)))==NULL){
		fprintf(stderr,"Error during malloc for tab size = %d = %d - %d\n",size,fin-1,debut);
		exit(1);
	}
	for(int i=0;i<size;i++){
		tab[i].col=tableau[i+debut];
		tab[i].val=values[i+debut];
	}
	qsort(tab,size,sizeof(ColVal),compare_col);

	for(int i=0;i<size;i++){
		tableau[i+debut]=tab[i].col;
		values[i+debut]=tab[i].val;
	}

	free(tab);
}

/**
 * \fn static int compare_col (void const *a, void const *b)
 * \brief Method which two labels of a ColVal struct
 * \param a The first value
 * \param b The second value
 */
static int compare_val (void const *a, void const *b)
{
   /* definir des pointeurs type's et initialise's
      avec les parametres */
   ValPos const *pa = (ValPos const*)a;
   ValPos const *pb = (ValPos const*)b;

   /* evaluer et retourner l'etat de l'evaluation (tri croissant) */
   return pa->val - pb->val;
}

//only perm is in/out
void CPLM_quickSortGetPerm(int **value, int debut, int fin, int **perm)
{
	ValPos *val = NULL;

	int size  = fin-debut;

	if( (val = (ValPos*)malloc(size*sizeof(ValPos)) ) ==  NULL){
		fprintf(stderr,"Error during malloc for val size = %d = %d - %d\n",size,fin,debut);
		exit(1);
	}

	for(int i=0;i<size;i++){
		val[i].pos  = (*perm)[i+debut];
		val[i].val  = (*value)[i+debut];
	}

	qsort(val,size,sizeof(ValPos),compare_val);

	for(int i=0;i<size;i++){
		(*perm)[i+debut]   = val[i].pos;
		(*value)[i+debut]   = val[i].val;
  }

	free(val);
}

void CPLM_quickSortDoubleWithPerm(double **value, int begin, int end, int **perm)
{
	ColVal *tab = NULL;
	int size = end + 1 - begin;
	tab = (ColVal*)malloc(size * sizeof(ColVal));
	if(tab == NULL){
		fprintf(stderr,"Error during malloc for tab size = %d = %d - %d\n",size,end-1,begin);
		exit(1);
	}
	for(int i = 0; i < size; i++){
		tab[i].col = (*perm)[i + begin];
		tab[i].val = (*value)[i + begin];
	}
	qsort(tab,size,sizeof(ColVal),compare_absval_colval);
	for(int i = 0; i < size; i++){
		(*perm)[i + begin] = tab[i].col;
  }
	free(tab);
}
