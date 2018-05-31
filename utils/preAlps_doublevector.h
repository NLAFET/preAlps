/*
============================================================================
Name        : preAlps_doublevector.h
Author      : Simplice Donfack
Version     : 0.1
Description : Basic and linear operations on vector of double.
Date        : Oct 13, 2017
============================================================================
*/

#ifndef PREALPS_DOUBLEVECTOR_H
#define PREALPS_DOUBLEVECTOR_H


/* Gather each part of a vector and Dump the result in a file*/
void preAlps_doubleVector_gathervDump(double *v_in, int mloc, char *fileName, MPI_Comm comm, char *header);

/*
 * Load a vector from a file.
 */
void preAlps_doubleVector_load(char *filename, double **v, int *vlen);

/*Compute the norm of a vector*/
double preAlps_doubleVector_norm2(double *v, int vlen);

/* x = b(p), for dense vectors x and b; p=NULL denotes identity */
void preAlps_doubleVector_permute(const int *p, const double *b_in, double *x_out, int n);

/*
 * Each processor print the vector of type double that it has.
 * Work only in debug (-DDEBUG) mode
 * v:
 *    input: The vector to print
 * vlen:
 *    input: The len of the vector to print
 * varname:
 *   The name of the vector
 * s:
 *   The string to display before the variable
 */
void preAlps_doubleVector_printSynchronized(double *v, int vlen, char *varname, char *s, MPI_Comm comm);


/*
 * Each processor print multiple vector of type double that it has. (similar to dense matrice stored as a set of vectors)
 * Work only in debug (-DDEBUG) mode
 * v:
 *    input: The vector to print
 * vlen:
 *    input: The len of each vector to print
 * n:
 *    input: the number of vectors in the set (Should be the same for all the processors)
 * ldv:
 *    input: the number of element between two vectors
 * varname:
 *   The name of the vector
 * s:
 *   The string to display before the variable
 */

void preAlps_doubleVectorSet_printSynchronized(double *v, int vlen, int n, int ldv, char *varname, char *s, MPI_Comm comm);

#endif
