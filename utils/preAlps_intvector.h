/*
============================================================================
Name        : preAlps_intvector.h
Author      : Simplice Donfack
Version     : 0.1
Description : Basic and linear operations on vector of int.
Date        : Oct 13, 2017
============================================================================
*/

/*
 * preAlps_intVector_* is a wrapper for some Ivector functions when the user do not want to create an Ivector object.
 * Although it is recommended to use the Ivector class directly for a better memory management and bugs tracking,
 * some users might still find simple to just use preAlps_intVector_* when these pointer has been already created in their program.
 * preAlps_intVector_ allocates and free internally all workspace required in order to call functions from preAlps_intVector_();
 *
 * Exple: a user want to compute c = a + b, where a, b, c are vectors declared as int *a, *b,*c;
 * Approach 1:  (using IVector)
 * ----------
 * int *a, *b, *c; int m;
 * //... some user initialisation for a and b
 * Ivector aWorK = IVectorNULL(), bWorK = IVectorNULL(), cWorK = IVectorNULL()
 * IVectorCreateFromPtr(aWork, m, a);
 * IVectorCreateFromPtr(bWork, m, b);
 * IVectorCreateFromPtr(cWork, m, c);
 * IVectorAdd(&c, &a, &b)
 * IvectorPrint(&c);
 *
 * Approach 2:  (using preAlps_intVector_)
 * ----------
 * int *a, *b, *c; int m;
 * //... some user initialisation for a and b
 * preAlps_intVector_add(m, c, a, b);
 * preAlps_intVector_Print(c);
*/

#ifndef PREALPS_INTVECTOR_H
#define PREALPS_INTVECTOR_H



/* Gather each part of a vector and Dump the result in a file*/
void preAlps_intVector_gathervDump(int *v_in, int mloc, char *fileName, MPI_Comm comm, char *header);

/*
 * Load a vector of integers from a file.
 */
void preAlps_intVector_load(char *filename, int **v, int *vlen);


/* x = b(p), for dense vectors x and b; p=NULL denotes identity */
void preAlps_intVector_permute(const int *p, const int *b_in, int *x_out, int n);


/*
 * Save a vector of integers in a file.
 */
void preAlps_intVector_save(int *v, int vlen, char *filename, char *header);



/*
 * Each processor print the vector of type int that it has.
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
void preAlps_intVector_printSynchronized(int *v, int vlen, char *varname, char *s, MPI_Comm comm);

#endif
