/*
 ============================================================================
 Name        : s_utils.h
 Author      : Simplice Donfack
 Version     : 0.1
 Description : Utilities
 Date        : Sept 27, 2016
 ============================================================================
 */
#ifndef S_UTILS_H
#define S_UTILS_H

#ifndef ABS
#define ABS(a)   (((a) < 0) ? -(a) : (a))
#endif

#ifndef MIN
#define MIN(a,b)   ((a < b) ? a : b)
#endif

#ifndef MAX
#define MAX(a,b)   ((a > b) ? a : b)
#endif

/*Terminate the execution of the program with an error*/
void s_abort(char *s);

/* 
 * Split n in P parts.
 * Returns the number of element, and the data offset for the specified processor.
 */
void s_nsplit(int n, int P, int index, int *n_i, int *offset_i);

/* create a permutation vector from a partition array partvec.
  * partvec: INPUT , partvec[i]=j means element i belongs to subdomain j; 
  * perm:OUTPUT, Permutation vector. 
 */
void s_partitionVector_to_permVector(int *partvec, int n, int nbparts, int *perm);

  
/* x(p) = b, for dense vectors x and b; p=NULL denotes identity */
void s_perm_inv_vec (const int *p, const double *b, double *x, int n);

/* x(p) = b, for dense vectors x and b; p=NULL denotes identity */
void s_perm_inv_vec_int (const int *p, const int *b, int *x, int n);

/* x = b(p), for dense vectors x and b; p=NULL denotes identity */
void s_perm_vec (const int *p, const double *b, double *x, int n);

/* x = b(p), for dense vectors x and b; p=NULL denotes identity */
void s_perm_vec_int (const int *p, const int *b, int *x, int n);

/* pinv = p', or p = pinv' */
int *s_return_pinv (int const *p, int n);

/*Force the current process to sleep fex seconds for debugging purpose*/
void s_sleep(int my_rank, int nbseconds);


#endif    
