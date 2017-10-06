/*
 ============================================================================
 Name        : utils.c
 Author      : Simplice Donfack
 Version     : 0.1
 Description : Utilities
 Date        : Sept 27, 2016
 ============================================================================
 */
#include <stdlib.h>
#include "mmio.h"
#include "s_utils.h"

#ifdef DEBUG
#include <unistd.h>
#endif
/*Terminate the execution of the program with an error*/
void s_abort(char *s){
  printf("[ABORT] %s\n", s);
  exit(1);
}





/* 
 * Split n in P parts.
 * Returns the number of element, and the data offset for the specified index.
 */
void s_nsplit(int n, int P, int index, int *n_i, int *offset_i)
{

  int r;

  r = n % P;
  
  *n_i = (int)(n-r)/P;


  *offset_i = index*(*n_i);


  if(index<r) (*n_i)++;


  if(index < r) *offset_i+=index;
  else *offset_i+=r;
}

/* create a permutation vector from a partition array partvec.
  * partvec: INPUT , partvec[i]=j means element i belongs to subdomain j; 
  * perm:OUTPUT, Permutation vector. 
 */
void s_partitionVector_to_permVector(int *partvec, int n, int nbparts, int *perm)
{

int i,j,k;
j=0;
  for(k=0;k<=nbparts-1;k++)
  {
   for (i=0;i<=n-1;i++)
    {
    if (partvec[i]==k) 
    {
     perm[j]=i; //inv
     j++;
    }
    }
  }
}


/* x(p) = b, for dense vectors x and b; p=NULL denotes identity */
void s_perm_inv_vec (const int *p, const double *b, double *x, int n)
{
    int k ;
    for (k = 0 ; k < n ; k++) x[p ? p[k] : k] = b[k] ;

}
/* x(p) = b, for dense vectors x and b; p=NULL denotes identity */
void s_perm_inv_vec_int (const int *p, const int *b, int *x, int n)
{
    int k ;
    for (k = 0 ; k < n ; k++) x[p ? p[k] : k] = b[k] ;

}

/* x = b(p), for dense vectors x and b; p=NULL denotes identity */
void s_perm_vec (const int *p, const double *b, double *x, int n)
{
    int k ;
    for (k = 0 ; k < n ; k++) x[k] = b [p ? p[k] : k] ;
}

/* x = b(p), for dense vectors x and b; p=NULL denotes identity */
void s_perm_vec_int (const int *p, const int *b, int *x, int n)
{
    int k ;
    for (k = 0 ; k < n ; k++) x[k] = b[p ? p[k] : k] ;
}




/* pinv = p', or p = pinv' */
int *s_return_pinv (int const *p, int n)
{
    int k, *pinv ;
    pinv = (int *) malloc (n *sizeof (int)) ;  /* allocate memory for the results */
    for (k = 0 ; k < n ; k++) pinv [p [k]] = k ;/* invert the permutation */
    return (pinv) ;        /* return result */
}


/*Force the current process to sleep fex seconds for debugging purpose*/
void s_sleep(int my_rank, int nbseconds){
#ifdef DEBUG  
  printf("[%d] Sleeping: %d (s)\n", my_rank, nbseconds);
  sleep(nbseconds);
#endif
}
