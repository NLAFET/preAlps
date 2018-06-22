/*
============================================================================
Name        : cplm_iarray.c
Author      : Simplice Donfack
Version     : 0.1
Description : Basic and linear operations on vector of type int.
Date        : June 22, 2018
============================================================================
*/

/*
 * Initialize a vector of integers with a value.
 */
int  CPLM_IArray_setValue(int *v, int vlen, int a){
 int i, ierr = 0;
 for(i=0;i<vlen;i++) v[i] = a;
 return ierr;
}
