/*
============================================================================
Name        : cplm_iarray.c
Author      : Simplice Donfack
Version     : 0.1
Description : Basic and linear operations on vector of type double.
Date        : June 22, 2018
============================================================================
*/

/*
 * Initialize a vector of double with a value.
 */
int  CPLM_DArray_setValue(double *v, int vlen, double a){
 int i, ierr = 0;
 for(i=0;i<vlen;i++) v[i] = a;
 return ierr;
}
