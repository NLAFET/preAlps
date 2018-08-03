/*
============================================================================
Name        : preAlps_doublevector.c
Author      : Simplice Donfack
Version     : 0.1
Description : Basic and linear operations on vector of double.
Date        : Oct 13, 2017
============================================================================
*/


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <preAlps_cplm_dvector.h>
#include "preAlps_param.h"
#include "preAlps_utils.h"


/* Gather each part of a vector and Dump the result in a file*/
void preAlps_doubleVector_gathervDump(double *v_in, int mloc, char *fileName, MPI_Comm comm, char *header){

  int nbprocs, my_rank, root = 0, m = 0;
  int i, *mcounts=NULL, *moffsets=NULL;

  double *v=NULL;

  MPI_Comm_size(comm, &nbprocs);
  MPI_Comm_rank(comm, &my_rank);

  if(my_rank==0){
    /* Allocate Workspace */
    if ( !(mcounts  = (int *) malloc((nbprocs) * sizeof(int))) ) preAlps_abort("Malloc fails for mcounts[].");
    if ( !(moffsets  = (int *) malloc((nbprocs+1) * sizeof(int))) ) preAlps_abort("Malloc fails for moffsets[].");
  }

  //Gather the number of elements in the vector for each procs

  MPI_Gather(&mloc, 1, MPI_INT, mcounts, 1, MPI_INT, root, comm);

  if(my_rank==0){
    /* Compute the begining of each part */
    moffsets[0] = 0;
    for(i=0;i<nbprocs;i++) moffsets[i+1] = moffsets[i] + mcounts[i];

    m = moffsets[nbprocs];

    if ( !(v  = (double *) malloc(m * sizeof(double))) ) preAlps_abort("Malloc fails for moffsets[].");
  }

  /* Each process send mloc element to proc 0 */
  MPI_Gatherv(v_in, mloc, MPI_DOUBLE, v, mcounts, moffsets, MPI_DOUBLE, root, comm);

  if(my_rank==0){
    CPLM_DVector_t Work1 = CPLM_DVectorNULL();
    CPLM_DVectorCreateFromPtr(&Work1, m, v);
    CPLM_DVectorSave(&Work1, fileName, header);

    free(mcounts);
    free(moffsets);
  }

}


/* Permute a vector by computing x = P^{-1}*b = P^{T}*b , for dense vectors x and b; p=NULL denotes identity */
void preAlps_doubleVector_invpermute(const int *p, const double *b_in, double *x_out, int n)
{
    int k ;
    for (k = 0 ; k < n ; k++) x_out [p ? p [k] : k] = b_in [k] ;
}


/*
 * Load a vector from a file.
 */
void preAlps_doubleVector_load(char *filename, double **v, int *vlen){

  CPLM_DVector_t Work1 = CPLM_DVectorNULL();

  CPLM_DVectorLoad (filename, &Work1, 0);

  *v = Work1.val;
  *vlen = Work1.nval;
}

/*Compute the norm of a vector*/
double preAlps_doubleVector_norm2(double *v, int vlen){
    double r = 0.0;
    for(int j=0;j<vlen;j++) r+=v[j]*v[j];
    return sqrt(r);
}

/* Permute a vector by computing x = P*b, for dense vectors x and b; p=NULL denotes identity */
void preAlps_doubleVector_permute(const int *p, const double *b_in, double *x_out, int n)
{
    int k ;
    for (k = 0 ; k < n ; k++) x_out [k] = b_in [p ? p [k] : k] ;
}

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
void preAlps_doubleVector_printSynchronized(double *v, int vlen, char *varname, char *s, MPI_Comm comm){
#if 0
  CPLM_DVector_t Work1 = CPLM_DVectorNULL();
  if(v) CPLM_DVectorCreateFromPtr(&Work1, vlen, v);
  preAlps_DVectorPrintSynchronized (&Work1, comm, varname, s);
#endif
#ifdef DEBUG
  int i,j,mark = 0;

  int TAG_PRINT = 4;

  CPLM_DVector_t vbuffer = CPLM_DVectorNULL();
  int my_rank, comm_size;

  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &comm_size);

  if(my_rank ==0){

    printf("[%d] %s, norm:%e\n", 0, s, preAlps_doubleVector_norm2(v, vlen));

    for(j=0;j<vlen;j++) {

      #ifdef PRINT_MOD
        //print only the borders and some values of the vector
        if((j>PRINT_DEFAULT_HEADCOUNT) && (j<vlen-1-PRINT_DEFAULT_HEADCOUNT) && (j%PRINT_MOD!=0)) {
          if(mark==0) {printf("%s[...]: ...\n", varname); mark=1;} //prevent multiple print of "..."
          continue;
        }
        mark = 0;
      #endif

      printf("%s[%d]: %20.19g\n", varname, j, v[j]);
    }

    for(i = 1; i < comm_size; i++) {

      /*Receive a Vector*/
      CPLM_DVectorRecv(&vbuffer, i, TAG_PRINT, comm);

      //printf("[%d] %s\n", i, s);
      printf("[%d] %s, norm:%e\n", i, s, preAlps_doubleVector_norm2(vbuffer.val, vbuffer.nval));
      mark = 0;
      for(j=0;j<vbuffer.nval;j++) {

        #ifdef PRINT_MOD
          //print only the borders and some values of the vector
          if((j>PRINT_DEFAULT_HEADCOUNT) && (j<vbuffer.nval-1-PRINT_DEFAULT_HEADCOUNT) && (j%PRINT_MOD!=0)) {
            if(mark==0) {printf("%s[...]: ...\n", varname); mark=1;} //prevent multiple print of "..."
            continue;
          }
          mark = 0;
        #endif

        printf("%s[%d]: %20.19g\n", varname, j, vbuffer.val[j]);

      }
    }
    printf("\n");

    CPLM_DVectorFree(&vbuffer);
  }
  else{
    CPLM_DVector_t Work1 = CPLM_DVectorNULL();
    if(v) CPLM_DVectorCreateFromPtr(&Work1, vlen, v);
    CPLM_DVectorSend(&Work1, 0, TAG_PRINT, comm);
  }

  MPI_Barrier(comm);

#endif
}


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

void preAlps_doubleVectorSet_printSynchronized(double *v, int vlen, int n, int ldv, char *varname, char *s, MPI_Comm comm){
#ifdef DEBUG
  int i,j;

  int my_rank;
  double *vbuffer;
  char sbuffer[80];

  MPI_Comm_rank(comm, &my_rank);

  vbuffer = (double*) malloc(vlen*sizeof(double));
  for(j=0;j<n;j++){ //for each column

    //if(my_rank == 0){
    //  printf("[Set][col:%d] %s\n", j, s);
    //}

    sprintf(sbuffer, "%s[col: %d]", varname, j);

    //the element are not necessary contiguous in memory (e.g if ldv>vlen),
    //so we create a contiguous copy of the data
    for(i=0;i<vlen;i++) vbuffer[i] = v[ldv*j+i];

    preAlps_doubleVector_printSynchronized(vbuffer, vlen, sbuffer, s, comm);
  }

  free(vbuffer);
#endif
}
