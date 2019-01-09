/*
============================================================================
Name        : preAlps_intvector.c
Author      : Simplice Donfack
Version     : 0.1
Description : Basic and linear operations on vector of int.
Date        : Oct 13, 2017
============================================================================
*/
#include <stdlib.h>
#include <stdio.h>
#include <cplm_v0_ivector.h>
#include "preAlps_param.h"
#include "preAlps_utils.h"



/* Gather each part of a vector and Dump the result in a file*/
void preAlps_intVector_gathervDump(int *v_in, int mloc, char *fileName, MPI_Comm comm, char *header){

  int nbprocs, my_rank, root = 0, m = 0;
  int i, *mcounts=NULL, *moffsets=NULL;

  int *v=NULL;

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

    if ( !(v  = (int *) malloc(m * sizeof(int))) ) preAlps_abort("Malloc fails for moffsets[].");
  }

  /* Each process send mloc element to proc 0 */
  MPI_Gatherv(v_in, mloc, MPI_INT, v, mcounts, moffsets, MPI_INT, root, comm);

  if(my_rank==0){
    CPLM_IVector_t Work1 = CPLM_IVectorNULL();
    CPLM_IVectorCreateFromPtr(&Work1, m, v);
    CPLM_IVectorSave(&Work1, fileName, header);

    free(mcounts);
    free(moffsets);
  }

}


/*
 * Load a vector of integers from a file.
 */
void preAlps_intVector_load(char *filename, int **v, int *vlen){

  CPLM_IVector_t Work1 = CPLM_IVectorNULL();

  CPLM_IVectorLoad (filename, &Work1, 0);

  *v = Work1.val;
  *vlen = Work1.nval;
}

/*
*
*/
/* x = b(p), for dense vectors x and b; p=NULL denotes identity */
void preAlps_intVector_permute(const int *p, const int *b_in, int *x_out, int n)
{
    int k ;
    for (k = 0 ; k < n ; k++) x_out [k] = b_in [p ? p [k] : k] ;
}


/*
 * Save a vector of integers in a file.
 */
void preAlps_intVector_save(int *v, int vlen, char *filename, char *header){

  CPLM_IVector_t Work1 = CPLM_IVectorNULL();
  CPLM_IVectorCreateFromPtr(&Work1, vlen, v);
  CPLM_IVectorSave (&Work1, filename, header);

}




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
void preAlps_intVector_printSynchronized(int *v, int vlen, char *varname, char *s, MPI_Comm comm){
  /*
  CPLM_IVector_t Work1 = CPLM_IVectorNULL();
  if(v) CPLM_IVectorCreateFromPtr(&Work1, vlen, v);
  preAlps_IVectorPrintSynchronized (&Work1, comm, varname, s);
  */
#ifdef DEBUG
  int i,j,mark=0;

  int TAG_PRINT = 4;

  CPLM_IVector_t vbuffer = CPLM_IVectorNULL();
  int my_rank, comm_size;

  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &comm_size);

  if(my_rank ==0){

    printf("[%d] %s\n", 0, s);

    for(j=0;j<vlen;j++) {
      #ifdef PRINT_MOD
        //print only the borders and some values of the vector
        if((j>PRINT_DEFAULT_HEADCOUNT) && (j<vlen-1-PRINT_DEFAULT_HEADCOUNT) && (j%PRINT_MOD!=0)) {

          if(mark==0) {printf("%s[...]: ...\n", varname); mark=1;} //prevent multiple print of "..."

          continue;
        }
        mark = 0;
      #endif

      printf("%s[%d]: %d\n", varname, j, v[j]);

    }

    for(i = 1; i < comm_size; i++) {

      /*Receive a Vector*/

      CPLM_IVectorRecv(&vbuffer, i, TAG_PRINT, comm);
      mark = 0;
      printf("[%d] %s\n", i, s);
      for(j=0;j<vbuffer.nval;j++) {
        #ifdef PRINT_MOD
          //print only the borders and some values of the vector
          if((j>PRINT_DEFAULT_HEADCOUNT) && (j<vbuffer.nval-1-PRINT_DEFAULT_HEADCOUNT) && (j%PRINT_MOD!=0)) {
            if(mark==0) {printf("%s[...]: ...\n", varname); mark=1;} //prevent multiple print of "..."
            continue;
          }

          mark = 0;
        #endif

        printf("%s[%d]: %d\n", varname, j, vbuffer.val[j]);

      }

    }
    printf("\n");

    CPLM_IVectorFree(&vbuffer);
  }
  else{
    CPLM_IVector_t Work1 = CPLM_IVectorNULL();
    if(v) CPLM_IVectorCreateFromPtr(&Work1, vlen, v);
    CPLM_IVectorSend(&Work1, 0, TAG_PRINT, comm);
  }

  MPI_Barrier(comm);

#endif

}
