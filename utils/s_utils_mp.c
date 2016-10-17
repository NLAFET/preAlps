/*
 ============================================================================
 Name        : test_spMSV.c
 Author      : Simplice Donfack
 Version     : 0.1
 Description : Parallel utilities
 Date        : Sept 27, 2016
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "s_utils.h"

/*
 * Each processor print its current value of an integer
 * Work only in debug (-DDEBUG) mode
 */
void s_int_print_mp(MPI_Comm comm, int a, char *s){

#ifdef DEBUG
  int i;
  int TAG_WRITE = 4;
  MPI_Status status;

  int b, my_rank, nbprocs;

  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &nbprocs);
  
  if(my_rank ==0){
    
    printf("[%d] %s: %d\n", my_rank, s, a);
       
    for(i = 1; i < nbprocs; i++) {

      MPI_Recv(&b, 1, MPI_INT, i, TAG_WRITE, comm, &status);
      printf("[%d] %s: %d\n", i, s, b);

    }
  }
  else{

    MPI_Send(&a, 1, MPI_INT, 0, TAG_WRITE, comm);
  }

  MPI_Barrier(comm);
#endif
}

/* Display statistiques from an integer*/
void s_stats_int_display(MPI_Comm comm, int a, char *str){

  int my_rank, nbprocs;
  int root = 0;
  int aMin, aMax, aSum;

  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &nbprocs);
  	
  MPI_Reduce(&a, &aMin, 1, MPI_INT, MPI_MIN, root, comm);
  MPI_Reduce(&a, &aMax, 1, MPI_INT, MPI_MAX, root, comm);
  MPI_Reduce(&a, &aSum, 1, MPI_INT, MPI_SUM, root, comm);
  
  if(my_rank==0){
	  printf("Stats: %s, min: %d, max: %d, avg: %.2f\n", str, aMin, aMax, (double) aSum/nbprocs);
  }					   
}

/* Display statistiques*/
void s_stats_display(MPI_Comm comm, double d, char *str, double dTotal){

  int my_rank, nbprocs;
  int root = 0;
  double dMin, dMax, dSum;

  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &nbprocs);
  	
  MPI_Reduce(&d, &dMin, 1, MPI_DOUBLE, MPI_MIN, root, comm);
  MPI_Reduce(&d, &dMax, 1, MPI_DOUBLE, MPI_MAX, root, comm);
  MPI_Reduce(&d, &dSum, 1, MPI_DOUBLE, MPI_SUM, root, comm);
  
  if(my_rank==0){
	  printf("Stats: %s, min: %.6f, max: %.6f, avg: %.6f, percentage: %.6f %%\n", str, dMin, dMax, (double) dSum/nbprocs, (double) 100*dMax/dTotal);
  }					   
}

/*
 * Each processor print a vector of double
 * Work only in debug (-DDEBUG) mode
 */
void s_vector_print_mp(MPI_Comm comm, double *u, int N, char *varname, char *s){
#ifdef DEBUG
  int i,j;

  int TAG_WRITE = 4;
  MPI_Status status;

  int MAX_SIZE = N;
  int N_recv;
  double * buffer;
  int my_rank, comm_size;

  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &comm_size);

  if(my_rank ==0){
    
    /* Allocate the max buffer */
    buffer =  (double*) malloc(MAX_SIZE * sizeof(double));
    
    printf("[%d] %s\n", 0, s);
    
    for(j=0;j<N;j++) {printf("%s[%d]: %e\n", varname, j, u[j]);}
    
    for(i = 1; i < comm_size; i++) {
      /*Receive the number of slides*/      
      MPI_Recv(&N_recv, 1, MPI_INT, i, TAG_WRITE, comm, &status);
      
      if(N_recv>MAX_SIZE){
        
        /* Redim the buffer */
        free(buffer);
        MAX_SIZE = N_recv;
        buffer =  (double*) malloc(MAX_SIZE * sizeof(double));
        
      }

      MPI_Recv(buffer, N_recv, MPI_DOUBLE, i, TAG_WRITE, comm, &status);

            printf("[%d] %s\n", i, s);
      for(j=0;j<N_recv;j++) {printf("%s[%d]: %e\n", varname, j, buffer[j]);}
        }
    printf("\n");
    
    free(buffer);
  }
  else{
    MPI_Send(&N, 1, MPI_INT, 0, TAG_WRITE, comm);
    MPI_Send(u, N, MPI_DOUBLE, 0, TAG_WRITE, comm);
  }

  MPI_Barrier(comm);

#endif
}

/*
 * Each processor print a vector of integer
 * Work only in debug (-DDEBUG) mode
 */

void s_ivector_print_mp (MPI_Comm comm, int *u, int N, char *varname, char *s){
#ifdef DEBUG
  int i,j;

  int TAG_WRITE = 4;
  MPI_Status status;

  int MAX_SIZE = N;
  int N_recv;
  int * buffer;
  int my_rank, comm_size;

    MPI_Comm_rank(comm, &my_rank);
    MPI_Comm_size(comm, &comm_size);

  if(my_rank ==0){
    
    /* Allocate the max buffer */
    buffer =  (int*) malloc(MAX_SIZE * sizeof(int));
    
    printf("[%d] %s\n", 0, s);
    
    for(j=0;j<N;j++) {printf("%s[%d]: %d\n", varname, j, u[j]);}
    
    for(i = 1; i < comm_size; i++) {
      /*Receive the number of slides*/      
      MPI_Recv(&N_recv, 1, MPI_INT, i, TAG_WRITE, comm, &status);
      
      if(N_recv>MAX_SIZE){
        
        /* Redim the buffer */
        free(buffer);
        MAX_SIZE = N_recv;
        buffer =  (int*) malloc(MAX_SIZE * sizeof(int));
        
      }

      MPI_Recv(buffer, N_recv, MPI_INT, i, TAG_WRITE, comm, &status);

            printf("[%d] %s\n", i, s);
      for(j=0;j<N_recv;j++) {printf("%s[%d]: %d\n", varname, j, buffer[j]);}
        }
    printf("\n");
    
    free(buffer);
  }
  else{
    MPI_Send(&N, 1, MPI_INT, 0, TAG_WRITE, comm);
    MPI_Send(u, N, MPI_INT, 0, TAG_WRITE, comm);
  }

  MPI_Barrier(comm);

#endif
}

/*The specified proc print its vector of double*/
void s_vector_print_single_mp(MPI_Comm comm, int proc_id, double *u, int N, char *varname, char *s){
#ifdef DEBUG
  
  int j;
  int prt_mod = (int) 1; //N/50
  int my_rank; 

  prt_mod = MAX(prt_mod, 1);
  
  MPI_Comm_rank(comm, &my_rank);
     


  if(my_rank ==0){
    printf("[%d] (single) %s\n", 0, s);
    for(j=0;j<N;j++) {if((j==0) || (j==N-1) || (j%prt_mod==0)) printf("%s[%d]: %e\n", varname, j, u[j]);}
  }

  MPI_Barrier(comm);
#endif
}

/*The specified proc prints its vector of double*/
void s_ivector_print_single_mp(MPI_Comm comm, int proc_id, int *u, int N, char *varname, char *s){
#ifdef DEBUG
  int j;

  int prt_mod = (int) 1; //N/50
  int my_rank; 

  prt_mod = MAX(prt_mod, 1);
  
  MPI_Comm_rank(comm, &my_rank);

  if(my_rank == proc_id){
    printf("[%d] (single) %s, prt_mod:%d\n", proc_id, s, prt_mod);
    for(j=0;j<N;j++) {if((j==0) || (j==N-1) || (j%prt_mod==0)) printf("%s[%d]: %d\n", varname, j, u[j]);}
  }

  MPI_Barrier(comm);
#endif
}

