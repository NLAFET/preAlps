/*
 ============================================================================
 Name        : test_spMSV.c
 Author      : Simplice Donfack
 Version     : 0.1
 Description : Parallel utilities
 Date        : Sept 27, 2016
 ============================================================================
 */
#ifndef S_UTILS_MP_H
#define S_UTILS_MP_  H

#include<mpi.h>
/*
 * Each processor print its current value of an integer
 * Work only in debug (-DDEBUG) mode
 */
void s_int_print_mp(MPI_Comm comm, int a, char *s);

/*
 * Each processor print a vector of double
 * Work only in debug (-DDEBUG) mode
 */
void s_vector_print_mp(MPI_Comm comm, double *u, int N, char *varname, char *s);
  
/*
 * Each processor print a vector of integer
 * Work only in debug (-DDEBUG) mode
 */
void s_ivector_print_mp (MPI_Comm comm, int *u, int N, char *varname, char *s);

/* Display statistiques from an integer*/
void s_stats_int_display(MPI_Comm comm, int a, char *str);

/* Display statistiques*/
void s_stats_display(MPI_Comm comm, double d, char *str, double dTotal);

/*The specified proc print its vector of double*/
void s_vector_print_single_mp(MPI_Comm comm, int proc_id, double *u, int N, char *varname, char *s);

/*The specified proc prints its vector of double*/
void s_ivector_print_single_mp(MPI_Comm comm, int proc_id, int *u, int N, char *varname, char *s);
#endif

