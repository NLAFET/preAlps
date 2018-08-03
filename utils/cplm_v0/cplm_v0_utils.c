/*
* This file contains utility functions
*
* Authors : Sebastien Cayrols
* Email   : sebastien.cayrols@[(gmail.com) | (inria.fr)]
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <mpi.h>

#include <cplm_utils.h>



void CPLM_efprintf(const char *signal, const char *fun, const char *format, va_list va)
{
  int rank = 0;
  char msg[1024];

  #ifdef MPIACTIVATE
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  #endif

  vsnprintf(msg, sizeof(msg), format, va);
  snprintf(msg, sizeof(msg), "[Proc: %u]" " %s\n", rank, format);

  fprintf(stderr,"\n%s from %s : %s\n"
  #ifdef STACKTRACE
                  "\tStackTrace::%s\n"
  #endif
                  ,signal
                  ,fun
                  ,msg
  #ifdef STACKTRACE
                  ,curStack->allFun
  #endif
                  );
}

void CPLM_FAbort(const char *fun,  const char *format, ...)
{
  int ierr = 1;
  va_list v;

  va_start(v, format);

  CPLM_efprintf(CPALAMEMSIGABRT, fun, format, v);

  va_end(v);

  #ifdef MPIACTIVATE
  MPI_Abort(MPI_COMM_WORLD, ierr);
  #else
  abort();
  #endif
}


void CPLM_esprintf( const char *signal,
                        const char *fun,
                        const char *format,
                        ...)
{
  va_list v;

  va_start(v, format);

  CPLM_efprintf(signal, fun, format, v);

  va_end(v);
}

/**
 * \fn void CPLM_checkMPIERR(int cr, int rank, const char *action)
 * \brief Method which checks the returned code returned by MPI
 * Note : This method is more explicit if DEBUG is defined
 * \param cr The value returned by MPI call
 * \param rank The rank of the process which calls a MPI function
 * \param *action The name of the action
 */
/*Function which checks the return of an MPI call*/
void CPLM_checkMPIERR(int cr, const char *action){

#ifdef DEBUG_MPIERR
	switch(cr){
		case MPI_SUCCESS :
				printf("%s -- SUCCESS\n",action);
				break;
		case MPI_ERR_COMM :
				printf("ERR_COMM of %s\n",action);
				break;
		case MPI_ERR_COUNT :
				printf("ERR_COUNT of %s\n",action);
				break;
		case MPI_ERR_TYPE :
				printf("ERR_TYPE of %s\n",action);
				break;
		case MPI_ERR_TAG :
				printf("ERR_TAG of %s\n",action);
				break;
		case MPI_ERR_RANK :
				printf("ERR_RANK of %s\n",action);
				break;
		default :
			fprintf(stderr,"(%d) unknown value of cr for %s\n",cr,action);
			exit(2);
	}
#else

	switch(cr){
		case MPI_SUCCESS :
/*		    printf("[%d]%s\n",rank,action);*/
				break;
		default :
			fprintf(stderr,"(%d) unknown value of cr for %s\nYou have to compile again with -DDEBUG\n",cr,action);
			exit(2);
	}

#endif
}

void CPLM_stdFErr(const char  *fun,
                      const char  *file,
                      const int   line,
                      const int   ierr)
{

  //Quick printing the error --Sim
  printf("ERROR from %s: Error in %s, line %d\tcode : %d\n", fun, file, line, ierr);

  //TODO: fix this function
  CPLM_esprintf(CPALAMEMSIGERR, fun, "Error in %s, line %d\tcode : %d\n",
      file,
      line,
      ierr);

}
