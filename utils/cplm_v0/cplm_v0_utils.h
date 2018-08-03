/*
* This file contains utility functions
*
* Authors : Sebastien Cayrols
* Email   : sebastien.cayrols@[(gmail.com) | (inria.fr)]
* Initial file: Sebastien Cayrols
* Simplified version: Simplice Donfack
*/
#ifndef PREALPS_CPLM_H
#define PREALPS_CPLM_H

#include <stdarg.h>

#ifndef MPIACTIVATE
  #define MPIACTIVATE
#endif

/*
 * Redefine MIN and MAX to avoid problem of crossdefinition
 */
#define CPLM_MAX(_a, _b) ((_a) > (_b) ? (_a) : (_b))
#define CPLM_MIN(_a, _b) ((_a) < (_b) ? (_a) : (_b))

#define CPLM_TRUE 1

#define CPALAMEMSIGERR  "ERROR"
#define CPALAMEMSIGWRN  "WARNING"
#define CPALAMEMSIGABRT "ABORTING"


/* Simplified version of CPLM_CHKERR */
#define CPLM_CHKERR(_e) if((_e) != 0)\
{CPLM_stdFErr(__FUNCTION__, __FILE__, __LINE__, (_e));}

#define CPLM_Abort(_format, _args...) CPLM_FAbort((__FUNCTION__),(_format),##_args)
#define CPLM_ASSERT(_t)  if(!(_t))\
    { CPLM_Abort(" wrong test '" #_t "' line %d", __LINE__);}


/**
 * \fn void CPLM_checkMPIERR(int cr, int rank, const char *action)
 * \brief Method which checks the returned code returned by MPI
 * Note : This method is more explicit if DEBUG is defined
 * \param cr The value returned by MPI call
 * \param rank The rank of the process which calls a MPI function
 * \param *action The name of the action
 */
/*Function which checks the return of an MPI call*/
void CPLM_checkMPIERR(int cr, const char *action);
void CPLM_efprintf(const char *signal, const char *fun, const char *format, va_list va);
void CPLM_esprintf( const char *signal,
                        const char *fun,
                        const char *format,
                        ...);
void CPLM_FAbort(const char *fun,  const char *format, ...);
void CPLM_stdFErr(const char *fun, const char *file, const int line, const int ierr);

/*
 * Split n in P parts and
 * returns the number of element, and the data offset for the specified index.
 */
void CPLM_nsplit(int n, int P, int index, int *n_i, int *offset_i);


#endif
