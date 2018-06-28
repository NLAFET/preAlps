/*
============================================================================
Name        : preAlps_ivector.c
Author      : Simplice Donfack, Sebastien Cayrols, Olivier Tissot,
Version     : 0.1
Description : Ivector class from CPaLAMeM. This file is provided for compatibility
and debugging purpose only, it will be removed later.
Date        : jun 13, 2018
============================================================================
*/

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <preAlps_utils.h>
#include <preAlps_cplm_utils.h>
#include <preAlps_cplm_timing.h>
#include <preAlps_cplm_ivector.h>

/**
 * \fn bCast_Vect(int *vec, int length, MPI_Comm comm, int root)
 * \brief Method which sends a CPLM_IVector_t to all processes
 * \param *vec The value array of the IVector
 * \param length The size of the IVector
 * \param comm The communicator for MPI
 * \param root The rank of the process which wants to send the IVector
 */
/*Function sends a CPLM_IVector_t to send_to*/
int CPLM_IVectorBcast(CPLM_IVector_t *v, MPI_Comm comm, int root)
{
CPLM_PUSH
CPLM_BEGIN_TIME

	int ierr = 0;
	int rank = 0;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	ierr = MPI_Bcast(&v->nval,
      1,
      MPI_INT,
      root,
      comm);CPLM_checkMPIERR(ierr,"bcast_length");
	//send length of the IVector

	if(rank != root && v->val == NULL)
  {
    ierr = CPLM_IVectorMalloc(v, v->nval);CPLM_CHKERR(ierr);
  }

	//send data of the IVector
	ierr = MPI_Bcast(v->val,
      v->nval,
      MPI_INT,
      root,
      comm);CPLM_checkMPIERR(ierr,"bcast_vec");

CPLM_END_TIME
CPLM_POP
  return ierr;
}


int CPLM_IVectorCalloc(CPLM_IVector_t *v_out, int length)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  CPLM_ASSERT(v_out      != NULL);
  CPLM_ASSERT(v_out->val == NULL);

  v_out->nval = length;
  v_out->size = length;
  v_out->val  = (int*)calloc(v_out->size, sizeof(int));

CPLM_END_TIME
CPLM_POP
  return !(v_out->val != NULL);
};


int CPLM_IVectorCreateFromPtr(CPLM_IVector_t *v, int length, int *val)
{
CPLM_PUSH
CPLM_BEGIN_TIME
  int ierr  = 0;

  CPLM_ASSERT(val    != NULL);
  CPLM_ASSERT(v->val == NULL);
  CPLM_ASSERT(length >= 0);

  v->nval = length;
  v->size = length;
  v->val  = val;

CPLM_END_TIME
CPLM_POP
  return ierr;
}

void CPLM_IVectorFree(CPLM_IVector_t *v_io)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  if(v_io)
  {
    if(v_io->val)
    {
      free(v_io->val);
    }
    v_io->nval = 0;
    v_io->size = 0;
    v_io->val  = NULL;
  }

CPLM_END_TIME
CPLM_POP
}


//ASSUMING v is sorted
int CPLM_IVectorGetPos(CPLM_IVector_t *v, int a, int *pos)
{
CPLM_PUSH
CPLM_BEGIN_TIME
  int not_found = 0;
  int begin     = 0;
  int end       = 0;
  int curPos    = 0;
  int ierr      = 0;

  end   = v->nval;
  *pos  = -1;

  while(!not_found)
  {
    curPos  = (end + begin) / 2;

    if(a < v->val[curPos])
    {
      end = curPos;
    }
    else if(a > v->val[curPos])
    {
      begin = curPos;
    }
    else
    {
      *pos      = curPos;
      not_found = 1;
    }

    if(begin == end)
    {
      //*pos      = begin + 1;//DO not understand why...
      //not_found = 1;
      break;
    }
  }

  if(*pos == -1)
    ierr = 1;

CPLM_END_TIME
CPLM_POP
  return ierr;

}
/**
 * \fn CPLM_IVector_t invertIVector(CPLM_IVector_t *v,int size)
 * \brief Function which inverts a IVector
 * \param *v The IVector
 * \return The inverted IVector
 */
/*Function which inverts a IVector*/
int CPLM_IVectorInvert(CPLM_IVector_t *v_in, CPLM_IVector_t *v_out)
{
CPLM_PUSH
CPLM_BEGIN_TIME
  int ierr = 0;

  ierr = handleMRealloc(v_out, v_in->nval);CPLM_CHKERR(ierr);

	for(int i = 0; i < v_in->nval; i++)
  {
		v_out->val[v_in->val[i]] = i;
  }

CPLM_END_TIME
CPLM_POP
	return ierr;
}


int CPLM_IVectorMalloc(CPLM_IVector_t *v_out,
                  int       length)
{
CPLM_PUSH
CPLM_BEGIN_TIME
  v_out->size = length;
  v_out->nval = length;

  CPLM_ASSERT(v_out      != NULL);
  CPLM_ASSERT(v_out->val == NULL);

  v_out->val  = (int*)malloc(v_out->size * sizeof(int));

CPLM_END_TIME
CPLM_POP
  return !(v_out->val != NULL);
}

/**
 * \fn void CPLM_IVectorPrint(CPLM_IVector_t *v)
 * \brief This method prints a CPLM_IVector_t
 * \param *v The CPLM_IVector_t to print
 */
/*Function prints a CPLM_IVector_t*/
void CPLM_IVectorPrint(CPLM_IVector_t *v)
{
CPLM_PUSH
CPLM_BEGIN_TIME

	for(int i = 0; i < v->nval; i++)
  {
    printf("%d ",v->val[i]);
  }
	printf("\n");

CPLM_END_TIME
CPLM_POP
}

/**
 * \fn void CPLM_IVectorPrintPartial(CPLM_IVector_t *v)
 * \brief Method which prints partially a CPLM_IVector_t when the number of
 * values is greater than IVECTOR_MAXVAL_PRINT macro
 * and otherwise the IVectorPrint routine is called instead
 * \param *v The CPLM_IVector_t to print
 */
/*Function prints a CPLM_IVector_t*/
void CPLM_IVectorPrintPartial(CPLM_IVector_t *v)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  if(v->nval > 2 * IVECTOR_MAXVAL_PRINT)
  {
    //Print the first part
    for(int i = 0; i < IVECTOR_MAXVAL_PRINT; i++)
    {
      printf("%d ",v->val[i]);
    }
    printf("... ");
    //Print the last part
    for(int i = v->nval - IVECTOR_MAXVAL_PRINT; i < v->nval; i++)
    {
      printf("%d ",v->val[i]);
    }
    printf("\n");
  }
  else
  {
    CPLM_IVectorPrint(v);
  }

CPLM_END_TIME
CPLM_POP
}

int CPLM_IVectorRealloc( CPLM_IVector_t *v_io,
                    int       length)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  v_io->size = length;
  v_io->nval = length;

  CPLM_ASSERT(v_io       != NULL);
  CPLM_ASSERT(v_io->val  != NULL);

  v_io->val  = (int*)realloc(v_io->val, v_io->size * sizeof(int));

CPLM_END_TIME
CPLM_POP
  return !(v_io->val != NULL);
}

/**
  *
  *
  * Computation complexity : 2 * number of lines in fileName
  * Memory complexity      : length of tmp + size of CPLM_IVector_t * size of int
**/
int CPLM_IVectorLoad(const char *fileName, CPLM_IVector_t *buf, int size)
{
CPLM_PUSH
CPLM_BEGIN_TIME
  FILE *fd = NULL;
  int ierr    = 0;
  int length  = 512;
  int nbLine  = size;
  int cpt     = 0;
  int nread   = 0;
  int v       = 0;
  int t       = 0;
  int begin   = 0;
  char *tmp = NULL;

  tmp = (char*)malloc(length * sizeof(char));
  CPLM_ASSERT(tmp != NULL);

  fd=fopen(fileName,"r");
  if(!fd)
  {
    CPLM_Abort("Error, impossible to load %s\n",fileName);
  }

  if(!nbLine)
  {
    //avoid comments
    while(fgets(tmp,length,fd))
    {
      if(tmp[0] == '%')
        continue;
      else
        break;
    }

    //test if number of lines is in the file
    cpt = sscanf(tmp,"%d %d",&v,&t);
    if(cpt == 2)
      nbLine = v;

    if(!nbLine)
    {
      nbLine++;
      while(fgets(tmp,length,fd))
      {
        nbLine++;
      }
    }

    fseek(fd,0,SEEK_SET);
  }

  ierr = CPLM_IVectorMalloc(buf, nbLine);CPLM_CHKERR(ierr);

  while(fgets(tmp,length,fd))
  {
    if(tmp[0] == '%')
      continue;
    else
      break;
  }

  cpt = sscanf(tmp,"%d %d",&v,&t);
  if(cpt==1)
  {
    buf->val[0] = v;
    begin = 1;
  }

  for(int i = begin; i < buf->nval; i++)
  {
    nread = fscanf(fd,"%d",&buf->val[i]);
  }

  free(tmp);
  fclose(fd);
CPLM_END_TIME
CPLM_POP
  return ierr || !nread;
}


//compute the sum of all elements of v
int CPLM_IVectorReduce(CPLM_IVector_t *v, int *sum)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  int ierr  = 0;
  int lsum  = 0;

  for(int i = 0; i < v->nval; i++)
  {
    lsum += v->val[i];
  }

  *sum = lsum;

CPLM_END_TIME
CPLM_POP
  return ierr;
}

/**
  *
  *
  * Computation complexity : size of the IVector
**/
int CPLM_IVectorSave(CPLM_IVector_t *v,const char *fileName, const char *header)
{
CPLM_PUSH
CPLM_BEGIN_TIME
  FILE *fd = NULL;

  CPLM_ASSERT(v->nval  > 0);
  CPLM_ASSERT(v->val  != NULL);

  fd = fopen(fileName,"w");
  CPLM_ASSERT(fd != NULL);
  if(!fd){
    CPLM_Abort("Error, impossible to create %s\n",fileName);
  }

  fprintf(fd,"%%%s\n%d 1\n",header, v->nval);

  for(int i = 0; i < v->nval; i++)
  {
    fprintf(fd,"%d\n", v->val[i]);
  }

  fclose(fd);

CPLM_END_TIME
CPLM_POP
  return 0;
}



int CPLM_IVectorSum(CPLM_IVector_t *u_in, CPLM_IVector_t *v_in, CPLM_IVector_t *w_out, int op)
{
CPLM_PUSH
CPLM_BEGIN_TIME
  int ierr = 0;

  if(u_in->nval != v_in->nval)
  {
    CPLM_Abort("Size of u %d/%d is not equal to size of v %d/%d",
        u_in->nval,
        u_in->size,
        v_in->nval,
        v_in->size);
  }

  if(w_out->size < u_in->nval)
  {
    if(w_out->val == NULL)
    {
      ierr = CPLM_IVectorMalloc(w_out, u_in->nval);CPLM_CHKERR(ierr);
    }
    else
    {
      ierr = CPLM_IVectorCalloc(w_out, u_in->nval);CPLM_CHKERR(ierr);
    }
  }

  switch(op)
  {
    case 0: //i.e. sum
      for(int i = 0; i < u_in->nval; i++)
      {
        w_out->val[i] = u_in->val[i] + v_in->val[i];
      }
      break;

    case 1:
      for(int i = 0; i < u_in->nval; i++)
      {
        w_out->val[i] = u_in->val[i] - v_in->val[i];
      }
      break;

    default:
      CPLM_Abort("op %d is unknown",op);
  }
CPLM_END_TIME
CPLM_POP
  return ierr;
}


/**
 * \fn void send_IVector(CPLM_IVector_t *vec, int dest, int tag, MPI_Comm comm)
 * \brief Method which sends a CPLM_IVector_t to a process
 * \param *vec    The CPLM_IVector_t sent to dest
 * \param dest    The number of the process which will receive the IVector
 * \param tag     The tag of the communication
 * \param comm    The communicator for MPI
 * \return        0 if the CPLM_IVector_t is sent
 */
/*Function sends a CPLM_IVector_t to send_to*/
int CPLM_IVectorSend(CPLM_IVector_t *v, int dest, int tag, MPI_Comm comm)
{
CPLM_PUSH
CPLM_BEGIN_TIME
	int ierr=0;
	//send length of the IVector
	ierr = MPI_Send(&v->nval,
      1,
      MPI_INT,
      dest,
      tag,
      comm);CPLM_checkMPIERR(ierr,"send_length");
	//send data of the IVector
	ierr = MPI_Send(v->val,
      v->nval,
      MPI_INT,
      dest,
      tag,
      comm);CPLM_checkMPIERR(ierr,"send_vec");
CPLM_END_TIME
CPLM_POP
  return ierr;
}

/**
 * \fn CPLM_IVector_t recv_Vect(int recv_from, int tag, MPI_Comm comm)
 * \brief Function which manage the reception of a CPLM_IVector_t and return it
 * \param recv_from The number of the process sending the IVector
 * \param tag The tag of the communication
 * \param comm The communicator for MPI
 * \return The CPLM_IVector_t received
 */
/*Function returns a CPLM_IVector_t received from recv_from*/
int CPLM_IVectorRecv(CPLM_IVector_t *v, int source, int tag, MPI_Comm comm)
{
CPLM_PUSH
CPLM_BEGIN_TIME

	MPI_Status status;
	int ierr    = 0;
  int length  = 0;

	//receive length of the IVector
	ierr = MPI_Recv(&length,
                  1,
                  MPI_INT,
                  source,
                  tag,
                  comm,
                  &status);CPLM_checkMPIERR(ierr,"recv_length");

  ierr = handleMRealloc(v, length);CPLM_CHKERR(ierr);

	//receive data of the IVector
	ierr = MPI_Recv(v->val,
                  v->size,
                  MPI_INT,
                  source,
                  tag,
                  comm,
                  &status);CPLM_checkMPIERR(ierr,"recv_vec");

  MPI_Get_count(&status, MPI_INT, &v->nval);

CPLM_END_TIME
CPLM_POP
	return ierr;
}
