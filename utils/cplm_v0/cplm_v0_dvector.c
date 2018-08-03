/*
============================================================================
Name        : preAlps_ivector.c
Author      : Simplice Donfack, Sebastien Cayrols, Olivier Tissot,
Version     : 0.1
Description : Dvector class from CPaLAMeM. This file is provided for compatibility
and debugging purpose only, it will be removed later.
Date        : jun 13, 2018
============================================================================
*/
#include <stdlib.h>
#include <stdio.h>
#include <cplm_utils.h>
#include <cplm_v0_timing.h>
#include <cplm_v0_dvector.h>

int CPLM_DVectorMalloc(CPLM_DVector_t   *v_out,
                  int         length)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  CPLM_ASSERT(v_out      != NULL);
  CPLM_ASSERT(v_out->val == NULL);

  v_out->nval = length;
  v_out->val  = (double*)malloc(v_out->nval * sizeof(double));

CPLM_END_TIME
CPLM_POP
  return !(v_out->val != NULL);
}

void CPLM_DVectorFree(CPLM_DVector_t   *v_io)
{
CPLM_PUSH

  if(v_io)
  {
    if(v_io->val)
    {
      free(v_io->val);
    }
    v_io->nval  = 0;
    v_io->val   = NULL;
  }

CPLM_POP
}

int CPLM_DVectorConstant(CPLM_DVector_t* v, double value){

  int ierr = 0;

  CPLM_ASSERT(v->val != NULL);

  for(int i=0; i<v->nval;i++)
    v->val[i] = value;


  return ierr;
}

int CPLM_DVectorCreateFromPtr(CPLM_DVector_t *v, int length, double *val){

  int ierr = 0;

  v->nval = length;
  v->val  = val;

  return ierr;
}


/**
  *
  *
  * Computation complexity : size of the vector
  * Memory complexity      : None
**/
int CPLM_DVectorSave(CPLM_DVector_t *v,const char *fileName, const char *header)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  FILE *fd = NULL;

  CPLM_ASSERT(v->nval   >  0);
  CPLM_ASSERT(v->val   !=  NULL);

  fd=fopen(fileName,"w");
  if(!fd)
  {
    fprintf(stderr,"Error, impossible to create %s\n",fileName);
CPLM_END_TIME
CPLM_POP
    return 1;
  }

  fprintf(fd,"%%%%%s\n%d 1\n",header, v->nval);

  for(int i = 0; i < v->nval; i++)
  {
    fprintf(fd,"%.16e\n",v->val[i]);
  }

  fclose(fd);

CPLM_END_TIME
CPLM_POP
  return 0;
}

/**
  *
  * \param size The size of the vector
  * Computation complexity : 2 * number of lines in fileName
  * Memory complexity      : length of tmp + size of CPLM_IVector_t * size of double
**/
int CPLM_DVectorLoad(const char *fileName, CPLM_DVector_t *buf, int size)
{
CPLM_PUSH
CPLM_BEGIN_TIME
  FILE *fd = NULL;
  int length  = 512;
  int nbLine  = size;
  int cpt     = 0;
  int v       = 0;
  int t       = 0;
  int begin   = 0;
  int nread   = 0;
  int ierr    = 0;
  double last = 0.0;
  char *tmp   = NULL;

  tmp = (char*)malloc(length * sizeof(char));
  CPLM_ASSERT(tmp != NULL);

  fd  = fopen(fileName,"r");
  if(!fd)
  {
    fprintf(stderr,"Error, impossible to load %s\n",fileName);
CPLM_END_TIME
CPLM_POP
    return 1;
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

  if(buf->val == NULL)
  {
    ierr = CPLM_DVectorMalloc(buf, nbLine);CPLM_CHKERR(ierr);
  }

  if(buf->nval < nbLine)
  {
    ierr = CPLM_DVectorRealloc(buf, nbLine);CPLM_CHKERR(ierr);
  }

  //Ignore comments
  while(fgets(tmp,length,fd))
  {
    if(tmp[0]=='%')
      continue;
    else
      break;
  }

  //Check if the size is given
  cpt = sscanf(tmp,"%d %d",&v,&t);
  if(cpt == 1)
  {
    buf->val[0] = atof(tmp);
    begin = 1;
    nread = 1;
  }

  //Read data
  for(int i = begin; i < nbLine; i++)
  {
    nread += fscanf(fd,"%lf", &buf->val[i]);
  }

  //CPLM_debug("%d values read\n",nread);
  CPLM_ASSERT(fscanf(fd,"%lf", &last) == EOF);

  free(tmp);
  fclose(fd);
CPLM_END_TIME
CPLM_POP
  return ierr;
}

int CPLM_DVectorRealloc( CPLM_DVector_t   *v_io,
                    int         length)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  CPLM_ASSERT(v_io      != NULL);
  CPLM_ASSERT(v_io->val != NULL);

  v_io->nval = length;
  v_io->val  = (double*)realloc(v_io->val, v_io->nval * sizeof(double));

CPLM_END_TIME
CPLM_POP
  return !(v_io->val != NULL);
}

int CPLM_DVectorAddSpace(CPLM_DVector_t *v_out, int length)
{
CPLM_PUSH
CPLM_BEGIN_TIME

  CPLM_ASSERT(v_out      != NULL);
  CPLM_ASSERT(v_out->val != NULL);

  v_out->nval  += length;
  v_out->val    = (double*)realloc(v_out->val,v_out->nval*sizeof(double));

CPLM_END_TIME
CPLM_POP
  return !(v_out->val != NULL);
}

/**
 * \fn void send_DVect(double *vec, int length, int send_to, int tag, MPI_Comm comm)
 * \brief Method which sends a CPLM_IVector_t to a process
 * \param *vec The value array of the vector
 * \param length The size of the vector
 * \param send_to The number of the process which will receive the vector
 * \param tag The tag of the communication
 * \param comm The communicator for MPI
 */
/*Function sends a CPLM_IVector_t to send_to*/

int CPLM_DVectorSend(CPLM_DVector_t *v, int dest, int tag, MPI_Comm comm){
CPLM_PUSH
CPLM_BEGIN_TIME
	int cr;
	//send length of the vector
	cr=MPI_Send(&(v->nval),1,MPI_INT,dest,tag,comm);CPLM_checkMPIERR(cr,"send_length");
	//send data of the vector
	cr=MPI_Send(v->val,v->nval,MPI_DOUBLE,dest,tag,comm);CPLM_checkMPIERR(cr,"send_vec");
CPLM_END_TIME
CPLM_POP
  return cr;
}


/**
 * \fn CPLM_DVector_t recv_DVect(int recv_from, int tag, MPI_Comm comm)
 * \brief Function which manage the reception of a CPLM_IVector_t and return it
 * \param recv_from The number of the process sending the vector
 * \param tag The tag of the communication
 * \param comm The communicator for MPI
 * \return The CPLM_IVector_t received
 */
/*Function returns a CPLM_IVector_t received from recv_from*/
int CPLM_DVectorRecv(CPLM_DVector_t *v, int source, int tag, MPI_Comm comm)
{
CPLM_PUSH
CPLM_BEGIN_TIME

	MPI_Status status;
	int ierr    = 0;
  int length  = 0;

	//receive length of the vector
	ierr = MPI_Recv(&length,
      1,
      MPI_INT,
      source,
      tag,
      comm,
      &status);CPLM_checkMPIERR(ierr,"recv_length");

  if(v->val == NULL)
  {
    ierr = CPLM_DVectorMalloc(v, length);CPLM_CHKERR(ierr);
  }
  else if(v->nval < length)
  {
    ierr = CPLM_DVectorAddSpace(v, length);CPLM_CHKERR(ierr);
  }

	//receive data of the vector
	ierr  = MPI_Recv(v->val,
      v->nval,
      MPI_DOUBLE,
      source,
      tag,
      comm,
      &status);CPLM_checkMPIERR(ierr,"recv_vec");

  v->nval = length;

CPLM_END_TIME
CPLM_POP
	return ierr;
}
