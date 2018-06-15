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
#include <preAlps_cplm_utils.h>
#include <preAlps_cplm_timing.h>
#include <preAlps_cplm_dvector.h>

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
