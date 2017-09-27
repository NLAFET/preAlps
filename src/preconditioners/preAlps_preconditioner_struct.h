/*
============================================================================
Name        : prec_struct.c
Author      : Simplice Donfack
Version     : 0.1
Description : A generic preconditioner structure that can be used externally to build
preconditioner compatible with EcgSolver
Date        : Sept 27, 2017
============================================================================
*/
#ifndef PREALPS_PRECONDITIONER_STRUCT_H
#define PREALPS_PRECONDITIONER_STRUCT_H
/* From which side the preconditioner needs to be applied: LEFT or SPLITTED */
typedef enum {
  LEFT_PREC,
  SPLIT_PREC
} Prec_Side_t;

/* Preconditioner type */
typedef enum {
  PREALPS_NOPREC,       /* No preconditioner */
  PREALPS_BLOCKJACOBI,  /* Block Jacobi preconditioner */
  PREALPS_LORASC,       /* Lorasc */
  PREALPS_PRESC         /* Preconditioner based on the Schur-Complement */
} Prec_Type_t;

/* Structure of the preconditioner */
typedef struct{
  Prec_Side_t side;
  Prec_Type_t type;
  void *data; /* Preconditioner data, cast depending on precond_type */
} PreAlps_preconditioner_t;
#endif
