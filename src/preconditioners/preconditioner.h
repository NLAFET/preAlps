#ifndef PRECONDITIONER_H
#define PRECONDITIONER_H

typedef enum {
  PREALPS_NOPREC,
  PREALPS_BLOCKJACOBI,
  PREALPS_PRESC /* Preconditioner based on the Schur-Complement */
} preAlps_preconditioner_t;

#endif
