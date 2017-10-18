#ifndef PREALPS_PARAM_H
#define PREALPS_PARAM_H


/* The default number of rows to print at the top and at the bottom for larger vector */
#ifndef PRINT_DEFAULT_HEADCOUNT
#define PRINT_DEFAULT_HEADCOUNT 10
#endif

/* Print only the lines of a matrix or vector which satisfies "line%PRINT_MOD==0" */
#ifndef PRINT_MOD
#define PRINT_MOD 250
#endif

#endif
