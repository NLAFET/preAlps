#define _CRT_SECURE_NO_WARNINGS 1

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <limits.h>
#include <sys/types.h>


#if !__GNUC__
typedef long ssize_t;
#endif


#if !defined(SSIZE_MAX)
#define SSIZE_MAX ((ssize_t)(SIZE_MAX/2))
#endif

#if !defined(EOVERFLOW)
#define EOVERFLOW (ERANGE)      /* is there something better to use? */
#endif

ssize_t nx_getdelim(char **lineptr, size_t *n, int delim, FILE *stream);
ssize_t nx_getdelim(char **lineptr, size_t *n, int delim, FILE *stream);
ssize_t nx_getline(char **lineptr, size_t *n, FILE *stream);
ssize_t parbcg_getdelim(char **lineptr, size_t *n, char delim, FILE *stream);
ssize_t parbcg_getline(char **lineptr, size_t *n, FILE *stream);
