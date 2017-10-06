/*
 ============================================================================
 Name        : preAlps_matrix_mp.h
 Author      : Simplice Donfack
 Version     : 0.1
 Description : Parallel matrix utilities
 Date        : Sept 27, 2016
 ============================================================================
 */
#ifndef PREALPS_MATRIX_UTILS_MP_H
#define PREALPS_MATRIX_UTILS_MP_H

#include <mpi.h>

#ifdef USE_PARMETIS
  #include <parmetis.h>
#else
  #include <metis.h>
#endif

#include "mmio.h"
#include "s_utils.h"


/* 
 * 1D block rows gatherv of the matrix to from the processors in the communicator using ncounts and displs.
 * The result is stored on processor 0.
 * ncounts: ncounts[i] = k means processor i has k rows. 
 */
int preAlps_matrix_assemble(MPI_Comm comm, int mloc, int *xa, int *asub, double *a, int *ncounts, int **xa1, int **asub1, double **a1);


/*
 * Create a sparse block structure of an CSR matrix.
 * The matrix is initially 1D row block distributed.
 * n:
 *		input: global number of columns
 * nparts:
 *		input: number of domain
 * ncounts:
 * 		input: number of columns in each subdomain
 * ABlockStruct
 *		output: array of size (nbprocs x nparts)
 */
int preAlps_matrix_createBlockStruct(MPI_Comm comm, preAlps_matrix_storage_t mtype, int mloc, int nloc, int *xa, int *asub, int nparts, int *ncounts, int *ABlockStruct);



/* 
 * 1D block rows distribution of the matrix to all the processors in the communicator
 */
int preAlps_matrix_distribute(MPI_Comm comm, int m, int **xa, int **asub, double **a, int *mloc);


/*
 * Assemble a matrix from all the processors in the communicator.
 * Each procs hold a matrix initially 1D block rows distributed.
*/
int preAlps_matrix_kpart_assemble(MPI_Comm comm, int m, int n, int *xa, int *asub, double *a, int mloc, int *part, int **xa1, int **asub1, double **a1);

/*
 * 1D block rows redistribution of the matrix to each proc after a partitioning.
 * part_loc: part_loc[i]=k means row i belongs to subdomain k.
*/
int preAlps_matrix_kpart_redistribute(MPI_Comm comm, int m, int n, int **xa, int **asub, double **a, int *mloc, int *part);

#ifdef USE_PARMETIS
/*
 * Partition a matrix using parMetis
 * part_loc:
 * 		output: part_loc[i]=k means rows i belongs to subdomain k
 */
int preAlps_matrix_partition(MPI_Comm comm, int *vtdist, int *xa, int *asub, int nbparts, int *part_loc);
#endif


/* Every process print its local matrix*/
void preAlps_matrix_print_global(MPI_Comm comm, preAlps_matrix_storage_t mtype, int m, int *xa, int *asub, double *a, char *s);

/* Only one process print its matrix, forces synchronisation between all the procs in the communicator */
void preAlps_matrix_print_single(MPI_Comm comm, preAlps_matrix_storage_t mtype, int root, int m, int *xa, int *asub, double *a, char *s);




/* 
 * 1D block rows distribution of the matrix to all the processors in the communicator using ncounts and displs.
 * The data are originally stored on processor 0.
 * ncounts: ncounts[i] = k means processor i has k rows. 
 * 
 */
int preAlps_matrix_scatterv(MPI_Comm comm, int mloc, int **xa, int **asub, double **a, int *ncounts);

/*
 * Create a sparse block structure of a CSR matrix.
 * The matrix is initially 1D row block distributed.
 * n:
 *		input: global number of columns
 * nparts:
 *		input: number of domain
 * ncounts:
 * 		input: number of columns in each subdomain
 * ABlockStruct
 *		output: array of size (nbprocs x nparts)
 */
int preAlps_matrix_sparseBlockStruct(MPI_Comm comm, int mloc, int *xa, int *asub, int n, int nparts, int *ncounts, int *ABlockStruct);



/* Distribute a vector v as Block Diagonal CSR matrix */
int preAlps_matrix_vpart_distribute(MPI_Comm comm, int m, double *v, int nbpartcol, int *part, int mloc, int *xb, int *bsub, double *b);

/* Distribute a vector v as Block Diagonal CSR matrix 
 * mloc:
 *     input: local number of rows
 * ncounts :
 *     input: number of rows for each part
 * b_nparts:
 *     input: number of block columns of the resulting matrix
 * v:
 *     input: initial vector to distribute
 * xb, bsub, b:
 *     output: sparse CSR matrix to create
*/
int preAlps_matrix_vshift_distribute(MPI_Comm comm, int mloc, int *ncounts, int b_nparts, double *v, int *xb, int **bsub, double **b);
#endif