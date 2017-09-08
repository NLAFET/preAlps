/*
============================================================================
Name        : preAlps_utils.h
Author      : Simplice Donfack
Version     : 0.1
Description : Utils for preAlps
Date        : Mai 15, 2017
============================================================================
*/
#ifndef PREALPS_UTILS_H
#define PREALPS_UTILS_H

#include <mpi.h>
#include <mat_csr.h>
#include <ivector.h>

#ifndef max
#define max(a,b) ((a) > (b) ? a : b)
#endif

#ifndef min
#define min(a,b) ((a) < (b) ? a : b)
#endif

//#define max(a,b) ((a) > (b) ? a : b)

/* tmp functions*/

/*
 * Split n in P parts.
 * Returns the number of element, and the data offset for the specified index.
 */
void s_nsplit(int n, int P, int index, int *n_i, int *offset_i);

/* pinv = p', or p = pinv' */
int *s_return_pinv (int const *p, int n);

/*Sort the row index of a CSR matrix*/
void preAlps_matrix_colIndex_sort(int m, int *xa, int *asub, double *a);

/*
 * Compute A1 = A(pinv,q) where pinv and q are permutations of 0..m-1 and 0..n-1.
 * if pinv or q is NULL it is considered as the identity
 */
void preAlps_matrix_permute (int n, int *xa, int *asub, double *a, int *pinv, int *q,int *xa1, int *asub1,double *a1);


/*
 * Move in Ivector.h
 */

/*
 * Each processor print a vector of integer
 * Work only in debug (-DDEBUG) mode
 * v:
 *    The vector to print
 */

void IVectorPrintSynchronized (IVector_t *v, MPI_Comm comm, char *varname, char *s);


/*
 * Move in Dvector.c
 */

/*
 * Each processor print a vector of double
 * Work only in debug (-DDEBUG) mode
 * v:
 *    The vector to print
 */

void DVectorPrintSynchronized (DVector_t *v, MPI_Comm comm, char *varname, char *s);











/*
 * Move in MatCSR.h
 */

 /*
  * Split the matrix in block column and remove the selected block column number,
  * Which is the same as replacing all values of that block with zeros.
  * This reoutine can be used to fill the diag of a Block diag of global matrix with zeros when each proc
  * has a rowPanel as locA.
  * A:
  *     input: the input matrix to remove the diag block
  * colCount:
  *     input: the global number of columns in each Block
  * numBlock:
  *     input: the number of the block to remove
  */

 int MatCSRBlockColRemove(Mat_CSR_t *A, int *colCount, int numBlock);


  /*
   * 1D block rows gather of the matrix from all the processors in the communicator .
   * Asend:
   *     input: the matrix to send
   * Arecv
   *     output: the matrix to assemble the block matrix received from all (relevant only on the root)
   * idxRowBegin:
   *     input: the global row indices of the distribution
   */
 int MatCSRBlockRowGatherv(Mat_CSR_t *Asend, Mat_CSR_t *Arecv, int *idxRowBegin, int root, MPI_Comm comm);

 /*
  * Gatherv a local matrix from each process and dump into a file
  *
  */
 int MatCSRBlockRowGathervDump(Mat_CSR_t *locA, char *filename, int *idxRowBegin, int root, MPI_Comm comm);


 /*
  * 1D block rows distribution of the matrix to all the processors in the communicator using ncounts and displs.
  * The data are originally stored on processor root. After this routine each processor i will have the row indexes from
  * idxRowBegin[i] to idxRowBegin[i+1] - 1 of the input matrix.
  *
  * Asend:
  *     input: the matrix to scatterv (relevant only on the root)
  * Arecv
  *     output: the matrix to store the block matrix received
  * idxRowBegin:
  *     input: the global row indices of the distribution
  */
int MatCSRBlockRowScatterv(Mat_CSR_t *Asend, Mat_CSR_t *Arecv, int *idxRowBegin, int root, MPI_Comm comm);

/*Create a matrix from a dense vector of type double, the matrix is stored in column major format*/
int MatCSRConvertFromDenseColumnMajorDVectorPtr(Mat_CSR_t *m_out, double *v_in, int M, int N);

/*Create a matrix from a dense vector of type double*/
int MatCSRConvertFromDenseDVectorPtr(Mat_CSR_t *m_out, double *v_in, int M, int N);

/*
 * Matrix vector product, y := alpha*A*x + beta*y
 */
int MatCSRMatrixVector(Mat_CSR_t *A, double alpha, double *x, double beta, double *y);

/* Print a CSR matrix as coordinate triplet (i,j, val)*/
void MatCSRPrintCoords(Mat_CSR_t *A, char *s);

/* Only one process print its matrix, forces synchronisation between all the procs in the communicator*/
void MatCSRPrintSingleCoords(Mat_CSR_t *A, MPI_Comm comm, int root, char *varname, char *s);

/*
 * Each processor print the matrix it has as coordinate triplet (i,j, val)
 * Work only in debug (-DDEBUG) mode
 * A:
 *    The matrix to print
 */

void MatCSRPrintSynchronizedCoords (Mat_CSR_t *A, MPI_Comm comm, char *varname, char *s);



/*
 *
 * Scale a scaling vectors R and C, and scale the matrix by computing A1 = R * A * C
 * A:
 *     input: the matrix to scale
 * R:
 *     output: a vector with the same size as the number of rows of the matrix
 * C:
 *     output: a vector with the same size as the number of columns of the matrix
 */

int MatCSRSymRACScaling(Mat_CSR_t *A, double *R, double *C);




/*
 * Macros
 */

#define preAlps_checkError(errnum) preAlps_checkError_srcLine((errnum), __LINE__, __FILE__)

/*
 * Functions
 */

/* Display a message and stop the execution of the program */
void preAlps_abort(char *s, ...);

/*
 * Check errors
 * No need to call this function directly, use preAlps_checkError() instead.
*/
void preAlps_checkError_srcLine(int err, int line, char *src);


/*
 * Load a vector from a file.
 */
void preAlps_doubleVector_load(char *filename, double **v, int *vlen);



/*
 * Each processor print the vector of type double that it has.
 * Work only in debug (-DDEBUG) mode
 * v:
 *    input: The vector to print
 * vlen:
 *    input: The len of the vector to print
 * varname:
 *   The name of the vector
 * s:
 *   The string to display before the variable
 */
void preAlps_doubleVector_printSynchronized(double *v, int vlen, char *varname, char *s, MPI_Comm comm);

/* Display statistiques min, max and avg of a double*/
void preAlps_dstats_display(MPI_Comm comm, double d, char *str);


/*
 * preAlps_intVector_* is a wrapper for some Ivector functions when the user do not want to create an Ivector object.
 * Although it is recommended to use the Ivector class directly for a better memory management and bugs tracking,
 * some users might still find simple to just use preAlps_intVector_* when these pointer has been already created in their program.
 * preAlps_intVector_ allocates and free internally all workspace required in order to call functions from preAlps_intVector_();
 *
 * Exple: a user want to compute c = a + b, where a, b, c are vectors declared as int *a, *b,*c;
 * Approach 1:  (using IVector)
 * ----------
 * int *a, *b, *c; int m;
 * //... some user initialisation for a and b
 * Ivector aWorK = IVectorNULL(), bWorK = IVectorNULL(), cWorK = IVectorNULL()
 * IVectorCreateFromPtr(aWork, m, a);
 * IVectorCreateFromPtr(bWork, m, b);
 * IVectorCreateFromPtr(cWork, m, c);
 * IVectorAdd(&c, &a, &b)
 * IvectorPrint(&c);
 *
 * Approach 2:  (using preAlps_intVector_)
 * ----------
 * int *a, *b, *c; int m;
 * //... some user initialisation for a and b
 * preAlps_intVector_add(m, c, a, b);
 * preAlps_intVector_Print(c);
*/

/*
 * Each processor print the value of type int that it has
 * Work only in debug (-DDEBUG) mode
 * a:
 *    The variable to print
 * s:
 *   The string to display before the variable
 */
void preAlps_int_printSynchronized(int a, char *s, MPI_Comm comm);

/*
 * Each processor print the vector of type int that it has.
 * Work only in debug (-DDEBUG) mode
 * v:
 *    input: The vector to print
 * vlen:
 *    input: The len of the vector to print
 * varname:
 *   The name of the vector
 * s:
 *   The string to display before the variable
 */
void preAlps_intVector_printSynchronized(int *v, int vlen, char *varname, char *s, MPI_Comm comm);


/*
 * Permute the rows of the matrix with offDiag elements at the bottom
 * locA:
 *     input: the local part of the matrix owned by the processor calling this routine
 * idxRowBegin:
 *     input: the global array to indicate the partition of each column
 * nbDiagRows:
 *     output: the number of rows permuted on the diagonal
 * colPerm
 *     output: a preallocated vector of the size of the number of columns of A
 *            to return the global permutation vector
*/
int preAlps_permuteOffDiagRowsToBottom(Mat_CSR_t *locA, int *idxColBegin, int *nbDiagRows, int *colPerm, MPI_Comm comm);


/*
 * Permute the matrix to reflect the global matrix structure where all the Block diag are ordered first
 * followed by the Schur complement.
 * The permuted local matrix will have the form locA = [... A_{i, Gamma};... A_{gamma,gamma}]
 *
 * nbDiagRowsloc:
       input: the number of diagonal block on the processor callinf this routine
 * locA:
 *     input: the local part of the matrix owned by the processor calling this routine
 * idxRowBegin:
 *     input: the global array to indicate the column partitioning
 * colPerm
 *     output: a preallocated vector of the size of the number of columns of A
 *            to return the global permutation vector
 * schur_ncols
 *    output: the number of column of the schur complement after the partitioning
 *
*/
int preAlps_permuteSchurComplementToBottom(Mat_CSR_t *locA, int nbDiagRows, int *idxColBegin, int *colPerm, int *schur_ncols, MPI_Comm comm);


/*
 * Check the permutation vector for consistency
 */
int preAlps_permVectorCheck(int *perm, int n);


/*
 * Extract the schur complement of a matrix A
 * A:
 *     input: the matrix from which we want to extract the schur complement
 * firstBlock_nrows:
 *     input: the number of rows of the top left block
 * firstBlock_ncols:
 *     input: the number of cols of the top left block
 * Agg
 *     output: the schur complement matrix
*/

int preAlps_schurComplementGet(Mat_CSR_t *A, int firstBlock_nrows, int firstBlock_ncols, Mat_CSR_t *Agg);

/*Force the current process to sleep few seconds for debugging purpose*/
void preAlps_sleep(int my_rank, int nbseconds);


#endif
