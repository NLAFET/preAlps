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






/*
 * Move in Ivector.h
 */

/*
 * Each processor print a vector of integer
 * Work only in debug (-DDEBUG) mode
 * v:
 *    The vector to print
 */

void CPLM_IVectorPrintSynchronized (CPLM_IVector_t *v, MPI_Comm comm, char *varname, char *s);


/*
 * Move in Dvector.c
 */

/*
 * Each processor print a vector of double
 * Work only in debug (-DDEBUG) mode
 * v:
 *    The vector to print
 */

void CPLM_DVectorPrintSynchronized (CPLM_DVector_t *v, MPI_Comm comm, char *varname, char *s);











/*
 * Move in CPLM_MatCSR.h
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

 int CPLM_MatCSRBlockColRemove(CPLM_Mat_CSR_t *A, int *colCount, int numBlock);


  /*
   * 1D block rows gather of the matrix from all the processors in the communicator .
   * Asend:
   *     input: the matrix to send
   * Arecv
   *     output: the matrix to assemble the block matrix received from all (relevant only on the root)
   * idxRowBegin:
   *     input: the global row indices of the distribution
   */
 int CPLM_MatCSRBlockRowGatherv(CPLM_Mat_CSR_t *Asend, CPLM_Mat_CSR_t *Arecv, int *idxRowBegin, int root, MPI_Comm comm);

 /*
  * Gatherv a local matrix from each process and dump into a file
  *
  */
 int CPLM_MatCSRBlockRowGathervDump(CPLM_Mat_CSR_t *locA, char *filename, int *idxRowBegin, int root, MPI_Comm comm);


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
int CPLM_MatCSRBlockRowScatterv(CPLM_Mat_CSR_t *Asend, CPLM_Mat_CSR_t *Arecv, int *idxRowBegin, int root, MPI_Comm comm);

/*Create a matrix from a dense vector of type double, the matrix is stored in column major format*/
int CPLM_MatCSRConvertFromDenseColumnMajorDVectorPtr(CPLM_Mat_CSR_t *m_out, double *v_in, int M, int N);

/*Create a matrix from a dense vector of type double*/
int CPLM_MatCSRConvertFromDenseDVectorPtr(CPLM_Mat_CSR_t *m_out, double *v_in, int M, int N);

/*
 * Matrix vector product, y := alpha*A*x + beta*y
 */
int CPLM_MatCSRMatrixVector(CPLM_Mat_CSR_t *A, double alpha, double *x, double beta, double *y);

/*
 * Perform an ordering of a matrix using parMetis
 *
 */

int CPLM_MatCSROrderingND(MPI_Comm comm, CPLM_Mat_CSR_t *A, int *vtdist, int *order, int *sizes);

/*
 * Partition a matrix using parMetis
 * part_loc:
 *     output: part_loc[i]=k means rows i belongs to subdomain k
 */

int CPLM_MatCSRPartitioningKway(MPI_Comm comm, CPLM_Mat_CSR_t *A, int *vtdist, int nparts, int *part_loc);


/* Print a CSR matrix as coordinate triplet (i,j, val)*/
void CPLM_MatCSRPrintCoords(CPLM_Mat_CSR_t *A, char *s);

/* Only one process print its matrix, forces synchronisation between all the procs in the communicator*/
void CPLM_MatCSRPrintSingleCoords(CPLM_Mat_CSR_t *A, MPI_Comm comm, int root, char *varname, char *s);

/*
 * Each processor print the matrix it has as coordinate triplet (i,j, val)
 * Work only in debug (-DDEBUG) mode
 * A:
 *    The matrix to print
 */

void CPLM_MatCSRPrintSynchronizedCoords (CPLM_Mat_CSR_t *A, MPI_Comm comm, char *varname, char *s);



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

int CPLM_MatCSRSymRACScaling(CPLM_Mat_CSR_t *A, double *R, double *C);



/* Transpose a matrix */
int CPLM_MatCSRTranspose(CPLM_Mat_CSR_t *A_in, CPLM_Mat_CSR_t *B_out);



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
 * From an array, set one when the node number is a leave.
 * The array should be initialized with zeros before calling this routine
 * n:  total number of nodes in the tree.
*/
void preAlps_binaryTreeIsLeaves(int nparts, int *isLeave);

/*
 * From an array, set one when the node number is a node at the target Level.
 * The array should be initialized with zeros before calling this routine
*/
void preAlps_binaryTreeIsNodeAtLevel(int targetLevel, int twoPowerLevel, int part_root, int *isNodeLevel);

/*
 * Create a block arrow structure from a matrix A
 * comm:
 *     input: the communicator for all the processors calling the routine
 * m:
 *     input: the number of rows of the global matrix
 * A:
 *     input: the input matrix
 * AP:
 *     output: the matrix permuted into a block arrow structure (relevant only on proc 0)
 * perm:
 *     output: the permutation vector
 * nbparts:
 *     output: the number of the partition created
 * partCount:
 *     output: the number of rows in each part
 * partBegin:
 *     output: the begining rows of each part.
 */
int preAlps_blockArrowStructCreate(MPI_Comm comm, int m, CPLM_Mat_CSR_t *A, CPLM_Mat_CSR_t *AP, int *perm, int *nbparts, int **partCount, int **partBegin);





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




/*Sort the row index of a CSR matrix*/
void preAlps_matrix_colIndex_sort(int m, int *xa, int *asub, double *a);

/*
 * Compute A1 = A(pinv,q) where pinv and q are permutations of 0..m-1 and 0..n-1.
 * if pinv or q is NULL it is considered as the identity
 */
void preAlps_matrix_permute (int n, int *xa, int *asub, double *a, int *pinv, int *q,int *xa1, int *asub1,double *a1);

/* Broadcast the matrix dimension from the root to the other procs*/
int preAlps_matrixDim_Bcast(MPI_Comm comm, CPLM_Mat_CSR_t *A, int root, int *m, int *n, int *nnz);


/*
 * We consider one binary tree A and two array part_in and part_out.
 * part_in stores the nodes of A as follows: first all the children at the last level n,
 * then all the children at the level n-1 from left to right, and so on,
 * while part_out stores the nodes of A in depth first search, so each parent node follows all its children.
 * The array part_in is compatible with the array sizes returned by ParMETIS_V3_NodeND.
 * part_out[i] = j means node i in part_in correspond to node j in part_in.
*/
void preAlps_NodeNDPostOrder(int nparts, int *part_in, int *part_out);


/*
 * Number the nodes at level targetLevel and decrease the value of pos.
*/
void preAlps_NodeNDPostOrder_targetLevel(int targetLevel, int twoPowerLevel, int part_root, int *part_in, int *part_out, int *pos);


/*
 * Split n in P parts.
 * Returns the number of element, and the data offset for the specified index.
 */
void preAlps_nsplit(int n, int P, int index, int *n_i, int *offset_i);

/* pinv = p', or p = pinv' */
int *preAlps_pinv (int const *p, int n);

/* pinv = p', or p = pinv' */
int preAlps_pinv_outplace (int const *p, int n, int *pinv);


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
int preAlps_permuteOffDiagRowsToBottom(CPLM_Mat_CSR_t *locA, int *idxColBegin, int *nbDiagRows, int *colPerm, MPI_Comm comm);


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
int preAlps_permuteSchurComplementToBottom(CPLM_Mat_CSR_t *locA, int nbDiagRows, int *idxColBegin, int *colPerm, int *schur_ncols, MPI_Comm comm);


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

int preAlps_schurComplementGet(CPLM_Mat_CSR_t *A, int firstBlock_nrows, int firstBlock_ncols, CPLM_Mat_CSR_t *Agg);

/*Force the current process to sleep few seconds for debugging purpose*/
void preAlps_sleep(int my_rank, int nbseconds);


#endif
