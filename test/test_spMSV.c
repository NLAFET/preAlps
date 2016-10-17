/*  
 ============================================================================
 Name        : test_spMSV.c
 Author      : Simplice Donfack
 Version     : 0.1
 Description : Sparse matrix matrix product
 Date        : Sept 27, 2016
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#ifdef VERIFY
#include <mkl.h>
#endif
#include "s_utils.h"
#include "s_utils_mp.h"
#include "preAlps_matrix.h"
#include "preAlps_matrix_mp.h"
#include "spMSV.h"



int main(int argc, char *argv[]) {
  
  MPI_Comm comm;

  int nbprocs, my_rank, root = 0;

  char matrix_filename[150]="";
  int matrixDim[3];
  int *xa, *ja;
  double *a;

#ifdef VERIFY
  int *xa_work, *ja_work;
  double *a_work;
  
  int *xb_work, *jb_work;
  double *b_work;
#endif
  
  int m, n, nnz, r;
  int i;
  
  int *ncounts=NULL, *noffsets=NULL, *part;
  int *b_ncounts = NULL;
  int mloc, nzloc, nloc, rloc;
  
  int nparts, b_nparts;
  
  int *A, A_size;
    
  int *xb, *jb;
  double *b;
    
  int *xc = NULL, *jc = NULL;
  double *c = NULL;

  double t_spmsv;

  int nrhs = -1;
  
#ifdef VERIFY
  int j;
  
  int *xc1, *jc1;
  double *c1;
  
  double *b2, *c2;
  int ldb2, ldc2;
  
  int *xc3 = NULL, *jc3 = NULL;
  double *c3 = NULL;
  
  double norm_r1, norm_r2 = 0.0;
  double t_mkl_dcsrmm, t_mkl_dcsrmultcsr = 0.0;
  
#endif
    
  double *v;
  
  int spmsv_options[3] = {0, 0, 0};
  
  
  /* Start MPI*/
  
  MPI_Init(&argc, &argv);
  
  MPI_Comm_dup(MPI_COMM_WORLD, &comm);
  
  MPI_Comm_size(comm, &nbprocs);

  MPI_Comm_rank(comm, &my_rank);

#ifdef DEBUG
  printf("I am proc %d over %d processors\n", my_rank, nbprocs);
#endif
  
  
  /* Get user parameters */
  for(i=1;i<argc;i+=2){
    if (strcmp(argv[i],"-m") == 0) strcpy(matrix_filename,argv[i+1]); 
	  if (strcmp(argv[i],"-nrhs") == 0) nrhs = atoi(argv[i+1]);
    if (strcmp(argv[i],"-h") == 0){
      if(my_rank==0){
        printf("* Purpose\n");
        printf("* =======\n");
        printf("* Perform a matrix matrix product C = A*B, \n");
        printf("* where A is a CSR matrix, B is a CSR matrix formed by a set of vectors, and C a CSR matrix.\n");
        printf("*\n");
        printf("* Usage\n");
        printf("* =========\n");
        printf("* ./test_spMSV -m <matrix_file_name> [-nrhs <nrhs>]\n");
        printf("*\n");
        printf("* Arguments\n");
        printf("* =========\n");
        printf("* -m: the matrix file\n");
        printf("*    the matrix file in matrix market format\n");
        printf("* -nrhs:\n");
        printf("*    The number of nrhs. If not given, the number of processors will be used.\n");
        
      }
      MPI_Finalize();
      return EXIT_SUCCESS;
    }
  }
  

  /* nrhs not provided, use the number of processors by default*/
  if(nrhs <=0) nrhs = nbprocs;
  
  /*
   * Load the matrix on proc 0
   */
  
  if(my_rank==0){  
    
    if(strlen(matrix_filename)==0){
	  
	    s_abort("Error: unknown Matrix. Usage ./test_spMSV -m <matrix_file_name> [-nrhs <nrhs>]");
         
    }
	
	
    printf("Matrix name: %s\n", matrix_filename);
    
    printf("Reading matrix ...\n");
    
    /* Read the matrix file */
    preAlps_matrix_readmm_csr(matrix_filename, &m, &n, &nnz, &xa, &ja, &a);
  
    //preAlps_matrix_print(MATRIX_CSR, M, xa, ja, a, "CSR matrix");
    
                  
    printf("Matrix %dx%d, nnz:%d, nbprocs:%d, nrhs:%d\n", m, n, nnz, nbprocs, nrhs);
    
    preAlps_matrix_print(MATRIX_CSR, m, xa, ja, a, "Matrix A");
    
    /* Prepare the matrix dimensions for the broadcast */
    matrixDim[0] = m;
    matrixDim[1] = n;
    matrixDim[2] = nnz;
    

    
  }
  
  
  /* Broadcast the global matrix dimension among all procs */

  MPI_Bcast(&matrixDim, 3, MPI_INT, root, comm);
  
  m   = matrixDim[0];
  n   = matrixDim[1];
  nnz = matrixDim[2];
      
  /* Read/create the vector v */   
  if ( !(v   = (double *) malloc(m*sizeof(double))) ) s_abort("Malloc fails for v[].");
  
  if(my_rank==0){ 
      for(i=0;i<m;i++) v[i] = i*1.0;
  }
  
  /*
   * Partition the matrix
   */
  
  /* Set the number of partitions to create. */
  nparts = nbprocs;
  
  /* Array to indicates in which subdomain appears each row after the partitioning */
  if ( !(part = (int *)     malloc((m*sizeof(int)))) )          s_abort("Malloc fails for part[].");
  
  /* Array to count the number of rows in each partitions */
  if ( !(ncounts   = (int *) malloc(nbprocs*sizeof(int))) )     s_abort("Malloc fails for ncounts[].");
  if ( !(noffsets  = (int *) malloc((nbprocs+1)*sizeof(int))) ) s_abort("Malloc fails for noffsets[].");
  
#ifdef USE_PARMETIS  
  
  int *part_loc;
    
  /*
   * Perform a 1D block rows distribution of the matrix
   */  

  if(my_rank==root) printf("Matrix distributing ...\n");
  
  /*TODO: values of "a" are not used at this step, it can be removed*/  
  
  preAlps_matrix_distribute(comm, m, &xa, &ja, &a, &mloc);
  
  s_int_print_mp(comm, mloc, "mloc");
  
  nloc = n;
  
  
  /*
   * Call parmetis to partition the matrix
   */
  
  if ( !(part_loc = (int *)     malloc((mloc*sizeof(int)))) ) s_abort("Malloc fails for part_loc[]."); 
  
  for(i=0;i<nbprocs;i++){
    s_nsplit(m, nbprocs, i, &ncounts[i], &noffsets[i]);
  }
    
  /* noffsets[i+1] - noffsets[i] returns the number of rows for processor i, as required by parMetis*/
  
  noffsets[nbprocs] = m;
  
   
  
  if(my_rank==root) printf("Matrix partitioning ...\n");
      
  preAlps_matrix_partition(comm, noffsets, xa, ja, nparts, part_loc); 
              
  s_ivector_print_mp (comm, part_loc, mloc, "partloc", "");  
    
  /*Gather the partitioning vector on each proc*/
  MPI_Allgatherv(part_loc, mloc, MPI_INT, part, ncounts, noffsets, MPI_INT, comm);
    
  s_ivector_print_mp (comm, part, m, "part", "");
             
  if(my_rank==root) printf("Matrix redistributing ...\n");
  
  /* Redistribute the matrix according to the global partitioning array 'part' */  
  preAlps_matrix_kpart_redistribute(comm, m, n, &xa, &ja, &a, &mloc, part);
  
  preAlps_matrix_print_global(comm, MATRIX_CSR, mloc, xa, ja, a, "CSR matrix partitioned");
  
  free(part_loc);
  
#else
  
  if(my_rank==root) printf("Matrix partitioning ...\n");
  
  if(my_rank==root){

#ifdef USE_HPARTITIONING
	 /* Use hypergraph partitioning */
	 preAlps_matrix_hpartition_sequential(m, n, xa, ja, nparts, part); 
#else
	 /* Use graph partitioning */ 
	 preAlps_matrix_partition_sequential(m, xa, ja, nparts, part); 
#endif  
	
  }else{
    xa = NULL; ja = NULL; a = NULL;
  }
  
  /*Broadcast global part to each proc*/
  
  s_ivector_print_single_mp(comm, root, part, m, "part", "");
  
  MPI_Bcast(part, m, MPI_INT, root, comm);
  
   
  if(my_rank==root) 
    printf("Matrix redistributing ...\n");
  
  /* Redistribute the matrix according to the global partitioning array 'part' */  
  preAlps_matrix_kpart_redistribute(comm, m, n, &xa, &ja, &a, &mloc, part);
  
  
  preAlps_matrix_print_global(comm, MATRIX_CSR, mloc, xa, ja, a, "CSR matrix partitioned");

  
#endif  
  
  
  /*Compute the number of columns in each block columns*/
  for(i=0;i<nparts;i++){
    ncounts[i] = 0; 
  }
  
  for(i=0;i<m;i++){
    ncounts[part[i]]++;
  }
  
  s_ivector_print_mp (comm, ncounts, nparts, "ncounts", "");

  mloc = ncounts[my_rank];
  nloc = n;
  nzloc = xa[mloc];
    
#ifdef VERIFY  
  
  /* Assemble the matrix into AWORK for further verification */
  
  preAlps_matrix_assemble(comm, mloc, xa, ja, a, ncounts, &xa_work, &ja_work, &a_work);
  
  preAlps_matrix_print_single(comm, MATRIX_CSR, root, m, xa_work, ja_work, a_work, "Matrix A assembled on proc 0");
  
#endif  
  
  /* 
   * Distribute v 
   */
  
  /*Total number of columns of B*/
  r = rloc = nrhs;
  
  /* Number of block columns to create (assume one column per part)*/
  b_nparts = rloc; //nbprocs;
  
  
  /* Allocate memory for the matrix B */
  
  if ( !(xb   = (int *) malloc((mloc+1)*sizeof(int))) ) s_abort("Malloc fails for xb[].");
  if ( !(jb   = (int *) malloc(nzloc*sizeof(int))) ) s_abort("Malloc fails for jb[].");
  if ( !(b   = (double *) malloc(nzloc*sizeof(double))) ) s_abort("Malloc fails for b[].");
  
  if ( !(b_ncounts   = (int *) malloc(b_nparts*sizeof(int))) ) s_abort("Malloc fails for b_ncounts[].");
  
  /* Distribute a vector v as Block Diagonal CSR matrix (one column per partition) */
  //preAlps_matrix_vpart_distribute(comm, m, v, b_nparts,  part, mloc, xb, jb, b);
  preAlps_matrix_vshift_distribute(comm, mloc, ncounts, b_nparts, v, xb, &jb, &b);
	    
  preAlps_matrix_print_global(comm, MATRIX_CSR, mloc, xb, jb, b, "v as CSR diag Block");
  
  
  /*Compute the number of columns for each block column*/
  for(i=0;i<b_nparts;i++){
    b_ncounts[i] = 1; 
  }
  
  //s_abort("Break vshift distrib");
   
  /*
   * Sparse matrix vector section  
   */
  
  /* Create a sparse block struct of A(nbprocs x nbprocs) to predict communication dependencies */
  
  /* Recall: nparts (Number of block columns) might be different to the number of processors.*/
  
  A_size = nbprocs*nbprocs;
  if ( !(A   = (int *) malloc(A_size*sizeof(int))) ) s_abort("Malloc fails for A[].");
  
  preAlps_matrix_createBlockStruct(comm, MATRIX_CSR, mloc, nloc, xa, ja, nbprocs, ncounts, A);
      
  s_ivector_print_single_mp(comm, 0, A, A_size, "A", "global A blockStruct");  
    
  
  /*Each processor converts its local part of B from csr to csc*/
  preAlps_matrix_convert_csr_to_csc(mloc, rloc, &xb, jb, b);
  
  preAlps_matrix_print_global(comm, MATRIX_CSC, rloc, xb, jb, b, "B as CSC");
  
  
  /* 
  * Call of spMSV
  */
  
  if(my_rank==root) printf("Doing spMSV ...\n");
  
  /*Compute the matric vector product*/  
  t_spmsv = MPI_Wtime();
  preAlps_spMSV(comm, mloc, nloc, rloc, 
             xa, ja, a, xb, jb, b, 
             nparts, ncounts,
             b_nparts, b_ncounts,
             A, 
             &xc, &jc, &c, 
			 spmsv_options);
  
  t_spmsv = MPI_Wtime() - t_spmsv;
  
  preAlps_matrix_print_global(comm, MATRIX_CSC, rloc, xc, jc, c, "C as CSC");

#ifdef VERIFY
  
    
  ldb2 = m;
  ldc2 = m;
  
  /*Allocate memory for dense matrices*/
  
  b2 =  (double*) malloc(ldb2 * rloc * sizeof(double));
  c2 =  (double*) malloc(ldc2 * rloc * sizeof(double));
	
  /* Convert B to CSR */
  preAlps_matrix_convert_csc_to_csr(mloc, rloc, &xb, jb, b);
  
  /*Assemble matrix B in BWORK*/
  preAlps_matrix_assemble(comm, mloc, xb, jb, b, ncounts, &xb_work, &jb_work, &b_work);
  
  preAlps_matrix_print_single(comm, MATRIX_CSR, root, m, xb_work, jb_work, b_work, "Matrix B assembled on proc 0");
      
  preAlps_matrix_convert_csc_to_csr(mloc, rloc, &xc, jc, c);
  
  preAlps_matrix_print_global(comm, MATRIX_CSR, mloc, xc, jc, c, "C as CSR");
  
  preAlps_matrix_assemble(comm, mloc, xc, jc, c, ncounts, &xc1, &jc1, &c1);
    
  if(my_rank==0){
  
    preAlps_matrix_print(MATRIX_CSR, m, xc1, jc1, c1, "Matrix C assembled on proc 0");
	
    /*Convert BWORK into a dense matrix B2 */
  
    preAlps_matrix_convert_csr_to_dense(m, rloc, xb_work, jb_work, b_work, MATRIX_COLUMN_MAJOR, b2, ldb2);
    
    preAlps_matrix_dense_print(MATRIX_COLUMN_MAJOR, m, rloc, b2, ldb2, "Matrix B dense "); 
    
    /* 
    * Call of mkl_dcsrmm
    */
      
    for(i=0;i<m*rloc;i++) c2[i]=0.0;
    
    char transa = 'N';
    double alpha = 1.0;
    char matdescra[6] ={'G','L','N', 'F'}; //'X'
    
    double beta = 0.0;
    int A_nnz = xa[m];
	
    /* B and C are stored as Colum major, so MKL assume that the input matrix is 1-based indexed */
    for (i = 0; i < m+1; i++) {
        xa_work[i] ++;
    }
    for (i = 0; i < A_nnz; i++) {
        ja_work[i] ++;
	}
    
    if(my_rank==root) printf("Doing mkl_dcsrmm ...\n");
    
	/* Call mkl_dcsrmm to perform a sparse matrix dense product */
	
    t_mkl_dcsrmm = MPI_Wtime();
    
    mkl_dcsrmm(&transa, &m, &r, &n, &alpha, matdescra, a_work, ja_work, xa_work, &xa_work[1], b2, &ldb2, &beta, c2, &ldc2);

    t_mkl_dcsrmm = MPI_Wtime() - t_mkl_dcsrmm;
    
#ifdef DEBUG    
    preAlps_matrix_dense_print(MATRIX_COLUMN_MAJOR, m, r, c2, ldc2, "C2");
#endif    
    
    /*Compute the norm (C1 is stored using CSR, C2 is dense)*/
    norm_r1 = 0.0;
    
    for (i=0; i<m; i++){
      for (j=xc1[i]; j<xc1[i+1]; j++){
        
#ifdef DEBUG        
        printf("C(i:%d, j:%d), c1:%e, c2:%e\n", i, jc1[j],  c1[j], c2[jc1[j]*ldc2 + i]);
#endif        
        norm_r1 += (c1[j] - c2[jc1[j]*ldc2 + i])*(c1[j] - c2[jc1[j]*ldc2 + i]);
        
      }
  	}
	
    norm_r1 = sqrt(norm_r1); 
    
    /* 
     * Call of mkl_dcsrmultcsr
     */
	
    int B_nnz = xb_work[m];
    
    for (i = 0; i < m+1; i++) {
        xb_work[i] ++;
    }
    for (i = 0; i < B_nnz; i++) {
        jb_work[i] ++;
    }
    
    /* Allocate memory for xc3 */
    xc3 =  (int*) malloc((m+1) * sizeof(int));
    char transa1 = 'N';
    int request = 1;
    int sort = 0;
    int nzmax;
    int info;
    
    t_mkl_dcsrmultcsr =  MPI_Wtime();
    
    /* First call of mkl_dcsrmultcsr required to determine the size of the result */
    mkl_dcsrmultcsr(&transa1, &request, &sort, &m, &n, &r, a_work, ja_work, xa_work, b_work, jb_work, xb_work , NULL, NULL, xc3, &nzmax, &info);
    
    if(info!=0) s_abort("mkl_dcsrmultcsr error \n");
      
    nzmax = xc3[m]; 

    /*Allocate memory for the result*/
    jc3 =  (int*) malloc(nzmax * sizeof(int));
    c3 =  (double*) malloc(nzmax * sizeof(double));
    
    /* Second call of mkl_dcsrmultcsr to compute the final sparse matrix product */
    request = 2;
    mkl_dcsrmultcsr(&transa, &request, &sort, &m, &n, &r, a_work, ja_work, xa_work, b_work, jb_work, xb_work , c3, jc3, xc3, &nzmax, &info);
    if(info!=0) s_abort("mkl_dcsrmultcsr error \n");
    t_mkl_dcsrmultcsr =  MPI_Wtime() - t_mkl_dcsrmultcsr;
    
    
    int C3_nnz = xc3[m];
    
    for (i = 0; i < m+1; i++) {
        xc3[i] --;
    }
    for (i = 0; i < C3_nnz; i++) {
        jc3[i] --;
    }
    
    /*Compute the norm (C3 is stored using CSR, C2 is dense)*/
    norm_r2 = 0.0;
	
#ifdef DEBUG
    printf("\n");
#endif
	    
    for (i=0; i<m; i++){
      for (j=xc3[i]; j<xc3[i+1]; j++){
        
#ifdef DEBUG        
        printf("C(i:%d, j:%d), c3:%e, c2:%e\n", i, jc3[j],  c3[j], c2[jc3[j]*ldc2 + i]);
#endif        
        norm_r2 += (c3[j] - c2[jc3[j]*ldc2 + i])*(c3[j] - c2[jc3[j]*ldc2 + i]);
        
      }
    }
	
    norm_r2 = sqrt(norm_r2); 
    
    printf("Error_norm(spmv-dcsrmm): %e Error_norm(spmv-dcsrmultcsr): %e"
		  "\nTime_spmsv: %f time_mkl_dcsrmm: %f time_mkl_dcsrmultcsr: %f"
		  "\nSpeedup(dcsrmm/spmv): %f speedup(dcsrmultcsr/spmv): %f\n", 
          norm_r1, norm_r2, 
          t_spmsv, t_mkl_dcsrmm, t_mkl_dcsrmultcsr,
          t_mkl_dcsrmm/t_spmsv, t_mkl_dcsrmultcsr/t_spmsv); 
  }
#else
  
  if(my_rank==0){
    printf("Time spmsv:%f\n", t_spmsv);
  }
  
#endif
    
    
          
  /* Free memory */
  
  free(part);
  
  
  
  free(A);
  free(xa);
  free(ja);
  free(a);
  
#ifdef VERIFY
  
if(my_rank==0){
    
  free(xa_work);
  free(ja_work);
  free(a_work);
    
    
  free(xb_work);
  free(jb_work);
  free(b_work);
    
    
  free(xc1);
  free(jc1);
  free(c1);
    
    
  free(b2);
  free(c2);

    
  free(xc3);
  free(jc3);
  free(c3);  
}  
  
#endif  
        
  free(xb);
  free(jb);
  free(b);
  
  if(xc!=NULL)
  free(xc);
  if(jc!=NULL)
  free(jc);
  if(c!=NULL)
  free(c);
  
  
  free(v);
  
  
  free(ncounts);
  free(b_ncounts);
  
  free(noffsets);
  
  
  MPI_Finalize();
  return EXIT_SUCCESS;
}

