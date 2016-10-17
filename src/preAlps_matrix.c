/*
 ============================================================================
 Name        : preAlps_matrix.c
 Author      : Simplice Donfack
 Version     : 0.1
 Description : Sequential matrix utilities
 Date        : Sept 27, 2016
 ============================================================================
 */

#ifdef USE_PARMETIS
  #include <parmetis.h>
#elif defined(USE_HPARTITIONING)
  #include <patoh.h>
#else
  #include <metis.h>
#endif


#include "s_utils.h"
#include "s_utils_mp.h"
#include "preAlps_matrix.h"

/*Sort the row index of a CSR matrix*/
void preAlps_matrix_colIndex_sort(int m, int *xa, int *asub, double *a){

  int i,j,k;

  int *asub_ptr, row_nnz, itmp;
  double *a_ptr, dtmp;

  for (k=0; k<m; k++){
  
    asub_ptr = &asub[xa[k]];
    a_ptr = &a[xa[k]];
    row_nnz = xa[k+1] - xa[k];
    for(i=0;i<row_nnz;i++){
      for(j=0;j<i;j++){
        if(asub_ptr[i]<asub_ptr[j]){
          /*swap column index*/
            itmp=asub_ptr[i];
            asub_ptr[i]=asub_ptr[j];
            asub_ptr[j]=itmp;
          
          /*swap values */
            dtmp=a_ptr[i]; 
            a_ptr[i]=a_ptr[j];
            a_ptr[j]=dtmp;
        }
      }
    }  
  }   
}

/*
 * Create a sparse column block structure of an CSR matrix.
 * nparts:
 *    input: number of domain
 * ncounts:
 *     input: number of columns in each subdomain
 * ABlockStruct
 *    output: array of size npart allocated on any process calling this routine.
 */
int preAlps_matrix_ColumnBlockStruct(  preAlps_matrix_storage_t mtype, int mloc, int nloc, int *xa, int *asub, int nparts, int *ncounts, int *AcolBlockStruct){
  
  int i, j, k;
  
  
  int *partcol;
  
  
  if ( !(partcol = (int *)   malloc(nloc*sizeof( int))) ) s_abort("Malloc fails for partcol[].");
  
  /*index table to indicate in which domain appears each rows*/
  k = 0;
  for(i=0;i<nparts;i++){
    for(j=0;j<ncounts[i];j++){
      partcol[k] = i;
      k++;
    }
  }
  
  for(i=0;i<nparts;i++){
    AcolBlockStruct[i] = 0;
  }
  
  /* Count the number of nonzeros in each part */
  
  if(mtype==MATRIX_CSR){
    for(i=0;i<mloc;i++){
       for (j=xa[i]; j<xa[i+1]; j++){
         AcolBlockStruct[partcol[asub[j]]]++;
       }
    }    
  }else{
    for (j=0; j<nloc; j++){
      AcolBlockStruct[partcol[j]]+=xa[j+1]-xa[j];
    }
  }
    
  free(partcol);  
  return 0;
}

/*Convert a matrix from csc to csr
 * if the matrix is not square then xa will be reallocated
*/
int preAlps_matrix_convert_csc_to_csr(int m, int n, int **xa, int *asub, double *a){
  
  int *xa_work, *asub_work;
  double *a_work;
  int *row_work;
  int i,j, pos, ipos;
  
  
  int nnz= (*xa)[n];
  
  /* Allocate memory */
  if ( !(a_work = (double *) malloc(nnz*sizeof(double))) ) s_abort("Malloc fails for a_work[].");
  if ( !(asub_work = (int *) malloc(nnz*sizeof(   int))) ) s_abort("Malloc fails for asub_work[].");
  if ( !(xa_work = (int *)   malloc((m+1)*sizeof( int))) ) s_abort("Malloc fails for xa_work[].");
  
  if ( !(row_work = (int *)   malloc((m+1)*sizeof( int))) ) s_abort("Malloc fails for row_work[].");
  
  
  for(i=0;i<m+1;i++){
    row_work[i] = 0;
  }
  
  /*Compute the number of elements in each rows*/
  
  for(i=0;i<nnz;i++){
    row_work[asub[i]]++;
  }
  
  /*Compute the index of each rows*/
  
  xa_work[0] = 0;
  
  for(i=0;i<m;i++){
    xa_work[i+1] = xa_work[i] + row_work[i];
    
    row_work[i] = 0; /*reused as current position for inserting the elements in the next step*/
  }
  
  
  /*Copy A to Awork*/
  
  for (j=0; j<n; j++){
    for (i=(*xa)[j]; i<(*xa)[j+1]; i++){
      /* copy element (asub[i], j, a[i]) to the final position*/
      ipos = asub[i];
      pos = xa_work[ipos]+row_work[ipos];
      row_work[ipos]++;
      
      asub_work[pos] = j;
      a_work[pos] = a[i]; 
    }
  }
  

  /* Copy back the matrix */
  
  if(m==n){
    for(i=0;i<m+1;i++){
      (*xa)[i] = xa_work[i];
    }
  }else{
    
    free(*xa);
    *xa = xa_work;
    
  }
  
  
  for(i=0;i<nnz;i++){
    asub[i] = asub_work[i];
    a[i]    = a_work[i];
  }
  
  /* Free memory*/
  free(a_work);
  free(asub_work);
  if(m==n)
    free(xa_work);
  
  free(row_work);
  
  return 0;

}

/*
 * Convert a matrix from csr to csc
 * if the matrix is not square then xa will be reallocated
 */

int preAlps_matrix_convert_csr_to_csc(int m, int n, int **xa, int *asub, double *a){
  
  int *xa_work, *asub_work;
  double *a_work;
  int *col_work;
  int i,j, pos, ipos;
  
  
  int nnz = (*xa)[m];
  
  /* Allocate memory */
  if ( !(a_work = (double *) malloc(nnz*sizeof(double))) ) s_abort("Malloc fails for a_work[].");
  if ( !(asub_work = (int *) malloc(nnz*sizeof(   int))) ) s_abort("Malloc fails for asub_work[].");
  if ( !(xa_work = (int *)   malloc((n+1)*sizeof( int))) ) s_abort("Malloc fails for xa_work[].");
  
  if ( !(col_work = (int *)   malloc((n+1)*sizeof( int))) ) s_abort("Malloc fails for row_work[].");
  
  
  for(i=0;i<n+1;i++){
    col_work[i] = 0;
  }
  
  /*Compute the number of elements in each rows*/
  
  for(i=0;i<nnz;i++){
    col_work[asub[i]]++;
  }
  
  /*Compute the index of each rows*/
  
  xa_work[0] = 0;
  
  for(i=0;i<n;i++){
    xa_work[i+1] = xa_work[i] + col_work[i];
    
    col_work[i] = 0; /*reused as current position for inserting the elements in the next step*/
  }
  
  
  /*transfer A to Awork*/
  
  for (j=0; j<m; j++){
    for (i=(*xa)[j]; i<(*xa)[j+1]; i++){
      /* copy element (asub[i], j, a[i]) to the final position*/
      ipos = asub[i];
      pos = xa_work[ipos]+col_work[ipos];
      col_work[ipos]++;
      
      asub_work[pos] = j;
      a_work[pos] = a[i]; 
    }
  }
  

  /* Copy back the matrix */
  
  /* If the matrix is rectangular xa_work maight be larger than xa*/
  if(m==n){
    for(i=0;i<n+1;i++){
      (*xa)[i] = xa_work[i];
    }
  }else{
    free(*xa);
    *xa = xa_work;
  }

  for(i=0;i<nnz;i++){
    asub[i] = asub_work[i];
    a[i]    = a_work[i];
  }
  
  /* Free memory*/
  free(a_work);
  free(asub_work);
  if(m==n)
    free(xa_work);
  free(col_work);
  
  return 0;

}

/*
 * Convert a matrix from csr to dense
 */

int preAlps_matrix_convert_csr_to_dense(int m, int n, int *xa, int *asub, double *a,   preAlps_matrix_layout_t mlayout, double *a1, int lda1){
  int i,j;
  
  for(i=0;i<m*n;i++) a1[i] = 0.0;
  
  for (i=0; i<m; i++){
    for (j=xa[i]; j<xa[i+1]; j++){
		
      if(mlayout==MATRIX_ROW_MAJOR) 
        a1[i*lda1+asub[j]] = a[j];
      else
        a1[asub[j]*lda1+i] = a[j];
    }  
  }
  
  return 0;
}

/* copy matrix A to A1 */
void preAlps_matrix_copy(int m, int *xa, int *asub, double *a, int *xa1, int *asub1, double *a1){
  
  int i,j;
  for (i=0; i<m; i++){
    xa1[i] = xa[i];
    for (j=xa[i]; j<xa[i+1]; j++){
      asub1[j] = asub[j];
    a1[j] = a[j];
    }  
    }
  
  xa1[m] = xa[m];
}

/*print a dense matrix*/
void preAlps_matrix_dense_print(preAlps_matrix_layout_t mlayout, int m, int n, double *a, int lda, char *s){
#ifdef DEBUG
  int i,j;
  if(s) printf("%s\n", s);
  for (i=0; i<m; i++){
    printf("row :%d\n", i);
    for (j=0; j<n; j++){
      if(mlayout==MATRIX_ROW_MAJOR)
        printf("%.2f ", a[i*lda+j]);
      else
        printf("%.2f ", a[j*lda+i]);
    }
    printf("\n");
   }
#endif
}


#ifdef USE_HPARTITIONING
/*
 * Partition a matrix using and hypergraph to represent its structure
 * part_loc:
 *     output: part_loc[i]=k means rows i belongs to subdomain k
 */
int preAlps_matrix_hpartition_sequential(int m, int n, int *xa, int *asub, int nparts, int *part){
  
  PaToH_Parameters patoh_args;
  
  int *cwghts;
  int *nwghts;
  int *partweights;
  int edgecut = 0;
  int i,err = 0;
  
  
  /*quick return*/
  if(nparts==1){
    
    for(i=0;i<m;i++) part[i] = 0;
    /* avoid division per zero bug when calling METIS with nparts=1*/
    return 0;
  }
  
  if ( !(cwghts  = (int *) malloc(m*sizeof(int))) ) s_abort("Malloc fails for cwghts[].");
  if ( !(nwghts  = (int *) malloc(n*sizeof(int))) ) s_abort("Malloc fails for cwghts[].");
  if ( !(partweights  = (int *) malloc(nparts*sizeof(int))) ) s_abort("Malloc fails for partweights[].");
  
  
  
  /* Set  parameters */
  //PaToH_Initialize_Parameters(&patoh_args, PATOH_CUTPART, PATOH_SUGPARAM_DEFAULT);
  PaToH_Initialize_Parameters(&patoh_args, PATOH_CUTPART, PATOH_SUGPARAM_QUALITY);
  
  patoh_args._k = nparts; /*Number of partitions to be created*/ 

  patoh_args.seed = 42; /* seed */ /*-1 for random seed*/
  
  
  for(i=0;i<m;i++) part[i]=-1;
  
  /*no constraints*/
  for(i=0;i<m;i++) {cwghts[i]=1;} 
  for(i=0;i<n;i++) {nwghts[i]=1;}
  
  PaToH_Partition(&patoh_args, m, n, cwghts, nwghts, xa, asub, part, partweights, &edgecut); 

  free(cwghts);
  free(nwghts);
  free(partweights);
  
  return err;  
}
#else
/*
 * Partition a matrix using Metis
 * part_loc:
 *     output: part_loc[i]=k means rows i belongs to subdomain k
 */
int preAlps_matrix_partition_sequential(int m, int *xa, int *asub, int nparts, int *part){
  
  
  
  idx_t options[METIS_NOPTIONS];
  

  idx_t ncon = 1;
  
  idx_t edgecut = 0;

  int i,err = 0;
  int nz = xa[m];
  
  /* Casting variables to silent metis Warning*/
  idx_t m_idx;
  idx_t nparts_idx;
  
  idx_t *xa_idx;
  idx_t *asub_idx;
  idx_t *part_idx;
  
  /*quick return*/
  if(nparts==1){
    
    for(i=0;i<m;i++) part[i] = 0;
    /* avoid division per zero bug when calling METIS with nparts=1*/
    return 0;
  }
  
  METIS_SetDefaultOptions(options);
    
  
  options[METIS_OPTION_NUMBERING] = 0;
  options[METIS_OPTION_SEED] = 42; /* Fixed Seed for reproducibility */
  
  m_idx      = (idx_t) m;
  nparts_idx = (idx_t) nparts;
  
  if (sizeof (int) == sizeof (idx_t)){
    
    xa_idx     = (idx_t*) xa;
    asub_idx   = (idx_t*) asub;
    part_idx   = (idx_t*) part;  
    
  }else{
    
    //s_abort("Size of int and idx_t differs");
    if ( !(xa_idx     = (idx_t *) malloc((m+1)*sizeof(idx_t))) ) s_abort("Malloc fails for xa_idx[].");
    if ( !(asub_idx   = (idx_t *) malloc(nz*sizeof(idx_t))))     s_abort("Malloc fails for asub_idx[].");
    if ( !(part_idx   = (idx_t *) malloc(m*sizeof(idx_t))) )     s_abort("Malloc fails for part_idx[].");
    
    for(i=0;i<m+1;i++) xa_idx[i] = (idx_t) xa[i];
    for(i=0;i<nz;i++)  asub_idx[i] = (idx_t) asub[i];
  }
  
  err = METIS_PartGraphKway(&m_idx, &ncon, xa_idx, asub_idx,
                            NULL, NULL, NULL, &nparts_idx, NULL,
                            NULL, options, &edgecut, part_idx);  
  
   if (sizeof (int) != sizeof (idx_t)){
     
     for(i=0;i<m;i++) part[i] = (int) part_idx[i];
     
     free(xa_idx);
     free(asub_idx);
     free(part_idx);
   }  
           
  if(err!=METIS_OK) {printf("METIS returned error:%d\n", err); s_abort("Metis Failed.");}
  
  return err;  
}
#endif






/* 
 * Compute A1 = A(pinv,q) where pinv and q are permutations of 0..m-1 and 0..n-1. 
 * if pinv or q is NULL it is considered as the identity
 */
void preAlps_matrix_permute (int n, int *xa, int *asub, double *a, int *pinv, int *q,int *xa1, int *asub1,double *a1)
{
  int j, jp, i, nz = 0;
    
  for (i = 0 ; i < n ; i++){
    xa1 [i] = nz ;  
    jp = q==NULL ? i: q [i];
    for (j = xa [jp] ; j < xa [jp+1] ; j++){
        asub1 [nz] = pinv==NULL ? asub [j]: pinv [asub [j]]  ;
        a1 [nz] = a [j] ;
        nz++;
    }
  }
  
  xa1 [n] = nz ;  
  /*Sort the row index of the matrix*/
  preAlps_matrix_colIndex_sort(n, xa1, asub1, a1);    
}

/* Print a CSR matrix */
void preAlps_matrix_print(preAlps_matrix_storage_t mtype, int m, int *xa, int *asub, double *a, char *s){
#ifdef DEBUG  
  int i,j;
  if(s) printf("%s\n", s);
  for (i=0; i<m; i++){
    for (j=xa[i]; j<xa[i+1]; j++){
      
      if(mtype==MATRIX_CSR)
        printf("%d %d %20.19g\n", i, asub[j], a[j]);
      else
        printf("%d %d %20.19g\n", asub[j], i, a[j]);
    }
  }
#endif      
}

/* Read a matrix market data file and stores the matrix using CSC format */
int preAlps_matrix_readmm_csc(char *filename, int *m, int *n, int *nnz, int **xa, int **asub, double **a){
  int ret_code;
  MM_typecode matcode;
  FILE *f;
  int M, N, nz;   
  int i, j, a_row, a_col;
  double a_val;

  int *xa_ptr, *asub_ptr;
  double *a_ptr;
  
  int *xa2, *asub2, nz2;
  double *a2;
  
  if ((f = fopen(filename, "r")) == NULL) {
        printf("Could not open the file: %s.\n", filename);
        exit(1);
  }
      

  if (mm_read_banner(f, &matcode) != 0){
          printf("Could not process Matrix Market banner.\n");
          exit(1);
  }


  /*  This is how one can screen matrix types if their application */
  /*  only supports a subset of the Matrix Market data types.      */

  if (mm_is_complex(matcode) && mm_is_matrix(matcode) && 
              mm_is_sparse(matcode) ){
          printf("Sorry, this application does not support ");
          printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
          exit(1);
  }

  /* find out size of sparse matrix .... */

  if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
        exit(1);


  /* reserve memory for matrices */

  if ( !(*a = (double *) malloc(nz*sizeof(double))) ) s_abort("Malloc fails for a[].");
  if ( !(*asub = (int *) malloc(nz*sizeof(   int))) ) s_abort("Malloc fails for asub[].");
  if ( !(*xa = (int *)   malloc((N+1)*sizeof( int))) ) s_abort("Malloc fails for xa[].");
    
  
  
  xa_ptr   = *xa;
  asub_ptr = *asub;
  a_ptr    = *a;
    
  /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
  /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
  /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

  j = 0;
  xa_ptr[0] = xa_ptr[1] = 0;
  
    
  for (i=0; i<nz; i++){
    
      fscanf(f, "%d %d %lg\n", &a_row, &a_col, &a_val);
    
    //printf("i:%d, j:%d, val=%f\n",a_row,a_col,a_val);
    
    /* adjust from 1-based to 0-based */
    a_row--; a_col--;
    
    
    
    /*Some columns might be empty, in that case we need to set the number of elements to 0.*/
    while(j < a_col) {
      
      j++;
      xa_ptr[j+1]=xa_ptr[j];
    }
    
    /*Increase the number of elements for the current column*/
    xa_ptr[j+1]++;

      asub_ptr[i] = a_row;
      a_ptr[i] = a_val;

  }

  printf("\n");
  
  /*Some of the last columns might be empty, in that case we need to set the number of elements to 0.*/
  while(j < N - 1) {
    
    j++;
    xa_ptr[j+1] = xa_ptr[j];
  }
  
  if (f !=stdin) fclose(f);



    if (mm_is_symmetric(matcode)){
    
    preAlps_matrix_symmetrize(M, N, nz, xa_ptr, asub_ptr, a_ptr, &nz2, &xa2, &asub2, &a2);
    
    
    preAlps_matrix_print(MATRIX_CSC, N, xa2, asub2, a2, "Symmetrized CSC matrix");    
    
    free(xa_ptr);
    free(asub_ptr);
    free(a_ptr);
    
    xa_ptr   = xa2;
    asub_ptr = asub2;
    a_ptr    = a2;
    nz = nz2;
    }  

  /************************/
  /* now write out matrix */
  /************************/

  mm_write_banner(stdout, matcode);
  mm_write_mtx_crd_size(stdout, M, N, nz);

  preAlps_matrix_print(MATRIX_CSC, M, xa_ptr, asub_ptr, a_ptr, "CSC matrix");
  
    
  *m     = M;
  *n     = N;
  *nnz   = nz;
  
  *xa   = xa_ptr;
  *asub   = asub_ptr;
  *a     = a_ptr;
  
  return 0;
}

/* Read a matrix market data file and stores the matrix using CSR format */
int preAlps_matrix_readmm_csr(char *filename, int *m, int *n, int *nnz, int **xa, int **asub, double **a){
  
  preAlps_matrix_readmm_csc(filename, m, n, nnz, xa, asub, a);
  
  /* Convert a matrix from csc to csr */
 preAlps_matrix_convert_csc_to_csr(*m, *n, xa, *asub, *a);
 
 return 0;
}

/*
 * Perform a product of two matrices, A(i_begin:i_end, :) stores as CSR and B as dense 
 *
 * ptrRowBegin:
 *    input: ptrRowBegin[i] = j means the first non zeros element of row i is in column j
 * ptrRowEnd:
 *    input: ptrRowEnd[i] = j means the last non zeros element of row i is in column j
 */
int preAlps_matrix_subMatrix_CSRDense_Product(int m, int *ptrRowBegin, int *ptrRowEnd, int a_colOffset, int *asub, double *a, double *b, int ldb, int b_nbcol, double *c, int ldc){
  
  int ip, k, i,j ;

  for(k=0;k<b_nbcol;k++){
    
    for (i=0; i<m; i++){  
      ip = i+k*ldb;
          c[ip] = 0.0;
        for (j=ptrRowBegin[i]; j<ptrRowEnd[i]; j++){
#ifdef DEBUG        
        printf("Computing c[%d]+=a(%d,%d):%e a[%d]:%e x b[%d - offset:%d]:%e\n", ip, i, asub[j], a[j], j, a[j], asub[j]+k*ldb, a_colOffset, b[asub[j]+k*ldb - a_colOffset]);
#endif        
          c[ip] += a[j]*b[asub[j]+k*ldb - a_colOffset];
        }
    }
  }
  
  return 0;
}


/* Create a full symmetric from a lower triangular matrix stored in CSC format */
int preAlps_matrix_symmetrize(int m, int n, int nnz, int *xa, int *asub, double *a, int *nnz2, int **xa2, int **asub2, double **a2){
  
  int *xa_work, *asub_work;
  double *a_work;
  int i,j, jpos;
  int *nnz_work;
  
  /* Allocate memory */
  if ( !(a_work = (double *) malloc(2*nnz*sizeof(double))) ) s_abort("Malloc fails for a_work[].");
  if ( !(asub_work = (int *) malloc(2*nnz*sizeof(   int))) ) s_abort("Malloc fails for asub_work[].");
  if ( !(xa_work = (int *)   malloc((n+1)*sizeof( int))) ) s_abort("Malloc fails for xa_work[].");
  
  
  if ( !(nnz_work = (int *)   malloc((n+1)*sizeof( int))) ) s_abort("Malloc fails for row_work[].");
  
  
  for(i=0;i<n+1;i++){
    nnz_work[i] = 0;
  }
  
  /*Compute the number of nnz per columns*/
  for (j=0; j<n; j++)
    for (i=xa[j]; i<xa[j+1]; i++)
    {
          
      nnz_work[j]++;
      
      if(j!=asub[i])
        nnz_work[asub[i]]++;
    }
  
  /*Compute the index of each row*/
  
  xa_work[0] = 0;
  
  for(i=0;i<n;i++){
    xa_work[i+1] = xa_work[i] + nnz_work[i];
    
    nnz_work[i] = xa_work[i]; /*reused as current position for inserting the elements in the next step*/
  }

  /* Fill the matrix */
  
  for (j=0; j<n; j++){
    for (i=xa[j]; i<xa[j+1]; i++){
      
      /* insert (asub[i], j) the element in column j */
      
      jpos = nnz_work[j];
      asub_work[jpos] = asub[i];
      a_work[jpos] = a[i];
      nnz_work[j]++;
      
      /* insert (j, asub[i]) the element in column asub[i] */
      if(j!=asub[i]){
        jpos = nnz_work[asub[i]];
        asub_work[jpos] = j;
        a_work[jpos] = a[i];
        nnz_work[asub[i]]++;
        }
    }
  }
  
  *a2     = a_work;
  *asub2  = asub_work;
  *xa2    = xa_work;
  
  *nnz2 = xa_work[n];
  
  /* Free memory*/
  free(nnz_work);
  
  return 0;

}
