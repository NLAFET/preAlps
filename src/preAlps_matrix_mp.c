/*
 ============================================================================
 Name        : preAlps_matrix_mp.c
 Author      : Simplice Donfack
 Version     : 0.1
 Description : Parallel matrix utilities
 Date        : Sept 27, 2016
 ============================================================================
 */
#include <mpi.h>
#include "preAlps_matrix.h"
#include "preAlps_matrix_mp.h"
#include "s_utils_mp.h"

/* 
 * 1D block rows gatherv of the matrix to from the processors in the communicator using ncounts and displs.
 * The result is stored on processor 0.
 * ncounts: ncounts[i] = k means processor i has k rows. 
 */
int preAlps_matrix_assemble(MPI_Comm comm, int mloc, int *xa, int *asub, double *a, int *ncounts, int **xa1, int **asub1, double **a1){
  
  int nbprocs, my_rank;
  int root = 0;
  
  int *nzcounts=NULL, *nzoffsets=NULL, *noffsets=NULL;
  
  int *xa1_ptr, *asub1_ptr;
  double *a1_ptr;
  
  int i, m = 0, j, nz = 0, pos, nzloc ;
  
  
  MPI_Comm_size(comm, &nbprocs);

  MPI_Comm_rank(comm, &my_rank);
  
  
  /* The root allocate memory for xa*/
  
  if(my_rank==root){
    
    /* determine the matrix number of rows*/
    m = 0;
    
    for(i=0;i<nbprocs;i++){
      m+=ncounts[i];
    }
    
    if ( !(*xa1 = (int *)   malloc((m+1)*sizeof(int))) ) s_abort("Malloc fails for xa[].");
    
    if ( !(noffsets  = (int *) malloc((nbprocs+1)*sizeof(int))) ) s_abort("Malloc fails for noffsets[].");
    
    if ( !(nzcounts  = (int *) malloc(nbprocs*sizeof(int))) ) s_abort("Malloc fails for nzcounts[].");
    
    
    /* Compute the offset */
    
    noffsets[0] = 0;
    for(i=0;i<nbprocs;i++){

      noffsets[i+1] = noffsets[i] + ncounts[i]; 
    }
    
    /* noffsets[i+1] - noffsets[i] returns the number of rows for processor i*/
    noffsets[nbprocs] = m;
    
    
  }
  
  xa1_ptr = *xa1;
  
  
  /* Shift to take into account the first element */
  if(my_rank==0) {
    
    xa1_ptr[0] = 0; //first element
    
    for(i=0;i<nbprocs+1;i++){

      noffsets[i]++;
    }
  }
  
  /*Each process send mloc element to proc 0*/
  MPI_Gatherv(&xa[1], mloc, MPI_INT, xa1_ptr, ncounts, noffsets, MPI_INT, root, comm);  
  
  
  /* convert xa from local to global by adding the last element of each subset*/
  if(my_rank==root){
    
    for(i=1;i<nbprocs;i++){
      
      pos = noffsets[i];
      
      for(j=0;j<ncounts[i];j++){
        
        /*add the number of non zeros of the previous proc */
        xa1_ptr[pos+j] = xa1_ptr[pos+j] + xa1_ptr[pos-1];
        
      }
    }
    
  }
  
  /* Compute number of non zeros in each rows*/
  
  if(my_rank==0){
  
    if ( !(nzoffsets = (int *) malloc((nbprocs+1)*sizeof(int))) ) s_abort("Malloc fails for nzoffsets[].");
    
    nzoffsets[0] = 0; nz = 0;
    for(i=0;i<nbprocs;i++){
    
    
      nzcounts[i] = xa1_ptr[noffsets[i+1]-1] - xa1_ptr[noffsets[i]-1];

      nzoffsets[i+1] = nzoffsets[i] + nzcounts[i];
      
      nz+=nzcounts[i];
    }
  
    if ( !(*asub1 = (int *)   malloc((nz*sizeof(int)))) ) s_abort("Malloc fails for asub[].");
    if ( !(*a1 = (double *)   malloc((nz*sizeof(double)))) ) s_abort("Malloc fails for a[].");
  }

  asub1_ptr = *asub1;
  a1_ptr = *a1;

  /*gather ja and a*/
  
  nzloc = xa[mloc];
  
  s_int_print_mp(comm, nzloc, "nzloc");
  
  
    MPI_Gatherv(asub, nzloc, MPI_INT, asub1_ptr, nzcounts, nzoffsets, MPI_INT, root, comm);  
  
    MPI_Gatherv(a, nzloc, MPI_DOUBLE, a1_ptr, nzcounts, nzoffsets, MPI_DOUBLE, root, comm);  
  
  if(my_rank==0){
    
    free(noffsets);
    
    free(nzcounts);
    free(nzoffsets);
  }

  return 0;
}

/*
 * Create a sparse block structure of a CSR matrix.
 * The matrix is initially 1D row block distributed.
 * n:
 *    input: global number of columns
 * nparts:
 *    input: number of domain
 * ncounts:
 *     input: number of columns in each subdomain
 * ABlockStruct
 *    output: dense matrix of size (nbprocs x nparts) stored using ROW_MAJOR_LAYOUT
 */
int preAlps_matrix_createBlockStruct(MPI_Comm comm, preAlps_matrix_storage_t mtype, int mloc, int nloc, int *xa, int *asub, int nparts, int *ncounts, int *ABlockStruct){
  
  int nbprocs, my_rank;
  int *RowBlockStruct;
  
  MPI_Comm_size(comm, &nbprocs);

  MPI_Comm_rank(comm, &my_rank);
  
  
  if ( !(RowBlockStruct = (int *)   malloc(nparts*sizeof( int))) ) s_abort("Malloc fails for RowBlockStruct[].");
  
  
  /*Create the sparse block column struct of the local matrix*/  
  preAlps_matrix_ColumnBlockStruct(mtype, mloc, nloc, xa, asub, nparts, ncounts, RowBlockStruct);
  
  
  
  s_ivector_print_mp (comm, RowBlockStruct, nparts, "RowBlockStruct", "");
  
  
  
  /*
  int ABlockStruct_size = nbprocs*nparts;
  MPI_Gather(RowBlockStruct, nparts, MPI_INT,
                   ABlockStruct, nparts, MPI_INT, root, comm);
  
  //Broadcast part to each proc
  MPI_Bcast(ABlockStruct, ABlockStruct_size, MPI_INT, root, comm);
  */
  
  MPI_Allgather(RowBlockStruct, nparts, MPI_INT, ABlockStruct, nparts, MPI_INT, comm);
						 
  
  free(RowBlockStruct);
  
  
  return 0;
}

/* 
 * 1D block rows distribution of the matrix to all the processors in the communicator.
 * Each proc has approximatively the same number of rows.
 */

int preAlps_matrix_distribute(MPI_Comm comm, int m, int **xa, int **asub, double **a, int *mloc){
  
  int nbprocs, my_rank;
  
  int *ncounts=NULL;
  int  offset;
  
  int i;
  
  MPI_Comm_size(comm, &nbprocs);

  MPI_Comm_rank(comm, &my_rank);
  
  
  
  /* Compute the local matrix dimension */
  s_nsplit(m, nbprocs, my_rank, mloc, &offset);
  
  /* Compute the number of rows and offsets for each procs*/
  if(my_rank==0){
    
    if ( !(ncounts   = (int *) malloc(nbprocs*sizeof(int))) ) s_abort("Malloc fails for ncounts[].");
    
    
    for(i=0;i<nbprocs;i++){
      s_nsplit(m, nbprocs, i, &ncounts[i], &offset);
    }
    
  }
    
  /* Distribute the matrix */
  preAlps_matrix_scatterv(comm, *mloc, xa, asub, a, ncounts);
  
  
  if(my_rank==0){
    
    free(ncounts);
    
  }

  
  return 0;
}






/*
 * Assemble a matrix from all the processors in the communicator.
 * Each procs hold a matrix initially 1D block rows distributed.
*/
int preAlps_matrix_kpart_assemble(MPI_Comm comm, int m, int n, int *xa, int *asub, double *a, int mloc, int *part, int **xa1, int **asub1, double **a1){
  
  int nbprocs, my_rank;
  
  int *ncounts = NULL;
  int i;
  
  MPI_Comm_size(comm, &nbprocs);

  MPI_Comm_rank(comm, &my_rank);
  
  
  
  if(my_rank==0){
  
    if ( !(ncounts   = (int *) malloc(nbprocs*sizeof(int))) ) s_abort("Malloc fails for ncounts[].");
  }
  
  /*Compute the new number of columns per process*/
  
  if(my_rank==0){
    
    /*Compute the number of rows per process*/
    for(i=0;i<nbprocs;i++){
      ncounts[i] = 0; 
    }
    
    for(i=0;i<m;i++){
      ncounts[part[i]]++;
    }
  }
  
  preAlps_matrix_assemble(comm, mloc, xa, asub, a, ncounts, xa1, asub1, a1);

    if(my_rank==0){
      free(ncounts);
    }
  
  return 0;
}

/*
 * 1D block rows redistribution of the matrix to each proc after a partitioning.
 * Each procs hold a matrix initially distributed using s_nsplit.
 * part_loc: part_loc[i]=k means row i belongs to subdomain k.
*/
int preAlps_matrix_kpart_redistribute(MPI_Comm comm, int m, int n, int **xa, int **asub, double **a, int *mloc, int *part){
  
  int nbprocs, my_rank;
  
  int *ncounts = NULL, *noffsets = NULL;
  int i, nz;
  
  int *perm, *part_tmp;
  int *xa_work, *asub_work;
  double *a_work;
  
  
  
  MPI_Comm_size(comm, &nbprocs);

  MPI_Comm_rank(comm, &my_rank);
  
  
  if(my_rank==0){
    
    if ( !(ncounts   = (int *) malloc(nbprocs*sizeof(int))) ) s_abort("Malloc fails for ncounts[].");
    if ( !(noffsets   = (int *) malloc((nbprocs+1)*sizeof(int))) ) s_abort("Malloc fails for noffsets[].");
  }
  
  if ( !(perm   = (int *) malloc(m*sizeof(int))) ) s_abort("Malloc fails for perm[].");
  if ( !(part_tmp   = (int *) malloc(m*sizeof(int))) ) s_abort("Malloc fails for part_tmp[].");
               
  
  /* Create a permutation vector from part */
  s_partitionVector_to_permVector(part, m, nbprocs, perm);
  
  s_ivector_print_mp (comm, perm, m, "perm", "");
  
  /* Permute part to order the rows part = inv(perm)*part */
  s_perm_vec_int (perm, part, part_tmp, m);
  
  s_ivector_print_mp (comm, part_tmp, m, "part_tmp", "");
  
  for(i=0;i<m;i++) part[i] = part_tmp[i];
  
  
  /* Compute the new number of columns per domain */
  
  if(my_rank==0){
    
    /*Compute the number of rows per domain */
    
    for(i=0;i<nbprocs;i++){
      ncounts[i] = 0; 
    }
    
    for(i=0;i<m;i++){
      ncounts[part[i]]++;
    }
  }
  
  /*Compute local number of rows after the distribution*/
  *mloc = 0;
  for(i=0;i<m;i++) 
    if(part[i]==my_rank) (*mloc)++;
  
  
  s_int_print_mp(comm, *mloc, "redistributed mloc");
  
  
    /*permute the initial matrix*/
    if(my_rank==0){
  
    int *iperm;

    iperm = s_return_pinv(perm, m);

#ifdef DEBUG    
    for(i=0;i<m;i++){
      printf("perm[%d]=%d\tiperm[%d]=%d\n", i, perm[i], i, iperm[i]);
    } 
#endif
      
     nz = (*xa)[m];
      
      if ( !(a_work = (double *) malloc(nz*sizeof(double))) ) s_abort("Malloc fails for a_work[].");
      if ( !(asub_work = (int *) malloc(nz*sizeof(   int))) ) s_abort("Malloc fails for asub_work[].");
      if ( !(xa_work = (int *)   malloc((n+1)*sizeof( int))) ) s_abort("Malloc fails for xa_work[].");
  
  
    preAlps_matrix_permute (m, *xa, *asub, *a, iperm, perm, xa_work, asub_work, a_work);
  
    preAlps_matrix_print(MATRIX_CSR, m, xa_work, asub_work, a_work, "permuted matrix");
    
    /* Free the old matrix*/
      free(*a);
      free(*asub);
      free(*xa);
  
    *xa   = xa_work;
    *asub   = asub_work;
    *a     = a_work;
    
    free(iperm);
    
  }else{
    if(*xa!=NULL)   free(*xa);
    if(*asub!=NULL) free(*asub);
    if(*a!=NULL)    free(*a);
  }
  
  /*redistribute the permuted matrix*/    
  preAlps_matrix_scatterv(comm, *mloc, xa, asub, a, ncounts);
  
  
  free(perm);
  free(part_tmp);
  
  if(my_rank==0){  
    free(ncounts);
    free(noffsets);
  }
  
  return 0;
}


#ifdef USE_PARMETIS
/*
 * Partition a matrix using parMetis
 * part_loc:
 *     output: part_loc[i]=k means rows i belongs to subdomain k
 */
int preAlps_matrix_partition(MPI_Comm comm, int *vtdist, int *xa, int *asub, int nparts, int *part_loc){
  
  int nbprocs;
  
  int options[METIS_NOPTIONS];
  
  int wgtflag = 0; /*No weights*/
  int numflag = 0; /*C-style numbering*/
  int ncon = 1;
  
  
  int edgecut = 0;
  float *tpwgts;
  float *ubvec;
  
  int i, err = 0;
  
  
  MPI_Comm_size(comm, &nbprocs);
  
  if ( !(tpwgts = (float *)   malloc((nparts*ncon*sizeof(float)))) ) s_abort("Malloc fails for tpwgts[].");
  if ( !(ubvec = (float *)    malloc((ncon*sizeof(float)))) ) s_abort("Malloc fails for ubvec[].");
  
  
  options[0] = 0;
  options[1] = 0;
  options[2] = 42; /* Fixed Seed for reproducibility */
  
  
  for(i=0;i<nparts*ncon;i++) tpwgts[i] = 1.0/(real_t)nparts;
  
  for(i=0;i<ncon;i++) ubvec[i] =  1.05;
  
  err = ParMETIS_V3_PartKway(vtdist, xa, asub, NULL, NULL, 
      &wgtflag, &numflag, &ncon, &nparts, tpwgts, ubvec, options, &edgecut, 
        part_loc, &comm);
        
    if(err!=METIS_OK) {printf("METIS returned error:%d\n", err); s_abort("Metis Failed.");}
  
  free(tpwgts);
  free(ubvec);
    
  return err;  
}
#endif


/* Every process print its local matrix*/
void preAlps_matrix_print_global(MPI_Comm comm, preAlps_matrix_storage_t mtype, int m, int *xa, int *asub, double *a, char *s){
#ifdef DEBUG  
  int i, my_rank, comm_size;
  
  int mp1_recv, nz, nz_recv, m_max, nz_max;
  
  int *buffer_xa, *buffer_asub;
  double *buffer_a;
  
  int TAG_WRITE = 4;
  MPI_Status status;
  
    MPI_Comm_rank(comm, &my_rank);
    MPI_Comm_size(comm, &comm_size);
  
  
  m_max = m;
  nz_max = nz = xa[m];
  
  if(my_rank ==0){
      
    /* Allocate the max buffer */
    buffer_xa =  (int*) malloc((m_max+1) * sizeof(int));
    buffer_asub =  (int*) malloc(nz_max * sizeof(int));
    buffer_a =  (double*) malloc(nz_max * sizeof(double));
    
    printf("[%d] %s\n", 0, s);
    
    preAlps_matrix_print(mtype, m, xa, asub, a, NULL);
    
    for(i = 1; i < comm_size; i++) {
      
      /*Receive the dimension*/      
      MPI_Recv(&mp1_recv,  1, MPI_INT, i, TAG_WRITE, comm, &status);
      MPI_Recv(&nz_recv, 1, MPI_INT, i, TAG_WRITE, comm, &status);
      
      
      if(mp1_recv>m_max){
        
        /* Redim the buffer */
        free(buffer_xa);
        m_max = mp1_recv;
        buffer_xa =  (int*) malloc(m_max * sizeof(int));
      }

      MPI_Recv(buffer_xa, mp1_recv, MPI_INT, i, TAG_WRITE, comm, &status);

      
      
      if(nz_recv>nz_max){
        
        /* Redim the buffer */
        free(buffer_asub);
        free(buffer_a);
        
        nz_max = nz_recv;
        
        buffer_asub =  (int*) malloc(nz_max * sizeof(int));
        buffer_a =  (double*) malloc(nz_max * sizeof(double));
      }

      MPI_Recv(buffer_asub, nz_recv, MPI_INT, i, TAG_WRITE, comm, &status);
      
      MPI_Recv(buffer_a, nz_recv, MPI_DOUBLE, i, TAG_WRITE, comm, &status);
      
      printf("[%d] %s\n", i, s);
      
      preAlps_matrix_print(mtype, mp1_recv-1, buffer_xa, buffer_asub, buffer_a, NULL);
        }
    printf("\n");
    
    free(buffer_xa);
    free(buffer_asub);
    free(buffer_a);
  }
  else{
    int mp1 = m+1;
    MPI_Send(&mp1, 1, MPI_INT, 0, TAG_WRITE, comm);
    MPI_Send(&nz, 1, MPI_INT, 0, TAG_WRITE, comm);
    MPI_Send(xa, mp1, MPI_INT, 0, TAG_WRITE, comm);
    MPI_Send(asub, nz, MPI_INT, 0, TAG_WRITE, comm);
    MPI_Send(a, nz, MPI_DOUBLE, 0, TAG_WRITE, comm);
  }
  

  MPI_Barrier(comm);
#endif
}

/* Only one process print its matrix, forces synchronisation between all the procs in the communicator*/
void preAlps_matrix_print_single(MPI_Comm comm, preAlps_matrix_storage_t mtype, int root, int m, int *xa, int *asub, double *a, char *s){
#ifdef DEBUG  
  int nbprocs, my_rank;
  
    MPI_Comm_rank(comm, &my_rank);
    MPI_Comm_size(comm, &nbprocs);
  
  if(my_rank==root) preAlps_matrix_print(mtype, m, xa, asub, a, s);
  
  MPI_Barrier(comm);
#endif
}

/* 
 * 1D block rows distribution of the matrix to all the processors in the communicator using ncounts and displs.
 * The data are originally stored on processor 0.
 * ncounts: ncounts[i] = k means processor i has k rows. 
 */
int preAlps_matrix_scatterv(MPI_Comm comm, int mloc, int **xa, int **asub, double **a, int *ncounts){
  
  int nbprocs, my_rank;
  int root = 0;
  
  int *nzcounts=NULL, *nzoffsets=NULL, *noffsets=NULL, *nxacounts = NULL;
  int nzloc;
  
  int *xa_ptr, *asub_ptr;
  double *a_ptr;
  int i,m;
  
  
  
  MPI_Comm_size(comm, &nbprocs);

  MPI_Comm_rank(comm, &my_rank);
  
  
  /* Compute the offsets */
  if(my_rank==0){
    
    if ( !(noffsets  = (int *) malloc((nbprocs+1)*sizeof(int))) ) s_abort("Malloc fails for noffsets[].");
    if ( !(nxacounts  = (int *) malloc((nbprocs+1)*sizeof(int))) ) s_abort("Malloc fails for noffsets[].");
  }
  
  
  /*Compute the new number of columns per process*/
  
  if(my_rank==0){

    /* determine the matrix number of rows*/
    m = 0;
    
    for(i=0;i<nbprocs;i++){
      m+=ncounts[i];
    }
    
    /*Compute the offset */  
    noffsets[0] = 0;
    
    for(i=0;i<nbprocs;i++){
      
      noffsets[i+1] = noffsets[i] + ncounts[i];
      
      nxacounts[i] = ncounts[i]+1;  /* add the n+1-th element required for the CSR format */
    }
    
    /* noffsets[i+1] - noffsets[i] returns the number of rows for processor i*/
    noffsets[nbprocs] = m;
  }
  
  /* The other process allocate memory for xa*/
  if(my_rank!=root){
    
    if ( !(*xa = (int *)   malloc((mloc+1)*sizeof(int))) ) s_abort("Malloc fails for xa[].");
    
  }
  
  xa_ptr = *xa;
  
  
  /* Distribute xa to each procs. Each proc has mloc+1 elements */
  
  MPI_Scatterv(xa_ptr, nxacounts, noffsets, MPI_INT, my_rank==0?MPI_IN_PLACE:xa_ptr, mloc+1, MPI_INT, root, comm);
  
  s_ivector_print_mp (comm, xa_ptr, mloc+1, "xa", "");
  
  /* Convert xa from global to local */
  for(i=mloc;i>=0;i--){
  
    xa_ptr[i] = xa_ptr[i] - xa_ptr[0];
    
  }
  
  s_ivector_print_mp (comm, xa_ptr, mloc+1, "xa", "xa local");
  
  /* 
   * Distribute asub and a to each procs 
   */
  
  nzloc = xa_ptr[mloc]; // - xa_ptr[0]


  /* Compute number of non zeros in each rows*/
  
  if(my_rank==0){
  
    if ( !(nzcounts  = (int *) malloc(nbprocs*sizeof(int))) ) s_abort("Malloc fails for nzcounts[].");
    if ( !(nzoffsets = (int *) malloc((nbprocs+1)*sizeof(int))) ) s_abort("Malloc fails for nzoffsets[].");
    
    
  
    nzoffsets[0] = 0;
    for(i=0;i<nbprocs;i++){
    
    
      nzcounts[i] = xa_ptr[noffsets[i+1]] - xa_ptr[noffsets[i]];

      nzoffsets[i+1] = nzoffsets[i] + nzcounts[i];
    }
  
  }else{
  
    if ( !(*asub = (int *)   malloc((nzloc*sizeof(int)))) ) s_abort("Malloc fails for asub[].");
    if ( !(*a = (double *)   malloc((nzloc*sizeof(double)))) ) s_abort("Malloc fails for a[].");
  }

  asub_ptr = *asub;;
  a_ptr = *a;

  /* Distribute ja and a */

  MPI_Scatterv(asub_ptr, nzcounts, nzoffsets, MPI_INT, my_rank==0?MPI_IN_PLACE:asub_ptr, nzloc, MPI_INT, root, comm);
  
  MPI_Scatterv(a_ptr, nzcounts, nzoffsets, MPI_DOUBLE, my_rank==0?MPI_IN_PLACE:a_ptr, nzloc, MPI_DOUBLE, root, comm);
  
  
  if(my_rank==0){
    
    free(noffsets);
    free(nxacounts);
    free(nzcounts);
    free(nzoffsets);
  }
  
  return 0;
}



/* Distribute a vector v as Block Diagonal CSR matrix */
int preAlps_matrix_vpart_distribute(MPI_Comm comm, int m, double *v, int nbpartcol, int *part,  int mloc, int *xb, int *bsub, double *b){
  
  int nbprocs, my_rank, root = 0;
  int *ncounts = NULL, *noffsets = NULL;
  int i, k;
  MPI_Comm_size(comm, &nbprocs);

  MPI_Comm_rank(comm, &my_rank);
  
  /*Compute the new number of columns per process*/
  
  if(my_rank==0){
    
    if ( !(ncounts   = (int *) malloc(nbprocs*sizeof(int))) ) s_abort("Malloc fails for ncounts[].");
    if ( !(noffsets  = (int *) malloc((nbprocs+1)*sizeof(int))) ) s_abort("Malloc fails for noffsets[].");
    
    /*Compute the number of rows per process*/
    for(i=0;i<nbprocs;i++){
      ncounts[i] = 0; 
    }
    
    for(i=0;i<m;i++){
      ncounts[part[i]]++;
    }  
    
    /*Compute the offset */
    
    noffsets[0] = 0;
    for(i=0;i<nbprocs;i++){

      noffsets[i+1] = noffsets[i] + ncounts[i];
    }  
  }
  
  /*Distribute the vector to each procs*/
  MPI_Scatterv(v, ncounts, noffsets, MPI_DOUBLE, b, mloc, MPI_DOUBLE, root, comm);
  
  
  /*The sub block vector received is dense*/
  
  xb[0] = 0, k =0;
  for (i=0; i<mloc; i++){
    xb[i+1] = xb[i]+1; /*Each row has at one element*/
    bsub[k++] = my_rank; /*fill the column with the rank of the process*/
  }
  
   
  if(my_rank==0){
    
    free(ncounts);
    free(noffsets);
  }
  
  return 0;
}

/* Distribute a vector v as Block Diagonal CSR matrix 
 * mloc:
 *     input: local number of rows
 * mcounts :
 *     input: number of rows for each part
 * b_nparts:
 *     input: number of block columns of the resulting matrix
 * v:
 *     input: initial vector to distribute
 * xb, bsub, b:
 *     output: sparse CSR matrix to create
*/
int preAlps_matrix_vshift_distribute(MPI_Comm comm, int mloc, int *ncounts, int b_nparts, double *v, int *xb, int **bsub, double **b){

  int nbprocs, my_rank, root = 0;
  int *noffsets = NULL;
  int i, k, nnz, J, Jblock_nprocs, offset, locPos, P_nbBlockCol, Jblock, level2_mloc, offset_level2;
  
  
  MPI_Comm_size(comm, &nbprocs);

  MPI_Comm_rank(comm, &my_rank);
  
  /*Compute the new number of columns per process*/
  
  if(my_rank==0){
    
    if ( !(noffsets  = (int *) malloc((nbprocs+1)*sizeof(int))) ) s_abort("Malloc fails for noffsets[]."); 
    
    /*Compute the offset */
    
      noffsets[0] = 0;
      for(i=0;i<nbprocs;i++){

        noffsets[i+1] = noffsets[i] + ncounts[i];
      }  
  } 
   
  /*Distribute the vector to each procs*/
  MPI_Scatterv(v, ncounts, noffsets, MPI_DOUBLE, my_rank==0?MPI_IN_PLACE:v, mloc, MPI_DOUBLE, root, comm);
  
  s_vector_print_mp (comm, v, mloc, "v", "v after scatterv");


  if(nbprocs>=b_nparts){
  	
    /* Determine my block column number */
	locPos = 0;
	for(J=0;J<b_nparts;J++){
		
      s_nsplit(nbprocs, b_nparts, J, &Jblock_nprocs, &offset);
		
	  if(locPos + Jblock_nprocs > my_rank){
        break;			
      }else{
        locPos+=Jblock_nprocs;		
	  }
    }
	
	s_int_print_mp(comm, J, "JBlock");
	s_int_print_mp(comm, locPos, "locPos");
	
	
    /*The sub block vector received is dense*/
	
	
    xb[0] = 0, k =0;
    for (i=0; i<mloc; i++){
      xb[i+1] = xb[i]+1; /*Each row has one element*/
      (*bsub)[k++] = J; 	/*fill the column with the rank of the process*/
	  
	  (*b)[i] = v[i];
    }
	
	s_vector_print_mp(comm, *b, mloc, "b", "b ");
	
  }else{ /* nbprocs < b_nparts*/
  	
	
	 /* Split the number of columns (b_nparts) among all processors and,
	    Determine my number of blocks 
	  */ 
	 s_nsplit(b_nparts, nbprocs, my_rank, &P_nbBlockCol, &Jblock);
	
	 s_int_print_mp(comm, Jblock, "Jblock");
	 s_int_print_mp(comm, P_nbBlockCol, "P_nbBlockCol");
	 
	 
	 /* Consider my subMatrix A(:, Jblock) of size (mloc x P_nbBlockCol) as dense and fill it with zeros*/
	 
	 /*Resize b*/
	 
	 nnz = mloc*P_nbBlockCol;
	 
	 free(*b);
	 free(*bsub);
	 
	 if ( !(*b  = (double *) malloc(nnz*sizeof(double))) )    s_abort("Malloc fails for b[].");
	 if ( !(*bsub  = (int *) malloc(nnz*sizeof(int))) ) s_abort("Malloc fails for bsub[].");
	 
	 /* Update the CSR indexes to indicates that the block A(:, Jblock) is dense*/
     xb[0] = 0; k =0;
     for (i=0; i<mloc; i++){
       xb[i+1] = xb[i]+P_nbBlockCol; //Each row has P_nbBlockCol elements
	   for(J=0;J<P_nbBlockCol;J++){
         (*bsub)[k++] = Jblock + J; 	 
       }
     }
	 
	 /* First fill it with zeros*/
	 for(J=0;J<P_nbBlockCol;J++){  	
		for (i=0; i<mloc; i++){
			(*b)[i*P_nbBlockCol + J] = 0.0;
		}
	 }
	 
	
	preAlps_matrix_print_global(comm, MATRIX_CSR, mloc, xb, *bsub, *b, "b as CSR diag Block with zeros");
	 
	 /* split my local number of rows and shift distribute to all my block columns */
	 
	 /* NOTE: if we know the number of rows in the second level of the partitioning,  it can be used here
	  * Otherwise, we equitably split and shift rows among the local number of columns 
	  */
	
	 k=0;
	 for(J=0;J<P_nbBlockCol;J++){
		  
		/* fill with b */
	 	s_nsplit(mloc, P_nbBlockCol, J, &level2_mloc, &offset_level2);
		
		for (i=0; i<level2_mloc; i++){
				
			(*b)[offset_level2*P_nbBlockCol + i*P_nbBlockCol + J] = v[offset_level2+i];//offset_level2+i;//k++
		}
		
	 }
	 
	 s_vector_print_mp(comm, *b, mloc*P_nbBlockCol, "b", "b ");
	 	 
  }

	
  if(my_rank==0){
    
    free(noffsets);
  }
		
	
  return 0;
}