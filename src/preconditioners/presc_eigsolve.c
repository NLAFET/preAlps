/*
============================================================================
Name        : presc_eigsolve.c
Author      : Simplice Donfack
Version     : 0.1
Description : Solve eigenvalues problem using ARPACK
Date        : Mai 15, 2017
============================================================================
*/
#include <stdlib.h>
#include <mpi.h>
#include "preAlps_utils.h"
#include "presc.h"

#include "preAlps_solver.h"

extern void pdsaupd_(MPI_Comm*, int*, char*, int*, char*, int*, double*, double*, int*, double*, int*,
                    int*, int*, double*, double*, int*, int*);

extern void pdnaupd_(MPI_Comm*, int*, char*, int*, char*, int*, double*, double*, int*, double*, int*,
                              int*, int*, double*, double*, int*, int*);

extern void pdseupd_(MPI_Comm*, int*, char*, int*, double*, double*, int*, double*, char*, int*,
                    char*, int*, double*, double*, int*, double*, int*, int*, int*,
                    double*, double*, int*, int*);


#define EIGVALUES_PRINT 0

#define USE_SYM 1

#define USE_GENERALIZED_SYSTEM 1


int Presc_eigSolve_init(char *bmat, char *which, int *maxit, int *iparam, double *eigs_tolerance, double *deflation_tol){

  int i;

  #if USE_GENERALIZED_SYSTEM
    *bmat = 'G'; //Generalized eigenvalue problem
  #else
    *bmat = 'I'; //standard eigenvalue problem
  #endif

  sprintf(which, "%s", "SM");

  #ifdef ARPACK_MAXIT
    *maxit = ARPACK_MAXIT;
  #else
    *maxit = 200;
  #endif

  for(i=0;i<11;i++) iparam[i] = 0;

  iparam[0] = 1;     //ishfts
  iparam[2] = *maxit; //maxitr
  iparam[3] = 1;     //blockSize

  #if USE_GENERALIZED_SYSTEM
    iparam[6] = 2;     //mode
  #else
    iparam[6] = 1;     //mode
  #endif

  *eigs_tolerance = 1e-3; // The tolerance of the arnoldi iterative solver

  *deflation_tol = 1e-2; //the deflation tolerance, all eigenvalues lower than this will be selected for deflation
  return 0;
}

/* Allocate workspace*/
int Presc_eigSolve_workspace_alloc(int m, int ncv, int ldv, int lworkl, double **resid, double **v, double **workd, double **workl){

  if ( !(*resid  = (double *) malloc(m * sizeof(double))) ) preAlps_abort("Malloc fails for resid[]."); //M
  if ( !(*v  = (double *) malloc(ldv * ncv * sizeof(double))) ) preAlps_abort("Malloc fails for v[].");

  if ( !(*workd  = (double *) malloc(3 * m * sizeof(double))) ) preAlps_abort("Malloc fails for workd[].");
  if ( !(*workl  = (double *) malloc(lworkl * sizeof(double))) ) preAlps_abort("Malloc fails for workl[].");

  return 0;
}

/* Free workspace*/
int Presc_eigSolve_workspace_free(double **resid, double **v, double **workd, double **workl){
  free(*resid);
  preAlps_int_printSynchronized(2, "eigsw free", MPI_COMM_WORLD);
  free(*v);
  preAlps_int_printSynchronized(3, "eigsw free", MPI_COMM_WORLD);
  free(*workd);
  preAlps_int_printSynchronized(4, "eigsw free", MPI_COMM_WORLD);
  ///free(*workl);
  return 0;
}

/*
 * Solve the eigenvalues problem (I + AggP*S_{loc}^{-1})u = \lambda u using arpack.
 * Where AggP and S_{loc} are two sparse matrices.
*  AggP is formed by the offDiag elements of Agg, and S_{loc} = Block-Diag(S);
 * Check the paper for the structure of AggP and S_{loc}.
 *
 * m:
 *    the number of rows of the global matrice. Also correspond to the size of the eigenvectors.
 * mcounts:
 *    input: the local of rows of each processor.
 * Sloc_sv
 *    input: the solver object to apply to compute  Sloc^{-1}v
*/
int Presc_eigSolve(Presc_t *presc, int *mcounts, preAlps_solver_t *Sloc_sv, Mat_CSR_t *Sloc, Mat_CSR_t *AggP, MPI_Comm comm){

  /**/
  char bmat;
  char which[2];
  int nev; //The number of eigenvalues to compute
  int maxit;
  int iparam[11];
  double eigs_tol;

  double deflation_tol;

  int ipntr[11] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  int info = 0;
  int ncv, lworkl;
  double *resid, *v, *workd, *workl, *Y, *X, *dwork, *ywork;
  int i, ldv, RCI_its = 0, ido = 0;
  int iterate = 1;
  int *mdispls;
  int m;

  /* */
  int my_rank, nbprocs, root = 0;
  double dONE = 1.0, dZERO = 0.0;
  double t = 0.0, tParpack=0.0, tSolve=0.0, tAggv=0.0, ttemp, tComm=0.0;

  /* */

  preAlps_int_printSynchronized(1, "Starting eigSolve", comm);

  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &nbprocs);

  t = MPI_Wtime();

  /* Get sizes */
  int mloc = mcounts[my_rank];
  m = 0;
  for(i=0;i<nbprocs;i++) m += mcounts[i];

  preAlps_int_printSynchronized(mloc, "mloc in PARPACK", comm);

  preAlps_int_printSynchronized(m, "m in PARPACK", comm);



  /*Initialize arpack with default parameters*/
  Presc_eigSolve_init(&bmat, which, &maxit, iparam, &eigs_tol, &deflation_tol);


  /* Set the number of eigenvalues to compute*/
  #ifdef NEV
   nev = NEV;
  #else
   //nev = (int) m*1e-2;
   nev = (int) m*2e-3;
   if(nev<=10) nev = 10;
  #endif


  /* Number of colums of the final set of Arnoldi basis vectors */
  ncv = 2 * nev + 1;

  //ncv = 10;
  //minimum required by ARPACK
  lworkl = ncv*(3*ncv + 6); //ncv * (ncv + 8);

  ldv = mloc; //ldv = m * ncv; //mloc

  // Allocate the memory
  /* Allocate workspace */
  Presc_eigSolve_workspace_alloc(m, ncv, ldv, lworkl, &resid, &v, &workd, &workl);

  if ( !(dwork  = (double *) malloc(mloc * sizeof(double))) ) preAlps_abort("Malloc fails for dwork[]."); //M
  if ( !(ywork  = (double *) malloc(m * sizeof(double))) ) preAlps_abort("Malloc fails for ywork[].");
  if ( !(mdispls  = (int *) malloc(nbprocs * sizeof(int))) ) preAlps_abort("Malloc fails for mdispls[].");

  /* Provide our own initialization for reproducibility purpose*/
  info = 1; //Provide our initialization vector
  for(i=0; i<m; i++){ //mloc
	  resid[i] = 1e-2;
  }

  //compute displacements
  mdispls[0] = 0;
  for(i=1;i<nbprocs;i++) mdispls[i] = mdispls[i-1] + mcounts[i-1];

  preAlps_intVector_printSynchronized(mcounts, nbprocs, "mcounts", "mcounts", comm);
  preAlps_intVector_printSynchronized(mdispls, nbprocs, "mdispls", "mdispls", comm);

  MatCSRPrintSynchronizedCoords (AggP, comm, "AggP", "AggP");

  if(my_rank==root) printf("Agg size: %d\n", m);

  if(mloc<=0) preAlps_abort("[PARPACK] mloc should be >0 for all procs ");


  preAlps_int_printSynchronized(comm, "Starting PARPACK comm", comm);




  iterate = 1;
  while(iterate){

  preAlps_int_printSynchronized(RCI_its, "Iteration", comm);
#ifdef USE_PARPACK

    RCI_its++;

    ttemp = MPI_Wtime();
    #if USE_SYM
     /* Call PARPACK */
     pdsaupd_(&comm, &ido, &bmat, &mloc, which,
            &nev, &eigs_tol, resid, &ncv,
            v, &ldv, iparam, ipntr,
            workd, workl, &lworkl,
            &info);
    #else
    pdnaupd_(&comm, &ido, &bmat, &mloc, which, &nev, &eigs_tol, resid, &ncv, v, &ldv,
          iparam, ipntr, workd, workl, &lworkl, &info );
    #endif
    tParpack += MPI_Wtime() - ttemp;

#else
    preAlps_abort("No other eigensolver is supported for the moment. Please Rebuild with PARPACK !");
#endif


    preAlps_int_printSynchronized(info, "info after pdnaupd", comm);
    preAlps_int_printSynchronized(ipntr[0] - 1, "ipntr[0] - 1", comm);
    preAlps_int_printSynchronized(ido, "ido", comm);

    X = &workd[ipntr[0] - 1];
    Y = &workd[ipntr[1] - 1];

    preAlps_doubleVector_printSynchronized(X, mloc, "X", "X after pdnaupd", comm);

    #ifdef DEBUG
      if(RCI_its==1) {
        MPI_Allgatherv(X, mloc, MPI_DOUBLE, ywork, mcounts, mdispls, MPI_DOUBLE, comm);

        if(my_rank==root) preAlps_doubleVector_printSynchronized(ywork, m, "X", "X gathered from all", MPI_COMM_SELF);

        DVector_t V0 = DVectorNULL();
        DVectorCreateFromPtr(&V0, m, ywork);
        if(my_rank==root) DVectorSave(&V0, "X_0.txt", "X_0 first step arpack");
      }
    #endif


    if(ido==-1||ido==1){
      ///if(RCI_its==1) {for(i=0;i<mloc;i++) X[i] = 1e-2; printf("dbgsimp1\n");}

      #if USE_GENERALIZED_SYSTEM

      /*
       * Compute Y = OP x X = S_{loc}^{-1}S*X = (I + S_{loc}^{-1}AggP) * X
      */

      //Gather the vector from each procs
      ttemp = MPI_Wtime();
      MPI_Allgatherv(X, mloc, MPI_DOUBLE, ywork, mcounts, mdispls, MPI_DOUBLE, comm);
      tComm+= MPI_Wtime() - ttemp;

      if(my_rank==root) preAlps_doubleVector_printSynchronized(ywork, m, "ywork", "ywork", MPI_COMM_SELF);

      ttemp = MPI_Wtime();
      MatCSRMatrixVector(AggP, dONE, ywork, dZERO, dwork);
      tAggv += MPI_Wtime() - ttemp;

      preAlps_doubleVector_printSynchronized(dwork, mloc, "dwork", "dwork after AggP*ywork", comm);

      /* Solve A x = b with the previous factorized matrix*/
      ttemp = MPI_Wtime();
      preAlps_solver_triangsolve(Sloc_sv, Sloc->info.m, Sloc->val, Sloc->rowPtr, Sloc->rowPtr, Y, dwork);
      tSolve += MPI_Wtime() - ttemp;

      preAlps_doubleVector_printSynchronized(Y, mloc, "Y", "Y after Sloc-1*v", comm);

      for(i=0;i<mloc;i++) Y[i] = X[i]+Y[i];

      /* Overwrite X with A*X (required mode=2)*/

      /* X = S*X = (Sloc+AoffDiag)*X_glob  =  Sloc*X_glob + AoffDiag*X_glob */
      /* X_i = Sloc_i*X_i + AoffDiag_i*ywork = Sloc_i*X_i + dwork;  */

      /* X = S*X_glob = (Sloc+AoffDiag)*X_glob = Sloc (I+Sloc^{-1}AoffDiag)*X_glob */
      /* X = Sloc*Y */
      //(Sloc+AoffDiag)*ywork =  Sloc*ywork+dwork

      ///for(i=0;i<mloc;i++) X[i] = dwork[i];

      ttemp = MPI_Wtime();
      //MatCSRMatrixVector(Sloc, dONE, ywork, dONE, X);
      //MatCSRMatrixVector(Sloc, dONE, X, dONE, X);
      MatCSRMatrixVector(Sloc, dONE, Y, dZERO, X);
      tAggv += MPI_Wtime() - ttemp;

      preAlps_doubleVector_printSynchronized(X, mloc, "X", "X = AX", comm);

      #else
      /*
       * Compute Y = OP x X = (I + AggP*S_{loc}^{-1}) x X
      */

      /* Compute  y = S_{loc}^{-1} x X => solve S_{loc} y = X */
      /* Solve A x = b with the previous factorized matrix*/
      ttemp = MPI_Wtime();
      preAlps_solver_triangsolve(Sloc_sv, Sloc->info.m, Sloc->val, Sloc->rowPtr, Sloc->rowPtr, dwork, X);
      tSolve += MPI_Wtime() - ttemp;

      preAlps_doubleVector_printSynchronized(dwork, mloc, "dwork", "dwork", comm);

      //Gather the vector from each
      ttemp = MPI_Wtime();
      //MPI_Allgather(dwork, mloc, MPI_INT, ywork, 1, MPI_INT, comm);
      MPI_Allgatherv(dwork, mloc, MPI_DOUBLE, ywork, mcounts, mdispls, MPI_DOUBLE, comm);
      tComm+= MPI_Wtime() - ttemp;

      //preAlps_doubleVector_printSynchronized(ywork, m, "ywork", "ywork", comm);

      if(my_rank==root) preAlps_doubleVector_printSynchronized(ywork, m, "ywork", "ywork", MPI_COMM_SELF);

      #ifdef DEBUG
        if(RCI_its==1) {
          MPI_Barrier(comm);
          DVector_t V1 = DVectorNULL();
          DVectorCreateFromPtr(&V1, m, ywork);

          printf("[%d] V1[0]:%e\n", my_rank, V1.val[0]);
          MPI_Barrier(comm);
          if(my_rank==root) DVectorSave(&V1, "Y_1.txt", "Y after Sloc^{-1}*x");
        }
      #endif

        /* Y = (I + AggP x S_{loc}^{-1})X = I*X +  AggP * S_{loc}^{-1}*X = I.X + AggP*ywork */
        //if(RCI_its==1) {for(i=0;i<m;i++) ywork[i] = 1e-2; printf("dbgsimp2\n");}
        ttemp = MPI_Wtime();
        MatCSRMatrixVector(AggP, dONE, ywork, dZERO, Y);
        tAggv += MPI_Wtime() - ttemp;

        preAlps_doubleVector_printSynchronized(Y, mloc, "Y", "Y after AggP*ywork", comm);

        #ifdef DEBUG
          if(RCI_its==1) {
            MPI_Allgatherv(Y, mloc, MPI_DOUBLE, ywork, mcounts, mdispls, MPI_DOUBLE, comm);

            if(my_rank==root) preAlps_doubleVector_printSynchronized(ywork, m, "ywork", "Y gathered from all", MPI_COMM_SELF);

            DVector_t V2 = DVectorNULL();
            DVectorCreateFromPtr(&V2, m,ywork);
            if(my_rank==root) DVectorSave(&V2, "Y_2.txt", "Y after AggP*ywork");
          }
        #endif

        for(i=0;i<mloc;i++) Y[i] = X[i]+Y[i];

        #ifdef DEBUG
          if(RCI_its==1) {
            MPI_Allgatherv(Y, mloc, MPI_DOUBLE, ywork, mcounts, mdispls, MPI_DOUBLE, comm);

            DVector_t V3 = DVectorNULL();
            DVectorCreateFromPtr(&V3, m,ywork);
            if(my_rank==root) DVectorSave(&V3, "Y_3.txt", "Y after AggP*ywork");
          }
        #endif

      #endif


      preAlps_doubleVector_printSynchronized(Y, mloc, "Y", "Y after matvec", comm);

    }else if(ido==2){

      /* Compute  Y = Sloc * X */
      ttemp = MPI_Wtime();
      MatCSRMatrixVector(Sloc, dONE, X, dZERO, Y);
      tAggv += MPI_Wtime() - ttemp;

      preAlps_doubleVector_printSynchronized(Y, mloc, "Y", "Y after Sloc*v", comm);
    }else{
       iterate = 0;
    } //ido

    //if(RCI_its>=5) iterate = 0; //DEBUG: force break for debugging purpose
  }

  preAlps_int_printSynchronized(ido, "End of ARPACK ido", comm);

  if(my_rank==root) printf("ido:%d\n", ido);

  if(ido!=99) preAlps_abort("[PARPACK] (Unhandled case) ido is not 99");

  /* After PARPACK */
  if(my_rank==root){
    if (info == 1) {
        printf("[PARPACK] Maximum number of iterations reached, found %d eigenvalues / %d requested\n",
                     iparam[4], nev);
    } else if (info != 0) {
        printf("[PARPACK] pdxaupd returned error info: %d\n", info);
        preAlps_abort("An error occured in PARPACK ");
    }
    printf("[PARPACK] Found %d eigenvalues after %d Arnoldi iterations, number OP * X: %d, RCI iterations: %d\n", iparam[4], iparam[2], iparam[8], RCI_its);

    for (i = 0; i < iparam[4]; i++) {
          /*
          printf("eigenvalues:\n");
          printf("\t%.16e", eigenvalues[i]);
          if (eigenvalues[i] > presc_eigs_tolerance) {
            printf(" (ignored)");
          }
          printf("\n");
          */
    }
  }

  /*
   * Compute eigenvectors
   */



  presc->nev =  iparam[4];
  if(presc->nev>0){

    /* Allocate workspace for the eigenvalues */

    if ( !(presc->eigvalues  = (double *) malloc(presc->nev * sizeof(double))) ) preAlps_abort("Malloc fails for presc->eigenvalues[].");

    int rvec = 1; // Compute the eigenvectors
    char howMany = 'A'; // Compute all the eigenvalues
    int *vselect; //Specify the eigenvectors to be computed
    double *z;
    int ldz;

    ldz = mloc; //counts[mynode_rank];

    #if USE_SYM

      double sigma=0.0;

      if ( !(vselect  = (int *) malloc(ncv * sizeof(int))) ) preAlps_abort("Malloc fails for vselect[].");
      if ( !(z  = (double *) malloc(ldz * presc->nev * sizeof(double))) ) preAlps_abort("Malloc fails for z[].");
      pdseupd_(&comm, &rvec, &howMany, vselect,
               presc->eigvalues, z, &ldz,
               &sigma, &bmat, &mloc, which,
               &nev, &eigs_tol, resid, &ncv,
               v, &ldv, iparam, ipntr,
               workd, workl, &lworkl, &info);

    #else

      double sigmaR = 0.0, sigmaI=0.0;

      double *workev, *DR, *DI;

      if ( !(vselect  = (int *) malloc(ncv * sizeof(int))) ) preAlps_abort("Malloc fails for vselect[].");
      if ( !(z  = (double *) malloc(ldz * (presc->nev+1) * sizeof(double))) ) preAlps_abort("Malloc fails for z[].");

      if ( !(DI  = (double *) malloc((presc->nev+1) * sizeof(double))) ) preAlps_abort("Malloc fails for D[].");

      if ( !(workev  = (double *) malloc((3*ncv) * sizeof(double))) ) preAlps_abort("Malloc fails for D[].");

      pdneupd_(&comm, &rvec, &howMany, vselect, presc->eigvalues, DI, z, &ldz, &sigmaR, &sigmaI, workev, &bmat, &mloc, which,
              &nev, &eigs_tol, resid, &ncv,
              v, &ldv, iparam, ipntr,
              workd, workl, &lworkl,
              &info);


      free(DI);
      free(workev);
    #endif

    if(my_rank==root){
      if (info != 0) {
          printf("[PARPACK] pdxeupd returned error info: %d\n", info);
          //preAlps_abort("An error occured in PARPACK ");
      }
    }
    free(vselect);
    free(z);


    /* Select the eigenvalues to deflate */
    if(my_rank == root){
      for (i = 0; i < presc->nev; i++) {
        if (presc->eigvalues[i] <= deflation_tol) {
          presc->eigvalues_deflation++;
        }
      }
    }


    #ifdef EIGVALUES_PRINT
      if(my_rank == root){
        printf("[%d] eigenvalues:\n", my_rank);
        for (i = 0; i < presc->nev; i++) {

          printf("\t%.16e", presc->eigvalues[i]);

          if (presc->eigvalues[i] <= deflation_tol) {
            printf(" (selected)\n");
          }else{
            printf(" (ignored)\n");
          }
        }
      }
    #endif


  }

  if(my_rank==root) printf("Eigenvalues selected for deflation: %d/%d\n", presc->eigvalues_deflation, presc->nev);


  t = MPI_Wtime() - t;

#ifdef EIG_DISPLAY_STATS
  preAlps_dstats_display(comm, tParpack, "Parpack time");
  preAlps_dstats_display(comm, tSolve, "Solve time");
  preAlps_dstats_display(comm, tAggv, "Agg*v time");
  preAlps_dstats_display(comm, tComm, "Comm time");
  preAlps_dstats_display(comm, t, "EigSolve Time");
#endif


  // Free the memory
  Presc_eigSolve_workspace_free(&resid, &v, &workd, &workl);

  free(dwork);
  free(ywork);
  free(mdispls);
  return 0;
}









/*
 * Version with ALOC
 */





/*
 * Solve the eigenvalues problem Su = \lambda Aloc u using arpack.
 * Where S and Aloc are two sparse matrices.
 *
 * S = Agg - Agi*Aii^{-1}*Agi
 * mloc:
 *    the local problem size.
*/
int Presc_eigSolve_SAloc(Presc_t *presc, int mloc, Mat_CSR_t *Aggloc, Mat_CSR_t *Agi, Mat_CSR_t *Aii, Mat_CSR_t *Aig, Mat_CSR_t *Aloc
                                       , preAlps_solver_t *Aii_sv, preAlps_solver_t *Aloc_sv, MPI_Comm comm){

  /**/
  char bmat;
  char which[2];
  int nev; //The number of eigenvalues to compute
  int maxit;
  int iparam[11];
  double eigs_tol;

  double deflation_tol;

  int ipntr[11] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  int info = 0;
  int ncv, lworkl;
  double *resid, *v, *workd, *workl, *Y, *X, *ywork;
  double *dwork1, *dwork2; //, *dwork
  int i, ldv, RCI_its = 0, ido = 0;
  int iterate = 1;
  int *mcounts, *mdispls;
  int m;

  /* */
  int my_rank, nbprocs, root = 0;
  double dONE = 1.0, dZERO = 0.0, dMONE =-1.0;
  double t = 0.0, tParpack=0.0, tSolve=0.0, tAggv=0.0, ttemp, tComm=0.0;
  double tSv = 0.0, tInvAggv = 0.0;
  /* */

  t = MPI_Wtime();

  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &nbprocs);

  preAlps_int_printSynchronized(1, "Starting eigSolve SAloc", comm);


  if ( !(mcounts  = (int *) malloc(nbprocs * sizeof(int))) ) preAlps_abort("Malloc fails for mcounts[].");
  if ( !(mdispls  = (int *) malloc(nbprocs * sizeof(int))) ) preAlps_abort("Malloc fails for mdispls[].");

  preAlps_int_printSynchronized(mloc, "mloc in PARPACK", comm);

  //Gather the number of elements in the diag for each procs
  MPI_Allgather(&mloc, 1, MPI_INT, mcounts, 1, MPI_INT, comm);

  //compute displacements
  mdispls[0] = 0;
  for(i=1;i<nbprocs;i++) mdispls[i] = mdispls[i-1] + mcounts[i-1];

  preAlps_intVector_printSynchronized(mcounts, nbprocs, "mcounts", "mcounts", comm);
  preAlps_intVector_printSynchronized(mdispls, nbprocs, "mdispls", "mdispls", comm);


  /* Compute the global size */
  m = 0;
  for(i=0;i<nbprocs;i++) m += mcounts[i];


  preAlps_int_printSynchronized(m, "m in PARPACK", comm);

  //if(my_rank==0)
  #ifdef DEBUG
    printf("mloc:%d, m:%d, Aii: (%d x %d), Aig: (%d x %d)\n", mloc, m, Aii->info.m, Aii->info.n, Aig->info.m, Aig->info.n);
    MPI_Barrier(comm); //debug only
  #endif
  /*Initialize arpack with default parameters*/
  Presc_eigSolve_init(&bmat, which, &maxit, iparam, &eigs_tol, &deflation_tol);

  bmat = 'G'; //Generalized eigenvalue problem
  iparam[6] = 2;  //Arpack mode

  /* Set the number of eigenvalues to compute*/
  #ifdef NEV
   nev = NEV;
  #else
   //nev = (int) m*1e-2;
   nev = (int) m*2e-3;
   if(nev<=10) nev = 10;
  #endif


  /* Number of colums of the final set of Arnoldi basis vectors */
  ncv = 2 * nev + 1;

  //ncv = 10;
  //minimum required by ARPACK
  lworkl = ncv*(3*ncv + 6); //ncv * (ncv + 8);

  ldv = mloc; //ldv = m * ncv; //mloc

  // Allocate the memory
  /* Allocate workspace */
  Presc_eigSolve_workspace_alloc(m, ncv, ldv, lworkl, &resid, &v, &workd, &workl);

  //if ( !(dwork  = (double *) malloc(m * sizeof(double))) ) preAlps_abort("Malloc fails for dwork[]."); //max(Aig.n, mloc) or m

  if ( !(dwork1  = (double *) malloc(Aii->info.m * sizeof(double))) ) preAlps_abort("Malloc fails for dwork1[].");
  if ( !(dwork2  = (double *) malloc(Aii->info.m * sizeof(double))) ) preAlps_abort("Malloc fails for dwork2[].");

  if ( !(ywork  = (double *) malloc(m * sizeof(double))) ) preAlps_abort("Malloc fails for ywork[].");


  /* Provide our own initialization for reproducibility purpose */
  info = 1; //Provide our initialization vector
  for(i=0; i<m; i++){ //mloc
	  resid[i] = 1e-2;
  }





  MatCSRPrintSynchronizedCoords (Aggloc, comm, "Aggloc", "Aggloc");

  if(my_rank==root) printf("Agg size: %d\n", m);

  if(mloc<=0) preAlps_abort("[PARPACK] mloc should be > 0 for all procs ");


  preAlps_int_printSynchronized(comm, "Starting PARPACK comm", comm);




  iterate = 1;
  while(iterate){

  preAlps_int_printSynchronized(RCI_its, "Iteration", comm);
#ifdef USE_PARPACK

    RCI_its++;

    ttemp = MPI_Wtime();
    #if USE_SYM
     /* Call PARPACK */
     pdsaupd_(&comm, &ido, &bmat, &mloc, which,
            &nev, &eigs_tol, resid, &ncv,
            v, &ldv, iparam, ipntr,
            workd, workl, &lworkl,
            &info);
    #else
    pdnaupd_(&comm, &ido, &bmat, &mloc, which, &nev, &eigs_tol, resid, &ncv, v, &ldv,
          iparam, ipntr, workd, workl, &lworkl, &info );
    #endif
    tParpack += MPI_Wtime() - ttemp;

#else
    preAlps_abort("No other eigensolver is supported for the moment. Please Rebuild with PARPACK !");
#endif


    preAlps_int_printSynchronized(info, "info after pdnaupd", comm);
    preAlps_int_printSynchronized(ipntr[0] - 1, "ipntr[0] - 1", comm);
    preAlps_int_printSynchronized(ido, "ido", comm);

    X = &workd[ipntr[0] - 1];
    Y = &workd[ipntr[1] - 1];

    preAlps_doubleVector_printSynchronized(X, mloc, "X", "X after pdnaupd", comm);

    #ifdef DEBUG
      if(RCI_its==1) {

        MPI_Allgatherv(X, mloc, MPI_DOUBLE, ywork, mcounts, mdispls, MPI_DOUBLE, comm);

        if(my_rank==root) preAlps_doubleVector_printSynchronized(ywork, m, "X", "X gathered from all", MPI_COMM_SELF);

        DVector_t V0 = DVectorNULL();
        DVectorCreateFromPtr(&V0, m, ywork);
        if(my_rank==root) DVectorSave(&V0, "X_0.txt", "X_0 first step arpack");
      }
    #endif


    if(ido==-1||ido==1){
      ///if(RCI_its==1) {for(i=0;i<mloc;i++) X[i] = 1e-2; printf("dbgsimp1\n");}

      /*
       * Compute Y = OP x X = A_{loc}^{-1}S*X
       * Si = Agg - Agi*Aii^{-1}*Aig
      */
      //Gather the vector from each procs
      ttemp = MPI_Wtime();
      MPI_Allgatherv(X, mloc, MPI_DOUBLE, ywork, mcounts, mdispls, MPI_DOUBLE, comm);
      tComm+= MPI_Wtime() - ttemp;

      if(my_rank==root) preAlps_doubleVector_printSynchronized(ywork, m, "ywork", "ywork", MPI_COMM_SELF);


      ttemp = MPI_Wtime();

      MatCSRMatrixVector(Aig, dONE, X, dZERO, dwork1);
      //tAggv += MPI_Wtime() - ttemp;

      preAlps_doubleVector_printSynchronized(dwork1, Aig->info.m, "dwork1", "dwork1 = Aig*X", comm);

      //ttemp = MPI_Wtime();
      preAlps_solver_triangsolve(Aii_sv, Aii->info.m, Aii->val, Aii->rowPtr, Aii->rowPtr, dwork2, dwork1);
      //tSolve += MPI_Wtime() - ttemp;

      preAlps_doubleVector_printSynchronized(dwork2, Aii->info.m, "dwork", "dwork2 = Aii^{-1}*dwork1", comm);

      //ttemp = MPI_Wtime();
      MatCSRMatrixVector(Agi, dONE, dwork2, dZERO, X);
      //tAggv += MPI_Wtime() - ttemp;


      preAlps_doubleVector_printSynchronized(X, Agi->info.m, "X", "X = Agi*dwork2", comm);

      /* Copy dwork = Su in X */
      //for(i=0;i<mloc;i++) X[i] = dwork[i];

      //preAlps_doubleVector_printSynchronized(X, mloc, "X", "X = SX", comm);


      //Copy of Su must be in X

      //ttemp = MPI_Wtime();
      MatCSRMatrixVector(Aggloc, dONE, ywork, dMONE, X);
      //tAggv += MPI_Wtime() - ttemp;

      tSv += MPI_Wtime() - ttemp;

      //preAlps_doubleVector_printSynchronized(dwork, mloc, "dwork", "dwork after Agg*ywork - dwork", comm);

      /* Copy dwork = Su in X */
      //for(i=0;i<mloc;i++) X[i] = dwork[i];

      preAlps_doubleVector_printSynchronized(X, mloc, "X", "X = SX", comm);

      #ifdef DEBUG
        if(RCI_its==1) {
          //MPI_Allgatherv(X, mloc, MPI_DOUBLE, ywork, mcounts, mdispls, MPI_DOUBLE, comm);

          DVector_t V2 = DVectorNULL();
          DVectorCreateFromPtr(&V2, mloc, X);
          if(my_rank==root) DVectorSave(&V2, "dump/SX_0.txt", "SX_0 first step arpack");
        }
      #endif



      /* Solve A x = b with the previous factorized matrix*/
      ttemp = MPI_Wtime();
      preAlps_solver_triangsolve(Aloc_sv, Aloc->info.m, Aloc->val, Aloc->rowPtr, Aloc->rowPtr, Y, X);
      tInvAggv += MPI_Wtime() - ttemp;

      preAlps_doubleVector_printSynchronized(Y, mloc, "Y", "Y after matvec", comm);

    }else if(ido==2){

      /* Compute  Y = Aloc * X */
      ttemp = MPI_Wtime();
      MatCSRMatrixVector(Aloc, dONE, X, dZERO, Y);
      tAggv += MPI_Wtime() - ttemp;

      preAlps_doubleVector_printSynchronized(Y, mloc, "Y", "Y after Aloc*v", comm);
    }else{
       iterate = 0;
    } //ido

    //if(RCI_its>=1) {printf("Break debug at iteration:%d\n", RCI_its); ido=99;iterate = 0;} //DEBUG: force break for debugging purpose
  }

  preAlps_int_printSynchronized(ido, "End of ARPACK ido", comm);

  if(my_rank==root) printf("ido:%d\n", ido);

  if(ido!=99) preAlps_abort("[PARPACK] (Unhandled case) ido is not 99");

  /* After PARPACK */
  if(my_rank==root){
    if (info == 1) {
        printf("[PARPACK] Maximum number of iterations reached, found %d eigenvalues / %d requested\n",
                     iparam[4], nev);
    } else if (info != 0) {
        printf("[PARPACK] pdxaupd returned error info: %d\n", info);
        preAlps_abort("An error occured in PARPACK ");
    }
    printf("[PARPACK] Found %d eigenvalues after %d Arnoldi iterations, number OP * X: %d, RCI iterations: %d\n", iparam[4], iparam[2], iparam[8], RCI_its);

    for (i = 0; i < iparam[4]; i++) {
          /*
          printf("eigenvalues:\n");
          printf("\t%.16e", eigenvalues[i]);
          if (eigenvalues[i] > presc_eigs_tolerance) {
            printf(" (ignored)");
          }
          printf("\n");
          */
    }
  }

  /*
   * Compute eigenvectors
   */



  presc->nev =  iparam[4];
  if(presc->nev>0){

    /* Allocate workspace for the eigenvalues */

    if ( !(presc->eigvalues  = (double *) malloc(presc->nev * sizeof(double))) ) preAlps_abort("Malloc fails for presc->eigenvalues[].");

    int rvec = 1; // Compute the eigenvectors
    char howMany = 'A'; // Compute all the eigenvalues
    int *vselect; //Specify the eigenvectors to be computed
    double *z;
    int ldz;

    ldz = mloc; //counts[mynode_rank];

    #if USE_SYM

      double sigma=0.0;

      if ( !(vselect  = (int *) malloc(ncv * sizeof(int))) ) preAlps_abort("Malloc fails for vselect[].");
      if ( !(z  = (double *) malloc(ldz * presc->nev * sizeof(double))) ) preAlps_abort("Malloc fails for z[].");
      pdseupd_(&comm, &rvec, &howMany, vselect,
               presc->eigvalues, z, &ldz,
               &sigma, &bmat, &mloc, which,
               &nev, &eigs_tol, resid, &ncv,
               v, &ldv, iparam, ipntr,
               workd, workl, &lworkl, &info);

    #else

      double sigmaR = 0.0, sigmaI=0.0;

      double *workev, *DR, *DI;

      if ( !(vselect  = (int *) malloc(ncv * sizeof(int))) ) preAlps_abort("Malloc fails for vselect[].");
      if ( !(z  = (double *) malloc(ldz * (presc->nev+1) * sizeof(double))) ) preAlps_abort("Malloc fails for z[].");

      if ( !(DI  = (double *) malloc((presc->nev+1) * sizeof(double))) ) preAlps_abort("Malloc fails for D[].");

      if ( !(workev  = (double *) malloc((3*ncv) * sizeof(double))) ) preAlps_abort("Malloc fails for D[].");

      pdneupd_(&comm, &rvec, &howMany, vselect, presc->eigvalues, DI, z, &ldz, &sigmaR, &sigmaI, workev, &bmat, &mloc, which,
              &nev, &eigs_tol, resid, &ncv,
              v, &ldv, iparam, ipntr,
              workd, workl, &lworkl,
              &info);


      free(DI);
      free(workev);
    #endif

    if(my_rank==root){
      if (info != 0) {
          printf("[PARPACK] pdxeupd returned error info: %d\n", info);
          //preAlps_abort("An error occured in PARPACK ");
      }
    }
    free(vselect);
    free(z);


    /* Select the eigenvalues to deflate */
    if(my_rank == root){
      for (i = 0; i < presc->nev; i++) {
        if (presc->eigvalues[i] <= deflation_tol) {
          presc->eigvalues_deflation++;
        }
      }
    }


    #ifdef EIGVALUES_PRINT
      if(my_rank == root){
        printf("[%d] eigenvalues:\n", my_rank);
        for (i = 0; i < presc->nev; i++) {

          printf("\t%.16e", presc->eigvalues[i]);

          if (presc->eigvalues[i] <= deflation_tol) {
            printf(" (selected)\n");
          }else{
            printf(" (ignored)\n");
          }
        }
      }
    #endif


  }

  if(my_rank==root) printf("Eigenvalues selected for deflation: %d/%d\n", presc->eigvalues_deflation, presc->nev);


  t = MPI_Wtime() - t;

#ifdef EIG_DISPLAY_STATS
  preAlps_dstats_display(comm, tParpack, "Parpack time");
  preAlps_dstats_display(comm, tSv, "S*v");
  preAlps_dstats_display(comm, tInvAggv, "Inv(Agg)*v time");
  preAlps_dstats_display(comm, tAggv, "Agg*v time");
  preAlps_dstats_display(comm, tComm, "Comm time");
  preAlps_dstats_display(comm, t, "EigSolve Time");
#endif


  // Free the memory
  preAlps_int_printSynchronized(1, "eigs free", comm);
  Presc_eigSolve_workspace_free(&resid, &v, &workd, &workl);

  preAlps_int_printSynchronized(2, "eigs free", comm);
  free(dwork1);
  free(dwork2);
  preAlps_int_printSynchronized(3, "eigs free", comm);
  free(ywork);
  preAlps_int_printSynchronized(4, "eigs free", comm);
  free(mcounts);
  free(mdispls);
  preAlps_int_printSynchronized(5, "eigs free", comm);
  return 0;
}
