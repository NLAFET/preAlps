/*
============================================================================
Name        : eigsolve.c
Author      : Simplice Donfack
Version     : 0.1
Description : eigenvalues interface using ARPACK
Date        : Sept 13, 2017
============================================================================
*/
#include <stdlib.h>
#include <stdio.h>
#include "eigsolver.h"
#include "preAlps_utils.h"

#ifdef USE_PARPACK
  extern void pdsaupd_(MPI_Comm*, int*, char*, int*, char*, int*, double*, double*, int*, double*, int*,
                    int*, int*, double*, double*, int*, int*);

  extern void pdnaupd_(MPI_Comm*, int*, char*, int*, char*, int*, double*, double*, int*, double*, int*,
                              int*, int*, double*, double*, int*, int*);

  extern void pdseupd_(MPI_Comm*, int*, char*, int*, double*, double*, int*, double*, char*, int*,
                    char*, int*, double*, double*, int*, double*, int*, int*, int*,
                    double*, double*, int*, int*);

  extern void pdneupd_(MPI_Comm* comm, int *rvec, char *howmny, int *select, double *dr, double *di, double *z, int *ldz, double *sigmaR, double *sigmaI,
                    double *workev, char *bmat, int *n, char *which, int *nev, double *tol, double *resid, int *ncv, double *v, int *ldv,
                    int *iparam, int *ipntr, double *workd, double *workl, int *lworkl, int *info );
#endif

/* Create an eigensolver object */
int Eigsolver_create(Eigsolver_t **eigs){

  int ierr = 0;
  if ( !(*eigs  = (Eigsolver_t *) malloc(sizeof(Eigsolver_t))) ) preAlps_abort("Malloc fails for eigsolver object"); //M
  return ierr;
}

/* Gather the local computed eigenvectors on the root process */
int Eigsolver_eigenvectorsGather(Eigsolver_t *eigs, MPI_Comm comm, int *mcounts, int *mdispls, double **eigenvectors){
  // Gather the eigenvectors on the root
  int ierr = 0, j, my_rank, nbprocs, root = 0;

  int TAG_WRITE = 4;
  MPI_Status status;

  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &nbprocs);

  if(my_rank==root){

    //Allocate memory for the global eigenvectors
    if ( !(*eigenvectors  = (double *) malloc(eigs->m * eigs->nevComputed * sizeof(double))) ) preAlps_abort("Malloc fails for *eigenvectors[].");

    int mrows = mcounts[my_rank];
    int ldz = mcounts[my_rank];

    dlacpy("A", &mrows, &eigs->nevComputed, eigs->eigvectorsloc, &ldz, *eigenvectors, &eigs->m);

    double *ztmp;
    int ztmp_size = mcounts[my_rank]*eigs->nevComputed;
    if ( !(ztmp  = (double *) malloc(ztmp_size * sizeof(double))) ) preAlps_abort("Malloc fails for ztmp[].");

    for (j = 1; j < nbprocs; j++) {

      if(ztmp_size<mcounts[j]*eigs->nevComputed){
        free(ztmp);
        ztmp_size = mcounts[j]*eigs->nevComputed;
        if ( !(ztmp  = (double *) malloc(ztmp_size * sizeof(double))) ) preAlps_abort("Malloc fails for ztmp[].");
      }

      MPI_Recv(ztmp, mcounts[j]*eigs->nevComputed, MPI_DOUBLE, j, TAG_WRITE, comm, &status);

      mrows = mcounts[j];
      ldz = mcounts[j];
      //double *e = *eigenvectors;
      dlacpy("A", &mrows, &eigs->nevComputed, ztmp, &ldz, &(*eigenvectors)[mdispls[j]], &eigs->m);
    }

    free(ztmp);
  }else{
    MPI_Send(eigs->eigvectorsloc, mcounts[my_rank]*eigs->nevComputed, MPI_DOUBLE, root, TAG_WRITE, comm);
  }

  return ierr;
}

/*Initialize the solver and allocate workspace*/
int Eigsolver_init(Eigsolver_t *eigs, int m, int mloc){

  int i, ierr = 0;

  /* Quick check*/
  if(mloc<=0) preAlps_abort("[PARPACK] mloc should be >0 for all procs ");
  eigs->m = m;

  eigs->iparam[0] = 1;     //ishfts
  eigs->iparam[2] = eigs->maxit; //maxitr
  eigs->iparam[3] = 1;     //blockSize


  if(eigs->bmat == 'G'){
    eigs->iparam[6] = 2;     //mode for the generalized problem
  }else{
    eigs->iparam[6] = 1;     //mode
  }


  /* Number of colums of the final set of Arnoldi basis vectors */
  eigs->ncv = 2 * eigs->nev + 1;

  //ncv = 10;
  //minimum required by ARPACK
  eigs->lworkl = eigs->ncv*(3*eigs->ncv + 6); //ncv * (ncv + 8);

  eigs->ldv = mloc; //ldv = m * ncv; //mloc

  if ( !(eigs->resid  = (double *) malloc(m * sizeof(double))) ) preAlps_abort("Malloc fails for resid[]."); //M
  if ( !(eigs->v  = (double *) malloc(eigs->ldv * eigs->ncv * sizeof(double))) ) preAlps_abort("Malloc fails for v[].");

  if ( !(eigs->workd  = (double *) malloc(3 * m * sizeof(double))) ) preAlps_abort("Malloc fails for workd[].");
  if ( !(eigs->workl  = (double *) malloc(eigs->lworkl * sizeof(double))) ) preAlps_abort("Malloc fails for workl[].");

  /* Provide our own initialization for reproducibility purpose*/
  eigs->info = 1; //Provide our initialization vector
  for(i=0; i<m; i++){ //mloc
	  eigs->resid[i] = 1e-2;
  }

  eigs->RCI_iter = 0;
  eigs->OPX_iter = 0;
  eigs->BX_iter  = 0;

  return ierr;
}

/* Terminate the solver and free the allocated workspace*/
int Eigsolver_finalize(Eigsolver_t **eigs){

  int ierr = 0;
  free((*eigs)->resid);
  free((*eigs)->v);
  free((*eigs)->workd);
  if((*eigs)->eigvalues!=NULL) free((*eigs)->eigvalues);
  if((*eigs)->eigvectorsloc!=NULL) free((*eigs)->eigvectorsloc);
  free(*eigs);

  return ierr;
}

/* Set the default parameters for the solver*/
int Eigsolver_setDefaultParameters(Eigsolver_t *eigs){
  int i, ierr = 0;

  eigs->bmat = 'G'; /* Generalized problem */

  sprintf(eigs->which, "%s", "SM"); /*Small eigenvalues*/

  /* Maximum number of iterations*/
  #ifdef ARPACK_MAXIT
    eigs->maxit = ARPACK_MAXIT;
  #else
    eigs->maxit = 200;
  #endif

  eigs->residual_tolerance  = 1e-8; // The tolerance of the arnoldi iterative solver


  eigs->nev         = 0;
  eigs->nevComputed = 0;
  eigs->eigvalues   = NULL;
  eigs->eigvectorsloc   = NULL;
  eigs->eigvectorsloc_size   = 0;
  eigs->issym       = 0;
  for(i=0;i<11;i++) eigs->iparam[i] = 0;
  for(i=0;i<14;i++) eigs->ipntr[i] = 0;

  eigs->info = 0;

  eigs->tEigVectors = 0.0;
  eigs->tEigValues  = 0.0;


  return ierr;
}


/* Perform one iteration of the eigensolver and return the hand to the RCI */
int Eigsolver_iterate(Eigsolver_t *eigs, MPI_Comm comm, int mloc, double **X, double **Y, int *ido){

  int ierr=0;

#ifdef USE_PARPACK
  int root = 0, my_rank, nbprocs;
  double ttemp;

  /* Retrieve parameters */

  char bmat        = eigs->bmat;
  char *which      = eigs->which;
  int nev          = eigs->nev;
  double residual_tol = eigs->residual_tolerance;
  double *resid    = eigs->resid;
  int ncv          = eigs->ncv;
  double *v        = eigs->v;
  int ldv          = eigs->ldv;
  int *iparam      = eigs->iparam;
  int *ipntr       = eigs->ipntr;
  double *workd    = eigs->workd;
  double *workl    = eigs->workl;
  int lworkl       = eigs->lworkl;


  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &nbprocs);

  #ifdef USE_PARPACK

    ttemp = MPI_Wtime();

    /* Call PARPACK */

    if(eigs->issym){

      pdsaupd_(&comm, ido, &bmat, &mloc, which, &nev, &residual_tol, resid, &ncv,
              v, &ldv, iparam, ipntr, workd, workl, &lworkl, &eigs->info);
    } else{
      pdnaupd_(&comm, ido, &bmat, &mloc, which, &nev, &residual_tol, resid, &ncv,
              v, &ldv, iparam, ipntr, workd, workl, &lworkl, &eigs->info );
    }

    eigs->tEigValues += MPI_Wtime() - ttemp;

  #else
    preAlps_abort("No other eigensolver is supported for the moment. Please Rebuild with PARPACK !");
  #endif

  eigs->RCI_iter++;
  if(*ido==1) eigs->OPX_iter++;
  if(*ido==2) eigs->BX_iter++;

  /* Set X and Y that will be used to compute the matrix vector product */

  *X = &workd[ipntr[0] - 1];
  *Y = &workd[ipntr[1] - 1];

  /* After PARPACK */
  if(*ido==99 && my_rank==root){

    if (eigs->info == 1) {
        printf("[PARPACK] Maximum number of iterations reached, found %d eigenvalues / %d requested\n",
                     iparam[4], nev);
    } else if (eigs->info != 0) {
        printf("[PARPACK] pdxaupd returned error info: %d\n", eigs->info);
        preAlps_abort("An error occured in PARPACK ");
    }
    printf("[PARPACK] Found %d eigenvalues after %d Arnoldi iterations, number OP * X: %d, RCI iterations: %d, OP*X: %d, B*X: %d\n",
           iparam[4], iparam[2], iparam[8], eigs->RCI_iter, eigs->OPX_iter, eigs->BX_iter);
  }

  /* Compute the eigenvectors when ido = 99 */
  if(*ido==99 && iparam[4]>0){

    ttemp = MPI_Wtime();

    #ifdef USE_PARPACK
      eigs->nevComputed =  iparam[4];

      /* Allocate workspace for the eigenvalues */

      if ( !(eigs->eigvalues  = (double *) malloc((nev +1) * sizeof(double))) ) preAlps_abort("Malloc fails for eigs->eigenvalues[].");

      int rvec = 1; // Compute the eigenvectors
      char howMany = 'A'; // Compute all the eigenvalues
      int *vselect; //Specify the eigenvectors to be computed
      //double *z;
      //int ldz;

      //ldz = mloc; //counts[mynode_rank];

      if(eigs->issym){

        double sigma=0.0;
        eigs->eigvectorsloc_size = mloc*nev;
        if ( !(vselect  = (int *) malloc(ncv * sizeof(int))) ) preAlps_abort("Malloc fails for vselect[].");
        if ( !(eigs->eigvectorsloc  = (double *) malloc(eigs->eigvectorsloc_size * sizeof(double))) ) preAlps_abort("Malloc fails for eigvectorsloc[].");
        pdseupd_(&comm, &rvec, &howMany, vselect, eigs->eigvalues, eigs->eigvectorsloc, &mloc,
               &sigma, &bmat, &mloc, which, &nev, &residual_tol, resid,
               &ncv, v, &ldv, iparam, ipntr, workd, workl, &lworkl, &eigs->info);

      } else {

        double sigmaR = 0.0, sigmaI=0.0;
        double *workev, *DI;
        eigs->eigvectorsloc_size = mloc * (nev+1);

        if ( !(vselect  = (int *) malloc(ncv * sizeof(int))) ) preAlps_abort("Malloc fails for vselect[].");
        if ( !(eigs->eigvectorsloc  = (double *) malloc(eigs->eigvectorsloc_size * sizeof(double))) ) preAlps_abort("Malloc fails for eigvectorsloc[].");

        if ( !(DI  = (double *) malloc((nev+1) * sizeof(double))) ) preAlps_abort("Malloc fails for D[].");

        if ( !(workev  = (double *) malloc((3*ncv) * sizeof(double))) ) preAlps_abort("Malloc fails for D[].");

        pdneupd_(&comm, &rvec, &howMany, vselect, eigs->eigvalues, DI, eigs->eigvectorsloc, &mloc,
               &sigmaR, &sigmaI, workev, &bmat, &mloc, which, &nev, &residual_tol, resid,
               &ncv, v, &ldv, iparam, ipntr, workd, workl, &lworkl, &eigs->info);


        free(DI);
        free(workev);
      }

      if(my_rank==root){
        if (eigs->info != 0) {
          printf("[PARPACK] pdxeupd returned error info: %d\n", eigs->info);
          //preAlps_abort("An error occured in PARPACK ");
        }
      }



    free(vselect);
    #else
      preAlps_abort("No other eigensolver is supported for the moment. Please Rebuild with PARPACK !");
    #endif

    eigs->tEigVectors += MPI_Wtime() - ttemp;
  }
#else
    preAlps_abort("No other eigensolver is supported at the moment. Please Rebuild with PARPACK !");
#endif

  return ierr;
}
