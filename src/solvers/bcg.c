/******************************************************************************/
/* Author     : Olivier Tissot                                                */
/* Creation   : 2016/06/24                                                    */
/* Description: Block Preconditioned C(onjugate) G(radient)                   */
/******************************************************************************/

/******************************************************************************/
/*                                  INCLUDE                                   */
/******************************************************************************/
#include "bcg.h"
/******************************************************************************/

/******************************************************************************/
/*                                    CODE                                    */
/******************************************************************************/
/* Private function to get the last token of a c-string */
int _strGetLastToken(char* str, char** lastToken, const char* delimiters) {
PUSH
  ASSERT(str != NULL);
  char* buffer = NULL;
  buffer = strtok(str, delimiters);
  while (buffer != NULL) {
    *lastToken = buffer;
    buffer = strtok(NULL,delimiters);
  }
  if (buffer) free(buffer);
POP
  return (*lastToken == NULL);
}

int BCGReadParamFromFile(BCG_t* bcg, const char* filename) {
PUSH
BEGIN_TIME
  int ierr = 0, rank, root = 0;
  MPI_Comm_rank(bcg->comm,&rank);
  if (rank == root) {
    char* line = NULL;
    char* lastToken = NULL;
    size_t len = 0;
    FILE* iFile = fopen(filename,"r");
    if (iFile == NULL)
      CPALAMEM_Abort("Impossible to open the file %s!",filename);
    // Read the file line by line
    while (parbcg_getline(&line, &len, iFile) != -1) {
      // Skip comments
      if (line[0] != '#') {
        // Switch over the parameters
        if (strncmp(line,"Ortho algorithm",10) == 0) {
          _strGetLastToken(line,&lastToken," ");
          if (strncmp(lastToken,"Orthodir",8) == 0)
            bcg->ortho_alg = ORTHODIR;
          else if (strncmp(lastToken,"Orthomin",8) == 0)
            bcg->ortho_alg = ORTHOMIN;
          else {
            fclose(iFile);
            CPALAMEM_Abort("Unknown ortho algorithm: %s!",lastToken);
          }
        }
        else if (strncmp(line,"Block size reduction",10) == 0) {
          _strGetLastToken(line,&lastToken," ");
          if (strncmp(lastToken,"Alpha-rank",5) == 0)
            bcg->bs_red = ALPHA_RANK;
          else if (strncmp(lastToken,"No",2) == 0)
            bcg->bs_red = NO_BS_RED;
          else {
            fclose(iFile);
            CPALAMEM_Abort("Unknown block size reduction algorithm: %s!",lastToken);
          }
        }
        // Block diagonal preconditioner
        else if (strncmp(line,"Precond type",8) == 0) {
          _strGetLastToken(line,&lastToken," ");
          if (strncmp(lastToken,"BJacobi",5) == 0){
            bcg->precond_side = LEFT_PREC;
            bcg->precond_type = PREALPS_BLOCKJACOBI;
          }
          else if (strncmp(lastToken,"No",2) == 0){
            bcg->precond_type = PREALPS_NOPREC;
          }
          else {
            fclose(iFile);
            CPALAMEM_Abort("Unknown preconditioner type option: %s!",lastToken);
          }
        }
      }
    }
    fclose(iFile);
  }
  // Broadcast the informations got
  ierr = MPI_Bcast(&(bcg->precond_side),
                   1,
                   MPI_INT,
                   root,
                   bcg->comm);
  checkMPIERR(ierr,"BCGReadParamFromFile::Bcast precond_side");
  ierr = MPI_Bcast(&(bcg->precond_type),
                   1,
                   MPI_INT,
                   root,
                   bcg->comm);
  checkMPIERR(ierr,"BCGReadParamFromFile::Bcast precond_type");
  ierr = MPI_Bcast(&(bcg->ortho_alg),
                   1,
                   MPI_INT,
                   root,
                   bcg->comm);
  checkMPIERR(ierr,"BCGReadParamFromFile::Bcast ortho_alg");
  ierr = MPI_Bcast(&(bcg->bs_red),
                   1,
                   MPI_INT,
                   root,
                   bcg->comm);
  checkMPIERR(ierr,"BCGReadParamFromFile::Bcast bs_red");

END_TIME
POP
  return ierr;
}

int BCGInitialize(BCG_t* bcg, double* rhs, int* rci_request) {
PUSH
BEGIN_TIME
  int rank, ierr = 0;
  int nCol = 0;
  // Simplify notations
  Mat_Dense_t* P        = bcg->P;
  Mat_Dense_t* R        = bcg->R;
  double*      ptrNormb = &(bcg->normb);

  // TODO remove this
  DVector_t b           = DVectorNULL();
  b.nval = P->info.m;
  b.val = rhs;
  // End TODO

  MPI_Comm_rank(bcg->comm, &rank);
  bcg->iter = 0;
  // Compute normb
  ierr = DVector2NormSquared(&b, ptrNormb);
  // Sum over all processes
  MPI_Allreduce(MPI_IN_PLACE,
                ptrNormb,
                1,
                MPI_DOUBLE,
                MPI_SUM,
                bcg->comm);
  *ptrNormb = sqrt(*ptrNormb);

  // First we construct R_0 by splitting b
  nCol = rank % (bcg->param.nbRHS);
  ierr = BCGSplit(rhs, R, nCol);CHKERR(ierr);

  // Then we need to construct R_0 and P_0
  *rci_request = 0;
END_TIME
POP
  return ierr;
}

int BCGMalloc(BCG_t* bcg, int M, int m, Usr_Param_t* param, const char* name) {
PUSH
BEGIN_TIME
  int ierr = 0;
  bcg->param = *param;
  bcg->name     = name;
  // Malloc the pointers
  bcg->X        = (Mat_Dense_t*) malloc(sizeof(Mat_Dense_t));
  bcg->R        = (Mat_Dense_t*) malloc(sizeof(Mat_Dense_t));
  bcg->P        = (Mat_Dense_t*) malloc(sizeof(Mat_Dense_t));
  bcg->AP       = (Mat_Dense_t*) malloc(sizeof(Mat_Dense_t));
  bcg->alpha    = (Mat_Dense_t*) malloc(sizeof(Mat_Dense_t));
  bcg->beta     = (Mat_Dense_t*) malloc(sizeof(Mat_Dense_t));
  bcg->Z        = (Mat_Dense_t*) malloc(sizeof(Mat_Dense_t));
  Info_Dense_t info = { .M = M,
                        .N = param->nbRHS,
                        .m = m,
                        .n = param->nbRHS,
                        .lda = m,
                        .nval = m*(param->nbRHS),
                        .stor_type = COL_MAJOR };
  ierr = MatDenseCreateZero(bcg->X,info);CHKERR(ierr);
  ierr = MatDenseCreateZero(bcg->R,info);CHKERR(ierr);
  ierr = MatDenseCreate(bcg->P,info);CHKERR(ierr);
  ierr = MatDenseCreate(bcg->AP,info);CHKERR(ierr);
  ierr = MatDenseCreateZero(bcg->Z,info);CHKERR(ierr);
  if (bcg->ortho_alg == ORTHODIR) {
    bcg->P_prev  = (Mat_Dense_t*) malloc(sizeof(Mat_Dense_t));
    bcg->AP_prev = (Mat_Dense_t*) malloc(sizeof(Mat_Dense_t));
    ierr = MatDenseCreateZero(bcg->P_prev,info);CHKERR(ierr);
    ierr = MatDenseCreateZero(bcg->AP_prev,info);CHKERR(ierr);
  }
  // H has nbBlockCG-1 columns maximum
  if (bcg->bs_red == ALPHA_RANK) {
    info.N--;
    info.n--;
    bcg->H = (Mat_Dense_t*) malloc(sizeof(Mat_Dense_t));
    bcg->AH = (Mat_Dense_t*) malloc(sizeof(Mat_Dense_t));
    ierr = MatDenseCreateZero(bcg->H,info);CHKERR(ierr);
    ierr = MatDenseCreateZero(bcg->AH,info);CHKERR(ierr);
  }
  Info_Dense_t info_step = { .M = param->nbRHS,
                             .N = param->nbRHS,
                             .m = param->nbRHS,
                             .n = param->nbRHS,
                             .lda = param->nbRHS,
                             .nval = (param->nbRHS)*(param->nbRHS),
                             .stor_type = COL_MAJOR};
  ierr = MatDenseCreate(bcg->alpha,info_step);CHKERR(ierr);
  ierr = MatDenseCreate(bcg->beta,info_step);CHKERR(ierr);
  if (bcg->ortho_alg == ORTHODIR) {
    bcg->gamma  = (Mat_Dense_t*) malloc(sizeof(Mat_Dense_t));
    ierr = MatDenseCreate(bcg->gamma,info_step);CHKERR(ierr);
  }
  // Malloc the working arrays
  bcg->work = (double*) malloc(bcg->P->info.nval*sizeof(double));
  bcg->iwork = (int*) malloc(param->nbRHS*sizeof(int));

END_TIME
POP
  return ierr;
}

int BCGSplit(double* x, Mat_Dense_t* XSplit, int colIndex) {
PUSH
BEGIN_TIME
  ASSERT(XSplit->val != NULL);
  ASSERT(x != NULL);
  int ierr = 0;
  int loop_index_1, loop_index_2;
  if (XSplit->info.stor_type == ROW_MAJOR) {
    loop_index_1 = XSplit->info.n;
    loop_index_2 = 1;
  }
  else {
    loop_index_1 = 1;
    loop_index_2 = XSplit->info.m;
  }
  // Assume that XSplit is zero everywhere
  for (int i = 0; i < XSplit->info.m; i++) {
    XSplit->val[i*loop_index_1 + colIndex*loop_index_2] = x[i];
  }
END_TIME
POP
  return ierr;
}

int BCGStoppingCriterion(BCG_t* bcg, int* stop) {
PUSH
BEGIN_TIME
int ierr = 0;
  // Simplify notations
  MPI_Comm comm    = bcg->comm;
  Mat_Dense_t* R   = bcg->R;
  double tolerance = bcg->param.tolerance;
  double normb     = bcg->normb;
  double* ptrRes   = &(bcg->res);
  double* work     = bcg->work;
  int iterMax      = bcg->param.iterMax;

  DVector_t r = DVectorNULL();
  r.nval = R->info.m;
  r.val  = work;

  ASSERT(stop != NULL);

  // Sum the columns of the block residual
  ierr = MatDenseKernelSumColumns(R, &r);CHKERR(ierr);
  // Sum over the line of the reduced residual
  *ptrRes = 0.0;
  ierr = DVector2NormSquared(&r, ptrRes);CHKERR(ierr);
  // Sum over all processes
  MPI_Allreduce(MPI_IN_PLACE,ptrRes,1,MPI_DOUBLE,MPI_SUM,comm);

  *ptrRes = sqrt(*ptrRes);

  // Stopping criterion
  if (*ptrRes > normb*tolerance && bcg->iter < iterMax )
    *stop = 0; // we continue
  else
    *stop = 1; // we stop

END_TIME
POP
  return ierr;
}

int BCGIterate(BCG_t* bcg, int* rci_request) {
PUSH
BEGIN_TIME
OPEN_TIMER
  int ierr = -1;
  /* int nrhs = bcg->param.nbRHS; */
  // Simplify notations
  MPI_Comm     comm    = bcg->comm;
  Mat_Dense_t* P       = bcg->P;
  Mat_Dense_t* AP      = bcg->AP;
  Mat_Dense_t* AP_prev = bcg->AP_prev;
  Mat_Dense_t* P_prev  = bcg->P_prev;
  Mat_Dense_t* X       = bcg->X;
  Mat_Dense_t* R       = bcg->R;
  /* Mat_Dense_t* H       = bcg->H; */
  /* Mat_Dense_t* AH      = bcg->AH; */
  Mat_Dense_t* alpha   = bcg->alpha;
  Mat_Dense_t* beta    = bcg->beta;
  Mat_Dense_t* gamma   = bcg->gamma;
  Mat_Dense_t* Z       = bcg->Z;
  /* int M = P->info.M; */
  /* int m = P->info.m; */
  /* int t = P->info.n; */

  if (*rci_request == 0) {
    // We have A*P here !!
    TIC(step1,"ACholQR")
    ierr = MatDenseACholQR(P,AP,beta,comm);
    TAC(step1)
    TIC(step2,"alpha = P^t*R")
    ierr = MatDenseMatDotProd(P,R,alpha,comm);
    TAC(step2)
    TIC(step3,"X = X + P*alpha")
    ierr = MatDenseKernelMatMult(P,'N',alpha,'N',X,1.0,1.0);
    TAC(step3)
    TIC(step4,"R = R - AP*alpha")
    ierr = MatDenseKernelMatMult(AP,'N',alpha,'N',R,-1.0,1.0);
    TAC(step4)
    // Iteration finished
    bcg->iter++;
    // Now we need the preconditioner
    *rci_request = 1;
  }
  else if (*rci_request == 1) {
    // We have the preconditioner here !!!
    TIC(step5,"beta = (AP)^t*Z")
    ierr = MatDenseMatDotProd(AP,Z,beta,comm);
    TAC(step5)
    TIC(step6,"Z = Z - P*beta")
    ierr = MatDenseKernelMatMult(P,'N',beta,'N',Z,-1.0,1.0);
    TAC(step6)
    if (bcg->ortho_alg == ORTHODIR) {
      TIC(step7,"gamma = (AP_prev)^t*Z")
      ierr = MatDenseMatDotProd(AP_prev,Z,gamma,comm);
      TAC(step7)
      TIC(step8,"Z = Z - P_prev*gamma")
      ierr = MatDenseKernelMatMult(P_prev,'N',gamma,'N',Z,-1.0,1.0);
      TAC(step8)
    }
    // Swapping time
    MatDenseSwap(P,Z);
    if (bcg->ortho_alg == ORTHODIR) {
      MatDenseSwap(AP,AP_prev);
      MatDenseSwap(P_prev,Z);
    }
    // Now we need A*P to continue
    *rci_request = 0;
  }
  else {
    CPALAMEM_Abort("Internal error: wrong rci_request value: %d",*rci_request);
  }
CLOSE_TIMER
END_TIME
POP
  return ierr;
}

int BCGFinalize(BCG_t* bcg, double* solution) {
PUSH
BEGIN_TIME
  int ierr = 0;
  // Simplify notations
  Mat_Dense_t* X = bcg->X;
  DVector_t sol = DVectorNULL();
  sol.nval = X->info.m;
  sol.val  = solution;
  // Get the solution
  ierr = MatDenseKernelSumColumns(X, &sol);CHKERR(ierr);
END_TIME
POP
  return ierr;
}

void BCGPrint(BCG_t* bcg) {
PUSH
BEGIN_TIME
  int rank;
  MPI_Comm_rank(bcg->comm,&rank);
  printf("[%d] prints BCG_t...\n", rank);
  MatDensePrintfInfo("X",    bcg->X);
  MatDensePrintfInfo("R",    bcg->R);
  MatDensePrintfInfo("P",    bcg->P);
  MatDensePrintfInfo("AP",   bcg->AP);
  MatDensePrintfInfo("Z",    bcg->Z);
  MatDensePrintfInfo("alpha",bcg->alpha);
  MatDensePrintfInfo("beta", bcg->beta);
  if (bcg->ortho_alg == ORTHODIR) {
    MatDensePrintfInfo("P_prev",bcg->P_prev);
    MatDensePrintfInfo("AP_prev",bcg->AP_prev);
    MatDensePrintfInfo("gamma",  bcg->gamma);
  }
  if (bcg->bs_red == ALPHA_RANK) {
    MatDensePrintfInfo("H",bcg->H);
    MatDensePrintfInfo("AH",  bcg->AH);
  }
  printf("\n");
  printf("iter: %d\n",bcg->iter);
  printf("[%d] ends printing BCG_t!\n", rank);
END_TIME
POP
}

/* int BCGSolve(BCG_t* bcg, DVector_t* rhs, Usr_Param_t* param, const char* name) { */
/* PUSH */
/* BEGIN_TIME */
/* OPEN_TIMER */
/*   int ierr = 0; */
/*   int M, m, rank; */
/*   MPI_Comm_rank(MPI_COMM_WORLD, &rank); */
/*   ierr = OperatorGetSizes(&M,&m);CHKERR(ierr); */
/*   // Allocate memory */
/*   ierr = BCGMalloc(bcg, M, m, param, name);CHKERR(ierr); */
/*   // Set-up rhs */
/*   bcg->b = rhs; */
/*   // Initialize variables */
/*   ierr = BCGInitialize(bcg,rhs->val);CHKERR(ierr); */
/*   int rci_request = 0; */
/*   int stop = 0; */
/*   // Main loop */
/*   BlockOperator(bcg->P,bcg->AP); */
/*   while (stop != 1) { */
/*     ierr = BCGIterate(bcg,&rci_request); */
/*     if (rci_request == 0) { */
/*       BlockOperator(bcg->P,bcg->AP); */
/*     } */
/*     else if (rci_request = 1) { */
/*       ierr = BCGStoppingCriterion(bcg,&stop); */
/*       if (stop == 1) break; */
/*       PrecondApply(bcg->precond_type,bcg->R,bcg->Z); */
/*     } */
/*   } */
/*   // Retrieve solution */
/*   //BCGFinalize(bcg); */
/*   // Release memory */
/*   BCGFree(bcg); */
/* CLOSE_TIMER */
/* END_TIME */
/* POP */
/*   return ierr; */
/* } */

void BCGFree(BCG_t* bcg) {
PUSH
BEGIN_TIME
  MatDenseFree(bcg->X);
  if (bcg->X != NULL)
    free(bcg->X);
  MatDenseFree(bcg->R);
  if (bcg->R != NULL)
    free(bcg->R);
  MatDenseFree(bcg->P);
  if (bcg->P != NULL)
    free(bcg->P);
  MatDenseFree(bcg->AP);
  if (bcg->AP != NULL)
    free(bcg->AP);
  MatDenseFree(bcg->alpha);
  if (bcg->alpha != NULL)
    free(bcg->alpha);
  MatDenseFree(bcg->beta);
  if (bcg->beta != NULL)
    free(bcg->beta);
  MatDenseFree(bcg->Z);
  if (bcg->Z != NULL)
    free(bcg->Z);
  if (bcg->ortho_alg == ORTHODIR) {
    MatDenseFree(bcg->P_prev);
    if (bcg->P_prev != NULL)
      free(bcg->P_prev);
    MatDenseFree(bcg->AP_prev);
    if (bcg->AP_prev != NULL)
      free(bcg->AP_prev);
    MatDenseFree(bcg->gamma);
    if (bcg->gamma != NULL)
      free(bcg->gamma);
  }
  if (bcg->bs_red == ALPHA_RANK) {
    MatDenseFree(bcg->H);
    if (bcg->H != NULL)
      free(bcg->H);
    MatDenseFree(bcg->AH);
    if (bcg->AH != NULL)
      free(bcg->AH);
  }
  if (bcg->work != NULL)
    free(bcg->work);
END_TIME
POP
}

/******************************************************************************/
