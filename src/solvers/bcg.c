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
#define VERBOSE 0
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

int BCGReadParamFromFile(BCG_t* bcg_solver, const char* filename) {
PUSH
BEGIN_TIME
  int ierr = 0, rank, root = 0;
  MPI_Comm_rank(bcg_solver->solver.comm,&rank);
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
        if (strncmp(line,"CG algorithm",10) == 0) {
          _strGetLastToken(line,&lastToken," ");
          if (strncmp(lastToken,"EK",2) == 0)
            bcg_solver->bcg_alg = EK;
          else if (strncmp(lastToken,"Coop",4) == 0)
            bcg_solver->bcg_alg = COOP;
          else if (strncmp(lastToken,"RRHS",4) == 0)
            bcg_solver->bcg_alg = RRHS;
          else {
            fclose(iFile);
            CPALAMEM_Abort("Unknown CG algorithm: %s!",lastToken);
          }
        }
        else if (strncmp(line,"Ortho algorithm",10) == 0) {
          _strGetLastToken(line,&lastToken," ");
          if (strncmp(lastToken,"Orthodir",8) == 0)
            bcg_solver->ortho_alg = ORTHODIR;
          else if (strncmp(lastToken,"Orthomin",8) == 0)
            bcg_solver->ortho_alg = ORTHOMIN;
          else {
            fclose(iFile);
            CPALAMEM_Abort("Unknown ortho algorithm: %s!",lastToken);
          }
        }
        else if (strncmp(line,"Block size reduction",10) == 0) {
          _strGetLastToken(line,&lastToken," ");
          if (strncmp(lastToken,"Alpha-rank",5) == 0)
            bcg_solver->bs_red = ALPHA_RANK;
          else if (strncmp(lastToken,"No",2) == 0)
            bcg_solver->bs_red = NO_BS_RED;
          else {
            fclose(iFile);
            CPALAMEM_Abort("Unknown block size reduction algorithm: %s!",lastToken);
          }
        }
        else if (strncmp(line,"R_k stabilization",10) == 0) {
          _strGetLastToken(line,&lastToken," ");
          if (strncmp(lastToken,"R-QR",3) == 0)
            bcg_solver->rqr_stab = R_QR;
          else if (strncmp(lastToken,"No",2) == 0)
            bcg_solver->rqr_stab = NO_QR_STAB;
          else {
            fclose(iFile);
            CPALAMEM_Abort("Unknown ortho algorithm: %s!",lastToken);
          }
        }
        // Block diagonal preconditioner
        else if (strncmp(line,"Block diagonal",10) == 0) {
          _strGetLastToken(line,&lastToken," ");
          if (strncmp(lastToken,"Yes",2) == 0)
            bcg_solver->prec_type = LEFT_PREC;
          else if (strncmp(lastToken,"No",2) == 0)
            bcg_solver->prec_type = NO_PREC;
          else {
            fclose(iFile);
            CPALAMEM_Abort("Unknown block diagonal option: %s!",lastToken);
          }
        }
        // RHS option
        // else if (strncmp(line,"RHS generation",10) == 0) {
        //   _strGetLastToken(line,&lastToken," ");
        //   if (strcmp(lastToken,"Random") == 0)
        //     bcg_solver->ortho_alg = ORTHODIR;
        //   else
        //     bcg_solver->ortho_alg = ORTHOMIN;
        //}
      }
    }
    fclose(iFile);
  }
  // Broadcast the informations got
  ierr = MPI_Bcast(&(bcg_solver->prec_type),
                   1,
                   MPI_INT,
                   root,
                   bcg_solver->solver.comm);
  checkMPIERR(ierr,"BCGReadParamFromFile::Bcast prec_type");
  ierr = MPI_Bcast(&(bcg_solver->bcg_alg),
                   1,
                   MPI_INT,
                   root,
                   bcg_solver->solver.comm);
  checkMPIERR(ierr,"BCGReadParamFromFile::Bcast bcg_alg");
  ierr = MPI_Bcast(&(bcg_solver->ortho_alg),
                   1,
                   MPI_INT,
                   root,
                   bcg_solver->solver.comm);
  checkMPIERR(ierr,"BCGReadParamFromFile::Bcast ortho_alg");
  ierr = MPI_Bcast(&(bcg_solver->bs_red),
                   1,
                   MPI_INT,
                   root,
                   bcg_solver->solver.comm);
  checkMPIERR(ierr,"BCGReadParamFromFile::Bcast bs_red");
  ierr = MPI_Bcast(&(bcg_solver->rqr_stab),
                   1,
                   MPI_INT,
                   root,
                   bcg_solver->solver.comm);
  checkMPIERR(ierr,"BCGReadParamFromFile::Bcast rqr_stab");
END_TIME
POP
  return ierr;
}

int BCGInitializeOutput(BCG_t* bcg_solver) {
PUSH
BEGIN_TIME
  int ierr = 0, rank;
  MPI_Comm_rank(bcg_solver->solver.comm,&rank);
  // Open output file
  if (rank == 0) {
    char date[20];
    time_t now = time(NULL);
    strftime(date, 20, "%Y-%m-%d_%H-%M-%S",localtime(&now));
    char rootName[19] = "parbcg_out_";
    sprintf(bcg_solver->solver.oFileName,
            "%s%s_%s.txt",
            rootName,
            bcg_solver->solver.name,
            date);
    FILE* oFile = fopen(bcg_solver->solver.oFileName,"w");
    if (oFile == NULL) {
      CPALAMEM_Abort("BCGInitializeOutput::Error: impossible to open the file %s!",bcg_solver->solver.oFileName);
    }
    fprintf(oFile,"########################################################\n");
    fprintf(oFile,"# Par(allel) B(lock) C(onjugate) G(radient)            #\n");
    fprintf(oFile,"# Author: Olivier Tissot                               #\n");
    fprintf(oFile,"# Mail  : olivier.tissot@inria.fr                      #\n");
    fprintf(oFile,"# Date  : 2016/08/02                                   #\n");
    fprintf(oFile,"########################################################\n");
    fprintf(oFile,"\n");
    fprintf(oFile,"########################################################\n");
    fprintf(oFile,
            "Matrix filename           : %s\n",
            bcg_solver->solver.param.matrixFilename);
    const char* strOpt;
    strOpt = (bcg_solver->prec_type == NO_PREC) ? "No" : "Yes";
    fprintf(oFile,
            "Preconditioner            : %s\n",strOpt);
    switch (bcg_solver->bcg_alg) {
      case EK:
        strOpt = "EK-CG";
      break;
      case COOP:
        strOpt = "Coop-CG";
      break;
      case RRHS:
        strOpt = "BRRHS-CG";
      break;
      default:
        strOpt = "Unknown algorithm";
    }
    fprintf(oFile,
            "CG algorithm              : %s\n",strOpt);
    strOpt = (bcg_solver->ortho_alg == ORTHODIR) ? "Orthodir" : "Orthomin";
    fprintf(oFile,
            "Ortho algorithm           : %s\n",strOpt);
    strOpt = (bcg_solver->bs_red == ALPHA_RANK) ? "Alpha-rank" : "No";
    fprintf(oFile,
            "Block size reduction      : %s\n",strOpt);
    strOpt = (bcg_solver->rqr_stab == R_QR) ? "R-QR" : "No";
    fprintf(oFile,
            "R_k stabilization         : %s\n",strOpt);
    fprintf(oFile,
            "tolerance                 : %e\n",
            bcg_solver->solver.param.tolerance);
    fprintf(oFile,
            "iteration maximum         : %d\n",
            bcg_solver->solver.param.iterMax);
    fprintf(oFile,
            "Number of METIS blocks    : %d\n",
            bcg_solver->solver.param.nbBlockPart);
    fprintf(oFile,
            "Number of right hand sides: %d\n",
            bcg_solver->solver.param.nbRHS);
    fprintf(oFile,"########################################################\n");
    fclose(oFile);
  }
END_TIME
POP
  return ierr;
}

int BCGInitialize(BCG_t* bcg_solver) {
PUSH
BEGIN_TIME
  int rank, ierr = 0;
  int nCol = 0;
  // Simplify notations
  Mat_Dense_t* P        = bcg_solver->P;
  Mat_Dense_t* AP       = bcg_solver->AP;
  Mat_Dense_t* X        = bcg_solver->X;
  Mat_Dense_t* R        = bcg_solver->R;
  DVector_t*   b        = bcg_solver->b;
  double*      ptrNormb = &(bcg_solver->normb);

  Mat_Dense_t B = MatDenseNULL();
  MPI_Comm_rank(bcg_solver->solver.comm, &rank);
  // Initialize outptut
  ierr = BCGInitializeOutput(bcg_solver);
  // Initialize iter
  bcg_solver->iter = 0;
  // Compute normb
  ierr = DVector2NormSquared(b, ptrNormb);
  // Sum over all processes
  MPI_Allreduce(MPI_IN_PLACE,
                ptrNormb,
                1,
                MPI_DOUBLE,
                MPI_SUM,
                bcg_solver->solver.comm);
  *ptrNormb = sqrt(*ptrNormb);
  // First we construct R_0
  switch (bcg_solver->bcg_alg) {
    // We assume that x_0 is 0
    case EK:
      // We split b
      nCol = rank % (bcg_solver->solver.param.nbRHS);
      ierr = BCGSplit(b, R, nCol);CHKERR(ierr);
    break;
    case COOP:
      // Create random initial solution
      ierr = MatDenseRandom(X,rank);CHKERR(ierr);
      // R = ones(1,t)*b - A*X_0
      ierr = MatDenseInit(&B,X->info);CHKERR(ierr);
      ierr = MatDenseMalloc(&B);CHKERR(ierr);CHKERR(ierr);
      for (int j = 0; j < B.info.n; ++j) {
        ierr = MatDenseSetColumn(&B, b, j);
        CHKERR(ierr);
      }
      // AP = A*X_0
      ierr = BlockOperator(X, AP);CHKERR(ierr);

      // R = B - AP
      ierr = MatDenseKernelMatAdd(&B,AP,R,-1.0,1.0);
      MatDenseFree(&B);
    break;
    case RRHS:
      // Create random RHS
      ierr = MatDenseRandom(R, rank);CHKERR(ierr);
      // Put the rhs in the first column
      ierr = MatDenseSetColumn(R, b, 0);
    break;
  }

  // Then we construct P_0
  if (bcg_solver->prec_type == NO_PREC) {
    ierr = MatDenseCopy(R,P);CHKERR(ierr);
  }
  else if (bcg_solver->prec_type == LEFT_PREC) {
    ierr = PrecondBlockOperator(R,P);CHKERR(ierr);
  }
END_TIME
POP
  return ierr;
}

int BCGMalloc(BCG_t* bcg_solver, int M, int m, Usr_Param_t* param, const char* name) {
PUSH
BEGIN_TIME
  int ierr = 0;
  // double *tptp = (double*) malloc(666);
  // Malloc the pointers
  bcg_solver->X     = (Mat_Dense_t*) malloc(sizeof(Mat_Dense_t));
  bcg_solver->R     = (Mat_Dense_t*) malloc(sizeof(Mat_Dense_t));
  bcg_solver->P     = (Mat_Dense_t*) malloc(sizeof(Mat_Dense_t));
  bcg_solver->AP    = (Mat_Dense_t*) malloc(sizeof(Mat_Dense_t));
  bcg_solver->alpha = (Mat_Dense_t*) malloc(sizeof(Mat_Dense_t));
  bcg_solver->beta  = (Mat_Dense_t*) malloc(sizeof(Mat_Dense_t));
  bcg_solver->Z     = (Mat_Dense_t*) malloc(sizeof(Mat_Dense_t));
  ierr = SolverInit(&(bcg_solver->solver), M, m, *param, name);CHKERR(ierr);
  Info_Dense_t info = { .M = M,
                        .N = param->nbRHS,
                        .m = m,
                        .n = param->nbRHS,
                        .lda = m,
                        .nval = m*(param->nbRHS),
                        .stor_type = COL_MAJOR };
  ierr = MatDenseCreateZero(bcg_solver->X,info);CHKERR(ierr);
  ierr = MatDenseCreateZero(bcg_solver->R,info);CHKERR(ierr);
  ierr = MatDenseCreate(bcg_solver->P,info);CHKERR(ierr);
  ierr = MatDenseCreate(bcg_solver->AP,info);CHKERR(ierr);
  ierr = MatDenseCreateZero(bcg_solver->Z,info);CHKERR(ierr);
  if (bcg_solver->ortho_alg == ORTHODIR) {
    bcg_solver->P_prev  = (Mat_Dense_t*) malloc(sizeof(Mat_Dense_t));
    bcg_solver->AP_prev = (Mat_Dense_t*) malloc(sizeof(Mat_Dense_t));
    ierr = MatDenseCreateZero(bcg_solver->P_prev,info);CHKERR(ierr);
    ierr = MatDenseCreateZero(bcg_solver->AP_prev,info);CHKERR(ierr);
  }
  // H has nbBlockCG-1 columns maximum
  if (bcg_solver->bs_red == ALPHA_RANK) {
    info.N--;
    info.n--;
    bcg_solver->H = (Mat_Dense_t*) malloc(sizeof(Mat_Dense_t));
    ierr = MatDenseCreateZero(bcg_solver->H,info);CHKERR(ierr);
  }
  Info_Dense_t info_step = { .M = param->nbRHS,
                             .N = param->nbRHS,
                             .m = param->nbRHS,
                             .n = param->nbRHS,
                             .lda = param->nbRHS,
                             .nval = (param->nbRHS)*(param->nbRHS),
                             .stor_type = COL_MAJOR};
  ierr = MatDenseCreate(bcg_solver->alpha,info_step);CHKERR(ierr);
  ierr = MatDenseCreate(bcg_solver->beta,info_step);CHKERR(ierr);
  if (bcg_solver->ortho_alg == ORTHODIR) {
    bcg_solver->gamma  = (Mat_Dense_t*) malloc(sizeof(Mat_Dense_t));
    ierr = MatDenseCreate(bcg_solver->gamma,info_step);CHKERR(ierr);
  }
  // Malloc the working array
  bcg_solver->work = (double*) malloc(bcg_solver->P->info.nval*sizeof(double));
END_TIME
POP
  return ierr;
}

int BCGCreateRandomRhs(BCG_t* bcg_solver, int generatorSeed) {
PUSH
BEGIN_TIME
  int ierr = DVectorRandom(bcg_solver->b, generatorSeed);
END_TIME
POP
  return ierr;
}

int BCGReadRhsFromFile(BCG_t* bcg_solver, const char* filename) {
PUSH
BEGIN_TIME
  IVector_t rowPos = IVectorNULL();
  OperatorGetRowPosPtr(&rowPos);
  DVectorFree(bcg_solver->b);
  int ierr = DVectorLoadAndDistribute(filename,
                                      bcg_solver->b,
                                      &rowPos,
                                      bcg_solver->solver.comm);
  CHKERR(ierr);
END_TIME
POP
  return ierr;
}

int BCGSplit(DVector_t* x, Mat_Dense_t* XSplit, int colIndex) {
PUSH
BEGIN_TIME
  ASSERT(XSplit->val != NULL);
  ASSERT(x->val != NULL);
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
    XSplit->val[i*loop_index_1 + colIndex*loop_index_2] = x->val[i];
  }
END_TIME
POP
  return ierr;
}

int BCGStoppingCriterion(BCG_t* bcg_solver, int* stop, int* min_index) {
PUSH
BEGIN_TIME
  int ierr = 0;
  double norm = 0.0;
  // Simplify notations
  int iterMax      = bcg_solver->solver.param.iterMax;
  double tolerance = bcg_solver->solver.param.tolerance;
  double normb     = bcg_solver->normb;
  MPI_Comm comm    = bcg_solver->solver.comm;
  Mat_Dense_t* R   = bcg_solver->R;

  DVector_t reducedR = DVectorNULL();
  Mat_Dense_t RtR = MatDenseNULL();
  ASSERT(stop != NULL);
  *min_index = -1;
  switch (bcg_solver->bcg_alg) {
    case EK:
      // Sum the columns of the block residual
      ierr = MatDenseKernelSumColumns(R, &reducedR);CHKERR(ierr);
      // Sum over the line of the reduced residual
      ierr = DVector2NormSquared(&reducedR, &norm);CHKERR(ierr);
      // Sum over all processes
      MPI_Allreduce(MPI_IN_PLACE,&norm,1,MPI_DOUBLE,MPI_SUM,comm);
    break;
    case COOP:
      ASSERT(min_index != NULL);
      ierr = 1;
      ierr = MatDenseMatDotProd(R,R,&RtR,comm);CHKERR(ierr);
      norm = 1e12;
      *min_index = -1;
      // RtR is a squared matrix
      for (int i = 0; i < RtR.info.m; ++i) {
        norm = fmin(norm, RtR.val[i*RtR.info.m + i]);
        *min_index = i;
      }
      MatDenseFree(&RtR);
    break;
    case RRHS:
      ierr = 1;
      // Get first column of R
      ierr = MatDenseGetColumn(R,&reducedR,0);CHKERR(ierr);
      // Sum over the line of the reduced residual
      ierr = DVector2NormSquared(&reducedR, &norm);CHKERR(ierr);
      // Sum over all processes
      MPI_Allreduce(MPI_IN_PLACE,&norm,1,MPI_DOUBLE,MPI_SUM,comm);
    break;
  }

  norm = sqrt(norm);

  // Stopping criterion
  if (norm > normb*tolerance && bcg_solver->iter < iterMax )
    *stop = 0; // we continue
  else {
    *stop = 1; // we stop
    bcg_solver->solver.finalRes = norm;
  }

  DVectorFree(&reducedR);
END_TIME
POP
  return ierr;
}

int BCGIterate(BCG_t* bcg_solver) {
PUSH
BEGIN_TIME
OPEN_TIMER
  int ierr = 0;
  // Simplify notations
  MPI_Comm     comm    = bcg_solver->solver.comm;
  Mat_Dense_t* P       = bcg_solver->P;
  Mat_Dense_t* AP      = bcg_solver->AP;
  Mat_Dense_t* AP_prev = bcg_solver->AP_prev;
  Mat_Dense_t* P_prev  = bcg_solver->P_prev;
  Mat_Dense_t* X       = bcg_solver->X;
  Mat_Dense_t* R       = bcg_solver->R;
  Mat_Dense_t* alpha   = bcg_solver->alpha;
  Mat_Dense_t* beta    = bcg_solver->beta;
  Mat_Dense_t* gamma   = bcg_solver->gamma;
  Mat_Dense_t* Z       = bcg_solver->Z;

  // AP = A*P
TIC
    ierr = BlockOperator(P,AP);
TAC("A*P")

TIC
  ierr = MatDenseACholQR(P,AP,alpha,comm);
TAC("A-CholQR")

  // alpha = P^t*R
TIC
  ierr = MatDenseMatDotProd(P,R,alpha,comm);
TAC("alpha = P^t*R")

  // X = X + P*alpha
TIC
  ierr = MatDenseKernelMatMult(P,'N',alpha,'N',X,1.0,1.0);
TAC("X = X + P*alpha")

  // R = R - AP*alpha
TIC
  ierr = MatDenseKernelMatMult(AP,'N',alpha,'N',R,-1.0,1.0);
TAC("R = R - AP*alpha")

TIC
  if (bcg_solver->ortho_alg == ORTHODIR) {
    // Z = AP
    if (bcg_solver->prec_type == NO_PREC) {
      ierr = MatDenseCopy(AP,Z);
    }
    // Z = precond(AP)
    else if (bcg_solver->prec_type == LEFT_PREC) {
      ierr = PrecondBlockOperator(AP,Z);
    }
TAC("Z = M^-1*AP")
  }
  else if (bcg_solver->ortho_alg == ORTHOMIN) {
    // Z = R
    if (bcg_solver->prec_type == NO_PREC) {
      ierr = MatDenseCopy(R,Z);
    }
    // Z = precond(R)
    else if (bcg_solver->prec_type == LEFT_PREC) {
      ierr = PrecondBlockOperator(R,Z);
    }
TAC("Z = M^-1*R")
  }

  // beta = (AP)^t*Z
TIC
  ierr = MatDenseMatDotProd(AP,Z,beta,comm);
TAC("beta = (AP)^t*Z")

  // Z = Z - P*beta
TIC
  ierr = MatDenseKernelMatMult(P,'N',beta,'N',Z,-1.0,1.0);
TAC("Z = Z - P*beta")

  if (bcg_solver->ortho_alg == ORTHODIR) {
    // gamma = (AP_prev)^t*Z
TIC
    ierr = MatDenseMatDotProd(AP_prev,Z,gamma,comm);
TAC("gamma = (AP_prev)^t*Z")

    // Z = Z - P_prev*gamma
TIC
    ierr = MatDenseKernelMatMult(P_prev,'N',gamma,'N',Z,-1.0,1.0);
TAC("Z = Z - P_prev*gamma")
  }

  // Swapping time
  MatDenseSwap(P,Z);
  if (bcg_solver->ortho_alg == ORTHODIR) {
    MatDenseSwap(AP,AP_prev);
    MatDenseSwap(P_prev,Z);
  }

  bcg_solver->iter++;
CLOSE_TIMER
END_TIME
POP
  return ierr;
}

// P = AP - PP^t AA P - P_prev P_prev^t AA P_prev (two times)
int BCGOrthodir(BCG_t* bcg_solver) {
PUSH
BEGIN_TIME
  int ierr = 0;
  double* swapVal = NULL;
  // Simplify notations
  MPI_Comm     comm    = bcg_solver->solver.comm;
  Mat_Dense_t* P       = bcg_solver->P;
  Mat_Dense_t* P_prev  = bcg_solver->P_prev;
  Mat_Dense_t* AP      = bcg_solver->AP;
  Mat_Dense_t* AP_prev = bcg_solver->AP_prev;
  Mat_Dense_t* beta    = bcg_solver->beta;
  Mat_Dense_t* gamma   = bcg_solver->gamma;
  double* work = bcg_solver->work;

  int algGS = 1;
  /* Classical Gram-Schmidt */
  if (algGS == 0) {
    // beta = P^t AA P
    ierr = MatDenseMatDotProd(AP,AP,beta,comm);CHKERR(ierr);
    // gamma = P_prev^t AA P
    ierr = MatDenseMatDotProd(AP_prev,AP,gamma,comm);CHKERR(ierr);
    // AP_prev = AP
    memcpy(AP_prev->val,AP->val,AP->info.nval*sizeof(double));
    // AP = AP - P*beta
    ierr = MatDenseKernelMatMult(P,'N',beta,'N',AP,-1.0,1.0);CHKERR(ierr);
    // AP = AP - P_prev*gamma
    ierr = MatDenseKernelMatMult(P_prev,'N',gamma,'N',AP,-1.0,1.0);CHKERR(ierr);

    // P_prev <- P ; P <- AP ; P_prev <- AP;
    swapVal     = P_prev->val;
    P_prev->val = P->val;
    P->val      = AP->val;
    AP->val     = swapVal;
  }
  /* Modified Gram-Schmidt */
  else if (algGS == 1) {
    // beta = P^t AA P
    ierr = MatDenseMatDotProd(AP,AP,beta,comm);CHKERR(ierr);
    // Copy AP_prev in order to do the 2nd iteration
    memcpy(work,AP_prev->val,AP->info.nval*sizeof(double));
    memcpy(AP_prev->val,AP->val,AP->info.nval*sizeof(double));
    // // AP = AP - P*beta
    ierr = MatDenseKernelMatMult(P,'N',beta,'N',AP,-1.0,1.0);CHKERR(ierr);
    // gamma = P_prev^t A AP
    swapVal = AP_prev->val;
    AP_prev->val = work;
    ierr = MatDenseMatDotProd(AP_prev,AP,gamma,comm);CHKERR(ierr);
    AP_prev->val = swapVal;
    // AP = AP - P_prev*gamma
    ierr = MatDenseKernelMatMult(P_prev,'N',gamma,'N',AP,-1.0,1.0);CHKERR(ierr);

    // P_prev <- P ; P <- AP ; P_prev <- AP;
    swapVal     = P_prev->val;
    P_prev->val = P->val;
    P->val      = AP->val;
    AP->val     = swapVal;
    CHKERR(ierr);
  }

END_TIME
POP
  return ierr;
}

// P = R - P P^t A R (two times)
int BCGOrthomin(BCG_t* bcg_solver) {
PUSH
BEGIN_TIME
  int ierr = 0;
  double* swapVal = NULL;
  // Simplify notations
  MPI_Comm     comm    = bcg_solver->solver.comm;
  Mat_Dense_t* P       = bcg_solver->P;
  Mat_Dense_t* AP      = bcg_solver->AP;
  Mat_Dense_t* R       = bcg_solver->R;
  Mat_Dense_t* beta    = bcg_solver->beta;

  /* Classical Gram-Schmidt */
  // beta = P^t A R
  ierr = MatDenseMatDotProd(AP,R,beta,comm);CHKERR(ierr);
  // AP = R
  memcpy(AP->val,R->val,AP->info.nval*sizeof(double));
  // AP = AP - P*beta
  ierr = MatDenseKernelMatMult(P,'N',beta,'N',AP,-1.0,1.0);CHKERR(ierr);

  // swap P and AP
  swapVal = P->val;
  P->val  = AP->val;
  AP->val = swapVal;

END_TIME
POP
  return ierr;
}

int BCGFinalize(BCG_t* bcg_solver, int min_index) {
PUSH
BEGIN_TIME
  int ierr;
  // Simplify notations
  Mat_Dense_t* X = bcg_solver->X;

  DVector_t solution = DVectorNULL();
  // Get the solution
  switch (bcg_solver->bcg_alg) {
    case EK:
      ierr = MatDenseKernelSumColumns(X, &solution);CHKERR(ierr);
    break;
    case COOP:
      // Get column of minimum residual
      ierr = MatDenseGetColumn(X, &solution, min_index);CHKERR(ierr);
    break;
    case RRHS:
      // Get column 0 of X
      ierr = MatDenseGetColumn(X, &solution, 0);CHKERR(ierr);
    break;
  }
  // Dump results
  ierr = BCGDump(bcg_solver);CHKERR(ierr);
  //  char oSolFileName[60];
  //  char rootName[22] = "parbcg_sol_out_";
  //  int rank;
  //  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  //  sprintf(oSolFileName,"%s%s_%d.txt",rootName, bcg_solver->solver.name, rank);
  //  FILE* oSolFile = fopen(oSolFileName,"w");
  //  for (int i = 0; i < solution.nval; ++i)
  //    fprintf(oSolFile,"%.14f\n",solution.val[i]);
  // Free memory
  DVectorFree(&solution);
  BCGFree(bcg_solver);
END_TIME
POP
  return ierr;
}

int BCGDump(BCG_t* bcg_solver) {
PUSH
BEGIN_TIME
  int ierr = 0, rank;
  MPI_Comm_rank(bcg_solver->solver.comm,&rank);
  if (rank == 0) {
    FILE* oFile = fopen(bcg_solver->solver.oFileName,"a");
    fprintf(oFile,"########################################################\n");
    if (bcg_solver->iter < bcg_solver->solver.param.iterMax) {
      fprintf(oFile, "The method converged!\n");
      fprintf(oFile, "Number of iteration: %d\n",bcg_solver->iter);
      fprintf(oFile, "Residual           : %.14e\n",bcg_solver->solver.finalRes);
      fprintf(oFile, "Normalized residual: %.14e\n",bcg_solver->solver.finalRes/bcg_solver->normb);
    }
    else
      fprintf(oFile, "/!\\ The method did not converge! /!\\\n");
    fprintf(oFile,"########################################################\n");
    fclose(oFile);
  }
END_TIME
POP
  return ierr;
}

void BCGPrint(BCG_t* bcg_solver) {
PUSH
BEGIN_TIME
  int rank;
  MPI_Comm_rank(bcg_solver->solver.comm,&rank);
  printf("[%d] prints BCG_t...\n", rank);
  MatDensePrintfInfo("X",    bcg_solver->X);
  MatDensePrintfInfo("R",    bcg_solver->R);
  MatDensePrintfInfo("P",    bcg_solver->P);
  MatDensePrintfInfo("AP",   bcg_solver->AP);
  MatDensePrintfInfo("alpha",bcg_solver->alpha);
  printf("\n");
  printf("iter: %d\n",bcg_solver->iter);
  printf("[%d] ends printing BCG_t!\n", rank);
END_TIME
POP
}

int BCGSolve(BCG_t* bcg_solver, DVector_t* rhs, Usr_Param_t* param, const char* name) {
PUSH
BEGIN_TIME
OPEN_TIMER
  int ierr = 0;
  int M, m, rank, min_index;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ierr = OperatorGetSizes(&M,&m);CHKERR(ierr);

  if (PRINT >= 1) {
    printf("[%d] BCGMalloc...\n",rank);
  }

  ierr = BCGMalloc(bcg_solver, M, m, param, name);CHKERR(ierr);

  if (PRINT >= 1) {
    printf("[%d] BCGMalloc done!\n",rank);
    printf("[%d] BCGCreateRandomRhs...\n", rank);
  }

  bcg_solver->b = rhs;
  // Change R when using block Jacobi
  if (bcg_solver->prec_type == LEFT_PREC)
    ierr = BlockJacobiInitialize(bcg_solver->b);

  if (PRINT >= 1) {
    printf("[%d] BCGCreateRandomRhs done!\n", rank);
    printf("[%d] BCGInitialize...\n", rank);
  }

  ierr = BCGInitialize(bcg_solver);CHKERR(ierr);

  if (PRINT >= 1) {
    printf("[%d] BCGInitialize done!\n", rank);
  }

  int stop = 0;
  while (stop != 1) {

    if (PRINT >= 1) {
        printf("[%d] BCGIterate:: %d\n", rank, bcg_solver->iter);
    }

    ierr = BCGIterate(bcg_solver);

    if (PRINT >= 3) {
        printf("[%d] BCGIterate done!\n", rank);
        printf("[%d] BCGIterateStoppingCriterion...\n", rank);
    }

    ierr = BCGStoppingCriterion(bcg_solver,&stop,&min_index);

    if (PRINT >= 3) {
        printf("[%d] BCGStoppingCriterion done!\n", rank);
    }

  }

  if (PRINT >= 1) {
    printf("[%d] BCGFinalize...\n", rank);
  }

  BCGFinalize(bcg_solver,min_index);

  if (PRINT >= 1) {
    printf("[%d] BCGFinalize done!\n", rank);
  }
CLOSE_TIMER
END_TIME
POP
  return ierr;
}

void BCGFree(BCG_t* bcg_solver) {
PUSH
BEGIN_TIME
  MatDenseFree(bcg_solver->X);
  if (bcg_solver->X != NULL)
    free(bcg_solver->X);
  MatDenseFree(bcg_solver->R);
  if (bcg_solver->R != NULL)
    free(bcg_solver->R);
  MatDenseFree(bcg_solver->P);
  if (bcg_solver->P != NULL)
    free(bcg_solver->P);
  MatDenseFree(bcg_solver->AP);
  if (bcg_solver->AP != NULL)
    free(bcg_solver->AP);
  MatDenseFree(bcg_solver->alpha);
  if (bcg_solver->alpha != NULL)
    free(bcg_solver->alpha);
  MatDenseFree(bcg_solver->beta);
  if (bcg_solver->beta != NULL)
    free(bcg_solver->beta);
  MatDenseFree(bcg_solver->Z);
  if (bcg_solver->Z != NULL)
    free(bcg_solver->Z);
  if (bcg_solver->ortho_alg == ORTHODIR) {
    MatDenseFree(bcg_solver->P_prev);
    if (bcg_solver->P_prev != NULL)
      free(bcg_solver->P_prev);
    MatDenseFree(bcg_solver->AP_prev);
    if (bcg_solver->AP_prev != NULL)
      free(bcg_solver->AP_prev);
    MatDenseFree(bcg_solver->gamma);
    if (bcg_solver->gamma != NULL)
      free(bcg_solver->gamma);
  }
  if (bcg_solver->bs_red == ALPHA_RANK) {
    MatDenseFree(bcg_solver->H);
    if (bcg_solver->H != NULL)
      free(bcg_solver->H);
  }
  if (bcg_solver->work != NULL)
    free(bcg_solver->work);
  SolverFree(&(bcg_solver->solver));
END_TIME
POP
}

/******************************************************************************/
