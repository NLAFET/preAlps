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
        else if (strncmp(line,"Block diagonal",10) == 0) {
          _strGetLastToken(line,&lastToken," ");
          if (strncmp(lastToken,"Yes",2) == 0){
            bcg->precond_side = LEFT_PREC;
            bcg->precond_type = PREALPS_BLOCKJACOBI;
          }
          else if (strncmp(lastToken,"No",2) == 0){
            bcg->precond_type = PREALPS_NOPREC;
          }
          else {
            fclose(iFile);
            CPALAMEM_Abort("Unknown block diagonal option: %s!",lastToken);
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

int BCGInitializeOutput(BCG_t* bcg) {
PUSH
BEGIN_TIME
  int ierr = 0, rank;
  MPI_Comm_rank(bcg->comm,&rank);
  // Open output file
  if (rank == 0) {
    char date[20];
    time_t now = time(NULL);
    strftime(date, 20, "%Y-%m-%d_%H-%M-%S",localtime(&now));
    char rootName[19] = "parbcg_out_";
    sprintf(bcg->oFileName,
            "%s%s_%s.txt",
            rootName,
            bcg->name,
            date);
    FILE* oFile = fopen(bcg->oFileName,"w");
    if (oFile == NULL) {
      CPALAMEM_Abort("BCGInitializeOutput::Error: impossible to open the file %s!",bcg->oFileName);
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
            bcg->param.matrixFilename);
    const char* strOpt;
    strOpt = (bcg->precond_type == PREALPS_NOPREC) ? "No" : "Yes";
    fprintf(oFile,
            "Preconditioner            : %s\n",strOpt);
    strOpt = (bcg->ortho_alg == ORTHODIR) ? "Orthodir" : "Orthomin";
    fprintf(oFile,
            "Ortho algorithm           : %s\n",strOpt);
    strOpt = (bcg->bs_red == ALPHA_RANK) ? "Alpha-rank" : "No";
    fprintf(oFile,
            "Block size reduction      : %s\n",strOpt);
    fprintf(oFile,
            "tolerance                 : %e\n",
            bcg->param.tolerance);
    fprintf(oFile,
            "iteration maximum         : %d\n",
            bcg->param.iterMax);
    fprintf(oFile,
            "Number of METIS blocks    : %d\n",
            bcg->param.nbBlockPart);
    fprintf(oFile,
            "Number of right hand sides: %d\n",
            bcg->param.nbRHS);
    fprintf(oFile,"########################################################\n");
    fclose(oFile);
  }
END_TIME
POP
  return ierr;
}

int BCGInitialize(BCG_t* bcg) {
PUSH
BEGIN_TIME
  int rank, ierr = 0;
  int nCol = 0;
  // Simplify notations
  Mat_Dense_t* P        = bcg->P;
  Mat_Dense_t* R        = bcg->R;
  DVector_t*   b        = bcg->b;
  double*      ptrNormb = &(bcg->normb);

  MPI_Comm_rank(bcg->comm, &rank);
  // Initialize outptut
  //  ierr = BCGInitializeOutput(bcg);
  // Initialize iter
  bcg->iter = 0;
  // Compute normb
  ierr = DVector2NormSquared(b, ptrNormb);
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
  ierr = BCGSplit(b, R, nCol);CHKERR(ierr);

  // Then we construct P_0
  if (bcg->precond_type == PREALPS_NOPREC) {
    ierr = MatDenseCopy(R,P);CHKERR(ierr);
  }
  else if (bcg->precond_side == LEFT_PREC) {
    ierr = PrecondBlockOperator(bcg->precond_type, R, P);CHKERR(ierr);
  }

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
  // Malloc the working array
  bcg->work = (double*) malloc(bcg->P->info.nval*sizeof(double));
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

int BCGIterate(BCG_t* bcg) {
PUSH
BEGIN_TIME
OPEN_TIMER
  int rank = -1;
  int ierr = 0;
  int nrhs = bcg->param.nbRHS;
  // Simplify notations
  MPI_Comm     comm    = bcg->comm;
  Mat_Dense_t* P       = bcg->P;
  Mat_Dense_t* AP      = bcg->AP;
  Mat_Dense_t* AP_prev = bcg->AP_prev;
  Mat_Dense_t* P_prev  = bcg->P_prev;
  Mat_Dense_t* X       = bcg->X;
  Mat_Dense_t* R       = bcg->R;
  Mat_Dense_t* H       = bcg->H;
  Mat_Dense_t* AH      = bcg->AH;
  Mat_Dense_t* alpha   = bcg->alpha;
  Mat_Dense_t* beta    = bcg->beta;
  Mat_Dense_t* gamma   = bcg->gamma;
  Mat_Dense_t* Z       = bcg->Z;
  int M = P->info.M;
  int m = P->info.m;
  int t = P->info.n;

  MPI_Comm_rank(comm,&rank);

  if (bcg->bs_red == ALPHA_RANK) {
    if (bcg->ortho_alg == ORTHOMIN) {
      MatDenseSetInfo(P, M, nrhs, m, nrhs, COL_MAJOR);
      MatDenseSetInfo(AP, M, nrhs, m, nrhs, COL_MAJOR);
      MatDenseSetInfo(alpha, nrhs, nrhs, nrhs, nrhs, COL_MAJOR);
      MatDenseSetInfo(beta, nrhs, nrhs, nrhs, nrhs, COL_MAJOR);
    }
    else if (bcg->ortho_alg == ORTHODIR) {
      MatDenseSetInfo(beta, t, t, t, t, COL_MAJOR);
      MatDenseSetInfo(alpha, t, nrhs, t, nrhs, COL_MAJOR);
      MatDenseSetInfo(P, M, t, m, t, COL_MAJOR);
      MatDenseSetInfo(AP, M, t, m, t, COL_MAJOR);
    }
  }

TIC(step1,"A*P")
    ierr = BlockOperator(P,AP);
TAC(step1)

TIC(step2,"ACholQR")
  ierr = MatDenseACholQR(P,AP,beta,comm);
TAC(step2)

  MatDenseSetInfo(alpha,
                  P->info.N,
                  R->info.N,
                  P->info.n,
                  R->info.n,
                  COL_MAJOR);
TIC(step3,"alpha = P^t*R")
  ierr = MatDenseMatDotProd(P,R,alpha,comm);
TAC(step3)

  if (bcg->bs_red == ALPHA_RANK) {
TIC(step4,"Reduce P")
  ierr = BCGReduceSearchDirections(bcg);
TAC(step4)
  }
  t = P->info.n;
TIC(step5,"X = X + P*alpha")
  ierr = MatDenseKernelMatMult(P,'N',alpha,'N',X,1.0,1.0);
TAC(step5)

TIC(step6,"R = R - AP*alpha")
  ierr = MatDenseKernelMatMult(AP,'N',alpha,'N',R,-1.0,1.0);
TAC(step6)

  if (bcg->ortho_alg == ORTHODIR) {
    MatDenseSetInfo(Z,M,t,m,t,COL_MAJOR);
    if (bcg->precond_type == PREALPS_NOPREC) {
TIC(step7,"Z = AP")
      ierr = MatDenseCopy(AP,Z);
    }
    else if (bcg->precond_side == LEFT_PREC) {
TIC(step7,"Z = M^-1*AP")
      ierr = PrecondBlockOperator(bcg->precond_type, AP, Z);
    }
TAC(step7)
  }
  else if (bcg->ortho_alg == ORTHOMIN) {
    if (bcg->precond_type == PREALPS_NOPREC) {
TIC(step7,"Z = R")
      ierr = MatDenseCopy(R,Z);
    }
    else if (bcg->precond_side == LEFT_PREC) {
TIC(step7,"Z = M^-1*R")
      ierr = PrecondBlockOperator(bcg->precond_type, R, Z);
TAC(step7)
    }
  }
  MatDenseSetInfo(beta,t,
                  nrhs,
                  Z->info.n,
                  AP->info.n,
                  COL_MAJOR);
TIC(step8,"beta = (AP)^t*Z")
  ierr = MatDenseMatDotProd(AP,Z,beta,comm);
TAC(step8)

TIC(step9,"Z = Z - P*beta")
  ierr = MatDenseKernelMatMult(P,'N',beta,'N',Z,-1.0,1.0);
TAC(step9)

  if (bcg->ortho_alg == ORTHODIR) {
    MatDenseSetInfo(gamma,
                    AP_prev->info.N,
                    Z->info.N,
                    AP_prev->info.n,
                    Z->info.n,
                    COL_MAJOR);
TIC(step10,"gamma = (AP_prev)^t*Z")
    ierr = MatDenseMatDotProd(AP_prev,Z,gamma,comm);
TAC(step10)
TIC(step11,"Z = Z - P_prev*gamma")
    ierr = MatDenseKernelMatMult(P_prev,'N',gamma,'N',Z,-1.0,1.0);
TAC(step11)
    if (bcg->bs_red == ALPHA_RANK && t < nrhs) {
      MatDenseSetInfo(beta,
                      AH->info.N,
                      Z->info.N,
                      AH->info.n,
                      Z->info.n,
                      COL_MAJOR);
TIC(step12,"beta = (AH)^t*Z")
      ierr = MatDenseMatDotProd(AH,Z,beta,comm);
TAC(step12)
TIC(step13,"Z = Z - H*beta")
    ierr = MatDenseKernelMatMult(H,'N',beta,'N',Z,-1.0,1.0);
TAC(step13)
    }
  }

  // Swapping time
  MatDenseSwap(P,Z);
  if (bcg->ortho_alg == ORTHODIR) {
    MatDenseSwap(AP,AP_prev);
    MatDenseSwap(P_prev,Z);
  }

  bcg->iter++;
CLOSE_TIMER
END_TIME
POP
  return ierr;
}

int BCGReduceSearchDirections(BCG_t* bcg) {
PUSH
BEGIN_TIME
  int ierr = 0;
  // Simplify notations
  MPI_Comm     comm  = bcg->comm;
  Mat_Dense_t* P     = bcg->P;
  Mat_Dense_t* AP    = bcg->AP;
  Mat_Dense_t* H     = bcg->H;
  Mat_Dense_t* AH     = bcg->AH;
  Mat_Dense_t* alpha = bcg->alpha;
  double* work       = bcg->work;
  int t  = alpha->info.n;
  int bs = alpha->info.m;

  double tolDef = sqrt(CPALAMEM_EPSILON);
  //(1.0/sqrt(t))*(bcg->param.tolerance);
  //(1.0/sqrt(t))*(bcg->param.tolerance)*(bcg->normb);

  // TODO remove this
  double* tau = NULL;
  tau = (double*) calloc(t,sizeof(double));
  double* superb = NULL;
  superb = (double*) calloc((t-1),sizeof(double));
  // TODO

  memcpy(work,alpha->val,sizeof(double)*bs*t);

  // Low-rank approximation of alpha
  // # Truncated SVD
  ierr = LAPACKE_dgesvd( LAPACK_COL_MAJOR,
                         'O',
                         'N',
                         bs,
                         t,
                         work,
                         alpha->info.lda,
                         tau,
                         NULL,
                         1,
                         NULL,
                         1,
                         superb);ASSERT(ierr == 0);

  if (ierr > 0)
  {
   CPALAMEM_Abort("the eigensolver did not converge!");
  }
  else if (ierr < 0)
  {
   CPALAMEM_Abort("parameter %d has an illegal value!",-ierr+1);
  }

  // /TODO # RRQR
  int rank = 0;
  MPI_Comm_rank(comm,&rank);

  int cpt = 0;
  for(int i = 0; i < t; i++){
  		if (tau[i] > tolDef) {
  			cpt++;
  		}
      else break;
      if (rank == 0) {
  		    printf("Singular value[%d] = %f \n", i, tau[i]);
      }
  	}

//  if (cpt == 0) cpt = 1;

  if (rank == 0) {
    printf("Block size: %d/%d\n",cpt,t);
    printf("tolDef = %.10f\n",tolDef);
  }

  // Reduction of the search directions
  if (cpt > 0 && cpt < t && cpt != bs) {
    Mat_Dense_t U_s = MatDenseNULL();
    MatDenseSetInfo(&U_s,bs,t,bs,t,COL_MAJOR);
    U_s.val = work;
    // Use QR on U to update in-place alpha, P and AP
    // Update alpha, P, AP
    MatDenseDormqr(alpha,&U_s,tau,'L','T');
    MatDenseDormqr(P    ,&U_s,tau,'R','N');
    MatDenseDormqr(AP   ,&U_s,tau,'R','N');
    // if (rank == 0) {
    //   MatDensePrintfInfo("P",P);
    // MatDensePrintf2D("P",P);
    // }
    // Reduce sizes
    MatDenseSetInfo(alpha,cpt,t,cpt,t,COL_MAJOR);
    MatDenseSetInfo(P,P->info.M,cpt,P->info.m,cpt,COL_MAJOR);
    MatDenseSetInfo(AP,AP->info.M,cpt,AP->info.m,cpt,COL_MAJOR);

    if (bcg->ortho_alg == ORTHODIR) {
      // Update H and AH
      MatDenseSetInfo(H, H->info.M, t-cpt, H->info.m, t-cpt, COL_MAJOR);
      memcpy(H->val+(t-bs)*H->info.m,
        P->val+cpt*P->info.m,
        sizeof(double)*(bs-cpt)*H->info.m);
      MatDenseSetInfo(AH,AH->info.M, t-cpt, AH->info.m, t-cpt, COL_MAJOR);
      memcpy(AH->val+(t-bs)*H->info.m,
        AP->val+cpt*AP->info.m,
        sizeof(double)*(bs-cpt)*AH->info.m);
    }
  }

  // TODO remove this
  if (tau != NULL) free(tau);
  if (superb != NULL) free(superb);
  // TODO

END_TIME
POP
  return ierr;
}

int BCGFinalize(BCG_t* bcg) {
PUSH
BEGIN_TIME
  int ierr = 0;
  // Simplify notations
  Mat_Dense_t* X = bcg->X;
  double* work = bcg->work;

  DVector_t solution = DVectorNULL();
  solution.nval = X->info.m;
  solution.val  = work;
  // Get the solution
  ierr = MatDenseKernelSumColumns(X, &solution);CHKERR(ierr);

  // Safecheck
  Mat_Dense_t* AP = bcg->AP;
  DVector_t* b = bcg->b;
  MPI_Comm comm = bcg->comm;
  BlockOperator(X,AP);
  DVector_t res = DVectorNULL();
  MatDenseKernelSumColumns(AP,&res);
  for (int i = 0; i < res.nval; ++i)
    res.val[i] = b->val[i] - res.val[i];
  double norm = 0.0;
  ierr = DVector2NormSquared(&res, &norm);CHKERR(ierr);
  // Sum over all processes
  MPI_Allreduce(MPI_IN_PLACE,&norm,1,MPI_DOUBLE,MPI_SUM,comm);
  norm = sqrt(norm);
  int rank = -1;
  MPI_Comm_rank(comm,&rank);
  if (rank == 0) {
    printf("/!\\ ECG safecheck /!\\\n\ttrue res: %e\n",norm);
  }
  DVectorFree(&res);
  //end safecheck

END_TIME
POP
  return ierr;
}

int BCGDump(BCG_t* bcg) {
PUSH
BEGIN_TIME
  int ierr = 0, rank;
  MPI_Comm_rank(bcg->comm,&rank);
  if (rank == 0) {
    FILE* oFile = fopen(bcg->oFileName,"a");
    fprintf(oFile,"########################################################\n");
    if (bcg->iter < bcg->param.iterMax) {
      fprintf(oFile, "The method converged!\n");
      fprintf(oFile, "Number of iteration: %d\n",bcg->iter);
      fprintf(oFile, "Residual           : %.14e\n",bcg->res);
      fprintf(oFile, "Normalized residual: %.14e\n",bcg->res/bcg->normb);
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

int BCGSolve(BCG_t* bcg, DVector_t* rhs, Usr_Param_t* param, const char* name) {
PUSH
BEGIN_TIME
OPEN_TIMER
  int ierr = 0;
  int M, m, rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ierr = OperatorGetSizes(&M,&m);CHKERR(ierr);
  // Allocate memory
  ierr = BCGMalloc(bcg, M, m, param, name);CHKERR(ierr);
  // Set-up rhs
  bcg->b = rhs;
  // Initialize variables
  ierr = BCGInitialize(bcg);CHKERR(ierr);
  int stop = 0;
  // Main loop
  while (stop != 1) {
    ierr = BCGIterate(bcg);
    ierr = BCGStoppingCriterion(bcg,&stop);
  }
  // Retrieve solution
  BCGFinalize(bcg);
  // Release memory
  BCGFree(bcg);
CLOSE_TIMER
END_TIME
POP
  return ierr;
}

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
