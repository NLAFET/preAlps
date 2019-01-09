
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <cplm_utils.h>
#include <cplm_timing.h>
#include <cplm_v0_ivector.h>
#include <cplm_matcsr.h>
#include <cplm_matdense.h>
#include <cplm_kernels.h>

// Convert colPos and shift colInd in order to extract more easily
// col panels of A during matmult: Fix of CPLM_MatCSRInitializeMatMult_v2()
int CPLM_MatCSRInitializeMatMult_v2(CPLM_Mat_CSR_t* A_io,
                            CPLM_IVector_t* pos_in,
                            CPLM_IVector_t* colPos_in,
                            CPLM_IVector_t* rowPtr_out,
                            CPLM_storage_type_t stor_type,
                            CPLM_IVector_t* dest,
                            int nrecv,
                            MPI_Comm comm) {
  CPLM_PUSH

  int ierr       = 0;
  int rank       = -1;
  int size       = -1;
  int cpt        = 0;
  int isColMajor = (stor_type == COL_MAJOR);
  CPLM_IVector_t symColInd = CPLM_IVectorNULL();
  CPLM_IVector_t rowPtrB   = CPLM_IVectorNULL();
  CPLM_IVector_t rowPtrE   = CPLM_IVectorNULL();
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  CPLM_ASSERT(rowPtr_out->val != NULL);
  // Convert colPos in order to have begin and end of col blocks contiguous in
  // memory
  ierr = CPLM_IVectorMalloc(&rowPtrB, (nrecv+1)*A_io->info.m);CPLM_CHKERR(ierr);
  ierr = CPLM_IVectorMalloc(&rowPtrE, (nrecv+1)*A_io->info.m);CPLM_CHKERR(ierr);
  for (int j = 0; j < nrecv; ++j) {
    for (int i = 0; i < A_io->info.m; ++i)
    {
      rowPtrB.val[cpt]   = colPos_in->val[i*size + dest->val[j]];
      rowPtrE.val[cpt++] = colPos_in->val[i*size + dest->val[j] + 1];
    }
  }
  for (int i = 0; i < A_io->info.m; ++i)
  {
    rowPtrB.val[cpt]   = colPos_in->val[i*size + rank];
    rowPtrE.val[cpt++] = colPos_in->val[i*size + rank + 1];
  }
  // Shift colInd
  ierr = CPLM_IVectorCreateFromPtr(&symColInd, A_io->info.lnnz, A_io->colInd);
  // First shift for proc from which we receive
  for (int k = 0; k < A_io->info.m; ++k)
    for (int i = 0; i < nrecv; ++i)
    for (int j = colPos_in->val[k*size + dest->val[i]]; j < colPos_in->val[k*size + dest->val[i]+1]; ++j)
       symColInd.val[j] += isColMajor - pos_in->val[dest->val[i]];
  // Then shift ourself
  for (int k = 0; k < A_io->info.m; ++k)
   for (int j = colPos_in->val[k*size + rank]; j < colPos_in->val[k*size + rank+1]; ++j)
     symColInd.val[j] += isColMajor - pos_in->val[rank];
 // Copy in rowPtr_out
 // TODO: remove this part
  memcpy(rowPtr_out->val               , rowPtrB.val, rowPtrB.nval*sizeof(int));
  memcpy(rowPtr_out->val + rowPtrB.nval, rowPtrE.val, rowPtrE.nval*sizeof(int));
  CPLM_IVectorFree(&rowPtrB);
  CPLM_IVectorFree(&rowPtrE);

  CPLM_POP
  return ierr;
}

// Convert back to get initial A_io: fix of  CPLM_MatCSRFinalizeMatMult()
int CPLM_MatCSRFinalizeMatMult_v2(CPLM_Mat_CSR_t* A_io,
                          CPLM_IVector_t* pos_in,
                          CPLM_IVector_t* colPos_in,
                          CPLM_storage_type_t stor_type,
                          CPLM_IVector_t* dest,
                          int nrecv,
                          MPI_Comm comm) {
  CPLM_PUSH

  int ierr       = 0;
  int rank       = -1;
  int size       = -1;
  int isColMajor = (stor_type == COL_MAJOR);
  CPLM_IVector_t symColInd = CPLM_IVectorNULL();
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  // Shift colInd
  ierr = CPLM_IVectorCreateFromPtr(&symColInd, A_io->info.lnnz, A_io->colInd);
  // First shift for proc from which we receive
  for (int k = 0; k < A_io->info.m; ++k)
    for (int i = 0; i < nrecv; ++i)
      for (int j = colPos_in->val[k*size + dest->val[i]]; j < colPos_in->val[k*size + dest->val[i]+1]; ++j)
        symColInd.val[j] -= isColMajor - pos_in->val[dest->val[i]];
  // Then shift ourself
  for (int k = 0; k < A_io->info.m; ++k)
    for (int j = colPos_in->val[k*size + rank]; j < colPos_in->val[k*size + rank+1]; ++j)
      symColInd.val[j] -= isColMajor - pos_in->val[rank];

CPLM_POP
  return ierr;
}

/* Fix of CPLM_MatCSRMatMult() */

int CPLM_MatCSRMatMult_v2(CPLM_Mat_CSR_t   *A_in,
                  CPLM_Mat_Dense_t *RHS_in,
                  CPLM_IVector_t   *dest,
                  int          nrecv,
                  CPLM_Mat_Dense_t *C_out,
                  CPLM_IVector_t   *pos,
                  CPLM_IVector_t   *colPos,
                  CPLM_IVector_t   *rowPtr_io,
                  MPI_Comm     comm,
                  int          matMultVersion)
{
CPLM_PUSH
CPLM_BEGIN_TIME
CPLM_OPEN_TIMER
  CPLM_Mat_Dense_t workRecv = CPLM_MatDenseNULL();
  CPLM_Mat_Dense_t workMult = CPLM_MatDenseNULL();
  int ierr       = 0;
  int srcIndest  = 0;
  int rank       = -1;
  int size       = 0;
  int src        = -1;
  int nrecvCurr  = 0;
  int tag        = 0;
  int flag       = 0;
  int count      = 0;
  int maxRow     = -1;
  int flagRowPtr = 0;
  CPLM_IVector_t offset   = CPLM_IVectorNULL();
  CPLM_IVector_t localPos = CPLM_IVectorNULL();
  CPLM_Info_Dense_t saveWorkInfo;
  // Symbolic variables (do not free)
  int *rowPtrB = NULL;
  int *rowPtrE = NULL;
  double* swapVal = NULL;
  // MPI variables
  MPI_Request *request_send = NULL;
  MPI_Request *tmp_s        = NULL; // Symbolic do not free
  MPI_Status *status_send   = NULL;
  MPI_Request request_recv;
  MPI_Status status_recv;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // Allocate workRecv
  for (int i = 0; i < pos->nval-1; ++i)
    maxRow = CPLM_MAX(maxRow, pos->val[i+1] - pos->val[i]);
  CPLM_MatDenseSetInfo(&workRecv, RHS_in->info.M, RHS_in->info.N, maxRow, RHS_in->info.n, RHS_in->info.stor_type);
  workMult.info = workRecv.info;
  CPLM_MatDenseMalloc(&workRecv);
  CPLM_MatDenseMalloc(&workMult);

  //Allocate C_out
  if(C_out->val == NULL)
  {
    ierr = CPLM_MatDenseSetInfo(C_out,
                           A_in->info.M,
                           RHS_in->info.N,
                           A_in->info.m,
                           RHS_in->info.n,
                           RHS_in->info.stor_type);
    CPLM_CHKERR(ierr);
    ierr = CPLM_MatDenseCalloc(C_out);CPLM_CHKERR(ierr);
  }

  request_send = (MPI_Request*) malloc(dest->nval*sizeof(MPI_Request));
  status_send  = (MPI_Status*)  malloc(dest->nval*sizeof(MPI_Status));

  CPLM_ASSERT(request_send != NULL);
  CPLM_ASSERT(status_send  != NULL);
  CPLM_ASSERT(pos->val     != NULL);
  CPLM_ASSERT(colPos->val  != NULL);
  CPLM_ASSERT(dest->val    != NULL);

CPLM_TIC(step1, "Send")
  //Send data
  for(int i=0;i<dest->nval;i++)
  {
    //CPLM_debug("Init ISendData to %d\n",dest->val[i]);
    // Dirty but it works...
    tmp_s = request_send + i;
    ierr = CPLM_MatDenseISendData(RHS_in,dest->val[i],tag,comm,&tmp_s);
    CPLM_checkMPIERR(ierr,"Send partial RHS");
  }
CPLM_TAC(step1)

  // Prepare to MatMult
  if (rowPtr_io == NULL)
  {
    rowPtr_io = (CPLM_IVector_t*) malloc(sizeof(CPLM_IVector_t));
    rowPtr_io->val  = NULL;
    rowPtr_io->nval = 0;
    flagRowPtr = 1;
  }
  if (rowPtr_io->val == NULL)
  {
    CPLM_IVectorMalloc(rowPtr_io, 2*(nrecv+1)*A_in->info.m);
    CPLM_MatCSRInitializeMatMult_v2(A_in,
                            pos,
                            colPos,
                            rowPtr_io,
                            RHS_in->info.stor_type,
                            dest,
                            nrecv, comm);
  }

CPLM_TIC(step2, "Diag SpMM")
  // Compute Adiag*RHS_in
  if (matMultVersion % 2 == 0) {
    // Use symbolic variables
    rowPtrB = rowPtr_io->val + nrecv * A_in->info.m;
    rowPtrE = rowPtr_io->val + (2*nrecv+1)*A_in->info.m;
    ierr = CPLM_MatCSRKernelGenMatDenseMult(A_in->val    + rowPtrB[0],
                                       A_in->colInd + rowPtrB[0],
                                       rowPtrB,
                                       rowPtrE,
                                       A_in->info.m,
                                       A_in->info.m,
                                       RHS_in,
                                       C_out,
                                       1.0,
                                       0.0);CPLM_CHKERR(ierr);
  }
CPLM_TAC(step2)

CPLM_TIC(step3, "Recv and SpMM")
  // Irecv version
  if (matMultVersion / 2 > 0)
  {
    saveWorkInfo = workRecv.info;
    ierr = CPLM_MatDenseIRecvData(&workRecv,MPI_ANY_SOURCE,tag,comm,&request_recv);
    for (int i = 0; i < nrecv; ++i)
    {
      MPI_Wait(&request_recv,&status_recv);
      swapVal = workMult.val;
      workMult.val = workRecv.val;
      workRecv.val = swapVal;
      workRecv.info = saveWorkInfo;
      if (i+1 < nrecv)
        ierr = CPLM_MatDenseIRecvData(&workRecv,MPI_ANY_SOURCE,tag,comm,&request_recv);
      src = status_recv.MPI_SOURCE;
      //CPLM_debug("Recv communication from %d\ttag %d\n",src,tag);

      MPI_Get_count(&status_recv, MPI_DOUBLE, &count);
      ierr = CPLM_IVectorGetPos(dest,src,&srcIndest);CPLM_CHKERR(ierr);

      // Set up workRecv infos
      ierr = CPLM_MatDenseSetInfo(&workMult,
                             RHS_in->info.M,
                             RHS_in->info.N,
                             count / RHS_in->info.n,
                             RHS_in->info.n,
                             RHS_in->info.stor_type);
      // Symbolic variables
      rowPtrB = rowPtr_io->val + srcIndest * A_in->info.m;
      rowPtrE = rowPtr_io->val + (nrecv+1+srcIndest)*A_in->info.m;
      ierr = CPLM_MatCSRKernelGenMatDenseMult(A_in->val    + rowPtrB[0],
                                         A_in->colInd + rowPtrB[0],
                                         rowPtrB,
                                         rowPtrE,
                                         A_in->info.m,
                                         workMult.info.m,
                                         &workMult,
                                         C_out,
                                         1.0,
                                         1.0);CPLM_CHKERR(ierr);
    }
    MPI_Waitall(dest->nval, request_send, status_send);
  }
  // Iprobe version
  else
  {
    while( nrecvCurr < nrecv )
    {
      MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &flag, &status_recv);

      if(flag)
      {
        tag = status_recv.MPI_TAG;
        src = status_recv.MPI_SOURCE;

        ierr = CPLM_IVectorGetPos(dest,src,&srcIndest);CPLM_CHKERR(ierr);

        //CPLM_debug("Recv communication from %d\ttag %d\n",src,tag);

        MPI_Get_count(&status_recv, MPI_DOUBLE, &count);

        // Set up workRecv infos
        ierr = CPLM_MatDenseSetInfo(&workRecv,
                               RHS_in->info.M,
                               RHS_in->info.N,
                               count / RHS_in->info.n,
                               RHS_in->info.n,
                               RHS_in->info.stor_type);
        ierr = CPLM_MatDenseRecvData(&workRecv, src, tag, comm);
        // Symbolic variables
        rowPtrB = rowPtr_io->val + srcIndest * A_in->info.m;
        rowPtrE = rowPtr_io->val + (nrecv+1+srcIndest)*A_in->info.m;
        ierr = CPLM_MatCSRKernelGenMatDenseMult(A_in->val    + rowPtrB[0],
                                           A_in->colInd + rowPtrB[0],
                                           rowPtrB,
                                           rowPtrE,
                                           A_in->info.m,
                                           workRecv.info.m,
                                           &workRecv,
                                           C_out,
                                           1.0,
                                           1.0);CPLM_CHKERR(ierr);

        nrecvCurr++;
      }
    }
    MPI_Waitall(dest->nval, request_send, status_send);
  }

CPLM_TAC(step3)


  if (flagRowPtr == 1)
  {
    CPLM_MatCSRFinalizeMatMult_v2(A_in,pos,colPos,RHS_in->info.stor_type,dest,nrecv, comm);
    CPLM_IVectorFree(rowPtr_io);
    if (rowPtr_io) free(rowPtr_io);
  }
  if (request_send != NULL) free(request_send);
  if (status_send  != NULL) free(status_send);
  CPLM_MatDenseFree(&workRecv);
  CPLM_MatDenseFree(&workMult);
  CPLM_IVectorFree(&localPos);
  CPLM_IVectorFree(&offset);
CPLM_CLOSE_TIMER
CPLM_END_TIME
CPLM_POP
  return ierr;

}
