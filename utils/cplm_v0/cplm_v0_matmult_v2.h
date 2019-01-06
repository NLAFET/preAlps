/*
* The version of Matmult which takes into account a different communicator
* Authors : Sebastien Cayrols
*         : Olivier Tissot
*         : Hussam Al Daas
*         : Simplice Donfack
*/
#ifndef CPLM_VO_MATMULT_V2
#define CPLM_V0_MATMULT_V2

#include <cplm_matcsr.h>
#include <cplm_matdense.h>

// Convert colPos and shift colInd in order to extract more easily
// col panels of A during matmult: Fix of CPLM_MatCSRInitializeMatMult_v2()
int CPLM_MatCSRInitializeMatMult_v2(CPLM_Mat_CSR_t* A_io,
                            CPLM_IVector_t* pos_in,
                            CPLM_IVector_t* colPos_in,
                            CPLM_IVector_t* rowPtr_out,
                            CPLM_storage_type_t stor_type,
                            CPLM_IVector_t* dest,
                            int nrecv,
                            MPI_Comm comm);

// Convert back to get initial A_io: fix of  CPLM_MatCSRFinalizeMatMult()
int CPLM_MatCSRFinalizeMatMult_v2(CPLM_Mat_CSR_t* A_io,
                          CPLM_IVector_t* pos_in,
                          CPLM_IVector_t* colPos_in,
                          CPLM_storage_type_t stor_type,
                          CPLM_IVector_t* dest,
                          int nrecv,
                          MPI_Comm comm);

int CPLM_MatCSRMatMult_v2(CPLM_Mat_CSR_t   *A_in,
                  CPLM_Mat_Dense_t *RHS_in,
                  CPLM_IVector_t   *dest,
                  int          nrecv,
                  CPLM_Mat_Dense_t *C_out,
                  CPLM_IVector_t   *pos,
                  CPLM_IVector_t   *colPos,
                  CPLM_IVector_t   *rowPtr_io,
                  MPI_Comm     comm,
                  int          matMultVersion);
#endif
