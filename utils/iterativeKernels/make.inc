#
# Environment of the project
#

BINDIR	  = bin
LIBDIR	  = lib
INCDIR    = include
UTILS     = utils
SRC       = src
TEST      = test


#
#   Compilers
#


CC           = mpicc
CFLAGS	     = -Wall -funroll-loops -std=c99 -fPIC 
OPT          = -O3 -march=native -mtune=native
LOADER       = mpicc
LOADOPTS     = 

#USE_DEBUG=1

ifdef USE_DEBUG
CFLAGS        = -g -std=c99 -O0
OPT           = 
endif


#
#   Include dir
#

INCLUDE_FLAGS = -I./ -I../include -I../utils

#
#   Directives:
#   Possible values:
#     -DDEBUG               enable printing for the matrix for debugging purpose.
#     -DUSE_PARMETIS        use parmetis, the external_lib_flags must be set later to -lparmetis.
#     -DVERIFY              run MKL on 1 proc and check the result with respect to spMSV.
#     -DUSE_HPARTITIONING   use hypergraph partitioning tools such as PaToH.
#     -DSPMSV_STATS         display time per operations and communication volume for the spmsv function.

ADDITIONAL_FLAGS = -DVERIFY 


#   Remark, for the matrix-matrix product part : 
#       If you need to verify the numerical result or measure the scalability, set mkl


#
#   OPTIONAL : 
#     To use one of the following library, uncomment the USE_* line
#

# BLAS: Choose at least one between MKL and LAPACK

USE_MKL=1

ifdef USE_MKL
  MKL_ROOT          = /opt/intel/mkl
  MKL_INC           = -I${MKL_ROOT}/include
  MKL_LIB           = -L${MKL_ROOT}/lib -Wl,-rpath,${MKL_ROOT}/lib -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl
  LIB_DIRECTIVES   += -DMKL
endif

#USE_LAPACK=1

ifdef USE_LAPACK
  LAPACK_DIR        = 
  LAPACK_INC        = -I$(LAPACK_DIR)/LAPACKE/include
  LAPACK_LIB        = -L$(LAPACK_DIR)/lib -llapacke -llapack -lrefblas -lgfortran -lm
  LIB_DIRECTIVES   += -DLAPACK 
endif


USE_SUITESPARSE=1

ifdef USE_SUITESPARSE
  SUITESPARSE_DIR   = /Users/donfack/app/soft/
  SUITESPARSE_INC   = -I$(SUITESPARSE_DIR)/include
  SUITESPARSE_LIB   = -L$(SUITESPARSE_DIR)/lib -lspqr -lcholmod -lcolamd -lcamd -lsuitesparseconfig -lamd -lccolamd -lmetis
endif

# PARTITIONING TOOLS: Choose at least one between METIS, PARMETIS and PATOH

#USE_PATOH=1

ifdef PATOH
  PATOH_INC         =
  PATOH_LIB         =
endif


#USE_PARMETIS=1

ifdef USE_PARMETIS
  PARMETIS_INC      =
  PARMETIS_LIB      =
endif

#preAlps can use METIS from SuiteSparse, if your version of SuiteSparse does not provide METIS then set it here.
#USE_METIS=1

ifdef USE_METIS
  METIS_DIR         = 
  METIS_INC         = -I$(METIS_DIR)/include
  METIS_LIB         = -L$(METIS_DIR)/lib -lmetis
endif


#=====================================================================
# (You won't probably need to change anything from here until the end) 
#=====================================================================

PARTITIONING_INC    = $(METIS_INC) $(PARMETIS_INC) $(PATOH_INC)
PARTITIONING_LIB    = $(METIS_LIB) $(PARMETIS_LIB) $(PATOH_LIB)


INCLUDE_FLAGS   +=  $(MKL_INC) $(PARTITIONING_INC) $(LAPACK_INC) $(SUITESPARSE_INC)
LINKER_LD       +=  $(MKL_LIB) $(PARTITIONING_LIB) $(LAPACK_LIB) $(SUITESPARSE_LIB)
CFLAGS          +=  $(ADDITIONAL_FLAGS) $(LIB_DIRECTIVES)


#
#   The archiver and the flag(s) to use when building archive (library)
#   If your system has no ranlib, set RANLIB = echo.
#

ARCH        = ar
ARCHFLAGS   = cr
RANLIB      = ranlib


#
#   You may also want to add some extension of the library to get different versions such as _debug or _mt
#

PLAT        = release


#
#   The library name can be changed here
#

LIBNAME     = preAlps_${PLAT}

