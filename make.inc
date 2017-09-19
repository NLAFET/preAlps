#
#   Locate the root directory of this library
#

HOME_DIR = .

#
# Environment of the project
#

BINDIR    = bin
LIBDIR    = lib
INCDIR    = include
UTILS     = utils
SRC       = src
EXAMPLES  = examples


#
#   Compilers
#


CC           = mpicc
CFLAGS       = -Wall -funroll-loops -std=c99 -fPIC
OPT          = -O3 -march=native -mtune=native
LOADER       = mpif90
LOADOPTS     = -lgfortran
LD_FLAGS     =

USE_DEBUG=0
#possible values: -DDEBUG -DPRINT_MOD=<number>
ifeq ($(USE_DEBUG), 1)
  CFLAGS        += -g -std=c99 -DDEBUG
  OPT           +=
endif


#
#   Include dir
#

INCLUDE_FLAGS = -I./ -I../include -I../utils -I../src/preconditioners -I../src/solvers

#
#   Directives:
#   Possible values: -DARPACK_MAXIT=<number> -DEIG_DISPLAY_STATS -DMAT_CUSTOM_PARTITIONING_FILE=<file_prefix>
#                    -DMAT_CUSTOM_PARTITIONING_FILE : The matrice provide its own permutation vector. <file_prefix>.perm.txt and <file_prefix>.rowPos.txt must exist
#                    -DBUILDING_MATRICES_DUMP

ADDITIONAL_FLAGS = -DAdd_

#
#   REQUIRED :
#

# Provide the path for cpalamem

  CPALAMEM_DIR      = ${CPALAMEM_ROOT}
  CPALAMEM_INC      = -I$(CPALAMEM_DIR)/includes
  CPALAMEM_LIB      = -L$(CPALAMEM_DIR)/lib -lcpalamem
  INCLUDE_FLAGS    += $(CPALAMEM_INC)
  LD_FLAGS         += $(CPALAMEM_LIB)
  ADDITIONAL_FLAGS += -DMPIACTIVATE -DMETIS -DCPALAMEM_BENCHMARK

# Provide the path for METIS

  METIS_DIR         = ${METIS_ROOT}
  METIS_INC         = -I$(METIS_DIR)/include
  METIS_LIB         = -L$(METIS_DIR)/lib -lmetis
  INCLUDE_FLAGS    += $(METIS_INC)
  LD_FLAGS         += $(METIS_LIB)


# Provide the path for PARPACK
USE_PARPACK=0
ifeq ($(USE_PARPACK), 1)
  PARPACK_DIR       = ${PARPACK_ROOT}
  PARPACK_LIB       = -L$(PARPACK_DIR) -lparpack -larpack
  INCLUDE_FLAGS    +=
  LD_FLAGS         += $(PARPACK_LIB)
  ADDITIONAL_FLAGS += -DUSE_PARPACK
endif




#
# Which Sparse Direct Solver will be used to compute the schur complement
# You will have to set the path to the solver library below.
# Possible values : 0 for MKL_PARDISO, 1 for PARDIDO,  2 for MUMPS
#


ADDITIONAL_FLAGS += -DSCHUR_COMPLEMENT_SOLVER=2

#
# Which Sparse Direct Solver will be factorize the matrix
# You will have to set the path to the solver library below.
# Possible values : 0 for MKL_PARDISO, 1 for PARDIDO,  2 for MUMPS
#

ADDITIONAL_FLAGS += -DSPARSE_SOLVER=2

#
# Which Sparse Direct Solver will be used
# Possible values : PARDIDO, MKL_PARDISO,
#

#Do not activate PARDISO and MKL_PARDISO at the same time

SPARSE_SOLVER = "MKL_PARDISO"

USE_SOLVER_MKL_PARDISO=1

ifeq ($(USE_SOLVER_MKL_PARDISO), 1)

  # Pardiso from mkl will be used. Make sure to link with Intel MKL (as described later in this file)
  # Important: It should be at least MKL 11.2 update 1. Check version in you mkl.h file
  ADDITIONAL_FLAGS += -DUSE_MKL_PARDISO -DUSE_SOLVER_MKL_PARDISO

endif


USE_SOLVER_PARDISO=0

ifeq ($(USE_SOLVER_PARDISO), 1)

  # Pardiso will be used (not the one from Intel)
  PARDISO_DIR       = $(HOME_DIR)
  PARDISO_LIB       = -L$(PARDISO_DIR)/lib -Wl,-rpath,$(PARDISO_DIR)/lib -lpardiso500-MACOS-X86-64
  INCLUDE_FLAGS    += -DUSE_PARDISO -DUSE_SOLVER_PARDISO
#-DPARDISO_SCHUR_COMPLEMENT_PATCH
  LD_FLAGS         += $(PARDISO_LIB)
endif

#
#   OPTIONAL :
#     To use one of the following library, set the variable USE_* to 1
#

USE_SOLVER_MUMPS=0

ifeq ($(USE_SOLVER_MUMPS), 1)
  MUMPS_DIR         = $(HOME_DIR)
  MUMPS_LIB         = -L$(MUMPS_DIR)/lib -Wl,-rpath,$(MUMPS_DIR)/lib -ldmumps -lmumps_common -lpord -qopenmp -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64
  INCLUDE_FLAGS    += -I$(MUMPS_DIR)/include -DUSE_SOLVER_MUMPS
  LD_FLAGS         += $(MUMPS_LIB)
endif


USE_MKL=1

ifeq ($(USE_MKL), 1)
  MKL_ROOT          = /opt/intel/mkl
  MKL_INC           = -I$(MKL_ROOT)/include
  MKL_LIB           = -L$(MKL_ROOT)/lib -Wl,-rpath,${MKL_ROOT}/lib -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl
# MKL_LIB           = -L$(MKL_ROOT)/lib/intel64 -Wl,-rpath,$(MKL_ROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lgomp -lm -ldl -lpthread -DMKL_LP64
# MKL_LIB            = -Wl,--no-as-needed -L$(MKL_ROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lgomp -lm -ldl -lpthread -DMKL_LP64

  INCLUDE_FLAGS    += $(MKL_INC)
  LD_FLAGS         += $(MKL_LIB)
  ADDITIONAL_FLAGS += -DMKLACTIVATE
endif


USE_LAPACK=0

ifeq ($(USE_LAPACK), 1)
  LAPACK_DIR        =
  LAPACK_INC        = -I$(LAPACK_DIR)/LAPACKE/include
  LAPACK_LIB        = -L$(LAPACK_DIR)/lib -llapacke -llapack -lrefblas -lgfortran -lm
  LIB_DIRECTIVES   += -DLAPACK
endif


USE_SUITESPARSE=0

ifeq ($(USE_SUITESPARSE), 1)
  SUITESPARSE_DIR   = /Users/donfack/app/soft/
  SUITESPARSE_INC   = -I$(SUITESPARSE_DIR)/include
  SUITESPARSE_LIB   = -L$(SUITESPARSE_DIR)/lib -lspqr -lcholmod -lcolamd -lcamd -lsuitesparseconfig -lamd -lccolamd -lmetis
endif

USE_PETSC=1
ifeq ($(USE_PETSC), 1)
  #PETSC_DIR   = ${PETSC_DIR}
  PETSC_INC   = -I$(PETSC_DIR)/include
  PETSC_LIB   = -L$(PETSC_DIR)/lib -L${PETSC_DIR}/${PETSC_ARCH}/lib -lpetsc
  INCLUDE_FLAGS    += $(PETSC_INC)
  LD_FLAGS         += $(PETSC_LIB)
  ADDITIONAL_FLAGS += -DPETSC
endif

#
#   You may also want to add some extension of the library to get different versions such as _debug or _mt
#

PLAT        = release


#=====================================================================
# (You won't probably need to change anything from here until the end)
#=====================================================================

CFLAGS             +=  $(ADDITIONAL_FLAGS) $(LIB_DIRECTIVES)


#
#   The archiver and the flag(s) to use when building archive (library)
#   If your system has no ranlib, set RANLIB = echo.
#

ARCH        = ar
ARCHFLAGS   = cr
RANLIB      = ranlib

#
#   The library name can be changed here
#

LIBNAME     = preAlps_${PLAT}
