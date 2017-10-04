#
#   Locate the root directory of this library
#

HOME_DIR = ${PWD}

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
LOADER       = mpicc
LOADOPTS     = -lgfortran
LD_FLAGS     =

USE_DEBUG=1
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
ADDITIONAL_LD_FLAGS =

#
#   REQUIRED : Select the solvers
#


#
# Sparse direct solver which support the computation of the schur complement computation.
# Which Sparse Direct Solver will be used to compute the schur complement
# You will have to set the path to the solver library below.
# Possible values : 0 for MKL_PARDISO, 1 for PARDIDO,  2 for MUMPS
#

SCHUR_COMPLEMENT_SOLVER_CHOICE = 2

#
# Which Sparse Direct Solver will be factorize the matrix
# You will have to set the path to the solver library below.
# Possible values : 0 for MKL_PARDISO, 1 for PARDIDO,  2 for MUMPS
#

SPARSE_SOLVER_CHOICE = 2

#
# Which Sparse Direct Solver will be used
# Possible values : PARDIDO, MKL_PARDISO,
#

#Do not activate PARDISO and MKL_PARDISO at the same time
#SPARSE_SOLVER = "MKL_PARDISO"


#
#   REQUIRED : Provide paths for the libraries
#

# Provide the path for cpalamem

  CPALAMEM_DIR         = ${HOME_DIR}/utils/cpalamem
  CPALAMEM_INC         = ${CPALAMEM_DIR}/include
  CPALAMEM_LIB         = ${CPALAMEM_DIR}/lib
  INCLUDE_FLAGS       += $(addprefix -I,$(CPALAMEM_INC))
  LD_FLAGS            += -L$(CPALAMEM_LIB) -lcpalamem_core
  ADDITIONAL_FLAGS    += -DMPIACTIVATE -DMETISACTIVATE -DCPALAMEM_BENCHMARK
  CPALAMEM_TAR        := $(shell ls -t $(HOME_DIR)/utils/CPALAMEM_*.tar.gz | head -1)

# Provide the path for METIS

  METIS_DIR            = ${METIS_ROOT}
  METIS_INC            = $(METIS_DIR)/include
  METIS_LIB            = $(METIS_DIR)/lib
  INCLUDE_FLAGS       += $(addprefix -I,$(METIS_INC))
  LD_FLAGS            += -L$(METIS_LIB) -lmetis
  ADDITIONAL_FLAGS    +=
  ADDITIONAL_LD_FLAGS += -lcpalamem_metis
# Provide the path for ParMetis

USE_PARMETIS=0
ifeq ($(USE_PARMETIS), 1)
  PARMETIS_DIR         = ${PARMETIS_ROOT}
  PARMETIS_INC         = $(PARMETIS_DIR)/include
  PARMETIS_LIB         = $(PARMETIS_DIR)/lib
  INCLUDE_FLAGS       += $(addprefix -I,$(PARMETIS_INC))
  LD_FLAGS            += -L$(PARMETIS_LIB) -Wl,-rpath,$(PARMETIS_DIR)/lib -lparmetis
  ADDITIONAL_FLAGS    += -DUSE_PARMETIS
endif

# Provide the path for PARPACK

USE_PARPACK=0
ifeq ($(USE_PARPACK), 1)
  PARPACK_DIR          = ${PARPACK_ROOT}
  PARPACK_LIB          = $(PARPACK_DIR)
  INCLUDE_FLAGS       +=
  LD_FLAGS            += -L$(PARPACK_LIB) -lparpack -larpack
  ADDITIONAL_FLAGS    += -DUSE_PARPACK
endif


USE_SOLVER_MKL_PARDISO=1
ifeq ($(USE_SOLVER_MKL_PARDISO), 1)

  # Pardiso from mkl will be used. Make sure to link with Intel MKL (as described later in this file)
  # Important: It should be at least MKL 11.2 update 1. Check version in you mkl.h file
  ADDITIONAL_FLAGS += -DUSE_MKL_PARDISO -DUSE_SOLVER_MKL_PARDISO

endif


USE_SOLVER_PARDISO=0
ifeq ($(USE_SOLVER_PARDISO), 1)

  # Pardiso will be used (not the one from Intel)
  PARDISO_DIR          = $(HOME_DIR)
  PARDISO_LIB          = $(PARDISO_DIR)/lib
  INCLUDE_FLAGS       +=
  LD_FLAGS            += -L$(PARDISO_LIB)  -Wl,-rpath,$(PARDISO_DIR)/lib -lpardiso500-MACOS-X86-64
  ADDITIONAL_FLAGS    += -DUSE_PARDISO -DUSE_SOLVER_PARDISO
  #-DPARDISO_SCHUR_COMPLEMENT_PATCH
endif


#
#   OPTIONAL :
#     To use one of the following library, set the variable USE_* to 1
#

USE_SOLVER_MUMPS=0

ifeq ($(USE_SOLVER_MUMPS), 1)
  MUMPS_DIR            = $(MUMPS_ROOT)
  MUMPS_INC            = $(MUMPS_DIR)/include
  MUMPS_LIB            = $(MUMPS_DIR)/lib
  INCLUDE_FLAGS       += $(addprefix -I,$(MUMPS_INC))
  LD_FLAGS            += -L$(MUMPS_LIB)  -Wl,-rpath,$(MUMPS_DIR)/lib -ldmumps -lmumps_common -lpord -qopenmp -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64
  ADDITIONAL_FLAGS    += -DUSE_SOLVER_MUMPS
endif


USE_MKL=1
ifeq ($(USE_MKL), 1)
  MKL_DIR              = ${MKL_ROOT}
  MKL_INC              = $(MKL_DIR)/include
  MKL_LIB              = $(MKL_DIR)/lib
  INCLUDE_FLAGS       += $(addprefix -I,$(MKL_INC))
  LD_FLAGS            += -L$(MKL_LIB) -Wl,-rpath,$(MKL_DIR)/lib -lmkl_core -lmkl_sequential -lmkl_intel_lp64 -lpthread -ldl
  # LD_FLAGS          += -L$(MKL_DIR)/lib/intel64 -Wl,-rpath,$(MKL_ROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lgomp -lm -ldl -lpthread -DMKL_LP64
  # LD_FLAGS          += -Wl,--no-as-needed -L$(MKL_DIR)/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lgomp -lm -ldl -lpthread -DMKL_LP64
  ADDITIONAL_FLAGS    += -DMKLACTIVATE -DUSE_MKL
  ADDITIONAL_LD_FLAGS += -lcpalamem_mkl
endif


USE_LAPACK=0
ifeq ($(USE_LAPACK), 1)
  LAPACK_DIR           = $(LAPACK_ROOT)
  LAPACK_INC           = $(LAPACK_DIR)/LAPACKE/include
  LAPACK_LIB           = $(LAPACK_DIR)/lib
  INCLUDE_FLAGS       += $(addprefix -I,$(LAPACK_INC))
  LIB_DIRECTIVES      += -DLAPACK
  LD_FLAGS            += -L$(LAPACK_LIB) -llapacke -llapack -lrefblas -lgfortran -lm
endif


USE_SUITESPARSE=0
ifeq ($(USE_SUITESPARSE), 1)
  SUITESPARSE_DIR      = ${SUITESPARSE_ROOT}
  SUITESPARSE_INC      = $(SUITESPARSE_DIR)/include
  SUITESPARSE_LIB      = $(SUITESPARSE_DIR)/lib
  INCLUDE_FLAGS       += $(addprefix -I,$(SUITESPARSE_INC))
  LD_FLAGS            += -L$(SUITESPARSE_LIB) -lspqr -lcholmod -lcolamd -lcamd -lsuitesparseconfig -lamd -lccolamd -lmetis
endif

USE_PETSC=0
ifeq ($(USE_PETSC), 1)
  PETSC_DIR            = ${PETSC_ROOT}
  PETSC_INC            = $(PETSC_DIR)/include
  PETSC_LIB            = $(PETSC_DIR)/lib
  INCLUDE_FLAGS       += $(addprefix -I,$(PETSC_INC))
  LD_FLAGS            += -L$(PETSC_LIB) -L$(PETSC_DIR)/${PETSC_ARCH}/lib -lpetsc
  ADDITIONAL_FLAGS    += -DPETSC
  ADDITIONAL_LD_FLAGS += -lcpalamem_petsc
endif

#
#   You may also want to add some extension of the library to get different versions such as _debug or _mt
#

PLAT        = release


#=====================================================================
# (You won't probably need to change anything from here until the end)
#=====================================================================

ADDITIONAL_FLAGS += -DSPARSE_SOLVER=$(SPARSE_SOLVER_CHOICE)
ADDITIONAL_FLAGS += -DSCHUR_COMPLEMENT_SOLVER=$(SCHUR_COMPLEMENT_SOLVER_CHOICE)

CFLAGS      +=  $(ADDITIONAL_FLAGS) $(LIB_DIRECTIVES)
LD_FLAGS    +=  $(ADDITIONAL_LD_FLAGS)

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
