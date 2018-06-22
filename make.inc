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
CPLMDIR   = utils/cplm_light
SRC       = src
EXAMPLES  = examples


#
#   Compilers
#

CC           = mpicc
CFLAGS       = -Wall -funroll-loops -std=c99 -fPIC
OPT          = -O3 -march=native -mtune=native
LOADER       = mpicc
LOADOPTS     = -lgfortran -lm
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
#                    -DBUILDING_MATRICES_DUMP -DCUSTOM_PARTITIONING -DMAT_CUSTOM_PARTITIONING

ADDITIONAL_FLAGS = -DAdd_
ADDITIONAL_LD_FLAGS =

#
#   REQUIRED : Select the solvers
#


#
# Sparse direct solvers which support the schur complement computation.
# Which Sparse Direct Solver will be used to compute the schur complement
# You will have to set the path to the selected solver in the file make.lib.inc.
# Important: If you use Pardiso from MKL here, make sure to link with at least MKL 11.2 update 1. Check the version in your mkl.h file.
# Possible values : 0 for MKL_PARDISO, 1 for PARDIDO,  2 for MUMPS
#

SCHUR_COMPLEMENT_SOLVER_CHOICE = 2

#
# Which Sparse Direct Solver will be factorize the matrix
# You will have to set the path to the solver library below.
# Possible values : 0 for MKL_PARDISO, 1 for PARDIDO,  2 for MUMPS
#

SPARSE_SOLVER_CHOICE = 0

#
#   You may also want to add some extension of the library to get different versions such as _debug or _mt
#

PLAT        = release


#=====================================================================
# (You won't probably need to change anything from here until the end)
#=====================================================================

# Include librairies path

SELF_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
include $(SELF_DIR)make.lib.inc

# Set additional compiler flags
ifeq ($(SPARSE_SOLVER_CHOICE), 0)
  # Pardiso from mkl will be used as solver. Intel MKL must be linked with preAlps.
  ADDITIONAL_FLAGS += -DUSE_SOLVER_MKL_PARDISO
else ifeq ($(SPARSE_SOLVER_CHOICE), 1)
  ADDITIONAL_FLAGS += -DUSE_SOLVER_PARDISO
else ifeq ($(SPARSE_SOLVER_CHOICE), 2)
  ADDITIONAL_FLAGS += -DUSE_SOLVER_MUMPS
endif

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
