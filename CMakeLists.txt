################################################################################
# Author: Olivier Tissot                                                       #
# Date: 2018/01/30                                                             #
# Description: CMake configuration file                                        #
################################################################################

################################################################################
#                                  SETUP                                      #
################################################################################

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
# Change C compiler because we use MPI
SET(CMAKE_C_COMPILER mpicc)
SET(CMAKE_CXX_COMPILER mpicxx)

################################################################################
#                                 PROJECT                                      #
################################################################################

# Name of the project
PROJECT(PREALPS)
# Version number
SET(PREALPS_VERSION_MAJOR 0)
SET(PREALPS_VERSION_MINOR 1)
# Put the executable in the bin directory
SET(EXECUTABLE_OUTPUT_PATH ${PREALPS_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE})
SET(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -m64 -Wall -O3 -std=c99")
# Default install is the source directory
SET(CMAKE_INSTALL_PREFIX "${PREALPS_SOURCE_DIR}")
# Command line option to run tests
OPTION(TEST "Enable tests" OFF)

################################################################################
#                               THIRDPARTIES                                   #
################################################################################

# Link MPI
INCLUDE_DIRECTORIES($ENV{MPI_ROOT}/include)
LINK_DIRECTORIES($ENV{MPI_ROOT}/lib)
SET(MPI_LIBRARIES "mpi")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DMPIACTIVATE")

# Link CPaLAMeM
INCLUDE_DIRECTORIES(${PREALPS_SOURCE_DIR}/utils/cpalamem/include)
LINK_DIRECTORIES(${PREALPS_SOURCE_DIR}/utils/cpalamem/lib)
SET(CPALAMEM_LIBRARIES "cpalamem_core")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DCPALAMEM_BENCHMARK")

# Link Metis: one can specify METIS_ROOT during cmake call
INCLUDE_DIRECTORIES($ENV{METIS_ROOT}/include)
LINK_DIRECTORIES($ENV{METIS_ROOT}/lib)
SET(METIS_LIBRARIES "metis" "cpalamem_metis")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DMETIS -DMETISACTIVATE")

# Link MKL
INCLUDE_DIRECTORIES($ENV{MKL_ROOT}/include)
LINK_DIRECTORIES($ENV{MKL_ROOT}/lib)
IF (APPLE)
  SET(MKL_LIBRARIES "mkl_core" "mkl_sequential" "mkl_intel_lp64" "cpalamem_mkl")
ELSE()
  SET(MKL_LIBRARIES "mkl_core" "mkl_intel_thread" "mkl_intel_lp64" "iomp5" "pthread" "m" "dl" "cpalamem_mkl")
ENDIF()
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DMKLACTIVATE -DUSE_MKL -DUSE_MKL_PARDISO -DUSE_SOLVER_MKL_PARDISO -DSPARSE_SOLVER=0")

# preAlps options (...)
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSCHUR_COMPLEMENT_SOLVER=2")

# No linking with PETSC for the moment
MESSAGE("/!\\ PETSC cannot be used when compiling with CMake (for the moment) /!\\")

MESSAGE("Compiler(s) details: ")
MESSAGE("   C Compiler      : ${CMAKE_C_COMPILER}")
MESSAGE("   C Flags         : ${CMAKE_C_FLAGS}")
MESSAGE("End of Compiler(s) details\n")

################################################################################
#                                CORE FILES                                   #
################################################################################

# Include header directory
INCLUDE_DIRECTORIES(${PREALPS_SOURCE_DIR}/src/solvers)
INCLUDE_DIRECTORIES(${PREALPS_SOURCE_DIR}/src/preconditioners)
INCLUDE_DIRECTORIES(${PREALPS_SOURCE_DIR}/utils)
# Set source files
SET(PREALPS_CORE ${PREALPS_SOURCE_DIR}/src/solvers/ecg.c
${PREALPS_SOURCE_DIR}/src/preconditioners/block_jacobi.c
${PREALPS_SOURCE_DIR}/src/preconditioners/lorasc.c
${PREALPS_SOURCE_DIR}/src/preconditioners/lorasc_eigsolve.c
${PREALPS_SOURCE_DIR}/src/preconditioners/preAlps_preconditioner.c
${PREALPS_SOURCE_DIR}/src/preconditioners/lorasc.c
${PREALPS_SOURCE_DIR}/src/preconditioners/presc.c
${PREALPS_SOURCE_DIR}/src/preconditioners/presc_eigsolve.c)
SET(PREALPS_UTILS ${PREALPS_SOURCE_DIR}/utils/eigsolver.c
${PREALPS_SOURCE_DIR}/utils/matrixVectorOp.c
${PREALPS_SOURCE_DIR}/utils/mkl_pardiso_solver.c
${PREALPS_SOURCE_DIR}/utils/mumps_solver.c
${PREALPS_SOURCE_DIR}/utils/operator.c
${PREALPS_SOURCE_DIR}/utils/pardiso_solver.c
${PREALPS_SOURCE_DIR}/utils/preAlps_cplm_matcsr.c
${PREALPS_SOURCE_DIR}/utils/preAlps_doublevector.c
${PREALPS_SOURCE_DIR}/utils/preAlps_intvector.c
${PREALPS_SOURCE_DIR}/utils/preAlps_solver.c
${PREALPS_SOURCE_DIR}/utils/preAlps_utils.c
${PREALPS_SOURCE_DIR}/utils/solverStats.c)

SET(PREALPS_SOURCES ${PREALPS_CORE} ${PREALPS_UTILS})

################################################################################
#                                  TARGETS                                     #
################################################################################

# Build the static library
SET(LIBS_LIBRARY_TYPE STATIC)
ADD_LIBRARY(preAlps ${LIBS_LIBRARY_TYPE} ${PREALPS_SOURCES})
# Link it to its dependencies
TARGET_LINK_LIBRARIES(preAlps ${MPI_LIBRARIES} ${CPALAMEM_LIBRARIES} ${METIS_LIBRARIES} ${MKL_LIBRARIES})

# Install the library
INSTALL(TARGETS preAlps DESTINATION lib)
INSTALL(FILES ${PREALPS_LIB_HEADERS} DESTINATION include)
INSTALL(FILES ${METIS_HEADERS} DESTINATION include)

################################################################################
#                                   TESTS                                      #
################################################################################

# Build the examples
ADD_EXECUTABLE(ecg_prealps_op ${PREALPS_SOURCE_DIR}/examples/ecg_prealps_op.c)
TARGET_LINK_LIBRARIES(ecg_prealps_op preAlps)
ADD_EXECUTABLE(ecg_petsc_op ${PREALPS_SOURCE_DIR}/examples/ecg_petsc_op.c)
TARGET_LINK_LIBRARIES(ecg_petsc_op preAlps)
ADD_EXECUTABLE(ecg_bench_petsc_pcg ${PREALPS_SOURCE_DIR}/examples/ecg_bench_petsc_pcg.c)
TARGET_LINK_LIBRARIES(ecg_bench_petsc_pcg preAlps)

# OT: A little dirty because some tests don't work without MPI
# OT: Sebastien we should fix this one day :)
# ENABLE_TESTING()
# ADD_TEST(t_dense_to_sparse ${PREALPS_SOURCE_DIR}/bin/denseToSparse)
# ADD_TEST(t_get_col_block_position mpirun -n 2 ${PREALPS_SOURCE_DIR}/bin/getColBlockPosition)
# ADD_TEST(t_load_matrix ${PREALPS_SOURCE_DIR}/bin/loadMatrix)
# ADD_TEST(t_load_matrix_kway mpirun -n 2 ${PREALPS_SOURCE_DIR}/bin/loadMatrixUsingKway)
# ADD_TEST(t_mat_mult ${PREALPS_SOURCE_DIR}/bin/matMult)
