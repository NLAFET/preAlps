# Computational Kernels for Preconditioned Iterative Methods

NOTE: the code in this directory is not used by ECG solver, it is provided as reference for D4.1.

This library provides computational kernels for preconditioned iterative methods. It includes routines such as a sparse matrix matrix product, and sparse communication avoiding low rank approximation using tournament pivoting. 
The library uses sequential sparse QR factorization and METIS from SuiteSparse.

Running instructions:

1. Download and install SuiteSparse from http://faculty.cse.tamu.edu/davis/suitesparse.html

2. Edit the make.inc file  at the top level of the root directory. 
 
3. Type 'make' at the top level of the root directory.

4. To run the test program, just type

   4.1 for the sparse matrix vector product:

     mpirun -np <nb processors> ./bin/test_spMSV -m <matrix_file.mtx>

   4.2 for the tournament pivoting QR factorization:
     mpirun -np <nb processors> ./bin/test_prototypeQR -m <matrix_file.mtx> -k <rank_of_approximation>

   4.3 for the tournament pivoting CUR factorization:
     mpirun -np <nb processors> ./bin/test_prototypeCUR -m <matrix_file.mtx> -k <rank_of_approximation>

 
For any question, please contact {simplice.donfack, alan.ayala-obregon, laura.grigori}@inria.fr 

