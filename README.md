# PreAlps
 
PreAlps is a library that implements communication avoiding solvers based on enlarged Krylov subspace methods and robust preconditioners. 

In its current state the library contains routines for solving symmetric positive definite (SPD) linear systems with enlarged Conjugate Gradient (ECG) in parallel. ECG is based on enriching the Krylov subspace used in classic methods that allows to reduce drastically the communication cost of the iterative solver (see [1]).

The next release will include LORASC, a robust algebraic preconditioner based on low rank approximation of the Schur complement (see [3]).

ECG can be used as an iterative solver and can be combined with block Jacobi, LORASC, or any other efficient preconditioner.  It is based on reverse communication such that it can be used also for matrix-free problems.  LORASC can be used in combination with ECG but also with any other Krylov solver.

# References
 
1. Laura Grigori, Sophie Moufawad, and Frederic Nataf. Enlarged Krylov subspace conjugate
gradient methods for reducing communication. SIAM J. Matrix Anal. Appl., 2016.

2. Laura Grigori and Olivier Tissot. Reducing the communication and computational costs of enlarged krylov subspaces conjugate gradient. In submission.

3. Laura Grigori , Frederic Nataf, Soleiman Yousef. Robust algebraic schur complement based on low rank
correction. Technical report, ALPINES-INRIA, Paris-Rocquencourt, 6 2014.

4. Laura Grigori , Frederic Nataf, Soleiman Yousef, Simplice Donfack, Remi Lacroix. Robust algebraic schur complement based on low rank
correction. In submission.

# Running instructions
 

1. Choose one option below and install the required librairies:

  - In order to use ECG Solver, install MKL and METIS. 
  
  - In order to use LORASC, install MKL, ParMETIS, PARPACK and MUMPS. 
  
  - For the full installation of preAlps, install MKL, METIS, ParMETIS, PARPACK and MUMPS.
  
2. Edit the make.inc file  at the top level of the root directory.

3. type 'make install_cpalamem' to install CPaLAMeM which is provided with preAlps.

4. Type 'make' to install the library.

5. To run the example program, just type
  
   5.1 for a test on a provided elasticity 3D matrix (see [1,3]): 
   
   mpirun -np 8 ./bin/test_ecgsolve matrix/elasticite3d_12x10x10_var.mtx
   
   5.2 for a general case: 
  
   mpirun -np <nb processors> ./bin/test_ecgsolve <matrix_file.mtx>
   
# Contact
  
For any question, please contact {simplice.donfack, olivier.tissot, laura.grigori}@inria.fr
