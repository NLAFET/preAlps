# PreAlps

PreAlps is a library that implements communication avoiding solvers based on enlarged Krylov subspace methods and robust preconditioners.

In its current state the library contains routines for solving symmetric positive definite (SPD) linear systems. The current version provides an efficient implementation of ECG and LORASC preconditioner. ECG is based on enriching the Krylov subspace used in classic methods that allows to reduce drastically the communication cost of the iterative solver (see [1]). LORASC is a robust algebraic preconditioner based on low rank approximation of the Schur complement (see [3]).

ECG can be used as an iterative solver and can be combined with block Jacobi, LORASC, or any other efficient preconditioner.  It is based on reverse communication such that it can be used also for matrix-free problems. LORASC can be used in combination with ECG but also with any other Krylov solver.



# Running instructions

  1. Make sure all these libraries are installed on your system or install them:  
    1.1 MPI  
    1.2 MKL  
    1.3 METIS (can be downloaded from http://glaros.dtc.umn.edu/gkhome/metis/metis/download)  

  2. If you want to enable LORASC preconditioner in preAlps, make sure all these libraries are installed or install them:  
    2.1 ParMETIS (can be downloaded from http://glaros.dtc.umn.edu/gkhome/metis/parmetis/download )  
    2.2 MUMPS (can be downloaded from http://mumps.enseeiht.fr/index.php?page=dwnld)  
    2.3 PARPACK (can be downloaded from http://www.caam.rice.edu/software/ARPACK/download.html)  

  3. If you will like to compare ECG results with PETSc, make sure PETSc is installed.  
    3.1 PETSc ( can be downloaded from https://www.mcs.anl.gov/petsc/download/index.html )  

  4. Get the latest version of preAlps.  
    ```
    $ git clone git@github.com:NLAFET/preAlps.git preAlps  
    ```
  5. Edit the make.inc file at the top level of the root directory, check compiler directives and flags.  

  6. Copy an example of make.lib.inc from the directory MAKES.  

    In order to use ECG Solver only, type:  
    ```
    $ cp MAKES/make.lib.inc-ecg make.lib.inc
    ```
    For the full installation of preAlps, type:
    ```
    $ cp MAKES/make.lib.inc make.lib.inc  
    ```
  7. Edit the make.lib.inc file to enable the libraries used and installed in 1,2 and 3. Make sure the LD_FLAGS of these libraries are correctly set. Disable unused libraries.  


  8. Type 'make' to compile the library.  
    ```
    $ make
    ```
  9. To run the example program  

    9.1 for a test on a provided elasticity 3D matrix (see [1,3]):  

      9.1.1 run ECG + Block Jacobi with 8 processors with an enlarging factor of 4.  
      ```
      $ mpirun -np 8 ./bin/test_ecg_prealps_op -m matrix/elasticity3d_12x10x10_var.mtx -o 0 -r 0 -e 4  
      ```
      9.1.1 run ECG + Lorasc Multilevel with 8 processors with enlarging factor of 2, use 4 domains at the first level of the parallelism for LORASC.  
      ```
      $ mpirun -np 8 bin/test_lorasc -m matrix/elasticity3d_12x10x10_var.mtx  -t 2 -p 2 -npLevel1 4  
      ```
    9.2 for obtaining the help about all the options provided with the test programs.
      ```
      $ ./bin/test_ecg_prealps_op -h  
      $ ./bin/test_lorasc -h  
      ```
    9.3 for a general case:  
      ```
      $ mpirun -np <nb_processors> mpirun ./test_ecg_prealps_op -e/--enlarging-factor <int> [-h/--help] [-i/--iteration-maximum <int>] -m/--matrix <matrix_file.mtx> -o/--ortho-alg <int> -r/--search-dir-red <int> [-t/--tolerance <double>]  

      $ mpirun -np <nb_processors> ./bin/test_lorasc -m <matrix_file.mtx> -t <enlarging factor> -p <preconditionner_number> -npLevel1 <number_domains_first_level>  
      ```

# License

  PreAlps is free software licensed under the [BSD-3 License](https://opensource.org/licenses/BSD-3-Clause).

  The preAlps software contains proprietary of Inria.  

  Version V1.0, August 2018  
  Authors: Simplice Donfack, Olivier Tissot, Laura Grigori, Sebastien Cayrols, Alan Ayala Obregon
  Copyright (C) 2018, Inria

# References

  1. Laura Grigori, Sophie Moufawad, and Frederic Nataf. Enlarged Krylov subspace conjugate gradient methods for reducing communication. SIAM J. Matrix Anal. Appl., 2016.

  2. Laura Grigori and Olivier Tissot. Reducing the communication and computational costs of enlarged krylov subspaces conjugate gradient. In submission.

  3. Laura Grigori , Frederic Nataf, Soleiman Yousef. Robust algebraic schur complement based on low rank correction. Technical report, ALPINES-INRIA, Paris-Rocquencourt, 6 2014.

  4. Laura Grigori , Frederic Nataf, Soleiman Yousef, Simplice Donfack, Remi Lacroix. Robust algebraic schur complement based on low rank correction. In submission.

# Contributors
  Simplice Donfack
  Olivier Tissot
  Laura Grigori
  Sebastien Cayrols
  Alan Ayala Obregon

# Contact

  For any question, please contact {simplice.donfack, olivier.tissot, laura.grigori}@inria.fr .
