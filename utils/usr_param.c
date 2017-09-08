/******************************************************************************/
/* Author     : Olivier Tissot                                                */
/* Creation   : 2016/06/23                                                    */
/* Description: Basic struct for user parameters                              */
/******************************************************************************/


/******************************************************************************/
/*                                  INCLUDE                                   */
/******************************************************************************/
#include "usr_param.h"
/******************************************************************************/

/******************************************************************************/
/*                                    CODE                                    */
/******************************************************************************/
#define PARBCG_DEFAULT_ITER_MAX 100000
#define PARBCG_DEFAULT_TOL 1e-5
#define PARBCG_DEFAULT_NB_RHS 0

/* Private function to print the help message */
void _print_help() {
  printf("DESCRPTION\n");
  printf("\tSolves Ax = b using a Parallel Block Conjugate Gradient."
          " A must be symmetric positive definite.\n");
  printf("USAGE\n");
  printf("\tmpirun -n nb_proc"
         "./parbcg"
         " [-h/--help]"
         " [-i/--iteration-maximum int]"
         " -m/--matrix file"
         " -n/--nb-block-part int"
         " [-r/--nb-rhs int]"
         " -s/--solver-param-filename file"
         " [-t/--tolerance double]\n");
  printf("OPTIONS\n");
  printf("\t-h/--help             : print this help message\n");
  printf("\t-i/--iteration-maximum: maximum of iteration count"
                                  " (default is %d)\n",PARBCG_DEFAULT_ITER_MAX);
  printf("\t-m/--matrix           : the .mtx file containing the matrix A\n");
  printf("\t-n/--nb-block-part    : the number of block in A's partitioning\n");
  printf("\t-r/--nb-rhs           : the number of rhs"
                                  " (default is number of processes)\n");
  printf("\t-s/--solver-filename  : the file containing parameters of the solver\n");
  printf("\t-t/--tolerance        : tolerance of the method"
                                  " (default is %e)\n",PARBCG_DEFAULT_TOL);
}

Usr_Param_t UsrParamNULL() {
  Usr_Param_t p;
  p.nbBlockPart = 0;
  p.nbRHS = 0;
  p.matrixFilename = NULL;
  p.solverFilename = NULL;
  p.tolerance = PARBCG_DEFAULT_TOL;
  p.iterMax = PARBCG_DEFAULT_ITER_MAX;
  return p;
}

/* Private function Fill the structure from command line arguments */
int _fillParamFromCline(Usr_Param_t* p, int argc, char** argv) {

  int c;

  static struct option long_options[] = {
    {"help"             , no_argument      , NULL, 'h'},
    {"iteration-maximum", optional_argument, NULL, 'i'},
    {"matrix"           , required_argument, NULL, 'm'},
    {"nb-block-part"    , required_argument, NULL, 'n'},
    {"nb-rhs"           , optional_argument, NULL, 'r'},
    {"solver-filename"  , required_argument, NULL, 's'},
    {"tolerance"        , optional_argument, NULL, 't'},
    {NULL               , 0                , NULL, 0}
  };

  int opterr = 0;
  int option_index = 0;

  while ((c = getopt_long(argc, argv, "hi:m:n:r:s:t:", long_options, &option_index)) != -1)
    switch (c) {
      case 'h':
        _print_help();
        MPI_Abort(MPI_COMM_WORLD, opterr);
      case 'i':
        if (optarg == NULL)
          p->iterMax = PARBCG_DEFAULT_ITER_MAX;
        else
          p->iterMax = atoi(optarg);
        break;
      case 'n':
        p->nbBlockPart = atoi(optarg);
        break;
      case 'm':
        p->matrixFilename = optarg;
        break;
      case 'r':
        if (optarg == NULL)
          p->nbRHS = PARBCG_DEFAULT_NB_RHS;
        else
          p->nbRHS = atoi(optarg);
        break;
      case 's':
          p->solverFilename = optarg;
          break;
      case 't':
        if (optarg == NULL)
          p->tolerance = PARBCG_DEFAULT_TOL;
        else
          p->tolerance = atof(optarg);
        break;
      case '?':
        if (optopt == 'i'
            || optopt == 'n'
            || optopt == 'm'
            || optopt == 'r'
            || optopt == 's'
            || optopt == 't')
          fprintf (stderr, "Option -%c requires an argument.\n", optopt);
        else if (isprint (optopt))
          fprintf (stderr, "Unknown option `-%c'.\n", optopt);
        else
          fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
        _print_help();
        MPI_Abort(MPI_COMM_WORLD, opterr);
      default:
        MPI_Abort(MPI_COMM_WORLD, opterr);
    }
    return 1;
}

int UsrParamReadFromCline(Usr_Param_t* param, int argc, char** argv) {
PUSH
  int ierr = 0, rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  // Get the parameters from command line
  ierr = _fillParamFromCline(param, argc, argv);
  // Default nbRHS is the number of processes
  if (param->nbRHS == 0)
    param->nbRHS = size;
  // TODO: handle the case where param.nbBlockCG <= param.nbBlockPart
  if (param->nbRHS > param->nbBlockPart)
    CPALAMEM_Abort("The number of block in A's partitioning has to be higher than the number of rhs!\n");
POP
  return ierr;
}

/******************************************************************************/
