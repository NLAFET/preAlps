include ../make.inc

PREC_DIR=preconditioners
SOLVER_DIR=solvers
PRECONDITIONERS_SRC = $(PREC_DIR)/block_jacobi.c $(PREC_DIR)/precond.c\
											$(PREC_DIR)/preAlps_preconditioner.c\
											$(PREC_DIR)/presc.c $(PREC_DIR)/presc_eigsolve.c\
											$(PREC_DIR)/lorasc.c

SOLVERS_SRC = $(SOLVER_DIR)/ecg.c

PRECONDITIONERS_OBJ = $(PRECONDITIONERS_SRC:.c=.o)
SOLVERS_OBJ = $(SOLVERS_SRC:.c=.o)
OBJ  = $(PRECONDITIONERS_OBJ) $(SOLVERS_OBJ)

all: $(OBJ) create_lib

%.o: %.c
	$(CC) -o $@ -c $^ $(INCLUDE_FLAGS) $(CFLAGS) $(OPT)

create_lib:
	$(ARCH) $(ARCHFLAGS) ../$(LIBDIR)/lib$(LIBNAME).a $(OBJ)
	$(RANLIB) ../$(LIBDIR)/lib$(LIBNAME).a

clean:
	-${RM} $(OBJ)
