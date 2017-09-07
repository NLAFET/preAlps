include ../make.inc

SRC = preAlps_utils.c preAlps_solver.c pardiso_solver.c mkl_pardiso_solver.c mumps_solver.c
OBJ = $(SRC:.c=.o)

all: $(OBJ) create_lib

%.o: %.c
	$(CC) -o $@ -c $< $(INCLUDE_FLAGS) $(CFLAGS) $(OPT)

create_lib:
	$(ARCH) $(ARCHFLAGS) ../$(LIBDIR)/lib$(LIBNAME).a $(OBJ)
	$(RANLIB) ../$(LIBDIR)/lib$(LIBNAME).a

clean:
	-${RM} $(OBJ)
