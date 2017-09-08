include ../make.inc

PRECONDITIONERS_SRC = preconditioners/presc.c preconditioners/presc_eigsolve.c solvers/bcg.c
SOLVERS_SRC =

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
