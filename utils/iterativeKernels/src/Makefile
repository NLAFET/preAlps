include ../make.inc

SRC0 = spMSV.c preAlps_matrix.c preAlps_matrix_mp.c
SRC1 = tournamentPivoting.c tournamentPivotingQR.c tournamentPivotingCUR.c
OBJ0 = $(SRC0:.c=.o)
OBJ1 = $(SRC1:.c=.o)
OBJ  = $(OBJ0) $(OBJ1)

all: $(OBJ) create_lib

spMatrix: $(OBJ0) create_lib0

sparseTP: $(OBJ1) create_lib1

%.o: %.c
	$(CC) -o $@ -c $^ $(INCLUDE_FLAGS) $(CFLAGS) $(OPT)

create_lib0:
	$(ARCH) $(ARCHFLAGS) ../$(LIBDIR)/lib$(LIBNAME).a $(OBJ0)
	$(RANLIB) ../$(LIBDIR)/lib$(LIBNAME).a

create_lib1:
	$(ARCH) $(ARCHFLAGS) ../$(LIBDIR)/lib$(LIBNAME).a $(OBJ1)
	$(RANLIB) ../$(LIBDIR)/lib$(LIBNAME).a

create_lib:
	$(ARCH) $(ARCHFLAGS) ../$(LIBDIR)/lib$(LIBNAME).a $(OBJ)
	$(RANLIB) ../$(LIBDIR)/lib$(LIBNAME).a
  
clean:
	-${RM} $(OBJ)
