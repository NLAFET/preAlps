include ../make.inc

SRC =  test_cgsolve.c test_presc.c
OBJ = $(SRC:.c=.o)
EXE  := $(OBJ:.o=)

all: $(OBJ) $(EXE)

%.o: %.c
	$(CC) -o $@ -c $< $(INCLUDE_FLAGS) $(CFLAGS) $(OPT)

$(EXE):	$(OBJ)
	$(LOADER) $(LOADOPTS) $(CFLAGS) $(INCLUDE_FLAGS) -L../$(LIBDIR) -l$(LIBNAME) $@.o -o ../$(BINDIR)/$@ -l$(LIBNAME) $(LD_FLAGS) $(OPT)

clean:
	-${RM} $(OBJ) ../$(BINDIR)/$(EXE)
