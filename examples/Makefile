include ../make.inc

SRC = test_ecgsolve.c #test_lorasc.c test_presc.c #test_cgsolve.c test_presc.c

OBJ = $(SRC:.c=.o)
EXE = $(OBJ:.o=)

all: $(OBJ) $(EXE)

%.o: %.c
	$(CC) -o $@ -c $< $(INCLUDE_FLAGS) $(CFLAGS) $(OPT)

$(EXE):	$(OBJ)
	$(LOADER) -o ../$(BINDIR)/$@ $@.o -L../$(LIBDIR) -l$(LIBNAME) $(LD_FLAGS) $(LOADOPTS) $(OPT)

clean:
	-${RM} $(OBJ) $(addprefix ../$(BINDIR)/,$(EXE))
