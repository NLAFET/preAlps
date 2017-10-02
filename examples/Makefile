include ../make.inc

SRC = test_ecgsolve.c #test_cgsolve.c test_presc.c test_lorasc.c

OBJ = $(SRC:.c=.o)
EXE  := $(OBJ:.o=)

all: $(OBJ) $(EXE)

%.o: %.c
	$(CC) -o $@ -c $< $(INCLUDE_FLAGS) $(CFLAGS) $(OPT)

$(EXE):	$(OBJ)
	$(LOADER) $(LOADOPTS) $(CFLAGS) $(LD_FLAGS) -L../$(LIBDIR) -l$(LIBNAME) $@.o -o ../$(BINDIR)/$@ $(OPT)

clean:
	-${RM} $(OBJ) ../$(BINDIR)/$(EXE)
