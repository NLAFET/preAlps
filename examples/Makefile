include ../make.inc

SRC = test_ecgsolve.c #test_lorasc.c test_cgsolve.c test_presc.c

OBJ = $(SRC:.c=.o)
EXE = $(OBJ:.o=)

all: $(OBJ) $(EXE)

%.o: %.c
	$(CC) -o $@ -c $< $(INCLUDE_FLAGS) $(CFLAGS) $(OPT)

$(EXE):	$(OBJ)
##$(LOADER) $(LOADOPTS) $(CFLAGS) $(LD_FLAGS) -L../$(LIBDIR) -l$(LIBNAME) $@.o -o ../$(BINDIR)/$@ $(OPT)
	$(LOADER) -o ../$(BINDIR)/$@ $< -L../$(LIBDIR) -l$(LIBNAME) $(LD_FLAGS) $(LOADOPTS) $(OPT)

clean:
	for file in $(EXE); do \
		rm -f ../$(BINDIR)/$$file; \
	done
	-${RM} $(OBJ)
