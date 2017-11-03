include ../make.inc

SRC = ecg_petsc_op.c ecg_prealps_op.c ecg_bench_petsc_pcg.c test_lorasc.c

OBJ = $(SRC:.c=.o)
EXE = $(OBJ:.o=)

all: $(OBJ) $(EXE)

%.o: %.c
	$(CC) $(CFLAGS) -o $@ -c $< $(INCLUDE_FLAGS) $(OPT)

$(EXE): $(OBJ)
	$(LOADER) -o ../$(BINDIR)/$@ $@.o -L../$(LIBDIR) -l$(LIBNAME) $(LD_FLAGS) $(LOADOPTS) $(OPT)

clean:
	-${RM} $(OBJ) $(addprefix ../$(BINDIR)/,$(EXE))
