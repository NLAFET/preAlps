include ../make.inc

SRC = test_ecg_petsc_op.c test_ecg_prealps_op.c test_ecg_bench_petsc_pcg.c test_lorasc.c test_ecg_bench_fused.c test_ecg_petsc_ela.c test_bench_spmm.c test_bench_bjacobi.c

OBJ = $(SRC:.c=.o)
EXE = $(OBJ:.o=)

all: $(OBJ) $(EXE)

%.o: %.c
	$(CC) $(CFLAGS) -o $@ -c $< $(INCLUDE_FLAGS) $(OPT)

$(EXE): $(OBJ)
	$(LOADER) -o ../$(BINDIR)/$@ $@.o -L../$(LIBDIR) -l$(LIBNAME) $(LD_FLAGS) $(LOADOPTS) $(OPT)

clean:
	-${RM} $(OBJ) $(addprefix ../$(BINDIR)/,$(EXE))
