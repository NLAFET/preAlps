include ../make.inc

SRC =  test_spMSV.c test_prototypeTP.c test_prototypeQR.c test_prototypeCUR.c
OBJ = $(SRC:.c=.o)
EXE  := $(OBJ:.o=)

all: $(OBJ) $(EXE)

%.o: %.c
	$(CC) -o $@ -c $< $(INCLUDE_FLAGS) $(CFLAGS) $(OPT)

$(EXE):	$(OBJ)
	$(LOADER) $(LOADOPTS) $(CFLAGS) $(INCLUDE_FLAGS) -L../$(LIBDIR) -l$(LIBNAME) $@.o -o ../$(BINDIR)/$@ -l$(LIBNAME) $(LINKER_LD) $(OPT)
  
clean:
	-${RM} $(OBJ) ../$(BINDIR)/$(EXE)
