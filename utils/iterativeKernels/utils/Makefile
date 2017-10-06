include ../make.inc

SRC = mmio.c s_utils.c s_utils_mp.c spTP_utils.c
OBJ = $(SRC:.c=.o)

all: $(OBJ) create_lib

%.o: %.c
	$(CC) -o $@ -c $< $(INCLUDE_FLAGS) $(CFLAGS) $(OPT)
  
create_lib:
	$(ARCH) $(ARCHFLAGS) ../$(LIBDIR)/lib$(LIBNAME).a $(OBJ)
	$(RANLIB) ../$(LIBDIR)/lib$(LIBNAME).a
  
clean:
	-${RM} $(OBJ)
