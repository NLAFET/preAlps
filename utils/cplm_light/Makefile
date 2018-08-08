include ../../make.inc

#The name of the library to create
LIBCPLM_NAME = cplm_light

#The full path for the library
LIBCPLM = ../../$(LIBDIR)/lib$(LIBCPLM_NAME).a

SRC = $(shell ls *.c)

OBJ = $(SRC:.c=.o)

all: $(OBJ) create_lib

%.o: %.c
	$(CC) -o $@ -c $< $(INCLUDE_FLAGS) $(CFLAGS) $(OPT)

create_lib:
	$(ARCH) $(ARCHFLAGS) $(LIBCPLM) $(OBJ)
	$(RANLIB) $(LIBCPLM)

clean:
	-${RM} $(OBJ) $(LIBCPLM)
