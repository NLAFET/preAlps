#
include make.inc

all: compilUtils compilSRC preAlps_test ok
spMatrix: compilSpMatrix compilUtils
sparseTP: compilSparseTP

compilUtils:
	make -C $(UTILS)
  
compilSpMatrix:
	make -C $(SRC) spMatrix

compilSparseTP:
	make -C $(SRC) sparseTP

compilSRC:
	make -C $(SRC)	

preAlps_test:
	make -C $(TEST)

clean:
	rm -f $(LIBDIR)/lib$(LIBNAME).a
	make clean -C $(UTILS)
	make clean -C $(SRC)
	make clean -C $(TEST)

distclean:
	clean
	make clean -C $(TEST)

ok:
	@echo ""
	@echo "****************************************************************************************"
	@echo "************** Compile sucessful    ****************************************************"
	@echo "************** Contact: {simplice.donfack, alan.ayala-obregon, laura.grigori}@inria.fr *"
	@echo "****************************************************************************************"
	@echo ""	
