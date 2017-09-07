#
include make.inc


all: compil_utils  compil_src compil_examples ok

compil_utils:
	make -C $(UTILS)

compil_src:
	make -C $(SRC)

compil_examples:
	make -C $(EXAMPLES)

clean:
	rm -f $(LIBDIR)/lib$(LIBNAME).a
	make clean -C $(UTILS)
	make clean -C $(SRC)
	make clean -C $(EXAMPLES)

distclean:
	clean
	make clean -C $(EXAMPLES)

ok:
	@echo ""
	@echo "****************************************************************************************"
	@echo "************** Compile sucessful    ****************************************************"
	@echo "****************************************************************************************"
	@echo ""
