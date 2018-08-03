#
include make.inc


all: compil_cplm compil_utils  compil_src compil_examples ok

compil_cplm:
		@if [ ! -d $(LIBDIR) ]; then mkdir -v $(LIBDIR);fi
		( cd $(CPLMDIR) ; $(MAKE) )
		( cd $(CPLM_V0_DIR) ; $(MAKE) )
compil_utils:
	@if [ ! -d $(LIBDIR) ]; then mkdir -v $(LIBDIR);fi
	make -C $(UTILS)

compil_src:
	@if [ ! -d $(LIBDIR) ]; then mkdir -v $(LIBDIR);fi
	make -C $(SRC)

compil_examples:
	@if [ ! -d $(BINDIR) ]; then mkdir -v $(BINDIR);fi
	make -C $(EXAMPLES)

clean:
	rm -f $(LIBDIR)/lib$(LIBNAME).a
	make clean -C $(UTILS)
	( cd $(CPLMDIR) ; $(MAKE) clean)
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
