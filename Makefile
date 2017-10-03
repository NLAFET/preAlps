#
include make.inc


all: compil_utils  compil_src compil_examples ok

install_cpalamem:
	@if [ ! -d $(CPALAMEM_DIR) ]; then mkdir -v $(CPALAMEM_DIR); tar xzvf utils/$(CPALAMEM_TAR) -C $(CPALAMEM_DIR);	cd $(CPALAMEM_DIR);./configure --cc $(CC) --with-metis --with-mkl;	make full;fi

compil_cpalamem:
	make -C $(CPALAMEM_ROOT) full

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
