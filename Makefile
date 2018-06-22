#
include make.inc


all: compil_cplm compil_utils  compil_src compil_examples ok

compil_cplm:
		@if [ ! -d $(LIBDIR) ]; then mkdir -v $(LIBDIR);fi
		( cd $(CPLMDIR) ; $(MAKE) )
		
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

install_cpalamem:$(CPALAMEM_DIR)
	cd $(CPALAMEM_DIR);./configure --cc $(CC) --with-metis --with-mkl
	make compil_cpalamem

$(CPALAMEM_TAR):$(CPALAMEM_DIR)/configure

$(CPALAMEM_DIR)/configure:
	@tar xzvf $(CPALAMEM_TAR) -C $(CPALAMEM_DIR)

$(CPALAMEM_DIR):
	@if [ ! -d $(CPALAMEM_DIR) ]; then mkdir -v $(CPALAMEM_DIR);fi
	make $(CPALAMEM_TAR)

compil_cpalamem:
	make -C $(CPALAMEM_DIR) full

clean_cpalamem:
	make -C $(CPALAMEM_DIR) cleanAll

remove_cpalamem:
	@if [ -d $(CPALAMEM_DIR) ]; then rm -vr $(CPALAMEM_DIR);fi

ok:
	@echo ""
	@echo "****************************************************************************************"
	@echo "************** Compile sucessful    ****************************************************"
	@echo "****************************************************************************************"
	@echo ""
