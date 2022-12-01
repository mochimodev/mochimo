
##
# GNUmakefile - C/C++ makefile for GNU Make
# Copyright 2021-2022 Adequate Systems, LLC. All Rights Reserved.
#

#####################
# vv CONFIGURATION vv

# system utils
CC:= /usr/bin/gcc
NVCC:= /usr/local/cuda/bin/nvcc
SHELL:= bash

# system directories
CUDADIR:= /usr/local/cuda
NVINCLUDEDIR:= /usr/local/cuda/include
NVLIBDIR:= /usr/local/cuda/lib64 /usr/local/cuda/lib64/stubs

# project directories
BINDIR:= bin
BUILDDIR:= build
INCLUDEDIR:= include
SOURCEDIR:= src
TESTBUILDDIR:= $(BUILDDIR)/test
TESTSOURCEDIR:= $(SOURCEDIR)/test
BINSOURCEDIR:= $(SOURCEDIR)/bin

# build definitions
GITVERSION:=$(shell git describe --always --dirty --tags || echo "<no-ver>")
VERDEF:=-DGIT_VERSION="\"$(GITVERSION)\""

# input flags
CUDEF:= $(filter -DCUDA,$(CFLAGS))
CFLAGS:= # RESERVED for user compile options
LFLAGS:= # RESERVED for user linking options
NVCFLAGS:= # RESERVED for user compilation options specific to NVCC

# module name (by default, the name of the root directory)
# NOTE: excessive makefile commands account for embedded make calls
MODULE:= $(notdir $(realpath $(dir $(lastword $(MAKEFILE_LIST)))))

# module library and coverage files
MODLIB:= $(BUILDDIR)/lib$(MODULE).a
COVERAGE:= $(BUILDDIR)/coverage.info

# source files: test (base/cuda), base, cuda
BCSRCS:= $(sort $(wildcard $(BINSOURCEDIR)/*.c))
CUSRCS:= $(sort $(wildcard $(SOURCEDIR)/*.cu))
CSRCS:= $(sort $(wildcard $(SOURCEDIR)/*.c))
TCUSRCS:= $(sort $(wildcard $(TESTSOURCEDIR)/*-cu.c))
TCSRCS:= $(sort $(filter-out %-cu.c,$(wildcard $(TESTSOURCEDIR)/*.c)))

# object files: extended filename derived from file ext
COBJS:= $(patsubst $(SOURCEDIR)/%.c,$(BUILDDIR)/%.o,$(CSRCS))
CUOBJS:= $(patsubst $(SOURCEDIR)/%.cu,$(BUILDDIR)/%.cu.o,$(CUSRCS))
TCOBJS:= $(patsubst $(SOURCEDIR)/%.c,$(BUILDDIR)/%.o,$(TCSRCS))
TCUOBJS:= $(patsubst $(SOURCEDIR)/%-cu.c,$(BUILDDIR)/%-cu.o,$(TCUSRCS))

# dependency files; compatible only with *.c files
DEPENDS:= $(patsubst $(SOURCEDIR)/%.c,$(BUILDDIR)/%.d, \
	$(BCSRCS) $(CSRCS) $(TCSRCS) $(TCUSRCS) $(TCLSRCS))

# dynamic working set of objects; dependant on compilation flags
OBJECTS:= $(COBJS) $(if $(CUDEF),$(CUOBJS),)

# dynamic test objects, names and components; eye candy during tests
TESTOBJECTS:= $(TCOBJS) $(if $(CUDEF),$(TCUOBJS),)
TESTNAMES:= $(basename $(patsubst $(TESTBUILDDIR)/%,%,$(TESTOBJECTS)))
TESTCOMPS:= $(shell echo $(TESTOBJECTS) | sed 's/\s/\n/g' | \
	sed -E 's/\S*\/([^-]*)[-.]+\S*/\1/g' | sort -u)

# includes and include/library directories
INCLUDES:= $(wildcard $(INCLUDEDIR)/**)
INCLUDEDIRS:= $(SOURCEDIR) $(addsuffix /$(SOURCEDIR),$(INCLUDES)) \
	$(if $(CUDEF),$(NVINCLUDEDIR),)
LIBDIRS:= $(BUILDDIR) $(addsuffix /$(BUILDDIR),$(INCLUDES)) \
	$(if $(CUDEF),$(NVLIBDIR),)
SUBLIBS:= $(join $(INCLUDES),\
	$(patsubst $(INCLUDEDIR)/%,/$(BUILDDIR)/lib%.a,$(INCLUDES)))
LIBFLAGS:= -l$(MODULE) $(patsubst $(INCLUDEDIR)/%,-l%,$(INCLUDES)) \
	$(if $(CUDEF),-lcudart -lnvidia-ml -lstdc++,) -lm

# compiler/linker flag macros
LDFLAGS:= $(addprefix -L,$(LIBDIRS)) -Wl,-\( $(LIBFLAGS) -Wl,-\) -pthread
CCFLAGS:= $(addprefix -I,$(INCLUDEDIRS)) -Werror -Wall -Wextra -MMD -MP
NVCCFLAGS:= $(addprefix -I,$(INCLUDEDIRS)) -Xptxas -Werror

## ^^ END CONFIGURATION ^^
##########################

.SUFFIXES: # disable rules predefined by MAKE
.PHONY: help all allcuda clean cleanall coverage docs library report \
	sublibraries test version

help: # default rule prints help information
	@echo ""
	@echo "Usage:  make [options] [FLAGS=FLAGVALUES]"
	@echo "   make               prints this usage information"
	@echo "   make all           build all object files"
	@echo "   make allcuda       build all CUDA object files"
	@echo "   make clean         removes build directory and files"
	@echo "   make cleanall      removes (all) build directories and files"
	@echo "   make coverage      build test coverage file"
	@echo "   make docs          build documentation files"
	@echo "   make report        build html report from test coverage"
	@echo "   make library       build a library file containing all objects"
	@echo "   make sublibraries  build all library files (incl. submodules)"
	@echo "   make test          build and run tests"
	@echo "   make test-*        build and run sub tests matching *"
	@echo "   make variable-*    show the value of a variable matching *"
	@echo "   make version       show the git repository version string"
	@echo ""

# build "all" base objects; redirect (DEFAULT RULE)
all: $(OBJECTS)

# build all CUDA object files; redirect
allcuda:
	@make $(CUOBJS) "CFLAGS=-DCUDA $(CFLAGS)" --no-print-directory

# remove build directory and files
clean:
	@rm -rf $(BUILDDIR)

# remove all build directories and files; recursive
cleanall: clean
	@$(foreach DIR,$(INCLUDES),make cleanall -C $(DIR); )

# build test coverage (requires lcov); redirect
coverage: $(COVERAGE)

# build documentation files under docs/ (requires doxygen)
docs:
	@mkdir -p docs
	@doxygen <( cat .github/docs/config; \
	 echo "PROJECT_NAME=$(MODULE)" | tr '[:lower:]' '[:upper:]'; \
	 echo "PROJECT_NUMBER=$(GITVERSION)" )

# build library file; redirect
library: $(MODLIB)

# build local html coverage report from coverage data
report: $(COVERAGE)
	genhtml $(COVERAGE) --output-directory $(BUILDDIR)

# initialize and build build submodule libraries; redirect
sublibraries: $(SUBLIBS)

# build and run specific tests matching pattern
test-%: $(SUBLIBS) $(MODLIB)
	@echo -e "\n[--------] Performing $(words $(filter $*%,$(TESTNAMES)))" \
		"tests matching \"$*\""
	@$(foreach TEST,\
		$(addprefix $(TESTBUILDDIR)/,$(filter $*%,$(TESTNAMES))),\
		make $(TEST) -s && ( $(TEST) && echo "[ ✔ PASS ] $(TEST)" || \
		( touch $(TEST).fail && echo "[ ✖ FAIL ] $(TEST)" ) \
	 ) || ( touch $(TEST).fail && \ echo "[  ERROR ] $(TEST), ecode=$$?" ); )

# build and run tests
test: $(SUBLIBS) $(MODLIB) $(TESTOBJECTS)
	@if test -d $(BUILDDIR); then find $(BUILDDIR) -name *.fail -delete; fi
	@echo -e "\n[========] Found $(words $(TESTNAMES)) tests" \
		"for $(words $(TESTCOMPS)) components in \"$(MODULE)\""
	@echo "[========] Performing all tests in \"$(MODULE)\" by component"
	@-$(foreach COMP,$(TESTCOMPS),make test-$(COMP) --no-print-directory; )
	@export FAILS=$$(find $(BUILDDIR) -name *.fail -delete -print | wc -l); \
	 echo -e "\n[========] Testing completed. Analysing results..."; \
	 echo -e "[ PASSED ] $$(($(words $(TESTNAMES))-FAILS)) tests passed."; \
	 echo -e "[ FAILED ] $$FAILS tests failed.\n"; \
	 exit $$FAILS

# echo the value of a variable matching pattern
variable-%:
	@echo $* = $($*)

# echo the value of the GITVERSION
version:
	@echo $(GITVERSION)

############################
# vv RECIPE CONFIGURATION vv

# include custom recipe configurations here

## ^^ END RECIPE CONFIGURATION ^^
#################################

# build module library, within build directory, from dynamic object set
$(MODLIB): $(OBJECTS)
	@mkdir -p $(dir $@)
	ar rcs $(MODLIB) $(OBJECTS)

# build submodule libraries, within associated directories
$(SUBLIBS): %:
	git submodule update --init --recursive
	@make library -C $(INCLUDEDIR)/$(word 2,$(subst /, ,$@))

# build coverage file, within out directory
$(COVERAGE):
	@mkdir -p $(dir $@)
	@make clean all --no-print-directory "CFLAGS=$(CFLAGS) --coverage -O0"
	lcov -c -i -d $(BUILDDIR) -o $(COVERAGE)_base
	@make test --no-print-directory "CFLAGS=$(CFLAGS) --coverage -O0"
	lcov -c -d $(BUILDDIR) -o $(COVERAGE)_test
	lcov -a $(COVERAGE)_base -a $(COVERAGE)_test -o $(COVERAGE) || \
		cp $(COVERAGE)_base $(COVERAGE)
	rm -rf $(COVERAGE)_base $(COVERAGE)_test
	lcov -r $(COVERAGE) '*/$(TESTSOURCEDIR)/*' -o $(COVERAGE)
	@$(foreach INC,$(INCLUDEDIRS),if test $(DEPTH) -gt 0; then \
		make coverage -C $(INC) DEPTH=$$(($(DEPTH) - 1)); fi; )

# build binaries, within build directory, from associated objects
$(BUILDDIR)/%: $(SUBLIBS) $(MODLIB) $(BUILDDIR)/%.o
	@mkdir -p $(dir $@)
	$(CC) $(BUILDDIR)/$*.o -o $@ $(LDFLAGS) $(LFLAGS) $(CFLAGS) $(VERDEF)

# build cuda objects, within build directory, from *.cu files
$(BUILDDIR)/%.cu.o: $(SUBLIBS) $(SOURCEDIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) -c $(SOURCEDIR)/$*.cu -o $@ $(NVCCFLAGS) $(NVCFLAGS) $(VERDEF)

# build c objects, within build directory, from *.c files
$(BUILDDIR)/%.o: $(SUBLIBS) $(SOURCEDIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) -c $(SOURCEDIR)/$*.c -o $@ $(CCFLAGS) $(CFLAGS) $(VERDEF)

# include depends rules created during "build object file" process
-include $(DEPENDS)
