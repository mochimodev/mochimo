##
# GNUmakefile - C/C++ makefile for GNU Make
# Copyright 2021-2022 Adequate Systems, LLC. All Rights Reserved.
#

#####################
# vv CONFIGURATION vv

SHELL:= bash

# directory macros
BINDIR:= bin
BUILDDIR:= build
INCLUDEDIR:= include
NVINCLUDEDIR:= /usr/local/cuda/include
NVLIBDIR:= /usr/local/cuda/lib64 /usr/local/cuda/lib64/stubs
SOURCEDIR:= src
TESTBUILDDIR:= $(BUILDDIR)/test
TESTSOURCEDIR:= $(SOURCEDIR)/test

# compiler macros (CFLAGS is reserved for user input)
NVCC:= /usr/local/cuda/bin/nvcc
CC:= gcc

# module name (by default, the name of the root directory)
# NOTE: excessive makefile commands account for embedded make calls
MODULE:= $(notdir $(realpath $(dir $(lastword $(MAKEFILE_LIST)))))

# module library and coverage files
MODLIB:= $(BUILDDIR)/lib$(MODULE).a
COVERAGE:= $(BUILDDIR)/coverage.info

# source files: test (base/cuda), base, cuda
CUSRCS:= $(sort $(wildcard $(SOURCEDIR)/*.cu))
CSRCS:= $(sort $(wildcard $(SOURCEDIR)/*.c))
TCUSRCS:= $(sort $(wildcard $(TESTSOURCEDIR)/*-cu.c))
TCSRCS:= $(sort $(filter-out %-cu.c,$(wildcard $(TESTSOURCEDIR)/*.c)))
# object files: extended filename derived from file ext
COBJS:= $(patsubst $(SOURCEDIR)/%.c,$(BUILDDIR)/%.o,$(CSRCS))
CUOBJS:= $(patsubst $(SOURCEDIR)/%.cu,$(BUILDDIR)/%.cu.o,$(CUSRCS))
TCOBJS:= $(patsubst $(SOURCEDIR)/%.c,$(BUILDDIR)/%.o,$(TCSRCS))
TCUOBJS:= $(patsubst $(SOURCEDIR)/%-cu.c,$(BUILDDIR)/%-cu.o,$(TCUSRCS))

# cuda definition flag
CUDEF:= $(filter -DCUDA,$(CFLAGS))
# cuda device link code object (only where CUOBJS exists)
CULINK:= $(if $(and $(CUDEF),$(CUOBJS)),$(BUILDDIR)/culink.o,)

# dependency files; *.c only
DEPENDS:= $(patsubst $(SOURCEDIR)/%.c,$(BUILDDIR)/%.d, \
	$(CSRCS) $(TCSRCS) $(TCUSRCS) $(TCLSRCS))

# dynamic working set of objects
OBJECTS:= $(COBJS) $(if $(CUDEF),$(CUOBJS),)

# dynamic test objects, names and components; eye candy during tests
TESTOBJECTS:= $(TCOBJS) \
	$(if $(CUDEF),$(TCUOBJS),)
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
LIBFLAGS:= -l$(MODULE) $(patsubst $(INCLUDEDIR)/%,-l%,$(INCLUDES)) -lm \
	$(if $(CUDEF),-lcudart -lcudadevrt -lnvidia-ml -lstdc++,)

# compiler/linker flag macros
LDFLAGS:= $(addprefix -L,$(LIBDIRS)) -Wl,-\( $(LIBFLAGS) -Wl,-\) -pthread
CCFLAGS:= -Werror -Wall -Wextra $(addprefix -I,$(INCLUDEDIRS))
NVLDFLAGS:= $(addprefix -L,$(LIBDIRS)) $(LIBFLAGS)
NVCCFLAGS:= -Werror=all-warnings $(addprefix -I,$(INCLUDEDIRS))

## ^^ END CONFIGURATION ^^
##########################

.SUFFIXES: # disable rules predefined by MAKE
.PHONY: help all allcuda allopencl clean cleanall coverage docs \
	library libraries report test

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
	@echo "   make library       build a library file containing all objects"
	@echo "   make libraries     build all library files (incl. submodules)"
	@echo "   make report        build html report from test coverage"
	@echo "   make test          build and run tests"
	@echo "   make test-*        build and run sub tests matching *"
	@echo ""

# build "all" base objects; redirect (DEFAULT RULE)
all: $(OBJECTS)

# build all CUDA object files; redirect
allcuda: $(CUOBJS)

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
	 echo "PROJECT_NUMBER=v$$(cat VERSION)" )

# build library file; redirect
library: $(MODLIB)

# build all libraries (incl. submodules); redirect
libraries: $(SUBLIBS) $(MODLIB)

# build local html coverage report from coverage data
report: $(COVERAGE)
	genhtml $(COVERAGE) --output-directory $(BUILDDIR)

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

############################
# vv RECIPE CONFIGURATION vv

# include custom recipe configurations here

## ^^ END RECIPE CONFIGURATION ^^
#################################

# build module library, within lib directory, from all base objects
$(MODLIB): $(OBJECTS)
	@mkdir -p $(dir $@)
	ar rcs $(MODLIB) $(OBJECTS)

$(SUBLIBS): %:
	git submodule update --init --recursive
	@make library -C $(INCLUDEDIR)/$(word 2,$(subst /, ,$@))

# build coverage file, within out directory
$(COVERAGE):
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

# build cuda device link code object, within build directory
$(CULINK): $(MODLIB)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCFLAGS) -dlink $(MODLIB) -o $(CULINK) $(NVLDFLAGS)

# build binaries, within build directory, from associated objects
$(BUILDDIR)/%: $(SUBLIBS) $(MODLIB) $(CULINK) $(BUILDDIR)/%.o
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(BUILDDIR)/$*.o $(CULINK) -o $@ $(LDFLAGS)
# build cuda objects, within build directory, from associated sources
$(BUILDDIR)/%.cu.o: $(SOURCEDIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCFLAGS) -dc $(abspath $<) -o $@ $(NVCCFLAGS)
# build c objects, within build directory, from associated sources
$(BUILDDIR)/%.o: $(SOURCEDIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -MMD -MP -c $(abspath $<) -o $@ $(CCFLAGS)

# include depends rules created during "build object file" process
-include $(DEPENDS)
