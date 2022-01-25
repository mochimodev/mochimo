##
# GNUmakefile - C/C++ makefile for GNU Make
# Copyright 2021-2022 Adequate Systems, LLC. All Rights Reserved.
#

SHELL:= bash

# directory macros
BINDIR = bin
BUILDDIR = build
INCLUDEDIR = include
SOURCEDIR = src
TESTBUILDDIR = $(BUILDDIR)/test
TESTSOURCEDIR = $(SOURCEDIR)/test

#####################
# vv CONFIGURATION vv

# module name (by default, the name of the root directory)
# NOTE: excessive makefile commands account for embedded make calls
MODULE:= $(notdir $(realpath $(dir $(lastword $(MAKEFILE_LIST)))))

# test sources, objects, depends, names and components
TESTSOURCES:= $(sort $(wildcard $(TESTSOURCEDIR)/*.c))
TESTOBJECTS:= $(patsubst $(SOURCEDIR)/%.c,$(BUILDDIR)/%.o,$(TESTSOURCES))
TESTDEPENDS:= $(patsubst $(SOURCEDIR)/%.c,$(BUILDDIR)/%.d,$(TESTSOURCES))
TESTNAMES:= $(patsubst $(TESTSOURCEDIR)/%.c,%,$(TESTSOURCES))
TESTCOMPS:= $(shell echo $(TESTSOURCES) | sed 's/\s/\n/g' | \
	sed -E 's/\S*\/([^-]*)[-.]+\S*/\1/g' | sort -u)
# base sources, objects and depends
BASESOURCES:= $(sort $(wildcard $(SOURCEDIR)/*.c))
BASEOBJECTS:= $(patsubst $(SOURCEDIR)/%.c,$(BUILDDIR)/%.o,$(BASESOURCES))
BASEDEPENDS:= $(patsubst $(SOURCEDIR)/%.c,$(BUILDDIR)/%.d,$(BASESOURCES))
# all sources, objects, depends, library and coverage files
SOURCES:= $(BASESOURCES) $(TESTSOURCES)
OBJECTS:= $(BASEOBJECTS) $(TESTOBJECTS)
DEPENDS:= $(BASEDEPENDS) $(TESTDEPENDS)
LIBRARY:= $(BUILDDIR)/lib$(MODULE).a
COVERAGE:= $(BUILDDIR)/coverage.info

# includes and include/library directories
INCLUDES:= $(wildcard $(INCLUDEDIR)/**)
INCLUDEDIRS:= $(SOURCEDIR) $(addsuffix /$(SOURCEDIR),$(INCLUDES))
LIBRARYDIRS:= $(BUILDDIR) $(addsuffix /$(BUILDDIR),$(INCLUDES))
LIBRARIES:= $(join $(INCLUDES),\
	$(patsubst $(INCLUDEDIR)/%,/$(BUILDDIR)/lib%.a,$(INCLUDES)))
NVINCLUDEDIR:= /usr/local/cuda/include

# compiler macros
LFLAGS:= -l$(MODULE) $(patsubst $(INCLUDEDIR)/%,-l%,$(INCLUDES))
LDFLAGS:= $(addprefix -L,$(LIBRARYDIRS)) -Wl,-\( $(LFLAGS) -Wl,-\)
CCFLAGS:= -Werror -Wall -Wextra $(addprefix -I,$(INCLUDEDIRS))
NVFLAGS:= -Werror all-warnings $(addprefix -I,$(INCLUDEDIRS) $(NVINCLUDEDIR))
NVCC:= /usr/local/cuda/bin/nvcc $(CFLAGS)
CC:= gcc $(CFLAGS) # CFLAGS is reserved for additional input

## ^^ END CONFIGURATION ^^
##########################

.SUFFIXES: # disable rules predefined by MAKE
.PHONY: help all clean cleanall coverage cuda docs library libraries report test

help: # default rule prints help information
	@echo ""
	@echo "Usage:  make [options] [FLAGS=FLAGVALUES]"
	@echo "   make               prints this usage information"
	@echo "   make all           build all object files"
	@echo "   make clean         removes build directory and files"
	@echo "   make cleanall      removes (all) build directories and files"
	@echo "   make coverage      build test coverage file"
	@echo "   make cuda          build cuda compatible object files"
	@echo "   make docs          build documentation files"
	@echo "   make library       build a library file containing all objects"
	@echo "   make libraries     build all library files required for binaries"
	@echo "   make report        build html report from test coverage"
	@echo "   make test          build and run all tests"
	@echo "   make test-<test>   build and run tests matching <test>*"
	@echo ""

# build "all" base objects; redirect (DEFAULT RULE)
all: $(BASEOBJECTS)

# remove build directory and files
clean:
	@rm -rf $(BUILDDIR)

# remove all build directories and files; recursive
cleanall: clean
	@$(foreach DIR,$(INCLUDES),make cleanall -C $(DIR); )

# build test coverage (requires lcov); redirect
coverage: $(COVERAGE)

# build CUDA compatible object files; recursive
cuda:
	@make all "CC=$(NVCC)" "CCFLAGS=$(NVFLAGS)" --no-print-directory

# build documentation files under docs/ (requires doxygen)
docs:
	@mkdir -p docs
	doxygen .github/docs/config

# build library file; redirect
library: $(LIBRARY)

# build all libraries (incl. submodules); redirect
libraries: $(LIBRARIES) $(LIBRARY)

# build local html coverage report from coverage data
report: $(COVERAGE)
	genhtml $(COVERAGE) --output-directory $(BUILDDIR)

# build and run all tests
test: $(LIBRARIES) $(LIBRARY) $(TESTOBJECTS)
	@if test -d $(BUILDDIR); then find $(BUILDDIR) -name *.fail -delete; fi
	@echo -e "\n[========] Found $(words $(TESTNAMES)) tests" \
		"for $(words $(TESTCOMPS)) components in \"$(MODULE)\""
	@echo "[========] Performing all tests in \"$(MODULE)\" by component"
	@$(foreach COMP,$(TESTCOMPS),make test-$(COMP) --no-print-directory; )
	@export FAILS=$$(find $(BUILDDIR) -name *.fail -delete -print | wc -l); \
	 echo -e "\n[========] Testing completed. Analysing results..."; \
	 echo -e "[ PASSED ] $$(($(words $(TESTNAMES))-FAILS)) tests passed."; \
	 echo -e "[ FAILED ] $$FAILS tests failed.\n"; \
	 exit $$FAILS

# build and run specific tests matching pattern
test-%: $(LIBRARIES) $(LIBRARY)
	@echo -e "\n[--------] Performing $(words $(filter $*%,$(TESTNAMES)))" \
		"tests matching \"$*\""
	@$(foreach TEST,\
		$(addprefix $(TESTBUILDDIR)/,$(filter $*%,$(TESTNAMES))),\
		make $(TEST) -s && ( $(TEST) && echo "[ ✔ PASS ] $(TEST)" || \
		( touch $(TEST).fail && echo "[ ✖ FAIL ] $(TEST)" ) \
	 ) || ( touch $(TEST).fail && \ echo "[  ERROR ] $(TEST), ecode=$$?" ); )

# build module library, within lib directory, from all base objects
$(LIBRARY): $(BASEOBJECTS)
	@mkdir -p $(dir $@)
	ar rcs $(LIBRARY) $(BASEOBJECTS)

$(LIBRARIES): %:
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

# build binaries, within build directory, from associated objects
$(BUILDDIR)/%: $(LIBRARIES) $(LIBRARY) $(BUILDDIR)/%.o
	@mkdir -p $(dir $@)
	$(CC) $(BUILDDIR)/$*.o -o $@ $(LDFLAGS)

# build objects, within build directory, from associated sources
$(BUILDDIR)/%.o: $(SOURCEDIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) -MMD -MP -c $(abspath $<) -o $@ $(CCFLAGS)

# include depends rules created during "build object file" process
-include $(DEPENDS)

##########################
# vv ADDITIONAL RECIPES vv
