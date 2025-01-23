
##
# GNUmakefile - C/C++ makefile for GNU Make
# Copyright 2021-2025 Adequate Systems, LLC. All Rights Reserved.
#

# REQUIRE bash shell
SHELL:= bash

# system directories
CUDADIR:= /usr/local/cuda

# project directories
BINDIR:= bin
BUILDDIR:= build
SUBDIR:= include
SOURCEDIR:= src
TESTBUILDDIR:= $(BUILDDIR)/test
TESTSOURCEDIR:= $(SOURCEDIR)/test
BINSOURCEDIR:= $(SOURCEDIR)/bin
SUBDIRS := $(wildcard $(SUBDIR)/**)
SUBSOURCEDIRS := $(addsuffix /$(SOURCEDIR),$(SUBDIRS))

# version info
VERSION := $(shell git describe --always --dirty --tags 2>/dev/null)
VERSION := \"$(or $(VERSION),$(shell date +u%g%j))\" # untracked version

# compilers
NVCC := $(if $(NO_CUDA),,$(shell ls $(CUDADIR)/bin/nvcc 2>/dev/null | head -n 1))
GCC := $(shell ls /usr/{,local/}bin/gcc-[0-9]* 2>/dev/null | sort -t- -k2,2V | tail -n 1)
CC := $(or $(GCC),$(CC)) # some systems alias gcc elsewhere

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
# ... collection of object files
OBJECTS := $(COBJS) $(if $(NVCC),$(CUOBJS))

# test coverage file
COVERAGE := $(BUILDDIR)/coverage.info

# library names: submodule, cuda, base
SUBLIBRARIES := $(patsubst $(SUBDIR)/%,%,$(SUBDIRS))
CULIBRARIES := $(if $(NVCC),cudart nvidia-ml)
LIBRARY := $(lastword $(notdir $(realpath .)))

# library files: submodule, base
SUBLIBRARYFILES := $(join $(SUBDIRS),$(patsubst $(SUBDIR)/%,/$(BUILDDIR)/lib%.a,$(SUBDIRS)))
LIBRARYFILE := $(BUILDDIR)/lib$(LIBRARY).a

# library directories: submodule, cuda
SUBLIBRARYDIRS := $(addsuffix /$(BUILDDIR),$(SUBDIRS))
CULIBRARYDIRS := $(if $(NVCC),$(addprefix $(CUDADIR),/lib64 /lib64/stubs))

# include directories: submodule, cuda
SUBINCLUDEDIRS := $(addsuffix /$(SOURCEDIR),$(SUBDIRS))
CUINCLUDEDIRS := $(if $(NVCC),$(CUDADIR)/include)

# linker and compiler flags
NVCFLAGS := -Xptxas -Werror
CFLAGS := -MMD -MP -Wall -Werror -Wextra -Wpedantic -fopenmp -g -rdynamic
DFLAGS := $(addprefix -D,$(DEFINES) VERSION=$(VERSION))
IFLAGS := $(addprefix -I,$(SOURCEDIR) $(CUINCLUDEDIRS) $(SUBINCLUDEDIRS))
LFLAGS := $(addprefix -L,$(BUILDDIR) $(CULIBRARYDIRS) $(SUBLIBRARYDIRS))
lFlags := -Wl,-\( $(addprefix -l,m $(LIBRARY) $(CULIBRARIES) $(SUBLIBRARIES)) -Wl,-\)
# ... working set of flags
CCFLAGS := $(IFLAGS) $(DFLAGS) $(CFLAGS) $(CCARGS)
LDFLAGS := $(LFLAGS) $(DFLAGS) $(CFLAGS) $(CCARGS) $(LDARGS) $(lFlags)
NVCCFLAGS := $(IFLAGS) $(DFLAGS) $(NVCFLAGS) $(NVCCARGS)

################################################################

.SUFFIXES: # disable rules predefined by MAKE
.PHONY: all clean coverage docs help library test

# default rule builds (local) library file containing all objects
all: $(SOURCEDIR) $(SUBLIBRARYFILES) $(LIBRARYFILE)

# remove build directory and files
clean:
	@rm -rf $(BUILDDIR)
	@$(if $(NO_RECURSIVE),,$(foreach DIR,$(SUBDIRS),make clean -C $(DIR);))

# build test coverage (requires lcov); depends on coverage file
coverage: $(COVERAGE)
	genhtml $(COVERAGE) --output-directory $(BUILDDIR)

# build documentation files (requires doxygen)
docs:
	@mkdir -p docs
	@cp .github/docs/config docs/config
	@echo 'PROJECT_NAME = "$(LIBRARY)"' | tr '[:lower:]' '[:upper:]' >>docs/config
	@echo 'PROJECT_NUMBER = "$(VERSION)"' >>docs/config
	@echo 'EXPAND_AS_DEFINED = EMCM__TABLE EMCM__ENUM' >>docs/config
	-doxygen docs/config
	rm docs/config

# echo the value of a variable matching pattern
echo-%:
	@echo $* = $($*)

# display help information
help:
	@echo
	@echo 'Usage:  make [options] [targets]'
	@echo '   		make --help # for make specific options'
	@echo
	@echo 'Options:'
	@echo '   DEFINES="<defs>"'
	@echo '      add preprocessor definitions to the C compiler'
	@echo '      e.g. make all DEFINES="_GNU_SOURCE _XOPEN_SOURCE=600"'
	@echo '   NO_CUDA=1           disable CUDA support'
	@echo '   NO_RECURSIVE=1      disable recursive submodule actions'
	@echo '   NVCCARGS="<flags>"  add compiler args to the NVIDIA compiler'
	@echo '   CCARGS="<flags>"    add compiler args to the C compiler'
	@echo '   LDARGS="<flags>"    add linker args to the C compiler'
	@echo '      e.g. make all CCARGS="-fsanitize=address"'
	@echo
	@echo 'Targets:'
	@echo '   make [all]       build all object files into a library'
	@echo '   make clean       remove build files (incl. within submodules)'
	@echo '   make coverage    build test coverage file and generate report'
	@echo '   make docs        build documentation files'
	@echo '   make echo-*      show the value of a variable matching *'
	@echo '   make help        prints this usage information'
	@echo '   make test        build and run (all) tests'
	@echo '   make test-*      build and run tests matching *'
	@echo
	@echo 'Elevated Targets:'
	@echo '   [sudo] make install'
	@echo '      Install the Mochimo Node to /opt/mochimo/, and create a'
	@echo '      system user 'mcm:mcm' for isolated service operation'
	@echo '   [sudo] make service'
	@echo '      Install the Mochimo Node as a background service'
	@echo '   [sudo] make service-logs'
	@echo '      Obtain Mochimo Node service logs from the latest session'
	@echo '      and save in a service.log file for service log review'
	@echo '   [sudo] make uninstall'
	@echo '      Remove Mochimo installations and services from the system'
	@echo
	@echo 'User Targets:'
	@echo '   make miner       build miner binary and install in bin/'
	@echo '   make mochimo     build mochimo binary and install in bin/'
	@echo

################################################################

# dynamic test objects, names and components; eye candy during tests
TESTOBJECTS:= $(TCOBJS) $(if $(NVCC),$(TCUOBJS))
TESTNAMES:= $(basename $(patsubst $(TESTBUILDDIR)/%,%,$(TESTOBJECTS)))
TESTCOMPS:= $(shell echo $(TESTOBJECTS) | sed 's/\s/\n/g' | \
	sed -E 's/\S*\/([^-]*)[-.]+\S*/\1/g' | sort -u)

# build and run specific tests matching pattern
test-%: $(SUBLIBRARYFILES) $(LIBRARYFILE)
	@echo -e "\n[--------] Performing $(words $(filter $*%,$(TESTNAMES)))" \
		"tests matching \"$*\""
	@$(foreach TEST,\
		$(addprefix $(TESTBUILDDIR)/,$(filter $*%,$(TESTNAMES))),\
		make $(TEST) -s && ( $(TEST) && echo "[ ✔ PASS ] $(TEST)" || \
		( touch $(TEST).fail && echo "[ ✖ FAIL ] $(TEST)" ) \
	 ) || ( touch $(TEST).fail && \ echo "[  ERROR ] $(TEST), ecode=$$?" ); )

# build and run tests
test: $(SUBLIBRARYFILES) $(LIBRARYFILE) $(TESTOBJECTS)
	@if test -d $(BUILDDIR); then find $(BUILDDIR) -name *.fail -delete; fi
	@echo -e "\n[========] Found $(words $(TESTNAMES)) tests" \
		"for $(words $(TESTCOMPS)) components in \"$(LIBRARY)\""
	@echo "[========] Performing all tests in \"$(LIBRARY)\" by component"
	@-$(foreach COMP,$(TESTCOMPS),make test-$(COMP) --no-print-directory; )
	@export FAILS=$$(find $(BUILDDIR) -name *.fail -delete -print | wc -l); \
	 echo -e "\n[========] Testing completed. Analysing results..."; \
	 echo -e "[ PASSED ] $$(($(words $(TESTNAMES))-FAILS)) tests passed."; \
	 echo -e "[ FAILED ] $$FAILS tests failed.\n"; \
	 exit $$FAILS

################################################################

INSTALLDIR := /opt/mochimo
SERVICE := /etc/systemd/system/mochimo.service

$(SERVICE): .github/systemd/mochimo.service
	@cp .github/systemd/mochimo.service $(SERVICE)
	@systemctl enable mochimo.service
	@echo "$(SERVICE) was updated..."

$(BINDIR)/gpuminer: $(BUILDDIR)/bin/gpuminer
	@cp $(BUILDDIR)/bin/gpuminer $(BINDIR)/
	@echo "$(BUILDDIR)/bin/gpuminer was updated..."

$(BINDIR)/mochimo: $(BUILDDIR)/bin/mochimo
	@mkdir -p $(BINDIR)/d/bc
	@mkdir -p $(BINDIR)/d/split
	@cp $(BUILDDIR)/bin/mochimo $(BINDIR)/
	@cp $(SOURCEDIR)/_init/* $(BINDIR)/
	@chmod +x $(BINDIR)/gomochi $(BINDIR)/*-external.sh
	@echo "$(BUILDDIR)/mochimo was updated..."

$(INSTALLDIR)/mochimo: $(BINDIR)/mochimo
	@mkdir -p /opt/mochimo/
	@cp -r $(BINDIR)/* /opt/mochimo/
	@useradd -M -r -s /usr/sbin/nologin mcm || true
	@chown -R mcm:mcm /opt/mochimo/
	@echo "$(INSTALLDIR)/mochimo was updated..."

install: $(INSTALLDIR)/mochimo
	@echo && echo "Mochimo installation directory:"
	@echo "   $(INSTALLDIR)/"
	@echo && echo "... install (done)" && echo

miner: $(BINDIR)/gpuminer
	@echo && echo "... miner (done)" && echo

mochimo: $(BINDIR)/mochimo
	@echo && echo "... mochimo (done)" && echo

package-%:
	@make clean --no-print-directory
	@mkdir $* # bail if exists
	@cp -r .github include src CHANGELOG.md GNUmakefile LICENSE.md $*/
	@rm -r $*/.github/docs $*/.github/media 2>/dev/null
	@rm -r $(addprefix $*/,$(wildcard $(INCLUDEDIR)/**/.git*)) 2>/dev/null
	@rm -r $(addprefix $*/,$(wildcard $(INCLUDEDIR)/**/docs)) 2>/dev/null
	@sed -i 's/<no-ver>/$*/g' $*/GNUmakefile
	@tar -czf $*.tar.gz $*
	@rm -r $*
	@echo && echo "Packaged to $*.tar.gz" && echo

service: $(INSTALLDIR)/mochimo $(SERVICE)
	@echo && echo "Manage the service with:"
	@echo "   [sudo] service mochimo [start|stop|status]"
	@echo && echo "Monitor service logs with:"
	@echo "   [sudo] journalctl -o cat -u mochimo [-f]"
	@echo && echo "... service (done)" && echo

service-logs: /etc/systemd/system/mochimo.service
	@journalctl _SYSTEMD_INVOCATION_ID=`systemctl show -p InvocationID --value "mochimo"` >service.log
	@echo && echo "... service-logs (done)" && echo

uninstall:
	@systemctl stop mochimo.service || true
	@systemctl disable mochimo.service || true
	@rm /etc/systemd/system/mochimo.service || true
	@rm -r /opt/mochimo/ || true
	@userdel mcm || true
	@echo && echo "... uninstall (done)" && echo

################################################################

# build (local) library, within build directory, from dynamic object set
$(LIBRARYFILE): $(OBJECTS)
	@mkdir -p $(dir $@)
	ar rcs $(LIBRARYFILE) $(OBJECTS)

# build coverage file, within out directory
$(COVERAGE):
	@mkdir -p $(dir $@)
	@make clean all NO_CUDA=1 "LCFLAGS=$(LCFLAGS) --coverage -O0"
	lcov -c -i -d $(BUILDDIR) -o $(COVERAGE)_base
	@make test NO_CUDA=1 "LCFLAGS=$(LCFLAGS) --coverage -O0"
	lcov -c -d $(BUILDDIR) -o $(COVERAGE)_test
	lcov -a $(COVERAGE)_base -a $(COVERAGE)_test -o $(COVERAGE) || \
		cp $(COVERAGE)_base $(COVERAGE)
	rm -rf $(COVERAGE)_base $(COVERAGE)_test
	lcov -r $(COVERAGE) '*/$(TESTSOURCEDIR)/*' -o $(COVERAGE)

# build test binaries ONLY [EXPERIMENTAL DEPENDENCY-LESS TEST CASES]
#$(TESTBUILDDIR)/%: $(TESTBUILDDIR)/%.o
#	@mkdir -p $(dir $@)
#	$(CC) $(TESTBUILDDIR)/$*.o -o $@

# build test objects ONLY [EXPERIMENTAL DEPENDENCY-LESS TEST CASES]
#$(TESTBUILDDIR)/%.o: $(TESTSOURCEDIR)/%.c
#	@mkdir -p $(dir $@)
#	$(CC) -c $(TESTSOURCEDIR)/$*.c -o $@ $(CCFLAGS)

# build binaries, within build directory, from associated objects
$(BUILDDIR)/%: $(SUBLIBRARYFILES) $(LIBRARYFILE) $(BUILDDIR)/%.o
	@mkdir -p $(dir $@)
	$(CC) $(BUILDDIR)/$*.o -o $@ $(LDFLAGS)
	@chmod +x $@

# build cuda objects, within build directory, from *.cu files
$(BUILDDIR)/%.cu.o: $(SOURCEDIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) -c $(SOURCEDIR)/$*.cu -o $@ $(NVCCFLAGS)

# build c objects, within build directory, from *.c files
$(BUILDDIR)/%.o: $(SOURCEDIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) -c $(SOURCEDIR)/$*.c -o $@ $(CCFLAGS)

# build submodule libraries: depends on submodule source directories
$(SUBLIBRARYFILES): %: $(SUBSOURCEDIRS)
	@make -C $(SUBDIR)/$(word 2,$(subst /, ,$@))

# initialize submodule source directories
$(SUBSOURCEDIRS): %:
	git config submodule.recurse true
	git submodule update --init --recursive

# include depends rules created during "build object file" process
-include $(patsubst $(SOURCEDIR)/%.c,$(BUILDDIR)/%.d,\
   $(BCSRCS) $(CSRCS) $(TCSRCS) $(TCUSRCS) $(TCLSRCS))
