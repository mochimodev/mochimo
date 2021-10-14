##
# GNUmakefile - Adequate Systems' (mostly) drop-in GNUmakefile for C.
#
# Copyright (c) 2021 Adequate Systems, LLC. All Rights Reserved.
# For more information, please refer to ../LICENSE
#
# Date: 11 August 2021
# Revised: 15 October 2021
#
##
# NOTES: PLEASE READ AND DELETE THE FOLLOWING NOTES BEFORE PRODUCTION USE !!!
#
# Before PRODUCTION USE of this makefile, be sure to:
# - update the GNUmakefile's header information, and
# - update the developer configuration section, as appropriate.
#
# THIS MAKEFILE is mostly drop-in, requiring very little adjustments.
# DEFAULT MACROS can be changed within the "Developer Configuration Section";
#
# Coverage integration was implemented using a combination of guides:
# https://about.codecov.io/blog/how-to-set-up-codecov-with-c-and-github-actions/
# https://wiki.documentfoundation.org/Development/Lcov
#


###############################
# vv DEVELOPER CONFIGURATION vv

SHELL:= bash

# command macros (verbose by default)
RM:= rm -rf
MKDIR:= mkdir -p

# directory macros
ROOTDIR = ..
BINDIR = $(ROOTDIR)/bin
BUILDDIR = $(ROOTDIR)/build
LIBDIR = $(ROOTDIR)/lib
OUTDIR = $(ROOTDIR)/out
PARENTDIR = $(ROOTDIR)/..
SRCDIR = $(ROOTDIR)/src

# module name (by default, the name of the root directory)
# NOTE: elaborate makefile commands, for embedded make calls
MODULE:= $(notdir $(realpath $(dir $(lastword $(MAKEFILE_LIST)))$(ROOTDIR)))
LIBRARY:= $(LIBDIR)/lib$(MODULE).a

# test sources, objects, depends, infos, names and components
TESTSOURCES:= $(sort $(wildcard $(SRCDIR)/test/*.c))
TESTOBJECTS:= $(patsubst $(SRCDIR)/%.c,$(BUILDDIR)/%.o,$(TESTSOURCES))
TESTDEPENDS:= $(patsubst $(SRCDIR)/%.c,$(BUILDDIR)/%.d,$(TESTSOURCES))
TESTNAMES:= $(patsubst $(SRCDIR)/test/%.c,%,$(TESTSOURCES))
TESTCOMPS:= $(shell echo $(TESTSOURCES) | sed 's/\s/\n/g' | \
	sed -E 's/\S*\/([^-]*)[-.]+\S*/\1/g' | sort -u)
# base sources, objects and depends
BASESOURCES:= $(sort $(wildcard $(SRCDIR)/*.c))
BASEOBJECTS:= $(patsubst $(SRCDIR)/%.c,$(BUILDDIR)/%.o,$(BASESOURCES))
BASEDEPENDS:= $(patsubst $(SRCDIR)/%.c,$(BUILDDIR)/%.d,$(BASESOURCES))
# all sources, objects and depends
SOURCES:= $(BASESOURCES) $(TESTSOURCES)
OBJECTS:= $(BASEOBJECTS) $(TESTOBJECTS)
DEPENDS:= $(BASEDEPENDS) $(TESTDEPENDS)

# binaries and associated names
BINARIES:= $(patsubst $(SRCDIR)/%.c,$(BUILDDIR)/%,$(TESTSOURCES))
BINARYNAMES:= $(patsubst $(BUILDDIR)/%,%,$(BINARIES))

# library and include dependencies
DEPENDENCIES:= # e.g. ../../path/to/include/dirs
INCLUDEDIRS:= # e.g. ../../path/to/library/libfile.a
LIBRARYDIRS:= $(LIBDIR) $(dir $(DEPENDENCIES))
LIBRARYFLAGS:= -l$(MODULE) $(patsubst lib%.a,-l%,$(notdir $(DEPENDENCIES))) -lm

# compiler macros
LDFLAGS:= $(addprefix -L,$(LIBRARYDIRS)) $(LIBRARYFLAGS)
CCFLAGS:= -Werror -Wall -Wextra $(addprefix -I,$(INCLUDEDIRS))
CC:= gcc $(CFLAGS) # CFLAGS is reserved for the user; make "CFLAGS=-DDEBUG -O3"

## ^^ END DEVELOPER CONFIGURATION ^^
####################################

.SUFFIXES: # disables predefined rules
.PHONY: all clean coverage library test help # ALWAYS execute

# DEFAULT RULE: build all objects, within build directory
all: $(BASEOBJECTS)

# remove build, lib and out directories
clean:
	@$(RM) $(BUILDDIR) $(INCLUDEDIR) $(LIBDIR) $(OUTDIR)
	@$(foreach DIR,$(INCLUDEDIRS),if test $(DEPTH) -gt 0; then \
		make clean -C $(DIR) DEPTH=$$(($(DEPTH) - 1)); fi; )

# build test coverage
coverage:
	@make clean all --no-print-directory "CFLAGS=--coverage -O0"
	@lcov -c -i -d $(BUILDDIR) -o coverage_base.info
	@make clean test --no-print-directory "CFLAGS=--coverage -O0"
	@lcov -c -d $(BUILDDIR) -o coverage_test.info
	@make clean --no-print-directory
	@lcov -a coverage_base.info -a coverage_test.info -o coverage.info || \
		cp coverage_base.info coverage.info
	@lcov -r coverage.info '*/test/*' -o coverage.info
	@$(RM) coverage_base.info coverage_test.info
	@$(foreach DIR,$(INCLUDEDIRS),if test $(DEPTH) -gt 0; then \
		make coverage -C $(DIR) DEPTH=$$(($(DEPTH) - 1)); fi; )

# build module library, within lib directory
library:
	@make $(LIBRARY) --no-print-directory

# build and run all tests
test: $(LIBRARY) $(DEPENDENCIES)
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
test-%: $(LIBRARY) $(DEPENDENCIES)
	@echo -e "\n[--------] Performing $(words $(filter $*%,$(TESTNAMES)))" \
		"tests matching \"$*\""
	@$(foreach TEST,$(addprefix $(BUILDDIR)/test/,$(filter $*%,$(TESTNAMES))), \
	 make $(TEST) -s && ( $(TEST) && echo "[ ✔ PASS ] $(TEST)" || \
		( touch $(TEST).fail && echo "[ ✖ FAIL ] $(TEST)" ) \
	 ) || echo "[  ERROR ] Could not make $(TEST), ecode=$$?"; )

# build module library, within lib directory
lib$(MODULE).a: $(BASEOBJECTS) # redirect
	@make $(LIBRARY) --no-print-directory
	@echo -e "\n  NOTE: $@ can be found at $(LIBDIR)/$@\n"
$(LIBRARY): $(BASEOBJECTS)
	@$(MKDIR) $(dir $@)
	ar rcs $(LIBRARY) $(BASEOBJECTS)

# build binaries, within build directory, from associated objects
$(BINARYNAMES): %: # redirect
	@make $(BUILDDIR)/$@ --no-print-directory
	@echo -e "\n  NOTE: $@ can be found at $(BUILDDIR)/$@\n"
$(BINARIES): $(BUILDDIR)/%: $(BUILDDIR)/%.o $(LIBRARY) $(DEPENDENCIES)
	@$(MKDIR) $(dir $@)
	$(CC) $< -o $@ $(LDFLAGS)

# build object files, within build directory
%.o: %.c $(SRCDIR)/GNUmakefile # redirect
	@make $(BUILDDIR)/$@ --no-print-directory
	@echo -e "\n  NOTE: $@ can be found at $(BUILDDIR)/$@\n"
$(OBJECTS): $(BUILDDIR)/%.o: $(SRCDIR)/%.c $(SRCDIR)/GNUmakefile
	@$(MKDIR) $(dir $@)
	$(CC) -MMD -MP -c $(abspath $<) -o $@ $(CCFLAGS)

# include depends rules created during "build object file" process
-include $(DEPENDS)

#######################################
# vv DEVELOPER RECIPES CONFIGURATION vv

help:
	@echo "Usage:"
	@echo " make               	build all objects"
	@echo " make <file>.o      	build individual objects"
	@echo " make clean         	removes build directories"
	@echo " make covempty      	build all (empty) coverage files"
	@echo " make coverage      	build all coverage files"
	@echo " make library       	build a library file containing all objects"
	@echo " make test          	build and run all tests"
	@echo " make test-<test>   	build and run tests matching <test>*"
	@echo " make help          	prints this usage information"
	@echo ""

# EXAMPLE Dependency Generation recipe
#../../path/to/library/libfile.a:
#	@make library -C ../../path/to/library/src
