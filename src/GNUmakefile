##
# The Adequate Systems' (mostly) drop-in makefile for building with GNU Make.
#   GNUmakefile (11 August 2021)
#
# Copyright (c) 2021 Adequate Systems, LLC. All Rights Reserved.
#
# For more information, please refer to ../LICENSE
#
## ( *** READ THE GUIDE BELOW BEFORE IMPLEMENTING FOR PRODUCTION USE *** )
# THIS MAKEFILE is mostly drop-in, requiring very little adjustments, if any,
#  before implementing for production use. By default, it handles test cases,
#  coverage reports, library building, and cleanup, without modification when
#  you use the default naming conventions and directory structures. Binaries,
#  however, will require additional recipes be placed at the end of the file.
# PRODUCTION USE of this makefile assumes the developer;
# - has updated the description on the second line of this file, and
# - understands the automation of this makefile, before removing this guide.
# DEFAULT MACROS can be changed within the "Developer Configuration Section";
# - "rm" and "mkdir" commands print to the console for every action taken.
# DEPENDENCY HIERARCHY:  by dependency type (external -> internal -> system);
# - Generally, when a project adopts both external and internal dependencies,
#   the project structure SHOULD BE RECONSIDERED, as this is NOT RECOMMENDED.
# - Dependencies MAY ONLY depend on subsequent dependency types OR, where the
#   subsequent dependency type is the same, dependants SHALL BE listed first.
# - External and internal dependencies SHALL EXIST within SRCDIR or PARENTDIR
#   respectively, and be of the same name as is listed. When dependencies MAY
#   NOT ALWAYS EXIST in their respective locations, recipes SHALL BE PROVIDED
#   to acquire said dependencies to their respective location (i.e. ../<dep>)
#

########################################
## DEVELOPER DEFINITIONS CONFIGURATION
#

# command macros
RM:= rm -vrf
MKDIR:= mkdir -vp
SHELL:= bash

# directory macros
ROOTDIR = ..
BINDIR = $(ROOTDIR)/bin
BUILDDIR = $(ROOTDIR)/build
LIBDIR = $(ROOTDIR)/lib
PARENTDIR = $(ROOTDIR)/..
SRCDIR = $(ROOTDIR)/src
TESTDIR = $(ROOTDIR)/src/test

# library name (by default, the name of the root directory)
LIBRARYNAME:= \
 $(notdir $(realpath $(dir $(lastword $(MAKEFILE_LIST)))$(ROOTDIR)))

# required dependencies
EXTERNALLIBS:= $(LIBRARYNAME)
INTERNALLIBS:=
DEPENDENCIES:= $(EXTERNALLIBS) $(INTERNALLIBS)
SYSTEMLIBS:=

#
## END DEVELOPER DEFINITIONS CONFIGURATION
############################################

# sources, objects, depends and libraries
SOURCES:= $(wildcard $(SRCDIR)/*.c)
OBJECTS:= $(patsubst $(SRCDIR)/%.c,$(BUILDDIR)/%.o,$(SOURCES))
DEPENDS:= $(patsubst $(SRCDIR)/%.c,$(BUILDDIR)/%.d,$(SOURCES))

# dependency paths (for includes and libraries)
INCLUDEPATHS:= \
 $(patsubst %,-I$(PARENTDIR)/%/src,$(EXTERNALLIBS)) \
 $(patsubst %,-I%/src,$(INTERNALLIBS))
LIBRARYPATHS:= \
 $(patsubst %,-L$(PARENTDIR)/%/lib,$(EXTERNALLIBS)) \
 $(patsubst %,-L%/lib,$(INTERNALLIBS))

# compiler macros
WARNINGS:= -Werror -Wall -Wextra
LIBFLAGS:= $(patsubst %,-l%,$(DEPENDENCIES)) $(patsubst %,-l%,$(SYSTEMLIBS))
LDFLAGS:= $(LIBRARYPATHS) $(LIBFLAGS)
CC:= gcc $(WARNINGS) $(CFLAGS) $(INCLUDEPATHS)

# .PHONY defines rules that should execute, even if they exist as files
.PHONY: all clean coverage library test $(DEPENDENCIES)
.SUFFIXES: # disables predefined rules

# default rule: "make all" and "make", makes all objects
all: $(OBJECTS)

# remove bin, build and out directories and their contents
clean:
	$(RM) $(BINFILES) $(BUILDDIR) $(LIBDIR)

# build hash test coverage binary and generate coverage.info file
coverage:
	@make test CFLAGS=-coverage
	@cd $(DIR_BUILD) && gcov -d -o $(DIR_BUILD) -s $(DIR_SRC) $(DIR_BIN)/hash-test
	lcov -c --directory $(DIR_BUILD) --output-file $(DIR_BUILD)/coverage.info
#	genhtml $(DIR_BUILD)/coverage.info --output-directory $(DIR_OUT)

library: $(LIBDIR)/lib$(LIBRARYNAME).a
$(LIBDIR)/lib$(LIBRARYNAME).a: $(OBJECTS) $(LIBDIR)
	ar rcs $(LIBDIR)/lib$(LIBRARYNAME).a $(OBJECTS)

# build and run all tests
test: $(wildcard $(TESTDIR)/*.c)
	@export COMPS=$$(ls $^ | sed -E 's/.*\/([^-]*)-.*/\1/g' | sort -u); \
	 export NCOMPS=$$(wc -l <<< "$$COMPS"); \
	 export NTESTS=$$(wc -w <<< "$^"); \
	 printf "[========] Performing all tests.\n"; \
	 printf "[========] Found $$NTESTS tests for $$NCOMPS components.\n"; \
	 for COMP in $$COMPS; do make test/$$COMP --no-print-directory; done; \
	 export FAILS=$$(ls -l $(BUILDDIR)/*.fail 2>/dev/null | wc -l); \
	 printf "[ FINISH ] Completed $$NTESTS tests for $$NCOMPS components.\n"; \
	 printf "[ PASSED ] $$((NTESTS-FAILS)) tests passed.\n"; \
	 printf "[ FAILED ] $$FAILS tests failed.\n"; \
	 exit $$FAILS;

# build and run specific tests
test/%: $(wildcard $(TESTDIR)/%*)
	@export TIMEFORMAT=" (%Es)"; \
	 export TESTS=$$(ls $(TESTDIR)/$** | sed -E 's/\S*\/(.*)\.c/test-\1/g'); \
	 export NUMTESTS=$$(wc -l <<< "$$TESTS"); \
	 printf "[--------] Remove previous test failures.\n"; \
	 $(RM) $(BUILDDIR)/test-$**.fail; \
	 printf "[--------] Making $$NUMTESTS tests matching \"$*\".\n"; \
	 for TEST in $$TESTS; do printf "[ MAKE   ] $$TEST...\n"; \
		time { make $(BUILDDIR)/$$TEST -s && printf "[   DONE ] $$TEST" || \
		 printf "[  ERROR ] $$TEST"; }; \
	 done; \
	 printf "[--------] Running $$NUMTESTS tests matching \"$*\".\n"; \
	 for TEST in $$TESTS; do printf "[ RUN    ] $$TEST...\n"; \
		time { $(BUILDDIR)/$$TEST && printf "[   PASS ] $$TEST" || \
		 ( touch $(BUILDDIR)/$$TEST.fail && printf "[   FAIL ] $$TEST" ); }; \
	 done;

# build test binaries, within build directory, from associated objects
$(BUILDDIR)/test-%: $(BUILDDIR)/test-%.o $(BUILDDIR) $(DEPENDENCIES)
	$(CC) $< -o $@ $(LDFLAGS)

# build external dependency libraries, within their respective directories
$(EXTERNALLIBS): %: $(PARENTDIR)/%/src
	@make library -C $<

# build internal dependency libraries, within their respective directories
$(INTERNALLIBS): %: %/src
	@make library -C $<

# build test object files, within build directory
test-%.o: $(BUILDDIR)/test-%.o # redirect
$(BUILDDIR)/test-%.o: $(TESTDIR)/%.c $(SRCDIR)/GNUmakefile $(BUILDDIR)
	$(CC) -MMD -MP -c $< -o $@

# build object file, within build directory
%.o: $(BUILDDIR)/%.o # redirect
	@echo -e "  $@ can be found at $<\n"
$(BUILDDIR)/%.o: $(SRCDIR)/%.c $(SRCDIR)/GNUmakefile $(BUILDDIR)
	$(CC) -MMD -MP -c $< -o $@

# include depends rules created during "build object file" process
-include $(DEPENDS)

# directory creation
$(BINDIR):
	@$(MKDIR) $(BINDIR)
$(BUILDDIR):
	@$(MKDIR) $(BUILDDIR)
$(LIBDIR):
	@$(MKDIR) $(LIBDIR)

####################################
## DEVELOPER RECIPES CONFIGURATION
#

help:
	@echo "Usage:"
	@echo " make                 build all objects"
	@echo " make <component>.o   build <component> object"
	@echo " make library         build a library file containing all objects"
	@echo " make test            build and run all tests"
	@echo " make test/<test>     build and run tests matching <test>*"
	@echo " make coverage        perform test coverage analysis"
	@echo " make clean           removes build directories"
	@echo " make help            prints this usage information"
	@echo ""
