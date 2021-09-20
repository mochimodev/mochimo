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
# vv OPTIONAL DEVELOPER CONFIGURATION vv

SHELL:= bash

# command macros (verbose by default)
RM:= rm -vrf
MKDIR:= mkdir -vp

# directory macros
ROOTDIR = ..
BINDIR = $(ROOTDIR)/bin
BUILDDIR = $(ROOTDIR)/build
LIBDIR = $(ROOTDIR)/lib
OUTDIR = $(ROOTDIR)/out
PARENTDIR = $(ROOTDIR)/..
SRCDIR = $(ROOTDIR)/src

# file macros
COVERAGEFILE:= $(OUTDIR)/coverage.info

# ^^ END OPTIONAL DEVELOPER CONFIGURATION ^^
############################################

# module name (by default, the name of the root directory)
MODULE:= $(notdir $(realpath $(dir $(lastword $(MAKEFILE_LIST)))$(ROOTDIR)))

# test sources, objects, depends, infos, names and components
TESTSOURCES:= $(sort $(wildcard $(SRCDIR)/test/*.c))
TESTOBJECTS:= $(patsubst $(SRCDIR)/%.c,$(BUILDDIR)/%.o,$(TESTSOURCES))
TESTDEPENDS:= $(patsubst $(SRCDIR)/%.c,$(BUILDDIR)/%.d,$(TESTSOURCES))
TESTINFOS:= $(patsubst $(SRCDIR)/%.c,$(BUILDDIR)/%.info,$(TESTSOURCES))
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

#########################################
## vv REQUIRED DEVELOPER CONFIGURATION vv

# binaries
BINARIES:= $(patsubst $(SRCDIR)/%.c,$(BUILDDIR)/%,$(TESTSOURCES))

# external, internal and system libraries required to compile binaries
EXTERNALLIBS:= # e.g. "math64" for "$(PARENTDIR)/math64" library al "-lmath64"
INTERNALLIBS:= # e.g. "math64" for "$(SRCDIR)/math64" library as "-lmath64"
SYSTEMLIBS:= # e.g. "m" for gcc math library as "-lm"

## ^^ END REQUIRED DEVELOPER CONFIGURATION ^^
#############################################

# dependencies and associated paths (for includes and libraries)
DEPENDENCIES:= $(MODULE) $(EXTERNALLIBS) $(INTERNALLIBS)
INCLUDEPATHS:= -I$(SRCDIR) \
 $(patsubst %,-I$(PARENTDIR)/%/src,$(EXTERNALLIBS)) \
 $(patsubst %,-I$(SRCDIR)/%/src,$(INTERNALLIBS))
LIBRARYPATHS:= -L$(LIBDIR) \
 $(patsubst %,-L$(PARENTDIR)/%/lib,$(EXTERNALLIBS)) \
 $(patsubst %,-L$(SRCDIR)/%/lib,$(INTERNALLIBS))

# compiler macros
WARNINGS:= -Werror -Wall -Wextra
LIBFLAGS:= $(patsubst %,-l%,$(DEPENDENCIES)) $(patsubst %,-l%,$(SYSTEMLIBS))
LDFLAGS:= $(LIBRARYPATHS) $(LIBFLAGS)
CC:= gcc $(WARNINGS) $(CFLAGS) $(INCLUDEPATHS)

# .PHONY defines rules that should execute, even if they exist as files
.PHONY: library clean coverage report test
.SUFFIXES: # disables predefined rules

# first rule is default: runs on "make" command
library: # redirect
	@make $(LIBDIR)/lib$(MODULE).a --no-print-directory
$(MODULE): # redirect
	@make $(LIBDIR)/lib$(MODULE).a --no-print-directory
$(LIBDIR)/lib$(MODULE).a: $(BASEOBJECTS) $(LIBDIR)
	ar rcs $(LIBDIR)/lib$(MODULE).a $(BASEOBJECTS)

# remove build, lib and out directories
clean:
	@$(RM) $(BUILDDIR) $(LIBDIR) $(OUTDIR)

# build all test coverage reports
coverage:
	@echo "[========] Generating test coverage reports for \"$(MODULE)\"..."
	@$(foreach INFO,$(sort $(TESTINFOS)),make $(INFO) --no-print-directory; )

# combine all test coverage reports
report: $(OUTDIR)
	@echo "[========] Combining coverage reports..."
	@lcov $$(find $(BUILDDIR) -name *.info -printf " -a %p") -o $(COVERAGEFILE)
#	@genhtml $(COVERAGEFILE) -o $(OUTDIR)

# build and run all tests
test: $(OBJECTS) $(DEPENDENCIES) $(BUILDDIR)
	@echo "[--------] Cleanup previous failures..."
	@find $(BUILDDIR) -name *.fail -delete
	@echo "[========] Found $(words $(TESTNAMES)) tests" \
		"for $(words $(TESTCOMPS)) components in \"$(MODULE)\"."
	@echo "[========] Perform all tests in \"$(MODULE)\" by component."
	@$(foreach COMP,$(TESTCOMPS),make test-$(COMP) --no-print-directory; )
	@export FAILS=$$(find $(BUILDDIR) -name *.fail -delete -print | wc -l); \
	 echo -e "\n[========] Testing completed. Analysing results..."; \
	 echo -e "[ PASSED ] $$(($(words $(TESTNAMES))-FAILS)) tests passed."; \
	 echo -e "[ FAILED ] $$FAILS tests failed.\n"; \
	 exit $$FAILS

# build and run specific tests matching pattern
test-%:
	@export TIMEFORMAT=" (%Es)"; \
	 export TESTS="$(addprefix $(BUILDDIR)/test/,$(filter $*%,$(TESTNAMES)))"; \
	 export NTESTS="$(words $(filter $*%,$(TESTNAMES)))"; \
	 echo "[--------] Performing $$NTESTS tests matching \"$*\"."; \
	 for TEST in $$TESTS; do \
		printf "[ MAKING ] $$TEST..." && make $$TEST -s && \
		printf "\33[2K\r[ RUNING ] $$TEST..." && \
		time { $$TEST && printf "\33[2K\r[ ✔ PASS ] $$TEST" || \
			( touch $$TEST.fail && printf "\33[2K\r[ ✖ FAIL ] $$TEST" ); } || \
		printf "\33[2K\r[  ERROR ] Could not test $$TEST, ecode=$$?\n"; \
	 done;

# build individual test coverage reports, within out directory
%.info: %.c # redirect
	@make $(BUILDDIR)/$@ --no-print-directory && \
		echo -e "\n  NOTE: $@ can be found at $(BUILDDIR)/$@\n"
$(TESTINFOS): $(BUILDDIR)/%.info: $(SRCDIR)/%.c
	@$(MKDIR) $(dir $@)
	@echo "[--------] Clean directories of interfering coverage data..."
	@find $(BUILDDIR) -type f -name *.gcda -delete -print
	@echo "[--------] Making coverage test for \"$*\"."
	@make $(BUILDDIR)/$* --no-print-directory CFLAGS=--coverage
	@echo "[--------] Running coverage test for \"$*\"."
	@$(BUILDDIR)/$*
	@echo "[--------] Building coverage report for \"$*\"."
	@cd $(BUILDDIR) && gcov -d -o $(dir $@) -s $(SRCDIR) $(BUILDDIR)/$*
	@lcov -c --directory $(BUILDDIR) --output-file $(BUILDDIR)/$*.info

# build binaries, within build directory, from associated objects
$(patsubst $(BUILDDIR)/%,%,$(BINARIES)): %: # redirect
	@make $(BUILDDIR)/$@ --no-print-directory && \
		echo -e "\n  NOTE: $@ can be found at $(BUILDDIR)/$@\n"
$(BINARIES): $(BUILDDIR)/%: $(BUILDDIR)/%.o $(DEPENDENCIES)
	@$(MKDIR) $(dir $@)
	$(CC) $< -o $@ $(LDFLAGS)

# build dependency libraries, within their respective directories
$(EXTERNALLIBS): %: $(PARENTDIR)/%/src
	@make library -C $<
$(INTERNALLIBS): %: $(SRCDIR)/%/src
	@make library -C $<

# build object files, within build directory
%.o: %.c # redirect
	@make $(BUILDDIR)/$@ --no-print-directory && \
		echo -e "\n  NOTE: $@ can be found at $(BUILDDIR)/$@\n"
$(OBJECTS): $(BUILDDIR)/%.o: $(SRCDIR)/%.c $(SRCDIR)/GNUmakefile
	@$(MKDIR) $(dir $@)
	$(CC) -MMD -MP -c $< -o $@

# include depends rules created during "build object file" process
-include $(DEPENDS)

# directory creation
$(BINDIR):
	$(MKDIR) $(BINDIR)
$(BUILDDIR):
	$(MKDIR) $(BUILDDIR)
$(LIBDIR):
	$(MKDIR) $(LIBDIR)
$(OUTDIR):
	$(MKDIR) $(OUTDIR)

####################################
## DEVELOPER RECIPES CONFIGURATION
#

help:
	@echo "Usage:"
	@echo " make                 build a library file containing all objects"
	@echo " make <component>.o   build <component> object"
	@echo " make test            build and run all tests"
	@echo " make test-<test>     build and run tests matching <test>*"
	@echo " make <test>.info     build coverage report for <test>"
	@echo " make coverage        build coverage reports for all tests"
	@echo " make report          combine coverage reports into single file"
	@echo " make help            prints this usage information"
	@echo " make clean           removes build directories"
	@echo ""
