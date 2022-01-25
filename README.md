<h1 align="center">
   <a href="http://adequate.biz">
      <img alt="Adequate Systems" src="https://raw.githubusercontent.com/adequatesystems/.github/main/media/adqlogo_banner.svg" /></a>
   <br/>C/C++ Build and Workflow Support<br/>
</h1>

This repository is designed for inclusion in C/C++ projects for easy build and workflow integration, intended for use in ongoing projects at Adequate Systems, LLC.

## Instructions
**NOTE: This repository contains it's own LICENSE.md and README.md files that may introduce artifacts in existing files of the target repostiory. Please ensure the LICENSE.md and README.md files are appropriate and/or correct for your target repostiory pulling this repository.**

To include or update, it makes no difference; add this repository as a remote and pull the latest (or desired) revision, specifying to "allow unrelated histories"...<br/>
<sup><i>Adding the repository as a remote may be omitted if previously performed</i></sup>
```sh
git remote add build-c https://github.com/adequatesystems/build-c.git
git pull build-c main --allow-unrelated-histories
```
Fix merge conflicts and commit with...<br/>
<sup><i>Commit may also be performed selectively</i></sup>
```sh
git commit -am "merge latest build-c repository files"
```

## Typical active project directory structure
```diff
# Base files
+ Manually managed
- Automatically managed
! Automatically generated only

# base-c
# ├── .github
# │   ├── docs
# │   │   ├── .nojekyll
# │   │   ├── config
# │   │   ├── layout.xml
# │   │   └── style.css
# │   └── workflows
# │       ├── codeql.yaml
# │       └── tests.yaml
- ├── build
- │   ├── test
- │   │   ├── sourcetest.d
- │   │   ├── sourcetest.o
- │   │   └── sourcetest
- │   ├── source
- │   ├── source.d
- │   └── source.o
! ├── docs
! │   └── htmlfiles...
! ├── include
! │   └── submoduledirs...
+ ├── src
+ │   ├── test
+ │   │   └── sourcetest.c
+ │   ├── source.c
+ │   └── source.h
# ├── .gitignore
# ├── GNUmakefile
# ├── LICENSE.md
# └── README.md
```

## Makefile usage
CLI usage information is revealed with the use of `make` or `make help` in the project's root directory.
```
Usage:  make [options] [FLAGS=FLAGVALUES]
   make               prints this usage information
   make all           build all object files
   make clean         removes build directory and files
   make cleanall      removes (all) build directories and files
   make coverage      build test coverage file
   make cuda          build cuda compatible object files
   make docs          build documentation files
   make library       build a library file containing all objects
   make libraries     build all library files required for binaries
   make report        build html report from test coverage
   make test          build and run all tests
   make test-<test>   build and run tests matching <test>*
```

## Configurable Flags
Most parameters used by the Makefile can be configured, either directly in the Makefile itself or on the command line by appending the flag and its value to the call to `make`. For a complete list of FLAGS it is recommended to peruse the GNUmakefile source.

## Integrated Documentation
*Requires at least `doxygen` v1.9.x (unavailable through `apt` on Ubuntu 20.04)*

C/C++ Documentation is made available with the help of Doxygen, using special comment style blocks before functions and definitions for automatic recognition and compilation into an easy to navigate html documentation front-end. 

Use:
* `make docs` after the code is commented appropriately

## Integrated Testing
Test files should be placed in the `src/test/` directory as `*.c` source files. Every test file will be compiled and run as it's own binary. So whether a test is broad or specific, a single test file can only count as a single failure.

Use:
* `make test` to run all tests, OR
* `make test-<pattern>` to run all tests matching `pattern*`

## Test coverage (local)
<sup><i>Note: Local test coverage may be incomplete if tests fail</i></sup>

Test coverage can be generated locally and viewed via a HTML report (generated separately). `lcov` is required to generate coverage data.

Use:
* `make coverage` to generate coverage data, AND
* `make report` to generate html report from coverage data, OR
* `make coverage report` to do both in one command

## Test coverage (workflow)
Test coverage is also generated automatically by the `tests.yaml` github workflow and automatically uploaded to [CodeCov.io](https://about.codecov.io/) upon success, **provided that all tests pass.**

**In some circumstances test coverage of a brand new repository will fail, specifying that the repository cannot be found.** Some causes of this error include:
* being too deadly;
  * I think there is some delay between creating/pushing to a repository and the repository being detectable by CodeCov. In this case, you can simply "re-run" the GitHub Action jobs after some time and it will pass ok.
* repository is private;
  * If the repository is private, one would normally question the necessity of coverage data, and recommend that the "coverage job" be removed from the `tests.yaml` workflow file. However, if coverage data is deemed necessary, you will need to obtain a CodeCov token from the website for your new repository, add it as a GitHub repository "secret" and include it in the `tests.yaml` workflow file.

**On the rare chance that test coverage remains in a failed state,** you may need to manually "activate" a repository on the CodeCov dashboard (website) and "re-run" all jobs again.

## Submodule support
Support for submodules is automatically built into the Makefile, provided that:
* the submodules use the same build and workflow structure
* the submodules are added to the `include/` directory

### Add a submodule
Adding a submodule can be done as part of a larger commit if desired.
```sh
cd project-repo
git submodule add <submodule-repo> include/<submodule-name>
git commit -m "add submodule to repostory"
```

### Update a submodule
**Operating the Makefile between any of the steps for updating a submodule may result in a misconfigured submodule.** It is recommended to complete steps below before operating the makefile. Updating a submodule can be done as part of a larger commit if desired.
```sh
cd project-repo
git -C include/<submodule-name> pull origin main
git commit -m "update submodule to latest revision"
```

## CUDA support
CUDA compilation of `*.c` source files is enabled by the Makefile for systems with appropriately installed CUDA Toolkit. The Makefile uses the NVCC compiler in place of the normal compiler (normally gcc) to compile identical object files. By default, the NVCC compiler is assumed to be accessible at the standard cuda toolkit install location `/usr/local/cuda/bin/nvcc`, however this is configurable via the command line using the `NVCC` flag like so:
```sh
make build/source.o NVCC=/path/to/nvcc
```

Use:
* `make cuda` to build all objects with NVCC

## License
This repository is licensed for use under an MPL 2.0 derivative Open Source license.<br/>
The community is free to develop and change the code with the caveat that any changes must be for the benefit of the Mochimo cryptocurrency network (with a number of exclusions).<br/>
Please read the LICENSE for more details on limitations and restrictions.
