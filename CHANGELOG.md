# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.2] - 2022-04-13
Documentation configuration adjustments, fixes to the compilation of cuda code for inclusion in parent projects, and CUDA compilation detection when "CFLAGS=-DCUDA" is specified AND `*.cu` files exist.

## Added
- Detection of CUDA files AND the CUDA definition in CFLAGS
- Documentation input exclusions for src/test/ directory and CHANGELOG.md

### Changed
At compile time:
- GNUmakefile checks for the existance of `*.cu` files AND `CFLAGS=-DCUDA`
- NVCC compiles Relocatable Device Code from `*.cu` into `*.cu.o` files
- Relocatable Device Code files are included in the module's library file
At link time:
- NVCC links Relocatable Device Code into Executable Device Code as `culink.o`
- CC links `culink.o` as an additional input file at link time


## [1.1.1] - 2022-04-06
Visual fixes and change of tag scheme to `build-c-<version>` to avoid conflict with dependency projects version numbers.

## [1.1.0] - 2022-04-06
CUDA Device compilation and testing support.

### Added
- Automatic CUDA device compilation, triggered by `CFLAGS=-DCUDA`
- Additional GNUmakefile recipes targeted at CUDA-only opertations
- Auto-recognised file extensions for targeted compilation and testing
- VERSION file, supporting auto-documentation version numbers
- CHANGELOG.md for chronological changes to the repository
- (re)Integrate testing assertion header as a testing utility

## [1.0.0] - 2022-01-26
Initial repository release.

### Added
- Automatic build utility GNUmakefile
- Automatic CI/CD workflows for testing, coverage reports and code quality
- Automatic documentation generation via doxygen configuration

[Unreleased]: https://github.com/adequatesystems/build-c/compare/build-c-1.1.2...HEAD
[1.1.2]: https://github.com/adequatesystems/build-c/compare/build-c-1.1.1...build-c-1.1.2
[1.1.1]: https://github.com/adequatesystems/build-c/compare/build-c-1.1.0...build-c-1.1.1
[1.1.0]: https://github.com/adequatesystems/build-c/compare/build-c-1.0.0...build-c-1.1.0
[1.0.0]: https://github.com/adequatesystems/build-c/releases/tag/build-c-1.0.0
