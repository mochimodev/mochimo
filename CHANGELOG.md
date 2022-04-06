# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/olivierlacan/keep-a-changelog/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/olivierlacan/keep-a-changelog/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/olivierlacan/keep-a-changelog/releases/tag/v1.0.0
