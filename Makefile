
##
# Makefile - C/C++ makefile for Microsoft NMAKE
# Copyright 2025 Adequate Systems, LLC. All Rights Reserved.
#

# Project directories
BINDIR=bin
BUILDDIR=build
SOURCEDIR=src

EXTCBUILDDIR=build\extended-c
EXTCSOURCEDIR=include\extended-c\src
CRYPTOCBUILDDIR=build\crypto-c
CRYPTOCSOURCEDIR=include\crypto-c\src

# CUDA support - adjust path to match Windows CUDA installation
CUDADIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8

# Version information
VERSION="\"windows-build\""

# Compilers
CC=cl.exe
NVCC="$(CUDADIR)\bin\nvcc.exe"

# Library base name
LIBRARY=mochimo

# (base) Output library and object files
LIBRARYFILE="$(BUILDDIR)\$(LIBRARY).lib"
BASEOBJECTS=\
	$(BUILDDIR)\bcon.obj \
	$(BUILDDIR)\bup.obj \
	$(BUILDDIR)\bval.obj \
	$(BUILDDIR)\device.objcu \
	$(BUILDDIR)\error.obj \
	$(BUILDDIR)\global.obj \
	$(BUILDDIR)\ledger.obj \
	$(BUILDDIR)\network.obj \
	$(BUILDDIR)\peach.objcu \
	$(BUILDDIR)\peach.obj \
	$(BUILDDIR)\peer.obj \
	$(BUILDDIR)\tfile.obj \
	$(BUILDDIR)\trigg.obj \
	$(BUILDDIR)\tx.obj \
	$(BUILDDIR)\wots.obj

# (extended-c) Output library and object files
LIBRARYEXTC="$(BUILDDIR)\extended-c.lib"
EXTCOBJECTS =\
	$(EXTCBUILDDIR)\extinet.obj \
	$(EXTCBUILDDIR)\extio.obj \
	$(EXTCBUILDDIR)\extlib.obj \
	$(EXTCBUILDDIR)\extmath.obj \
	$(EXTCBUILDDIR)\extstring.obj \
	$(EXTCBUILDDIR)\extthrd.obj

LIBRARYCRYC="$(BUILDDIR)\crypto-c.lib"
CRYCOBJECTS=\
	$(CRYPTOCBUILDDIR)\base58.obj \
	$(CRYPTOCBUILDDIR)\blake2b.obj \
	$(CRYPTOCBUILDDIR)\crc16.obj \
	$(CRYPTOCBUILDDIR)\crc32.obj \
	$(CRYPTOCBUILDDIR)\md2.obj \
	$(CRYPTOCBUILDDIR)\md5.obj \
	$(CRYPTOCBUILDDIR)\ripemd160.obj \
	$(CRYPTOCBUILDDIR)\sha1.obj \
	$(CRYPTOCBUILDDIR)\sha256.obj \
	$(CRYPTOCBUILDDIR)\sha256.objcu \
	$(CRYPTOCBUILDDIR)\sha3.obj

# Compiler and linker flags
CFLAGS=/nologo /MP /W4 /WX /DVERSION=$(VERSION) /D_CRT_SECURE_NO_WARNINGS /D_WINSOCK_DEPRECATED_NO_WARNINGS
IFLAGS=/I "$(CUDADIR)\include" /I "include\crypto-c\src" /I "include\extended-c\src" /I "$(SOURCEDIR)"
# Use static OpenMP library with dynamic runtime
LCFLAGS=/openmp:llvm

# Include static CUDA runtime, but keep dynamic C runtime
LDFLAGS=/LIBPATH:"$(CUDADIR)\lib\x64" /LIBPATH:"$(BUILDDIR)" cudart_static.lib

# Use consistent runtime model in NVCC
NVIFLAGS=-I "$(CUDADIR)\include" -I "include\crypto-c\src" -I "include\extended-c\src" -I "$(SOURCEDIR)"
##############################################################

# Default target
all: dirs $(LIBRARYFILE) $(LIBRARYEXTC) $(LIBRARYCRYC)
	@echo ... library (done)

# Create necessary directories
dirs:
	@if not exist bin mkdir bin
	@if not exist build mkdir build
	@if not exist build\bin mkdir build\bin

# Clean build files
clean:
	@if exist build rmdir /S /Q build
	@echo Cleaned build directory

# Build miner binary if CUDA is available
miner: dirs $(BUILDDIR)\bin\gpuminer.exe
	@copy $(BUILDDIR)\bin\gpuminer.exe bin
	@echo ... miner (done)

# Show help information
help:
	@echo.
	@echo Usage: nmake [options] [targets]
	@echo.
	@echo Options:
	@echo   *no currently implemented options*
	@echo.
	@echo Targets:
	@echo   nmake [all]     Build library
	@echo   nmake clean     Remove build files
	@echo   nmake miner     Build miner binary (requires CUDA)
	@echo   nmake help      Show this help information
	@echo.

##############################################################

# Rules for building

.SUFFIXES: .cu .objcu

$(BUILDDIR)\bin\gpuminer.exe: dirs $(LIBRARYFILE) $(LIBRARYCRYC) $(LIBRARYEXTC) $(BUILDDIR)\bin\gpuminer.obj
	@if not exist $(BUILDDIR)\bin mkdir $(BUILDDIR)\bin
	@$(CC) $(BUILDDIR)\bin\gpuminer.obj $(LIBRARYFILE) $(LIBRARYCRYC) $(LIBRARYEXTC) /Fe$(BUILDDIR)\bin\gpuminer.exe /link $(LDFLAGS)

# Build library from object files (base directory)
$(LIBRARYFILE): $(BASEOBJECTS)
	@if not exist $(BUILDDIR) mkdir $(BUILDDIR)
	@lib /OUT:$(LIBRARYFILE) $(BASEOBJECTS)

# Build library from object files (extended-c directory)
$(LIBRARYEXTC): $(EXTCOBJECTS)
	@if not exist $(BUILDDIR) mkdir $(BUILDDIR)
	@if not exist $(EXTCBUILDDIR) mkdir $(EXTCBUILDDIR)
	@lib /OUT:$(LIBRARYEXTC) $(EXTCOBJECTS)

# Build library from object files (crypto-c directory)
$(LIBRARYCRYC): $(CRYCOBJECTS)
	@if not exist $(BUILDDIR) mkdir $(BUILDDIR)
	@if not exist $(CRYPTOCBUILDDIR) mkdir $(CRYPTOCBUILDDIR)
	@lib /OUT:$(LIBRARYCRYC) $(CRYCOBJECTS)

# Compile C sources (base directory)
{$(SOURCEDIR)\bin}.c{$(BUILDDIR)\bin}.obj:
	@if not exist $(BUILDDIR)\bin mkdir $(BUILDDIR)\bin
	@$(CC) /c $(CFLAGS) $(LCFLAGS) $(IFLAGS) /Fo$(BUILDDIR)\bin\ $<

{$(SOURCEDIR)}.c{$(BUILDDIR)}.obj:
	@$(CC) $(CFLAGS) $(LCFLAGS) $(IFLAGS) /Fo$(BUILDDIR)\ /c $<

# Compile CUDA sources
{$(SOURCEDIR)}.cu{$(BUILDDIR)}.objcu:
	@$(NVCC) $(NVIFLAGS) -o $@ -c $<

# Compile C sources (extended-c directory)
{$(EXTCSOURCEDIR)}.c{$(EXTCBUILDDIR)}.obj:
	@if not exist $(EXTCBUILDDIR) mkdir $(EXTCBUILDDIR)
	@$(CC) /c $(CFLAGS) $(LCFLAGS) $(IFLAGS) /Fo$(EXTCBUILDDIR)\ $<

# Compile C sources (crypto-c directory)
{$(CRYPTOCSOURCEDIR)}.c{$(CRYPTOCBUILDDIR)}.obj:
	@if not exist $(CRYPTOCBUILDDIR) mkdir $(CRYPTOCBUILDDIR)
	@$(CC) /c $(CFLAGS) $(LCFLAGS) $(IFLAGS) /Fo$(CRYPTOCBUILDDIR)\ $<

# Compile CUDA sources (crypto-c directory)
{$(CRYPTOCSOURCEDIR)}.cu{$(CRYPTOCBUILDDIR)}.objcu:
	@if not exist $(CRYPTOCBUILDDIR) mkdir $(CRYPTOCBUILDDIR)
	@$(NVCC) $(NVIFLAGS) -o $@ -c $<
