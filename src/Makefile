# Makefile for IDE integration
# To build without CUDA support, use:
# make COMMAND=bin-without-cuda

COMMAND?=bin
all:
	./makeunx $(COMMAND)
	./makeunx install
debug:
	./makeunx $(COMMAND) -g
	./makeunx install
clean:
	./makeunx clean
