###############################################
###### TO BE USED FOR DEBUG PURPOSE ONLY ######
######   NEEDS UPDATE FOR V2.4 SUPPORT!  ######
###############################################

SHELL := /bin/bash
CC=gcc -DUNIXLIKE -DLONG64 -I/usr/lib -L/usr/local/cuda/lib64
CFLAGS=-W -Wall -ansi -pedantic -g
NVCC=/usr/local/cuda/bin/nvcc -DLONG64 --generate-code arch="compute_37,code=\"sm_37,compute_37\""
LDFLAGS=-lcudart -lcuda -lstdc++
EXEC=mochimo bval bcon bup sortlt neogen wallet minertest
SOURCE_FOLDER=./
BIN_FOLDER=../bin/
OBJ_FOLDER=obj/Debug

all: $(EXEC)
debug: $(EXEC)

sha256.o: $(SOURCE_FOLDER)crypto/sha256.c
	$(CC) -g -c $(SOURCE_FOLDER)crypto/sha256.c -o $(SOURCE_FOLDER)crypto/sha256.o $(LDFLAGS)

blake2b-ref.o: $(SOURCE_FOLDER)crypto/blake2/blake2b-ref.c
	$(CC) -g -c $(SOURCE_FOLDER)crypto/blake2/blake2b-ref.c -o $(SOURCE_FOLDER)crypto/blake2b-ref.o $(LDFLAGS)

blake2s-ref.o: $(SOURCE_FOLDER)crypto/blake2/blake2s-ref.c
	$(CC) -g -c $(SOURCE_FOLDER)crypto/blake2/blake2s-ref.c -o $(SOURCE_FOLDER)crypto/blake2/blake2s-ref.o $(LDFLAGS)

sha3.o: $(SOURCE_FOLDER)crypto/sha3/sha3.c
	$(CC) -g -c $(SOURCE_FOLDER)crypto/sha3/sha3.c -o $(SOURCE_FOLDER)crypto/sha3/sha3.o $(LDFLAGS)

md5.o: $(SOURCE_FOLDER)crypto/md5/md5.c
	$(CC) -g -c $(SOURCE_FOLDER)crypto/md5/md5.c -o $(SOURCE_FOLDER)crypto/md5/md5.o $(LDFLAGS)
	
sha1.o: $(SOURCE_FOLDER)crypto/sha1/sha1.c
	$(CC) -g -c $(SOURCE_FOLDER)crypto/sha1/sha1.c -o $(SOURCE_FOLDER)crypto/sha1/sha1.o $(LDFLAGS)
	
wotshash.o: $(SOURCE_FOLDER)wots/wotshash.c
	$(CC) -g -o $(SOURCE_FOLDER)crypto/wots/wotshash.o -c $(SOURCE_FOLDER)crypto/wots/wotshash.c $(LDFLAGS)
	
wots.o: $(SOURCE_FOLDER)crypto/wots/wots.c
	$(CC) -g -c $(SOURCE_FOLDER)crypto/wots/wots.c -o $(SOURCE_FOLDER)crypto/wots/wots.o $(LDFLAGS)

cuda_trigg.o:
	$(NVCC) -g -c $(SOURCE_FOLDER)algo/trigg/cuda_trigg.cu -o $(SOURCE_FOLDER)algo/trigg/cuda_trigg.o $(LDFLAGS)

trigg.o: $(SOURCE_FOLDER)algo/trigg/trigg.c
	$(CC) -g -c $(SOURCE_FOLDER)algo/trigg/trigg.c -o $(SOURCE_FOLDER)algo/trigg/trigg.o $(LDFLAGS)
	
mochimo: wots.o trigg.o sha256.o blake2b-ref.o blake2s-ref.o sha3.o md5.o sha1.o
	$(CC) -g -o $(BIN_FOLDER)mochimo $(SOURCE_FOLDER)mochimo.c $(SOURCE_FOLDER)algo/trigg/trigg.o $(SOURCE_FOLDER)crypto/wots/wots.o $(SOURCE_FOLDER)crypto/sha256.o $(LDFLAGS)
	
cuda_mochimo: wots.o cuda_trigg.o trigg.o sha256.o
	$(CC) -g -o $(BIN_FOLDER)mochimo $(SOURCE_FOLDER)mochimo.c $(SOURCE_FOLDER)algo/trigg/trigg.o $(SOURCE_FOLDER)algo/trigg/cuda_trigg.o $(SOURCE_FOLDER)crypto/wots/wots.o $(SOURCE_FOLDER)crypto/sha256.o $(LDFLAGS)
	
wallet: wots.o sha256.o
	$(CC) -g -o $(BIN_FOLDER)wallet $(SOURCE_FOLDER)wallet.c $(SOURCE_FOLDER)crypto/wots/wots.o $(SOURCE_FOLDER)crypto/sha256.o $(LDFLAGS)

bval: wots.o trigg.o sha256.o
	$(CC) -g -o $(BIN_FOLDER)bval $(SOURCE_FOLDER)bval.c $(SOURCE_FOLDER)algo/trigg/trigg.o $(SOURCE_FOLDER)crypto/wots/wots.o $(SOURCE_FOLDER)crypto/sha256.o $(LDFLAGS)

bcon: sha256.o
	$(CC) -g -o $(BIN_FOLDER)bcon $(SOURCE_FOLDER)bcon.c $(SOURCE_FOLDER)crypto/sha256.o $(LDFLAGS)
	
bup: sha256.o
	$(CC) -g -o $(BIN_FOLDER)bup $(SOURCE_FOLDER)bup.c $(SOURCE_FOLDER)crypto/sha256.o $(LDFLAGS)
	
sortlt: sha256.o
	$(CC) -g -o $(BIN_FOLDER)sortlt $(SOURCE_FOLDER)sortlt.c $(SOURCE_FOLDER)crypto/sha256.o $(LDFLAGS)
	
neogen: sha256.o
	$(CC) -g -o $(BIN_FOLDER)neogen $(SOURCE_FOLDER)neogen.c $(SOURCE_FOLDER)crypto/sha256.o $(LDFLAGS)

txclean: sha256.o
	$(CC) -g -o $(BIN_FOLDER)txclean $(SOURCE_FOLDER)txclean.c $(SOURCE_FOLDER)crypto/sha256.o $(LDFLAGS)


minertest: wots.o trigg.o sha256.o
	$(CC) -g -o $(BIN_FOLDER)minertest $(SOURCE_FOLDER)minertest.c $(SOURCE_FOLDER)algo/trigg/trigg.o $(SOURCE_FOLDER)crypto/wots/wots.o $(SOURCE_FOLDER)crypto/sha256.o $(LDFLAGS)



clean:
	rm -rf *.o
