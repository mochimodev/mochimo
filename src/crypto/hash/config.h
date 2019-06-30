#pragma once
#define USE_MD2 1
#define USE_MD5 1
#define USE_SHA1 1
#define USE_SHA256 1
#define USE_BLAKE2B 1
#define USE_KECCAK 1

#define CUDA_HASH 1
#define OCL_HASH 0

typedef unsigned char BYTE;
typedef unsigned int  WORD;

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
