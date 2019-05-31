/*
 * cuda_v24.cu  Multi-GPU CUDA Mining
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 23 May 2019
 *
 */

#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <unistd.h>
#include <cuda_runtime.h>
extern "C" {
#include "../../crypto/sha256.h"
}

#include "../../config.h"
#include "sha256.cuh"
#include "v24.h"


#define CUDA_CHECK_ERROR \
{\
  cudaError_t err = cudaGetLastError(); \
  if (err != cudaSuccess) \
  { \
    printf("GPU error: %s \n", cudaGetErrorString(err)); \
  } \
}

__constant__ uint8_t c_BTRAILER[TRLSIZE]; //TODO: reduce to 124 uint8_ts
__constant__ uint8_t c_NONCE1[16];

__global__ void cuda_gen_workfield(uint64_t * workfield, uint64_t * tfile, uint32_t tfsize)
{
    uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    workfield[tidx] = tfile[tidx % tfsize];
}

__constant__ static uint8_t Z_ADJ[64]  = {61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,
                                          87,88,89,90,91,92,94,95,96,97,98,99,100,101,102,103,104,105,107,108,109,110,
                                          112,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128};
__constant__ static uint8_t Z_NS[64] = {129,130,131,132,133,134,135,136,137,138,145,149,154,155,156,157,177,178,179,180,
                                        182,183,184,185,186,187,188,189,190,191,192,193,194,196,197,198,199,200,201,202,
                                        203,204,205,206,207,208,209,210,211,212,213,241,244,245,246,247,248,249,250,251,
                                        252,253,254,255};
__constant__ static uint8_t Z_PREP[2][4]  = {{12,13,14,15},{16,17,12,13}};
__constant__ static uint8_t Z_TIME[2][8] = {{82,83,84,85,86,87,88,253},{249,250,251,252,253,254,255,243}};

__constant__ static uint8_t Z_ING[16]  = {18,19,20,21,22,23,24,25,
                                          26,27,28,29,30,31,32,33};
__constant__ static uint8_t Z_MASS[16] = {214,215,216,217,218,219,
                                          220,221,222,223,224,225,
                                          226,227,228,229};

__device__ int gpu_v24_eval(uint8_t *bp, uint8_t d)
{
  uint8_t x, i, j, n;

  x = i = j = n = 0;

  for (i = 0; i < HASHLEN; i++) {
      x = *(bp + i);
      if (x != 0) {
          for(j = 7; j > 0; j--) {
              x >>= 1;
              if(x == 0) {
                  n += j;
                  break;
              }
          }
          break;
      }
      n += 8;
      continue;
  }
  if(n >= d) return 1;
  return 0;
}

__global__ void v24_kernel(const uint32_t offset,
                           int *g_found, uint8_t *g_seed,
                           uint8_t diff, uint8_t* workfield)
{
    const uint32_t thread = offset + blockDim.x * blockIdx.x + threadIdx.x;
    uint8_t seed[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    uint8_t input[128];
    uint8_t indexhash[HASHLEN], solution[HASHLEN];
    seed[ 0] = Z_ING[(thread & 15)];
    seed[ 1] = Z_PREP[0][(thread >> 4) & 3];
    seed[ 2] = Z_TIME[0][(thread >> 6) & 7];
    seed[ 3] = Z_MASS[(thread >> 9) & 15];//9+4
    seed[ 4] = 1;
    seed[ 5] = Z_MASS[(thread >> 13) & 15];// 13 + 4
    seed[ 6] = Z_ING[(thread >> 17) & 15]; //17
    seed[ 7] = 3;
    seed[ 8] = 1;
    seed[ 9] = 5;
    cuda_SHA256_CTX ictx, mctx;
    memcpy(input, c_BTRAILER, 92);
    memcpy(input + 92, c_NONCE1, 16);
    // seed 6, 9 in loop
    for (uint32_t pid = 0; pid < 4096 /*64*64*/; pid++)
    {
        seed[10] = Z_ADJ[pid & 63];
        seed[11] = Z_NS[(pid >> 6) & 63];
        memcpy(input + 92 + 16, seed, 16);

        cuda_sha256_init(&ictx);
        cuda_sha256_update(&ictx, input, 124);
        memcpy(&mctx, &ictx, sizeof(cuda_SHA256_CTX));
        cuda_sha256_final(&ictx, indexhash);
        uint64_t index = 0;
        for(int j = 0; j < 8; j++)
        {
            index += *((uint32_t *) indexhash + j);
            if (index > (WORKFIELD - 32)) index %= (WORKFIELD - 32);
            cuda_sha256_update(&mctx, workfield + index, HASHLEN);
        }
        cuda_sha256_final(&mctx, solution);
        if (gpu_v24_eval(solution, diff))
        {
            if (atomicExch(g_found, 1) == 0)
            {
                memcpy(g_seed, seed, 16);
            }
            return;
        }
    }
}


extern "C"
{

typedef struct __trigg_cuda_ctx
{
    uint8_t pCPUSeed[16]; // first haiku frame
    uint8_t *pGPUSeed; // second haiku frame
    uint8_t pCPUResult[16]; // for downloading from GPU
    uint8_t *pTFile;
    uint8_t *pWorkfield;
    int pHostFound;
    int *pDevFound; // nonce counter
    uint64_t mBlockNumber;
    cudaStream_t mStream;
    cudaStream_t mDataStream;// reserved for optimization
    uint64_t mThreadOffset;

    int mGrid;
    int mBlock;
    uint64_t mHashPerLoop;
    uint64_t mThreadPerLoop;
} TriggCudaCTX;

/* Prototypes from trigg.o dependency */
byte *trigg_gen(byte *in);
void trigg_expand2(byte *in, byte *out);

TriggCudaCTX gCTX[64];    /* Max 64 GPUs Supported */
int nGPU = 0;
uint8_t* gTFile;

/*
* Allocating all needed variables, filling it with CPU data
* output : 0 (error), otherwise, number of gpu
*/
int cuda_v24_init(uint8_t *pBtrailer, uint32_t blocknum)
{
    static const uint32_t BASE_THREAD = 256;
    /* Obtain and check system GPU count */
    uint32_t tfsize;
    uint32_t tfsizechk;

    cudaGetDeviceCount(&nGPU);
    if(nGPU<1 || nGPU>64) return nGPU;
    FILE* fp = fopen("tfile.dat", "rb");
    if(fp == NULL)
    {
       printf("Fatal: Unable to open T-file.\n");
       return 0;
    }
    /* Get the first previous neogenesis block number */
    blocknum = ((blocknum / 256) * 256) - 256;

    /* Determine Expected trailer file size */
    tfsize = blocknum * TRLSIZE;

    /* Collect size of Tfile in bytes, perform sanity check */
    fseek(fp, 0, SEEK_END);
    tfsizechk = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if(tfsize > tfsizechk){ /* She needs something bigger. */
      printf("Fatal: T-File Size on disk is just too small.\n");
      return 0;
    }

    if(tfsizechk > WORKFIELD) { /* Won't happen for 200+ years */
      printf("Fatal: T-File Size > WORKFIELD.\n");
      return 0;
    }
    /* Allocate T-file on the Heap */
    gTFile = (uint8_t*)malloc(tfsize);
    if(gTFile == NULL) {
      printf("Fatal: Can't allocate space on the Heap for T-file.\n");
      return 0;
    }
    memset(gTFile, 0, tfsize);

    /* Copy T-file to memory */
    uint32_t count = 0;
    uint32_t total = 0;
    for(int j = 0; j < tfsize/TRLSIZE ; j++)
    {
      count = fread(gTFile + j*TRLSIZE, 1, TRLSIZE, fp);
      total += count;
      if(count != TRLSIZE) {
         printf("Fatal: Bad read on T-file\n");
         return 0;
      }
    }

    for ( int i = 0 ; i<nGPU; i++)
    {
        cudaSetDevice(i);
        /* Create Stream */
        cudaStreamCreate(&gCTX[i].mStream);
        /* Allocate device memory */
        cudaMalloc(&gCTX[i].pDevFound, 4);
        cudaMemsetAsync(gCTX[i].pDevFound, 0, 4, gCTX[i].mStream);

        cudaMalloc(&gCTX[i].pGPUSeed, 16);
        cudaMemsetAsync(gCTX[i].pGPUSeed, 0, 16, gCTX[i].mStream);

        /* Allocate associated device-host memory */
        cudaMalloc(&gCTX[i].pTFile, tfsize);
        cudaMalloc(&gCTX[i].pWorkfield, WORKFIELD);

        cudaMemcpyAsync(gCTX[i].pTFile, gTFile, tfsize, cudaMemcpyHostToDevice, gCTX[i].mStream);
        cudaMemcpyToSymbolAsync(c_BTRAILER, pBtrailer, TRLSIZE, 0,
                                cudaMemcpyHostToDevice, gCTX[i].mStream);
        cuda_gen_workfield<<<WORKFIELD / BASE_THREAD / 8, BASE_THREAD, 0, gCTX[i].mStream>>>(
        (uint64_t *)gCTX[i].pWorkfield, (uint64_t *)gCTX[i].pTFile, tfsize/8);

        gCTX[i].mThreadOffset = 1L << 30;
        cudaOccupancyMaxPotentialBlockSize(&gCTX[i].mGrid, &gCTX[i].mBlock, v24_kernel, 0, 0);
        gCTX[i].mThreadPerLoop = gCTX[i].mGrid * gCTX[i].mBlock;
        gCTX[i].mHashPerLoop = gCTX[i].mGrid * gCTX[i].mBlock * 64*64;
    }
    for (int i = 0; i < nGPU; i++)
    {
      cudaStreamSynchronize(gCTX[i].mStream); // make sure everything are done
      CUDA_CHECK_ERROR;
    }
    return nGPU;
}

void cuda_v24_free()
{
    /* Free pinned host memory */
    for (int i = 0 ; i<nGPU; i++)
    {
        cudaSetDevice(i);
        /* Destroy Stream */
        cudaStreamDestroy(gCTX[i].mStream);
        /* Free device memory */
        cudaFree(gCTX[i].pGPUSeed);
        cudaFree(gCTX[i].pTFile);
        cudaFreeHost(gCTX[i].pDevFound);
    }
}


/***
* Simple code for new algorithm (v24), code path must be clear (straight, no
* precompute or hacky stuff) and easy to understand. Further optimization should
* be in another function and kernels
***/
int cuda_v24_mine(uint8_t *pBtrailer, uint32_t difficulty, byte *pHaiku,
                  uint32_t *pHashrate, unsigned char *pExitSignal)
{
  while (1)
  {
    /* pExitSignal is set to 0 on SIGTERM */
    if(!(*pExitSignal)) break;

    for (int i=0; i<nGPU; i++)
    {
        /* Check if GPU has finished */
        cudaSetDevice(i);

        if(cudaStreamQuery(gCTX[i].mStream) == cudaSuccess)
        {
            cudaMemcpyAsync(&(gCTX[i].pHostFound), gCTX[i].pDevFound, 4,
                            cudaMemcpyDeviceToHost, gCTX[i].mStream);
            cudaStreamSynchronize(gCTX[i].mStream);
            if(gCTX[i].pHostFound != 0)
            {
                /* SOLVED A BLOCK! */
                cudaMemcpyAsync(gCTX[i].pCPUResult, gCTX[i].pGPUSeed, 16,
                                cudaMemcpyDeviceToHost, gCTX[i].mStream);
                cudaStreamSynchronize(gCTX[i].mStream);
                memcpy(pBtrailer + 92, gCTX[i].pCPUSeed, 16); // nonce start from 92
                memcpy(pBtrailer + 92 + 16, gCTX[i].pCPUResult, 16);
                return 1;
            }
            gCTX[i].mThreadOffset += gCTX[i].mThreadPerLoop;
            *pHashrate += gCTX[i].mHashPerLoop;
            // current kernel can run with thread of [0, 2^21]
            if (gCTX[i].mThreadOffset > 2097151)
            {
                // generate first haiku frame, upload it to a symbol
                trigg_gen(gCTX[i].pCPUSeed);
                cudaMemcpyToSymbolAsync(c_NONCE1, gCTX[i].pCPUSeed, 16, 0,
                                        cudaMemcpyHostToDevice, gCTX[i].mStream);
                gCTX[i].mThreadOffset = 0;
            }
            v24_kernel<<<gCTX[i].mGrid, gCTX[i].mBlock, 0, gCTX[i].mStream>>>
            (gCTX[i].mThreadOffset, gCTX[i].pDevFound, gCTX[i].pGPUSeed,
             difficulty, gCTX[i].pWorkfield);
        }
        else
        {
          continue;  /* Waiting on GPU ... */
        }
    }
    usleep(1);
  }

  return 0;
}
}
