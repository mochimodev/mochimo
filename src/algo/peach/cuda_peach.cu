/*
 * cuda_trigg.cu  Multi-GPU CUDA Mining
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 10 August 2018
 * Revision: 31
 */

#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <unistd.h>
#include <cuda_runtime.h>

#include "../../config.h"
#include "peach.h"
#include "nighthash.cu"

__constant__ static uint8_t __align__(8) c_phash[32];
__constant__ static uint8_t __align__(8) c_input[108];
__constant__ static uint8_t __align__(8) c_difficulty;
__constant__ static int Z_MASS[4] = {238,239,240,242};
__constant__ static int Z_ING[2]  = {42,43};
__constant__ static int Z_TIME[16] =
   {82,83,84,85,86,87,88,243,249,250,251,252,253,254,255,253};
__constant__ static int Z_AMB[16] =
   {77,94,95,96,126,214,217,218,220,222,223,224,225,226,227,228};
__constant__ static int Z_ADJ[64] =
   {61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,
    88,89,90,91,92,94,95,96,97,98,99,100,101,102,103,104,105,107,108,109,110,112,114,
    115,116,117,118,119,120,121,122,123,124,125,126,127,128};

inline int cudaCheckError( const char *msg, uint32_t gpu, const char *file)
{
   cudaError err = cudaGetLastError();
   if(cudaSuccess != err) {
      fprintf(stderr, "%s Error (#%d) in %s: %s\n",
              msg, gpu, file, cudaGetErrorString(err));
      return 1;
   }
   return 0;
}


__device__ uint32_t cuda_next_index(uint32_t index, uint8_t *g_map, uint8_t *nonce)
{
   CUDA_NIGHTHASH_CTX nighthash;
   byte seed[HASHLEN + 4 + TILE_LENGTH];
   byte hash[HASHLEN];
   int i, seedlen;

   /* Create nighthash seed for this index on the map */
   seedlen = HASHLEN + 4 + TILE_LENGTH;
   memcpy(seed, nonce, HASHLEN);
   memcpy(seed + HASHLEN, (byte *) &index, 4);
   memcpy(seed + HASHLEN + 4, &g_map[index * TILE_LENGTH], TILE_LENGTH);
   
   /* Setup nighthash the seed, NO TRANSFORM */
   cuda_nighthash_init(&nighthash, seed, seedlen, index, 0);

   /* Update nighthash with the seed data */
   cuda_nighthash_update(&nighthash, seed, seedlen);

   /* Finalize nighthash into the first 32 byte chunk of the tile */
   cuda_nighthash_final(&nighthash, hash);

   /* Convert 32-byte Hash Value Into 8x 32-bit Unsigned Integer */
   for(i = 0, index = 0; i < 8; i++)
      index += ((uint32_t *) hash)[i];

   return index % MAP;
}


__device__ void cuda_gen_tile(uint32_t index, uint8_t *g_map)
{
   CUDA_NIGHTHASH_CTX nighthash;
   byte seed[4 + HASHLEN];
   byte *tilep;
   int i, j, seedlen;

   /* Set map pointer */
   tilep = &g_map[index * TILE_LENGTH];

   /* Create nighthash seed for this index on the map */
   seedlen = 4 + HASHLEN;
   memcpy(seed, (byte *) &index, 4);
   memcpy(seed + 4, c_phash, HASHLEN);

   /* Setup nighthash with a transform of the seed */
   cuda_nighthash_init(&nighthash, seed, seedlen, index, 1);

   /* Update nighthash with the seed data */
   cuda_nighthash_update(&nighthash, seed, seedlen);

   /* Finalize nighthash into the first 32 byte chunk of the tile */
   cuda_nighthash_final(&nighthash, tilep);

   /* Begin constructing the full tile */
   for(i = 0; i < TILE_LENGTH; i += HASHLEN) { /* For each tile row */
      /* Set next row's pointer location */
      j = i + HASHLEN;

      /* Hash the current row to the next, if not at the end */
      if(j < TILE_LENGTH) {
         /* Setup nighthash with a transform of the current row */
         cuda_nighthash_init(&nighthash, &tilep[i], HASHLEN, index, 1);

         /* Update nighthash with the seed data and tile index */
         cuda_nighthash_update(&nighthash, &tilep[i], HASHLEN);
         cuda_nighthash_update(&nighthash, (byte *) &index, 4);

         /* Finalize nighthash into the first 32 byte chunk of the tile */
         cuda_nighthash_final(&nighthash, &tilep[j]);
      }
   }
}


__global__ void cuda_build_map(uint8_t *g_map)
{
   const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;
   if (thread < MAP)
      cuda_gen_tile(thread, g_map);
}


__global__ void cuda_find_peach(uint32_t threads, uint8_t *g_map,
                                int32_t *g_found, uint8_t *g_seed)
{
   const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;

   CUDA_SHA256_CTX ictx;
   uint8_t seed[16] = {0}, nonce[32] = {0};
   uint8_t bt_hash[32], fhash[32];
   int32_t i, j, n, x;
   uint32_t sm;

   if (thread < threads) {
      /* Determine second seed */
      if(thread < 131072) { /* This frame permutations: 131,072 */
         seed[ 0] = Z_TIME[(thread & 15)];
         seed[ 1] = Z_AMB[(thread >> 4) & 15];
         seed[ 2] = 1;
         seed[ 3] = Z_ADJ[(thread >> 8) & 63];
         seed[ 4] = Z_MASS[(thread >> 14) & 3];
         seed[ 5] = 1;
         seed[ 6] = Z_ING[(thread >> 16) & 1];
      }

      /* store full nonce */
      #pragma unroll
      for (i = 0; i < 16; i++)
         nonce[i] = c_input[i + 92];

      #pragma unroll
      for (i = 0; i < 16; i++)
         nonce[i+16] = seed[i];

      /*********************************************************/
      /* Hash 124 bytes of Block Trailer, including both seeds */

      cuda_sha256_init(&ictx);
      cuda_sha256_update(&ictx, c_input, 108);
      cuda_sha256_update(&ictx, seed, 16);
      cuda_sha256_final(&ictx, bt_hash);

      /****************************************************/
      /* Follow the tile path based on the selected nonce */
      
      sm = bt_hash[0];
      #pragma unroll
      for(i = 1; i < HASHLEN; i++)
         sm *= bt_hash[i];
      sm %= MAP;

      /* make <JUMP> tile jumps to find the final tile */
      #pragma unroll
      for(j = 0; j < JUMP; j++)
        sm = cuda_next_index(sm, g_map, nonce);

      /****************************************************************/
      /* Check the hash of the final tile produces the desired result */

      cuda_sha256_init(&ictx);
      cuda_sha256_update(&ictx, bt_hash, HASHLEN);
      cuda_sha256_update(&ictx, &g_map[sm * TILE_LENGTH], TILE_LENGTH);
      cuda_sha256_final(&ictx, fhash);

      /* Evaluate hash */
      for (x = i = j = n = 0; i < HASHLEN; i++) {
         x = fhash[i];
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
      }

      if(n >= c_difficulty && !atomicExch(g_found, 1)) {
         /* PRINCESS FOUND! */
         #pragma unroll
         for (i = 0; i < 16; i++)
            g_seed[i] = seed[i];
      }
      /* Our princess is in another castle ! */
   }
}


extern "C" {

typedef struct __peach_cuda_ctx {
   byte init, curr_seed[16], next_seed[16];
   byte *seed, *d_seed;
   byte *input, *d_map;
   int32_t *d_found;
   cudaStream_t stream;
} PeachCudaCTX;

/* Max 63 GPUs Supported */
PeachCudaCTX ctx[64];
dim3 grid(512);
dim3 block(256);
uint32_t threads = 131072;
int32_t nGPU = 0;
int32_t *found;
byte gpuInit = 0;
byte bnum[8] = {0};
byte *diff;
byte *phash;

int init_cuda_peach(byte difficulty, byte *prevhash, byte *blocknumber) {
   int i;
   
   /* Obtain and check system GPU count */
   nGPU = 0;
   cudaGetDeviceCount(&nGPU);
   if(nGPU<1 || nGPU>64) return nGPU;
   
   /* Allocate pinned host memory */
   cudaMallocHost(&found, 4);
   cudaMallocHost(&diff, 1);
   cudaMallocHost(&phash, 32);
   
   /* Copy immediate block data to pinned memory */
   *found = 0;
   *diff = difficulty;
   memcpy(phash, prevhash, 32);
   
   /* Initialize GPU context init variable incase
    * it holds a random number from memory */
   if(gpuInit == 0) {
      gpuInit = 1;
      for (i = 0; i < nGPU; i++)
         ctx[i].init = 0;
   }
   
   /* Initialize GPU data asynchronously */
   for (i = 0; i < nGPU; i++) {
      cudaSetDevice(i);
      
      /* Create Stream */
      cudaStreamCreate(&ctx[i].stream);
      
      /* Allocate device memory */
      cudaMalloc(&ctx[i].d_found, 4);
      cudaMalloc(&ctx[i].d_seed, 16);
      
      /* Allocate associated device-host memory */
      cudaMallocHost(&ctx[i].seed, 16);
      cudaMallocHost(&ctx[i].input, 108);
      
      /* Copy immediate block data to device memory */
      cudaMemcpyToSymbolAsync(c_difficulty, diff, 1, 0,
                              cudaMemcpyHostToDevice, ctx[i].stream);
      cudaMemcpyToSymbolAsync(c_phash, phash, 32, 0,
                              cudaMemcpyHostToDevice, ctx[i].stream);
      
      /* Set remaining device memory */
      cudaMemsetAsync(ctx[i].d_found, 0, 4, ctx[i].stream);
      cudaMemsetAsync(ctx[i].d_seed, 0, 16, ctx[i].stream);
      
      /* Set initial round variables */
      ctx[i].next_seed[0] = 0;
      
      /* If first init, setup map and cache */
      if(ctx[i].init == 0) {
         /* NOTE: The device MAP that holds the data of a map DOES NOT
          * explicitly get free()'d. The reason behind this is because
          * we reuse the map variable between blocks, and just rebuild
          * the map once every block. The GPU free's the MAP when the
          * program ends by default. This can be adjusted later. */
         cudaMalloc(&ctx[i].d_map, MAP_LENGTH);
         ctx[i].init = 1;
      }
      
      /* (re)Build map if new block */
      if(memcmp(bnum, blocknumber, 8) != 0)
         cuda_build_map<<<4096, 256, 0, ctx[i].stream>>>(ctx[i].d_map);
   }
   
   /* Check for any GPU initialization errors */
   for(i = 0; i < nGPU; i++) {
      cudaSetDevice(i);
      cudaStreamSynchronize(ctx[i].stream);
      if(cudaCheckError("init_cuda_peach()", i, __FILE__))
         return -1;
   }
   
   /* Update block number */
   memcpy(bnum, blocknumber, 8);

   return nGPU;
}

void free_cuda_peach() {
   int i;
   
   /* Free pinned host memory */
   cudaFreeHost(diff);
   cudaFreeHost(found);
   cudaFreeHost(phash);
   
   /* Free GPU data */
   for (i = 0; i<nGPU; i++) {
      cudaSetDevice(i);
      
      /* Destroy Stream */
      cudaStreamDestroy(ctx[i].stream);
      
      /* Free device memory */
      cudaFree(ctx[i].d_found);
      cudaFree(ctx[i].d_seed);
      cudaFree(ctx[i].d_map);
      
      /* Free associated device-host memory */
      cudaFreeHost(ctx[i].seed);
      cudaFreeHost(ctx[i].input);
   }
}

extern byte *trigg_gen(byte *in);

__host__ void cuda_peach(byte *bt, uint32_t *hps, byte *runflag)
{
   int i;
   uint64_t lastnHaiku, nHaiku = 0;
   time_t seconds = time(NULL);
   for( ; *runflag && *found == 0; ) {
      for (i=0; i<nGPU; i++) {
         /* Prepare next seed for GPU... */
         if(ctx[i].next_seed[0] == 0) {
            /* ... generate first GPU seed (and expand as Haiku) */
            trigg_gen(ctx[i].next_seed);

            /* ... and prepare round data */
            memcpy(ctx[i].input, bt, 92);
            memcpy(ctx[i].input+92, ctx[i].next_seed, 16);
         }
         /* Check if GPU has finished */
         cudaSetDevice(i);
         if(cudaStreamQuery(ctx[i].stream) == cudaSuccess) {
            cudaMemcpy(found, ctx[i].d_found, 4, cudaMemcpyDeviceToHost);
            if(*found==1) { /* SOLVED A BLOCK! */
               cudaMemcpy(ctx[i].seed, ctx[i].d_seed, 16, cudaMemcpyDeviceToHost);
               memcpy(bt + 92, ctx[i].curr_seed, 16);
               memcpy(bt + 92 + 16, ctx[i].seed, 16);
               break;
            }
            /* Send new GPU round Data */
            cudaMemcpyToSymbolAsync(c_input, ctx[i].input, 108, 0,
                                    cudaMemcpyHostToDevice, ctx[i].stream);
            /* Start GPU round */
            cuda_find_peach<<<grid, block, 0, ctx[i].stream>>>(threads,
                                 ctx[i].d_map, ctx[i].d_found, ctx[i].d_seed);

            /* Add to haiku count */
            nHaiku += threads;

            /* Store round vars aside for checks next loop */
            memcpy(ctx[i].curr_seed,ctx[i].next_seed,16);
            ctx[i].next_seed[0] = 0;
         }
         
         /* Waiting on GPU? ... */
         if(cudaCheckError("cuda_peach()", i, __FILE__)) {
            *runflag = 0;
            return;
         }
      }
      
      /* Chill a bit if nothing is happening */
      if(lastnHaiku == nHaiku) usleep(1000);
      else lastnHaiku = nHaiku;
   }
    
   seconds = time(NULL) - seconds;
   if(seconds == 0) seconds = 1;
   nHaiku /= seconds;
   *hps = (uint32_t) nHaiku;
}


}
