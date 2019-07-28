/* cuda_peach.cu   Multi-GPU CUDA Mining
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 10 July 2019
 * Revised: 22 July 2019
 * Revision: 3
 *
 * Optimized version, cloning from cuda_peach.cu.
 *
 */

#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <nvml.h>

#include "../../crypto/hash/cpu/sha256.c"
#include "../../config.h"
#include "peach.h"
#include "nighthash.cu"
#include "cuda_peach.h"

__constant__ static CUDA_SHA256_CTX __align__(8) c_precomputed_sha256;
__constant__ static uint8_t __align__(8) c_phash[32];
__constant__ static uint8_t __align__(8) c_input[16384];
__constant__ static uint8_t __align__(8) c_difficulty;


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


__device__ uint32_t cuda_next_index(uint32_t index, uint8_t *g_map,
                                    uint8_t *first_seed, uint8_t *last_seed)
{
   CUDA_NIGHTHASH_CTX nighthash;
   byte seed[HASHLEN + 4 + TILE_LENGTH];
   byte hash[HASHLEN];
   int i, seedlen;

   /* Create nighthash seed for this index on the map */
   seedlen = HASHLEN + 4 + TILE_LENGTH;
   
   memcpy(seed, first_seed, 16);
   memcpy(seed + 16, last_seed, 16);
   memcpy(seed + HASHLEN, &index, 4);
   memcpy(seed + 36, &g_map[index * TILE_LENGTH], TILE_LENGTH);

   /* Setup nighthash the seed, NO TRANSFORM */
   cuda_nighthash_init(&nighthash, seed, seedlen, index, 0);

   /* Update nighthash with the seed data */
   cuda_nighthash_update(&nighthash, seed, seedlen);

   /* Finalize nighthash into the first 32 byte chunk of the tile */
   cuda_nighthash_final(&nighthash, hash);

   /* Convert 32-byte Hash Value Into 8x 32-bit Unsigned Integer */
   for(i = 0, index = 0; i < 8; i++)
      index += ((uint32_t *) hash)[i];

   return index & (MAP-1);
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
   
   memcpy(seed, &index, 4);
   memcpy(seed + 4, c_phash, 32);

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


__global__ void cuda_find_peach(uint32_t offset, uint8_t *g_map,
                                int32_t *g_found, uint8_t *g_nonce)
{
   __shared__
      uint8_t first_seed[16];
      uint8_t last_seed[16], hash[32];
   CUDA_SHA256_CTX ictx;
   int32_t i, j, n;
   uint32_t sm;
   
   /******************/
   /* Assemble seeds */
   if(threadIdx.x < 16)
      first_seed[threadIdx.x] = c_input[((offset + blockIdx.x) << 4) + threadIdx.x];
   
   __syncthreads();
   
   memcpy(last_seed, &c_input[threadIdx.x << 4], 16);
   
   /*********************************************************/
   /* Hash 124 bytes of Block Trailer, including both seeds */
   memcpy(&ictx, &c_precomputed_sha256, sizeof(CUDA_SHA256_CTX));
   
   cuda_sha256_update(&ictx, first_seed, 16);
   cuda_sha256_update(&ictx, last_seed, 16);
   cuda_sha256_final(&ictx, hash);
   
   /****************************************************/
   /* Follow the tile path based on the selected nonce */
   sm = hash[0];
   #pragma unroll
   for(i = 1; i < HASHLEN; i++)
       sm *= hash[i];
   sm = (sm & (MAP - 1));
   
   /* make <JUMP> tile jumps to find the final tile */
   #pragma unroll
   for(j = 0; j < JUMP; j++)
      sm = cuda_next_index(sm, g_map, first_seed, last_seed);
   
   /****************************************************************/
   /* Check the hash of the final tile produces the desired result */
   cuda_sha256_init(&ictx);
   cuda_sha256_update(&ictx, hash, HASHLEN);
   cuda_sha256_update(&ictx, &g_map[sm * TILE_LENGTH], TILE_LENGTH);
   cuda_sha256_final(&ictx, hash);
   
   /* Evaluate hash */
   for(i = j = n = 0; i < 8 && n == j; i++, j += 32)
      n = __clz( __byte_perm( ((int32_t *) hash)[i], 0, 0x0123 ) );

   if(n >= c_difficulty && !atomicExch(g_found, 1)) {
      /* PRINCESS FOUND! */
      memcpy(g_nonce, first_seed, 16);
      memcpy(g_nonce + 16, last_seed, 16);
   }
   /* Our princess is in another castle ! */
}


extern "C" {


uint8_t enable_nvml = 0;
GPU_t gpus[MAX_GPUS] = { 0 };
uint32_t num_gpus = 0;
/* Max 63 GPUs Supported */
PeachCudaCTX peach_ctx[64];
PeachCudaCTX *ctx = peach_ctx;
int32_t nGPU = 0;
SHA256_CTX *precompute_ctx;
int32_t *found;
byte *diff;
byte *phash;

int init_nvml() {
   int32_t num_cuda = 0;
   cudaError_t cr = cudaGetDeviceCount(&num_cuda);
   if (num_cuda > MAX_GPUS) num_cuda = MAX_GPUS;

   for (int i = 0; i < num_cuda; i++) {
      struct cudaDeviceProp p = { 0 };
      cudaError_t cr = cudaGetDeviceProperties(&p, i);
      printf("CUDA pciDomainID: %x, pciBusID: %x, pciDeviceID: %x\n", p.pciDomainID, p.pciBusID, p.pciDeviceID);
      gpus[i].pciDomainId = p.pciDomainID;
      gpus[i].pciBusId = p.pciBusID;
      gpus[i].pciDeviceId = p.pciDeviceID;
      gpus[i].cudaNum = i;
      num_gpus++;
   }


   nvmlReturn_t r = nvmlInit();
   if (r != NVML_SUCCESS) {
      printf("Failed to initialize NVML: %s\n", nvmlErrorString(r));
      enable_nvml = 0;
      return 0;
   }
   uint32_t nvml_device_count;
   r = nvmlDeviceGetCount(&nvml_device_count);
   if (r != NVML_SUCCESS) {
      printf("Failed to get NVML device count: %s\n", nvmlErrorString(r));
      enable_nvml = 0;
      return 0;
   }
   printf("NVML Devices: %d\n", nvml_device_count);
   for (int i = 0; i < nvml_device_count; i++) {
      nvmlDevice_t dev;
      r = nvmlDeviceGetHandleByIndex(i, &dev);
      if (r != NVML_SUCCESS) {
         printf("nvmlDeviceGetHandleByIndex failed: %s\n", nvmlErrorString(r));
         nvml_device_count = i;
         break;
      }
      nvmlPciInfo_t pci;
      r = nvmlDeviceGetPciInfo(dev, &pci);
      if (r != NVML_SUCCESS) {
         printf("nvmlDeviceGetPciInfo failed: %s\n", nvmlErrorString(r));
         continue;
      }
      printf("NVML PCI: pciDeviceId: %x, pciSubSystemId: %x, domain: %x, device: %x, bus: %x\n", pci.pciDeviceId, pci.pciSubSystemId, pci.domain, pci.device, pci.bus);

      for (int j = 0; j < num_cuda; j++) {
         if (gpus[j].pciDomainId == pci.domain && gpus[j].pciBusId == pci.bus && gpus[i].pciDeviceId == pci.device) {
            printf("NVML device is CUDA Device: %d\n", gpus[j].cudaNum);
            gpus[j].nvml_dev = dev;
            break;
         }
      }

      char device_name[128];
      r = nvmlDeviceGetName(dev, device_name, 128);
      if (r != NVML_SUCCESS) {
         printf("nvmlDeviceGetName failed: %s\n", nvmlErrorString(r));
      }
      else {
         printf("Device: %d, Name: %s\n", i, device_name);
      }
   }
   enable_nvml = 1;
   return 1;
}

int init_cuda_peach(byte difficulty, byte *prevhash, byte *bt) {
   int i;
   
   /* Obtain and check system GPU count */
   nGPU = 0;
   cudaGetDeviceCount(&nGPU);
   if(nGPU<1 || nGPU>64) return nGPU;
   
   /* Allocate pinned host memory */
   cudaMallocHost(&found, 4);
   cudaMallocHost(&diff, 1);
   cudaMallocHost(&phash, 32);
   cudaMallocHost(&precompute_ctx, sizeof(SHA256_CTX));
   
   /* Copy immediate block data to pinned memory */
   *found = 0;
   *diff = difficulty;
   memcpy(phash, prevhash, 32);
   
   /* Precompute SHA256 */
   sha256_init(precompute_ctx);
   sha256_update(precompute_ctx, bt, 92);
   
   /* Initialize GPU data asynchronously */
   for (i = 0; i < nGPU; i++) {
      cudaSetDevice(i);
      
      /* Get the best block/thread configs */
      cudaOccupancyMaxPotentialBlockSize(&ctx[i].nblock, &ctx[i].nthread,
                                         cuda_find_peach, 16, 1024);
      ctx[i].total_threads = ctx[i].nblock * ctx[i].nthread;
      
      /* Create Stream */
      cudaStreamCreate(&ctx[i].stream);
      
      /* Allocate device memory */
      cudaMalloc(&ctx[i].d_found, 4);
      cudaMalloc(&ctx[i].d_nonce, 32);
      
      /* Allocate associated device-host memory */
      cudaMallocHost(&ctx[i].found, 4);
      cudaMallocHost(&ctx[i].nonce, 32);
      cudaMallocHost(&ctx[i].input, ctx[i].nthread * 16);
      
      /* Copy immediate block data to device memory */
      cudaMemcpyToSymbolAsync(c_difficulty, diff, 1, 0,
                              cudaMemcpyHostToDevice, ctx[i].stream);
      cudaMemcpyToSymbolAsync(c_phash, phash, 32, 0,
                              cudaMemcpyHostToDevice, ctx[i].stream);
      cudaMemcpyToSymbolAsync(c_precomputed_sha256, precompute_ctx,
                              sizeof(SHA256_CTX), 0, cudaMemcpyHostToDevice,
                              ctx[i].stream);
      
      /* Set remaining device memory */
      cudaMemsetAsync(ctx[i].d_found, 0, 4, ctx[i].stream);
      memset(ctx[i].found, 0, 4);
      
      /* Set scan offset to 1024*/
      ctx[i].scan_offset = ctx[i].nthread;
      
      /* Setup map and cache */
      cudaMalloc(&ctx[i].d_map, MAP_LENGTH);
      cuda_build_map<<<4096, 256, 0, ctx[i].stream>>>(ctx[i].d_map);
   }
   
   /* Check for any GPU initialization errors */
   for(i = 0; i < nGPU; i++) {
      cudaSetDevice(i);
      cudaStreamSynchronize(ctx[i].stream);
      if(cudaCheckError("init_cuda_peach()", i, __FILE__))
         return -1;
   }

   return nGPU;
}

void free_cuda_peach() {
   int i;
   
   /* Free pinned host memory */
   cudaFreeHost(diff);
   cudaFreeHost(found);
   cudaFreeHost(phash);
   cudaFreeHost(precompute_ctx);
   
   /* Free GPU data */
   for (i = 0; i<nGPU; i++) {
      cudaSetDevice(i);
      
      /* Destroy Stream */
      cudaStreamDestroy(ctx[i].stream);
      
      /* Free device memory */
      cudaFree(ctx[i].d_found);
      cudaFree(ctx[i].d_nonce);
      cudaFree(ctx[i].d_map);
      
      /* Free associated device-host memory */
      cudaFreeHost(ctx[i].found);
      cudaFreeHost(ctx[i].nonce);
      cudaFreeHost(ctx[i].input);
   }
}

extern byte *trigg_gen(byte *in);

__host__ void cuda_peach(byte *bt, uint32_t *hps, byte *runflag)
{
   int i, j, k;
   double tdiff;
   uint32_t shps;
   uint64_t lastnHaiku, nHaiku, ustart, uend;
   timeval nSeconds;
   time_t gpu_stats_time = time(NULL);
   
   gettimeofday(&nSeconds, NULL);
   for(nHaiku = 0; *runflag && *found == 0; ) {
      for (i=0; i<nGPU; i++) {
         /* Check if GPU has finished */
         cudaSetDevice(i);
         if(cudaStreamQuery(ctx[i].stream) == cudaSuccess) {
            /* Obtain haiku/s calc data */
            gettimeofday(&(ctx[i].t_end), NULL);
            ustart = 1000000 * ctx[i].t_start.tv_sec + ctx[i].t_start.tv_usec;
            if (ustart > 0) {
               uend = 1000000 * ctx[i].t_end.tv_sec + ctx[i].t_end.tv_usec;
               tdiff = (uend - ustart) / 1000.0 / 1000.0;
            }
            gettimeofday(&(ctx[i].t_start), NULL);
            
            /* Check for a solved block */
            if(*ctx[i].found==1) { /* SOLVED A BLOCK! */
               cudaMemcpy(ctx[i].nonce, ctx[i].d_nonce, 32, cudaMemcpyDeviceToHost);
               memcpy(bt + 92, ctx[i].nonce, 32);
               *found = 1;
               break;
            }
            
            /* Init GPU data if necessary */
            if (ctx[i].scan_offset + ctx[i].nblock >= ctx[i].nthread) {
               /* Reset offset */
               ctx[i].scan_offset = 0;
               /* Generate random seed array data */
               for(j = 0, k = ctx[i].nthread * 16; j < k; j += 16)
                  trigg_gen(ctx[i].input + j);
               /* Send new seed array data */
               cudaMemcpyToSymbolAsync(c_input, ctx[i].input, k, 0,
                                       cudaMemcpyHostToDevice, ctx[i].stream);
            }
            /* Start GPU round */
            cuda_find_peach<<<ctx[i].nblock,ctx[i].nthread,16,ctx[i].stream>>>
            (ctx[i].scan_offset, ctx[i].d_map, ctx[i].d_found, ctx[i].d_nonce);
            /* Retrieve GPU found status */
            cudaMemcpyAsync(ctx[i].found, ctx[i].d_found, 4, cudaMemcpyDeviceToHost);

            /* Add to haiku count */
            nHaiku += ctx[i].total_threads;
            ctx[i].scan_offset += ctx[i].nblock;
            
            /* Perform per GPU Haiku/s cacluation */
            if (ustart > 0) {
               ctx[i].hps_index = (ctx[i].hps_index + 1) % 3;
               ctx[i].hps[ctx[i].hps_index] = ctx[i].total_threads / tdiff;
               shps = 0;
               for (j = 0; j < 3; j++) {
                  shps += ctx[i].hps[j];
               }
               ctx[i].ahps = shps / 3;
            }
         }
         
         /* Waiting on GPU? ... */
         if(cudaCheckError("cuda_peach()", i, __FILE__)) {
            *runflag = 0;
            return;
         }
      }
      
      /* Print GPU stats and chill if waiting on all GPUs */
      if(lastnHaiku == nHaiku) {
         /* Print GPU stats every 5 seconds */
         if ( (time(NULL) - gpu_stats_time) > 5 ) {
            for (j = 0; j < nGPU; j++) {
               if (enable_nvml) {
                  uint32_t temp = 0;
                  uint32_t power = 0;
                  nvmlReturn_t r = nvmlDeviceGetTemperature(gpus[j].nvml_dev, NVML_TEMPERATURE_GPU, &temp);
                  if (r != NVML_SUCCESS) {
                     printf("nvmlDeviceGetTemperature failed: %s\n", nvmlErrorString(r));
                  }

                  r = nvmlDeviceGetPowerUsage(gpus[j].nvml_dev, &power);
                  if (r != NVML_SUCCESS) {
                     printf("nvmlDeviceGetPowerUsage Failed: %s\n", nvmlErrorString(r));
                  }
                  gpus[j].temp = temp;
                  gpus[j].power = power;

                  printf("GPU %d: %7d H/s, Temperature: %d C, Power: %6.2f W\n", j,
                        ctx[j].ahps, gpus[j].temp, gpus[j].power / 1000.0);
               } /* else {
                  printf("GPU %d: %7d H/s\n", j, ctx[j].ahps);
               } */
            }
            gpu_stats_time = time(NULL);
         }
         /* Chill for 1ms */
         usleep(1000);
      }
      else lastnHaiku = nHaiku;
   }
   
   /* Calculate Final Haiku/s */
   ustart = 1000000 * nSeconds.tv_sec + nSeconds.tv_usec;
   gettimeofday(&nSeconds, NULL);
   uend = 1000000 * nSeconds.tv_sec + nSeconds.tv_usec;
   tdiff = (uend - ustart) / 1000.0 / 1000.0;
   *hps = (uint32_t) (nHaiku / tdiff);
   
   /* Reset Miner Data */
   *found = 0;
   for (i=0; i<nGPU; i++) {
      cudaSetDevice(i);
      memset(ctx[i].found, 0, 4);
      cudaMemset(ctx[i].d_found, 0, 4);
      ctx[i].scan_offset = ctx[i].nthread;
   }
}


}
