#pragma once
#ifndef CUDA_PEACH_H
#define CUDA_PEACH_H

#include <stdint.h>
#include <sys/time.h>
#include "../../config.h"
#include <cuda_runtime.h>
#include <nvml.h>

#ifdef __cplusplus
extern "C" {
#endif
   int init_cuda_peach(byte difficulty, byte *prevhash, byte *blocknumber);
   void free_cuda_peach();
   void cuda_peach(byte *bt, uint32_t *hps, byte *runflag);

   typedef struct __peach_cuda_ctx {
      byte init, curr_seed[16], next_seed[16];
      byte *seed, *d_seed;
      byte *input, *d_map;
      int32_t *d_found;
      cudaStream_t stream;
      struct timeval t_start, t_end;
      uint32_t hps[3];
      uint8_t hps_index;
      uint32_t ahps;
   } PeachCudaCTX;

   extern PeachCudaCTX peach_ctx[64];

   int init_nvml();
   typedef struct {
      uint32_t pciDomainId;
      uint32_t pciBusId;
      uint32_t pciDeviceId;
      nvmlDevice_t nvml_dev;
      uint32_t cudaNum;
      uint32_t temp;
      uint32_t power;
   } GPU_t;
#define MAX_GPUS 64
   extern GPU_t gpus[MAX_GPUS];
#ifdef __cplusplus
}
#endif

#endif
