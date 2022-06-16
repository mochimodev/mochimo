/**
 * @file peach.cuh
 * @brief Peach CUDA Proof-of-Work algorithm support.
 * @details See peach.h for details on peach algorithm.
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_PEACH_CUH
#define MOCHIMO_PEACH_CUH


#include <cuda_runtime.h>
#include <nvml.h>

#include <stdint.h>
#include <time.h>
#include "peach.h"

__global__ void kcu_peach_build
   (word32 offset, word64 *d_map, word32 *d_phash);
__global__ void kcu_peach_solve
   (word64 *d_map, SHA256_CTX *d_ictx, word8 diff, word64 *d_solve);
__global__ void kcu_peach_checkhash
   (BTRAILER *d_bt, word8 *d_out, word8 *d_eval);

/* end include guard */
#endif
