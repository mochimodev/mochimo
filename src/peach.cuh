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
#include <device_launch_parameters.h>

#include <stdint.h>
#include <time.h>
#include "peach.h"

/* WORKAROUND for annoying limitations of intellisense, due to the arguably
 * questionable choice of CUDA delimiter for Kernel Function Arguments.
 * Based on contributions to a stackoverflow question here:
 *    https://stackoverflow.com/a/63084481
 */
#ifdef __INTELLISENSE__
#define CUDA_KERNEL(...)
#else
#define CUDA_KERNEL(FN, ...) FN <<< __VA_ARGS__ >>>
#endif
/* end WORKAROUND */

__global__ void kcu_peach_build
   (word32 offset, word64 *d_map, word32 *d_phash);
__global__ void kcu_peach_solve
   (word64 *d_map, BTRAILER *d_bt, word64 *d_state, word8 diff, word64 *d_solve);
__global__ void kcu_peach_checkhash
   (BTRAILER *d_bt, word8 *d_out, word8 *d_eval);

/* end include guard */
#endif
