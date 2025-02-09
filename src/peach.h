/**
 * @file peach.h
 * @brief Peach Proof-of-Work algorithm support.
 * @details The Peach algorithm was designed, specificaly, with the
 * intention of permitting a "mining advantage" to modern GPUs with
 * more than 1GiB VRAM where it can cache data faster than it would
 * take to re-compute it.
 * <br />
 * The cache is made of 1048576 x 1KibiByte chunks (a.k.a tiles) of
 * data, generated deterministically from the previous blocks hash,
 * making it unique per block solve. The generation process, dubbed
 * Nighthash, generates chunks using deterministic single precision
 * floating point operations, a selection of eight different memory
 * transformations, and finally a selection of eight different hash
 * algorithms. The final digest is then placed within the first row
 * of a tile. Subsequent rows are filled in the same manner, except
 * they use the previous row as input until the chunk is completed.
 * <br />
 * Peach also utilizes the nonce restrictions designed for use with
 * Trigg's algorithm, to retain the pleasantries of using haikus.
 * ```
 * a raindrop
 * on sunrise air--
 * drowned
 * ```
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
 * @note If compiled with `ENABLE_CPU_PEACH_CACHE`, peach_solve()
 * generates and stores tiles in a statically aallocated Peach Map
 * and cache taking up 1 Gibibyte and 1 Mibibyte, respectively,
 * enabling a "mining advantage" with the reuse of generated tiles.
*/

/* include guard */
#ifndef MOCHIMO_PEACH_H
#define MOCHIMO_PEACH_H


#include <string.h>  /* for mem handling */
#include <time.h>    /* for clock() timing */
#include "extint.h"  /* for word types */
#include "types.h"   /* for Mochimo types */
#include "trigg.h"   /* for BTRAILER, generation and evaluation */


/**
 * Number of rounds of Peach hashing (Nighthash) to arrive at a result.
*/
#define PEACHROUNDS      8

/**
 * The initial length of input data hashed when generating a tile.
 * HASHLEN + 4
*/
#define PEACHGENLEN      36

/**
 * The initial length of input data hashed when jumping tiles.
 * HASHLEN + 4 + PEACH_TILE
*/
#define PEACHJUMPLEN     1060

/**
 * Peach Map length (1 GiByte), in bytes.
 * PEACHCACHELEN * PEACHTILELEN or 1048576 * 1024
*/
#define PEACHMAPLEN      1073741824

/**
 * Peach Map Cache length (1 MiByte), in bytes.
 * PEACHTILELEN * PEACHTILELEN or 1024 * 1024.
*/
#define PEACHCACHELEN    1048576

/**
 * Peach Map Cache length, PEACHCACHELEN, minus 1. Used primarily for
 * restricting results to the bounds of [0 - PEACHCACHELEN], without using
 * a modulo operator. Example:
 * @code int cache_idx = result & PEACHCACHELEN_M1; @endcode
*/
#define PEACHCACHELEN_M1 1048575

/**
 * 64-bit variant of Peach Map Cache length, PEACHCACHELEN. Used primarily
 * for iterating through Peach Map Cache in 64-bit chunks.
*/
#define PEACHCACHELEN64  131072

/**
 * Peach Tile length (1 KiByte), in bytes.
*/
#define PEACHTILELEN     1024

/**
 * 64-bit variant of Peach Map Tile length, PEACHTILELEN. Used primarily
 * for looping copies with 32-bit pointer to a Peach Map tile.
*/
#define PEACHTILELEN32  256

/**
 * 64-bit variant of Peach Map Tile length, PEACHTILELEN. Used primarily
 * for selecting a 64-bit pointer to a Peach Map tile.
*/
#define PEACHTILELEN64  128

/**
 * Check the Peach Proof of Work of a Block Trailer is valid. Checks Proof
 * of Work against the difficulty within the block trailer and ignores the
 * final hash
*/
#define peach_check(btp)  peach_checkhash(btp, (btp)->difficulty[0], NULL)

/* C/C++ compatible function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

int peach_checkhash(const BTRAILER *bt, word8 diff, void *out);
int peach_init(const BTRAILER *bt);
int peach_solve(const BTRAILER *bt, word8 diff, void *out);

/* CUDA functions */
int peach_checkhash_cuda(int count, BTRAILER bt[], void *out);
int peach_init_cuda_device(DEVICE_CTX *devp);
int peach_solve_cuda(DEVICE_CTX *dev, BTRAILER *bt, word8 diff, BTRAILER *out);

#ifdef __cplusplus
}  /* end extern "C" */
#endif

/* end include guard */
#endif
