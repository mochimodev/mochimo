/**
 * @private
 * @headerfile peach.cuh <peach.cuh>
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
 * @note A note on variable naming in this file...
 * [2] suffix implies dual stream variables; for maximum GPU utilisation
 * h_ prefix implies host memory; requires cudaMallocHost()
 * c_ prefix implies (constant) device memory
 * g_ prefix implies (global) device memory
 * d_ prefix implies device memory
 * *h_ pointers require cudaMallocHost();
 * *d_/*g_ pointers require cudaMalloc();
*/

/* include guard */
#ifndef MOCHIMO_PEACH_CU
#define MOCHIMO_PEACH_CU


#include "peach.cuh"

/* external support */
#include "extint.h"
#include "extmath.h"
#include "error.h"
/* external support -- Nighthash */
#include "blake2b.h"
#include "md2.h"
#include "md5.h"
#include "sha1.h"
#include "sha256.cu"
#include "sha3.h"

/**
 * @private
 * Definitions for embedding strings.
*/
#define cuSTRING(x) #x
#define cuSTR(x) cuSTRING(x)

#define cu__log_error(err) \
   do { \
      palert("CUDA ERROR: (%d) %s", (int) err, cudaGetErrorString(err)); \
      set_errno(EMCM_CUDA); \
   } while(0);

/* sm_61 performs MUCH better with the __constant__ qualifier */
#if __CUDA_ARCH__ == 610
   #define cuCONSTn860 __constant__
#else
   #define cuCONSTn860
#endif

/**
 * @private
 * Peach CUDA context. Managed internally by cross referencing parameters
 * of DEVICE_CTX passed to functions.
*/
typedef struct {
   cudaStream_t stream[2];             /**< asynchronous streams */
   BTRAILER *h_bt[2], *d_bt[2];        /**< BTRAILER (current) */
   word64 *h_solve[2], *d_solve[2];    /**< solve seeds */
   word64 *d_state[2];                 /**< PRNG state */
   word64 *d_map;                      /**< Peach Map */
   word32 *d_phash;                    /**< previous hash */
} PEACH_CUDA_CTX;

/* pointer to peach CUDA context/s */
static PEACH_CUDA_CTX *PeachCudaCTX;

__device__ cuCONSTn860 static word64 Z_ING[32] = {
   18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
   34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 23, 24, 31, 32, 33, 34
};
__device__ cuCONSTn860 static word64 Z_NS[64] = {
   129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 145, 149, 154,
   155, 156, 157, 177, 178, 179, 180, 182, 183, 184, 185, 186, 187,
   188, 189, 190, 191, 192, 193, 194, 196, 197, 198, 199, 200, 201,
   202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 241,
   244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255
};
__device__ cuCONSTn860 static word64 Z_MASS[32] = {
   214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224,
   225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235,
   236, 237, 238, 239, 240, 242, 214, 215, 216, 219
};
__device__ cuCONSTn860 static word64 Z_PREP[8] = {
   12, 13, 14, 15, 16, 17, 12, 13
};
__device__ cuCONSTn860 static word64 Z_ADJ[64] = {
   61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
   76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
   91, 92, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104,
   105, 107, 108, 109, 110, 112, 114, 115, 116, 117, 118,
   119, 120, 121, 122, 123, 124, 125, 126, 127, 128
};

/**
 * @private
 * 256-bit Blake2b (w/ key) computation optimized for the Peach algorithm.
 * Places the resulting hash in @a out.
 * @param in Pointer to data to hash
 * @param inlen Length of @a in data, in bytes
 * @param keylen Length of optional @a key input, in bytes
 * @param out Pointer to location to place the message digest
*/
__device__ void cu_peach_blake2b(const word64 *in, size_t inlen, int keylen,
   word64 *out)
{
   /* Blake2b compression constant */
   cuCONSTn860 static word8 c_sigma[12][16] = {
      { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
      { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
      { 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
      { 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
      { 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
      { 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
      { 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
      { 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
      { 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
      { 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
      { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
      { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 }
   };

   /* blake2b_init - outlen is always 256-bits in Peach */
   word64 v[16];
   word64 state[8];
   word64 final[16];
   word64 t[2] = { 128, 0 };

   /* FAST-FORWARD state to known keylen states */
   if (keylen == 64) {
      state[0] = WORD64_C(0x00B8AA23C261EF69);
      state[1] = WORD64_C(0xD38AE6ABCA237B9E);
      state[2] = WORD64_C(0x67FB881E5EE89069);
      state[3] = WORD64_C(0x3E5B8BD06B58D002);
      state[4] = WORD64_C(0x252D3F68395AAE91);
      state[5] = WORD64_C(0xD25465E23C6C1B27);
      state[6] = WORD64_C(0x852B4CC2E13303B5);
      state[7] = WORD64_C(0x3F38B9FF245BE7C1);
   } else {
      state[0] = WORD64_C(0x63320ACE264383EB);
      state[1] = WORD64_C(0x012AF5FD045A2737);
      state[2] = WORD64_C(0xF4F49C55E6BE39DF);
      state[3] = WORD64_C(0x791C5BC8AFFB11A7);
      state[4] = WORD64_C(0xC9BCACC002C0EA21);
      state[5] = WORD64_C(0x8295B8ABE2FDEDD6);
      state[6] = WORD64_C(0xB711490E5F9F41C8);
      state[7] = WORD64_C(0x3F8E4D1D9EBEAF1A);
   }

   /* blake2b_update */
   for(; inlen > 128; inlen -= 128, in = &in[16]) {
      t[0] += 128;
      blake2b_compress_init(v, state, t, 0);
      blake2b_compress_rounds(v, in, c_sigma);
      blake2b_compress_set(v, state);
   }

   /* blake2b_final - somewhat conveniently (and exclusive to Peach)...
    * the remaining datalen will always be 36... */
   final[0] = in[0];
   final[1] = in[1];
   final[2] = in[2];
   final[3] = in[3];
   final[4] = (word64) ((word32 *) in)[8];
   final[5] = 0;
   final[6] = 0;
   final[7] = 0;
   final[8] = 0;
   final[9] = 0;
   final[10] = 0;
   final[11] = 0;
   final[12] = 0;
   final[13] = 0;
   final[14] = 0;
   final[15] = 0;

   t[0] += 36;
   blake2b_compress_init(v, state, t, 1);
   blake2b_compress_rounds(v, final, c_sigma);

   /* blake2b_output */
   out[0] = state[0] ^ v[0] ^ v[8];
   out[1] = state[1] ^ v[1] ^ v[9];
   out[2] = state[2] ^ v[2] ^ v[10];
   out[3] = state[3] ^ v[3] ^ v[11];
}  /* end cu_peach_blake2b() */

/**
 * @private
 * 128-bit MD2 computation optimized for the Peach algorithm.
 * Places the resulting hash in @a out.
 * @param in Pointer to data to hash
 * @param inlen Length of @a in data, in bytes
 * @param out Pointer to location to place the message digest
*/
__device__ void cu_peach_md2(const word64 *in, size_t inlen, word64 *out)
{
   /* MD2 transformation constant */
   cuCONSTn860 static word8 s[256] = {
      41, 46, 67, 201, 162, 216, 124, 1, 61, 54, 84, 161, 236, 240, 6,
      19, 98, 167, 5, 243, 192, 199, 115, 140, 152, 147, 43, 217, 188,
      76, 130, 202, 30, 155, 87, 60, 253, 212, 224, 22, 103, 66, 111, 24,
      138, 23, 229, 18, 190, 78, 196, 214, 218, 158, 222, 73, 160, 251,
      245, 142, 187, 47, 238, 122, 169, 104, 121, 145, 21, 178, 7, 63,
      148, 194, 16, 137, 11, 34, 95, 33, 128, 127, 93, 154, 90, 144, 50,
      39, 53, 62, 204, 231, 191, 247, 151, 3, 255, 25, 48, 179, 72, 165,
      181, 209, 215, 94, 146, 42, 172, 86, 170, 198, 79, 184, 56, 210,
      150, 164, 125, 182, 118, 252, 107, 226, 156, 116, 4, 241, 69, 157,
      112, 89, 100, 113, 135, 32, 134, 91, 207, 101, 230, 45, 168, 2, 27,
      96, 37, 173, 174, 176, 185, 246, 28, 70, 97, 105, 52, 64, 126, 15,
      85, 71, 163, 35, 221, 81, 175, 58, 195, 92, 249, 206, 186, 197,
      234, 38, 44, 83, 13, 110, 133, 40, 132, 9, 211, 223, 205, 244, 65,
      129, 77, 82, 106, 220, 55, 200, 108, 193, 171, 250, 36, 225, 123,
      8, 12, 189, 177, 74, 120, 136, 149, 139, 227, 99, 232, 109, 233,
      203, 213, 254, 59, 0, 29, 57, 242, 239, 183, 14, 102, 88, 208, 228,
      166, 119, 114, 248, 235, 117, 75, 10, 49, 68, 80, 180, 143, 237,
      31, 26, 219, 153, 141, 51, 159, 17, 131, 20
   };

   /* md2_init */
   word64 state[6] = { 0 };
   word64 checksum[2] = { 0 };
   word64 pad64;
   word8 pad;

   /* prepare padding */
   pad = 16 - (inlen & 0xf);
   pad64 = pad | pad << 8;
   pad64 = pad64 | pad64 << 16;
   pad64 = pad64 | pad64 << 32;

   /* md2_update */
   for (; inlen >= 16; inlen -= 16, in = &in[2]) {
      md2_transform_init64(state, in);
      md2_transform_checksum(((word8 *) checksum), ((word8 *) in), s);
      md2_transform_state(((word8 *) state), s);
   }

   /* md2_final - only 4 bytes left, so 12 remaining bytes are pad */
   state[4] = (state[2] = *((word32 *) in) | (pad64 << 32)) ^ state[0];
   state[5] = (state[3] = pad64) ^ state[1];
   /* final transform part1 */
   md2_transform_checksum(((word8 *) checksum), ((word8 *) &state[2]), s);
   md2_transform_state(((word8 *) state), s);
   /* final transform part2 */
   md2_transform_init64(state, checksum);
   md2_transform_state(((word8 *) state), s);

   /* MD2 hash = 128 bits, zero fill remaining... */
   out[0] = state[0];
   out[1] = state[1];
   out[2] = 0;
   out[3] = 0;
}  /* end cu_peach_md2 */

/**
 * @private
 * 128-bit MD5 computation optimized for the Peach algorithm.
 * Places the resulting hash in @a out.
 * @param in Pointer to data to hash
 * @param inlen Length of @a in data, in bytes
 * @param out Pointer to location to place the message digest
*/
__device__ void cu_peach_md5(const word32 *in, size_t inlen, word64 *out)
{
   /* md5_init */
   word32 final[16];
   word32 state[4] = { 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476 };

   /* prepare bitlen in final data */
   final[14] = inlen << 3;

   /* md5_update */
   for (; inlen >= 64; inlen -= 64, in = &in[16]) {
      md5_tranform_unrolled(state, in);
   }

   /* md5_final - somewhat conveniently (and exclusive to Peach)...
    * the remaining datalen will always be 36, so:
    * in[9] = 0x80; and in[10+] = 0; */
   final[0] = in[0];
   final[1] = in[1];
   final[2] = in[2];
   final[3] = in[3];
   final[4] = in[4];
   final[5] = in[5];
   final[6] = in[6];
   final[7] = in[7];
   final[8] = in[8];
   final[9] = 0x80;
   final[10] = 0;
   final[11] = 0;
   final[12] = 0;
   final[13] = 0;
   final[15] = 0;

   md5_tranform_unrolled(state, final);

   /* MD5 hash = 128 bits, zero fill remaining... */
   out[0] = ((word64 *) state)[0];
   out[1] = ((word64 *) state)[1];
   out[2] = 0;
   out[3] = 0;
}  /* end cuda_peach_md5 */

/**
 * @private
 * 160-bit Sha1 computation optimized for the Peach algorithm.
 * Places the resulting hash in @a out.
 * @param in Pointer to data to hash
 * @param inlen Length of @a in data, in bytes
 * @param out Pointer to location to place the message digest
*/
__device__ void cu_peach_sha1(const word32 *in, size_t inlen, word32 *out)
{
   /* SHA1 transformation constant */
   cuCONSTn860 static word32 c_k[4] =
      { 0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xca62c1d6 };
   /* sha1_init */
   word32 state[5] =
      { 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0 };
   word32 final[16];

   final[15] = inlen << 3;
   final[15] = bswap32(final[15]);

   /* sha1_update */
   for(; inlen >= 64; inlen -= 64, in = &in[16]) {
      sha1_transform_unrolled(state, in, c_k);
   }

   /* sha1_final - somewhat conveniently (and exclusive to Peach)...
    * the remaining datalen will always be 36, so in[9] = 0x80. */
   final[0] = in[0];
   final[1] = in[1];
   final[2] = in[2];
   final[3] = in[3];
   final[4] = in[4];
   final[5] = in[5];
   final[6] = in[6];
   final[7] = in[7];
   final[8] = in[8];
   final[9] = 0x80;
   final[10] = 0;
   final[11] = 0;
   final[12] = 0;
   final[13] = 0;
   final[14] = 0;

   sha1_transform_unrolled(state, final, c_k);

   /* SHA1 hash = 160 bits, zero fill remaining... */
   out[0] = bswap32(state[0]);
   out[1] = bswap32(state[1]);
   out[2] = bswap32(state[2]);
   out[3] = bswap32(state[3]);
   out[4] = bswap32(state[4]);
   out[5] = 0;
   out[6] = 0;
   out[7] = 0;
}  /* end cu_peach_sha1() */

/**
 * @private
 * 256-bit SHA256 computation optimized for the Peach algorithm.
 * Places the resulting hash in @a out.
 * @param in Pointer to data to hash
 * @param inlen Length of @a in data, in bytes
 * @param out Pointer to location to place the message digest
*/
__device__ void cu_peach_sha256(const word32 *in, size_t inlen, word32 *out)
{
   /* SHA256 transformation constant */
   cuCONSTn860 static word32 c_k[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
      0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
      0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
      0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
      0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
      0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
      0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
      0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
      0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
      0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
      0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
      0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
      0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
      0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
      0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
      0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
   };

   /* sha256_init */
   word32 final[16];
   word32 state[8] = {
      0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
   };

   /* prepare bitlen in final */
   final[15] = inlen << 3;
   final[15] = bswap32(final[15]);

   /* sha256_update */
   for(; inlen >= 64; inlen -= 64, in = &in[16]) {
      sha256_tranform_unrolled(state, in, c_k);
   }

   /* sha256_final - somewhat conveniently (and exclusive to Peach)...
    * the remaining datalen will always be 36, so in[9] = 0x80. */
   final[0] = in[0];
   final[1] = in[1];
   final[2] = in[2];
   final[3] = in[3];
   final[4] = in[4];
   final[5] = in[5];
   final[6] = in[6];
   final[7] = in[7];
   final[8] = in[8];
   final[9] = 0x80;
   final[10] = 0;
   final[11] = 0;
   final[12] = 0;
   final[13] = 0;
   final[14] = 0;
   sha256_tranform_unrolled(state, final, c_k);

   /* Since this implementation uses little endian byte ordering and
    * SHA uses big endian, reverse all the bytes when copying the
    * final state to the output hash. */
   out[0] = bswap32(state[0]);
   out[1] = bswap32(state[1]);
   out[2] = bswap32(state[2]);
   out[3] = bswap32(state[3]);
   out[4] = bswap32(state[4]);
   out[5] = bswap32(state[5]);
   out[6] = bswap32(state[6]);
   out[7] = bswap32(state[7]);
}  /* end cu_peach_sha256() */

/**
 * @private
 * 256-bit Sha3 (Keccak) computation optimized for the Peach algorithm.
 * Places the resulting hash in @a out.
 * @param in Pointer to data to hash
 * @param inlen Length of @a in data, in bytes
 * @param keccak_final Flag indicates hash should be finalized as Keccak
 * @param out Pointer to location to place the message digest
*/
__device__ void cu_peach_sha3(const word64 *in, size_t inlen,
   int keccak_final, word64 *out)
{
   /* Keccak permutation constant */
   cuCONSTn860 static word64 keccakf_rndc[24] = {
      WORD64_C(0x0000000000000001), WORD64_C(0x0000000000008082),
      WORD64_C(0x800000000000808a), WORD64_C(0x8000000080008000),
      WORD64_C(0x000000000000808b), WORD64_C(0x0000000080000001),
      WORD64_C(0x8000000080008081), WORD64_C(0x8000000000008009),
      WORD64_C(0x000000000000008a), WORD64_C(0x0000000000000088),
      WORD64_C(0x0000000080008009), WORD64_C(0x000000008000000a),
      WORD64_C(0x000000008000808b), WORD64_C(0x800000000000008b),
      WORD64_C(0x8000000000008089), WORD64_C(0x8000000000008003),
      WORD64_C(0x8000000000008002), WORD64_C(0x8000000000000080),
      WORD64_C(0x000000000000800a), WORD64_C(0x800000008000000a),
      WORD64_C(0x8000000080008081), WORD64_C(0x8000000000008080),
      WORD64_C(0x0000000080000001), WORD64_C(0x8000000080008008)
   };

   /* sha3_init */
   word8 state[200] = { 0 };
   word64 *st64 = (word64 *) state;
	int i;

   /* sha3_update - 136 is ctx->rsiz, fill only 17x 64-bit words in state */
   for(; inlen >= 136; inlen -= 136, in = &in[17]) {
      for (i = 0; i < 17; i++) st64[i] ^= in[i];
	   sha3_keccakf_unrolled(st64, keccakf_rndc);
   }

   /* sha3_final */
   st64[0] ^= in[0];
   st64[1] ^= in[1];
   st64[2] ^= in[2];
   st64[3] ^= in[3];
   if (inlen > PEACHGENLEN) {
      st64[4] ^= in[4];
      st64[5] ^= in[5];
      st64[6] ^= in[6];
      st64[7] ^= in[7];
      st64[8] ^= in[8];
      st64[9] ^= in[9];
      st64[10] ^= in[10];
      st64[11] ^= in[11];
      st64[12] ^= in[12];
      ((word32 *) st64)[26] ^= ((word32 *) in)[26];
      state[108] ^= keccak_final ? 0x01 : 0x06;
   } else {
      ((word32 *) st64)[8] ^= ((word32 *) in)[8];
      state[36] ^= keccak_final ? 0x01 : 0x06;
   }
   state[135] ^= 0x80;
	sha3_keccakf_unrolled(st64, keccakf_rndc);

   /* sha3_output */
   out[0] = st64[0];
   out[1] = st64[1];
   out[2] = st64[2];
   out[3] = st64[3];
}  /* end cu_peach_sha3 */

/**
 * @private
 * Perform deterministic (single precision) floating point operations on
 * @a len bytes of @a data (in 4 byte operations).
 * @param data Pointer to data to use in operations
 * @param len Length of @a data to use in operations
 * @param index Peach tile index number
 * @param txf Flag indicates @a data should be transformed by operations
 * @returns 32-bit unsigned operation code for subsequent Peach algo steps
 * @note Operations are guaranteed "deterministic" within the Peach
 * algorithm for all IEEE-754 compliant hardware on "round-to-nearest"
 * rounding mode. This is ensured by the use of CUDA intrinsics:
 * - __fdiv_rn(), __fmul_rn(), __fsub_rn(), __fadd_rn() operations, and
 * - __int2float_rn(), __uint2float_rn() conversions
*/
__device__ word32 cu_peach_dflops(void *data, size_t len,
   word32 index, int txf)
{
   cuCONSTn860 static word32 c_float[4] = {
      WORD32_C(0x26C34), WORD32_C(0x14198),
      WORD32_C(0x3D6EC), WORD32_C(0x80000000)
   };
   word8 *bp;
   float *flp, temp, flv;
   int32 operand;
   word32 op;
   unsigned i;
   word8 shift;

   /* process entire length of input data; limit to 4 byte multiples */
   /* len = len - (len & 3); // uncomment if (len % 4 != 0) is expected */
   for (op = i = 0; i < len; i += 4) {
      bp = &((word8 *) data)[i];
      if (txf) {
         /* input data is modified directly */
         flp = (float *) bp;
      } else {
         /* temp variable is modified, input data is unchanged */
         temp = *((float *) bp);
         flp = &temp;
      }
      /* first byte allocated to determine shift amount */
      shift = ((*bp & 7) + 1) << 1;
      /* remaining bytes are selected for 3 different operations based on
       * the first bytes resulting shift on precomputed contants to...
       * ... 1) determine the floating point operation type */
      op += bp[((c_float[0] >> shift) & 3)];
      /* ... 2) determine the value of the operand */
      operand = bp[((c_float[1] >> shift) & 3)];
      /* ... 3) determine the upper most bit of the operand
       *        NOTE: must be performed AFTER the allocation of the operand */
      if (bp[((c_float[2] >> shift) & 3)] & 1) operand ^= c_float[3];
      /* interpret operand as SIGNED integer and cast to float */
      flv = __int2float_rn(operand);
      /* Replace pre-operation NaN with index */
      if (isnan(*flp)) *flp = __uint2float_rn(index);
      /* Perform predetermined floating point operation */
      switch (op & 3) {
         case 3: *flp = __fdiv_rn(*flp, flv);  break;
         case 2: *flp = __fmul_rn(*flp, flv);  break;
         case 1: *flp = __fsub_rn(*flp, flv);  break;
         case 0: *flp = __fadd_rn(*flp, flv);  break;
      }
      /* Replace post-operation NaN with index */
      if (isnan(*flp)) *flp = __uint2float_rn(index);
      /* Add result of the operation to `op` as an array of bytes */
      bp = (word8 *) flp;
      op += bp[0];
      op += bp[1];
      op += bp[2];
      op += bp[3];
   }  /* end for(i = 0; ... */

   return op;
}  /* end cu_peach_dflops() */

/**
 * @private
 * Perform deterministic memory transformations on @a len bytes of @a data.
 * @param data Pointer to data to use in operations
 * @param len Length of @a data to use in operations
 * @param op Operating code from previous Peach algo steps
 * @returns 32-bit unsigned operation code for subsequent Peach algo steps
*/
__device__ word32 cu_peach_dmemtx(void *data, size_t len, word32 op)
{
   cuCONSTn860 static word64 c_flip64 = WORD64_C(0x8181818181818181);
   cuCONSTn860 static word32 c_flip32 = WORD64_C(0x81818181);
   word64 *qp = (word64 *) data;
   word32 *dp = (word32 *) data;
   word8 *bp = (word8 *) data;
   size_t len16, len32, len64, y;
   unsigned i, z;
   word8 temp;

   /* prepare memory pointers and lengths */
   len64 = (len32 = (len16 = len >> 1) >> 1) >> 1;
   /* perform memory transformations multiple times */
   for (i = 0; i < PEACHROUNDS; i++) {
      /* determine operation to use for this iteration */
      op += bp[i];
      /* select "random" transformation based on value of `op` */
      switch (op & 7) {
         case 0:  /* flip the first and last bit in every byte */
            for (z = 0; z < len64; z++) qp[z] ^= c_flip64;
            for (z <<= 1; z < len32; z++) dp[z] ^= c_flip32;
            break;
         case 1:  /* Swap bytes */
            for (y = len16, z = 0; z < len16; y++, z++) {
               temp = bp[z]; bp[z] = bp[y]; bp[y] = temp;
            }
            break;
         case 2:  /* 1's complement, all bytes */
            for (z = 0; z < len64; z++) qp[z] = ~qp[z];
            for (z <<= 1; z < len32; z++) dp[z] = ~dp[z];
            break;
         case 3:  /* Alternate +1 and -1 on all bytes */
            for (z = 0; z < len; z++) bp[z] += (z & 1) ? -1 : 1;
            break;
         case 4:  /* Alternate -i and +i on all bytes */
            for (z = 0; z < len; z++) bp[z] += (word8) ((z & 1) ? i : -i);
            break;
         case 5:  /* Replace every occurrence of 104 with 72 */ 
            for (z = 0; z < len; z++) if(bp[z] == 104) bp[z] = 72;
            break;
         case 6:  /* If byte a is > byte b, swap them. */
            for (y = len16, z = 0; z < len16; y++, z++) {
               if(bp[z] > bp[y]) {
                  temp = bp[z]; bp[z] = bp[y]; bp[y] = temp;
               }
            }
            break;
         case 7:  /* XOR all bytes */
            for (y = 0, z = 1; z < len; y++, z++) bp[z] ^= bp[y];
            break;
      } /* end switch(op & 7)... */
   } /* end for(i = 0; ... */

   return op;
}  /* end cu_peach_dmemtx() */

/**
 * @private
 * Perform Nighthash on @a inlen bytes of @a in and place result in @a out.
 * Utilizes deterministic float operations and memory transformations.
 * @param in Pointer to input data
 * @param inlen Length of data from @a in, used in non-transform steps
 * @param index Peach tile index number
 * @param txlen Length of data from @a in, used in transform steps
 * @param out Pointer to location to place resulting hash
*/
__device__ void cu_peach_nighthash(word64 *in, size_t inlen,
   word32 index, size_t txlen, word64 *out)
{
   /* Perform flops to determine initial algo type.
    * When txlen is non-zero the transformation of input data is enabled,
    * as well as the additional memory transformation process. */
   if (txlen) {
      index = cu_peach_dflops(in, txlen, index, 1);
      index = cu_peach_dmemtx(in, txlen, index);
   } else index = cu_peach_dflops(in, inlen, index, 0);

   /* reduce algorithm selection to 1 of 8 choices */
   switch (index & 7) {
      case 0: cu_peach_blake2b(in, inlen, 32, out); break;
      case 1: cu_peach_blake2b(in, inlen, 64, out); break;
      case 2: cu_peach_sha1((word32 *) in, inlen, (word32 *) out); break;
      case 3: cu_peach_sha256((word32 *) in, inlen, (word32 *) out); break;
      case 4: cu_peach_sha3(in, inlen, 0, out); break;
      case 5: cu_peach_sha3(in, inlen, 1, out); break;
      case 6: cu_peach_md2(in, inlen, out); break;
      case 7: cu_peach_md5((word32 *) in, inlen, out); break;
   }  /* end switch(algo_type)... */
}  /* end cu_peach_nighthash() */

/**
 * @private
 * Generate a tile of the Peach map.
 * @param index Index number of tile to generate
 * @param tilep Pointer to location to place generated tile
*/
__device__ void cu_peach_generate
   (word32 index, word64 *tilep, word32 *phash)
{
   int i;

   /* place initial data into seed */
   ((word32 *) tilep)[0] = index;
   ((word32 *) tilep)[1] = phash[0];
   ((word32 *) tilep)[2] = phash[1];
   ((word32 *) tilep)[3] = phash[2];
   ((word32 *) tilep)[4] = phash[3];
   ((word32 *) tilep)[5] = phash[4];
   ((word32 *) tilep)[6] = phash[5];
   ((word32 *) tilep)[7] = phash[6];
   ((word32 *) tilep)[8] = phash[7];
   /* perform initial nighthash into first row of tile */
   cu_peach_nighthash(tilep, PEACHGENLEN, index, PEACHGENLEN, tilep);
   /* fill the rest of the tile with the preceding Nighthash result */
   for (i = 0; i < (PEACHTILELEN64 - 4); i += 4) {
      tilep[i + 4] = index;
      cu_peach_nighthash(&tilep[i], PEACHGENLEN, index, SHA256LEN,
         &tilep[i + 4]);
   }
}  /* end cu_peach_generate() */

/**
 * @private
 * Perform an index jump using the hash result of the Nighthash function.
 * @param index Index number of (current) tile on Peach map
 * @param nonce Nonce for use as entropy in jump direction
 * @param tilep Pointer to tile data at @a index
 * @returns 32-bit unsigned index of next tile
*/
__device__ void cu_peach_jump(word32 *index, word64 *nonce, word64 *tilep)
{
   word64 seed[(PEACHJUMPLEN / 8) + 1];
   word32 *dp = (word32 *) seed;
   int i;

   /* construct seed for use as Nighthash input for this index on the map */
   seed[0] = nonce[0];
   seed[1] = nonce[1];
   seed[2] = nonce[2];
   seed[3] = nonce[3];
   dp[8] = *index;
#pragma unroll
   for (i = 0; i < PEACHTILELEN32; i++) {
      dp[i + 9] = ((word32 *) tilep)[i];
   }

   /* perform nighthash on PEACHJUMPLEN bytes of seed */
   cu_peach_nighthash(seed, PEACHJUMPLEN, *index, 0, seed);
   /* sum hash as 8x 32-bit unsigned integers */
   *index = dp[0] + dp[1] + dp[2] + dp[3] + dp[4] + dp[5] + dp[6] + dp[7];
   *index &= PEACHCACHELEN_M1;
}  /* end cu_peach_jump() */

/**
 * CUDA kernel for bulk generation of Peach Map tiles.
 * @param d_map Device pointer to location of Peach Map
 * @param offset Index number offset to generate tiles from
 */
__global__ void kcu_peach_build
   (word32 offset, word64 *d_map, word32 *d_phash)
{
   const word32 index = ((blockDim.x * blockIdx.x) + threadIdx.x) + offset;
   if (index < PEACHCACHELEN) {
      cu_peach_generate(index, &d_map[index * PEACHTILELEN64], d_phash);
   }
}  /* end kcu_peach_build() */

/**
 * CUDA kernel to expand a seed into a long state for parallel cu_rand64().
 * State generation based on SplitMix64 by Sebastiano Vigna.
 * @param d_seed Device pointer to location of state
 * @param seed 64-bit unsigned integer value of seed
*/
__global__ void kcu_srand64(word64 *d_state, word64 seed)
{
   word64 index = (blockDim.x * blockIdx.x) + threadIdx.x;
   d_state[index] = (seed ^ (index * WORD64_C(0x9e3779b97f4a7c15))) * WORD64_C(0xc6bc279692b5c323);
}  /* end kcu_srand64() */

/**
 * CUDA device function to generate a 64-bit random number.
 * State generation based on SplitMix64 by Sebastiano Vigna.
 * @param d_state Pointer to location of state
 * @return (word64) value representing a 64-bit random number
 */
__device__ __forceinline__ word64 cu_rand64(word64 *d_state)
{
   word64 index = (blockIdx.x * blockDim.x) + threadIdx.x;
   word64 z = (d_state[index] += WORD64_C(0x9e3779b97f4a7c15));
	z = (z ^ (z >> 30)) * WORD64_C(0xbf58476d1ce4e5b9);
	z = (z ^ (z >> 27)) * WORD64_C(0x94d049bb133111eb);
	return (d_state[index] = z ^ (z >> 31));
}  /* end cu_rand64() */

/**
 * CUDA kernel for solving a tokenized haiku as nonce output for Peach proof
 * of work. Combine haiku protocols implemented in the Trigg Algorithm with
 * the memory intensive protocols of the Peach algorithm to generate haiku
 * output as proof of work.
 * @param d_map Device pointer to Peach Map
 * @param d_ictx Device pointer to incomplete hashing contexts
 * @param d_solve Device pointer to location to place nonce on solve
*/
__global__ void kcu_peach_solve
   (word64 *d_map, BTRAILER *d_bt, word64 *d_state, word8 diff, word64 *d_solve)
{
   SHA256_CTX ictx;
   word64 nonce[4], seed;
   word8 hash[SHA256LEN];
   word32 *x, mario, i;

   /* extract nonce from trailer and seed list*/
   for (i = 0; i < 4; i++) {
      ((word32 *) nonce)[i] = ((word32 *) d_bt->nonce)[i];
   }

   /* generate last half of nonce from seed (w/ largest known frame) */
   seed = cu_rand64(d_state);
   nonce[2] =    WORD64_C(0x10000050000) | /* nonce8bit[2,5] */
          Z_ING[(seed     )  & 31]       | /* nonce8bit[0] */
         Z_PREP[(seed >> 5)  &  7] <<  8 | /* nonce8bit[1] */
          Z_ADJ[(seed >> 8)  & 63] << 24 | /* nonce8bit[3] */
           Z_NS[(seed >> 14) & 63] << 32 | /* nonce8bit[4] */
         Z_MASS[(seed >> 20) & 31] << 48 | /* nonce8bit[6] */
          Z_ING[(seed >> 25) & 31] << 56;        /* nonce8bit[7] */
   nonce[3] =  WORD64_C(0x50103) | /* nonce8bit[8:10] */
          Z_ADJ[(seed >> 30) & 63] << 24 | /* nonce8bit[11] */
           Z_NS[(seed >> 36) & 63] << 32;  /* nonce8bit[12] */

   /* sha256 hash trailer and nonce */
   cu_sha256_init(&ictx);
   cu_sha256_update(&ictx, d_bt, 92);
   cu_sha256_update(&ictx, nonce, 32);
   cu_sha256_final(&ictx, hash);
   /* initialize mario's starting index on the map, bound to PEACHCACHELEN */
   for (mario = hash[0], i = 1; i < SHA256LEN; i++) {
      mario *= hash[i];
   }
   mario &= PEACHCACHELEN_M1;
   /* perform tile jumps to find the final tile x8 */
   for (i = 0; i < PEACHROUNDS; i++) {
      cu_peach_jump(&mario, nonce, &d_map[mario * PEACHTILELEN64]);
   }
   /* hash block trailer with final tile */
   cu_sha256_init(&ictx);
   cu_sha256_update(&ictx, hash, SHA256LEN);
   cu_sha256_update(&ictx, &d_map[mario * PEACHTILELEN64], PEACHTILELEN);
   cu_sha256_final(&ictx, hash);
   /* Coarse/Fine evaluation checks */
   x = (word32 *) hash;
   for (i = diff >> 5; i; i--) if(*(x++) != 0) return;
   if (__clz(__byte_perm(*x, 0, 0x0123)) < (diff & 31)) return;

   /* check first to solve with atomic solve handling */
   if (!atomicCAS((int *) d_solve, 0, *((int *) nonce))) {
      d_solve[0] = nonce[0];
      d_solve[1] = nonce[1];
      d_solve[2] = nonce[2];
      d_solve[3] = nonce[3];
   }
}  /* end kcu_peach_solve() */

/**
 * CUDA kernel for checking Peach Proof-of-Work. The haiku must be
 * syntactically correct AND have the right vibe. Also, entropy MUST match
 * difficulty.
 * @param ictx Device pointer to incomplete hashing context
 * @param out Pointer to location to place final hash
 * @param eval Evaluation result: VEOK on success, else VERROR
*/
__global__ void kcu_peach_checkhash
   (BTRAILER *d_bt, word8 *d_out, word8 *d_eval)
{
   word64 data[(SHA256LEN + PEACHTILELEN) / 8] = { 0 };
   word64 nonce[4];
   BTRAILER *btp;
   word8 *hash = (word8 *) data;
   word64 *tile = (word64 *) &data[SHA256LEN / 8];
   word32 *x, mario;
   unsigned int tid;
   int i;

   /* init */
   tid = (blockDim.x * blockIdx.x) + threadIdx.x;
   btp = &d_bt[tid];

   /* copy nonce */
#pragma unroll
   for (i = 0; i < 8; i++) {
      ((word32 *) nonce)[i] = ((word32 *) btp->nonce)[i];
   }

   /* hash partial trailer */
   cu_sha256(btp, 124, hash);
   /* initialize mario's starting index on the map, bound to PEACHCACHELEN */
   for(mario = hash[0], i = 1; i < SHA256LEN; i++) mario *= hash[i];
   mario &= PEACHCACHELEN_M1;
   /* generate and perform tile jumps to find the final tile x8 */
   for (i = 0; i < PEACHROUNDS; i++) {
      cu_peach_generate(mario, tile, (word32 *) btp->phash);
      cu_peach_jump(&mario, nonce, tile);
   }
   /* generate the last tile */
   cu_peach_generate(mario, tile, (word32 *) btp->phash);
   /* hash bthash and final tile */
   cu_sha256(data, SHA256LEN + PEACHTILELEN, hash);
   /* pass final hash to out */
   memcpy(&d_out[SHA256LEN * tid], hash, SHA256LEN);
   /* Coarse/Fine evaluation checks */
   x = (word32 *) hash;
   i = btp->difficulty[0] >> 5;
   for (; i; i--) if(*(x++) != 0) { *d_eval = 1; return; }
   if (__clz(__byte_perm(*x, 0, 0x0123)) < (btp->difficulty[0] & 31)) {
      *d_eval = 1;
      return;
   }
}  /* end kcu_peach_checkhash() */

/**
 * Check Peach proof of work with a CUDA device.
 * Uses the first available Cuda device to check multiple POW.
 * @param count Number of block trailers to check
 * @param bt Pointer to block trailer array
 * @param out Pointer to final hash array, if non-null
 * @returns VEOK on success, else VERROR
*/
int peach_checkhash_cuda(int count, BTRAILER bt[], void *out)
{
   size_t btsz = sizeof(BTRAILER) * count;
   size_t outsz = SHA256LEN * count;
   BTRAILER *d_bt;
   word8 *d_out, *d_eval;
   word8 eval = 0;
   int cuda_count;

#undef cuCHK
#define cuCHK(cuFN) \
   do { \
      cudaError_t err = (cuFN); \
      if (err != cudaSuccess) { \
         cu__log_error(err); \
         return VERROR; \
      } \
   } while(0)

   cuCHK(cudaGetDeviceCount(&cuda_count));
   if (cuda_count < 1) {
      palert("No CUDA devices...");
      return -1;
   }
   cuCHK(cudaSetDevice(0));
   cuCHK(cudaMalloc(&d_bt, btsz));
   cuCHK(cudaMalloc(&d_out, outsz));
   cuCHK(cudaMalloc(&d_eval, 1));
   /* transfer data to device */
   cuCHK(cudaMemcpy(d_bt, bt, btsz, cudaMemcpyHostToDevice));
   cuCHK(cudaMemset(d_out, 0, outsz));
   cuCHK(cudaMemset(d_eval, 0, 1));
   /* launch kernel to check Peach */
   kcu_peach_checkhash<<<1, count>>>(d_bt, d_out, d_eval);
   cuCHK(cudaGetLastError());
   /* retrieve hash/eval data */
   cuCHK(cudaMemcpy(out, d_out, outsz, cudaMemcpyDeviceToHost));
   cuCHK(cudaMemcpy(&eval, d_eval, 1, cudaMemcpyDeviceToHost));
   /* wait for device to finish */
   cuCHK(cudaDeviceSynchronize());
   /* free memory */
   cuCHK(cudaFree(d_bt));
   cuCHK(cudaFree(d_out));
   cuCHK(cudaFree(d_eval));

   return (int) eval;
}  /* end peach_checkhash_cuda() */

/**
 * Free CUDA memory allocated to a previously initialized device context.
 * @param devp Pointer to DEVICE_CTX to free
 * @returns VEOK on valid DEVICE_CTX pointer, else VERROR
*/
int peach_free_cuda_device(DEVICE_CTX *devp, int status)
{
   /* check device pointer */
   if (devp == NULL) return VERROR;
   /* set device status */
   devp->status = status;
   /* free pointers -- if set */
   PEACH_CUDA_CTX *ctxp = &PeachCudaCTX[devp->id];
   if (ctxp->stream[0]) cudaStreamDestroy(ctxp->stream[0]);
   if (ctxp->stream[1]) cudaStreamDestroy(ctxp->stream[1]);
   if (ctxp->h_solve[0]) cudaFreeHost(ctxp->h_solve[0]);
   if (ctxp->h_solve[1]) cudaFreeHost(ctxp->h_solve[1]);
   if (ctxp->h_bt[0]) cudaFreeHost(ctxp->h_bt[0]);
   if (ctxp->h_bt[1]) cudaFreeHost(ctxp->h_bt[1]);
   if (ctxp->d_solve[0]) cudaFree(ctxp->d_solve[0]);
   if (ctxp->d_solve[1]) cudaFree(ctxp->d_solve[1]);
   if (ctxp->d_state[0]) cudaFree(ctxp->d_state[0]);
   if (ctxp->d_state[1]) cudaFree(ctxp->d_state[1]);
   if (ctxp->d_bt[0]) cudaFree(ctxp->d_bt[0]);
   if (ctxp->d_bt[1]) cudaFree(ctxp->d_bt[1]);
   if (ctxp->d_phash) cudaFree(ctxp->d_phash);
   if (ctxp->d_map) cudaFree(ctxp->d_map);
   /* attempt to clear last error */
   (void) cudaGetLastError();

   return VEOK;
}  /* end peach_free_cuda_device() */

/**
 * (re)Initialize a device context with a CUDA device.
 * @param devp Pointer to DEVICE_CTX to initialize
 * @param id Index of CUDA device to initialize to DEVICE_CTX
 * @returns VEOK on success, else VERROR
 * @note The `id` parameter of the DEVICE_CTX must be set to an appropriate
 * CUDA device number. If not performing a re-initialization, recommend
 * using peach_init_cuda() first.
*/
int peach_init_cuda_device(DEVICE_CTX *ctx)
{
   PEACH_CUDA_CTX *p_ctx;
   size_t btsz, seedsz;

#undef cuCHK
#define cuCHK(cuFN) \
   do { \
      cudaError_t err = (cuFN); \
      if (err != cudaSuccess) { \
         ctx->status = DEV_FAIL; \
         cu__log_error(err); \
         return VERROR; \
      } \
   } while(0)

   /* check for double init */
   if (ctx->peach) {
      set_errno(EINVAL);
      return VERROR;
   }

   /* allocate peach context */
   p_ctx = (PEACH_CUDA_CTX *) malloc(sizeof(PEACH_CUDA_CTX));
   if (p_ctx == NULL) return VERROR;
   ctx->peach = p_ctx;

   /* set context to CUDA id */
   cuCHK(cudaSetDevice(ctx->id));
   /* set CUDA configuration for device */
   if (cudaOccupancyMaxPotentialBlockSize(&(ctx->grid), &(ctx->block),
         kcu_peach_solve, 0, 0) != cudaSuccess) {
      pdebug("cudaOccupancy~BlockSize(%d) failed...", ctx->id);
      pdebug("Using conservative defaults for <<<512/128>>>");
      ctx->grid = 512;
      ctx->block = 128;
   }
   /* calculate total threads */
   ctx->threads = ctx->grid * ctx->block;
   seedsz = sizeof(word64) * ctx->threads;
   btsz = sizeof(BTRAILER);
   /* create generator for unsigned long long */
   /* create streams for device */
   cuCHK(cudaStreamCreate(&(p_ctx->stream[0])));
   cuCHK(cudaStreamCreate(&(p_ctx->stream[1])));
   /* allocate pinned host memory for host/device transfers */
   cuCHK(cudaMallocHost(&(p_ctx->h_solve[0]), 32));
   cuCHK(cudaMallocHost(&(p_ctx->h_solve[1]), 32));
   cuCHK(cudaMallocHost(&(p_ctx->h_bt[0]), btsz));
   cuCHK(cudaMallocHost(&(p_ctx->h_bt[1]), btsz));
   /* allocate device memory for host/device transfers */
   cuCHK(cudaMalloc(&(p_ctx->d_solve[0]), 32));
   cuCHK(cudaMalloc(&(p_ctx->d_solve[1]), 32));
   cuCHK(cudaMalloc(&(p_ctx->d_state[0]), seedsz));
   cuCHK(cudaMalloc(&(p_ctx->d_state[1]), seedsz));
   cuCHK(cudaMalloc(&(p_ctx->d_bt[0]), btsz));
   cuCHK(cudaMalloc(&(p_ctx->d_bt[1]), btsz));
   /* allocate memory for Peach map on device */
   cuCHK(cudaMalloc(&(p_ctx->d_phash), 32));
   cuCHK(cudaMalloc(&(p_ctx->d_map), PEACHMAPLEN));
   /* clear device/host allocated memory */
   cuCHK(cudaMemsetAsync(p_ctx->d_bt[0], 0, btsz, cudaStreamDefault));
   cuCHK(cudaMemsetAsync(p_ctx->d_bt[1], 0, btsz, cudaStreamDefault));
   cuCHK(cudaMemsetAsync(p_ctx->d_state[0], 0, seedsz, cudaStreamDefault));
   cuCHK(cudaMemsetAsync(p_ctx->d_state[1], 0, seedsz, cudaStreamDefault));
   cuCHK(cudaMemsetAsync(p_ctx->d_solve[0], 0, 32, cudaStreamDefault));
   cuCHK(cudaMemsetAsync(p_ctx->d_solve[1], 0, 32, cudaStreamDefault));
   cuCHK(cudaMemsetAsync(p_ctx->d_phash, 0, 32, cudaStreamDefault));
   memset(p_ctx->h_bt[0], 0, btsz);
   memset(p_ctx->h_bt[1], 0, btsz);
   memset(p_ctx->h_solve[0], 0, 32);
   memset(p_ctx->h_solve[1], 0, 32);

   /* wait for all operations in cudaStreamDefault to complete */
   cuCHK(cudaStreamSynchronize(cudaStreamDefault));

   /* generate prng state */
   kcu_srand64<<<ctx->grid, ctx->block>>>(p_ctx->d_state[0], rand32());
   kcu_srand64<<<ctx->grid, ctx->block>>>(p_ctx->d_state[1], rand32());
   cudaDeviceSynchronize();

   /* set device as initialized */
   ctx->status = DEV_INIT;

   return VEOK;
}  /* end peach_init_cuda_device() */

/**
 * Try solve for a tokenized haiku as nonce output for Peach proof of work
 * on CUDA devices. Combine haiku protocols implemented in the Trigg
 * Algorithm with the memory intensive protocols of the Peach algorithm to
 * generate haiku output as proof of work.
 * @param ctx Pointer to DEVICE_CTX to perform work with
 * @param bt Pointer to block trailer to solve for
 * @param diff Difficulty to test against entropy of final hash
 * @param btout Pointer to location to place solved block trailer
 * @returns VEOK on solve, VERROR on no solve, or VETIMEOUT if GPU is
 * either stopped or unrecoverable.
*/
int peach_solve_cuda(DEVICE_CTX *ctx, BTRAILER *bt, word8 diff, BTRAILER *btout)
{
   int id, sid, grid, block, build;
   PEACH_CUDA_CTX *P;
   cudaError_t err;

#undef cuCHK
#define cuCHK(cuFN) \
   do { \
      err = (cuFN); \
      if (err != cudaSuccess) { \
         ctx->status = DEV_FAIL; \
         cu__log_error(err); \
         return VERROR; \
      } \
   } while(0)

   /* init */
   id = ctx->id;
   P = (PEACH_CUDA_CTX *) ctx->peach;
   /* report unuseable GPUs */
   if (ctx->status < DEV_NULL) return VETIMEOUT;

   /* set/check cuda device */
   cuCHK(cudaSetDevice(id));
   cuCHK(cudaGetLastError());

   /* build peach map */
   if (ctx->status == DEV_INIT) {
      /* build peach map -- init */
      if (ctx->work == 0) {
         /* ensure both streams have finished */
         if (cudaStreamQuery(P->stream[1]) == cudaSuccess
            && cudaStreamQuery(P->stream[0]) == cudaSuccess) {
            /* synchronize device before initializing new peach map */
            cudaDeviceSynchronize();
            /* clear any late solves */
            cuCHK(cudaMemset(P->d_solve[0], 0, 32));
            cuCHK(cudaMemset(P->d_solve[1], 0, 32));
            memset(P->h_solve[0], 0, 32);
            memset(P->h_solve[1], 0, 32);
            /* update block trailer */
            memcpy(P->h_bt[0], bt, sizeof(BTRAILER));
            memcpy(P->h_bt[1], bt, sizeof(BTRAILER));
            /* update device phash */
            cuCHK(cudaMemcpy(P->d_phash, P->h_bt[0]->phash, 32, cudaMemcpyHostToDevice));
            /* synchronize memory transfers before building peach map */
            cudaDeviceSynchronize();
            /* flag build ready */
            build = 1;
         }
      }
      /* build peach map -- build */
      if (ctx->work < PEACHCACHELEN) {
         for (sid = 0; sid < 2 && (build || ctx->work > 0); sid++) {
            /* ensure stream is ready for next section of build */
            if (cudaStreamQuery(P->stream[sid]) != cudaSuccess) continue;
            /* set CUDA configuration for generating peach map */
            if (cudaOccupancyMaxPotentialBlockSize(&grid, &block,
                  kcu_peach_build, 0, 0) != cudaSuccess) {
               pdebug("cudaOccupancy~BlockSize(%d) failed...", id);
               pdebug("Using conservative defaults, <<<128/128>>>");
               grid = 128;
               block = 128;
            }
            /* launch kernel to generate map */
            kcu_peach_build<<<grid, block, 0, P->stream[sid]>>>
               ((word32) ctx->work, P->d_map, P->d_phash);
            cuCHK(cudaGetLastError());
            /* update build progress */
            ctx->work += grid * block;
         }
      } else {
         /* ensure both streams have finished */
         if (cudaStreamQuery(P->stream[1]) == cudaSuccess
            && cudaStreamQuery(P->stream[0]) == cudaSuccess) {
            /* build is complete */
            ctx->last = time(NULL);
            ctx->status = DEV_IDLE;
            ctx->work = 0;
         }
      }
   }

   /* switch to WORK mode when all conditions are satisfied:
    * - transactions to solve
    * - block NOT already solved
    * - block NOT expired
    */
   switch (ctx->status) {
      case DEV_IDLE:{
         if (get32(bt->tcount) == 0) break;
         if (cmp64(bt->bnum, btout->bnum) == 0) break;
         if (difftime(time(NULL), get32(bt->time0)) >= BRIDGEv3) break;
         ctx->last = time(NULL);
         ctx->status = DEV_WORK;
         ctx->work = 0;
         break;
      }
   }

   /* solve work in block trailer */
   if (ctx->status == DEV_WORK) {
      for(sid = 0; sid < 2; sid++) {
         err = cudaStreamQuery(P->stream[sid]);
         if (err == cudaErrorNotReady) continue;
         if (err != cudaSuccess) {
            ctx->status = DEV_FAIL;
            cu__log_error(err);
            return VERROR;
         }
         /* check trailer for block update */
         if (memcmp(P->h_bt[sid]->phash, bt->phash, HASHLEN)) {
            ctx->status = DEV_INIT;
            ctx->work = 0;
            break;
         }
         /* switch to IDLE mode when reasonable:
          * - no transaction to solve
          * - block already solved
          * - block expired
          */
         if (get32(bt->tcount) == 0 || cmp64(bt->bnum, btout->bnum) == 0 ||
               difftime(time(NULL), get32(bt->time0)) >= BRIDGEv3) {
            ctx->status = DEV_IDLE;
            ctx->work = 0;
            break;
         }
         /* check for solves */
         if (*(P->h_solve[sid])) {
            /* combine solved nonce with bt */
            memcpy(P->h_bt[sid]->nonce, P->h_solve[sid], 32);
            /* clear solve from host/device */
            cudaMemsetAsync(P->d_solve[sid], 0, 32, P->stream[sid]);
            memset(P->h_solve[sid], 0, 32);
            /* move solved block trailer to btout */
            memcpy(btout, P->h_bt[sid], sizeof(BTRAILER));
            /* return a solve */
            return VEOK;
         }
         /* check for "on-the-fly" difficulty changes */
         diff = diff && diff < bt->difficulty[0] ? diff : bt->difficulty[0];
         /* ensure host block trailer is updated */
         memcpy(P->h_bt[sid], bt, 92);
         /* generate (first) nonce directly into block trailer */
         trigg_generate(P->h_bt[sid]->nonce);
         /* copy trailer updates w/ nonce ELSE just nonce */
         cuCHK(cudaMemcpyAsync(P->d_bt[sid], P->h_bt[sid],
            92 + 16, cudaMemcpyHostToDevice, P->stream[sid]));
         cuCHK(cudaGetLastError());
         /* launch kernel to solve Peach */
         kcu_peach_solve<<<ctx->grid, ctx->block, 0, P->stream[sid]>>>
            (P->d_map, P->d_bt[sid], P->d_state[sid], diff, P->d_solve[sid]);
         cuCHK(cudaGetLastError());
         /* retrieve solve seed */
         cudaMemcpyAsync(P->h_solve[sid], P->d_solve[sid], 32,
            cudaMemcpyDeviceToHost, P->stream[sid]);
         cuCHK(cudaGetLastError());
         /* increment progress counters */
         ctx->work += ctx->threads;
         double delta = difftime(time(NULL), ctx->last);
         ctx->hps = ctx->work / (delta ? delta : 1);
      }
   }

   return VERROR;
}  /* end peach_solve_cuda() */

/* end include guard */
#endif
