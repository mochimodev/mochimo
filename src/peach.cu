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
#include "extprint.h"
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

/**
 * @private
 * Peach specific cuda error checking definition.
 * Destroys/frees resources on error.
*/
#define cuCHK(_cmd, _dev, _exec) \
   do { \
      cudaError_t _cerr = _cmd; \
      if (_cerr != cudaSuccess) { \
         int _n; cudaGetDevice(&_n); cudaDeviceSynchronize(); \
         const char *_err = cudaGetErrorString(_cerr); \
         pfatal("CUDA#%d->%s: %s", _n, cuSTR(_cmd), _err); \
         peach_free_cuda_device(_dev, DEV_FAIL); \
         _exec; \
      } \
   } while(0)

/**
 * @private
 * Peach CUDA context. Managed internally by cross referencing parameters
 * of DEVICE_CTX passed to functions.
*/
typedef struct {
   nvmlDevice_t nvml_device;           /**< nvml device for monitoring */
   cudaStream_t stream[2];             /**< asynchronous streams */
   SHA256_CTX *h_ictx[2], *d_ictx[2];  /**< sha256 ictx lists */
   BTRAILER *h_bt[2];                  /**< BTRAILER (current) */
   word8 *h_solve[2], *d_solve[2];     /**< solve seeds */
   word8 *d_map;                       /**< Peach Map */
   int nvml_enabled;                   /**< Flags NVML capable */
} PEACH_CUDA_CTX;


/* pointer to peach CUDA context/s */
static PEACH_CUDA_CTX *PeachCudaCTX;
/* host phash and diff (paged memory txfer) */
static word8 *h_phash, *h_diff;
/* device symbol memory (unique per device) */
__device__ __constant__ static word8 c_phash[SHA256LEN];
__device__ __constant__ static word8 c_diff;

/**
 * @private
 * 256-bit Blake2b (w/ key) computation optimized for the Peach algorithm.
 * Places the resulting hash in @a out.
 * @param in Pointer to data to hash
 * @param inlen Length of @a in data, in bytes
 * @param keylen Length of optional @a key input, in bytes
 * @param out Pointer to location to place the message digest
*/
__device__ void cu_peach_blake2b(const void *in, size_t inlen, int keylen,
   const void *out)
{
   static word8 c_sigma[12][16] = {
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
   static word64 c_iv[8] = {
      0x6A09E667F3BCC908, 0xBB67AE8584CAA73B,
      0x3C6EF372FE94F82B, 0xA54FF53A5F1D36F1,
      0x510E527FADE682D1, 0x9B05688C2B3E6C1F,
      0x1F83D9ABFB41BD6B, 0x5BE0CD19137E2179
   };

   /* blake2b_init - outlen is always 256-bits in Peach */
   word64 v[16];
   word64 final[16];
   word64 state[8];
   word64 t = 128;
   word64 *in64 = (word64 *) in;
   int i;

   if (keylen == 64) {
      state[0] = WORD64_C(0x00b8aa23c261ef69);
      state[1] = WORD64_C(0xd38ae6abca237b9e);
      state[2] = WORD64_C(0x67fb881e5ee89069);
      state[3] = WORD64_C(0x3e5b8bd06b58d002);
      state[4] = WORD64_C(0x252d3f68395aae91);
      state[5] = WORD64_C(0xd25465e23c6c1b27);
      state[6] = WORD64_C(0x852b4cc2e13303b5);
      state[7] = WORD64_C(0x3f38b9ff245be7c1);
   } else {
      state[0] = WORD64_C(0x63320ace264383eb);
      state[1] = WORD64_C(0x012af5fd045a2737);
      state[2] = WORD64_C(0xf4f49c55e6be39df);
      state[3] = WORD64_C(0x791c5bc8affb11a7);
      state[4] = WORD64_C(0xc9bcacc002c0ea21);
      state[5] = WORD64_C(0x8295b8abe2fdedd6);
      state[6] = WORD64_C(0xb711490e5f9f41c8);
      state[7] = WORD64_C(0x3f8e4d1d9ebeaf1a);
   }

   /* blake2b_update */
   for(; inlen > 128; inlen -= 128, in64 = &in64[16]) {
      t += 128;

      v[0] = state[0];
      v[1] = state[1];
      v[2] = state[2];
      v[3] = state[3];
      v[4] = state[4];
      v[5] = state[5];
      v[6] = state[6];
      v[7] = state[7];
      v[8] = c_iv[0];
      v[9] = c_iv[1];
      v[10] = c_iv[2];
      v[11] = c_iv[3];
      v[12] = c_iv[4] ^ t;
      v[13] = c_iv[5];
      v[14] = c_iv[6];
      v[15] = c_iv[7];

      for (i = 0; i < BLAKE2BROUNDS; i++) {
         B2B_G( 0, 4,  8, 12, in64[c_sigma[i][ 0]], in64[c_sigma[i][ 1]]);
         B2B_G( 1, 5,  9, 13, in64[c_sigma[i][ 2]], in64[c_sigma[i][ 3]]);
         B2B_G( 2, 6, 10, 14, in64[c_sigma[i][ 4]], in64[c_sigma[i][ 5]]);
         B2B_G( 3, 7, 11, 15, in64[c_sigma[i][ 6]], in64[c_sigma[i][ 7]]);
         B2B_G( 0, 5, 10, 15, in64[c_sigma[i][ 8]], in64[c_sigma[i][ 9]]);
         B2B_G( 1, 6, 11, 12, in64[c_sigma[i][10]], in64[c_sigma[i][11]]);
         B2B_G( 2, 7,  8, 13, in64[c_sigma[i][12]], in64[c_sigma[i][13]]);
         B2B_G( 3, 4,  9, 14, in64[c_sigma[i][14]], in64[c_sigma[i][15]]);
      }

      state[0] ^= v[0] ^ v[8];
      state[1] ^= v[1] ^ v[9];
      state[2] ^= v[2] ^ v[10];
      state[3] ^= v[3] ^ v[11];
      state[4] ^= v[4] ^ v[12];
      state[5] ^= v[5] ^ v[13];
      state[6] ^= v[6] ^ v[14];
      state[7] ^= v[7] ^ v[15];
   }

   /* blake2b_final - somewhat conveniently (and exclusive to Peach)...
    * the remaining datalen will always be 36... */
   final[0] = in64[0];
   final[1] = in64[1];
   final[2] = in64[2];
   final[3] = in64[3];
   final[4] = (word64) ((word32 *) in64)[8];
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

   t += 36;

   v[0] = state[0];
   v[1] = state[1];
   v[2] = state[2];
   v[3] = state[3];
   v[4] = state[4];
   v[5] = state[5];
   v[6] = state[6];
   v[7] = state[7];
   v[8] = c_iv[0];
   v[9] = c_iv[1];
   v[10] = c_iv[2];
   v[11] = c_iv[3];
   v[12] = c_iv[4] ^ t;
   v[13] = c_iv[5];
   v[14] = ~c_iv[6];
   v[15] = c_iv[7];

   for (i = 0; i < BLAKE2BROUNDS; i++) {
      B2B_G( 0, 4,  8, 12, final[c_sigma[i][ 0]], final[c_sigma[i][ 1]]);
      B2B_G( 1, 5,  9, 13, final[c_sigma[i][ 2]], final[c_sigma[i][ 3]]);
      B2B_G( 2, 6, 10, 14, final[c_sigma[i][ 4]], final[c_sigma[i][ 5]]);
      B2B_G( 3, 7, 11, 15, final[c_sigma[i][ 6]], final[c_sigma[i][ 7]]);
      B2B_G( 0, 5, 10, 15, final[c_sigma[i][ 8]], final[c_sigma[i][ 9]]);
      B2B_G( 1, 6, 11, 12, final[c_sigma[i][10]], final[c_sigma[i][11]]);
      B2B_G( 2, 7,  8, 13, final[c_sigma[i][12]], final[c_sigma[i][13]]);
      B2B_G( 3, 4,  9, 14, final[c_sigma[i][14]], final[c_sigma[i][15]]);
   }

   /* blake2b_output */
   ((word64 *) out)[0] = state[0] ^ v[0] ^ v[8];
   ((word64 *) out)[1] = state[1] ^ v[1] ^ v[9];
   ((word64 *) out)[2] = state[2] ^ v[2] ^ v[10];
   ((word64 *) out)[3] = state[3] ^ v[3] ^ v[11];
}  /* end cu_peach_blake2b() */

/**
 * @private
 * 128-bit MD2 computation optimized for the Peach algorithm.
 * Places the resulting hash in @a out.
 * @param in Pointer to data to hash
 * @param inlen Length of @a in data, in bytes
 * @param out Pointer to location to place the message digest
*/
__device__ void cu_peach_md2(const void *in, size_t inlen,
   void *out)
{
   static word8 s[256] = {
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
   word8 state[48] = { 0 };
   word8 checksum[16] = { 0 };
   word64 *checksum64 = (word64 *) checksum;
   word64 *state64 = (word64 *) state;
   word64 *in64 = (word64 *) in;
   word8 *in8 = (word8 *) in;
   size_t j, k;

   word8 pad = 16 - (inlen & 0xf);

   /* md2_update */
   for (; inlen >= 16; inlen -= 16, in8 = &in8[16], in64 = &in64[2]) {
      state64[2] = in64[0];
      state64[3] = in64[1];
      state64[4] = state64[2] ^ state64[0];
      state64[5] = state64[3] ^ state64[1];

	   state[0] ^= s[0];
      for (k = 1; k < 48; ++k) {
	   	state[k] ^= s[state[k - 1]];
	   }
	   for (j = 1; j < 18; ++j) {
         state[0] ^= s[(state[47] + (j - 1)) & 0xFF];
         for (k = 1; k < 48; ++k) {
            state[k] ^= s[state[k - 1]];
         }
	   }
      checksum[0] ^= s[in8[0] ^ checksum[15]];
      for (j = 1; j < 16; ++j) {
         checksum[j] ^= s[in8[j] ^ checksum[j - 1]];
      }
   }

   /* md2_final - only 4 bytes left, so 12 remaining bytes are pad *//*
   state64[2] = *((word32 *) in64) | (pad64 & WORD64_C(0xFFFFFFFF00000000));
   state64[3] = pad64; */
   for(j = 0; j < inlen; j++) state[j + 16] = in8[j];
   for(; j < 16; j++) state[j + 16] = pad;
   state64[4] = state64[2] ^ state64[0];
   state64[5] = state64[3] ^ state64[1];

   state[0] ^= s[0];
   for (k = 1; k < 48; ++k) {
		state[k] ^= s[state[k - 1]];
	}
	for (j = 1; j < 18; ++j) {
      state[0] ^= s[(state[47] + (j - 1)) & 0xFF];
      for (k = 1; k < 48; ++k) {
         state[k] ^= s[state[k - 1]];
      }
	}
   checksum[0] ^= s[in8[0] ^ checksum[15]];
   checksum[1] ^= s[in8[1] ^ checksum[0]];
   checksum[2] ^= s[in8[2] ^ checksum[1]];
   checksum[3] ^= s[in8[3] ^ checksum[2]];
   for (j = 4; j < 16; ++j) {
      checksum[j] ^= s[pad ^ checksum[j - 1]];
   }

   state64[2] = checksum64[0];
   state64[3] = checksum64[1];
   state64[4] = state64[2] ^ state64[0];
   state64[5] = state64[3] ^ state64[1];

   state[0] ^= s[0];
   for (k = 1; k < 48; ++k) {
		state[k] ^= s[state[k - 1]];
	}
	for (j = 1; j < 18; ++j) {
      state[0] ^= s[(state[47] + (j - 1)) & 0xFF];
      for (k = 1; k < 48; ++k) {
         state[k] ^= s[state[k - 1]];
      }
	}

   /* md2_output */
   ((word64 *) out)[0] = state64[0];
   ((word64 *) out)[1] = state64[1];
   /* MD2 hash = 128 bits, zero fill remaining... */
   ((word64 *) out)[2] = 0;
   ((word64 *) out)[3] = 0;
}  /* end cu_peach_md2 */

/**
 * @private
 * 128-bit MD5 computation optimized for the Peach algorithm.
 * Places the resulting hash in @a out.
 * @param in Pointer to data to hash
 * @param inlen Length of @a in data, in bytes
 * @param out Pointer to location to place the message digest
*/
__device__ void cu_peach_md5(const void *in, size_t inlen, void *out)
{
   /* md5_init -- NOT STATIC */
   word32 state[4] = {
      0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476
   };
   word32 a, b, c, d;
   word32 bitlen = inlen << 3;
   word32 *in32 = (word32 *) in;

   /* md5_update */
   for (; inlen >= 64; inlen -= 64, in32 = &in32[16]) {
      a = state[0];
      b = state[1];
      c = state[2];
      d = state[3];

      FF(a, b, c, d, in32[0],   7, 0xd76aa478);
      FF(d, a, b, c, in32[1],  12, 0xe8c7b756);
      FF(c, d, a, b, in32[2],  17, 0x242070db);
      FF(b, c, d, a, in32[3],  22, 0xc1bdceee);
      FF(a, b, c, d, in32[4],   7, 0xf57c0faf);
      FF(d, a, b, c, in32[5],  12, 0x4787c62a);
      FF(c, d, a, b, in32[6],  17, 0xa8304613);
      FF(b, c, d, a, in32[7],  22, 0xfd469501);
      FF(a, b, c, d, in32[8],   7, 0x698098d8);
      FF(d, a, b, c, in32[9],  12, 0x8b44f7af);
      FF(c, d, a, b, in32[10], 17, 0xffff5bb1);
      FF(b, c, d, a, in32[11], 22, 0x895cd7be);
      FF(a, b, c, d, in32[12],  7, 0x6b901122);
      FF(d, a, b, c, in32[13], 12, 0xfd987193);
      FF(c, d, a, b, in32[14], 17, 0xa679438e);
      FF(b, c, d, a, in32[15], 22, 0x49b40821);

      GG(a, b, c, d, in32[1],   5, 0xf61e2562);
      GG(d, a, b, c, in32[6],   9, 0xc040b340);
      GG(c, d, a, b, in32[11], 14, 0x265e5a51);
      GG(b, c, d, a, in32[0],  20, 0xe9b6c7aa);
      GG(a, b, c, d, in32[5],   5, 0xd62f105d);
      GG(d, a, b, c, in32[10],  9, 0x02441453);
      GG(c, d, a, b, in32[15], 14, 0xd8a1e681);
      GG(b, c, d, a, in32[4],  20, 0xe7d3fbc8);
      GG(a, b, c, d, in32[9],   5, 0x21e1cde6);
      GG(d, a, b, c, in32[14],  9, 0xc33707d6);
      GG(c, d, a, b, in32[3],  14, 0xf4d50d87);
      GG(b, c, d, a, in32[8],  20, 0x455a14ed);
      GG(a, b, c, d, in32[13],  5, 0xa9e3e905);
      GG(d, a, b, c, in32[2],   9, 0xfcefa3f8);
      GG(c, d, a, b, in32[7],  14, 0x676f02d9);
      GG(b, c, d, a, in32[12], 20, 0x8d2a4c8a);

      HH(a, b, c, d, in32[5],   4, 0xfffa3942);
      HH(d, a, b, c, in32[8],  11, 0x8771f681);
      HH(c, d, a, b, in32[11], 16, 0x6d9d6122);
      HH(b, c, d, a, in32[14], 23, 0xfde5380c);
      HH(a, b, c, d, in32[1],   4, 0xa4beea44);
      HH(d, a, b, c, in32[4],  11, 0x4bdecfa9);
      HH(c, d, a, b, in32[7],  16, 0xf6bb4b60);
      HH(b, c, d, a, in32[10], 23, 0xbebfbc70);
      HH(a, b, c, d, in32[13],  4, 0x289b7ec6);
      HH(d, a, b, c, in32[0],  11, 0xeaa127fa);
      HH(c, d, a, b, in32[3],  16, 0xd4ef3085);
      HH(b, c, d, a, in32[6],  23, 0x04881d05);
      HH(a, b, c, d, in32[9],   4, 0xd9d4d039);
      HH(d, a, b, c, in32[12], 11, 0xe6db99e5);
      HH(c, d, a, b, in32[15], 16, 0x1fa27cf8);
      HH(b, c, d, a, in32[2],  23, 0xc4ac5665);

      II(a, b, c, d, in32[0],   6, 0xf4292244);
      II(d, a, b, c, in32[7],  10, 0x432aff97);
      II(c, d, a, b, in32[14], 15, 0xab9423a7);
      II(b, c, d, a, in32[5],  21, 0xfc93a039);
      II(a, b, c, d, in32[12],  6, 0x655b59c3);
      II(d, a, b, c, in32[3],  10, 0x8f0ccc92);
      II(c, d, a, b, in32[10], 15, 0xffeff47d);
      II(b, c, d, a, in32[1],  21, 0x85845dd1);
      II(a, b, c, d, in32[8],   6, 0x6fa87e4f);
      II(d, a, b, c, in32[15], 10, 0xfe2ce6e0);
      II(c, d, a, b, in32[6],  15, 0xa3014314);
      II(b, c, d, a, in32[13], 21, 0x4e0811a1);
      II(a, b, c, d, in32[4],   6, 0xf7537e82);
      II(d, a, b, c, in32[11], 10, 0xbd3af235);
      II(c, d, a, b, in32[2],  15, 0x2ad7d2bb);
      II(b, c, d, a, in32[9],  21, 0xeb86d391);

      state[0] += a;
      state[1] += b;
      state[2] += c;
      state[3] += d;
   }

   /* md5_final - somewhat conveniently (and exclusive to Peach)...
    * the remaining datalen will always be 36, so:
    * in32[9] = 0x80; and in32[10+] = 0; */
   a = state[0];
   b = state[1];
   c = state[2];
   d = state[3];

   FF(a, b, c, d, in32[0],  7, 0xd76aa478);
   FF(d, a, b, c, in32[1], 12, 0xe8c7b756);
   FF(c, d, a, b, in32[2], 17, 0x242070db);
   FF(b, c, d, a, in32[3], 22, 0xc1bdceee);
   FF(a, b, c, d, in32[4],  7, 0xf57c0faf);
   FF(d, a, b, c, in32[5], 12, 0x4787c62a);
   FF(c, d, a, b, in32[6], 17, 0xa8304613);
   FF(b, c, d, a, in32[7], 22, 0xfd469501);
   FF(a, b, c, d, in32[8],  7, 0x698098d8);
   FF(d, a, b, c,    0x80, 12, 0x8b44f7af);
   FF(c, d, a, b,    0x00, 17, 0xffff5bb1);
   FF(b, c, d, a,    0x00, 22, 0x895cd7be);
   FF(a, b, c, d,    0x00,  7, 0x6b901122);
   FF(d, a, b, c,    0x00, 12, 0xfd987193);
   FF(c, d, a, b,  bitlen, 17, 0xa679438e);
   FF(b, c, d, a,    0x00, 22, 0x49b40821);

   GG(a, b, c, d, in32[1],  5, 0xf61e2562);
   GG(d, a, b, c, in32[6],  9, 0xc040b340);
   GG(c, d, a, b,    0x00, 14, 0x265e5a51);
   GG(b, c, d, a, in32[0], 20, 0xe9b6c7aa);
   GG(a, b, c, d, in32[5],  5, 0xd62f105d);
   GG(d, a, b, c,    0x00,  9, 0x02441453);
   GG(c, d, a, b,    0x00, 14, 0xd8a1e681);
   GG(b, c, d, a, in32[4], 20, 0xe7d3fbc8);
   GG(a, b, c, d,    0x80,  5, 0x21e1cde6);
   GG(d, a, b, c,  bitlen,  9, 0xc33707d6);
   GG(c, d, a, b, in32[3], 14, 0xf4d50d87);
   GG(b, c, d, a, in32[8], 20, 0x455a14ed);
   GG(a, b, c, d,    0x00,  5, 0xa9e3e905);
   GG(d, a, b, c, in32[2],  9, 0xfcefa3f8);
   GG(c, d, a, b, in32[7], 14, 0x676f02d9);
   GG(b, c, d, a,    0x00, 20, 0x8d2a4c8a);

   HH(a, b, c, d, in32[5],  4, 0xfffa3942);
   HH(d, a, b, c, in32[8], 11, 0x8771f681);
   HH(c, d, a, b,    0x00, 16, 0x6d9d6122);
   HH(b, c, d, a,  bitlen, 23, 0xfde5380c);
   HH(a, b, c, d, in32[1], 4, 0xa4beea44);
   HH(d, a, b, c, in32[4], 11, 0x4bdecfa9);
   HH(c, d, a, b, in32[7], 16, 0xf6bb4b60);
   HH(b, c, d, a,    0x00, 23, 0xbebfbc70);
   HH(a, b, c, d,    0x00,  4, 0x289b7ec6);
   HH(d, a, b, c, in32[0], 11, 0xeaa127fa);
   HH(c, d, a, b, in32[3], 16, 0xd4ef3085);
   HH(b, c, d, a, in32[6], 23, 0x04881d05);
   HH(a, b, c, d,    0x80,  4, 0xd9d4d039);
   HH(d, a, b, c,    0x00, 11, 0xe6db99e5);
   HH(c, d, a, b,    0x00, 16, 0x1fa27cf8);
   HH(b, c, d, a, in32[2], 23, 0xc4ac5665);

   II(a, b, c, d, in32[0],  6, 0xf4292244);
   II(d, a, b, c, in32[7], 10, 0x432aff97);
   II(c, d, a, b,  bitlen, 15, 0xab9423a7);
   II(b, c, d, a, in32[5], 21, 0xfc93a039);
   II(a, b, c, d,    0x00,  6, 0x655b59c3);
   II(d, a, b, c, in32[3], 10, 0x8f0ccc92);
   II(c, d, a, b,    0x00, 15, 0xffeff47d);
   II(b, c, d, a, in32[1], 21, 0x85845dd1);
   II(a, b, c, d, in32[8], 6, 0x6fa87e4f);
   II(d, a, b, c,    0x00, 10, 0xfe2ce6e0);
   II(c, d, a, b, in32[6], 15, 0xa3014314);
   II(b, c, d, a,    0x00, 21, 0x4e0811a1);
   II(a, b, c, d, in32[4], 6, 0xf7537e82);
   II(d, a, b, c,    0x00, 10, 0xbd3af235);
   II(c, d, a, b, in32[2], 15, 0x2ad7d2bb);
   II(b, c, d, a,    0x80, 21, 0xeb86d391);

   state[0] += a;
   state[1] += b;
   state[2] += c;
   state[3] += d;

   /* md5_output */
   ((word64 *) out)[0] = ((word64 *) state)[0];
   ((word64 *) out)[1] = ((word64 *) state)[1];
   /* MD5 hash = 128 bits, zero fill remaining... */
   ((word64 *) out)[2] = 0;
   ((word64 *) out)[3] = 0;
}  /* end cuda_peach_md5 */

/**
 * @private
 * 160-bit Sha1 computation optimized for the Peach algorithm.
 * Places the resulting hash in @a out.
 * @param in Pointer to data to hash
 * @param inlen Length of @a in data, in bytes
 * @param out Pointer to location to place the message digest
*/
__device__ void cu_peach_sha1(const void *in, size_t inlen, void *out)
{
   static word32 k[4] = {
      0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xca62c1d6
   };

   /* Since this implementation uses little endian byte ordering and
    * SHA uses big endian, reverse all the bytes upon input, and
    * re-reverse them on output */

   /* sha1_init */
   word32 state[5]  = {
      0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0
   };
   word32 bitlen = inlen << 3;
   word32 W[16], a, b, c, d, e;
   word32 *in32 = (word32 *) in;

   /* sha1_update */
   for(; inlen >= 64; inlen -= 64, in32 = &in32[16]) {
      W[0] = in32[0];
      W[1] = in32[1];
      W[2] = in32[2];
      W[3] = in32[3];
      W[4] = in32[4];
      W[5] = in32[5];
      W[6] = in32[6];
      W[7] = in32[7];
      W[8] = in32[8];
      W[9] = in32[9];
      W[10] = in32[10];
      W[11] = in32[11];
      W[12] = in32[12];
      W[13] = in32[13];
      W[14] = in32[14];
      W[15] = in32[15];

      a = state[0];
      b = state[1];
      c = state[2];
      d = state[3];
      e = state[4];

      /* SHA1 round 1 */
      sha1_r0(a, b, c, d, e, 0);
      sha1_r0(e, a, b, c, d, 1);
      sha1_r0(d, e, a, b, c, 2);
      sha1_r0(c, d, e, a, b, 3);
      sha1_r0(b, c, d, e, a, 4);
      sha1_r0(a, b, c, d, e, 5);
      sha1_r0(e, a, b, c, d, 6);
      sha1_r0(d, e, a, b, c, 7);
      sha1_r0(c, d, e, a, b, 8);
      sha1_r0(b, c, d, e, a, 9);
      sha1_r0(a, b, c, d, e, 10);
      sha1_r0(e, a, b, c, d, 11);
      sha1_r0(d, e, a, b, c, 12);
      sha1_r0(c, d, e, a, b, 13);
      sha1_r0(b, c, d, e, a, 14);
      sha1_r0(a, b, c, d, e, 15);
      /* alternate round computation */
      sha1_r1(e, a, b, c, d, 16);
      sha1_r1(d, e, a, b, c, 17);
      sha1_r1(c, d, e, a, b, 18);
      sha1_r1(b, c, d, e, a, 19);
      sha1_r2(a, b, c, d, e, 20);

      /* SHA1 round 2 */
      sha1_r2(e, a, b, c, d, 21);
      sha1_r2(d, e, a, b, c, 22);
      sha1_r2(c, d, e, a, b, 23);
      sha1_r2(b, c, d, e, a, 24);
      sha1_r2(a, b, c, d, e, 25);
      sha1_r2(e, a, b, c, d, 26);
      sha1_r2(d, e, a, b, c, 27);
      sha1_r2(c, d, e, a, b, 28);
      sha1_r2(b, c, d, e, a, 29);
      sha1_r2(a, b, c, d, e, 30);
      sha1_r2(e, a, b, c, d, 31);
      sha1_r2(d, e, a, b, c, 32);
      sha1_r2(c, d, e, a, b, 33);
      sha1_r2(b, c, d, e, a, 34);
      sha1_r2(a, b, c, d, e, 35);
      sha1_r2(e, a, b, c, d, 36);
      sha1_r2(d, e, a, b, c, 37);
      sha1_r2(c, d, e, a, b, 38);
      sha1_r2(b, c, d, e, a, 39);

      /* SHA1 round 3 */
      sha1_r3(a, b, c, d, e, 40);
      sha1_r3(e, a, b, c, d, 41);
      sha1_r3(d, e, a, b, c, 42);
      sha1_r3(c, d, e, a, b, 43);
      sha1_r3(b, c, d, e, a, 44);
      sha1_r3(a, b, c, d, e, 45);
      sha1_r3(e, a, b, c, d, 46);
      sha1_r3(d, e, a, b, c, 47);
      sha1_r3(c, d, e, a, b, 48);
      sha1_r3(b, c, d, e, a, 49);
      sha1_r3(a, b, c, d, e, 50);
      sha1_r3(e, a, b, c, d, 51);
      sha1_r3(d, e, a, b, c, 52);
      sha1_r3(c, d, e, a, b, 53);
      sha1_r3(b, c, d, e, a, 54);
      sha1_r3(a, b, c, d, e, 55);
      sha1_r3(e, a, b, c, d, 56);
      sha1_r3(d, e, a, b, c, 57);
      sha1_r3(c, d, e, a, b, 58);
      sha1_r3(b, c, d, e, a, 59);

      /* SHA1 round 4 */
      sha1_r4(a, b, c, d, e, 60);
      sha1_r4(e, a, b, c, d, 61);
      sha1_r4(d, e, a, b, c, 62);
      sha1_r4(c, d, e, a, b, 63);
      sha1_r4(b, c, d, e, a, 64);
      sha1_r4(a, b, c, d, e, 65);
      sha1_r4(e, a, b, c, d, 66);
      sha1_r4(d, e, a, b, c, 67);
      sha1_r4(c, d, e, a, b, 68);
      sha1_r4(b, c, d, e, a, 69);
      sha1_r4(a, b, c, d, e, 70);
      sha1_r4(e, a, b, c, d, 71);
      sha1_r4(d, e, a, b, c, 72);
      sha1_r4(c, d, e, a, b, 73);
      sha1_r4(b, c, d, e, a, 74);
      sha1_r4(a, b, c, d, e, 75);
      sha1_r4(e, a, b, c, d, 76);
      sha1_r4(d, e, a, b, c, 77);
      sha1_r4(c, d, e, a, b, 78);
      sha1_r4(b, c, d, e, a, 79);

      state[0] += a;
      state[1] += b;
      state[2] += c;
      state[3] += d;
      state[4] += e;
   }

   /* sha1_final - somewhat conveniently (and exclusive to Peach)...
    * the remaining datalen will always be 36, so in32[9] = 0x80. */
   W[0] = in32[0];
   W[1] = in32[1];
   W[2] = in32[2];
   W[3] = in32[3];
   W[4] = in32[4];
   W[5] = in32[5];
   W[6] = in32[6];
   W[7] = in32[7];
   W[8] = in32[8];
   W[9] = 0x80;
   W[10] = 0;
   W[11] = 0;
   W[12] = 0;
   W[13] = 0;
   W[14] = 0;
   W[15] = bswap32(bitlen);

   a = state[0];
   b = state[1];
   c = state[2];
   d = state[3];
   e = state[4];

   /* SHA1 round 1 */
   sha1_r0(a, b, c, d, e, 0);
   sha1_r0(e, a, b, c, d, 1);
   sha1_r0(d, e, a, b, c, 2);
   sha1_r0(c, d, e, a, b, 3);
   sha1_r0(b, c, d, e, a, 4);
   sha1_r0(a, b, c, d, e, 5);
   sha1_r0(e, a, b, c, d, 6);
   sha1_r0(d, e, a, b, c, 7);
   sha1_r0(c, d, e, a, b, 8);
   sha1_r0(b, c, d, e, a, 9);
   sha1_r0(a, b, c, d, e, 10);
   sha1_r0(e, a, b, c, d, 11);
   sha1_r0(d, e, a, b, c, 12);
   sha1_r0(c, d, e, a, b, 13);
   sha1_r0(b, c, d, e, a, 14);
   sha1_r0(a, b, c, d, e, 15);
   /* alternate round computation */
   sha1_r1(e, a, b, c, d, 16);
   sha1_r1(d, e, a, b, c, 17);
   sha1_r1(c, d, e, a, b, 18);
   sha1_r1(b, c, d, e, a, 19);
   sha1_r2(a, b, c, d, e, 20);

   /* SHA1 round 2 */
   sha1_r2(e, a, b, c, d, 21);
   sha1_r2(d, e, a, b, c, 22);
   sha1_r2(c, d, e, a, b, 23);
   sha1_r2(b, c, d, e, a, 24);
   sha1_r2(a, b, c, d, e, 25);
   sha1_r2(e, a, b, c, d, 26);
   sha1_r2(d, e, a, b, c, 27);
   sha1_r2(c, d, e, a, b, 28);
   sha1_r2(b, c, d, e, a, 29);
   sha1_r2(a, b, c, d, e, 30);
   sha1_r2(e, a, b, c, d, 31);
   sha1_r2(d, e, a, b, c, 32);
   sha1_r2(c, d, e, a, b, 33);
   sha1_r2(b, c, d, e, a, 34);
   sha1_r2(a, b, c, d, e, 35);
   sha1_r2(e, a, b, c, d, 36);
   sha1_r2(d, e, a, b, c, 37);
   sha1_r2(c, d, e, a, b, 38);
   sha1_r2(b, c, d, e, a, 39);

   /* SHA1 round 3 */
   sha1_r3(a, b, c, d, e, 40);
   sha1_r3(e, a, b, c, d, 41);
   sha1_r3(d, e, a, b, c, 42);
   sha1_r3(c, d, e, a, b, 43);
   sha1_r3(b, c, d, e, a, 44);
   sha1_r3(a, b, c, d, e, 45);
   sha1_r3(e, a, b, c, d, 46);
   sha1_r3(d, e, a, b, c, 47);
   sha1_r3(c, d, e, a, b, 48);
   sha1_r3(b, c, d, e, a, 49);
   sha1_r3(a, b, c, d, e, 50);
   sha1_r3(e, a, b, c, d, 51);
   sha1_r3(d, e, a, b, c, 52);
   sha1_r3(c, d, e, a, b, 53);
   sha1_r3(b, c, d, e, a, 54);
   sha1_r3(a, b, c, d, e, 55);
   sha1_r3(e, a, b, c, d, 56);
   sha1_r3(d, e, a, b, c, 57);
   sha1_r3(c, d, e, a, b, 58);
   sha1_r3(b, c, d, e, a, 59);

   /* SHA1 round 4 */
   sha1_r4(a, b, c, d, e, 60);
   sha1_r4(e, a, b, c, d, 61);
   sha1_r4(d, e, a, b, c, 62);
   sha1_r4(c, d, e, a, b, 63);
   sha1_r4(b, c, d, e, a, 64);
   sha1_r4(a, b, c, d, e, 65);
   sha1_r4(e, a, b, c, d, 66);
   sha1_r4(d, e, a, b, c, 67);
   sha1_r4(c, d, e, a, b, 68);
   sha1_r4(b, c, d, e, a, 69);
   sha1_r4(a, b, c, d, e, 70);
   sha1_r4(e, a, b, c, d, 71);
   sha1_r4(d, e, a, b, c, 72);
   sha1_r4(c, d, e, a, b, 73);
   sha1_r4(b, c, d, e, a, 74);
   sha1_r4(a, b, c, d, e, 75);
   sha1_r4(e, a, b, c, d, 76);
   sha1_r4(d, e, a, b, c, 77);
   sha1_r4(c, d, e, a, b, 78);
   sha1_r4(b, c, d, e, a, 79);

   state[0] += a;
   state[1] += b;
   state[2] += c;
   state[3] += d;
   state[4] += e;

   /* sha1_output */
   ((word32 *) out)[0] = bswap32(state[0]);
   ((word32 *) out)[1] = bswap32(state[1]);
   ((word32 *) out)[2] = bswap32(state[2]);
   ((word32 *) out)[3] = bswap32(state[3]);
   ((word32 *) out)[4] = bswap32(state[4]);
   /* sha1 hash = 160 bits, zero fill remaining... */
   ((word32 *) out)[5] = 0;
   ((word64 *) out)[3] = 0;
}  /* end cu_peach_sha1() */

/**
 * @private
 * 256-bit SHA256 computation optimized for the Peach algorithm.
 * Places the resulting hash in @a out.
 * @param in Pointer to data to hash
 * @param inlen Length of @a in data, in bytes
 * @param out Pointer to location to place the message digest
*/
__device__ void cu_peach_sha256(const void *in, size_t inlen, void *out)
{
   static word32 k[64] = {
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

   /* Since this implementation uses little endian byte ordering and
    * SHA uses big endian, reverse all the bytes upon input, and
    * re-reverse them on output */

   /* sha256_init */
   word32 state[8] = {
      0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
   };
   uint32_t W[16], a, b, c, d, e, f, g, h;
   word32 bitlen = inlen << 3;
   word32 *in32 = (word32 *) in;

   /* sha256_update */
   for(; inlen >= 64; inlen -= 64, in32 = &in32[16]) {
      W[0] = in32[0];
      W[1] = in32[1];
      W[2] = in32[2];
      W[3] = in32[3];
      W[4] = in32[4];
      W[5] = in32[5];
      W[6] = in32[6];
      W[7] = in32[7];
      W[8] = in32[8];
      W[9] = in32[9];
      W[10] = in32[10];
      W[11] = in32[11];
      W[12] = in32[12];
      W[13] = in32[13];
      W[14] = in32[14];
      W[15] = in32[15];

      a = state[0];
      b = state[1];
      c = state[2];
      d = state[3];
      e = state[4];
      f = state[5];
      g = state[6];
      h = state[7];

      /* initial 16 rounds */
      RX0_8(0); RX0_8(8);
      /* rounds 16 - 32 */
      RX_8(0, 16); RX_8(8, 16);
      /* rounds 32 - 48 */
      RX_8(0, 32); RX_8(8, 32);
      /* rounds 48 - 64 */
      RX_8(0, 48); RX_8(8, 48);

      state[0] += a;
      state[1] += b;
      state[2] += c;
      state[3] += d;
      state[4] += e;
      state[5] += f;
      state[6] += g;
      state[7] += h;
   }

   /* sha256_final - somewhat conveniently (and exclusive to Peach)...
    * the remaining datalen will always be 36, so in32[9] = 0x80. */
   W[0] = in32[0];
   W[1] = in32[1];
   W[2] = in32[2];
   W[3] = in32[3];
   W[4] = in32[4];
   W[5] = in32[5];
   W[6] = in32[6];
   W[7] = in32[7];
   W[8] = in32[8];
   W[9] = 0x80;
   W[10] = 0;
   W[11] = 0;
   W[12] = 0;
   W[13] = 0;
   W[14] = 0;
   W[15] = bswap32(bitlen);

   a = state[0];
   b = state[1];
   c = state[2];
   d = state[3];
   e = state[4];
   f = state[5];
   g = state[6];
   h = state[7];

   /* initial 16 rounds */
   RX0_8(0); RX0_8(8);
   /* rounds 16 - 32 */
   RX_8(0, 16); RX_8(8, 16);
   /* rounds 32 - 48 */
   RX_8(0, 32); RX_8(8, 32);
   /* rounds 48 - 64 */
   RX_8(0, 48); RX_8(8, 48);

   state[0] += a;
   state[1] += b;
   state[2] += c;
   state[3] += d;
   state[4] += e;
   state[5] += f;
   state[6] += g;
   state[7] += h;

   /* Since this implementation uses little endian byte ordering and
    * SHA uses big endian, reverse all the bytes when copying the
    * final state to the output hash. */
   ((uint32_t *) out)[0] = bswap32(state[0]);
   ((uint32_t *) out)[1] = bswap32(state[1]);
   ((uint32_t *) out)[2] = bswap32(state[2]);
   ((uint32_t *) out)[3] = bswap32(state[3]);
   ((uint32_t *) out)[4] = bswap32(state[4]);
   ((uint32_t *) out)[5] = bswap32(state[5]);
   ((uint32_t *) out)[6] = bswap32(state[6]);
   ((uint32_t *) out)[7] = bswap32(state[7]);
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
__device__ void cu_peach_sha3(const void *in, size_t inlen,
   int keccak_final, void *out)
{
   static word64 keccakf_rndc[24] = {
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
	uint64_t Ba, Be, Bi, Bo, Bu;
	uint64_t Ca, Ce, Ci, Co, Cu;
	uint64_t Da, De, Di, Do, Du;
   word64 *st64 = (word64 *) state;
   word64 *in64 = (word64 *) in;
	int r;

   /* sha3_update - 136 is ctx->rsiz, fill only 17x 64-bit words in state */
   for(; inlen >= 136; inlen -= 136, in64 = &in64[17]) {
      for (r = 0; r < 17; r++) st64[r] ^= in64[r];
      for (r = 0; r < KECCAKFROUNDS; r += 4) {
         /* Unrolled 4 rounds at a time */

         Ca = st64[0] ^ st64[5] ^ st64[10] ^ st64[15] ^ st64[20];
         Ce = st64[1] ^ st64[6] ^ st64[11] ^ st64[16] ^ st64[21];
         Ci = st64[2] ^ st64[7] ^ st64[12] ^ st64[17] ^ st64[22];
         Co = st64[3] ^ st64[8] ^ st64[13] ^ st64[18] ^ st64[23];
         Cu = st64[4] ^ st64[9] ^ st64[14] ^ st64[19] ^ st64[24];
         Da = Cu ^ rol64(Ce, 1);
         De = Ca ^ rol64(Ci, 1);
         Di = Ce ^ rol64(Co, 1);
         Do = Ci ^ rol64(Cu, 1);
         Du = Co ^ rol64(Ca, 1);

         Ba = (st64[0] ^ Da);
         Be = rol64((st64[6] ^ De), 44);
         Bi = rol64((st64[12] ^ Di), 43);
         Bo = rol64((st64[18] ^ Do), 21);
         Bu = rol64((st64[24] ^ Du), 14);
         st64[0]  = Ba ^ ((~Be) & Bi) ^ keccakf_rndc[r];
         st64[6]  = Be ^ ((~Bi) & Bo);
         st64[12] = Bi ^ ((~Bo) & Bu);
         st64[18] = Bo ^ ((~Bu) & Ba);
         st64[24] = Bu ^ ((~Ba) & Be);

         Bi = rol64((st64[10] ^ Da), 3);
         Bo = rol64((st64[16] ^ De), 45);
         Bu = rol64((st64[22] ^ Di), 61);
         Ba = rol64((st64[3] ^ Do), 28);
         Be = rol64((st64[9] ^ Du), 20);
         st64[10] = Ba ^ ((~Be) & Bi);
         st64[16] = Be ^ ((~Bi) & Bo);
         st64[22] = Bi ^ ((~Bo) & Bu);
         st64[3] = Bo ^ ((~Bu) & Ba);
         st64[9] = Bu ^ ((~Ba) & Be);

         Bu = rol64((st64[20] ^ Da), 18);
         Ba = rol64((st64[1] ^ De), 1);
         Be = rol64((st64[7] ^ Di), 6);
         Bi = rol64((st64[13] ^ Do), 25);
         Bo = rol64((st64[19] ^ Du), 8);
         st64[20] = Ba ^ ((~Be) & Bi);
         st64[1] = Be ^ ((~Bi) & Bo);
         st64[7] = Bi ^ ((~Bo) & Bu);
         st64[13] = Bo ^ ((~Bu) & Ba);
         st64[19] = Bu ^ ((~Ba) & Be);

         Be = rol64((st64[5] ^ Da), 36);
         Bi = rol64((st64[11] ^ De), 10);
         Bo = rol64((st64[17] ^ Di), 15);
         Bu = rol64((st64[23] ^ Do), 56);
         Ba = rol64((st64[4] ^ Du), 27);
         st64[5] = Ba ^ ((~Be) & Bi);
         st64[11] = Be ^ ((~Bi) & Bo);
         st64[17] = Bi ^ ((~Bo) & Bu);
         st64[23] = Bo ^ ((~Bu) & Ba);
         st64[4] = Bu ^ ((~Ba) & Be);

         Bo = rol64((st64[15] ^ Da), 41);
         Bu = rol64((st64[21] ^ De), 2);
         Ba = rol64((st64[2] ^ Di), 62);
         Be = rol64((st64[8] ^ Do), 55);
         Bi = rol64((st64[14] ^ Du), 39);
         st64[15] = Ba ^ ((~Be) & Bi);
         st64[21] = Be ^ ((~Bi) & Bo);
         st64[2] = Bi ^ ((~Bo) & Bu);
         st64[8] = Bo ^ ((~Bu) & Ba);
         st64[14] = Bu ^ ((~Ba) & Be);

         Ca = st64[0] ^ st64[10] ^ st64[20] ^ st64[5] ^ st64[15];
         Ce = st64[6] ^ st64[16] ^ st64[1] ^ st64[11] ^ st64[21];
         Ci = st64[12] ^ st64[22] ^ st64[7] ^ st64[17] ^ st64[2];
         Co = st64[18] ^ st64[3] ^ st64[13] ^ st64[23] ^ st64[8];
         Cu = st64[24] ^ st64[9] ^ st64[19] ^ st64[4] ^ st64[14];
         Da = Cu ^ rol64(Ce, 1);
         De = Ca ^ rol64(Ci, 1);
         Di = Ce ^ rol64(Co, 1);
         Do = Ci ^ rol64(Cu, 1);
         Du = Co ^ rol64(Ca, 1);

         Ba = (st64[0] ^ Da);
         Be = rol64((st64[16] ^ De), 44);
         Bi = rol64((st64[7] ^ Di), 43);
         Bo = rol64((st64[23] ^ Do), 21);
         Bu = rol64((st64[14] ^ Du), 14);
         st64[0] = Ba ^ ((~Be) & Bi) ^ keccakf_rndc[r + 1];
         st64[16] = Be ^ ((~Bi) & Bo);
         st64[7] = Bi ^ ((~Bo) & Bu);
         st64[23] = Bo ^ ((~Bu) & Ba);
         st64[14] = Bu ^ ((~Ba) & Be);

         Bi = rol64((st64[20] ^ Da), 3);
         Bo = rol64((st64[11] ^ De), 45);
         Bu = rol64((st64[2] ^ Di), 61);
         Ba = rol64((st64[18] ^ Do), 28);
         Be = rol64((st64[9] ^ Du), 20);
         st64[20] = Ba ^ ((~Be) & Bi);
         st64[11] = Be ^ ((~Bi) & Bo);
         st64[2] = Bi ^ ((~Bo) & Bu);
         st64[18] = Bo ^ ((~Bu) & Ba);
         st64[9] = Bu ^ ((~Ba) & Be);

         Bu = rol64((st64[15] ^ Da), 18);
         Ba = rol64((st64[6] ^ De), 1);
         Be = rol64((st64[22] ^ Di), 6);
         Bi = rol64((st64[13] ^ Do), 25);
         Bo = rol64((st64[4] ^ Du), 8);
         st64[15] = Ba ^ ((~Be) & Bi);
         st64[6] = Be ^ ((~Bi) & Bo);
         st64[22] = Bi ^ ((~Bo) & Bu);
         st64[13] = Bo ^ ((~Bu) & Ba);
         st64[4] = Bu ^ ((~Ba) & Be);

         Be = rol64((st64[10] ^ Da), 36);
         Bi = rol64((st64[1] ^ De), 10);
         Bo = rol64((st64[17] ^ Di), 15);
         Bu = rol64((st64[8] ^ Do), 56);
         Ba = rol64((st64[24] ^ Du), 27);
         st64[10] = Ba ^ ((~Be) & Bi);
         st64[1] = Be ^ ((~Bi) & Bo);
         st64[17] = Bi ^ ((~Bo) & Bu);
         st64[8] = Bo ^ ((~Bu) & Ba);
         st64[24] = Bu ^ ((~Ba) & Be);

         Bo = rol64((st64[5] ^ Da), 41);
         Bu = rol64((st64[21] ^ De), 2);
         Ba = rol64((st64[12] ^ Di), 62);
         Be = rol64((st64[3] ^ Do), 55);
         Bi = rol64((st64[19] ^ Du), 39);
         st64[5] = Ba ^ ((~Be) & Bi);
         st64[21] = Be ^ ((~Bi) & Bo);
         st64[12] = Bi ^ ((~Bo) & Bu);
         st64[3] = Bo ^ ((~Bu) & Ba);
         st64[19] = Bu ^ ((~Ba) & Be);

         Ca = st64[0] ^ st64[20] ^ st64[15] ^ st64[10] ^ st64[5];
         Ce = st64[16] ^ st64[11] ^ st64[6] ^ st64[1] ^ st64[21];
         Ci = st64[7] ^ st64[2] ^ st64[22] ^ st64[17] ^ st64[12];
         Co = st64[23] ^ st64[18] ^ st64[13] ^ st64[8] ^ st64[3];
         Cu = st64[14] ^ st64[9] ^ st64[4] ^ st64[24] ^ st64[19];
         Da = Cu ^ rol64(Ce, 1);
         De = Ca ^ rol64(Ci, 1);
         Di = Ce ^ rol64(Co, 1);
         Do = Ci ^ rol64(Cu, 1);
         Du = Co ^ rol64(Ca, 1);

         Ba = (st64[0] ^ Da);
         Be = rol64((st64[11] ^ De), 44);
         Bi = rol64((st64[22] ^ Di), 43);
         Bo = rol64((st64[8] ^ Do), 21);
         Bu = rol64((st64[19] ^ Du), 14);
         st64[0] = Ba ^ ((~Be) & Bi) ^ keccakf_rndc[r + 2];
         st64[11] = Be ^ ((~Bi) & Bo);
         st64[22] = Bi ^ ((~Bo) & Bu);
         st64[8] = Bo ^ ((~Bu) & Ba);
         st64[19] = Bu ^ ((~Ba) & Be);

         Bi = rol64((st64[15] ^ Da), 3);
         Bo = rol64((st64[1] ^ De), 45);
         Bu = rol64((st64[12] ^ Di), 61);
         Ba = rol64((st64[23] ^ Do), 28);
         Be = rol64((st64[9] ^ Du), 20);
         st64[15] = Ba ^ ((~Be) & Bi);
         st64[1] = Be ^ ((~Bi) & Bo);
         st64[12] = Bi ^ ((~Bo) & Bu);
         st64[23] = Bo ^ ((~Bu) & Ba);
         st64[9] = Bu ^ ((~Ba) & Be);

         Bu = rol64((st64[5] ^ Da), 18);
         Ba = rol64((st64[16] ^ De), 1);
         Be = rol64((st64[2] ^ Di), 6);
         Bi = rol64((st64[13] ^ Do), 25);
         Bo = rol64((st64[24] ^ Du), 8);
         st64[5] = Ba ^ ((~Be) & Bi);
         st64[16] = Be ^ ((~Bi) & Bo);
         st64[2] = Bi ^ ((~Bo) & Bu);
         st64[13] = Bo ^ ((~Bu) & Ba);
         st64[24] = Bu ^ ((~Ba) & Be);

         Be = rol64((st64[20] ^ Da), 36);
         Bi = rol64((st64[6] ^ De), 10);
         Bo = rol64((st64[17] ^ Di), 15);
         Bu = rol64((st64[3] ^ Do), 56);
         Ba = rol64((st64[14] ^ Du), 27);
         st64[20] = Ba ^ ((~Be) & Bi);
         st64[6] = Be ^ ((~Bi) & Bo);
         st64[17] = Bi ^ ((~Bo) & Bu);
         st64[3] = Bo ^ ((~Bu) & Ba);
         st64[14] = Bu ^ ((~Ba) & Be);

         Bo = rol64((st64[10] ^ Da), 41);
         Bu = rol64((st64[21] ^ De), 2);
         Ba = rol64((st64[7] ^ Di), 62);
         Be = rol64((st64[18] ^ Do), 55);
         Bi = rol64((st64[4] ^ Du), 39);
         st64[10] = Ba ^ ((~Be) & Bi);
         st64[21] = Be ^ ((~Bi) & Bo);
         st64[7] = Bi ^ ((~Bo) & Bu);
         st64[18] = Bo ^ ((~Bu) & Ba);
         st64[4] = Bu ^ ((~Ba) & Be);

         Ca = st64[0] ^ st64[15] ^ st64[5] ^ st64[20] ^ st64[10];
         Ce = st64[11] ^ st64[1] ^ st64[16] ^ st64[6] ^ st64[21];
         Ci = st64[22] ^ st64[12] ^ st64[2] ^ st64[17] ^ st64[7];
         Co = st64[8] ^ st64[23] ^ st64[13] ^ st64[3] ^ st64[18];
         Cu = st64[19] ^ st64[9] ^ st64[24] ^ st64[14] ^ st64[4];
         Da = Cu ^ rol64(Ce, 1);
         De = Ca ^ rol64(Ci, 1);
         Di = Ce ^ rol64(Co, 1);
         Do = Ci ^ rol64(Cu, 1);
         Du = Co ^ rol64(Ca, 1);

         Ba = (st64[0] ^ Da);
         Be = rol64((st64[1] ^ De), 44);
         Bi = rol64((st64[2] ^ Di), 43);
         Bo = rol64((st64[3] ^ Do), 21);
         Bu = rol64((st64[4] ^ Du), 14);
         st64[0] = Ba ^ ((~Be) & Bi) ^ keccakf_rndc[r + 3];
         st64[1] = Be ^ ((~Bi) & Bo);
         st64[2] = Bi ^ ((~Bo) & Bu);
         st64[3] = Bo ^ ((~Bu) & Ba);
         st64[4] = Bu ^ ((~Ba) & Be);

         Bi = rol64((st64[5] ^ Da), 3);
         Bo = rol64((st64[6] ^ De), 45);
         Bu = rol64((st64[7] ^ Di), 61);
         Ba = rol64((st64[8] ^ Do), 28);
         Be = rol64((st64[9] ^ Du), 20);
         st64[5] = Ba ^ ((~Be) & Bi);
         st64[6] = Be ^ ((~Bi) & Bo);
         st64[7] = Bi ^ ((~Bo) & Bu);
         st64[8] = Bo ^ ((~Bu) & Ba);
         st64[9] = Bu ^ ((~Ba) & Be);

         Bu = rol64((st64[10] ^ Da), 18);
         Ba = rol64((st64[11] ^ De), 1);
         Be = rol64((st64[12] ^ Di), 6);
         Bi = rol64((st64[13] ^ Do), 25);
         Bo = rol64((st64[14] ^ Du), 8);
         st64[10] = Ba ^ ((~Be) & Bi);
         st64[11] = Be ^ ((~Bi) & Bo);
         st64[12] = Bi ^ ((~Bo) & Bu);
         st64[13] = Bo ^ ((~Bu) & Ba);
         st64[14] = Bu ^ ((~Ba) & Be);

         Be = rol64((st64[15] ^ Da), 36);
         Bi = rol64((st64[16] ^ De), 10);
         Bo = rol64((st64[17] ^ Di), 15);
         Bu = rol64((st64[18] ^ Do), 56);
         Ba = rol64((st64[19] ^ Du), 27);
         st64[15] = Ba ^ ((~Be) & Bi);
         st64[16] = Be ^ ((~Bi) & Bo);
         st64[17] = Bi ^ ((~Bo) & Bu);
         st64[18] = Bo ^ ((~Bu) & Ba);
         st64[19] = Bu ^ ((~Ba) & Be);

         Bo = rol64((st64[20] ^ Da), 41);
         Bu = rol64((st64[21] ^ De), 2);
         Ba = rol64((st64[22] ^ Di), 62);
         Be = rol64((st64[23] ^ Do), 55);
         Bi = rol64((st64[24] ^ Du), 39);
         st64[20] = Ba ^ ((~Be) & Bi);
         st64[21] = Be ^ ((~Bi) & Bo);
         st64[22] = Bi ^ ((~Bo) & Bu);
         st64[23] = Bo ^ ((~Bu) & Ba);
         st64[24] = Bu ^ ((~Ba) & Be);
      }
   }

   for (r = 0; inlen >= 8; inlen -= 8, r++) st64[r] ^= in64[r];
   ((word32 *) st64)[r << 1] ^= *((word32 *) &in64[r]);

   /* sha3_final */
   state[(r << 3) + 4] ^= keccak_final ? 0x01 : 0x06;
   state[135] ^= 0x80;
   for (r = 0; r < KECCAKFROUNDS; r += 4) {
      /* Unrolled 4 rounds at a time */

      Ca = st64[0] ^ st64[5] ^ st64[10] ^ st64[15] ^ st64[20];
      Ce = st64[1] ^ st64[6] ^ st64[11] ^ st64[16] ^ st64[21];
      Ci = st64[2] ^ st64[7] ^ st64[12] ^ st64[17] ^ st64[22];
      Co = st64[3] ^ st64[8] ^ st64[13] ^ st64[18] ^ st64[23];
      Cu = st64[4] ^ st64[9] ^ st64[14] ^ st64[19] ^ st64[24];
      Da = Cu ^ rol64(Ce, 1);
      De = Ca ^ rol64(Ci, 1);
      Di = Ce ^ rol64(Co, 1);
      Do = Ci ^ rol64(Cu, 1);
      Du = Co ^ rol64(Ca, 1);

      Ba = (st64[0] ^ Da);
      Be = rol64((st64[6] ^ De), 44);
      Bi = rol64((st64[12] ^ Di), 43);
      Bo = rol64((st64[18] ^ Do), 21);
      Bu = rol64((st64[24] ^ Du), 14);
      st64[0]  = Ba ^ ((~Be) & Bi) ^ keccakf_rndc[r];
      st64[6]  = Be ^ ((~Bi) & Bo);
      st64[12] = Bi ^ ((~Bo) & Bu);
      st64[18] = Bo ^ ((~Bu) & Ba);
      st64[24] = Bu ^ ((~Ba) & Be);

      Bi = rol64((st64[10] ^ Da), 3);
      Bo = rol64((st64[16] ^ De), 45);
      Bu = rol64((st64[22] ^ Di), 61);
      Ba = rol64((st64[3] ^ Do), 28);
      Be = rol64((st64[9] ^ Du), 20);
      st64[10] = Ba ^ ((~Be) & Bi);
      st64[16] = Be ^ ((~Bi) & Bo);
      st64[22] = Bi ^ ((~Bo) & Bu);
      st64[3] = Bo ^ ((~Bu) & Ba);
      st64[9] = Bu ^ ((~Ba) & Be);

      Bu = rol64((st64[20] ^ Da), 18);
      Ba = rol64((st64[1] ^ De), 1);
      Be = rol64((st64[7] ^ Di), 6);
      Bi = rol64((st64[13] ^ Do), 25);
      Bo = rol64((st64[19] ^ Du), 8);
      st64[20] = Ba ^ ((~Be) & Bi);
      st64[1] = Be ^ ((~Bi) & Bo);
      st64[7] = Bi ^ ((~Bo) & Bu);
      st64[13] = Bo ^ ((~Bu) & Ba);
      st64[19] = Bu ^ ((~Ba) & Be);

      Be = rol64((st64[5] ^ Da), 36);
      Bi = rol64((st64[11] ^ De), 10);
      Bo = rol64((st64[17] ^ Di), 15);
      Bu = rol64((st64[23] ^ Do), 56);
      Ba = rol64((st64[4] ^ Du), 27);
      st64[5] = Ba ^ ((~Be) & Bi);
      st64[11] = Be ^ ((~Bi) & Bo);
      st64[17] = Bi ^ ((~Bo) & Bu);
      st64[23] = Bo ^ ((~Bu) & Ba);
      st64[4] = Bu ^ ((~Ba) & Be);

      Bo = rol64((st64[15] ^ Da), 41);
      Bu = rol64((st64[21] ^ De), 2);
      Ba = rol64((st64[2] ^ Di), 62);
      Be = rol64((st64[8] ^ Do), 55);
      Bi = rol64((st64[14] ^ Du), 39);
      st64[15] = Ba ^ ((~Be) & Bi);
      st64[21] = Be ^ ((~Bi) & Bo);
      st64[2] = Bi ^ ((~Bo) & Bu);
      st64[8] = Bo ^ ((~Bu) & Ba);
      st64[14] = Bu ^ ((~Ba) & Be);

      Ca = st64[0] ^ st64[10] ^ st64[20] ^ st64[5] ^ st64[15];
      Ce = st64[6] ^ st64[16] ^ st64[1] ^ st64[11] ^ st64[21];
      Ci = st64[12] ^ st64[22] ^ st64[7] ^ st64[17] ^ st64[2];
      Co = st64[18] ^ st64[3] ^ st64[13] ^ st64[23] ^ st64[8];
      Cu = st64[24] ^ st64[9] ^ st64[19] ^ st64[4] ^ st64[14];
      Da = Cu ^ rol64(Ce, 1);
      De = Ca ^ rol64(Ci, 1);
      Di = Ce ^ rol64(Co, 1);
      Do = Ci ^ rol64(Cu, 1);
      Du = Co ^ rol64(Ca, 1);

      Ba = (st64[0] ^ Da);
      Be = rol64((st64[16] ^ De), 44);
      Bi = rol64((st64[7] ^ Di), 43);
      Bo = rol64((st64[23] ^ Do), 21);
      Bu = rol64((st64[14] ^ Du), 14);
      st64[0] = Ba ^ ((~Be) & Bi) ^ keccakf_rndc[r + 1];
      st64[16] = Be ^ ((~Bi) & Bo);
      st64[7] = Bi ^ ((~Bo) & Bu);
      st64[23] = Bo ^ ((~Bu) & Ba);
      st64[14] = Bu ^ ((~Ba) & Be);

      Bi = rol64((st64[20] ^ Da), 3);
      Bo = rol64((st64[11] ^ De), 45);
      Bu = rol64((st64[2] ^ Di), 61);
      Ba = rol64((st64[18] ^ Do), 28);
      Be = rol64((st64[9] ^ Du), 20);
      st64[20] = Ba ^ ((~Be) & Bi);
      st64[11] = Be ^ ((~Bi) & Bo);
      st64[2] = Bi ^ ((~Bo) & Bu);
      st64[18] = Bo ^ ((~Bu) & Ba);
      st64[9] = Bu ^ ((~Ba) & Be);

      Bu = rol64((st64[15] ^ Da), 18);
      Ba = rol64((st64[6] ^ De), 1);
      Be = rol64((st64[22] ^ Di), 6);
      Bi = rol64((st64[13] ^ Do), 25);
      Bo = rol64((st64[4] ^ Du), 8);
      st64[15] = Ba ^ ((~Be) & Bi);
      st64[6] = Be ^ ((~Bi) & Bo);
      st64[22] = Bi ^ ((~Bo) & Bu);
      st64[13] = Bo ^ ((~Bu) & Ba);
      st64[4] = Bu ^ ((~Ba) & Be);

      Be = rol64((st64[10] ^ Da), 36);
      Bi = rol64((st64[1] ^ De), 10);
      Bo = rol64((st64[17] ^ Di), 15);
      Bu = rol64((st64[8] ^ Do), 56);
      Ba = rol64((st64[24] ^ Du), 27);
      st64[10] = Ba ^ ((~Be) & Bi);
      st64[1] = Be ^ ((~Bi) & Bo);
      st64[17] = Bi ^ ((~Bo) & Bu);
      st64[8] = Bo ^ ((~Bu) & Ba);
      st64[24] = Bu ^ ((~Ba) & Be);

      Bo = rol64((st64[5] ^ Da), 41);
      Bu = rol64((st64[21] ^ De), 2);
      Ba = rol64((st64[12] ^ Di), 62);
      Be = rol64((st64[3] ^ Do), 55);
      Bi = rol64((st64[19] ^ Du), 39);
      st64[5] = Ba ^ ((~Be) & Bi);
      st64[21] = Be ^ ((~Bi) & Bo);
      st64[12] = Bi ^ ((~Bo) & Bu);
      st64[3] = Bo ^ ((~Bu) & Ba);
      st64[19] = Bu ^ ((~Ba) & Be);

      Ca = st64[0] ^ st64[20] ^ st64[15] ^ st64[10] ^ st64[5];
      Ce = st64[16] ^ st64[11] ^ st64[6] ^ st64[1] ^ st64[21];
      Ci = st64[7] ^ st64[2] ^ st64[22] ^ st64[17] ^ st64[12];
      Co = st64[23] ^ st64[18] ^ st64[13] ^ st64[8] ^ st64[3];
      Cu = st64[14] ^ st64[9] ^ st64[4] ^ st64[24] ^ st64[19];
      Da = Cu ^ rol64(Ce, 1);
      De = Ca ^ rol64(Ci, 1);
      Di = Ce ^ rol64(Co, 1);
      Do = Ci ^ rol64(Cu, 1);
      Du = Co ^ rol64(Ca, 1);

      Ba = (st64[0] ^ Da);
      Be = rol64((st64[11] ^ De), 44);
      Bi = rol64((st64[22] ^ Di), 43);
      Bo = rol64((st64[8] ^ Do), 21);
      Bu = rol64((st64[19] ^ Du), 14);
      st64[0] = Ba ^ ((~Be) & Bi) ^ keccakf_rndc[r + 2];
      st64[11] = Be ^ ((~Bi) & Bo);
      st64[22] = Bi ^ ((~Bo) & Bu);
      st64[8] = Bo ^ ((~Bu) & Ba);
      st64[19] = Bu ^ ((~Ba) & Be);

      Bi = rol64((st64[15] ^ Da), 3);
      Bo = rol64((st64[1] ^ De), 45);
      Bu = rol64((st64[12] ^ Di), 61);
      Ba = rol64((st64[23] ^ Do), 28);
      Be = rol64((st64[9] ^ Du), 20);
      st64[15] = Ba ^ ((~Be) & Bi);
      st64[1] = Be ^ ((~Bi) & Bo);
      st64[12] = Bi ^ ((~Bo) & Bu);
      st64[23] = Bo ^ ((~Bu) & Ba);
      st64[9] = Bu ^ ((~Ba) & Be);

      Bu = rol64((st64[5] ^ Da), 18);
      Ba = rol64((st64[16] ^ De), 1);
      Be = rol64((st64[2] ^ Di), 6);
      Bi = rol64((st64[13] ^ Do), 25);
      Bo = rol64((st64[24] ^ Du), 8);
      st64[5] = Ba ^ ((~Be) & Bi);
      st64[16] = Be ^ ((~Bi) & Bo);
      st64[2] = Bi ^ ((~Bo) & Bu);
      st64[13] = Bo ^ ((~Bu) & Ba);
      st64[24] = Bu ^ ((~Ba) & Be);

      Be = rol64((st64[20] ^ Da), 36);
      Bi = rol64((st64[6] ^ De), 10);
      Bo = rol64((st64[17] ^ Di), 15);
      Bu = rol64((st64[3] ^ Do), 56);
      Ba = rol64((st64[14] ^ Du), 27);
      st64[20] = Ba ^ ((~Be) & Bi);
      st64[6] = Be ^ ((~Bi) & Bo);
      st64[17] = Bi ^ ((~Bo) & Bu);
      st64[3] = Bo ^ ((~Bu) & Ba);
      st64[14] = Bu ^ ((~Ba) & Be);

      Bo = rol64((st64[10] ^ Da), 41);
      Bu = rol64((st64[21] ^ De), 2);
      Ba = rol64((st64[7] ^ Di), 62);
      Be = rol64((st64[18] ^ Do), 55);
      Bi = rol64((st64[4] ^ Du), 39);
      st64[10] = Ba ^ ((~Be) & Bi);
      st64[21] = Be ^ ((~Bi) & Bo);
      st64[7] = Bi ^ ((~Bo) & Bu);
      st64[18] = Bo ^ ((~Bu) & Ba);
      st64[4] = Bu ^ ((~Ba) & Be);

      Ca = st64[0] ^ st64[15] ^ st64[5] ^ st64[20] ^ st64[10];
      Ce = st64[11] ^ st64[1] ^ st64[16] ^ st64[6] ^ st64[21];
      Ci = st64[22] ^ st64[12] ^ st64[2] ^ st64[17] ^ st64[7];
      Co = st64[8] ^ st64[23] ^ st64[13] ^ st64[3] ^ st64[18];
      Cu = st64[19] ^ st64[9] ^ st64[24] ^ st64[14] ^ st64[4];
      Da = Cu ^ rol64(Ce, 1);
      De = Ca ^ rol64(Ci, 1);
      Di = Ce ^ rol64(Co, 1);
      Do = Ci ^ rol64(Cu, 1);
      Du = Co ^ rol64(Ca, 1);

      Ba = (st64[0] ^ Da);
      Be = rol64((st64[1] ^ De), 44);
      Bi = rol64((st64[2] ^ Di), 43);
      Bo = rol64((st64[3] ^ Do), 21);
      Bu = rol64((st64[4] ^ Du), 14);
      st64[0] = Ba ^ ((~Be) & Bi) ^ keccakf_rndc[r + 3];
      st64[1] = Be ^ ((~Bi) & Bo);
      st64[2] = Bi ^ ((~Bo) & Bu);
      st64[3] = Bo ^ ((~Bu) & Ba);
      st64[4] = Bu ^ ((~Ba) & Be);

      Bi = rol64((st64[5] ^ Da), 3);
      Bo = rol64((st64[6] ^ De), 45);
      Bu = rol64((st64[7] ^ Di), 61);
      Ba = rol64((st64[8] ^ Do), 28);
      Be = rol64((st64[9] ^ Du), 20);
      st64[5] = Ba ^ ((~Be) & Bi);
      st64[6] = Be ^ ((~Bi) & Bo);
      st64[7] = Bi ^ ((~Bo) & Bu);
      st64[8] = Bo ^ ((~Bu) & Ba);
      st64[9] = Bu ^ ((~Ba) & Be);

      Bu = rol64((st64[10] ^ Da), 18);
      Ba = rol64((st64[11] ^ De), 1);
      Be = rol64((st64[12] ^ Di), 6);
      Bi = rol64((st64[13] ^ Do), 25);
      Bo = rol64((st64[14] ^ Du), 8);
      st64[10] = Ba ^ ((~Be) & Bi);
      st64[11] = Be ^ ((~Bi) & Bo);
      st64[12] = Bi ^ ((~Bo) & Bu);
      st64[13] = Bo ^ ((~Bu) & Ba);
      st64[14] = Bu ^ ((~Ba) & Be);

      Be = rol64((st64[15] ^ Da), 36);
      Bi = rol64((st64[16] ^ De), 10);
      Bo = rol64((st64[17] ^ Di), 15);
      Bu = rol64((st64[18] ^ Do), 56);
      Ba = rol64((st64[19] ^ Du), 27);
      st64[15] = Ba ^ ((~Be) & Bi);
      st64[16] = Be ^ ((~Bi) & Bo);
      st64[17] = Bi ^ ((~Bo) & Bu);
      st64[18] = Bo ^ ((~Bu) & Ba);
      st64[19] = Bu ^ ((~Ba) & Be);

      Bo = rol64((st64[20] ^ Da), 41);
      Bu = rol64((st64[21] ^ De), 2);
      Ba = rol64((st64[22] ^ Di), 62);
      Be = rol64((st64[23] ^ Do), 55);
      Bi = rol64((st64[24] ^ Du), 39);
      st64[20] = Ba ^ ((~Be) & Bi);
      st64[21] = Be ^ ((~Bi) & Bo);
      st64[22] = Bi ^ ((~Bo) & Bu);
      st64[23] = Bo ^ ((~Bu) & Ba);
      st64[24] = Bu ^ ((~Ba) & Be);
   }

   /* sha3_output */
   ((word64 *) out)[0] = st64[0];
   ((word64 *) out)[1] = st64[1];
   ((word64 *) out)[2] = st64[2];
   ((word64 *) out)[3] = st64[3];
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
   __constant__ static uint32_t c_float[4] = {
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
            for (z = 0; z < len64; z++) qp[z] ^= WORD64_C(0x8181818181818181);
            for (z <<= 1; z < len32; z++) dp[z] ^= WORD32_C(0x81818181);
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
__device__ void cu_peach_nighthash(void *in, size_t inlen,
   word32 index, size_t txlen, void *out)
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
      case 2: cu_peach_sha1(in, inlen, out); break;
      case 3: cu_peach_sha256(in, inlen, out); break;
      case 4: cu_peach_sha3(in, inlen, 0, out); break;
      case 5: cu_peach_sha3(in, inlen, 1, out); break;
      case 6: cu_peach_md2(in, inlen, out); break;
      case 7: cu_peach_md5(in, inlen, out); break;
   }  /* end switch(algo_type)... */
}  /* end cu_peach_nighthash() */

/**
 * @private
 * Generate a tile of the Peach map.
 * @param index Index number of tile to generate
 * @param tilep Pointer to location to place generated tile
*/
__device__ void cu_peach_generate(word32 index, word32 *tilep)
{
   int i;

   /* place initial data into seed */
   tilep[0] = index;
   tilep[1] = ((word32 *) c_phash)[0];
   tilep[2] = ((word32 *) c_phash)[1];
   tilep[3] = ((word32 *) c_phash)[2];
   tilep[4] = ((word32 *) c_phash)[3];
   tilep[5] = ((word32 *) c_phash)[4];
   tilep[6] = ((word32 *) c_phash)[5];
   tilep[7] = ((word32 *) c_phash)[6];
   tilep[8] = ((word32 *) c_phash)[7];
   /* perform initial nighthash into first row of tile */
   cu_peach_nighthash(tilep, PEACHGENLEN, index, PEACHGENLEN, tilep);
   /* fill the rest of the tile with the preceding Nighthash result */
   for (i = 0; i < (PEACHTILELEN - 32) / 4; i += 8) {
      tilep[i + 8] = index;
      cu_peach_nighthash(&tilep[i], PEACHGENLEN, index, SHA256LEN,
         &tilep[i + 8]);
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
__device__ void cu_peach_jump(word32 *index, word64 *nonce, word32 *tilep)
{
   word32 seed[PEACHJUMPLEN / 4];
   word32 dhash[SHA256LEN / 4];
   int i;

   /* construct seed for use as Nighthash input for this index on the map */
   ((word64 *) seed)[0] = nonce[0];
   ((word64 *) seed)[1] = nonce[1];
   ((word64 *) seed)[2] = nonce[2];
   ((word64 *) seed)[3] = nonce[3];
   seed[8] = *index;
#pragma unroll
   for (i = 0; i < PEACHTILELEN / 4; i++) {
      seed[i + 9] = tilep[i];
   }

   /* perform nighthash on PEACHJUMPLEN bytes of seed */
   cu_peach_nighthash(seed, PEACHJUMPLEN, *index, 0, dhash);
   /* sum hash as 8x 32-bit unsigned integers */
   *index = (
      dhash[0] + dhash[1] + dhash[2] + dhash[3] +
      dhash[4] + dhash[5] + dhash[6] + dhash[7]
   ) & PEACHCACHELEN_M1;
}  /* end cu_peach_jump() */

/**
 * CUDA kernel for bulk generation of Peach Map tiles.
 * @param d_map Device pointer to location of Peach Map
 * @param offset Index number offset to generate tiles from
 */
__global__ void kcu_peach_build(word8 *d_map, word32 offset)
{
   const word32 index = ((blockDim.x * blockIdx.x) + threadIdx.x) + offset;
   if (index < PEACHCACHELEN) {
      cu_peach_generate(index, (word32 *) &d_map[index * PEACHTILELEN]);
   }
}  /* end kcu_peach_build() */

/**
 * CUDA kernel for solving a tokenized haiku as nonce output for Peach proof
 * of work. Combine haiku protocols implemented in the Trigg Algorithm with
 * the memory intensive protocols of the Peach algorithm to generate haiku
 * output as proof of work.
 * @param d_map Device pointer to Peach Map
 * @param d_ictx Device pointer to incomplete hashing contexts
 * @param d_solve Device pointer to location to place nonce on solve
*/
__global__ void kcu_peach_solve(word8 *d_map, SHA256_CTX *d_ictx,
   word8 *d_solve)
{
   word64 nonce[4];
   word8 hash[SHA256LEN];
   SHA256_CTX ictx;
   word32 *x, mario, tid, i;

   tid = (blockIdx.x * blockDim.x) + threadIdx.x;

   /* shift ictx to appropriate location and extract nonce */
#pragma unroll
   for (i = 0; i < sizeof(ictx) / 4; i++) {
      ((word32 *) &ictx)[i] = ((word32 *) &d_ictx[tid])[i];
   }
#pragma unroll
   for (i = 0; i < 8; i++) {
      ((word32 *) nonce)[i] = ((word32 *) &ictx.data[28])[i];
   }
   /* finalise incomplete sha256 hash */
   cu_sha256_final(&ictx, hash);
   /* initialize mario's starting index on the map, bound to PEACHCACHELEN */
   for (mario = hash[0], i = 1; i < SHA256LEN; i++) {
      mario *= hash[i];
   }
   mario &= PEACHCACHELEN_M1;
   /* perform tile jumps to find the final tile x8 */
   for (i = 0; i < PEACHROUNDS; i++) {
      cu_peach_jump(&mario, nonce, (word32 *) &d_map[mario * PEACHTILELEN]);
   }
   /* hash block trailer with final tile */
   cu_sha256_init(&ictx);
   cu_sha256_update(&ictx, hash, SHA256LEN);
   cu_sha256_update(&ictx, &d_map[mario * PEACHTILELEN], PEACHTILELEN);
   cu_sha256_final(&ictx, hash);
   /* Coarse/Fine evaluation checks */
   x = (word32 *) hash;
   for(i = c_diff >> 5; i; i--) if(*(x++) != 0) return;
   if(__clz(__byte_perm(*x, 0, 0x0123)) < (c_diff & 31)) return;

   /* check first to solve with atomic solve handling */
   if(!atomicCAS((int *) d_solve, 0, *((int *) nonce))) {
      ((word64 *) d_solve)[0] = nonce[0];
      ((word64 *) d_solve)[1] = nonce[1];
      ((word64 *) d_solve)[2] = nonce[2];
      ((word64 *) d_solve)[3] = nonce[3];
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
__global__ void kcu_peach_checkhash(SHA256_CTX *ictx, word8 *out,
   word8 *eval)
{
   word64 nonce[4] = { 0 };
   word32 tile[PEACHTILELEN] = { 0 };
   word8 hash[SHA256LEN] = { 0 };
   word32 *x, mario;
   int i;

   /* restricted to a single thread for debug purposes */
   if ((blockDim.x * blockIdx.x) + threadIdx.x > 0) return;

   /* extract nonce */
#pragma unroll
   for (i = 0; i < 8; i++) {
      ((word32 *) nonce)[i] = ((word32 *) &ictx->data[28])[i];
   }
   /* finalise incomplete sha256 hash */
   cu_sha256_final(ictx, hash);
   /* initialize mario's starting index on the map, bound to PEACHCACHELEN */
   for(mario = hash[0], i = 1; i < SHA256LEN; i++) mario *= hash[i];
   mario &= PEACHCACHELEN_M1;
   /* generate and perform tile jumps to find the final tile x8 */
   for (i = 0; i < PEACHROUNDS; i++) {
      cu_peach_generate(mario, tile);
      cu_peach_jump(&mario, nonce, tile);
   }
   /* generate the last tile */
   cu_peach_generate(mario, tile);
   /* hash block trailer with final tile */
   cu_sha256_init(ictx);
   cu_sha256_update(ictx, hash, SHA256LEN);
   cu_sha256_update(ictx, tile, PEACHTILELEN);
   cu_sha256_final(ictx, hash);
   /* pass final hash to out */
   memcpy(out, hash, SHA256LEN);
   /* Coarse/Fine evaluation checks */
   *eval = 1;
   x = (word32 *) hash;
   for(i = c_diff >> 5; i; i--) if(*(x++) != 0) return;
   if(__clz(__byte_perm(*x, 0, 0x0123)) < (c_diff & 31)) return;
   *eval = 0;
}  /* end kcu_peach_checkhash() */

/**
 * Check Peach proof of work algorithm with a CUDA device.
 * @param bt Pointer to block trailer to check
 * @param diff Difficulty to test against entropy of final hash
 * @param out Pointer to location to place final hash, if non-null
 * @returns VEOK on success, else VERROR
*/
int peach_checkhash_cuda(BTRAILER *btp, word8 diff, void *out)
{
   SHA256_CTX *d_ictx, ictx;
   word8 *d_hash, *d_eval;
   word8 eval = 0;
   int count;

   cuCHK(cudaGetDeviceCount(&count), NULL, return (-1));
   if (count < 1) {
      pfatal("No CUDA devices...");
      return -1;
   }
   cuCHK(cudaSetDevice(0), NULL, return (-1));
   cuCHK(cudaMalloc(&d_ictx, sizeof(SHA256_CTX)), NULL, return (-1));
   cuCHK(cudaMalloc(&d_hash, SHA256LEN), NULL, return (-1));
   cuCHK(cudaMalloc(&d_eval, 1), NULL, return (-1));
   cuCHK(cudaMemset(d_eval, 0xff, 1), NULL, return (-1));
   /* prepare intermediate state for next round */
   sha256_init(&ictx);
   sha256_update(&ictx, btp, 124);
   /* transfer phash to device */
   cuCHK(cudaMemcpyToSymbol(c_diff, btp->difficulty, 1, 0,
      cudaMemcpyHostToDevice), NULL, return (-1));
   /* transfer phash to device */
   cuCHK(cudaMemcpyToSymbol(c_phash, btp->phash, SHA256LEN, 0,
      cudaMemcpyHostToDevice), NULL, return (-1));
   /* transfer ictx to device */
   cuCHK(cudaMemcpy(d_ictx, &ictx, sizeof(SHA256_CTX),
      cudaMemcpyHostToDevice), NULL, return (-1));
   /* launch kernel to check Peach */
   kcu_peach_checkhash<<<1, 1>>>(d_ictx, d_hash, d_eval);
   cuCHK(cudaGetLastError(), NULL, return (-1));
   /* retrieve hash/eval data */
   cuCHK(cudaMemcpy(out, d_hash, SHA256LEN,
      cudaMemcpyDeviceToHost), NULL, return (-1));
   cuCHK(cudaMemcpy(&eval, d_eval, 1,
      cudaMemcpyDeviceToHost), NULL, return (-1));
   /* wait for device to finish */
   cuCHK(cudaDeviceSynchronize(), NULL, return (-1));
   /* free memory */
   cuCHK(cudaFree(d_ictx), NULL, return (-1));
   cuCHK(cudaFree(d_hash), NULL, return (-1));
   cuCHK(cudaFree(d_eval), NULL, return (-1));
   /* return */
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
   if (ctxp->h_solve) cudaFreeHost(ctxp->h_solve);
   if (ctxp->h_ictx) cudaFreeHost(ctxp->h_ictx);
   if (ctxp->d_solve[0]) cudaFree(ctxp->d_solve[0]);
   if (ctxp->d_solve[1]) cudaFree(ctxp->d_solve[1]);
   if (ctxp->d_ictx[0]) cudaFree(ctxp->d_ictx[0]);
   if (ctxp->d_ictx[1]) cudaFree(ctxp->d_ictx[1]);
   if (ctxp->d_map) cudaFree(ctxp->d_map);

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
int peach_init_cuda_device(DEVICE_CTX *devp, int id)
{
   static int nvml_initialized = 0;
   static unsigned nvml_count = 0;

   struct cudaDeviceProp props;
   nvmlPciInfo_t pci;
   nvmlDevice_t *nvmlp;
   size_t ictxlen;
   unsigned i, gen, width;

   if (nvml_initialized == 0) {
      /* set nvml initialized */
      nvml_initialized = 1;
      /* initialize nvml */
      if (nvmlInit() != NVML_SUCCESS) {
         nvml_count = pdebug("Unable to initialize NVML");
         plog("No NVML devices detected...");
      } else if (nvmlDeviceGetCount(&nvml_count) != NVML_SUCCESS) {
         nvml_count = pdebug("Unable to obtain NVML count");
         plog("No NVML devices detected...");
      }
   }

   devp->id = id;
   devp->type = CUDA_DEVICE;
   nvmlp = &(PeachCudaCTX[id].nvml_device);
   /* get CUDA properties for verification with nvml device */
   if (cudaGetDeviceProperties(&props, id) != cudaSuccess) {
      perr("cudaGetDeviceProperties(%d)", id);
   } else {
      /* scan nvml devices for match */
      PeachCudaCTX[id].nvml_enabled = 0;
      for (i = 0; i < nvml_count; i++) {
         memset(nvmlp, 0, sizeof(nvmlDevice_t));
         if (nvmlDeviceGetHandleByIndex(i, nvmlp) == NVML_SUCCESS ||
            (nvmlDeviceGetPciInfo(*nvmlp, &pci) == NVML_SUCCESS) ||
            (pci.device == props.pciDeviceID) ||
            (pci.domain == props.pciDomainID) ||
            (pci.bus == props.pciBusID)) {
            /* obtain link gen/width */
            if (nvmlDeviceGetCurrPcieLinkGeneration(*nvmlp, &gen)
                  != NVML_SUCCESS) gen = 0;
            if (nvmlDeviceGetCurrPcieLinkWidth(*nvmlp, &width)
                  != NVML_SUCCESS) width = 0;
            PeachCudaCTX[id].nvml_enabled = 1;
            break;
         }
      }
      /* store GPU name, PCI Id and gen info in nameId */
      snprintf(devp->nameId, sizeof(devp->nameId),
         "%04u:%02u:%02u:%.128s Gen%1ux%02u", props.pciDomainID,
         props.pciDeviceID, props.pciBusID, props.name, gen, width);
   }
   /* set context to CUDA id */
   cuCHK(cudaSetDevice(id), devp, return VERROR);
   /* set CUDA configuration for device */
   if (cudaOccupancyMaxPotentialBlockSize(&(devp->grid), &(devp->block),
         kcu_peach_solve, 0, 0) != cudaSuccess) {
      pdebug("cudaOccupancy~BlockSize(%d) failed...", id);
      pdebug("Using conservative defaults for <<<512/128>>>");
      devp->grid = 512;
      devp->block = 128;
   }
   /* calculate total threads and ictxlist size */
   devp->threads = devp->grid * devp->block;
   ictxlen = sizeof(SHA256_CTX) * devp->threads;
   /* create streams for device */
   cuCHK(cudaStreamCreate(&(PeachCudaCTX[id].stream[0])), devp, return VERROR);
   cuCHK(cudaStreamCreate(&(PeachCudaCTX[id].stream[1])), devp, return VERROR);
   /* allocate pinned host memory for host/device transfers */
   cuCHK(cudaMallocHost(&(PeachCudaCTX[id].h_solve[0]), 32), devp, return VERROR);
   cuCHK(cudaMallocHost(&(PeachCudaCTX[id].h_solve[1]), 32), devp, return VERROR);
   cuCHK(cudaMallocHost(&(PeachCudaCTX[id].h_ictx[0]), ictxlen), devp, return VERROR);
   cuCHK(cudaMallocHost(&(PeachCudaCTX[id].h_ictx[1]), ictxlen), devp, return VERROR);
   cuCHK(cudaMallocHost(&(PeachCudaCTX[id].h_bt[0]), sizeof(BTRAILER)), devp, return VERROR);
   cuCHK(cudaMallocHost(&(PeachCudaCTX[id].h_bt[1]), sizeof(BTRAILER)), devp, return VERROR);
   /* allocate device memory for host/device transfers */
   cuCHK(cudaMalloc(&(PeachCudaCTX[id].d_solve[0]), 32), devp, return VERROR);
   cuCHK(cudaMalloc(&(PeachCudaCTX[id].d_solve[1]), 32), devp, return VERROR);
   cuCHK(cudaMalloc(&(PeachCudaCTX[id].d_ictx[0]), ictxlen), devp, return VERROR);
   cuCHK(cudaMalloc(&(PeachCudaCTX[id].d_ictx[1]), ictxlen), devp, return VERROR);
   /* allocate memory for Peach map on device */
   cuCHK(cudaMalloc(&(PeachCudaCTX[id].d_map), PEACHMAPLEN), devp, return VERROR);
   /* clear device/host allocated memory */
   cudaMemsetAsync(PeachCudaCTX[id].d_ictx[0], 0, ictxlen, cudaStreamDefault);
   cuCHK(cudaGetLastError(), devp, return VERROR);
   cudaMemsetAsync(PeachCudaCTX[id].d_ictx[1], 0, ictxlen, cudaStreamDefault);
   cuCHK(cudaGetLastError(), devp, return VERROR);
   cudaMemsetAsync(PeachCudaCTX[id].d_solve[0], 0, 32, cudaStreamDefault);
   cuCHK(cudaGetLastError(), devp, return VERROR);
   cudaMemsetAsync(PeachCudaCTX[id].d_solve[1], 0, 32, cudaStreamDefault);
   cuCHK(cudaGetLastError(), devp, return VERROR);
   memset(PeachCudaCTX[id].h_bt[0], 0, sizeof(BTRAILER));
   memset(PeachCudaCTX[id].h_bt[1], 0, sizeof(BTRAILER));
   memset(PeachCudaCTX[id].h_ictx[0], 0, ictxlen);
   memset(PeachCudaCTX[id].h_ictx[1], 0, ictxlen);
   memset(PeachCudaCTX[id].h_solve[0], 0, 32);
   memset(PeachCudaCTX[id].h_solve[1], 0, 32);

   return VEOK;
}  /* end peach_init_cuda_device() */

/**
 * Initialize a DEVICE_CTX list with CUDA devices for solving the Peach
 * proof of work algorithm.
 * @param devlist Pointer to DEVICE_CTX list to initialize
 * @param max Maximum number of CUDA devices to initialize
 * @returns number of CUDA devices available for initialization
 * @note It is possible to have "some" CUDA devices fail to initialize.
*/
int peach_init_cuda(DEVICE_CTX devlist[], int max)
{
   static int cuda_initialized = 0;
   static int cuda_num = 0;

   int id;

   /* avoid re-initialization attempts */
   if (cuda_initialized) return cuda_num;

   /* check for cuda driver and devices */
   switch(cudaGetDeviceCount(&cuda_num)) {
      case cudaErrorNoDevice:
         return plog("No CUDA devices detected...");
      case cudaErrorInsufficientDriver:
         pfatal("Insufficient CUDA Driver. Update display drivers...");
         return 0;
      case cudaSuccess:
         if (cuda_num > max) {
            cuda_num = max;
            plog("CUDA Devices: %d (limited)\n", cuda_num);
            perr("CUDA device count EXCEEDED maximum count parameter!");
            pwarn("Some CUDA devices will not be utilized.");
            plog("Please advise developers if this is an issue...");
         }
         break;
      default:
         pfatal("Unknown CUDA initialization error occured...");
         return 0;
   }

   /* set initialized */
   cuda_initialized = 1;
   if (cuda_num < 1) return (cuda_num = 0);

   /* allocate memory for PeachCudaCTX */
   PeachCudaCTX =
      (PEACH_CUDA_CTX *) malloc(sizeof(PEACH_CUDA_CTX) * cuda_num);

   /* allocate pinned host memory for data consistant across devices */
   cuCHK(cudaMallocHost(&h_phash, SHA256LEN), NULL, return (cuda_num = 0));
   cuCHK(cudaMallocHost(&h_diff, 1), NULL, return (cuda_num = 0));
   memset(h_phash, 0, SHA256LEN);
   memset(h_diff, 0, 1);

   /* initialize device contexts for CUDA num devices */
   for(id = 0; id < cuda_num; id++) {
      peach_init_cuda_device(&devlist[id], id);
   }

   return cuda_num;
}  /* end peach_init_cuda() */

/**
 * Try solve for a tokenized haiku as nonce output for Peach proof of work
 * on CUDA devices. Combine haiku protocols implemented in the Trigg
 * Algorithm with the memory intensive protocols of the Peach algorithm to
 * generate haiku output as proof of work.
 * @param dev Pointer to DEVICE_CTX to perform work with
 * @param bt Pointer to block trailer to solve for
 * @param diff Difficulty to test against entropy of final hash
 * @param btout Pointer to location to place solved block trailer
 * @returns VEOK on solve, else VERROR
*/
int peach_solve_cuda(DEVICE_CTX *dev, BTRAILER *bt, word8 diff, BTRAILER *btout)
{
   int i, id, sid, grid, block, build;
   PEACH_CUDA_CTX *P;
   nvmlReturn_t nr;
   size_t ictxlen;

   /* check for GPU failure */
   if (dev->status == DEV_FAIL) return VERROR;

   id = dev->id;
   P = &PeachCudaCTX[id];
   /* set/check cuda device */
   cuCHK(cudaSetDevice(id), dev, return VERROR);
   cuCHK(cudaGetLastError(), dev, return VERROR);

   /* ensure initialization is complete */
   if (dev->status == DEV_NULL) {
      if (cudaStreamQuery(cudaStreamDefault) != cudaSuccess) return VERROR;
      /* set next action to build Peach map */
      dev->status = DEV_INIT;
      dev->last_work = time(NULL);
      dev->total_work = 0;
      dev->work = 0;
   }

   /* build peach map */
   if (dev->status == DEV_INIT) {
      /* build peach map -- init */
      if (dev->work == 0) {
         /* ensure both streams have finished */
         if (cudaStreamQuery(P->stream[1]) == cudaSuccess
            && cudaStreamQuery(P->stream[0]) == cudaSuccess) {
            /* synchronize device before initializing new peach map */
            cudaDeviceSynchronize();
            /* clear any late solves */
            cuCHK(cudaMemset(P->d_solve[0], 0, 32), dev, return VERROR);
            cuCHK(cudaMemset(P->d_solve[1], 0, 32), dev, return VERROR);
            memset(P->h_solve[0], 0, 32);
            memset(P->h_solve[1], 0, 32);
            /* update block trailer */
            memcpy(P->h_bt[0], bt, sizeof(BTRAILER));
            memcpy(P->h_bt[1], bt, sizeof(BTRAILER));
            /* ensure phash is set */
            memcpy(h_phash, bt->phash, SHA256LEN);
            /* asynchronous copy to phash and difficulty symbols */
            cuCHK(cudaMemcpyToSymbol(c_phash, h_phash, SHA256LEN, 0,
               cudaMemcpyHostToDevice), dev, return VERROR);
            /* update h_diff with diff or bt->difficulty[0] */
            *h_diff = diff ? diff : bt->difficulty[0];
            if (*h_diff > bt->difficulty[0]) *h_diff = bt->difficulty[0];
            cuCHK(cudaMemcpyToSymbol(c_diff, h_diff, 1, 0,
               cudaMemcpyHostToDevice), dev, return VERROR);
            /* synchronize memory transfers before building peach map */
            cudaDeviceSynchronize();
            /* flag build ready */
            build = 1;
         }
      }
      /* build peach map -- build */
      if (dev->work < PEACHCACHELEN) {
         for (sid = 0; sid < 2 && (build || dev->work > 0); sid++) {
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
               (P->d_map, (word32) dev->work);
            cuCHK(cudaGetLastError(), dev, return VERROR);
            /* update build progress */
            dev->work += grid * block;
         }
      } else {
         /* ensure both streams have finished */
         if (cudaStreamQuery(P->stream[1]) == cudaSuccess
            && cudaStreamQuery(P->stream[0]) == cudaSuccess) {
            /* build is complete */
            dev->last_work = time(NULL);
            dev->status = DEV_IDLE;
            dev->work = 0;
         }
      }
   }

   /* check for unsolved work in block trailer */
   if (dev->status == DEV_IDLE && get32(bt->tcount)) {
      if (cmp64(bt->bnum, btout->bnum)) dev->status = DEV_WORK;
   }

   /* solve work in block trailer */
   if (dev->status == DEV_WORK) {
      for(sid = 0; sid < 2; sid++) {
         if (cudaStreamQuery(P->stream[sid]) != cudaSuccess) continue;
         /* check trailer for block update */
         if (memcmp(P->h_bt[sid]->phash, bt->phash, HASHLEN)) {
            dev->status = DEV_INIT;
            dev->work = 0;
            break;
         }
         /* switch to idle mode if no transactions or already solved bnum */
         if (get32(bt->tcount) == 0 || cmp64(bt->bnum, btout->bnum) == 0) {
            dev->status = DEV_IDLE;
            dev->work = 0;
            break;
         }
         /* check for solves */
         if (*(P->h_solve[sid])) {
            /* move solved nonce */
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
         if (diff != *h_diff) {
            *h_diff = diff;
            cuCHK(cudaMemcpyToSymbol(c_diff, h_diff, 1, 0,
               cudaMemcpyHostToDevice), dev, return VERROR);
         }
         /* ensure block trailer is updated */
         memcpy(P->h_bt[sid], bt, BTSIZE);
         /* generate nonce directly into block trailer */
         trigg_generate_fast(P->h_bt[sid]->nonce);
         trigg_generate_fast(P->h_bt[sid]->nonce + 16);
         /* prepare intermediate state for next round */
         sha256_init(P->h_ictx[sid]);
         sha256_update(P->h_ictx[sid], P->h_bt[sid], 124);
         /* duplicate intermediate state with random second seed */
         for(i = 1; i < dev->threads; i++) {
            memcpy(&(P->h_ictx[sid][i]), P->h_ictx[sid], sizeof(SHA256_CTX));
            trigg_generate_fast(P->h_ictx[sid][i].data + 44);
         }
         /* transfer ictx to device */
         ictxlen = sizeof(SHA256_CTX) * dev->threads;
         cudaMemcpyAsync(P->d_ictx[sid], P->h_ictx[sid], ictxlen,
            cudaMemcpyHostToDevice, P->stream[sid]);
         cuCHK(cudaGetLastError(), dev, return VERROR);
         /* launch kernel to solve Peach */
         kcu_peach_solve<<<dev->grid, dev->block, 0, P->stream[sid]>>>
            (P->d_map, P->d_ictx[sid], P->d_solve[sid]);
         cuCHK(cudaGetLastError(), dev, return VERROR);
         /* retrieve solve seed */
         cudaMemcpyAsync(P->h_solve[sid], P->d_solve[sid], 32,
            cudaMemcpyDeviceToHost, P->stream[sid]);
         cuCHK(cudaGetLastError(), dev, return VERROR);
         /* increment progress counters */
         dev->total_work += dev->threads;
         dev->work += dev->threads;
      }
   }

   /* power and temperature monitoring (1 second interval) */
   if (P->nvml_enabled && difftime(time(NULL), dev->last_monitor)) {
      dev->last_monitor = time(NULL);
      /* get GPU device power */
      unsigned int fan;
      nr = nvmlDeviceGetFanSpeed(P->nvml_device, &fan);
      if (nr != NVML_SUCCESS) {
         perr("nvml(%d) fan speed: %s\n", id, nvmlErrorString(nr));
         memset(&(P->nvml_device), 0, sizeof(nvmlDevice_t));
         P->nvml_enabled = 0;
      } else dev->fan = fan;
      /* get GPU device power */
      unsigned int power;
      nr = nvmlDeviceGetPowerUsage(P->nvml_device, &power);
      if (nr != NVML_SUCCESS) {
         perr("nvml(%d) power usage: %s\n", id, nvmlErrorString(nr));
         memset(&(P->nvml_device), 0, sizeof(nvmlDevice_t));
         P->nvml_enabled = 0;
      } else dev->pow = power / 1000;
      /* get GPU device temperature */
      unsigned int temperature;
      nr = nvmlDeviceGetTemperature(P->nvml_device, NVML_TEMPERATURE_GPU,
         &temperature);
      if (nr != NVML_SUCCESS) {
         perr("nvml(%d) temperature: %s\n", id, nvmlErrorString(nr));
         memset(&(P->nvml_device), 0, sizeof(nvmlDevice_t));
         P->nvml_enabled = 0;
      } else dev->temp = temperature;
      /* get GPU device utilization */
      nvmlUtilization_t utilization;
      nr = nvmlDeviceGetUtilizationRates(P->nvml_device, &utilization);
      if (nr != NVML_SUCCESS) {
         perr("nvml(%d) utilization rates: %s\n", id, nvmlErrorString(nr));
         memset(&(P->nvml_device), 0, sizeof(nvmlDevice_t));
         P->nvml_enabled = 0;
      } else dev->util = utilization.gpu;
   }

   return VERROR;
}  /* end peach_solve_cuda() */

/* end include guard */
#endif
