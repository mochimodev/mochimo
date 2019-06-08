/*
 * cuda_trigg.cu  Multi-GPU CUDA Mining
 *
 * Copyright (c) 2018 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 10 August 2018
 * Revision: 31
 *
 * Attribution:  The portions of this code on lines 20 through 233 are work
 * made for hire by a Mochimo Discord user, and are not subject to to Mochimo
 * Cryptocurrency Engine License Agreement.  The remainder of this file below
 * line 233 is subject to the license as found in LICENSE.PDF
 * 
 * Anon Discord User: Let Stack know if you want attribution, and we'll give
 * you a proper credit here.  As of our last conversation you just wanted to
 * be paid with no attribution, which frankly feels a little weird to everyone.
 *
 */

#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

#include <cuda_runtime.h>

extern "C" {
#include "../../crypto/sha256.h"
}

#include "../../config.h"
#include "peach.h"

__constant__ static uint32_t __align__(8) c_midstate256[8];
__constant__ static uint32_t __align__(8) c_phash[8];
__constant__ static uint32_t __align__(8) c_input32[11];
__constant__ static uint8_t __align__(8) c_difficulty;
__constant__ static uint32_t __align__(8) c_K[64] =
{
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2
};
__constant__ static int Z_PREP[4]  = {12,13,14,15};
__constant__ static int Y_PREP[2]  = {16,17};
__constant__ static int Z_ING[16]  = {18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33};
__constant__ static int Y_ING[8]   = {34,35,36,37,38,39,40,41};
__constant__ static int X_ING[2]   = {42,43};
__constant__ static int Z_INF[16]  = {44,45,46,47,48,50,51,52,53,54,55,56,57,58,59,60};
__constant__ static int Z_ADJ[64]  = {61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,
                                      81,82,83,84,85,86,87,88,89,90,91,92,94,95,96,97,98,99,100,
                                      101,102,103,104,105,107,108,109,110,112,114,115,116,117,118,
                                      119,120,121,122,123,124,125,126,127,128};
__constant__ static int Z_AMB[16]  = {77,94,95,96,126,214,217,218,220,222,223,224,225,226,227,228};
__constant__ static int Z_TIMED[8] = {84,243,249,250,251,252,253,255};
__constant__ static int Z_NS[64]   = {129,130,131,132,133,134,135,136,137,138,145,149,154,155,156,
                                      157,177,178,179,180,182,183,184,185,186,187,188,189,190,191,
                                      192,193,194,196,197,198,199,200,201,202,203,204,205,206,207,
                                      208,209,210,211,212,213,241,244,245,246,247,248,249,250,251,
                                      252,253,254,255};
__constant__ static int Z_NPL[32]  = {139,140,141,142,143,144,146,147,148,150,151,153,158,159,160,
                                      161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,
                                      176,181};
__constant__ static int Z_MASS[16] = {214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229};
__constant__ static int Y_MASS[8]  = {230,231,232,233,234,235,236,237};
__constant__ static int X_MASS[4]  = {238,239,240,242};
__constant__ static int Z_INGINF[32] = {18,19,20,21,22,25,26,27,28,29,30,36,37,38,39,40,41,42,44,
                                        46,47,48,49,51,52,53,54,55,56,57,58,59};
__constant__ static int Z_TIME[16] = {82,83,84,85,86,87,88,243,249,250,251,252,253,254,255,253};
__constant__ static int Z_INGADJ[64] = {18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,
                                        37,38,39,40,41,42,43,23,24,31,32,33,34,61,62,63,64,65,66,
                                        67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,
                                        86,87,88,89,90,91,92};









// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}









#ifdef __CUDA_ARCH__
__device__ __forceinline__ uint32_t cuda_swab32(uint32_t x)
{
    /* device */
    return __byte_perm(x, x, 0x0123);
}
#else
    /* host */
    #define cuda_swab32(x) \
    ((((x) << 24) & 0xff000000u) | (((x) << 8) & 0x00ff0000u) | \
        (((x) >> 8) & 0x0000ff00u) | (((x) >> 24) & 0x000000ffu))
#endif

#define xor3b(a,b,c) (a ^ b ^ c)
#define ROTR32(x, n) __funnelshift_r( (x), (x), (n) )

__device__ __forceinline__
uint64_t xandx(uint64_t a, uint64_t b, uint64_t c)
{
    uint64_t result;
    asm("{ // xandx \n\t"
        ".reg .u64 n;\n\t"
        "xor.b64 %0, %2, %3;\n\t"
        "and.b64 n, %0, %1;\n\t"
        "xor.b64 %0, n, %3;\n\t"
    "}\n" : "=l"(result) : "l"(a), "l"(b), "l"(c));
    return result;
}

#define AS_UINT2(addr) *((uint2*)(addr))

__device__ __forceinline__ uint32_t bsg2_0(const uint32_t x)
{
    return xor3b(ROTR32(x,2),ROTR32(x,13),ROTR32(x,22));
}

__device__ __forceinline__ uint32_t bsg2_1(const uint32_t x)
{
    return xor3b(ROTR32(x,6),ROTR32(x,11),ROTR32(x,25));
}

__device__ __forceinline__ uint32_t ssg2_0(const uint32_t x)
{
    return xor3b(ROTR32(x,7),ROTR32(x,18),(x>>3));
}

__device__ __forceinline__ uint32_t ssg2_1(const uint32_t x)
{
    return xor3b(ROTR32(x,17),ROTR32(x,19),(x>>10));
}

__device__ __forceinline__ uint32_t andor32(const uint32_t a, const uint32_t b, const uint32_t c)
{
    uint32_t result;
    asm("{\n\t"
        ".reg .u32 m,n,o;\n\t"
        "and.b32 m,  %1, %2;\n\t"
        " or.b32 n,  %1, %2;\n\t"
        "and.b32 o,   n, %3;\n\t"
        " or.b32 %0,  m, o ;\n\t"
        "}\n\t" : "=r"(result) : "r"(a), "r"(b), "r"(c)
    );
    return result;
}

__device__ __forceinline__ uint2 vectorizeswap(uint64_t v)
{
    uint2 result;
    asm("mov.b64 {%0,%1},%2; \n\t"
        : "=r"(result.y), "=r"(result.x) : "l"(v));
    return result;
}

__device__
static void sha2_step1(uint32_t a, uint32_t b, uint32_t c, uint32_t &d, uint32_t e, uint32_t f, uint32_t g, uint32_t &h,
    uint32_t in, const uint32_t Kshared)
{
    uint32_t t1,t2;
    uint32_t vxandx = xandx(e, f, g);
    uint32_t bsg21 = bsg2_1(e);
    uint32_t bsg20 = bsg2_0(a);
    uint32_t andorv = andor32(a,b,c);

    t1 = h + bsg21 + vxandx + Kshared + in;
    t2 = bsg20 + andorv;
    d = d + t1;
    h = t1 + t2;
}

__device__
static void sha2_step2(uint32_t a, uint32_t b, uint32_t c, uint32_t &d, uint32_t e, uint32_t f, uint32_t g, uint32_t &h,
    uint32_t* in, uint32_t pc, const uint32_t Kshared)
{
    uint32_t t1,t2;

    int pcidx1 = (pc-2) & 0xF;
    int pcidx2 = (pc-7) & 0xF;
    int pcidx3 = (pc-15) & 0xF;

    uint32_t inx0 = in[pc];
    uint32_t inx1 = in[pcidx1];
    uint32_t inx2 = in[pcidx2];
    uint32_t inx3 = in[pcidx3];

    uint32_t ssg21 = ssg2_1(inx1);
    uint32_t ssg20 = ssg2_0(inx3);
    uint32_t vxandx = xandx(e, f, g);
    uint32_t bsg21 = bsg2_1(e);
    uint32_t bsg20 = bsg2_0(a);
    uint32_t andorv = andor32(a,b,c);

    in[pc] = ssg21 + inx2 + ssg20 + inx0;

    t1 = h + bsg21 + vxandx + Kshared + in[pc];
    t2 = bsg20 + andorv;
    d =  d + t1;
    h = t1 + t2;
}

__device__
static void sha256_round(uint32_t* in, uint32_t* state, uint32_t* const Kshared)
{
    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];

    sha2_step1(a,b,c,d,e,f,g,h,in[ 0], Kshared[ 0]);
    sha2_step1(h,a,b,c,d,e,f,g,in[ 1], Kshared[ 1]);
    sha2_step1(g,h,a,b,c,d,e,f,in[ 2], Kshared[ 2]);
    sha2_step1(f,g,h,a,b,c,d,e,in[ 3], Kshared[ 3]);
    sha2_step1(e,f,g,h,a,b,c,d,in[ 4], Kshared[ 4]);
    sha2_step1(d,e,f,g,h,a,b,c,in[ 5], Kshared[ 5]);
    sha2_step1(c,d,e,f,g,h,a,b,in[ 6], Kshared[ 6]);
    sha2_step1(b,c,d,e,f,g,h,a,in[ 7], Kshared[ 7]);
    sha2_step1(a,b,c,d,e,f,g,h,in[ 8], Kshared[ 8]);
    sha2_step1(h,a,b,c,d,e,f,g,in[ 9], Kshared[ 9]);
    sha2_step1(g,h,a,b,c,d,e,f,in[10], Kshared[10]);
    sha2_step1(f,g,h,a,b,c,d,e,in[11], Kshared[11]);
    sha2_step1(e,f,g,h,a,b,c,d,in[12], Kshared[12]);
    sha2_step1(d,e,f,g,h,a,b,c,in[13], Kshared[13]);
    sha2_step1(c,d,e,f,g,h,a,b,in[14], Kshared[14]);
    sha2_step1(b,c,d,e,f,g,h,a,in[15], Kshared[15]);

    #pragma unroll
    for (int i=0; i<3; i++)
    {
        sha2_step2(a,b,c,d,e,f,g,h,in,0, Kshared[16+16*i]);
        sha2_step2(h,a,b,c,d,e,f,g,in,1, Kshared[17+16*i]);
        sha2_step2(g,h,a,b,c,d,e,f,in,2, Kshared[18+16*i]);
        sha2_step2(f,g,h,a,b,c,d,e,in,3, Kshared[19+16*i]);
        sha2_step2(e,f,g,h,a,b,c,d,in,4, Kshared[20+16*i]);
        sha2_step2(d,e,f,g,h,a,b,c,in,5, Kshared[21+16*i]);
        sha2_step2(c,d,e,f,g,h,a,b,in,6, Kshared[22+16*i]);
        sha2_step2(b,c,d,e,f,g,h,a,in,7, Kshared[23+16*i]);
        sha2_step2(a,b,c,d,e,f,g,h,in,8, Kshared[24+16*i]);
        sha2_step2(h,a,b,c,d,e,f,g,in,9, Kshared[25+16*i]);
        sha2_step2(g,h,a,b,c,d,e,f,in,10,Kshared[26+16*i]);
        sha2_step2(f,g,h,a,b,c,d,e,in,11,Kshared[27+16*i]);
        sha2_step2(e,f,g,h,a,b,c,d,in,12,Kshared[28+16*i]);
        sha2_step2(d,e,f,g,h,a,b,c,in,13,Kshared[29+16*i]);
        sha2_step2(c,d,e,f,g,h,a,b,in,14,Kshared[30+16*i]);
        sha2_step2(b,c,d,e,f,g,h,a,in,15,Kshared[31+16*i]);
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

__device__ static void sha256_tile(uint32_t *in, uint32_t *tile, uint32_t *out) {
  /**
   * Assume in[8] pointer
   * Plus til[256] pointer
   * Plus an additional unsigned - 0x120 bits */
  int i, j;
  uint32_t input[16] = {in[0],in[1],in[2],in[3],in[4],in[5],in[6],in[7],
                        0,0,0,0,0,0,0,0};
  /**
   * Initialize hash */
  out[0] = 0x6a09e667L;
  out[1] = 0xbb67ae85L;
  out[2] = 0x3c6ef372L;
  out[3] = 0xa54ff53aL;
  out[4] = 0x510e527fL;
  out[5] = 0x9b05688cL;
  out[6] = 0x1f83d9abL;
  out[7] = 0x5be0cd19L;
  /**
   * Feed in data and hash */
  for(i = 0, j = 8; i < 248; ) {
    if(j == 16) {
      sha256_round(input, out, c_K);
      j = 0;
    }
    input[j++] = tile[i++];
  }
  /* Feed in remaining data and final hash */
  for(i = 0; i < 8; i++)
    input[i] = tile[248 + i];

  input[8] = 0x80000000;
  for(i = 9; i < 15; i++)
    input[i] = 0;

  input[15] = 0x2100;
  
  sha256_round(input, out, c_K);
}

__device__ static void sha256_row(uint32_t *in, uint32_t index,
                                   uint32_t *out) {
  /**
   * Assume in[8] pointer
   * Plus an additional unsigned - 0x120 bits */
  uint32_t input[16] = {in[0],in[1],in[2],in[3],in[4],in[5],in[6],in[7],
                        cuda_swab32(index),0x80000000,0,0,0,0,0,0x120};
  /**
   * Initialize hash */
  out[0] = 0x6a09e667L;
  out[1] = 0xbb67ae85L;
  out[2] = 0x3c6ef372L;
  out[3] = 0xa54ff53aL;
  out[4] = 0x510e527fL;
  out[5] = 0x9b05688cL;
  out[6] = 0x1f83d9abL;
  out[7] = 0x5be0cd19L;
  /**
   * Hash in data */
  sha256_round(input, out, c_K);
}

__device__ uint32_t cuda_next_index(uint32_t tileNum, uint32_t* tile, uint32_t* nonce) {
  int i, j;
  uint32_t index, hash[8];
  
  /**
   * Assume tile[256] pointer and nonce[4] pointer
   * Plus an additional unsigned - 0x20A0 bits */
  uint32_t input[16] = {nonce[0],nonce[1],nonce[2],nonce[3],nonce[4],
                        nonce[5],nonce[6],nonce[7],cuda_swab32(tileNum),
                        0,0,0,0,0,0,0};
  /**
   * Feed in data and hash */
  for(i = 0, j = 9; i < 247; ) {
    if(j == 16) {
      sha256_round(input, hash, c_K);
      j = 0;
    }
    input[j++] = tile[i++];
  }
  /* Feed in remaining data and final hash */
  for(i = 0; i < 9; i++)
    input[i] = tile[247 + i];

  input[9] = 0x80000000;
  for(i = 10; i < 15; i++)
    input[i] = 0;
    
  input[15] = 0x2120;
  
  sha256_round(input, hash, c_K);

  index =  *(uint32_t*)&hash[0]; //read first 4 bytes as unsigned int
  index += *(uint32_t*)&hash[4]; //read next 4 bytes as unsigned int

  return index % MAP;
}

__device__ void cuda_gen_tile(uint32_t tnum, uint32_t *phash,
                        uint32_t *g_map, uint8_t *g_cache) {
  /* declarations */
  uint32_t *mapp, op;
  uint8_t *tilep, _104, _72, bits;
  int i, j, k, l, t, z, exp;
  float *floatp;

  /* set tile and map pointer */
  tilep = (uint8_t*)&g_map[tnum*256];
  mapp = &g_map[tnum*256];
  
  /* begin tile data */
  sha256_row(phash, tnum, mapp);
  
  /* set operation variables */
  _104 = 104;
  _72 = 72;

    for(j = k = l = 0; j < 256; j+=8) {
      for(op = 0; k < j+8; k++) {
        /* set float pointer */
        floatp = (float*)&mapp[k];
        
        /**
         * Order of operations dependant on initial 8 bits
         * Operations:
         *   1) right shift by 4 to obtain the exponent value
         *   2) 50% chance of exponent being negative
         *   3) 50% chance of changing sign of float */
        if(tilep[l] & 1) {
          l++;
          exp = tilep[l++] >> 4;
          if(tilep[l++] & 1) exp ^= 0x80000000;
          if(tilep[l++] & 1) *floatp = -(*floatp);
        } else
        if(tilep[l] & 2) {
          l++;
          exp = tilep[l++] >> 4;
          if(tilep[l++] & 1) *floatp = -(*floatp);
          if(tilep[l++] & 1) exp ^= 0x80000000;
        } else {
          l++;
          if(tilep[l++] & 1) *floatp = -(*floatp);
          exp = tilep[l++] >> 4;
          if(tilep[l++] & 1) exp ^= 0x80000000;
        }

        /* replace NaN's with tnum */
        if(isnan(*floatp))
          *floatp = (float) tnum;

        /* perform floating point operation */
        *floatp = ldexpf(*floatp, exp);
        
        /* pre-scramble op */
        op ^= mapp[k];
      }
      
      /* perform 8x bit manipulations per row */
      for(t = 0; t < 8; t++) {
        /* determine tile byte offset and operation to use */
        op += mapp[i + (t % 8)];
        switch(op & 7) {
		  case 0: /* Swap the first and last bit in each byte. */
		  {
			  for(z = 0;z<32;z++)
				  tilep[i + z] ^= 0x81;
          }
            break;
          case 1: /* Swap bytes */
          {
        	  for(z = 0;z<16;z++)
        	  {
        		 bits = tilep[i + z];
				 tilep[i + z] = tilep[i + 16 + z];
				 tilep[i + 16 + z] = bits;
        	  }
          }
            break;
          case 2: /* Complement One all bytes */
          {
        	  for(z = 1; z < 32; z++)
        		  tilep[i + z] = ~tilep[i + z];
          }
            break;
          case 3: /* Alternate +1 and -1 on all bytes */
          {
            for(z = 0; z < 32; z++)
              tilep[i + z] += (z & 1 == 0) ? 1 : -1;
          }
            break;
          case 4: /* Alternate +t and -t on all bytes */
          {
            for(z = 0; z < 32; z++)
              tilep[i + z] += (z & 1 == 0) ? -t : t;
          }
            break;
          case 5: /* Replace every occurrence of h with H */
          {
        	  for(z = 0;z<32;z++)
        		  if(tilep[i + z] == _104)
        			  tilep[i + z] = _72;
          }
            break;
          case 6: /* If byte a is > byte b, swap them. */
          {
        	for(z = 0;z<16;z++)
        	{
        		if(tilep[i + z] > tilep[i + 16 + z])
        		{
        			bits = tilep[i + z];
					tilep[i + z] = tilep[i + 16 + z];
					tilep[i + 16 + z] = bits;
				 }
        	}
          }
            break;
          case 7 : /* XOR all bytes */
          {
            for(z = 1; z < 32; z++)
              tilep[i + z] ^= tilep[i + z - 1];
          }
          	  break;
		}
      }
      
      /* hash the result of the previous tile's row to the next */
      if(k < 256)
        sha256_row(&mapp[j], tnum, &mapp[k]);
    }
  
  g_cache[tnum] = 1;
}

__global__ void cuda_find_peach(uint32_t threads, uint32_t *g_map, uint8_t *g_cache, 
                           int *g_found, uint8_t *g_seed) {

  const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t sm, input[16], bt_hash[8], fhash[8], nonce[8] = {0}, *tilep, *bp, n;
  uint8_t seed[16] = {0};
  int i, j;

  /* split second seed workload between threads */
   if (thread <= threads) {
      /* Frame 1 -> Split 6 ways */
      if(thread < 32768) { /* Total Permutations, this frame: 32,768 ( 1 << 15 ) */
         seed[ 0] = Z_PREP[(thread & 3)];       // 2^2
         seed[ 1] = Z_TIMED[(thread >> 2) & 7]; // 2^3
         seed[ 2] = 1;
         seed[ 3] = 5;
         seed[ 4] = Z_NS[(thread >> 5) & 63];   // 2^6
         seed[ 5] = 1;
         seed[ 6] = Z_ING[(thread >> 11) & 15]; // 2^4
      } else
      if(thread < 49152) { /* Total Permutations, this frame: 16,384 ( 1 << 14 ) */
         seed[ 0] = Y_PREP[(thread & 1)];       // 2^1
         seed[ 1] = Z_TIMED[(thread >> 1) & 7]; // 2^3
         seed[ 2] = 1;
         seed[ 3] = 5;
         seed[ 4] = Z_NS[(thread >> 4) & 63];   // 2^6
         seed[ 5] = 1;
         seed[ 6] = Z_ING[(thread >> 10) & 15]; // 2^4
      } else
      if(thread < 65536) { /* Total Permutations, this frame: 16,384 ( 1 << 14 ) */
         seed[ 0] = Z_PREP[(thread & 3)];       // 2^2
         seed[ 1] = Z_TIMED[(thread >> 2) & 7]; // 2^3
         seed[ 2] = 1;
         seed[ 3] = 5;
         seed[ 4] = Z_NS[(thread >> 5) & 63];   // 2^6
         seed[ 5] = 1;
         seed[ 6] = Y_ING[(thread >> 11) & 7]; // 2^3
      } else
      if(thread < 73728) { /* Total Permutations, this frame: 8,192 ( 1 << 13 ) */
         seed[ 0] = Y_PREP[(thread & 1)];       // 2^1
         seed[ 1] = Z_TIMED[(thread >> 1) & 7]; // 2^3
         seed[ 2] = 1;
         seed[ 3] = 5;
         seed[ 4] = Z_NS[(thread >> 4) & 63];   // 2^6
         seed[ 5] = 1;
         seed[ 6] = Y_ING[(thread >> 10) & 7]; // 2^3
      } else
      if(thread < 81920) { /* Total Permutations, this frame: 16,384 ( 1 << 13 ) */
         seed[ 0] = Z_PREP[(thread & 3)];       // 2^2
         seed[ 1] = Z_TIMED[(thread >> 2) & 7]; // 2^3
         seed[ 2] = 1;
         seed[ 3] = 5;
         seed[ 4] = Z_NS[(thread >> 5) & 63];   // 2^6
         seed[ 5] = 1;
         seed[ 6] = X_ING[(thread >> 11) & 3]; // 2^2
      } else
      if(thread < 86016) { /* Total Permutations, this frame: 8,192 ( 1 << 12 ) */
         seed[ 0] = Y_PREP[(thread & 1)];       // 2^1
         seed[ 1] = Z_TIMED[(thread >> 1) & 7]; // 2^3
         seed[ 2] = 1;
         seed[ 3] = 5;
         seed[ 4] = Z_NS[(thread >> 4) & 63];   // 2^6
         seed[ 5] = 1;
         seed[ 6] = X_ING[(thread >> 10) & 3];  // 2^2
      } else
      /* END Frame 1 */
      /* Frame 2 -> Split 3 ways */
      if(thread <= 151552) { /* Total Permutations, this frame: 65,536 (1 << 16) */
         seed[ 0] = Z_TIME[(thread & 15)];      // 2^4
         seed[ 1] = Y_MASS[(thread >> 4) & 15]; // 2^4
         seed[ 2] = 1;
         seed[ 3] = Z_INF[(thread >> 8) & 15];  // 2^4
         seed[ 4] = 9;
         seed[ 5] = 2;
         seed[ 6] = 1;
         seed[ 7] = Z_AMB[(thread >> 12) & 15]; // 2^4
      } else
      if(thread <= 184320) { /* Total Permutations, this frame: 32,768 (1 << 15) */
         seed[ 0] = Z_TIME[(thread & 15)];      // 2^4
         seed[ 1] = X_MASS[(thread >> 3) & 7];  // 2^3
         seed[ 2] = 1;
         seed[ 3] = Z_INF[(thread >> 7) & 15];  // 2^4
         seed[ 4] = 9;
         seed[ 5] = 2;
         seed[ 6] = 1;
         seed[ 7] = Z_AMB[(thread >> 11) & 15]; // 2^4
      } else
      if(thread <= 200704) { /* Total Permutations, this frame: 16,384 (1 << 14) */
         seed[ 0] = Z_TIME[(thread & 15)];      // 2^4
         seed[ 1] = Z_MASS[(thread >> 2) & 3];  // 2^2
         seed[ 2] = 1;
         seed[ 3] = Z_INF[(thread >> 6) & 15];  // 2^4
         seed[ 4] = 9;
         seed[ 5] = 2;
         seed[ 6] = 1;
         seed[ 7] = Z_AMB[(thread >> 10) & 15]; // 2^4
      } else
      /* END Frame 2 */
      /* Frame 3 -> Split 2 ways */
      if(thread < 2297856) { /* Total Permutations, this frame: 2,097,152 ( 1 << 21 )*/
         seed[ 0] = Z_PREP[(thread & 3)];          // 2^2
         seed[ 1] = Z_TIMED[(thread >> 2) & 7];    // 2^3
         seed[ 2] = 1;
         seed[ 3] = Z_ADJ[(thread >> 5) & 63];     // 2^6
         seed[ 4] = Z_NPL[(thread >> 11) & 31];    // 2^5
         seed[ 5] = 1;
         seed[ 6] = Z_INGINF[(thread >> 16) & 31]; // 2^5
      } else
      if(thread < 3346432) { /* Total Permutations, this frame: 1,048,576 ( 1 << 20 )*/
         seed[ 0] = Y_PREP[(thread & 1)];          // 2^1
         seed[ 1] = Z_TIMED[(thread >> 1) & 7];    // 2^3
         seed[ 2] = 1;
         seed[ 3] = Z_ADJ[(thread >> 4) & 63];     // 2^6
         seed[ 4] = Z_NPL[(thread >> 10) & 31];    // 2^5
         seed[ 5] = 1;
         seed[ 6] = Z_INGINF[(thread >> 15) & 31]; // 2^5
      } else
      /* END Frame 3 */
      /* Frame 4 -> Split 6 ways */
      if(thread < 5443584) { /* Total Permutations, this frame: 2,097,152 ( 1 << 21 ) */
         seed[ 0] = 5;
         seed[ 1] = Z_NS[(thread & 63)];           // 2^6
         seed[ 2] = 1;
         seed[ 3] = Z_PREP[(thread >> 6) & 3];     // 2^2
         seed[ 4] = Z_TIMED[(thread >> 8) & 7];    // 2^3
         seed[ 5] = Z_MASS[(thread >> 11) & 15];   // 2^4
         seed[ 6] = 3;
         seed[ 7] = 1;
         seed[ 8] = Z_ADJ[(thread >> 15) & 63];    // 2^6
      } else
      if(thread < 6492160) { /* Total Permutations, this frame: 1,048,576 ( 1 << 20 ) */
         seed[ 0] = 5;
         seed[ 1] = Z_NS[(thread & 63)];           // 2^6
         seed[ 2] = 1;
         seed[ 3] = Y_PREP[(thread >> 6) & 1];     // 2^1
         seed[ 4] = Z_TIMED[(thread >> 7) & 7];    // 2^3
         seed[ 5] = Z_MASS[(thread >> 10) & 15];   // 2^4
         seed[ 6] = 3;
         seed[ 7] = 1;
         seed[ 8] = Z_ADJ[(thread >> 14) & 63];    // 2^6
      } else
      if(thread < 7540736) { /* Total Permutations, this frame: 1,048,576 ( 1 << 20 ) */
         seed[ 0] = 5;
         seed[ 1] = Z_NS[(thread & 63)];           // 2^6
         seed[ 2] = 1;
         seed[ 3] = Z_PREP[(thread >> 6) & 3];     // 2^2
         seed[ 4] = Z_TIMED[(thread >> 8) & 7];    // 2^3
         seed[ 5] = Y_MASS[(thread >> 11) & 7];    // 2^3
         seed[ 6] = 3;
         seed[ 7] = 1;
         seed[ 8] = Z_ADJ[(thread >> 14) & 63];    // 2^6
      } else
      if(thread < 8065024) { /* Total Permutations, this frame: 524,288 ( 1 << 19 ) */
         seed[ 0] = 5;
         seed[ 1] = Z_NS[(thread & 63)];           // 2^6
         seed[ 2] = 1;
         seed[ 3] = Y_PREP[(thread >> 6) & 1];     // 2^1
         seed[ 4] = Z_TIMED[(thread >> 7) & 7];    // 2^3
         seed[ 5] = Y_MASS[(thread >> 10) & 7];    // 2^3
         seed[ 6] = 3;
         seed[ 7] = 1;
         seed[ 8] = Z_ADJ[(thread >> 13) & 63];    // 2^6
      } else
      if(thread < 8589312) { /* Total Permutations, this frame: 524,288 ( 1 << 19 ) */
         seed[ 0] = 5;
         seed[ 1] = Z_NS[(thread & 63)];           // 2^6
         seed[ 2] = 1;
         seed[ 3] = Z_PREP[(thread >> 6) & 3];     // 2^2
         seed[ 4] = Z_TIMED[(thread >> 8) & 7];    // 2^3
         seed[ 5] = X_MASS[(thread >> 11) & 3];    // 2^2
         seed[ 6] = 3;
         seed[ 7] = 1;
         seed[ 8] = Z_ADJ[(thread >> 13) & 63];    // 2^6
      } else
      if(thread < 8851456) { /* Total Permutations, this frame: 262,144 ( 1 << 18 ) */
         seed[ 0] = 5;
         seed[ 1] = Z_NS[(thread & 63)];           // 2^6
         seed[ 2] = 1;
         seed[ 3] = Y_PREP[(thread >> 6) & 1];     // 2^1
         seed[ 4] = Z_TIMED[(thread >> 7) & 7];    // 2^3
         seed[ 5] = X_MASS[(thread >> 10) & 3];    // 2^2
         seed[ 6] = 3;
         seed[ 7] = 1;
         seed[ 8] = Z_ADJ[(thread >> 12) & 63];    // 2^6
      } else
      /* END Frame 4 */
      /* Frame 5 -> Split 6 ways */
      if(thread < 13045760) { /* Total Permutations, this frame: 4,194,304 ( 1 << 22 ) */
         seed[ 0] = Z_PREP[thread & 3];            // 2^2
         seed[ 1] = Z_ADJ[(thread >> 2) & 63];     // 2^6
         seed[ 2] = Z_MASS[(thread >> 8) & 15];    // 2^4
         seed[ 3] = 1;
         seed[ 4] = Z_NPL[(thread >> 12) & 31];    // 2^5
         seed[ 5] = 1;
         seed[ 6] = Z_INGINF[(thread >> 17) & 31]; // 2^5
      } else
      if(thread < 15142912) { /* Total Permutations, this frame: 2,097,152 ( 1 << 21 ) */
         seed[ 0] = Y_PREP[thread & 1];            // 2^1
         seed[ 1] = Z_ADJ[(thread >> 1) & 63];     // 2^6
         seed[ 2] = Z_MASS[(thread >> 7) & 15];    // 2^4
         seed[ 3] = 1;
         seed[ 4] = Z_NPL[(thread >> 11) & 31];    // 2^5
         seed[ 5] = 1;
         seed[ 6] = Z_INGINF[(thread >> 16) & 31]; // 2^5
      } else
      if(thread < 17240064) { /* Total Permutations, this frame: 2,097,152 ( 1 << 21 ) */
         seed[ 0] = Z_PREP[thread & 3];            // 2^2
         seed[ 1] = Z_ADJ[(thread >> 2) & 63];     // 2^6
         seed[ 2] = Y_MASS[(thread >> 8) & 7];     // 2^3
         seed[ 3] = 1;
         seed[ 4] = Z_NPL[(thread >> 11) & 31];    // 2^5
         seed[ 5] = 1;
         seed[ 6] = Z_INGINF[(thread >> 16) & 31]; // 2^5
      } else
      if(thread < 18288640) { /* Total Permutations, this frame: 1,048,576 ( 1 << 20 ) */
         seed[ 0] = Y_PREP[thread & 1];            // 2^1
         seed[ 1] = Z_ADJ[(thread >> 1) & 63];     // 2^6
         seed[ 2] = Y_MASS[(thread >> 7) & 7];     // 2^3
         seed[ 3] = 1;
         seed[ 4] = Z_NPL[(thread >> 10) & 31];    // 2^5
         seed[ 5] = 1;
         seed[ 6] = Z_INGINF[(thread >> 15) & 31]; // 2^5
      } else
      if(thread < 19337216) { /* Total Permutations, this frame: 1,048,576 ( 1 << 20 ) */
         seed[ 0] = Z_PREP[thread & 3];            // 2^2
         seed[ 1] = Z_ADJ[(thread >> 2) & 63];     // 2^6
         seed[ 2] = X_MASS[(thread >> 8) & 3];     // 2^2
         seed[ 3] = 1;
         seed[ 4] = Z_NPL[(thread >> 10) & 31];    // 2^5
         seed[ 5] = 1;
         seed[ 6] = Z_INGINF[(thread >> 15) & 31]; // 2^5
      } else
      if(thread < 19861504) { /* Total Permutations, this frame: 524,288 ( 1 << 19 ) */
         seed[ 0] = Y_PREP[thread & 1];            // 2^1
         seed[ 1] = Z_ADJ[(thread >> 1) & 63];     // 2^6
         seed[ 2] = X_MASS[(thread >> 7) & 3];     // 2^2
         seed[ 3] = 1;
         seed[ 4] = Z_NPL[(thread >> 9) & 31];     // 2^5
         seed[ 5] = 1;
         seed[ 6] = Z_INGINF[(thread >> 14) & 31]; // 2^5
      } else
      /* END Frame 5 */
      /* Frame 6 -> Split 6 ways */
      if(thread < 24055808) { /* Total Permutations, this frame: 4,194,304 ( 1 << 22 ) */
         seed[ 0] = Z_PREP[(thread & 3)];          // 2^2
         seed[ 1] = Z_MASS[(thread >> 2) & 15];    // 2^4
         seed[ 2] = 1;
         seed[ 3] = Z_ADJ[(thread >> 6) & 63];     // 2^6
         seed[ 4] = Z_NPL[(thread >> 12) & 31];    // 2^5
         seed[ 5] = 1;
         seed[ 6] = Z_INGINF[(thread >> 17) & 31]; // 2^5
      } else
      if(thread < 26152960) { /* Total Permutations, this frame: 2,097,152 ( 1 << 21 ) */
         seed[ 0] = Y_PREP[(thread & 1)];          // 2^1
         seed[ 1] = Z_MASS[(thread >> 1) & 15];    // 2^4
         seed[ 2] = 1;
         seed[ 3] = Z_ADJ[(thread >> 5) & 63];     // 2^6
         seed[ 4] = Z_NPL[(thread >> 11) & 31];    // 2^5
         seed[ 5] = 1;
         seed[ 6] = Z_INGINF[(thread >> 16) & 31]; // 2^5
      } else
      if(thread < 28250112) { /* Total Permutations, this frame: 2,097,152 ( 1 << 21 ) */
         seed[ 0] = Z_PREP[(thread & 3)];          // 2^2
         seed[ 1] = Y_MASS[(thread >> 2) & 7];     // 2^3
         seed[ 2] = 1;
         seed[ 3] = Z_ADJ[(thread >> 5) & 63];     // 2^6
         seed[ 4] = Z_NPL[(thread >> 11) & 31];    // 2^5
         seed[ 5] = 1;
         seed[ 6] = Z_INGINF[(thread >> 16) & 31]; // 2^5
      } else
      if(thread < 29298688) { /* Total Permutations, this frame: 1,048,576 ( 1 << 20 ) */
         seed[ 0] = Y_PREP[(thread & 1)];          // 2^1
         seed[ 1] = Y_MASS[(thread >> 1) & 7];     // 2^3
         seed[ 2] = 1;
         seed[ 3] = Z_ADJ[(thread >> 4) & 63];     // 2^6
         seed[ 4] = Z_NPL[(thread >> 10) & 31];    // 2^5
         seed[ 5] = 1;
         seed[ 6] = Z_INGINF[(thread >> 15) & 31]; // 2^5
      } else
      if(thread < 30347264) { /* Total Permutations, this frame: 1,048,576 ( 1 << 20 ) */
         seed[ 0] = Z_PREP[(thread & 3)];          // 2^2
         seed[ 1] = X_MASS[(thread >> 2) & 3];     // 2^2
         seed[ 2] = 1;
         seed[ 3] = Z_ADJ[(thread >> 5) & 63];     // 2^6
         seed[ 4] = Z_NPL[(thread >> 11) & 31];    // 2^5
         seed[ 5] = 1;
         seed[ 6] = Z_INGINF[(thread >> 16) & 31]; // 2^5
      } else
      if(thread < 30871552) { /* Total Permutations, this frame: 524,288 ( 1 << 19 ) */
         seed[ 0] = Y_PREP[(thread & 1)];          // 2^1
         seed[ 1] = X_MASS[(thread >> 1) & 3];     // 2^2
         seed[ 2] = 1;
         seed[ 3] = Z_ADJ[(thread >> 4) & 63];     // 2^6
         seed[ 4] = Z_NPL[(thread >> 10) & 31];    // 2^5
         seed[ 5] = 1;
         seed[ 6] = Z_INGINF[(thread >> 15) & 31]; // 2^5
      } else
      /* END Frame 6 */
      /* Frame 7 -> Split 9 ways */
      if(thread < 35065856) { /* Total Permutations, this frame: 4,194,304 ( 1 << 22 ) */
         seed[ 0] = Z_TIME[(thread & 15)];         // 2^4
         seed[ 1] = Z_AMB[(thread >> 4) & 15];     // 2^4
         seed[ 2] = 1;
         seed[ 3] = Z_ADJ[(thread >> 8) & 63];     // 2^6
         seed[ 4] = Z_MASS[(thread >> 14) & 15];   // 2^4
         seed[ 5] = 1;
         seed[ 6] = Z_ING[(thread >> 18) & 15];    // 2^4
      } else
      if(thread < 37163088) { /* Total Permutations, this frame: 2,097,152 ( 1 << 21 ) */
         seed[ 0] = Z_TIME[(thread & 15)];         // 2^4
         seed[ 1] = Z_AMB[(thread >> 4) & 15];     // 2^4
         seed[ 2] = 1;
         seed[ 3] = Z_ADJ[(thread >> 8) & 63];     // 2^6
         seed[ 4] = Y_MASS[(thread >> 14) & 7];    // 2^3
         seed[ 5] = 1;
         seed[ 6] = Z_ING[(thread >> 17) & 15];    // 2^4
      } else
      if(thread < 38211584) { /* Total Permutations, this frame: 1,048,576 ( 1 << 20 ) */
         seed[ 0] = Z_TIME[(thread & 15)];         // 2^4
         seed[ 1] = Z_AMB[(thread >> 4) & 15];     // 2^4
         seed[ 2] = 1;
         seed[ 3] = Z_ADJ[(thread >> 8) & 63];     // 2^6
         seed[ 4] = X_MASS[(thread >> 14) & 3];    // 2^2
         seed[ 5] = 1;
         seed[ 6] = Z_ING[(thread >> 16) & 15];    // 2^4
      } else
      if(thread < 40308736) { /* Total Permutations, this frame: 2,097,152 ( 1 << 21 ) */
         seed[ 0] = Z_TIME[(thread & 15)];         // 2^4
         seed[ 1] = Z_AMB[(thread >> 4) & 15];     // 2^4
         seed[ 2] = 1;
         seed[ 3] = Z_ADJ[(thread >> 8) & 63];     // 2^6
         seed[ 4] = Z_MASS[(thread >> 14) & 15];   // 2^4
         seed[ 5] = 1;
         seed[ 6] = Y_ING[(thread >> 18) & 7];     // 2^3
      } else
      if(thread < 41357312) { /* Total Permutations, this frame: 1,048,576 ( 1 << 20 ) */
         seed[ 0] = Z_TIME[(thread & 15)];         // 2^4
         seed[ 1] = Z_AMB[(thread >> 4) & 15];     // 2^4
         seed[ 2] = 1;
         seed[ 3] = Z_ADJ[(thread >> 8) & 63];     // 2^6
         seed[ 4] = Y_MASS[(thread >> 14) & 7];    // 2^3
         seed[ 5] = 1;
         seed[ 6] = Y_ING[(thread >> 17) & 7];     // 2^3
      } else
      if(thread < 41881600) { /* Total Permutations, this frame: 524,288 ( 1 << 19 ) */
         seed[ 0] = Z_TIME[(thread & 15)];         // 2^4
         seed[ 1] = Z_AMB[(thread >> 4) & 15];     // 2^4
         seed[ 2] = 1;
         seed[ 3] = Z_ADJ[(thread >> 8) & 63];     // 2^6
         seed[ 4] = X_MASS[(thread >> 14) & 3];    // 2^2
         seed[ 5] = 1;
         seed[ 6] = Y_ING[(thread >> 16) & 7];     // 2^3
      } else
      if(thread < 42405888) { /* Total Permutations, this frame: 524,288 ( 1 << 19  ) */
         seed[ 0] = Z_TIME[(thread & 15)];         // 2^4
         seed[ 1] = Z_AMB[(thread >> 4) & 15];     // 2^4
         seed[ 2] = 1;
         seed[ 3] = Z_ADJ[(thread >> 8) & 63];     // 2^6
         seed[ 4] = Z_MASS[(thread >> 14) & 15];   // 2^4
         seed[ 5] = 1;
         seed[ 6] = X_ING[(thread >> 18) & 1];     // 2^1
      } else
      if(thread < 42668032) { /* Total Permutations, this frame: 262,144 ( 1 << 18 ) */
         seed[ 0] = Z_TIME[(thread & 15)];         // 2^4
         seed[ 1] = Z_AMB[(thread >> 4) & 15];     // 2^4
         seed[ 2] = 1;
         seed[ 3] = Z_ADJ[(thread >> 8) & 63];     // 2^6
         seed[ 4] = Y_MASS[(thread >> 14) & 7];    // 2^3
         seed[ 5] = 1;
         seed[ 6] = X_ING[(thread >> 17) & 1];     // 2^1
      } else
      if(thread < 42799104) { /* Total Permutations, this frame: 131,072 ( 1 << 17 ) */
         seed[ 0] = Z_TIME[(thread & 15)];         // 2^4
         seed[ 1] = Z_AMB[(thread >> 4) & 15];     // 2^4
         seed[ 2] = 1;
         seed[ 3] = Z_ADJ[(thread >> 8) & 63];     // 2^6
         seed[ 4] = X_MASS[(thread >> 14) & 3];    // 2^2
         seed[ 5] = 1;
         seed[ 6] = X_ING[(thread >> 16) & 1];     // 2^1
      } else
      /* END Frame 7 */
      /* Frame 8 -> Split 2 ways */
      if(thread < 311234560) { /* Total Permutations, this frame: 268,435,456 ( 1 << 28 ) */
         seed[ 0] = Z_TIME[(thread & 15)];         // 2^4
         seed[ 1] = Z_AMB[(thread >> 4) & 15];     // 2^4
         seed[ 2] = 1;
         seed[ 3] = Z_PREP[(thread >> 8) & 3];     // 2^2
         seed[ 4] = 5;
         seed[ 5] = Z_ADJ[(thread >> 10) & 63];    // 2^6
         seed[ 6] = Z_NS[(thread >> 16) & 63];     // 2^6
         seed[ 7] = 3;
         seed[ 8] = 1;
         seed[ 9] = Z_INGADJ[(thread >> 22) & 63]; // 2^6
      } else
      if(thread < 445452288) { /* Total Permutations, this frame: 134,217,728 ( 1 << 27 ) */
         seed[ 0] = Z_TIME[(thread & 15)];         // 2^4
         seed[ 1] = Z_AMB[(thread >> 4) & 15];     // 2^4
         seed[ 2] = 1;
         seed[ 3] = Y_PREP[(thread >> 8) & 1];     // 2^1
         seed[ 4] = 5;
         seed[ 5] = Z_ADJ[(thread >> 9) & 63];     // 2^6
         seed[ 6] = Z_NS[(thread >> 15) & 63];     // 2^6
         seed[ 7] = 3;
         seed[ 8] = 1;
         seed[ 9] = Z_INGADJ[(thread >> 21) & 63]; // 2^6
      }

/* Below Two Frames are Valid, But Require 64-Bit Math: if extra entropy req'd.
   if( < thread <= ) { /* Total Permutations, this frame: 549,755,813,888
	seed[ 0] = Z_ING[(thread & 31)]; 
	seed[ 1] = Z_PREP[(thread << 5) & 7];
	seed[ 2] = Z_TIME[(thread << 8) & 15]; 
	seed[ 3] = Z_MASS[(thread << 12) & 31]; 
	seed[ 4] = 1;
        seed[ 5] = Z_MASS[(thread << 17) & 31]; 
	seed[ 6] = Z_ING[(thread << 22) & 31];  
	seed[ 7] = 3; 
	seed[ 8] = 1;
        seed[ 9] = 5; 
	seed[10] = Z_ADJ[(thread << 27) & 63];
	seed[11] = Z_NS[(thread << 33) & 63];
   }
   if( < thread <= ) { /* Total Permutations, this frame: 4,398,046,511,104
	seed[ 0] = Z_ING[(thread & 31)]; 
	seed[ 1] = Z_PREP[(thread << 5) & 7]; 
	seed[ 2] = 5; 
	seed[ 3] = Z_ADJ[(thread << 8) & 63]; 
	seed[ 4] = Z_NS[(thread << 14) & 63]; 
	seed[ 5] = 1;
        seed[ 6] = Z_MASS[(thread << 19) & 31]; 
	seed[ 7] = Z_ING[(thread << 24) & 31];  
	seed[ 8] = 3; 
	seed[ 9] = 1;
        seed[10] = 5; 
	seed[11] = Z_ADJ[(thread << 30) & 63]; 
	seed[12] = Z_NS[(thread << 36) & 63];
   }
End 64-bit Frames */
     
     /* store full nonce */
     #pragma unroll
     for (i = 0; i < 4; i++)
       nonce[i] = c_input32[i + 7];
     
     #pragma unroll
     for (i = 0; i < 4; i++)
       nonce[i+4] = seed[i];
     
     
     /*********************************************************/
     /* Hash 124 bytes of Block Trailer, including both seeds */
     /* Get the wizard to draw you a map to the princess      */
     
     /* prepare the midstate */
     #pragma unroll
     for (i = 0; i < 8; i += 2)
       AS_UINT2(&bt_hash[i]) = AS_UINT2(&c_midstate256[i]);
     
     /* prepare the remaining mroot (28 bytes) for hashing */
     #pragma unroll
     for (i = 0; i < 7; i++)
       input[i] = c_input32[i];
     
     /* swab and prepare the nonce (16 bytes) for hashing */
     #pragma unroll
     for (i = 0; i < 8; i++)
       input[i+7] = cuda_swab32(nonce[i]);

     /* end of data */
     input[15] = 0x80000000;
     
     /* perform sha256 transform */
     sha256_round(input, bt_hash, c_K);

     /* zero fill remaining input */
     #pragma unroll
     for (i = 0; i < 15; i++)
       input[i] = 0;
     
     /* 0x1e0 bits of data processed */
     input[15] = 0x1e0;
     
     /* finalize sha256 transform */
     sha256_round(input, bt_hash, c_K);
     
     
     /*****************************************************/
     /* Determine the final tile based on selected nonce  */
     /* Follow the wizard's map to find the princess!     */
     
     /* determine first tile index */
     for(i = 0; i < HASHLEN; i++) {
       if(i == 0) sm = bt_hash[i];
       else sm *= bt_hash[i];
     }
     sm %= MAP;

     /* get cached tile, or generate one if it doesn't exist */
     if(g_cache[sm])
       tilep = &g_map[sm * TILE_LENGTH_32BIT];
     else
       cuda_gen_tile(sm, c_phash, g_map, g_cache);
     
     /* make <JUMP> tile jumps to find the final tile */
	 for(j = 0; j < JUMP; j++) {
       /* determine next tile index */
       sm = cuda_next_index(sm, tilep, nonce);
       
       /* get cached tile, or generate one if it doesn't exist */
       if(g_cache[sm])
         tilep = &g_map[sm * TILE_LENGTH_32BIT];
       else
         cuda_gen_tile(sm, c_phash, g_map, g_cache);
	 }


     /****************************************************************/
     /* Check the hash of the final tile produces the desired result */
     /* Search the castle for the princess!                          */
     
     sha256_tile(bt_hash, tilep, fhash);
     
     for (bp = fhash, n = c_difficulty >> 5; n; n--)
       if (*bp++ != 0) return; /* Our princess is in another castle ! */
     if (__clz(*bp) >= (c_difficulty & 31)) {
       /* PRINCESS FOUND! */
       *g_found = 1;
       #pragma unroll
       for (i = 0; i < 16; i++)
         g_seed[i] = seed[i];
     }
   }
}



extern "C" {

typedef struct __peach_cuda_ctx {
    byte curr_seed[16], next_seed[16];
    char cp[256], next_cp[256];
    int *found, *d_found;
    uint8_t *seed, *d_seed, *d_cache;
    uint8_t *midstate, *input, *d_map;
    cudaStream_t stream;
} PeachCudaCTX;

PeachCudaCTX ctx[63];    /* Max 63 GPUs Supported */
uint32_t threads = 65536;
dim3 grid(256);
dim3 block(256);
char nullcp = '\0';
byte *diff;
byte *phash;
byte bnum[8] = {0};
byte initGPU = 0;
int nGPU = 0;

int init_cuda_peach(byte difficulty, byte *prevhash, byte *blocknumber) {
    /* Obtain and check system GPU count */
    cudaGetDeviceCount(&nGPU);
    if(nGPU<1 || nGPU>63) return nGPU;
    /* Allocate pinned host memory */
    cudaMallocHost(&diff, 1);
    cudaMallocHost(&phash, 32);
    /* Copy immediate block data to pinned memory */
    memcpy(diff, &difficulty, 1);
    memcpy(phash, prevhash, 32);

    int i = 0;
    for ( ; i<nGPU; i++) {
        cudaSetDevice(i);
        /* Create Stream */
        cudaStreamCreate(&ctx[i].stream);
        /* Allocate device memory */
        cudaMalloc(&ctx[i].d_found, 4);
        cudaMalloc(&ctx[i].d_seed, 16);
        /* Allocate associated device-host memory */
        cudaMallocHost(&ctx[i].found, 4);
        cudaMallocHost(&ctx[i].seed, 16);
        cudaMallocHost(&ctx[i].midstate, 32);
        cudaMallocHost(&ctx[i].input, 44);
        /* Copy immediate block data to device memory */
        cudaMemcpyToSymbolAsync(c_difficulty, diff, 1, 0,
                                cudaMemcpyHostToDevice, ctx[i].stream);
        cudaMemcpyToSymbolAsync(c_phash, phash, 32, 0,
                                cudaMemcpyHostToDevice, ctx[i].stream);
        /* Set remaining device memory */
        cudaMemsetAsync(ctx[i].d_found, 0, 4, ctx[i].stream);
        cudaMemsetAsync(ctx[i].d_seed, 0, 16, ctx[i].stream);
        /* Set initial round variables */
        ctx[i].next_cp[0] = nullcp;
        /* If first init, setup map and cache */
        if(initGPU == 0) {
            cudaMalloc(&ctx[i].d_map, MAP_LENGTH);
            cudaMalloc(&ctx[i].d_cache, MAP);
            initGPU = 1;
        }
        /* Wipe cache if new block */
        if(initGPU && memcmp(blocknumber, bnum, 8) != 0) {
            cudaMemsetAsync(ctx[i].d_cache, 0, MAP, ctx[i].stream);
            memcpy(bnum, blocknumber, 8);
        }
    }

    return nGPU;
}

void free_cuda_peach() {
    /* Free pinned host memory */
    cudaFreeHost(diff);
    cudaFreeHost(phash);

    int i = 0;
    for ( ; i<nGPU; i++) {
        cudaSetDevice(i);
        /* Destroy Stream */
        cudaStreamDestroy(ctx[i].stream);
        /* Free device memory */
        cudaFree(ctx[i].d_found);
        cudaFree(ctx[i].d_seed);
        /* Free associated device-host memory */
        cudaFreeHost(ctx[i].found);
        cudaFreeHost(ctx[i].seed);
        cudaFreeHost(ctx[i].midstate);
        cudaFreeHost(ctx[i].input);
    }
}

extern byte *trigg_gen(byte *in);
extern char *trigg_expand2(byte *in, byte *out);

__host__ char *cuda_peach(byte *bt, char *haiku, uint32_t *hps, byte *runflag)
{
    int i;
    uint64_t nHaiku;
    time_t start = time(NULL);
    for(haiku = NULL; *runflag && haiku == NULL; ) {
        for (i=0; i<nGPU; i++) {
            /* Prepare next seed for GPU... */
            if(ctx[i].next_cp[0] == nullcp) {
                /* ... generate first GPU seed (and expand as Haiku) */
                trigg_gen(ctx[i].next_seed);
                trigg_expand2(ctx[i].next_seed, (byte*)ctx[i].next_cp);

                /* ... and prepare sha256 midstate for next round */
                SHA256_CTX sha256;
                sha256_init(&sha256);
                sha256_update(&sha256, bt, 64);
                memcpy(ctx[i].midstate, sha256.state, 32);
                memcpy(ctx[i].input, &bt[64], 44);
            }
            /* Check if GPU has finished */
            CudaCheckError();
            cudaSetDevice(i);
            if(cudaStreamQuery(ctx[i].stream) == cudaSuccess) {
                cudaMemcpy(ctx[i].found, ctx[i].d_found, 4, cudaMemcpyDeviceToHost);
                if(*ctx[i].found==1) { /* SOLVED A BLOCK! */
                    cudaMemcpy(ctx[i].seed, ctx[i].d_seed, 16, cudaMemcpyDeviceToHost);
                    memcpy(bt + 92, ctx[i].curr_seed, 16);
                    memcpy(bt + 92 + 16, ctx[i].seed, 16);
                    haiku = ctx[i].cp;
                    break;
                }
                /* Send new GPU round Data */
                cudaMemcpyToSymbolAsync(c_midstate256, ctx[i].midstate, 32, 0,
                                        cudaMemcpyHostToDevice, ctx[i].stream);
                cudaMemcpyToSymbolAsync(c_input32, ctx[i].input, 44, 0,
                                        cudaMemcpyHostToDevice, ctx[i].stream);
                /* Start GPU round */
                cuda_find_peach<<<grid, block, 0, ctx[i].stream>>>(threads,
                ctx[i].d_map, ctx[i].d_cache, ctx[i].d_found, ctx[i].d_seed);

                /* Add to haiku count */
                nHaiku += threads;

                /* Store round vars aside for checks next loop */
                memcpy(ctx[i].curr_seed,ctx[i].next_seed,16);
                strcpy(ctx[i].cp,ctx[i].next_cp);
                ctx[i].next_cp[0] = nullcp;
            } else continue;  /* Waiting on GPU ... */
        }
    }
    
    start = time(NULL) - start;
    if(start == 0) start = 1;
    *hps = (uint32_t)(nHaiku / (uint64_t)start);
         
    return haiku;
}


}
