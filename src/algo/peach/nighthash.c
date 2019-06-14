/*
 * nighthash.c  FPGA-Confuddling Hash Algo
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 12 June 2019
 * Revision: 1
 *
 * This file is subject to the license as found in LICENSE.PDF
 *
 */

#include "nighthash.h"
#include "../../crypto/hash/cpu/keccak.c"
#include "../../crypto/hash/cpu/blake2b.c"
#include "../../crypto/hash/cpu/md2.c"
#include "../../crypto/hash/cpu/md5.c"

int nighthash_seed_init(nighthash_ctx_t *ctx, byte *algo_type_seed, uint32_t algo_type_seed_length, uint32_t digestbitlen)
{
   uint32_t algo_type;
   algo_type = 0;

   /* TODO: Replace with floating point arithmetic */
  for(int i = 0; i < algo_type_seed_length; i++)
     algo_type += algo_type_seed[i];

  return nighthash_init(ctx, algo_type & 7, digestbitlen);
}

int nighthash_init(nighthash_ctx_t *ctx, uint32_t algo_type, uint32_t digestbitlen)
{
   if(digestbitlen != 256 && digestbitlen != 512)
      return -1;

   memset(ctx, 0, sizeof(nighthash_ctx_t));

   switch(algo_type)
   {
      case 0:
      {
         byte key[32];
         memset(key, algo_type, 32);
         blake2b_init(&(ctx->blake2b), key, 32, digestbitlen);
      }
         break;
      case 1:
      {
         byte key[64];
         memset(key, algo_type, 64);
         blake2b_init(&(ctx->blake2b), key, 32, digestbitlen);
      }
         break;
      case 2:
      {
         sha1_init(&(ctx->sha1));
      }
         break;
      case 3:
      {
         sha256_init(&(ctx->sha256));
      }
         break;
      case 4:
      {
         keccak_sha3_init(&(ctx->sha3), digestbitlen);
      }
         break;
      case 5:
      {
         keccak_init(&(ctx->keccak), digestbitlen);
      }
         break;
      case 6:
      {
         md2_init(&(ctx->md2));
      }
         break;
      case 7:
      {
         md5_init(&(ctx->md5));
      }
         break;
      default:
         error("Fatal: Invalid night hash algo type (%i)\n", algo_type);
         return -1;
   }

   ctx->digestlen = digestbitlen >> 3;
   ctx->algo_type = algo_type;
   return 0;
}

int nighthash_update(nighthash_ctx_t *ctx, byte *in, uint32_t inlen)
{
   switch(ctx->algo_type)
   {
      case 0:
         blake2b_update(&(ctx->blake2b), in, inlen);
         break;
      case 1:
         blake2b_update(&(ctx->blake2b), in, inlen);
         break;
      case 2:
         sha1_update(&(ctx->sha1), in, inlen);
         break;
      case 3:
         sha256_update(&(ctx->sha256), in, inlen);
         break;
      case 4:
         keccak_update(&(ctx->sha3), in, inlen);
         break;
      case 5:
         keccak_update(&(ctx->keccak), in, inlen);
         break;
      case 6:
         md2_update(&(ctx->md2), in, inlen);
         break;
      case 7:
         md5_update(&(ctx->md5), in, inlen);
         break;
      default:
         error("Fatal: Invalid night hash algo type (%i)\n", ctx->algo_type);
         return -1;
   }
   return 0;
}

int nighthash_final(nighthash_ctx_t *ctx, byte *out)
{
   switch(ctx->algo_type)
   {
      case 0:
         blake2b_final(&(ctx->blake2b), out);
         break;
      case 1:
         blake2b_final(&(ctx->blake2b), out);
         break;
      case 2:
         sha1_final(&(ctx->sha1), out);
         memset(out + 20, 0, ctx->digestlen - 20);
         break;
      case 3:
         sha256_final(&(ctx->sha256), out);
         if(ctx->digestlen > 32)
            memset(out + 32, 0, ctx->digestlen - 32);
         break;
      case 4:
         keccak_final(&(ctx->sha3), out);
         break;
      case 5:
         keccak_final(&(ctx->keccak), out);
         break;
      case 6:
         md2_final(&(ctx->md2), out);
         memset(out + 16, 0, ctx->digestlen - 16);
         break;
      case 7:
         md5_final(&(ctx->md5), out);
         memset(out + 16, 0, ctx->digestlen - 16);
         break;
      default:
         error("Fatal: Invalid night hash algo type (%i)\n", ctx->algo_type);
         return -1;
   }
   return 0;
}

void nighthashold(byte *out, uint32_t index, byte *in, uint32_t inlen, byte *in2, uint32_t inlen2)
{
   uint32_t op;
   op = 0;

   /* TODO: Replace with floating point arithmetic */
   for(int i = 0; i < inlen; i++)
      op += in[i];

   switch(op & 7)
   {
      case 0:
      {
         /* Blake2b key 32 bytes
         * CUDA impl:
         *          https://github.com/tromp/equihash/blob/master/blake2b.cu
         *          https://github.com/nicehash/nheqminer/blob/master/cuda_tromp/blake2b.cu
         */
         byte key[HASHLEN];
         memset(key, in[inlen - 1], HASHLEN);
         blake2b_ctx_t blake2b;
         blake2b_init(&blake2b, key, HASHLEN, 256);
         blake2b_update(&blake2b, in, inlen);
         if(in2 != NULL) blake2b_update(&blake2b, in2, inlen2);
         blake2b_update(&blake2b, (byte*) &index, sizeof(uint32_t));
         blake2b_final(&blake2b, out);
      }
         break;
      case 1:
      {
         /* Blake2b key 64 bytes
         * CUDA impl:
         *          https://github.com/tromp/equihash/blob/master/blake2b.cu
         *          https://github.com/nicehash/nheqminer/blob/master/cuda_tromp/blake2b.cu
         */
         byte key[64];
         memset(key, in[0], 64);
         blake2b_ctx_t blake2b;
         blake2b_init(&blake2b, key, HASHLEN, 256);
         blake2b_update(&blake2b, in, inlen);
         if(in2 != NULL) blake2b_update(&blake2b, in2, inlen2);
         blake2b_update(&blake2b, (byte*) &index, sizeof(uint32_t));
         blake2b_final(&blake2b, out);
      }
         break;
      case 2:
      {
         /* SHA1
          *
          */
         SHA1_CTX sha1;
         sha1_init(&sha1);
         sha1_update(&sha1, in, inlen);
         if(in2 != NULL) sha1_update(&sha1, in2, inlen2);
         sha1_update(&sha1, (byte*) &index, sizeof(uint32_t));
         sha1_final(&sha1, out);
      }
         break;
      case 3:
      {
         /* SHA256
          *
          */
         SHA256_CTX sha256;
         sha256_init(&sha256);
         sha256_update(&sha256, in, inlen);
         if(in2 != NULL) sha256_update(&sha256, in2, inlen2);
         sha256_update(&sha256, (byte*) &index, sizeof(uint32_t));
         sha256_final(&sha256, out);
      }
         break;
      case 4:
      {
         /* SHA3 256
         * CUDA impl:
         *        https://github.com/smoes/SHA1-CUDA-bruteforce
         */

         keccak_ctx_t sha3;
         keccak_sha3_init(&sha3, 256);
         keccak_update(&sha3, in, inlen);
         if(in2 != NULL) keccak_update(&sha3, in2, inlen2);
         keccak_update(&sha3, (byte*) &index, sizeof(uint32_t));
         keccak_final(&sha3, out);
      }
         break;
      case 5:
      {
         /* Keccak
         * CUDA impl:
         *    https://github.com/cbuchner1/CudaMiner/blob/master/keccak.cu
         *    http://www.cayrel.net/?Keccak-implementation-on-GPU
         *    https://sites.google.com/site/keccaktreegpu/
         */

         keccak_ctx_t keccak;
         keccak_init(&keccak, (uint32_t)256);
         keccak_update(&keccak, in, inlen);
         if(in2 != NULL) keccak_update(&keccak, in2, inlen2);
         keccak_update(&keccak, (byte*) &index, sizeof(uint32_t));
         keccak_final(&keccak, out);
     }
         break;
      case 6:
      {
         /* MD2
         *
         */
         MD2_CTX md2;
         md2_init(&md2);
         md2_update(&md2, in, inlen);
         if(in2 != NULL) md2_update(&md2, in2, inlen2);
         md2_update(&md2, (byte*) &index, sizeof(uint32_t));
         md2_final(&md2, out);
      }
         break;
      case 7:
      {
         /* MD5
         * CUDA impl:
         *        https://github.com/xpn/CUDA-MD5-Crack
         */
         MD5_CTX md5;
         md5_init(&md5);
         md5_update(&md5, in, inlen);
         if(in2 != NULL) md5_update(&md5, in2, inlen2);
         md5_update(&md5, (byte*) &index, sizeof(uint32_t));
         md5_final(&md5, out);
      }
         break;
      default:
         error("Fatal: Peach night hash OP is outside the expected range (%i)\n", op);
         assert(0);
         break;
   }

}
