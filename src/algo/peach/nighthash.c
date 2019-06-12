/*
 * nighthash.c  FPGA-Confuddling Hash Algo Selector
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

#include "../../crypto/blake2/blake2b.c"
#include "../../crypto/sha1/sha1.c"
#include "../../crypto/sha256.c"
#include "../../crypto/keccak/keccak.c"
#include "../../crypto/md5/md5.c"

void night_hash(byte *out, byte *in, uint32_t inlen)
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
         sha256_final(&sha256, out);

         /* Keccak
         * CUDA impl:
         *       https://github.com/cbuchner1/CudaMiner/blob/master/keccak.cu
         *    http://www.cayrel.net/?Keccak-implementation-on-GPU
         *    https://sites.google.com/site/keccaktreegpu/
         */
         //sha3_HashBuffer(256, SHA3_FLAGS_KECCAK, in, inlen, out, HASHLEN);
      }
         break;
      case 4:
      {
         /* SHA3 256
         * CUDA impl:
         *        https://github.com/smoes/SHA1-CUDA-bruteforce
         */
         keccak_ctx_t sha3;
         keccack_sha3_init(&sha3, 256);
         keccack_update(&sha3, in, inlen);
         keccack_final(&sha3, out);
      }
         break;
      case 5:
      {
         /* SHA3 256
         *
         */
         keccak_ctx_t sha3;
         keccack_init(&sha3, 256);
         keccack_update(&sha3, in, inlen);
         keccack_final(&sha3, out);

      }
         break;

      case 6:
      {
         /* MD4
         * TODO: implement
         */
      }
         break;
      case 7:
      {
         /* MD5
         * CUDA impl:
         *        https://github.com/xpn/CUDA-MD5-Crack
         */
         MD5_CTX ctx;
         md5_init(&ctx);
         md5_update(&ctx, in, inlen);
         md5_final(&ctx, out);

      }
         break;
      default:
         error("Peach night hash OP is outside the expected range (%i)\n", op);
         assert(0);
         break;
   }
}
