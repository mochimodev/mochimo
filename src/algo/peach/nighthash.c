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
#include "../../crypto/hash/cpu/blake2b.c"
#include "../../crypto/hash/cpu/sha1.c"
#include "../../crypto/hash/cpu/keccak.c"
#include "../../crypto/hash/cpu/md2.c"
#include "../../crypto/hash/cpu/md5.c"

/**
 * Performs data transformation on 32 bit chunks (4 bytes) of data
 * using deterministic floating point operations on IEEE 754
 * compliant machines and devices.
 * @param *data     - pointer to in data (at least 32 bytes)
 * @param len       - length of data
 * @param index     - the current tile
 * @param *op       - pointer to the operator value
 * @param transform - flag indicates to transform the input data */
void fp_operation(uint8_t *data, uint32_t len, uint32_t index, uint32_t *op,
                  uint8_t transform)
{
   uint8_t *temp;
   uint32_t adjustedlen = (len >> 2) << 2; /* Adjust the length to a multiple of 4 */
   int32_t i, j, operand;
   float floatv, floatv1, *floatp;

   /* Work on data 4 bytes at a time */
   for(i = 0; i < adjustedlen; i += 4)
   {
      /* Cast 4 byte piece to float pointer */
      if(transform)
         floatp = (float *) &data[i];
      else {
         floatv1 = *(float *) &data[i];
         floatp = &floatv1;
      }

      /* 4 byte separation order depends on initial byte:
       * #1) *op = data... determine floating point operation type
       * #2) operand = ... determine the value of the operand
       * #3) if(data[i ... determine the sign of the operand
       *                   ^must always be performed after #2) */
      switch(data[i] & 7)
      {
         case 0:
            *op += (uint32_t) data[i + 1];
            operand = (int32_t) data[ (data[i + 2] & 31) ];
            if(data[i + 3] & 1) operand ^= 0x80000000;
            break;
         case 1:
            operand = (int32_t) data[ (data[i + 1] & 31) ];
            if(data[i + 2] & 1) operand ^= 0x80000000;
            *op += (uint32_t) data[i + 3];
            break;
         case 2:
            *op += (uint32_t) data[i];
            operand = (int32_t) data[ (data[i + 2] & 31) ];
            if(data[i + 3] & 1) operand ^= 0x80000000;
            break;
         case 3:
            *op += (uint32_t) data[i];
            operand = (int32_t) data[ (data[i + 1] & 31) ];
            if(data[i + 2] & 1) operand ^= 0x80000000;
            break;
         case 4:
            operand = (int32_t) data[ (data[i] & 31) ];
            if(data[i + 1] & 1) operand ^= 0x80000000;
            *op += (uint32_t) data[i + 3];
            break;
         case 5:
            operand = (int32_t) data[ (data[i] & 31) ];
            if(data[i + 1] & 1) operand ^= 0x80000000;
            *op += (uint32_t) data[i + 2];
            break;
         case 6:
            *op += (uint32_t) data[i + 1];
            operand = (int32_t) data[ (data[i + 1] & 31) ];
            if(data[i + 3] & 1) operand ^= 0x80000000;
            break;
         case 7:
            operand = (int32_t) data[ (data[i + 1] & 31) ];
            *op += (uint32_t) data[i + 2];
            if(data[i + 3] & 1) operand ^= 0x80000000;
            break;
      } /* end switch(data[j] & 31... */

      /* Cast operand to float */
      floatv = (float) operand;

      /* Replace pre-operation NaN with index */
      if(isnan(*floatp)) *floatp = (float) index;

      /* Perform predetermined floating point operation */
      switch(*op & 3) {
         case 0:
            *floatp += floatv;
            break;
         case 1:
            *floatp -= floatv;
            break;
         case 2:
            *floatp *= floatv;
            break;
         case 3:
            *floatp /= floatv;
            break;
      }

      /* Replace post-operation NaN with index */
      if(isnan(*floatp)) *floatp = (float) index;

      /* Add result of floating point operation to op */
      temp = (uint8_t *) floatp;
      for(j = 0; j < 4; j++) {
         *op += (uint32_t) temp[j];
      }
   } /* end for(*op = 0... */
}

/**
 * Performs bit/byte operations on all data (len) of data using
 * random bit/byte transform operations, for increased complexity
 * @param *data     - pointer to in data
 * @param len       - length of data
 * @param *op       - pointer to the operator value */
void bitbyte_transform(uint8_t *data, uint32_t len, uint32_t *op)
{
   int32_t i, z;
   uint32_t len2;
   uint8_t temp, _104, _72;

   /* Perform <TILE_TRANSFORMS> number of bit/byte manipulations */
   for(i = 0, _104 = 104, _72 = 72, len2 = len/2; i < TILE_TRANSFORMS; i++)
   {
      /* Determine operation to use this iteration */
      *op += (uint32_t) data[i & 31];

      /* Perform random operation */
      switch(*op & 7) {
         case 0: /* Swap the first and last bit in each byte. */
            for(z = 0; z < len; z++)
               data[z] ^= 0x81;
            break;
         case 1: /* Swap bytes */
            for(z = 0; z < len2; z++) {
               temp = data[z];
               data[z] = data[z + len2];
               data[z + len2] = temp;
            }
            break;
         case 2: /* Complement One, all bytes */
            for(z = 1; z < len; z++)
               data[z] = ~data[z];
            break;
         case 3: /* Alternate +1 and -1 on all bytes */
            for(z = 0; z < len; z++)
               data[z] += ((z & 1) == 0) ? 1 : -1;
            break;
         case 4: /* Alternate +i and -i on all bytes */
            for(z = 0; z < len; z++)
               data[z] += ((z & 1) == 0) ? -i : i;
            break;
         case 5: /* Replace every occurrence of _104 with _72 */ 
            for(z = 0; z < len; z++)
               if(data[z] == _104) data[z] = _72;
            break;
         case 6: /* If byte a is > byte b, swap them. */
            for(z = 0; z < len2; z++) {
               if(data[z] > data[z + len2]) {
                  temp = data[z];
                  data[z] = data[z + len2];
                  data[z + len2] = temp;
               }
            }
            break;
         case 7: /* XOR all bytes */
            for(z = 1; z < HASHLEN; z++)
               data[z] ^= data[z - 1];
            break;
      } /* end switch(... */
   } /* end for(i = 0... */ 
}

/**
 * Nighthash function optimised for generating a tile */
int nighthash_transform_init(nighthash_ctx_t *ctx, byte *algo_type_seed,
                             uint32_t algo_type_seed_length, uint32_t index,
                             uint32_t digestbitlen)
{
   uint32_t algo_type;
   algo_type = 0;
   
   /* Perform floating point operations to transform input data
    * and determine algo type */
   fp_operation(algo_type_seed, algo_type_seed_length, index, &algo_type, 1);
   
   /* Perform bit/byte transform operations to transform input data
    * and determine algo type */
   bitbyte_transform(algo_type_seed, algo_type_seed_length, &algo_type);

   return nighthash_init(ctx, algo_type & 7, digestbitlen);
}

/**
 * Nighthash function optimised for determining the next index */
int nighthash_seed_init(nighthash_ctx_t *ctx, byte *algo_type_seed,
                        uint32_t algo_type_seed_length, uint32_t index,
                        uint32_t digestbitlen)
{
   uint32_t algo_type;
   algo_type = 0;

   /* Perform floating point operations to determine algo type
    * without transforming input data */
   fp_operation(algo_type_seed, algo_type_seed_length, index, &algo_type, 0);

   return nighthash_init(ctx, algo_type & 7, digestbitlen);
}

int nighthash_init(nighthash_ctx_t *ctx, uint32_t algo_type, uint32_t digestbitlen)
{
   byte key32[32], key64[64];
   if(digestbitlen != 256 && digestbitlen != 512)
      return -1;

   memset(ctx, 0, sizeof(nighthash_ctx_t));

   switch(algo_type)
   {
      case 0:
         memset(key32, algo_type, 32);
         blake2b_init(&(ctx->blake2b), key32, 32, digestbitlen);
         break;
      case 1:
         memset(key64, algo_type, 64);
         blake2b_init(&(ctx->blake2b), key64, 64, digestbitlen);
         break;
      case 2:
         sha1_init(&(ctx->sha1));
         break;
      case 3:
         sha256_init(&(ctx->sha256));
         break;
      case 4:
         keccak_sha3_init(&(ctx->sha3), digestbitlen);
         break;
      case 5:
         keccak_init(&(ctx->keccak), digestbitlen);
         break;
      case 6:
         md2_init(&(ctx->md2));
         break;
      case 7:
         md5_init(&(ctx->md5));
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
