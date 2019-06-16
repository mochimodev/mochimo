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

#include "../../crypto/hash/cuda/blake2b.cu"
#include "../../crypto/hash/cuda/keccak.cu"
#include "../../crypto/hash/cuda/sha256.cu"
#include "../../crypto/hash/cuda/sha1.cu"
#include "../../crypto/hash/cuda/md5.cu"
#include "../../crypto/hash/cuda/md2.cu"

typedef struct {

   uint32_t digestlen;
   uint32_t algo_type;

   CUDA_BLAKE2B_CTX blake2b;
   CUDA_SHA1_CTX sha1;
   CUDA_SHA256_CTX sha256;
   CUDA_KECCAK_CTX sha3;
   CUDA_KECCAK_CTX keccak;
   CUDA_MD2_CTX md2;
   CUDA_MD5_CTX md5;

} CUDA_NIGHTHASH_CTX;

/**
 * Performs data transformation on 32 bit chunks (4 bytes) of data
 * using deterministic floating point operations on IEEE 754
 * compliant machines and devices.
 * @param *data     - pointer to in data (at least 32 bytes)
 * @param len       - length of data
 * @param index     - the current tile
 * @param *op       - pointer to the operator value
 * @param transform - flag indicates to transform the input data */
__device__ void cuda_fp_operation(uint8_t *data, uint32_t len, uint32_t index,
                                  uint32_t *op, uint8_t transform)
{
   uint8_t *temp;
   int32_t i, j, operand;
   float floatv, floatv1, *floatp;

   /* Work on data 4 bytes at a time */
   for(i = 0, len = (len >> 2) << 2; i < len; i += 4)
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
__device__ void cuda_bitbyte_transform(uint8_t *data, uint32_t len, uint32_t *op)
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

__device__ void cuda_nighthash_init(CUDA_NIGHTHASH_CTX *ctx, byte *algo_type_seed,
                                    uint32_t algo_type_seed_length, uint32_t index,
                                    uint8_t transform)
{
   uint32_t algo_type;
   byte key32[32], key64[64];
   algo_type = 0;
   
   /* Perform floating point operations to transform (if transform byte is set)
    * input data and determine algo type */
   cuda_fp_operation(algo_type_seed, algo_type_seed_length, index, &algo_type, transform);
   
   /* Perform bit/byte transform operations to transform (if transform byte is set)
    * input data and determine algo type */
   if(transform)
      cuda_bitbyte_transform(algo_type_seed, algo_type_seed_length, &algo_type);
   
   /* Clear nighthash context */
   memset(ctx, 0, sizeof(CUDA_NIGHTHASH_CTX));

   ctx->digestlen = 32;
   ctx->algo_type = algo_type & 7;
   
   switch(ctx->algo_type)
   {
      case 0:
         memset(key32, ctx->algo_type, 32);
         cuda_blake2b_init(&(ctx->blake2b), key32, 32, 256);
         break;
      case 1:
         memset(key64, ctx->algo_type, 64);
         cuda_blake2b_init(&(ctx->blake2b), key64, 64, 256);
         break;
      case 2:
         cuda_sha1_init(&(ctx->sha1));
         break;
      case 3:
         cuda_sha256_init(&(ctx->sha256));
         break;
      case 4:
         cuda_keccak_sha3_init(&(ctx->sha3), 256);
         break;
      case 5:
         cuda_keccak_init(&(ctx->keccak), 256);
         break;
      case 6:
         cuda_md2_init(&(ctx->md2));
         break;
      case 7:
         cuda_md5_init(&(ctx->md5));
         break;
   } /* end switch(algo_type)... */
}

__device__ void cuda_nighthash_update(CUDA_NIGHTHASH_CTX *ctx, byte *in, uint32_t inlen)
{
   switch(ctx->algo_type)
   {
      case 0:
         cuda_blake2b_update(&(ctx->blake2b), in, inlen);
         break;
      case 1:
         cuda_blake2b_update(&(ctx->blake2b), in, inlen);
         break;
      case 2:
         cuda_sha1_update(&(ctx->sha1), in, inlen);
         break;
      case 3:
         cuda_sha256_update(&(ctx->sha256), in, inlen);
         break;
      case 4:
         cuda_keccak_update(&(ctx->sha3), in, inlen);
         break;
      case 5:
         cuda_keccak_update(&(ctx->keccak), in, inlen);
         break;
      case 6:
         cuda_md2_update(&(ctx->md2), in, inlen);
         break;
      case 7:
         cuda_md5_update(&(ctx->md5), in, inlen);
         break;
   } /* end switch(ctx->... */
}

__device__ void cuda_nighthash_final(CUDA_NIGHTHASH_CTX *ctx, byte *out)
{
   switch(ctx->algo_type)
   {
      case 0:
         cuda_blake2b_final(&(ctx->blake2b), out);
         break;
      case 1:
         cuda_blake2b_final(&(ctx->blake2b), out);
         break;
      case 2:
         cuda_sha1_final(&(ctx->sha1), out);
         memset(out + 20, 0, 12);
         break;
      case 3:
         cuda_sha256_final(&(ctx->sha256), out);
         break;
      case 4:
         cuda_keccak_final(&(ctx->sha3), out);
         break;
      case 5:
         cuda_keccak_final(&(ctx->keccak), out);
         break;
      case 6:
         cuda_md2_final(&(ctx->md2), out);
         memset(out + 16, 0, 16);
         break;
      case 7:
         cuda_md5_final(&(ctx->md5), out);
         memset(out + 16, 0, 16);
         break;
   } /* end switch(ctx->... */
}