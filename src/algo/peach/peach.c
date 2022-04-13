/**
 * @private
 * @headerfile peach.h <peach.h>
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_PEACH_C
#define MOCHIMO_PEACH_C


#include <math.h>  /* for isnan() */
#include "peach.h"

/* hashing functions used by Peach's nighthash */
#include "blake2b.h"
#include "md2.h"
#include "md5.h"
#include "sha1.h"
#include "sha256.h"
#include "sha3.h"

/* Define restricted use Peach semaphores */
static SHA256_CTX PeachICTX;
#ifdef ENABLE_CPU_PEACH_CACHE
   static word8 PeachMap[PEACHMAPLEN];      /* 1GiByte! */
   static word8 PeachCache[PEACHCACHELEN];  /* 1MiByte! */
   static word8 PeachCleared[SHA256LEN];    /* clearhash */

#endif

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
 * rounding mode. This should be covered by default on most 21st century
 * hardware, otherwise it may need to be specified at compile-time.
*/
static word32 peach_dflops(void *data, size_t len, word32 index, int txf)
{
   float *flp, temp, flv;
   word8 *bp, shift;
   int32 operand;
   word32 op;
   unsigned i;

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
      op += bp[((WORD32_C(0x26C34) >> shift) & 3)];
      /* ... 2) determine the value of the operand */
      operand = bp[((WORD32_C(0x14198) >> shift) & 3)];
      /* ... 3) determine the upper most bit of the operand
       *        NOTE: must be performed AFTER operand allocation */
      if (bp[((WORD32_C(0x3D6EC) >> shift) & 3)] & 1) {
         operand ^= WORD32_C(0x80000000);
      }
      /* interpret operand as SIGNED integer and cast to float */
      flv = (float) operand;
      /* Replace pre-operation NaN with index */
      if (isnan(*flp)) *flp = (float) index;
      /* Perform predetermined floating point operation */
      switch (op & 3) {
         case 3:  *flp /= flv; break;
         case 2:  *flp *= flv; break;
         case 1:  *flp -= flv; break;
         default: *flp += flv; break;
      }
      /* Replace post-operation NaN with index */
      if (isnan(*flp)) *flp = (float) index;
      /* Add result of the operation to `op` as an array of bytes */
      bp = (word8 *) flp;
      op += bp[0];
      op += bp[1];
      op += bp[2];
      op += bp[3];
   }  /* end for(i = 0; ... */

   return op;
}  /* end peach_dflops() */

/**
 * @private
 * Perform deterministic memory transformations on @a len bytes of @a data.
 * @param data Pointer to data to use in operations
 * @param len Length of @a data to use in operations
 * @param op Operating code from previous Peach algo steps
 * @returns 32-bit unsigned operation code for subsequent Peach algo steps
*/
static word32 peach_dmemtx(void *data, size_t len, word32 op)
{
   unsigned i, z;
   size_t len16, len32, len64, y;
   word64 *qp;
   word32 *dp;
   word8 *bp, temp;

   /* prepare memory pointers and lengths */
   qp = (word64 *) data;
   dp = (word32 *) data;
   bp = (word8 *) data;
   len64 = len >> 3;
   len32 = len >> 2;
   len16 = len >> 1;
   /* perform memory transformations multiple times */
   for (i = 0; i < PEACHROUNDS; i++) {
      /* determine operation to use for this iteration */
      op += bp[i & 31];
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
}  /* end peach_dmemtx() */

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
static void peach_nighthash(void *in, size_t inlen, word32 index,
   size_t txlen, void *out)
{
   static const word64 key32B[4] = { 0, 0, 0, 0 };
   static const word64 key64B[8] = {
      WORD64_C(0x0101010101010101), WORD64_C(0x0101010101010101),
      WORD64_C(0x0101010101010101), WORD64_C(0x0101010101010101),
      WORD64_C(0x0101010101010101), WORD64_C(0x0101010101010101),
      WORD64_C(0x0101010101010101), WORD64_C(0x0101010101010101),
   };

   /* Perform flops to determine initial algo type.
    * When txlen is non-zero the transformation of input data is enabled,
    * as well as the additional memory transformation process. */
   if (txlen) {
      index = peach_dflops(in, txlen, index, 1);
      index = peach_dmemtx(in, txlen, index);
   } else index = peach_dflops(in, inlen, index, 0);

   /* reduce algorithm selection to 1 of 8 choices */
   switch (index & 7) {
      case 0: blake2b(in, inlen, key32B, 32, out, BLAKE2BLEN256); break;
      case 1: blake2b(in, inlen, key64B, 64, out, BLAKE2BLEN256); break;
      case 2: {
         sha1(in, inlen, out);
         /* SHA1 hash is only 20 bytes long, zero fill remaining... */
         ((word32 *) out)[5] = 0;
         ((word64 *) out)[3] = 0;
         break;
      }
      case 3: sha256(in, inlen, out); break;
      case 4: sha3(in, inlen, out, SHA3LEN256); break;
      case 5: keccak(in, inlen, out, KECCAKLEN256); break;
      case 6: {
         md2(in, inlen, out);
         /* MD2 hash is only 16 bytes long, zero fill remaining... */
         ((word64 *) out)[2] = 0;
         ((word64 *) out)[3] = 0;
         break;
      }
      case 7: {
         md5(in, inlen, out);
         /* MD5 hash is only 16 bytes long, zero fill remaining... */
         ((word64 *) out)[2] = 0;
         ((word64 *) out)[3] = 0;
         break;
      }
   }  /* end switch(algo_type)... */
}  /* end peach_nighthash() */

/**
 * @private
 * Generate a tile of the Peach map.
 * @param index Index number of tile to generate
 * @param phash Previous block hash for use in tile generation
 * @param out Pointer to location to place generated tile
*/
static void peach_generate(word32 index, void *phash, word8 *out)
{
   int i;

   /* place initial data into seed */
   memcpy(out, &index, 4);
   memcpy(out + 4, phash, SHA256LEN);
   /* perform initial nighthash into first row of tile */
   peach_nighthash(out, PEACHGENLEN, index, PEACHGENLEN, out);
   /* fill the rest of the tile with the preceding Nighthash result */
   for (i = 0; i < 992; i += 32) {
      memcpy(out + i + 32, &index, 4);
      peach_nighthash(&out[i], PEACHGENLEN, index, SHA256LEN, &out[i + 32]);
   }
}  /* end peach_generate() */

/**
 * @private
 * Generate and/or retrieve a tile of the Peach map. CPU solving only.
 * @param index Index number of tile on Peach map
 * @param phash Previous block hash for use in tile generation
 * @param out Pointer to location to place generated tile
 * @returns Pointer to (previously) generated tile data
*/
static inline word8 *peach_gencache(word32 index, void *phash, word8 *out)
{
#ifdef ENABLE_CPU_PEACH_CACHE
   /* return cache or redirect out to correct map tile */
   if (PeachCache[index]) {
      return &PeachMap[index * PEACHTILELEN];
   } else out = &PeachMap[index * PEACHTILELEN];

#endif

   /* generaion tile to out */
   peach_generate(index, phash, out);

#ifdef ENABLE_CPU_PEACH_CACHE
   /* flag index as generated */
   PeachCache[index] = 1;

#endif

   return out;
}  /* end peach_gencache() */

/**
 * @private
 * Perform an index jump using the hash result of the Nighthash function.
 * @param index Index number of (current) tile on Peach map
 * @param nonce Nonce for use as entropy in jump direction
 * @param tilep Pointer to tile data at @a index
 * @returns 32-bit unsigned index of next tile
*/
static void peach_jump(word32 *index, word8 *nonce, word8 *tilep)
{
   word8 seed[PEACHJUMPLEN];
   word32 dhash[SHA256LEN / 4];

   /* construct seed for use as Nighthash input for this index on the map */
   memcpy(seed, nonce, HASHLEN);
   memcpy(seed + HASHLEN, index, 4);
   memcpy(seed + HASHLEN + 4, tilep, PEACHTILELEN);
   /* perform nighthash on PEACHJUMPLEN bytes of seed */
   peach_nighthash(seed, PEACHJUMPLEN, *index, 0, dhash);
   /* sum hash as 8x 32-bit unsigned integers */
   *index = (
      dhash[0] + dhash[1] + dhash[2] + dhash[3] +
      dhash[4] + dhash[5] + dhash[6] + dhash[7]
   ) & PEACHCACHELEN_M1;
}  /* end peach_jump() */

/**
 * Check Peach Proof-of-Work. The haiku must be syntactically correct AND
 * have the right vibe. Also, entropy MUST match difficulty.
 * @param bt Pointer to block trailer to check
 * @param diff Difficulty to test against entropy of final hash
 * @param out Pointer to location to place final hash, if non-null
 * @returns VEOK on success, else VERROR
*/
int peach_checkhash(BTRAILER *bt, word8 diff, void *out)
{
   SHA256_CTX ictx;
   word8 hash[SHA256LEN], tile[PEACHTILELEN];
   word32 mario;
   int i;

   /* check syntax, semantics, and vibe... */
   if(trigg_syntax(bt->nonce)) return VERROR;
   if(trigg_syntax(bt->nonce + 16)) return VERROR;
   /* hash block trailer (with nonce) to find starting tile */
   sha256(bt, 124, hash);
   /* initialize mario's starting tile index, bound to PEACHCACHELEN */
   for(mario = hash[0], i = 1; i < SHA256LEN; i++) {
      mario *= hash[i];
   }
   mario &= PEACHCACHELEN_M1;
   /* generate tile at index, determine next index, x PEACHROUNDS */
   for(i = 0; i < PEACHROUNDS; i++) {
      peach_generate(mario, bt->phash, tile);
      peach_jump(&mario, bt->nonce, tile);
   } /* ... then generate final tile for hashing */
   peach_generate(mario, bt->phash, tile);
   /* hash block trailer with final tile */
   sha256_init(&ictx);
   sha256_update(&ictx, hash, SHA256LEN);
   sha256_update(&ictx, tile, PEACHTILELEN);
   sha256_final(&ictx, hash);
   /* where `out` pointer is supplied, copy final hash */
   if(out != NULL) memcpy(out, hash, SHA256LEN);
   /* return trigg's evaluation of the final hash */
   return trigg_eval(hash, diff);
}  /* end peach_checkhash() */

/**
 * Initialize configuration parameters for solving a Block Trailer with
 * the Peach Proof-of-Work algorithm.
 * @param bt Pointer to block trailer to initialize for work
 * @returns VEOK
*/
int peach_init(BTRAILER *bt)
{
#ifdef ENABLE_CPU_PEACH_CACHE
   word8 hashphash[SHA256LEN];
   word64 *zp;
   int i;

   sha256(bt->phash, SHA256LEN, hashphash);
   if (memcmp(PeachCleared, hashphash, SHA256LEN)) {
      /* store last hash the cache was cleared for */
      memcpy(PeachCleared, hashphash, SHA256LEN);
      /* clear Cache data if phash does not match block trailer's */
      zp = (word64 *) PeachCache;
      for (i = 0; i < PEACHCACHELEN64; zp[i++] = 0);
   }

#endif

   /* pre-compute partial SHA256 of block trailer */
   sha256_init(&PeachICTX);
   sha256_update(&PeachICTX, bt, 92);

   return VEOK;
}  /* end peach_init() */

/**
 * Try solve for a tokenized haiku as nonce output for Peach proof of work.
 * Combine haiku protocols implemented in the Trigg Algorithm with the
 * memory intensive protocols of the Peach algorithm to generate haiku
 * output as proof of work.
 * @param bt Pointer to block trailer to solve for
 * @param diff Difficulty to test against entropy of final hash
 * @param out Pointer to location to place nonce on solve
 * @returns VEOK on solve, else VERROR
*/
int peach_solve(BTRAILER *bt, word8 diff, void *out)
{
   static const size_t SHA256_CTX_SIZE = sizeof(SHA256_CTX);

   SHA256_CTX ictx;
   word8 *tilep, hash[SHA256LEN], tile[PEACHTILELEN], nonce[HASHLEN];
   word32 mario;
   int i;

   /* set (initial) tile pointer */
   tilep = tile;
   /* generate (full) nonce */
   trigg_generate_fast(nonce);
   trigg_generate_fast(nonce + 16);
   /* copy pre-computed SHA256 */
   memcpy(&ictx, &PeachICTX, SHA256_CTX_SIZE);
   /* update pre-computed SHA256 with nonce and finalize */
   sha256_update(&ictx, nonce, SHA256LEN);
   sha256_final(&ictx, hash);
   /* initialize mario's starting index on the map, bound to PEACHCACHELEN */
   for(mario = hash[0], i = 1; i < SHA256LEN; i++) {
      mario *= hash[i];
   }
   mario &= PEACHCACHELEN_M1;
   /* generate tile at index, then determine next jump, for PEACHROUNDS, ... */
   for(i = 0; i < PEACHROUNDS; i++) {
      tilep = peach_gencache(mario, bt->phash, tilep);
      peach_jump(&mario, nonce, tilep);
   } /* ... then generate final tile for hashing */
   tilep = peach_gencache(mario, bt->phash, tilep);
   /* hash block trailer with final tile */
   sha256_init(&ictx);
   sha256_update(&ictx, hash, SHA256LEN);
   sha256_update(&ictx, tilep, PEACHTILELEN);
   sha256_final(&ictx, hash);
   /* evaluate result against required difficulty */
   if(trigg_eval(hash, diff) == VEOK) {
      /* copy successful (full) nonce to `out` */
      memcpy(out, nonce, SHA256LEN);
      return VEOK;
   }

   return VERROR;
}  /* end peach_solve() */

/* end include guard */
#endif
