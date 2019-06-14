/*
 * peach.c  FPGA-Tough CPU Mining Algo
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 05 June 2019
 * Revision: 1
 *
 * This file is subject to the license as found in LICENSE.PDF
 *
 */

#include "peach.h"
#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <sys/time.h>

#include "../../crypto/hash/cpu/sha1.c"
//#include "nighthash.c"

/* Prototypes from trigg.o dependency */
byte *trigg_gen(byte *in);
void trigg_expand2(byte *in, char *out);

/* Prototype for forward reference.  To-do: Clean this up. */
void generate_tile(byte **out, uint32_t index, byte *seed, byte *map);

/*
 * Return 0 if solved, else 1.
 * Note: We can probably just use trigg_eval here.
 */
int peach_eval(byte *bp, byte d)
{
   byte x, i, j, n;

   x = i = j = n = 0;

   for (i = 0; i < HASHLEN; i++) {
      x = *(bp + i);
      if (x != 0) {
         for(j = 7; j > 0; j--) {
            x >>= 1;
            if(x == 0) {
               n += j;
               break;
            }
         }
      break;
      }
      n += 8;
      continue;
   }
   if(n >= d) return 0;
   return 1;
}

uint32_t next_index(uint32_t current_index, byte *current_tile, byte *nonce)
{
   SHA256_CTX ictx;
   uint32_t index;
   byte hash[HASHLEN];
   int i;
   
   sha256_init(&ictx);

   /* Hash nonce in first to prevent caching of index's hash. */
   sha256_update(&ictx, nonce, HASHLEN);
   sha256_update(&ictx, (byte*) &current_index,sizeof(uint32_t));
   sha256_update(&ictx, current_tile, TILE_LENGTH);
   sha256_final(&ictx, hash);
   
   /* night_hash2(hash, j, current_tile, TILE_LENGTH); */

   /* Convert 32-byte Hash Value Into 32-bit Unsigned Integer */
   for(i = 0, index = 0; i < (HASHLEN / 4); i++) index += *((uint32_t *) &hash[i]);

   return index % MAP;
}

void get_tile(byte **out, uint32_t index, byte *seed, byte *map, byte *cache)
{
   /* Check cache to see if we've already generated the tile. */
   if(cache != NULL && cache[index]) {
      *out = map + index * TILE_LENGTH;
      return;
   }

   /* Tile not yet generated, generate it, and flag the cache accordingly. */   
   generate_tile(out, index, seed, map);
   if(cache != NULL) cache[index] = 1;
}

void night_hash(byte *out, byte *in, uint32_t inlen)
{
   uint32_t op;
   op = 0;
   SHA1_CTX sha1;
   SHA256_CTX sha256;


   /* TODO: Replace with floating point arithmetic */
   for(int i = 0; i < inlen; i++)
      op += in[i];

   switch(op & 7)
   {
      case 0:
         /* Production: Blake2b key 32 bytes, Placeholder: SHA1 */
         sha1_init(&sha1);
         sha1_update(&sha1, in, inlen);
         sha1_final(&sha1, out);
         break;
      case 1:
         /* Production: Blake2b key 64 bytes, Placeholder: SHA256 */
         sha256_init(&sha256);
         sha256_update(&sha256, in, inlen);
         sha256_final(&sha256, out);
         break;
      case 2:
         /* Production: SHA1 */
         sha1_init(&sha1);
         sha1_update(&sha1, in, inlen);
         sha1_final(&sha1, out);
         break;
      case 3:
         /* Production: SHA256 */
         sha256_init(&sha256);
         sha256_update(&sha256, in, inlen);
         sha256_final(&sha256, out);
         break;
      case 4:
         /* Production SHA3, Placeholder: SHA1 */
         sha1_init(&sha1);
         sha1_update(&sha1, in, inlen);
         sha1_final(&sha1, out);
         break;         
      case 5:
         /* Production Keccak, Placeholder: SHA256 */
         sha256_init(&sha256);
         sha256_update(&sha256, in, inlen);
         sha256_final(&sha256, out);
         break;
      case 6:
         /* Production MD4, Placeholder: SHA1 */
         sha1_init(&sha1);
         sha1_update(&sha1, in, inlen);
         sha1_final(&sha1, out);
         break;
      case 7:
         /* Production MD5, Placeholder: SHA256 */
         sha256_init(&sha256);
         sha256_update(&sha256, in, inlen);
         sha256_final(&sha256, out);
         break;
      default:
         error("Fatal: Peach night hash OP is outside the expected range (%i)\n", op);
         assert(0);
         break;
   }
}

void generate_tile(byte **out, uint32_t index, byte *seed, byte *map)
{
   SHA256_CTX ictx;
   uint32_t op;
   byte bits, _104, _72, selector, *tilep;
   int i, j, k, t, z;
   float floatv, *floatp;

   _104 = 104;
   _72 = 72;
  
   /* Set map pointer. */
   if(map == NULL) tilep = *out;
   if(map != NULL) tilep = &map[index * TILE_LENGTH];
  
   /* Create tile hashing context. */
   sha256_init(&ictx);

   /* Hash nonce in first to prevent caching of index's hash. */
   sha256_update(&ictx, seed, HASHLEN);    
   sha256_update(&ictx, (byte*)&index, sizeof(uint32_t));
   sha256_final(&ictx, tilep);
  
   for(i = j = k = 0; i < TILE_LENGTH; i += HASHLEN) { /* For each tile row */
      for(op = 0; j < i + HASHLEN; j += 4) {
         
         /* set float pointers */
         floatp = (float *) &tilep[j];
         
         /* Byte selections depend on initial 8 bits
          * Note: Trying not to perform "floatv =" first */
         switch(tilep[k] & 7) {
            case 0:
            {
               // skip a byte
               k++;
               // determine floating point operation type
               op = tilep[k++];
               // determine which byte to select on the current 32 byte series
               selector = tilep[k++] & (HASHLEN - 1); // & (HASHLEN - 1), returns 0-31
               floatv = (float) tilep[i + selector]; // i + selector, index in 32 byte series
               // determine if floating point operation is performed on a negative number
               if(tilep[k++] & 1) floatv = (float)((int)floatv ^ 0x80000000);
            }
               break;
            case 1:
            {
               k++;
               selector = tilep[k++] & (HASHLEN - 1);
               floatv = (float) tilep[i + selector];
               if(tilep[k++] & 1) floatv = (float)((int)floatv ^ 0x80000000);
               op = tilep[k++];
            }
               break;
            case 2:
            {
               op = tilep[k++];
               k++;
               selector = tilep[k++] & (HASHLEN - 1);
               floatv = (float) tilep[i + selector];
               if(tilep[k++] & 1) floatv = (float)((int)floatv ^ 0x80000000);
            }
               break;
            case 3:
            {
               op = tilep[k++];
               selector = tilep[k++] & (HASHLEN - 1);
               floatv = (float) tilep[i + selector];
               if(tilep[k++] & 1) floatv = (float)((int)floatv ^ 0x80000000);
               k++;
            }
               break;
            case 4:
            {
               selector = tilep[k++] & (HASHLEN - 1);
               floatv = (float) tilep[i + selector];
               if(tilep[k++] & 1) floatv = (float)((int)floatv ^ 0x80000000);
               k++;
               op = tilep[k++];
            }
               break;
            case 5:
            {
               selector = tilep[k++] & (HASHLEN - 1);
               floatv = (float) tilep[i + selector];
               if(tilep[k++] & 1) floatv = (float)((int)floatv ^ 0x80000000);
               op = tilep[k++];
               k++;
            }
               break;
            case 6:
            {
               op = tilep[k++];
               selector = tilep[k++] & (HASHLEN - 1);
               floatv = (float) tilep[i + selector];
               k++;
               if(tilep[k++] & 1) floatv = (float)((int)floatv ^ 0x80000000);
            }
               break;
            case 7:
            {
               k++;
               selector = tilep[k++] & (HASHLEN - 1);
               floatv = (float) tilep[i + selector];
               op = tilep[k++];
               if(tilep[k++] & 1) floatv = (float)((int)floatv ^ 0x80000000);
            }
               break;
            default:
               error("Fatal: Peach float OP is outside the expected range (%i)\n", op);
               assert(0);
               break;
         }

         /* Replace NaN's with tileNum. */
         if(isnan(*floatp)) *floatp = (float) index;
         if(isnan(floatv)) floatv = (float) index;

         /* Float operation depends on final 8 bits.
          * Perform floating point operation. */
         switch(op & 3) {
            case 0:
               {
                  *floatp += floatv;
               }
                  break;
            case 1:
               {
                  *floatp -= floatv;
               }
                  break;
            case 2:
               {
                  *floatp *= floatv;
               }
                  break;
            case 3:
               {
                  *floatp /= floatv;
               }
                  break;
         }
         
      } /* end for(op = 0... */
      
      /* Execute bit manipulations per tile row. */
      for(t = 0; t < TILE_TRANSFORMS; t++) {
         /* Determine tile byte offset and operation to use. */
         op += (uint32_t) tilep[i + (t % HASHLEN)];

         switch(op & 7) {
            case 0: /* Swap the first and last bit in each byte. */
               for(z = 0; z < HASHLEN; z++)
                  tilep[i + z] ^= 0x81;
               break;
            case 1: /* Swap bytes */
               for(z = 0;z<HASHLENMID;z++) {
                  bits = tilep[i + z];
                  tilep[i + z] = tilep[i + HASHLENMID + z];
                  tilep[i + HASHLENMID + z] = bits;
               }
               break;
            case 2: /* Complement One, all bytes */
               for(z = 1; z < HASHLEN; z++) tilep[i + z] = ~tilep[i + z];
               break;
            case 3: /* Alternate +1 and -1 on all bytes */
               for(z = 0; z < HASHLEN; z++) {
                  tilep[i + z] += ((z & 1) == 0) ? 1 : -1;
               }
               break;
            case 4: /* Alternate +t and -t on all bytes */
               for(z = 0; z < HASHLEN; z++) {
                  tilep[i + z] += ((z & 1) == 0) ? -t : t;
               }
               break;
            case 5: /* Replace every occurrence of h with H */ 
               for(z = 0; z < HASHLEN; z++) {
                  if(tilep[i + z] == _104) tilep[i + z] = _72;
               }
               break;
            case 6: /* If byte a is > byte b, swap them. */
               for(z = 0; z < HASHLENMID; z++) {
                  if(tilep[i + z] > tilep[i + HASHLENMID + z]) {
                     bits = tilep[i + z];
                     tilep[i + z] = tilep[i + HASHLENMID + z];
                     tilep[i + HASHLENMID + z] = bits;
                  }
               }
               break;
            case 7: /* XOR all bytes */
               for(z = 1; z < HASHLEN; z++) tilep[i + z] ^= tilep[i + z - 1];
               break;
            default:
               error("Fatal: Peach transform OP is outside the expected range (%i)\n", op);
               assert(0);
               break;
         } /* end switch(... */
      } /* end for(t = 0... */ 
      
      /* Hash the result of the current tile's row to the next. */
      if(j < TILE_LENGTH) {
         sha256_init(&ictx);
         sha256_update(&ictx, &tilep[i], HASHLEN);
         sha256_update(&ictx, (byte*) &index, sizeof(uint32_t));
         sha256_final(&ictx, &tilep[j]);
         
         night_hash(&tilep[j], &tilep[j], HASHLEN);
         /* night_hash2(&tilep[j], j, &tilep[j], HASHLEN); */
      }
   } /* end for(i = j = k = 0... */

   if(map != NULL) *(out) = map + (index * TILE_LENGTH);
} /* end generate_tile() */

int is_solution(byte diff, byte* tile, byte* bt_hash)
{
   SHA256_CTX ictx;
   byte hash[HASHLEN];

   sha256_init(&ictx);
   sha256_update(&ictx, bt_hash, HASHLEN);
   sha256_update(&ictx, tile, TILE_LENGTH);
   sha256_final(&ictx, hash);

   return peach_eval(hash, diff) == 0;
}

/**
 * Mode 0: Mining
 * Mode 1: Validating
 *
 */
int peach(BTRAILER *bt, word32 difficulty, word32 *hps, int mode)
{
   SHA256_CTX ictx;

   uint32_t sm;
   uint64_t j, h;
   struct timeval tstart, tend, telapsed;
   byte *map, *cache, *tile, diff, bt_hash[HASHLEN];
   int solved, cached;

   diff = difficulty; /* down-convert passed-in 32-bit difficulty to 8-bit */
   h = 0;
   map = NULL;
   cache = NULL;
   tile = NULL;
   solved = 0;
   
   gettimeofday(&tstart, NULL);
   
   plog("Peach mode %i, diff %i", mode, diff);   
   if(mode == 0) {
      /* Allocate MAP on the Heap */
      map = malloc(MAP_LENGTH);
      if(map == NULL) {
         if(Trace) plog("Fatal: Unable to allocate memory for map.\n");
         goto out;
      }
      memset(map, 0, MAP_LENGTH);
  
      /* Allocate MAP cache on the Heap */
      cache = malloc(MAP);
      if(cache == NULL) {
         if(Trace) plog("Fatal: Unable to allocate memory for cache.\n");
         goto out;
      }
      memset(cache, 0, MAP);
   } else {
      tile = malloc(TILE_LENGTH);
      if(tile == NULL) {
         if(Trace) plog("Fatal: Unable to allocate memory for tile.\n");
         goto out;
      }
   }

   for(;;) {
      if(!Running && mode == 0) goto out; /* SIGTERM Received */
      h += 1;
      sm = 0;
      if(mode == 0) {
         /* In mode 0, add random haiku to passed-in candidate block trailer */
         memset(&bt->nonce[0], 0, HASHLEN);
         trigg_gen(&bt->nonce[0]);
         trigg_gen(&bt->nonce[16]);
      }
       
      sha256_init(&ictx);
      sha256_update(&ictx, (byte *) bt, 124 /*BTSIZE - 4 - HASHLEN*/);
      sha256_final(&ictx, bt_hash);

      for(int i = 0; i < HASHLEN; i++){
         if(i == 0) {
            sm = bt_hash[i];
         } else {
            sm *= bt_hash[i];
         }
      }

      sm %= MAP;
      
      get_tile(&tile, sm, bt->phash, map, cache);
     
      for(j = 0; j < JUMP; j++) {
         sm = next_index(sm, tile, bt->nonce);
         get_tile(&tile, sm, bt->phash, map, cache);
      }
       
      solved = is_solution(diff, tile, bt_hash);
      /* include the mining address and transactions as part of the solution */

      if(mode == 1) { /* Just Validating, not Mining, check once and return */
         gettimeofday(&tend, NULL);
         timersub(&tend, &tstart, &telapsed);
         plog("Peach validated in %ld.%06ld seconds", 
             (long int) telapsed.tv_sec, (long int) telapsed.tv_usec);

         goto out;
      }

      if(solved) { /* We're Mining & We Solved! */
         gettimeofday(&tend, NULL);
         timersub(&tend, &tstart, &telapsed);

         if(peach(bt, difficulty, NULL, 1)) {
            byte* bt_bytes = (byte*) bt;
            char hex[124 * 4];
            for(int i = 0; i < 124; i++){
               sprintf(hex + i * 4, "%03i ", bt_bytes[i]);
            }

            error("!!!!!Peach Validation failed IN THE CONTEXT!!!!!");
            error("BT -> %s", hex);
            plog("!!!!!Peach Validation failed IN THE CONTEXT!!!!!");
            plog("BT -> %s", hex);
         }

         cached = 0;
         for(int i = 0; i < MAP; i++) {
            if(cache[i]) cached++;
         }
         plog("Peach found in %ld.%06ld seconds, %li iterations, %i cached", 
             (long int) telapsed.tv_sec, (long int)telapsed.tv_usec, h, cached);
         *hps = h;

         char haiku[256];
         trigg_expand2(bt->nonce, haiku);
         printf("\nS:%s\n\n", haiku);
 
         goto out;
      } /* end if(solved)... */

	  /*
            THANK YOUR MARIO!
         BUT OUR PRINCESS IS IN 
	     ANOTHER CASTLE!
		 
      */
	  
   } /* end for(;;)... */

out:
   if(map != NULL) free(map);
   if(cache != NULL) free(cache);

   if(mode == 1 && tile != NULL) free(tile); /* When validating... */

   tile = map = cache = NULL;

   if(mode == 1 && solved == 0) plog("?????Peach Validation failed?????");

   if(mode != 1 && !Running) return 1; /* SIGTERM RECEIVED */

   return solved ? 0 : 1;  /* Return 0 if valid, 1 if not valid */
} /* End peach() */
