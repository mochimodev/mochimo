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

#include "nighthash.c"

/* Prototypes from trigg.o dependency */
byte *trigg_gen(byte *in);
int trigg_syntax(byte *in);
void trigg_expand2(byte *in, char *out);

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
   nighthash_ctx_t nighthash;
   byte seed[HASHLEN + 4 + TILE_LENGTH];
   byte hash[HASHLEN];
   int i, seedlen;
   uint32_t index;

   /* Create nighthash seed for this index on the map */
   seedlen = HASHLEN + 4 + TILE_LENGTH;
   memcpy(seed, nonce, HASHLEN);
   memcpy(seed + HASHLEN, (byte *) &current_index, 4);
   memcpy(seed + HASHLEN + 4, current_tile, TILE_LENGTH);
   
   /* Setup nighthash the seed, NO TRANSFORM */
   nighthash_seed_init(&nighthash, seed, seedlen, current_index, 256);

   /* Update nighthash with the seed data */
   nighthash_update(&nighthash, seed, seedlen);

   /* Finalize nighthash into the first 32 byte chunk of the tile */
   nighthash_final(&nighthash, hash);

   /* Convert 32-byte Hash Value Into 8x 32-bit Unsigned Integer */
   for(i = 0, index = 0; i < (HASHLEN >> 2); i++)
      index += ((uint32_t *) &hash)[i];

   return index % MAP;
}

void generate_tile(byte **out, uint32_t index, byte *phash, byte *map)
{
   nighthash_ctx_t nighthash;
   byte seed[4 + HASHLEN];
   byte *tilep;
   int i, j, seedlen;

   /* Set tile pointer */
   if(map == NULL) tilep = *out;
   else tilep = &map[index * TILE_LENGTH];

   /* Create nighthash seed for this index on the map */
   seedlen = 4 + HASHLEN;
   memcpy(seed, (byte *) &index, 4);
   memcpy(seed + 4, phash, HASHLEN);

   /* Setup nighthash with a transform of the seed */
   nighthash_transform_init(&nighthash, seed, seedlen, index, 256);

   /* Update nighthash with the seed data */
   nighthash_update(&nighthash, seed, seedlen);

   /* Finalize nighthash into the first 32 byte chunk of the tile */
   nighthash_final(&nighthash, tilep);

   /* Begin constructing the full tile */
   for(i = 0; i < TILE_LENGTH; i += HASHLEN) { /* For each tile row */
      /* Set next row's pointer location */
      j = i + HASHLEN;

      /* Hash the current row to the next, if not at the end */
      if(j < TILE_LENGTH) {
         /* Setup nighthash with a transform of the current row */
         nighthash_transform_init(&nighthash, &tilep[i], HASHLEN,
                                  index, 256);

         /* Update nighthash with the seed data and tile index */
         nighthash_update(&nighthash, &tilep[i], HASHLEN);
         nighthash_update(&nighthash, (byte *) &index, 4);

         /* Finalize nighthash into the first 32 byte chunk of the tile */
         nighthash_final(&nighthash, &tilep[j]);
      }
   }

   if(map != NULL) *(out) = tilep;
} /* end generate_tile() */

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
   
   if(Trace) plog("Peach mode %i, diff %i", mode, diff);   
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
      } else if(mode == 1) {
         /* Validation Precheck, The haiku must be syntactically correct
          * and have the right vibe */
         if(trigg_syntax(&bt->nonce[0]) == 0 ||
            trigg_syntax(&bt->nonce[16]) == 0) {
            solved = 0;
            goto out;
         }
      }
       
      sha256_init(&ictx);
      sha256_update(&ictx, (byte *) bt, 124 /*BTSIZE - 4 - HASHLEN*/);
      sha256_final(&ictx, bt_hash);

      sm = bt_hash[0];
      for(int i = 1; i < HASHLEN; i++)
         sm *= bt_hash[i];

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
         if(Trace)
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
