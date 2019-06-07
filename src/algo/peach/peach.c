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

/* Prototypes from trigg.o dependency */
byte *trigg_gen(byte *in);
void trigg_expand2(byte *in, byte *out);

void generate_tile(byte** out, uint64_t index, byte* seed, byte * map,  byte * cache);

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

uint64_t next_index(uint64_t current_index, byte* current_tile, byte* nonce)
{
	SHA256_CTX ictx;
	uint64_t index;
	byte hash[HASHLEN];

	sha256_init(&ictx);
	sha256_update(&ictx, nonce, HASHLEN);//hash nonce first because we dont want to allow caching of index computation
	sha256_update(&ictx, (byte*) &current_index, 8);
	sha256_update(&ictx, current_tile, TILE);

	sha256_final(&ictx, hash);

	index = *((word32 *) hash);//read first 4 bytes as unsigned int
	index += ((word32 *) hash)[1];//read last 4 bytes as unsigned int

	return index % MAP;
}

void get_tile(byte** out, uint64_t index, byte* seed, byte * map,  byte * cache)
{
	if(cache[index])
	{
		*out = map + index * TILE;
		return;
	}

	generate_tile(out, index, seed, map, cache);

	cache[index] = 1;
}

void generate_tile(byte** out, uint64_t index, byte* seed, byte * map,  byte * cache)
{
	SHA256_CTX ictx;
	byte hash[HASHLEN], *b, *b1, *b2, *b3, *b4, _104, _72;
	uint64_t op, tile_start, tile_offset, i1, i2, i3, i4;

	sha256_init(&ictx);
	sha256_update(&ictx, seed, HASHLEN);//hash seed first because we don't want to allow caching of index computation
	sha256_update(&ictx, (byte*) &index, 8);
	sha256_final(&ictx, hash);


	int h = *((byte*)hash);// is this right ?
	i1 = *((word32 *) hash);//read first 4 bytes as unsigned int
	i2 = ((word32 *) hash)[1];//read last 4 bytes as unsigned int
	i3 = i1 ^ i2;
	i4 = i1 + i2;

	i1 = i1 % HASHLEN;
	i2 = i2 % HASHLEN;
	i3 = i3 % HASHLEN;
	i4 = i4 % HASHLEN;

	word32 w = 0x6D617474;
	_104 = 104;
	_72 = 72;

	tile_start = index * TILE;

	byte ops[4];
	ops[0] = 6;
	ops[1] = 6;
	ops[2] = 6;
	ops[3] = 6;
	op = 0;

	for(int i=0;i<TILE_FACTOR;i++)
	{
		//printf("######\n");

		tile_offset = tile_start + i * HASHLEN;

		memcpy(map + tile_offset, hash, HASHLEN);
		//printf("%li\n", tile_offset);


		for(int t=0;t<TILE_TRANSFORM;t++)
		{
			//Use some floating point calculation to compute op ?

			//printf("OP value on new iteration: %lu \n", op);
			for(int z = (h ^ i ^ t) % (HASHLEN >> 1);z<HASHLEN;z++)
				op += hash[z];

			//op = hash[0] ;


			op %= 8;
			//printf("OP value on mid iteration: %lu \n", op);
			//op = ops[op];

			//printf("%li\n", op);

			switch(op)
			{
			  case 0: /* Swap the first and last bit of a byte. */
			  {
				  if(i1 % 2 == 0){
					  b = map + tile_offset + i1;
				  }
				  else{
					  b = map + tile_offset + i2;
				  }

				  *b ^= (1 << 0);
				  *b ^= (1 << 7);
			  }
				  break;
			  case 1: /* Swap the first and last byte. */
			  {
				  byte tmp = map[tile_offset];
				  map[tile_offset] = map[tile_offset + HASHLEN -1];
				  map[tile_offset + HASHLEN -1] = tmp;
			  }
				  break;
			  case 2: /* XOR two bytes */
			  {
				  map[tile_offset + i4] = map[tile_offset + i4] ^ map[tile_offset + i3];
			  }
				  break;
			  case 3: /* Alternate +1 and -1 on all bytes */
			  {
				  for(int j=0;j<HASHLEN;j++)
					  map[tile_offset + j] += j % 2 ==0 ? 1 : -1;
			  }
				  break;
			  case 4: /* Alternate +t and -t on all bytes */
			  {
				  for(int j=0;j<HASHLEN;j++)
					  map[tile_offset + j] += j % 2 == 0 ? t : -t;
			  }
				  break;
			  case 5: /* Replace every occurence of h with H */
			  {
				  if(map[tile_offset + i1] == _104)
					  map[tile_offset + i1] = _72;

				  if(map[tile_offset + i2] == _104)
					  map[tile_offset + i2] = _72;

				  if(map[tile_offset + i3] == _104)
					  map[tile_offset + i3] = _72;

				  if(map[tile_offset + i4] == _104)
					  map[tile_offset + i4] = _72;
			  }
				  break;
			  case 6: /* If byte a is > byte b, swap them. */
			  {
				  byte x;
				  if(map[tile_offset + i1] > map[tile_offset + i3])
				  {
					  x = map[tile_offset + i1];
					  map[tile_offset + i1] = map[tile_offset + i3];
					  map[tile_offset + i3] = x;
				  }
			  }
			  	  break;
			  case 7: /* XOR all bytes */
			  {
				  for(int j=1;j<HASHLEN;j++)
					  map[tile_offset + j] ^= map[tile_offset + j - 1];
			  }
			  	  break;
			  default:
				printf("Outside operation range\n");
				break;
			}
			//printf("OP value on end iteration: %lu \n", op);
		}
	}

	*(out) = map + tile_start;
}

int is_solution(byte diff, byte* tile, byte* nonce)
{
	SHA256_CTX ictx;
	uint64_t index;
	byte hash[HASHLEN];

	sha256_init(&ictx);
	sha256_update(&ictx, nonce, HASHLEN);//hash nonce first because we dont want to allow caching of index computation
	sha256_update(&ictx, tile, TILE);
	sha256_final(&ictx, hash);

	return peach_eval(hash, diff) == 0;
}

/**
 * Mode 0: Mining
 * Mode 1: Validating
 *
 */
int peach(BTRAILER *bt, word32 difficulty, byte *haiku, word32 *hps, int mode)
{
   printf("Peach mode %i\n", mode);
   SHA256_CTX ictx, mctx; /* Index & Mining Contexts */

   uint64_t map_length, sm, sm2;
   map_length = MAP_LENGTH;

   byte * map, *cache, *tile, *tile2, diff, bt_hash[HASHLEN];
   diff = difficulty; /* down-convert passed-in 32-bit difficulty to 8-bit */
   printf("diff %i\n", diff);

   uint64_t j, h;
   h = 0;


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
   long start = time(NULL);
   int solved = 0;

   for(;;)
   {
	   if(!Running && mode == 0) goto out; /* SIGTERM Received */

	   h += 1;

	   sm = 0;
	   tile = NULL;

	   if(mode == 0) {
		   /* In mode 0, add random haiku to the passed-in candidate block trailer */
		  memset(&bt->nonce[0], 0, HASHLEN);
		  trigg_gen(&bt->nonce[0]);
		  trigg_gen(&bt->nonce[16]);
	   }

	   sha256_init(&ictx);
	   sha256_update(&ictx, (byte *) bt, BTSIZE - 4 - HASHLEN);
	   sha256_final(&ictx, bt_hash);

	   for(int i=0;i<HASHLEN;i++)
		   if(i == 0){
			   sm = bt_hash[i];
		   }else{
			   sm *= bt_hash[i];
		   }

	   sm %= MAP;

	   get_tile(&tile, sm, bt->phash, map, cache);
	   sm = next_index(sm, tile, bt->nonce);

	   for(j = 0; j < JUMP; j++)
	   {
		  // sm2 = next_index(sm, tile, bt->nonce);
		   sm = next_index(sm, tile, bt->nonce);
		   //assert(sm == sm2);

		   /*
			 generate_tile(&tile2, sm, bt->phash, map, cache);
		   generate_tile(&tile, sm, bt->phash, map, cache);
		   assert(memcmp(tile, tile2, TILE) == 0);
		    */
		   get_tile(&tile, sm, bt->phash, map, cache);
	   }

	   solved = is_solution(diff, tile, bt_hash);//include the mining address and transactions as part of the solution

	   if(mode == 1) { /* Just Validating, not Mining, check once and return */
		  trigg_expand2(bt->nonce, &haiku[0]);
		  if(Trace) plog("\nV:%s\n\n", haiku);

		  goto out;
	   }

	   if(solved)
	   { /* We're Mining & We Solved! */

		  byte v24haiku[256];
		  if(peach(bt, difficulty, &v24haiku[0], NULL, 1))
			  printf("????Validation failed IN THE CONTEXT??????\n");
		  long end = time(NULL);
		  long elapsed = end - start;
		  int cached = 0;
		  for (int i =0;i<MAP;i++)
			  if(cache[i])
				  cached++;
		  printf("Solved in %li seconds, %li iterations, %i cached\n", elapsed, h, cached);
		  *hps = h;
		  trigg_expand2(bt->nonce, &haiku[0]);
		  if(Trace) plog("\nS:%s\n\n", haiku);

		  goto out;
	   }
	}


out:
	if(map != NULL) free(map);
	if(cache != NULL) free(cache);

	map = cache = NULL;

	if(mode == 1 && solved == 0)
		printf("####Validation failed#####\n");

	if(mode != 1/*not validating*/ && !Running)
		return 1; /* SIGTERM RECEIVED */
	return solved ? 0 : 1;  /* Return 0 if valid, 1 if not valid */
} /* End v24() */
