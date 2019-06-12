/*
 * blake2b.c  Implementation of Blake2b digest
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


#include "blake2b.h"

/*
 * Max keylen = 64
 */
void blake2b_init(blake2b_ctx_t *ctx, byte* key, uint32_t keylen, uint32_t digestbitlen)
{
   memset(ctx, 0, sizeof(blake2b_ctx_t));

   ctx->key = key;
   ctx->keylen = keylen;
   ctx->digestlen = digestbitlen >> 3;
   ctx->pos = 0;
   ctx->t0 = 0;
   ctx->t1 = 0;
   ctx->f0 = 0;
   ctx->chain[0] = BLAKE2B_IVS[0] ^ (ctx->digestlen | (ctx->keylen << 8) | 0x1010000);
   ctx->chain[1] = BLAKE2B_IVS[1];
   ctx->chain[2] = BLAKE2B_IVS[2];
   ctx->chain[3] = BLAKE2B_IVS[3];
   ctx->chain[4] = BLAKE2B_IVS[4];
   ctx->chain[5] = BLAKE2B_IVS[5];
   ctx->chain[6] = BLAKE2B_IVS[6];
   ctx->chain[7] = BLAKE2B_IVS[7];

	memcpy(ctx->buff, ctx->key, ctx->keylen);
	ctx->pos = BLAKE2B_BLOCK_LENGTH;
}

void blake2b_update(blake2b_ctx_t *ctx, byte* in, uint64_t inlen)
{
	if (inlen == 0)
	   return;

	uint32_t start = 0;
	int64_t inIndex = 0, blockIndex = 0;
	
	if (ctx->pos)
	{
	   start = BLAKE2B_BLOCK_LENGTH - ctx->pos;
		if (start < inlen)
		{ 
         memcpy(ctx->buff + ctx->pos, in, start);
			ctx->t0 += BLAKE2B_BLOCK_LENGTH;
			if (ctx->t0 == 0)
				ctx->t1++;
			
			blake2b_compress(ctx, ctx->buff, 0);
			ctx->pos = 0;
			memset(ctx->buff, 0, BLAKE2B_BLOCK_LENGTH);
		}
		else
		{
			memcpy(ctx->buff + ctx->pos, in, inlen);//read the whole *in
			ctx->pos += inlen;
			return;
		}
	}

	blockIndex =  inlen - BLAKE2B_BLOCK_LENGTH;
	for (inIndex = start; inIndex < blockIndex; inIndex += BLAKE2B_BLOCK_LENGTH)
	{
	   ctx->t0 += BLAKE2B_BLOCK_LENGTH;
		if (ctx->t0 == 0)
	      ctx->t1++;

		blake2b_compress(ctx, in, inIndex);
	}

	memcpy(ctx->buff, in + inIndex, inlen - inIndex);
   ctx->pos += inlen - inIndex;
}

void blake2b_final(blake2b_ctx_t *ctx, byte* out)
{
	ctx->f0 = 0xFFFFFFFFFFFFFFFFL;
	ctx->t0 += ctx->pos;
	if (ctx->pos > 0 && ctx->t0 == 0)
		ctx->t1++;
	
	blake2b_compress(ctx, ctx->buff, 0);
	memset(ctx->buff, 0, BLAKE2B_BLOCK_LENGTH);
	memset(ctx->state, 0, BLAKE2B_STATE_LENGTH);
	
	int i8 = 0;
	for (int i = 0; i < BLAKE2B_CHAIN_SIZE && ((i8 = i * 8) < ctx->digestlen); i++)
	{
		byte * bytes = (byte*)(&ctx->chain[i]);
		if (i8 < ctx->digestlen - 8)
		   memcpy(out + i8, bytes, 8);
		else
           memcpy(out + i8, bytes, ctx->digestlen - i8);
	}
}

void blake2b_init_state(blake2b_ctx_t *ctx)
{
	memcpy(ctx->state, ctx->chain, BLAKE2B_CHAIN_LENGTH);
	for (int i = 0; i < 4; i++)
		ctx->state[BLAKE2B_CHAIN_SIZE + i] = BLAKE2B_IVS[i];

	ctx->state[12] = ctx->t0 ^ BLAKE2B_IVS[4];
	ctx->state[13] = ctx->t1 ^ BLAKE2B_IVS[5];
	ctx->state[14] = ctx->f0 ^ BLAKE2B_IVS[6];
	ctx->state[15] = BLAKE2B_IVS[7];
}

void blake2b_compress(blake2b_ctx_t *ctx, byte* in, uint32_t inoffset)
{
	blake2b_init_state(ctx);

	uint64_t  m[16] = {0};
	for (int j = 0; j < 16; j++)
   	   m[j] = blake2b_leuint64(in + inoffset + j * 8);
	
	for (int round = 0; round < BLAKE2B_ROUNDS; round++)
	{
		blake2b_G(ctx, m[BLAKE2B_SIGMAS[round][0]], m[BLAKE2B_SIGMAS[round][1]], 0, 4, 8, 12);
		blake2b_G(ctx, m[BLAKE2B_SIGMAS[round][2]], m[BLAKE2B_SIGMAS[round][3]], 1, 5, 9, 13);
		blake2b_G(ctx, m[BLAKE2B_SIGMAS[round][4]], m[BLAKE2B_SIGMAS[round][5]], 2, 6, 10, 14);
		blake2b_G(ctx, m[BLAKE2B_SIGMAS[round][6]], m[BLAKE2B_SIGMAS[round][7]], 3, 7, 11, 15);
		blake2b_G(ctx, m[BLAKE2B_SIGMAS[round][8]], m[BLAKE2B_SIGMAS[round][9]], 0, 5, 10, 15);
		blake2b_G(ctx, m[BLAKE2B_SIGMAS[round][10]], m[BLAKE2B_SIGMAS[round][11]], 1, 6, 11, 12);
		blake2b_G(ctx, m[BLAKE2B_SIGMAS[round][12]], m[BLAKE2B_SIGMAS[round][13]], 2, 7, 8, 13);
		blake2b_G(ctx, m[BLAKE2B_SIGMAS[round][14]], m[BLAKE2B_SIGMAS[round][15]], 3, 4, 9, 14);
	}

	for (int offset = 0; offset < BLAKE2B_CHAIN_SIZE; offset++)
		ctx->chain[offset] = ctx->chain[offset] ^ ctx->state[offset] ^ ctx->state[offset + 8];
}

uint64_t blake2b_leuint64(byte *in)
{
	uint64_t a;
	a = *((uint64_t *)in);
	return a;

	/* If memory is not little endian
	uint8_t *a = (uint8_t *)in;
	return ((uint64_t)(a[0]) << 0) | ((uint64_t)(a[1]) << 8) | ((uint64_t)(a[2]) << 16) | ((uint64_t)(a[3]) << 24) |((uint64_t)(a[4]) << 32) 
		| ((uint64_t)(a[5]) << 40) | ((uint64_t)(a[6]) << 48) | 	((uint64_t)(a[7]) << 56);
	 */
}

uint64_t blake2b_ROTR64(uint64_t a, uint8_t b)
{
   return (a >> b) | (a << (64 - b));
}

void blake2b_G(blake2b_ctx_t *ctx, int64_t m1, int64_t m2, int32_t a, int32_t b, int32_t c, int32_t d)
{
	ctx->state[a] = ctx->state[a] + ctx->state[b] + m1;
	ctx->state[d] = blake2b_ROTR64(ctx->state[d] ^ ctx->state[a], 32);
	ctx->state[c] = ctx->state[c] + ctx->state[d];
	ctx->state[b] = blake2b_ROTR64(ctx->state[b] ^ ctx->state[c], 24);
	ctx->state[a] = ctx->state[a] + ctx->state[b] + m2;
	ctx->state[d] = blake2b_ROTR64(ctx->state[d] ^ ctx->state[a], 16);
	ctx->state[c] = ctx->state[c] + ctx->state[d];
	ctx->state[b] = blake2b_ROTR64(ctx->state[b] ^ ctx->state[c], 63);
}

