#include "keccak.h"

/*
*Digestbitlen must be 128 224 256 288 384 512
*/
void keccak_init(keccak_ctx_t *ctx, uint32_t digestbitlen) 
{
	memset(ctx, 0, sizeof(keccak_ctx_t));

	ctx->digestbitlen = digestbitlen;
	
	ctx->rate_bits = 1600 - ((ctx->digestbitlen) << 1);
	ctx->rate_bytes = ctx->rate_bits >> 3;
	
	ctx->absorb_round = ctx->rate_bits >> 6;

	ctx->bits_in_queue = 0;
}

/*
*Digestbitlen must be 224 256 384 512
*/
void keccak_sha3_init(keccak_ctx_t *ctx, uint32_t digestbitlen)
{
	keccak_init(ctx, digestbitlen);
	ctx->sha3_flag = 1;
}


void keccak_update(keccak_ctx_t *ctx, byte *in, uint64_t inlen) 
{
	int64_t bytes = ctx->bits_in_queue >> 3;
	int64_t count = 0;
	while (count < inlen) {
		if (bytes == 0 && count <= (int64_t)(inlen - ctx->rate_bytes)) {
			do {
				keccak_aborb(ctx, in + count);
				count += ctx->rate_bytes;
			} while (count <= (inlen - ctx->rate_bytes));
		}
		else {
			int64_t partial = MIN(ctx->rate_bytes - bytes, inlen - count);
			memcpy(ctx->q + bytes, in + count, partial);
			
			bytes += partial;
			count += partial;

			if (bytes == ctx->rate_bytes) {
				keccak_aborb(ctx, ctx->q);
				bytes = 0;
			}
		}
	}

	ctx->bits_in_queue = bytes << 3;
}


void keccak_final(keccak_ctx_t *ctx, byte *out)
{
	if (ctx->sha3_flag)
	{
		int mask = (1 << 2) - 1;
		ctx->q[ctx->bits_in_queue >> 3] = (byte)(0x02 & mask);
		ctx->bits_in_queue += 2;
	}

	keccak_pad(ctx);
	uint64_t i = 0;
	
	while (i < ctx->digestbitlen) {
		if (ctx->bits_in_queue == 0) {
			keccak_permutations(ctx);
			keccak_extract(ctx);
			ctx->bits_in_queue = ctx->rate_bits;
		}

		uint64_t partialBlock = UMIN(ctx->bits_in_queue, ctx->digestbitlen - i);
		memcpy(out + (i >> 3), ctx->q + (ctx->rate_bytes - (ctx->bits_in_queue >> 3)), partialBlock >> 3);
		ctx->bits_in_queue -= partialBlock;
		i += partialBlock;
	}
}



void keccak_pad(keccak_ctx_t *ctx)
{
	ctx->q[ctx->bits_in_queue >> 3] |= (1L << (ctx->bits_in_queue & 7));

	if (++ctx->bits_in_queue == ctx->rate_bits) {
		keccak_aborb(ctx, ctx->q);
		ctx->bits_in_queue = 0;
		
	}
	
	uint64_t full = ctx->bits_in_queue >> 6;
	uint64_t partial = ctx->bits_in_queue & 63;

		uint64_t offset = 0;
		for (int i = 0; i < full; ++i) {
			ctx->state[i] ^= leuint64(ctx->q + offset);
			offset += 8;
		}

		uint64_t one = 1;
		if (partial > 0) {
			uint64_t mask = (one << partial) - 1;
			ctx->state[full] ^= leuint64(ctx->q + offset) & mask;
			one = 1;
		}

		ctx->state[(ctx->rate_bits - 1) >> 6] ^= 9223372036854775808ULL;/* 1 << 63 */
	

		keccak_permutations(ctx);


		keccak_extract(ctx);

	ctx->bits_in_queue = ctx->rate_bits;

}


int64_t MIN(int64_t a, int64_t b)
{
	if (a > b)
		return b;
	return a;
}

uint64_t UMIN(uint64_t a, uint64_t b)
{
	if (a > b)
		return b;
	return a;
}

uint64_t leuint64(void *in)
{
	uint64_t a;
	a = *((uint64_t *)in);
	return a;
	/*
#if defined(NATIVE_LITTLE_ENDIAN)
	uint64_t a;
	a = *((uint64_t *)in);
#else
	uint8_t *a = (uint8_t *)in;
	return ((uint64_t)(a[0]) << 0) | ((uint64_t)(a[1]) << 8) | ((uint64_t)(a[2]) << 16) | ((uint64_t)(a[3]) << 24) | ((uint64_t)(a[4]) << 32)
		| ((uint64_t)(a[5]) << 40) | ((uint64_t)(a[6]) << 48) | ((uint64_t)(a[7]) << 56);
#endif*/
}

void keccak_aborb(keccak_ctx_t *ctx, byte* in) {
	
	uint64_t offset = 0;
	for (uint64_t i = 0; i < ctx->absorb_round; ++i) {
		ctx->state[i] ^= leuint64(in + offset);
		offset += 8;
	}

	keccak_permutations(ctx);
}

void keccak_extract(keccak_ctx_t *ctx)
{
	uint64_t len = ctx->rate_bits >> 6;
	int64_t a;
	int s = sizeof(uint64_t);
	for (int i = 0;i < len;i++) {
		a = leuint64((int64_t*)&ctx->state[i]);
		//ctx->q[i]+ = ((uint64_t *)in);

		//ctx->q[i] = ;
		memcpy(ctx->q + (i * s), &a, s);
	}
}

void keccak_permutations(keccak_ctx_t * ctx) {
	
	int64_t* A = ctx->state;;
	
	int64_t *a00 = A, *a01 = A + 1, *a02 = A + 2, *a03 = A + 3, *a04 = A + 4;
	int64_t *a05 = A + 5, *a06 = A + 6, *a07 = A + 7, *a08 = A + 8, *a09 = A + 9;
	int64_t *a10 = A + 10, *a11 = A + 11, *a12 = A + 12, *a13 = A + 13, *a14 = A + 14;
	int64_t *a15 = A + 15, *a16 = A + 16, *a17 = A + 17, *a18 = A + 18, *a19 = A + 19;
	int64_t *a20 = A + 20, *a21 = A + 21, *a22 = A + 22, *a23 = A + 23, *a24 = A + 24;

	for (int i = 0; i < KECCAK_ROUND; i++) {
		
		/* Theta */
		int64_t c0 = *a00 ^ *a05 ^ *a10 ^ *a15 ^ *a20;
		int64_t c1 = *a01 ^ *a06 ^ *a11 ^ *a16 ^ *a21;
		int64_t c2 = *a02 ^ *a07 ^ *a12 ^ *a17 ^ *a22;
		int64_t c3 = *a03 ^ *a08 ^ *a13 ^ *a18 ^ *a23;
		int64_t c4 = *a04 ^ *a09 ^ *a14 ^ *a19 ^ *a24;

		int64_t d1 = ROTL64b(c1, 1) ^ c4;
		int64_t d2 = ROTL64b(c2, 1) ^ c0;
		int64_t d3 = ROTL64b(c3, 1) ^ c1;
		int64_t d4 = ROTL64b(c4, 1) ^ c2;
		int64_t d0 = ROTL64b(c0, 1) ^ c3;

		*a00 ^= d1;
		*a05 ^= d1;
		*a10 ^= d1;
		*a15 ^= d1;
		*a20 ^= d1;
		*a01 ^= d2;
		*a06 ^= d2;
		*a11 ^= d2;
		*a16 ^= d2;
		*a21 ^= d2;
		*a02 ^= d3;
		*a07 ^= d3;
		*a12 ^= d3;
		*a17 ^= d3;
		*a22 ^= d3;
		*a03 ^= d4;
		*a08 ^= d4;
		*a13 ^= d4;
		*a18 ^= d4;
		*a23 ^= d4;
		*a04 ^= d0;
		*a09 ^= d0;
		*a14 ^= d0;
		*a19 ^= d0;
		*a24 ^= d0;


		/* Rho pi */
		c1 = ROTL64b(*a01, 1);
		*a01 = ROTL64b(*a06, 44);
		*a06 = ROTL64b(*a09, 20);
		*a09 = ROTL64b(*a22, 61);
		*a22 = ROTL64b(*a14, 39);
		*a14 = ROTL64b(*a20, 18);
		*a20 = ROTL64b(*a02, 62);
		*a02 = ROTL64b(*a12, 43);
		*a12 = ROTL64b(*a13, 25);
		*a13 = ROTL64b(*a19, 8);
		*a19= ROTL64b(*a23, 56);
		*a23 = ROTL64b(*a15, 41);
		*a15 = ROTL64b(*a04, 27);
		*a04 = ROTL64b(*a24, 14);
		*a24 = ROTL64b(*a21, 2);
		*a21 = ROTL64b(*a08, 55);
		*a08 = ROTL64b(*a16, 45);
		*a16 = ROTL64b(*a05, 36);
		*a05 = ROTL64b(*a03, 28);
		*a03 = ROTL64b(*a18, 21);
		*a18 = ROTL64b(*a17, 15);
		*a17 = ROTL64b(*a11, 10);
		*a11 = ROTL64b(*a07, 6);
		*a07 = ROTL64b(*a10, 3);
		*a10 = c1;

		/* Chi */
		c0 = *a00 ^ (~*a01 & *a02);
		c1 = *a01 ^ (~*a02 & *a03);
		*a02 ^= ~*a03 & *a04;
		*a03 ^= ~*a04 & *a00;
		*a04 ^= ~*a00 & *a01;
		*a00 = c0;
		*a01 = c1;

		c0 = *a05 ^ (~*a06 & *a07);
		c1 = *a06 ^ (~*a07 & *a08);
		*a07 ^= ~*a08 & *a09;
		*a08 ^= ~*a09 & *a05;
		*a09 ^= ~*a05 & *a06;
		*a05 = c0;
		*a06 = c1;

		c0 = *a10 ^ (~*a11 & *a12);
		c1 = *a11 ^ (~*a12 & *a13);
		*a12 ^= ~*a13 & *a14;
		*a13 ^= ~*a14 & *a10;
		*a14 ^= ~*a10 & *a11;
		*a10 = c0;
		*a11 = c1;

		c0 = *a15 ^ (~*a16 & *a17);
		c1 = *a16 ^ (~*a17 & *a18);
		*a17 ^= ~*a18 & *a19;
		*a18 ^= ~*a19 & *a15;
		*a19 ^= ~*a15 & *a16;
		*a15 = c0;
		*a16 = c1;

		c0 = *a20 ^ (~*a21 & *a22);
		c1 = *a21 ^ (~*a22 & *a23);
		*a22 ^= ~*a23 & *a24;
		*a23 ^= ~*a24 & *a20;
		*a24 ^= ~*a20 & *a21;
		*a20 = c0;
		*a21 = c1;

		/* Iota */
		*a00 ^= KECCAK_CONSTS[i];
	}
}

uint64_t ROTL64b(uint64_t a, uint64_t  b)
{
	return (a << b) | (a >> (64 - b));
}
