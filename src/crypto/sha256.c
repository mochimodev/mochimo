/* sha256b.c  Implements SHA2-256 algorithm
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * sha256.c is based on public domain code by Brad Conte
 * (brad AT bradconte.com).
 * https://raw.githubusercontent.com/B-Con/crypto-algorithms/master/sha256.c
 *
 * Algorithm specification can be found here:
 * http://csrc.nist.gov/publications/fips/fips180-2/fips180-2withchangenotice.pdf
 * This implementation uses little endian byte order.
 *
 * Date: 5 January 2018
 *
*/

#include <stdlib.h>
#include <string.h>
#include "../config.h"  /* to check for LONG64 */
#include "sha256.h"

/* LOCAL MACROS */
#define ROTLEFT(a,b) (((a) << (b)) | ((a) >> (32-(b))))
#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))

#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x,2) ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))
#define EP1(x) (ROTRIGHT(x,6) ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))
#define SIG0(x) (ROTRIGHT(x,7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))

/**************************** VARIABLES *****************************/
static const word32 k[64] = {
  0x428a2f98L,0x71374491L,0xb5c0fbcfL,0xe9b5dba5L,0x3956c25bL,0x59f111f1L,
  0x923f82a4L,0xab1c5ed5L,
  0xd807aa98L,0x12835b01L,0x243185beL,0x550c7dc3L,0x72be5d74L,0x80deb1feL,
  0x9bdc06a7L,0xc19bf174L,
  0xe49b69c1L,0xefbe4786L,0x0fc19dc6L,0x240ca1ccL,0x2de92c6fL,0x4a7484aaL,
  0x5cb0a9dcL,0x76f988daL,
  0x983e5152L,0xa831c66dL,0xb00327c8L,0xbf597fc7L,0xc6e00bf3L,0xd5a79147L,
  0x06ca6351L,0x14292967L,
  0x27b70a85L,0x2e1b2138L,0x4d2c6dfcL,0x53380d13L,0x650a7354L,0x766a0abbL,
  0x81c2c92eL,0x92722c85L,
  0xa2bfe8a1L,0xa81a664bL,0xc24b8b70L,0xc76c51a3L,0xd192e819L,0xd6990624L,
  0xf40e3585L,0x106aa070L,
  0x19a4c116L,0x1e376c08L,0x2748774cL,0x34b0bcb5L,0x391c0cb3L,0x4ed8aa4aL,
  0x5b9cca4fL,0x682e6ff3L,
  0x748f82eeL,0x78a5636fL,0x84c87814L,0x8cc70208L,0x90befffaL,0xa4506cebL,
  0xbef9a3f7L,0xc67178f2L
};


void sha256_transform(SHA256_CTX *ctx, const byte data[])
{
   word32 a, b, c, d, e, f, g, h, t1, t2, m[64];
   int i, j;

   for(i = j = 0; i < 16; ++i, j += 4)
      m[i] = ((word32) data[j] << 24) | ((word32) data[j + 1] << 16)
              | ((word32) data[j + 2] << 8) | ((word32) data[j + 3]);
      for( ; i < 64; ++i)
         m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];

	a = ctx->state[0];
	b = ctx->state[1];
	c = ctx->state[2];
	d = ctx->state[3];
	e = ctx->state[4];
	f = ctx->state[5];
	g = ctx->state[6];
	h = ctx->state[7];

   for (i = 0; i < 64; ++i) {
	t1 = h + EP1(e) + CH(e,f,g) + k[i] + m[i];
	t2 = EP0(a) + MAJ(a,b,c);
	h = g;
	g = f;
	f = e;
	e = d + t1;
	d = c;
	c = b;
	b = a;
	a = t1 + t2;
   }

	ctx->state[0] += a;
	ctx->state[1] += b;
	ctx->state[2] += c;
	ctx->state[3] += d;
	ctx->state[4] += e;
	ctx->state[5] += f;
	ctx->state[6] += g;
	ctx->state[7] += h;
}


void sha256_init(SHA256_CTX *ctx)
{
	ctx->datalen = 0;
#ifdef LONG64
	ctx->bitlen = 0;
#else
        ctx->bitlen = ctx->bitlen2 = 0;
#endif
	ctx->state[0] = 0x6a09e667L;
	ctx->state[1] = 0xbb67ae85L;
	ctx->state[2] = 0x3c6ef372L;
	ctx->state[3] = 0xa54ff53aL;
	ctx->state[4] = 0x510e527fL;
	ctx->state[5] = 0x9b05688cL;
	ctx->state[6] = 0x1f83d9abL;
	ctx->state[7] = 0x5be0cd19L;
}


/* data[] is less than 64k bytes in length on 16-bit machines. */
void sha256_update(SHA256_CTX *ctx, const byte data[], unsigned len)
{
   unsigned i;
   word32 old;

	for(i = 0; i < len; ++i) {
		ctx->data[ctx->datalen] = data[i];
		ctx->datalen++;
		if(ctx->datalen == 64) {
			sha256_transform(ctx, ctx->data);
#ifdef LONG64
			ctx->bitlen += 512;
#else
            old = ctx->bitlen;
            ctx->bitlen += 512;
            if(ctx->bitlen < old) ctx->bitlen2++;  /* add in carry */
#endif
			ctx->datalen = 0;
		}
	}
}


void sha256_final(SHA256_CTX *ctx, byte hash[])
{
   unsigned i, j;
   word32 old;

	i = ctx->datalen;

	/* Pad whatever data is left in the buffer. */
	if(ctx->datalen < 56) {
		ctx->data[i++] = 0x80;
		while (i < 56)
			ctx->data[i++] = 0x00;
	}
	else {
		ctx->data[i++] = 0x80;
		while(i < 64)
			ctx->data[i++] = 0x00;
		sha256_transform(ctx, ctx->data);
		memset(ctx->data, 0, 56);
	}

#ifdef LONG64
	ctx->bitlen += ctx->datalen * 8;
#else
    old = ctx->bitlen;
    ctx->bitlen += ctx->datalen * 8;
    if(ctx->bitlen < old) ctx->bitlen2++;  /* add in carry */
#endif
	ctx->data[63] = ctx->bitlen;
	ctx->data[62] = ctx->bitlen >> 8;
	ctx->data[61] = ctx->bitlen >> 16;
	ctx->data[60] = ctx->bitlen >> 24;
#ifndef LONG64
	ctx->data[59] = ctx->bitlen2;
	ctx->data[58] = ctx->bitlen2 >> 8;
	ctx->data[57] = ctx->bitlen2 >> 16;
	ctx->data[56] = ctx->bitlen2 >> 24;
#else
	ctx->data[59] = ctx->bitlen >> 32;  /* on 64-bit machines */
	ctx->data[58] = ctx->bitlen >> 40;
	ctx->data[57] = ctx->bitlen >> 48;
	ctx->data[56] = ctx->bitlen >> 56;
#endif
	sha256_transform(ctx, ctx->data);

    /* Since this implementation uses little endian byte ordering and 
     * SHA uses big endian, reverse all the bytes when copying the final
     * state to the output hash.
     */
   for(i = j = 0; i < 4; ++i, j += 8) {
	hash[i]      = (ctx->state[0] >> (24 - j));
	hash[i + 4]  = (ctx->state[1] >> (24 - j));
	hash[i + 8]  = (ctx->state[2] >> (24 - j));
	hash[i + 12] = (ctx->state[3] >> (24 - j));
	hash[i + 16] = (ctx->state[4] >> (24 - j));
	hash[i + 20] = (ctx->state[5] >> (24 - j));
	hash[i + 24] = (ctx->state[6] >> (24 - j));
	hash[i + 28] = (ctx->state[7] >> (24 - j));
   }
   memset(ctx, 0, sizeof(SHA256_CTX));  /* security */
}


void sha256(const byte *in, int inlen, byte *hashout)
{
   SHA256_CTX ctx;

   sha256_init(&ctx);
   sha256_update(&ctx, in, inlen);
   sha256_final(&ctx, hashout);
}
