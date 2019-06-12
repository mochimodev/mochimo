/* xo4.c  Crypto for shylock.c
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 17 February 2018
 *
*/

#include "hash/cpu/sha256.h"

/* --------  XO4 Cipher package  --------
 * Courtesy Patrick Cargill -- EYES ONLY!
*/

typedef struct {
   byte s[64];
   byte rnd[32];
   int j;
} XO4CTX;


/* Initialise Cipher XO4
 * Key is a random seed of length len <= 64 bytes.
 */
void xo4_init(XO4CTX *ctx, byte *key, int len)
{
   int i, j, len2;

   for(i = 0, j = 0, len2 = len; i < 64; i++) {
      ctx->s[i] = key[j++];
      if(--len2 == 0) { j = 0; len2 = len; }
   }
   ctx->j = 0;
}


/* Return a random number between 0 and 255 */
byte xo4_rand(XO4CTX *ctx)
{
   int n;
   byte b;

   if(ctx->j == 0) {
      /* increment big number in ctx->s[] */
      for(n = 0; n < 64; n++) {
         if(++(ctx->s[n]) != 0) break;
      }       
      sha256(ctx->s, 64, ctx->rnd);
   }
   b = ctx->rnd[ctx->j++];
   if(ctx->j >= 32) ctx->j = 0;
   return b;
}


void xo4_crypt(XO4CTX *ctx, void *input, void *output, int len)
{
   byte *in, *out;

   in = input;
   out = output;

   for(  ; len; len--)
      *out++ = *in++ ^ xo4_rand(ctx);
}

/*--------  End XO4 package  --------*/
