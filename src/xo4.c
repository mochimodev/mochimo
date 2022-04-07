/**
 * @private
 * @headerfile xo4.h <xo4.h>
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_XO4_C
#define MOCHIMO_XO4_C


#include "xo4.h"
#include "sha256.h"

/**
 * Initialise Cipher XO4. Key is a random seed of length len <= 64 bytes.
 * @param ctx Pointer to XO4 context
 * @param key Pointer to (private) key used as seed
 * @param len Length of key input, in bytes
*/
void xo4_init(XO4_CTX *ctx, void *key, size_t len)
{
   int i, j, len2;

   for(i = 0, j = 0, len2 = len; i < 64; i++) {
      ctx->s[i] = ((word8 *) key)[j++];
      if(--len2 == 0) { j = 0; len2 = len; }
   }
   ctx->j = 0;
}  /* end xo4_init() */

/**
 * @private
 * Returns a random number between 0 and 255, based on XO4 context state.
 * @param ctx Pointer to XO4 context
*/
static word8 xo4_rand(XO4_CTX *ctx)
{
   int n;
   word8 b;

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
}  /* end xo4_rand() */

/**
 * Multipurpose XO4 encryption/decryption function.
 * @param ctx Pointer to XO4 context
 * @param input Pointer to input data
 * @param output Pointer to place output data in
 * @param len Length of data, in bytes
 * @note The XO4 context needs to be (re)initialized before decrypting
 * already encrypted data.
*/
void xo4_crypt(XO4_CTX *ctx, void *input, void *output, size_t len)
{
   word8 *in, *out;

   in = input;
   out = output;

   for(  ; len; len--) {
      *out++ = *in++ ^ xo4_rand(ctx);
   }
}  /* end xo4_crypt() */

/* end include guard */
#endif
