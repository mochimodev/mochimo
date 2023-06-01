
#include "wots.h"

#include "extmath.h"
#include "sha256.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

#define TXADDRLEN  64
#define TXWOTSLEN 2208
#define ITERATIONS 1000

word8 match[TXADDRLEN] = { 0xd1, 0x60, 0 };

/* Copy outlen random bytes to out. 64-byte seed is incremented. */
void rndbytes(word8 *out, word32 outlen, word8 *seed)
{
   static word8 state;
   static word8 rnd[64];
   word8 hash[32];  /* output for sha256() */
   int n;

   if(state == 0) {
      memcpy(rnd, seed, 64);
      state = 1;
   }
   for( ; outlen; ) {
      /* increment big number in rnd and seed */
      for(n = 0; n < 64; n++) {
         if(++seed[n] != 0) break;
      }
      sha256(seed, 64, hash);
      if(outlen < 32) n = outlen; else n = 32;
      memcpy(out, hash, n);
      out += n;
      outlen -= n;
   }  /* end for outlen */
}  /* end rndbytes() */

int main()
{
   word32 ADRS[8] = { 0 };
   word8 seed[64] = { 0 };
   word8 secret[SHA256LEN];
   word8 address[TXWOTSLEN];
   word8 hashed[TXADDRLEN];
   time_t last;
   int i, j, best;

   /* Public WOTS+ address subpointers */
   word8 *PKp = address;
   word8 *PSp = PKp + WOTSSIGBYTES;
   word8 *PAp = PSp + SHA256LEN;

   /* init seed and message */
   time((time_t *) seed);
   best = 0;

   time(&last);
   for (i = 0; ; i++) {
      /* create address */
      rndbytes(secret, 32, seed);
      rndbytes(PSp, SHA256LEN, seed);
      rndbytes(PAp, SHA256LEN, seed);
      /* addr is modified by wots_pkgen() */
      memcpy(ADRS, PAp, SHA256LEN);
      /* generate a good addr */
      wots_pkgen(PKp, secret, PSp, ADRS);
      memcpy(PAp, ADRS, SHA256LEN);  /* default tag */
      /* convert to hashed address */
      sha256(PKp, WOTSSIGBYTES, hashed);
      memcpy(hashed + SHA256LEN, ADRS, SHA256LEN);  /* default tag */
      for (j = 0; j < SHA256LEN; j++) {
         if (hashed[j] != match[j]) break;
      }
      if (j > best) {
         best = j;
         for (j = 0; j < best; j++) {
            printf("%02x", hashed[j]);
         }
         printf("... %.02lf/s\n", (float) i / difftime(time(NULL), last));
         time(&last);
         i = 0;
      }
   }
}