
#include "wots.h"

#include "extmath.h"
#include "sha256.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

#define TXADDRLEN 64
#define TXWOTSLEN 2208
#define ITERATIONS 1000

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
   word8 val[WOTSSIGBYTES];
   word8 sig[WOTSSIGBYTES];
   word8 sig_msg[SHA256LEN];
   word8 secret[SHA256LEN];
   word8 address[TXWOTSLEN];
   word8 hashed[TXADDRLEN];
   clock_t genclks, sigclks, valclks, valw2aclks;
   clock_t start, end;
   int i;

   /* Public WOTS+ address subpointers */
   word8 *PKp = address;
   word8 *PSp = PKp + WOTSSIGBYTES;
   word8 *PAp = PSp + SHA256LEN;

   /* init seed and message */
   genclks = sigclks = valclks = valw2aclks = 0;
   time((time_t *) seed);

   for (i = 0; i < ITERATIONS; i++) {

      start = clock();
      /* create address */
      rndbytes(secret, 32, seed);
      rndbytes(PSp, SHA256LEN, seed);
      rndbytes(PAp, SHA256LEN, seed);
      /* addr is modified by wots_pkgen() */
      memcpy(ADRS, PAp, SHA256LEN);
      /* generate a good addr */
      wots_pkgen(PKp, secret, PSp, ADRS);
      memcpy(PAp, ADRS, SHA256LEN);  /* default tag */
      end = clock();

      genclks += end - start;

      start = clock();
      /* generate signature */
      sha256(PKp, WOTSSIGBYTES, sig_msg);
      memcpy(ADRS, PAp, SHA256LEN);
      wots_sign(sig, sig_msg, secret, PSp, ADRS);
      end = clock();

      sigclks += end - start;

      start = clock();
      /* generate public key from signature */
      memcpy(ADRS, PAp, SHA256LEN);
      wots_pk_from_sig(val, sig, sig_msg, PSp, ADRS);
      end = clock();

      valclks += end - start;
      valw2aclks += end - start;

      start = clock();
      /* copy to hashed */
      sha256(val, WOTSSIGBYTES, hashed);
      memcpy(val + SHA256LEN, PAp, SHA256LEN);
      end = clock();

      valw2aclks += end - start;
   }

   printf("\n%dx WOTS+ address generations:\n", ITERATIONS);
   printf("- Total Clocks: %zu (%lfs)\n", genclks,
      (float) genclks / CLOCKS_PER_SEC);
   printf("- Clocks / gen: %zu (%lfs)\n", genclks / ITERATIONS,
      (float) (genclks / ITERATIONS) / CLOCKS_PER_SEC);
   printf("- Est. Capable: %.02lf /s /core\n",
      1.0f / ((float) (genclks / ITERATIONS) / CLOCKS_PER_SEC));

   printf("\n%dx WOTS+ signature generation:\n", ITERATIONS);
   printf("- Total Clocks: %zu (%lfs)\n", sigclks,
      (float) sigclks / CLOCKS_PER_SEC);
   printf("- Clocks / sig: %zu (%lfs)\n", sigclks / ITERATIONS,
      (float) (sigclks / ITERATIONS) / CLOCKS_PER_SEC);
   printf("- Est. Capable: %.02lf /s /core\n",
      1.0f / ((float) (sigclks / ITERATIONS) / CLOCKS_PER_SEC));

   printf("\n%dx WOTS+ signature validation:\n", ITERATIONS);
   printf("- Total Clocks: %zu (%lfs)\n", valclks,
      (float) valclks / CLOCKS_PER_SEC);
   printf("- Clocks / val: %zu (%lfs)\n", valclks / ITERATIONS,
      (float) (valclks / ITERATIONS) / CLOCKS_PER_SEC);
   printf("- Est. Capable: %.02lf /s /core\n",
      1.0f / ((float) (valclks / ITERATIONS) / CLOCKS_PER_SEC));

   printf("\n%dx WOTS+ signature validation (hashed):\n", ITERATIONS);
   printf("- Total Clocks: %zu (%lfs)\n", valw2aclks,
      (float) valw2aclks / CLOCKS_PER_SEC);
   printf("- Clocks / val: %zu (%lfs)\n", valw2aclks / ITERATIONS,
      (float) (valw2aclks / ITERATIONS) / CLOCKS_PER_SEC);
   printf("- Est. Capable: %.02lf /s /core\n",
      1.0f / ((float) (valw2aclks / ITERATIONS) / CLOCKS_PER_SEC));
}