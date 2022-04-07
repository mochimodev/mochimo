
#include "_assert.h"
#include "sha256.h"
#include "wots.h"
#include <time.h>

#define TXADDRLEN 2208
#define ITERATIONS 100

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
   word8 pub_key2[WOTSSIGBYTES + 32 + 32] = { 0 };
   word8 pub_key[WOTSSIGBYTES + 32 + 32] = { 0 };
   word8 sig[WOTSSIGBYTES] = { 0 };
   word8 secret[32] = { 0 };
   word32 addr[8] = { 0 };
   word8 seed[64] = { 0 };
   word8 sig_msg[SHA256LEN] = { 0 };
   int i;

   /* init seed and message */
   time((time_t *) seed);
   rndbytes(sig_msg, SHA256LEN, seed);

   for (i = 0; i < ITERATIONS; i++) {
      /* create address */
      rndbytes(secret, 32, seed);
      rndbytes(pub_key, TXADDRLEN, seed);
      /* addr is modified by wots_pkgen() */
      memcpy(addr, &pub_key[WOTSSIGBYTES + 32], 32);
      /* generate a good addr */
      wots_pkgen(pub_key, secret, &pub_key[WOTSSIGBYTES], addr);
      memcpy(&pub_key[WOTSSIGBYTES + 32], addr, 32);  /* default tag */

      /* generate signature */
      sha256(pub_key, WOTSSIGBYTES, sig_msg);
      wots_sign(sig, sig_msg, secret, &pub_key[WOTSSIGBYTES], addr);

      /* generate public key from signature */
      wots_pk_from_sig(pub_key2, sig, sig_msg, &pub_key[WOTSSIGBYTES], addr);

      ASSERT_CMP_MSG(pub_key, pub_key2, WOTSSIGBYTES,
         "Derived public key should match initial public key");
   }
}