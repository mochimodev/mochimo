
#include "_assert.h"
#include "sha256.h"
#include "types.h"
#include "sort.c"

#include <stdio.h>
#include <time.h>

#define LISTLEN 1000

TXQENTRY TXS[LISTLEN];

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
   FILE *fp;
   word8 seed[64] = { 0 };
   int i, compare;

   /* init seed and message */
   time((time_t *) seed);
   rndbytes((word8 *) TXS, sizeof(TXS), seed);

   /* write TXS to file */
   fp = fopen("txclean.dat", "wb");
   ASSERT_NE(fp, NULL);
   ASSERT_EQ(fwrite(TXS, sizeof(TXQENTRY), LISTLEN, fp), LISTLEN);
   fclose(fp);

   /* sort (from) file */
   ASSERT_EQ_MSG(sorttx("txclean.dat"), VEOK,
      "sorttx() should succeed under normal conditions");

   /* global pointers/values should be set */
   ASSERT_NE_MSG(Tx_ids, NULL, "Tx_ids should not be NULL");
   ASSERT_NE_MSG(Txidx, NULL, "Txidx should not be NULL");
   ASSERT_EQ_MSG(Ntx, LISTLEN, "Number of transactions should be LISTLEN");

   /* check sorted-ness */
   for (i = 0; i < LISTLEN - 1; i++) {
      compare = memcmp(&Tx_ids[Txidx[i] * HASHLEN],
         &Tx_ids[Txidx[i + 1] * HASHLEN], HASHLEN);
      ASSERT_LT_MSG(compare, 1, "preceding list items should compare < 1");
   }

   /* append to file, invalid size */
   fp = fopen("txclean.dat", "ab");
   ASSERT_NE(fp, NULL);
   ASSERT_EQ(fwrite(seed, sizeof(seed), 1, fp), 1);
   fclose(fp);

   /* sort file - VERROR */
   ASSERT_EQ_MSG(sorttx("txclean.dat"), VERROR,
      "sorttx() should FAIL on invalid size");

   remove("txclean.dat");

   /* sort file - VERROR */
   ASSERT_EQ_MSG(sorttx("txclean.dat"), VERROR,
      "sorttx() should FAIL on missing file");
}
