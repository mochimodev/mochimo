
#include "_assert.h"
#include "sha256.h"
#include "types.h"
#include "sort.c"

#include <stdio.h>
#include <time.h>

#define LISTLEN 1000

LTRAN LTS[LISTLEN];

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

   set_print_level(0);

   /* init seed and message */
   time((time_t *) seed);
   rndbytes((word8 *) LTS, sizeof(LTS), seed);

   /* write LTS to file */
   fp = fopen("ltran.dat", "wb");
   ASSERT_NE(fp, NULL);
   ASSERT_EQ(fwrite(LTS, sizeof(LTRAN), LISTLEN, fp), LISTLEN);
   fclose(fp);

   /* sort file */
   ASSERT_EQ_MSG(sortlt("ltran.dat"), VEOK,
      "sortlt() should succeed under normal conditions");

   /* read file to LTS */
   fp = fopen("ltran.dat", "rb");
   ASSERT_NE(fp, NULL);
   ASSERT_EQ(fread(LTS, sizeof(LTRAN), LISTLEN, fp), LISTLEN);
   fclose(fp);

   /* check sorted-ness */
   for (i = 0; i < LISTLEN - 1; i++) {
      compare = memcmp(&LTS[i].addr, &LTS[i + 1].addr, TXWOTSLEN);
      compare = compare ? compare :
         memcmp(&LTS[i].trancode, &LTS[i + 1].trancode, 1);
      ASSERT_LT_MSG(compare, 1, "preceding list items should compare < 1");
   }

   /* append to file, invalid size */
   fp = fopen("ltran.dat", "ab");
   ASSERT_NE(fp, NULL);
   ASSERT_EQ(fwrite(seed, sizeof(seed), 1, fp), 1);
   fclose(fp);

   /* sort file - VERROR */
   ASSERT_EQ_MSG(sortlt("ltran.dat"), VERROR,
      "sortlt() should FAIL on invalid size");

   remove("ltran.dat");

   /* sort file - VERROR */
   ASSERT_EQ_MSG(sortlt("ltran.dat"), VERROR,
      "sortlt() should FAIL on missing file");
}
