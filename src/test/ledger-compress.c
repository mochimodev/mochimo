
#include "wots.h"
#include "ledger.h"
#include "error.h"
#include "extlib.h"
#include "extmath.h"
#include <string.h>

#include "_assert.h"
#include "_testutils.h"

#define WOTS_SEEDp(addr) ((((word8 *) (addr)) + WOTSSIGBYTES))

LENTRY *random_ledger(size_t count)
{
   LENTRY_W lew;
   LENTRY *le;
   size_t i;
   word32 ADRS[8];
   word32 balance[2];
   word8 *tag;

   /* init */
   balance[1] = 0;
   le = calloc(count, sizeof(*le));

   /* generate WOTS+ addresses */
   for (i = 0; le && i < count; i++) {
      ADRS[0] = i;
      balance[0] = (i*i) + MFEE + 1;
      wots_pkgen(lew.addr, (word8 *) ADRS, WOTS_SEEDp(lew.addr), ADRS);
      memcpy(lew.addr + (TXWOTSLEN - HASHLEN), ADRS, HASHLEN);
      put64(lew.balance, balance);
      if (i && (i % 2) == 0) {
         tag = WOTS_TAGp(lew.addr);
         tag[0] = 0x01;
         *((word32 *) &tag[4]) = i;
      }
      /* convert to hashed lentry */
      le_convert(le[i].addr, lew.addr);
      put64(le[i].balance, lew.balance);
   }
   /* sort ledger entries (WOTS+) */
   if (le) qsort(le, count, sizeof(*le), le_cmp);

   return le;
}  /* end random_ledger() */

void modify_ledger(LENTRY *lep, size_t count, size_t mod)
{
   size_t i;

   /* modify ledger */
   for (i = 0; lep && i < count; i++) {
      if ((i % mod) == 0) {
         memset(lep[i].balance, 0, sizeof(lep[i].balance));
      } else lep[i].balance[0] += (word8) mod;
   }
}  /* end random_ledger() */

int write_ledger(char *fname, LENTRY *le, size_t count)
{
   FILE *fp;

   /* write data */
   if ((fp = fopen(fname, "wb")) == NULL) goto FAIL_IO;
   if (fwrite(le, sizeof(*le), count, fp) != count) goto FAIL_IO;
   /* cleanup */
   fclose(fp);

   return 0;

FAIL_IO:
   if (fp) fclose(fp);
   return 1;
}

void search_and_compare(LENTRY *lep, size_t count)
{
   LENTRY *lef;
   word8 *tag;
   size_t i;

   /* try find all ledger entries by address and tag */
   for (i = 0; i < count; i++) {
      lef = le_find(lep[i].addr);
      if (iszero(lep[i].balance, sizeof(lep[i].balance))) {
         ASSERT_EQ_MSG(lef, NULL, "LENTRY found on zero balance");
      } else {
         /* if (lef == NULL) perrno(errno, "le_find()"); */
         ASSERT_NE_MSG(lef, NULL, "LENTRY not returned on balance");
         ASSERT_CMP_MSG(lef->addr, lep[i].addr, sizeof(*lef), "addr mismatch");
      }
      if (ADDR_HAS_TAG(&lep[i])) {
         tag = ADDR_TAGp(&lep[i]);
         lef = tag_find(tag);
         if (iszero(lep[i].balance, sizeof(lep[i].balance))) {
            ASSERT_EQ_MSG(lef, NULL, "TAG found on zero balance");
         } else {
            /* if (lef == NULL) perrno(errno, "le_find()"); */
            ASSERT_NE_MSG(lef, NULL, "TAG not returned on balance");
            ASSERT_CMP_MSG(lef->addr, lep[i].addr, sizeof(*lef), "(tag) addr mismatch");
         }
      }
   }
}

void check_sort(int depth, size_t count)
{
   LENTRY le, prev_le;
   TAGIDX ti, prev_ti;
   FILE *fp;
   size_t i;
   char fname[FILENAME_MAX];

   /* check sort of ledger */
   snprintf(fname, FILENAME_MAX, "%s.%d", Lefname_opt, depth);
   fp = fopen(fname, "rb");
   ASSERT_NE_MSG(fp, NULL, "failed to open ledger");
   for (i = 0; ; i++) {
      if (fread(&le, sizeof(le), 1, fp) != 1) {
         ASSERT_EQ_MSG(ferror(fp), 0, "I/O error");
         break;
      }
      if (i != 0) {
         ASSERT_GE_MSG(le_cmpw(&le, &prev_le), 0, "bad ledger sort");
      }
      memcpy(&prev_le, &le, sizeof(le));
   }
   fclose(fp);

   ASSERT_EQ_MSG(i, count, "ledger entry count mismatch");

   /* check sort of tag index */
   snprintf(fname, FILENAME_MAX, "%s.%d", Tifname_opt, depth);
   fp = fopen(fname, "rb");
   ASSERT_NE_MSG(fp, NULL, "failed to open tagidx");
   for (i = 0; ; i++) {
      if (fread(&ti, sizeof(ti), 1, fp) != 1) {
         ASSERT_EQ_MSG(ferror(fp), 0, "I/O error");
         break;
      }
      if (i != 0) {
         ASSERT_GE_MSG(tag_cmp(&ti, &prev_ti), 0, "bad tag index sort");
      }
      memcpy(&prev_ti, &ti, sizeof(ti));
   }
   fclose(fp);

   ASSERT_LE_MSG(i, count, "iterated too many tags");
}

int main()
{
   LENTRY *lep;
   size_t count;

   /* init many address, with some tag */
   count = 1234;
   lep = random_ledger(count);
   ASSERT_NE_MSG(lep, NULL, "random ledgerw creation failed");
   ASSERT_EQ_MSG(write_ledger("ledger.dat", lep, count), 0, "rlg failed");
   ASSERT_EQ_MSG(le_append("ledger.dat", NULL), VEOK, "le_append() failed");

   modify_ledger(lep, count, 3);
   ASSERT_EQ_MSG(write_ledger("ledger.dat", lep, count), 0, "rlg failed");
   ASSERT_EQ_MSG(le_append("ledger.dat", NULL), VEOK, "le_append() failed");

   modify_ledger(lep, count, 7);
   ASSERT_EQ_MSG(write_ledger("ledger.dat", lep, count), 0, "rlg failed");
   ASSERT_EQ_MSG(le_append("ledger.dat", NULL), VEOK, "le_append() failed");

   /* compress and splice, depth > 0 */
   ASSERT_EQ_MSG(le_compress("ledger.co", 1, 2), VEOK, "le_compress fail");
   ASSERT_EQ_MSG(le_splice("ledger.co", 1, 2), VEOK, "le_splice fail");

   /* check sort of compressed ledger */
   check_sort(1, count);

   /* do searching */
   search_and_compare(lep, count);

   /* compress and splice, depth == 0 */
   ASSERT_EQ_MSG(le_compress("ledger.co", 0, 2), VEOK, "le_compress fail");
   ASSERT_EQ_MSG(le_splice("ledger.co", 0, 2), VEOK, "le_splice fail");


   /* check sort of compressed ledger */
   check_sort(0, count - ((count + 7 - 1) / 7));

   /* do searching */
   search_and_compare(lep, count);

   free(lep);
   le_delete(0);

   return 0;
}
