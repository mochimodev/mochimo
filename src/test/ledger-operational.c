
#include "ledger_.h"

void compare_remaining(LENTRY_W *lep, size_t count)
{
   LENTRY_W *lef;
   size_t i, counted;

   /* try find all ledger entries by address and tag */
   for (counted = i = 0; i < count; i++) {
      lef = le_find(&lep[i]);
      if (lef) {
         ASSERT_LT_MSG(get32(lef->balance), get32(lep[i].balance),
            "remaining balance should be less than original");
         counted++;
      }
   }
   ASSERT_LT_MSG(counted, count, "remaining entries should be less");
}

void search_and_compare(LENTRY_W *lep, size_t count)
{
   LENTRY_W *lef;
   word8 *tag;
   size_t i;

   /* try find all ledger entries by address and tag */
   for (i = 0; i < count; i++) {
      lef = le_find(&lep[i]);
      ASSERT_NE_MSG(lef, NULL, "selected ledger entry not found");
      ASSERT_CMP_MSG(lef, &lep[i], sizeof(*lep), "entry mismatch");
      if (WOTS_HAS_TAG(&lep[i])) {
         tag = WOTS_TAGp(&lep[i]);
         lef = tag_find(tag);
         ASSERT_NE_MSG(lef, NULL, "selected (tag) ledger entry not found");
         ASSERT_CMP_MSG(lef, &lep[i], sizeof(*lep), "(tag) entry mismatch");
      }
   }
}

int main()
{
   word32 mfee[2] = { MFEE, 0 };
   LENTRY_W *lep, le, prev_le;
   TAGIDX ti, prev_ti;
   FILE *fp;
   size_t count, i;
   int ecode;

   set_print_level(PLEVEL_LOG);

   /* init */
   count = 1234;
   lep = random_ledgerw(count);
   ASSERT_NE_MSG(lep, NULL, "random ledgerw creation failed");
   ASSERT_EQ_MSG(random_neogenw("neogen.dat", lep, count), 0,
      "random neogenesis creation failed");

   /* begin extract and find test */
   ASSERT_EQ_MSG(le_extractw("neogen.dat"), VEOK, "le_extractw() failed");

   /* check sort of extracted ledger */
   fp = fopen("ledger.dat.0", "rb");
   ASSERT_NE_MSG(fp, NULL, "failed to open ledger.dat.0");
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

   ASSERT_EQ_MSG(i, count, "not enough ledger entries");

   /* check sort of extracted ledger */
   fp = fopen("tagidx.dat.0", "rb");
   ASSERT_NE_MSG(fp, NULL, "failed to open tagidx.dat.0");
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

   /* do searching */
   search_and_compare(lep, count);

   /* adjust balances to immitate an update */
   for (i = 0; i < (count / 4); i++) lep[i].balance[2]++;
   write2file("ledger.update", lep, sizeof(*lep) * (count / 4));

   /* apply the update */
   ecode = le_update("ledger.update", count / 4);
   if (ecode) perrno(errno, "le_update() FAILURE");
   ASSERT_EQ_MSG(ecode, VEOK, "failed to update ledger");

   /* do searching again */
   search_and_compare(lep, count);

   /* adjust balances (again) to immitate a "compression" update */
   for (i = 0; i < (count / 2); i++) lep[i].balance[2]++;
   write2file("ledger.update", lep, sizeof(*lep) * (count / 2));

   /* apply the "compression" update */
   ecode = le_update("ledger.update", count / 2);
   if (ecode) perrno(errno, "le_update() FAILURE");
   ASSERT_EQ_MSG(ecode, VEOK, "failed to (compression) update ledger");

   /* compression update should consume all file depths */
   ASSERT_EQ_MSG(fexists("ledger.dat.1"), 0,
      "compression update should consume ledger depths");

   /* do searching again */
   search_and_compare(lep, count);

   /* perform ledger renewal */
   Sanctuary = 0x100;
   ecode = le_reneww(mfee);
   if (ecode) perrno(errno, "le_reneww() FAILURE");
   ASSERT_EQ_MSG(ecode, VEOK, "failed to renew ledger");

   /* reduce and compare remaining balances */
   compare_remaining(lep, count);

   /* apply non-compression update */
   for (i = 0; i < (count / 4); i++) lep[i].balance[2]++;
   write2file("ledger.update", lep, sizeof(*lep) * (count / 4));
   le_update("ledger.update", count / 4);

   /* perform ledger renewal (with compression) */
   Sanctuary = 0x10000;
   ecode = le_reneww(mfee);
   if (ecode) perrno(errno, "le_reneww() FAILURE");
   ASSERT_EQ_MSG(ecode, VEOK, "failed to renew ledger (with compression)");

   /* check close ledger closes the ledger but does not delete file */
   le_close(0);
   ASSERT_EQ_MSG(fexists("ledger.dat.0"), 1, "ledger depth 0 should exist");
   ASSERT_EQ_MSG(le_find(lep), NULL, "le_find should fail");
   ASSERT_EQ_MSG(errno, EMCMLENOTAVAIL, "errno should indicate no ledger");

   /* cleanup */
   remove("ledger.dat.0");
   remove("tagidx.dat.0");
   remove("neogen.dat");
   remove("ledger.dat");
   free(lep);

   return 0;
}
