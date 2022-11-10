
#include "wots.h"
#include "ledger.h"
#include "error.h"
#include "extlib.h"
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

int random_neogen(char *fname, LENTRY *le, size_t count)
{
   FILE *fp;
   NGHEADER ngh;
   word32 hdrlen;
   long long lbytes;
   BTRAILER bt = { 0 };

   /* init data */
   hdrlen = sizeof(ngh);
   lbytes = (sizeof(*le) * count);
   put32(ngh.hdrlen, hdrlen);
   put64(ngh.lbytes, &lbytes);

   /* write data */
   if ((fp = fopen(fname, "wb")) == NULL) goto FAIL_IO;
   if (fwrite(&ngh, sizeof(ngh), 1, fp) != 1) goto FAIL_IO;
   if (fwrite(le, sizeof(*le), count, fp) != count) goto FAIL_IO;
   if (fwrite(&bt, sizeof(bt), 1, fp) != 1) goto FAIL_IO;

   /* cleanup */
   fclose(fp);

   return 0;

FAIL_IO:
   if (fp) fclose(fp);
   return 1;
}  /* end random_neogenw() */

void compare_remaining(LENTRY *lep, size_t count)
{
   LENTRY *lef;
   size_t i, counted;

   /* try find all ledger entries by address and tag */
   for (counted = i = 0; i < count; i++) {
      lef = le_find(lep[i].addr);
      if (lef) {
         ASSERT_LT_MSG(get32(lef->balance), get32(lep[i].balance),
            "remaining balance should be less than original");
         counted++;
      }
   }
   ASSERT_LT_MSG(counted, count, "remaining entries should be less");
}

void search_and_compare(LENTRY *lep, size_t count)
{
   LENTRY *lef;
   word8 *tag;
   size_t i;

   /* try find all ledger entries by address and tag */
   for (i = 0; i < count; i++) {
      lef = le_find(lep[i].addr);
      if (lef == NULL) perrno(errno, "le_find()");
      ASSERT_NE_MSG(lef, NULL, "selected ledger entry not found");
      ASSERT_CMP_MSG(lef->addr, lep[i].addr, sizeof(*lef), "entry mismatch");
      if (ADDR_HAS_TAG(&lep[i])) {
         tag = ADDR_TAGp(&lep[i]);
         lef = tag_find(tag);
         ASSERT_NE_MSG(lef, NULL, "selected (tag) ledger entry not found");
         ASSERT_CMP_MSG(lef->addr, lep[i].addr, sizeof(*lef),
            "(tag) entry mismatch");
      }
   }
}

int main()
{
   word32 mfee[2] = { MFEE, 0 };
   NGHEADER ngh;
   LENTRY le, prev_le, *lep;
   TAGIDX ti, prev_ti;
   FILE *fp;
   size_t count, i;
   word32 hdrlen;
   long long lbytes;
   int ecode;

   memset(&le, 0, sizeof(le));

   /* check initial failure modes for some functions */
   set_errno(0);
   ASSERT_NE(auto_compression_depth(), 0);
   ASSERT_EQ(errno, EMCMLENOTAVAIL);
   set_errno(0);
   ASSERT_NE(le_append("dummy.file.name", NULL), 0);
   ASSERT_NE(le_append(NULL, NULL), 0);
   ASSERT_EQ(errno, EINVAL);
   set_errno(0);
   ASSERT_NE(le_compress(NULL, 0, 0), 0);
   ASSERT_EQ(errno, EINVAL);
   set_errno(0);
   ASSERT_EQ(le_find(NULL), NULL);
   ASSERT_EQ(errno, EINVAL);
   set_errno(0);
   ASSERT_EQ(le_find(NULL), NULL);
   ASSERT_EQ(errno, EINVAL);
   set_errno(0);
   ASSERT_EQ(tag_find(NULL), NULL);
   ASSERT_EQ(errno, EINVAL);
   set_errno(0);
   ASSERT_EQ(le_find(le.addr), NULL);
   ASSERT_EQ(errno, EMCMLENOTAVAIL);
   set_errno(0);
   ASSERT_EQ(tag_find(WOTS_TAGp(le.addr)), NULL);
   ASSERT_EQ(errno, EMCMLENOTAVAIL);
   /* write bad info */
   ASSERT_NE((fp = fopen("ledger.dat", "wb")), NULL);
   ASSERT_EQ(fwrite(&le, sizeof(le), 1, fp), 1);
   fclose(fp);
   ASSERT_NE(le_extract("ledger.dat"), 0);
   ASSERT_EQ(errno, EMCM_HDRLEN);
   remove("ledger.dat");
   ASSERT_NE((fp = fopen("neogen.dat", "wb")), NULL);
   hdrlen = sizeof(ngh);
   lbytes = (sizeof(le) * 2);
   put32(ngh.hdrlen, hdrlen);
   put64(ngh.lbytes, &lbytes);
   ASSERT_EQ(fwrite(&ngh, sizeof(ngh), 1, fp), 1);
   ASSERT_EQ(fwrite(&le, sizeof(le), 1, fp), 1);
   ASSERT_EQ(fwrite(&le, sizeof(le), 1, fp), 1);
   fclose(fp);
   ASSERT_NE(le_extract("neogen.dat"), 0);
   ASSERT_EQ(errno, EMCM_LE_SORT);
   remove("neogen.dat");

   /* init many address, with some tag */
   count = 1234;
   lep = random_ledger(count);
   ASSERT_NE_MSG(lep, NULL, "random ledgerw creation failed");
   ASSERT_EQ_MSG(random_neogen("neogen.dat", lep, count), 0,
      "random neogenesis creation failed");

   /* begin extract and find test */
   ecode = le_extract("neogen.dat");
   if (ecode) perrno(errno, "le_extract() FAILURE");
   ASSERT_EQ_MSG(ecode, VEOK, "le_extract() failed");

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

   ASSERT_EQ_MSG(i, count, "ledger entry count mismatch");

   /* check sort of extracted tag index */
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

   /* perform ledger renewal */
   Sanctuary_opt = 0x100;
   ASSERT_EQ_MSG(le_renew(mfee), VEOK, "failed to renew ledger");

   /* reduce and compare remaining balances */
   compare_remaining(lep, count);

   free(lep);

   /* do auto compressed renewal -- add a depth */
   lep = random_ledger(123);
   ASSERT_NE_MSG(lep, NULL, "random ledgerw creation failed");
   ASSERT_EQ_MSG(write_ledger("ledger.update", lep, 123), 0, "rng failed");
   ASSERT_EQ_MSG(le_append("ledger.update", NULL), VEOK, "le_append() failed");
   ASSERT_EQ_MSG(le_renew(mfee), VEOK, "failed to renew ledger");
   ASSERT_EQ_MSG(fexists("ledger.dat.1"), 0,
      "ledger depth 1 should NOT exist");
   free(lep);

   /* create one address, no tags */
   count = 1;
   lep = random_ledger(count);
   ASSERT_NE_MSG(lep, NULL, "random ledgerw creation failed");
   ASSERT_EQ_MSG(random_neogen("neogen.dat", lep, count), 0, "rng failed");
   ASSERT_EQ_MSG(le_extract("neogen.dat"), VEOK, "le_extract() failed");

   /* do searching */
   search_and_compare(lep, count);

   free(lep);

   /* add LEDEPTHMAX ledgers to trigger auto compression */
   for ( ; count < LEDEPTHMAX; count++) {
      lep = random_ledger(count);
      ASSERT_NE_MSG(lep, NULL, "random ledgerw creation failed");
      ASSERT_EQ_MSG(write_ledger("ledger.update", lep, count), 0, "rng failed");
      ASSERT_EQ_MSG(le_append("ledger.update", NULL), VEOK, "le_append() failed");
      free(lep);
   }
   /* ledger.dat.7 should exist */
   ASSERT_EQ_MSG(fexists("ledger.dat.7"), 1, "ledger depth 7 should exist");

   lep = random_ledger(count);
   ASSERT_NE_MSG(lep, NULL, "random ledgerw creation failed");
   ASSERT_EQ_MSG(write_ledger("ledger.update", lep, count), 0, "rng failed");
   ASSERT_EQ_MSG(le_append("ledger.update", NULL), VEOK, "le_append() failed");
   free(lep);

   /* ledger.dat.7 should no longer exist */
   ASSERT_EQ_MSG(fexists("ledger.dat.7"), 0,
      "ledger depth 7 should no longer exist");

   /* check close ledger closes the ledger but does not delete file */
   le_delete(1);
   le_close(0);
   ASSERT_EQ_MSG(fexists("ledger.dat.0"), 1, "ledger depth 0 should exist");
   ASSERT_EQ_MSG(le_find(lep), NULL, "le_find should fail");
   ASSERT_EQ_MSG(errno, EMCMLENOTAVAIL, "errno should indicate no ledger");

   /* reinstate ledger (rebuilds tag index) */
   remove("tagidx.dat.0");
   ASSERT_EQ(le_append("ledger.dat.0", NULL), 0);
   ASSERT_EQ_MSG(fexists("ledger.dat.0"), 1, "ledger depth 0 should exist");
   ASSERT_EQ_MSG(fexists("tagidx.dat.0"), 1, "tagidx depth 0 should exist");

   /* check delete ledger closes and deletes the ledger */
   le_delete(0);
   ASSERT_EQ_MSG(fexists("ledger.dat.0"), 0, "ledger should not exist");
   ASSERT_EQ_MSG(fexists("tagidx.dat.0"), 0, "tagidx should not exist");
   ASSERT_EQ_MSG(le_find(lep), NULL, "le_find should fail");
   ASSERT_EQ_MSG(errno, EMCMLENOTAVAIL, "errno should indicate no ledger");

   /* cleanup */
   remove("neogen.dat");
   remove("ledger.dat");
   remove("ledger.dat.0.tags");
   remove("ledger.dat.co");

   return 0;
}
