
#include <stdlib.h>
#include <stdio.h>
#include "_assert.h"
#include "extprint.h"
#include "extlib.h"
#include "types.h"
#include "tag.h"
#include "ledger.h"

int main()
{
   FILE *fp;
   LENTRY le[2];
   TXQENTRY txqe;
   word8 zero[8] = { 0 };
   word8 trigger[8] = { 0 };
   word8 src_wots[TXADDRLEN], dst_wots[TXADDRLEN], chg_wots[TXADDRLEN];
   word8 src_tag[TXADDRLEN], dst_tag[TXADDRLEN], chg_tag[TXADDRLEN];

   put32(trigger, RTRIGGER31);

   set_print_level(0);

   /* init _wots, _tag, le and txqe */
   memset(src_wots, 0, TXADDRLEN);
   memset(dst_wots, 1, TXADDRLEN);
   memset(chg_wots, 2, TXADDRLEN);
   memset(src_tag, 3, TXADDRLEN);
   memset(dst_tag, 4, TXADDRLEN);
   memset(chg_tag, 5, TXADDRLEN);
   memset(ADDR_TAG_PTR(src_wots), 0x42, TXTAGLEN);
   memset(ADDR_TAG_PTR(dst_wots), 0x42, TXTAGLEN);
   memset(ADDR_TAG_PTR(chg_wots), 0x42, TXTAGLEN);
   memset(ADDR_TAG_PTR(src_tag), 0, TXTAGLEN);
   memset(ADDR_TAG_PTR(dst_tag), 0, TXTAGLEN);
   memset(ADDR_TAG_PTR(chg_tag), 0, TXTAGLEN);
   strcpy((char *) ADDR_TAG_PTR(src_tag), "123");
   strcpy((char *) ADDR_TAG_PTR(dst_tag), "456");
   strcpy((char *) ADDR_TAG_PTR(chg_tag), "789");
   memcpy(le[0].addr, dst_tag, TXADDRLEN);
   memcpy(le[1].addr, chg_tag, TXADDRLEN);
   memcpy(txqe.chg_addr, chg_tag, TXADDRLEN);

   /* test conditions without ledger */
   ASSERT_EQ_MSG(tag_valid(src_wots, chg_wots, dst_wots, NULL), 0,
      "expected VEOK for wots transaction; called from txval()");
   ASSERT_EQ_MSG(tag_valid(src_wots, chg_wots, dst_wots, zero), 0,
      "expected VEOK for wots transaction; called from bval(<17185)");
   ASSERT_EQ_MSG(tag_valid(src_wots, chg_wots, dst_wots, trigger), 0,
      "expected VEOK for wots transaction; called from bval(>=17185)");
   ASSERT_EQ_MSG(tag_valid(src_wots, chg_wots, dst_tag, NULL), VERROR,
      "expected VERROR for dst_tag NOT in ledger; from txval()");
   ASSERT_EQ_MSG(tag_valid(src_wots, chg_wots, dst_tag, trigger), VERROR,
      "expected VERROR for dst_tag NOT in ledger; from bval(>=17185)");

   /* write ledger containing dst_tag and chg_tag */
   ASSERT_NE_MSG((fp = fopen("ledger.dat", "wb")), NULL, "no ledger.dat");
   ASSERT_EQ_MSG(fwrite(le, sizeof(LENTRY), 2, fp), 2,
      "expected fwrite()=2 ledger entries to ledger.dat");
   fclose(fp); le_open("ledger.dat", "rb");

   /* test conditions with ledger */
   ASSERT_EQ_MSG(tag_valid(src_wots, chg_wots, dst_tag, NULL), VEOK,
      "expected VEOK dst_tag in ledger; called from txval");
   ASSERT_EQ_MSG(tag_valid(src_wots, chg_wots, dst_tag, trigger), VEOK,
      "expected VEOK dst_tag in ledger; called from bval(>=17185)");
   ASSERT_EQ_MSG(tag_valid(src_tag, src_tag, dst_wots, NULL), VEOK,
      "expected VEOK for src_tag==chg_tag to dst_wots; from txval()");
   ASSERT_EQ_MSG(tag_valid(src_tag, chg_tag, dst_wots, NULL), VERROR,
      "expected VERROR for src_tag!=chg_tag to dst_wots; from txval()");
   ASSERT_EQ_MSG(tag_valid(src_wots, chg_tag, dst_wots, NULL), VERROR,
      "expected VERROR for create chg_tag(already exists); from txval()");

   /* ledger and Tagidx is no longer required */
   remove("ledger.dat");
   tag_free();

   /* write txclean.dat containing chg_tag */
   ASSERT_NE_MSG((fp = fopen("txclean.dat", "wb")), NULL, "no txclean.dat");
   ASSERT_EQ_MSG(fwrite(&txqe, 2 * sizeof(TXQENTRY), 1, fp), 1,
      "expected fwrite()=1 transaction queue entry to txclean.dat");
   fclose(fp);

   /* test conditions with txclean */
   ASSERT_EQ_MSG(tag_valid(src_wots, chg_tag, dst_wots, NULL), VERROR,
      "expected VERROR for duplicate chg_tag funding; from txval()");
   ASSERT_EQ_MSG(tag_valid(src_wots, chg_tag, dst_wots, zero), 0,
      "expected VEOK for change_tag funding; from bval(<=17185)");

   /* cleanup */
   remove("txclean.dat");
}
