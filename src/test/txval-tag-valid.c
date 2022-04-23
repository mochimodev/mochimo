
#include <stdlib.h>
#include "_assert.h"

#define EXCLUDE_NODES
#include "../tag.c"

int main()
{
   FILE *fp;
   LENTRY le[2];
   TXQENTRY txqe;
   word64 zero = 0;
   word64 trigger = RTRIGGER31;
   word8 src_wots[TXADDRLEN], dst_wots[TXADDRLEN], chg_wots[TXADDRLEN];
   word8 src_tag[TXADDRLEN], dst_tag[TXADDRLEN], chg_tag[TXADDRLEN];

   /* init _wots, _tag, le and txqe */
   memset(src_wots, 0, TXADDRLEN);
   memset(dst_wots, 1, TXADDRLEN);
   memset(chg_wots, 2, TXADDRLEN);
   memset(src_tag, 3, TXADDRLEN);
   memset(dst_tag, 4, TXADDRLEN);
   memset(chg_tag, 5, TXADDRLEN);
   memset(ADDR_TAG_PTR(src_wots), 0x42, ADDR_TAG_LEN);
   memset(ADDR_TAG_PTR(dst_wots), 0x42, ADDR_TAG_LEN);
   memset(ADDR_TAG_PTR(chg_wots), 0x42, ADDR_TAG_LEN);
   memset(ADDR_TAG_PTR(src_tag), 0, ADDR_TAG_LEN);
   memset(ADDR_TAG_PTR(dst_tag), 0, ADDR_TAG_LEN);
   memset(ADDR_TAG_PTR(chg_tag), 0, ADDR_TAG_LEN);
   strcpy((char *) ADDR_TAG_PTR(src_tag), "123");
   strcpy((char *) ADDR_TAG_PTR(dst_tag), "456");
   strcpy((char *) ADDR_TAG_PTR(chg_tag), "789");
   memcpy(le[0].addr, dst_tag, TXADDRLEN);
   memcpy(le[1].addr, chg_tag, TXADDRLEN);
   memcpy(txqe.chg_addr, chg_tag, TXADDRLEN);

   /* test conditions where tag_valid()= 1 */
   ASSERT_EQ_MSG(tag_valid(src_wots, chg_wots, dst_tag, NULL), 1,
      "expect ecode=1, for src to dst_tag NOT in ledger; called from txval");
   ASSERT_EQ_MSG(tag_valid(src_wots, chg_wots, dst_tag, &trigger), 1,
      "expect ecode=1, for src to dst_tag NOT in ledger; called from bval, "
      "ON/AFTER block 17185");
   /* test conditions where tag_valid()= 0, first VEOK */
   ASSERT_EQ_MSG(tag_valid(src_wots, chg_wots, dst_wots, &zero), 0,
      "expect ecode=0, for plain wots transaction; called from bval, "
      "BEFORE block 17185");
   ASSERT_EQ_MSG(tag_valid(src_wots, chg_wots, dst_wots, NULL), 0,
      "expect ecode=0, for plain wots transaction; called from txval");
   ASSERT_EQ_MSG(tag_valid(src_wots, chg_wots, dst_wots, &trigger), 0,
      "expect ecode=0, for plain wots transaction; called from bval, "
      "ON/AFTER block 17185");

   /* write ledger containing dst_tag and chg_tag */
   ASSERT_NE_MSG((fp = fopen("ledger.dat", "wb")), NULL, "expect ledger.dat");
   ASSERT_EQ_MSG(fwrite(le, sizeof(LENTRY), 2, fp), 2,
      "expect write 2 ledger entries to ledger.dat");
   fclose(fp); le_open("ledger.dat", "rb");

   /* test conditions where tag_valid()= 0, first VEOK (CONTINUED) */
   ASSERT_EQ_MSG(tag_valid(src_wots, chg_wots, dst_tag, NULL), 0,
      "expect ecode=0, for src to dst_tag in ledger; called from txval");
   ASSERT_EQ_MSG(tag_valid(src_wots, chg_wots, dst_tag, &trigger), 0,
      "expect ecode=0, for src to dst_tag in ledger; called from bval, "
      "ON/AFTER block 17185");
   /* test conditions where tag_valid()= 0, second VEOK */
   ASSERT_EQ_MSG(tag_valid(src_tag, src_tag, dst_wots, NULL), 0,
      "expect ecode=0, for src_tag to dst, where src_tag == chg_tag; "
      "called from txval");
   /* test conditions where tag_valid()= 2 */
   ASSERT_EQ_MSG(tag_valid(src_tag, chg_tag, dst_wots, NULL), 2,
      "expect ecode=2, for src_tag to dst, where src_tag != chg_tag; "
      "called from txval");
   /* test conditions where tag_valid()= 3 */
   ASSERT_EQ_MSG(tag_valid(src_wots, chg_tag, dst_wots, NULL), 3,
      "expect ecode=3, for src to dst to chg_tag in Tagidx; called from txval");

   /* ledger and Tagidx is no longer required */
   remove("ledger.dat");
   tag_free();

   /* write txclean.dat containing chg_tag */
   ASSERT_NE_MSG((fp = fopen("txclean.dat", "wb")), NULL, "expect txclean.dat");
   ASSERT_EQ_MSG(fwrite(&txqe, 2 * sizeof(TXQENTRY), 1, fp), 1,
      "expect write transaction queue entry to txclean.dat");
   fclose(fp);

   /* test conditions where tag_valid()= 4 */
   ASSERT_EQ_MSG(tag_valid(src_wots, chg_tag, dst_wots, NULL), 4,
      "expect ecode=4, for src to dst to chg_tag in txclean.dat; "
      "called from txval");
   /* test conditions where tag_valid()= 0, final VEOK */
   ASSERT_EQ_MSG(tag_valid(src_wots, chg_tag, dst_wots, &zero), 0,
      "expect ecode=4, for src to dst to chg_tag; called from bval");

   /* cleanup */
   remove("txclean.dat");
}
