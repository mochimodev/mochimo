
#include <stdlib.h>
#include "_assert.h"
#include "tag.c"

int main()
{
   FILE *fp;
   TXQENTRY txqe;
   LENTRY bad;

   /* init good and bad search tags */
   memset(ADDR_TAG_PTR(txqe.chg_addr), 0, TXTAGLEN);
   memset(ADDR_TAG_PTR(bad.addr), 0, TXTAGLEN);
   strcpy((char *) ADDR_TAG_PTR(txqe.chg_addr), "456");
   strcpy((char *) ADDR_TAG_PTR(bad.addr), "badtag");

   /* write dummy txclean.dat */
   ASSERT_NE_MSG((fp = fopen("txclean.dat", "wb")), NULL, "expect txclean.dat");
   ASSERT_EQ_MSG(fwrite(&txqe, sizeof(TXQENTRY), 1, fp), 1,
      "expect write TXQENTRY to txclean.dat");
   fclose(fp);
   /* search for tag "456" and "badtag" */
   ASSERT_EQ_MSG(tag_qfind(txqe.chg_addr), VEOK, "expect VEOK, tag found");
   ASSERT_EQ_MSG(tag_qfind(bad.addr), VERROR, "expect VERROR, tag not found");
   /* remove txclean.dat file for further tests */
   remove("txclean.dat");
   /* write dummy txq1.dat */
   ASSERT_NE_MSG((fp = fopen("txq1.dat", "wb")), NULL, "expect txq1.dat");
   ASSERT_EQ_MSG(fwrite(&txqe, sizeof(TXQENTRY), 1, fp), 1,
      "expect write TXQENTRY to txq1.dat");
   fclose(fp);
   /* search for tag "456" and "badtag" */
   ASSERT_EQ_MSG(tag_qfind(txqe.chg_addr), VEOK, "expect VEOK, tag found");
   ASSERT_EQ_MSG(tag_qfind(bad.addr), VERROR, "expect VERROR, tag not found");
   /* remove txq1.dat file for final test */
   remove("txq1.dat");
   /* search for tag "456", expecting tag not found (VERROR) */
   ASSERT_EQ_MSG(tag_qfind(bad.addr), VERROR,
      "expect VERROR, since neither queue files are present");
}
