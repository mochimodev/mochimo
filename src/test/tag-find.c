
#include <stdlib.h>
#include <stdio.h>
#include "_assert.h"
#include "extprint.h"
#include "extlib.h"
#include "types.h"
#include "tag.h"

#define GOODADDRSTR  "goodaddr"
#define GOODTAGSTR   "goodtag"
#define BADTAGSTR    "badtag"

void make_Tagidx(int withN)
{
   int n = withN + 1;
   size_t size = (size_t) (n * TXTAGLEN);
   Ntagidx = (word32) n;
   Tagidx = (word8 *) malloc(size);
   memset(Tagidx, 0, size);
   strcpy((char *) &Tagidx[withN * TXTAGLEN], GOODTAGSTR);
}

int main()
{
   FILE *fp;
   LENTRY le[6];

   set_print_level(0);

   /* write ledger entry/s to test ledger */
   memset(le, 0, sizeof(le));
   strcpy((char *) le[4].addr, GOODADDRSTR);
   strcpy((char *) ADDR_TAG_PTR(le[4].addr), GOODTAGSTR);
   put32(le[4].balance, 16);
   ASSERT_NE_MSG((fp = fopen("ledger.dat", "wb")), NULL, "expect ledger.dat");
   ASSERT_EQ_MSG(fwrite(le, sizeof(le), 1, fp), 1,
      "expect write all ledger entries to ledger.dat");
   fclose(fp); memset(&le[4].addr, 0, strlen(GOODADDRSTR));

   /* search for tag BADTAGSTR, expecting tag not found (VERROR) */
   strcpy((char *) ADDR_TAG_PTR(le[0].addr), BADTAGSTR);
   ASSERT_EQ(tag_find(le[0].addr, NULL, NULL, TXTAGLEN), VERROR);

   /* clear LENTRY and search for tag GOODTAGSTR, expecting tag found (VEOK) */
   ASSERT_EQ(tag_find(le[4].addr, le[4].addr, le[4].balance, TXTAGLEN), 0);
   /* check partial search of 7 bytes, expecting tag found (VEOK) */
   ASSERT_EQ(tag_find(le[4].addr, le[4].addr, le[4].balance, 7), 0);
   /* check partial search of 4 bytes, expecting tag found (VEOK) */
   ASSERT_EQ(tag_find(le[4].addr, le[4].addr, le[4].balance, 4), 0);
   /* check tag_find() built Tagidx & Ntagidx as well*/
   ASSERT_NE_MSG(Tagidx, NULL, "expect (Tagidx != NULL) after VEOK");
   ASSERT_GT_MSG(Ntagidx, 0, "expect (Ntagidx > 0) after VEOK");
   /* check tag address is as expected */
   ASSERT_STR_MSG((char *) le[4].addr, GOODADDRSTR, strlen(GOODADDRSTR),
      "should be = " GOODADDRSTR);
   /* check tag balance is as expected */
   ASSERT_EQ_MSG(get32(le[4].balance), 16, "should be = 16");
   /* check tag_free() free's Tagidx & resets Ntagidx */
   tag_free();
   ASSERT_EQ_MSG(Tagidx, NULL, "expect (Tagidx == NULL) after tag_free()");
   ASSERT_EQ_MSG(Ntagidx, 0, "expect (Ntagidx == 0) after tag_free()");

   /* check tag_find() fails on IO error */
   make_Tagidx(8);
   ASSERT_NE(tag_find(le[4].addr, le[4].addr, NULL, TXTAGLEN), VEOK);
   ASSERT_EQ_MSG(Tagidx, NULL, "expect (Tagidx == NULL) on I/O error");
   ASSERT_EQ_MSG(Ntagidx, 0, "expect (Ntagidx == 0) on I/O error");
   /* check tag_find() fails on bad fread */
   make_Tagidx(3);
   ASSERT_NE(tag_find(le[4].addr, le[4].addr, NULL, TXTAGLEN), VEOK);
   ASSERT_EQ_MSG(Tagidx, NULL, "expect (Tagidx == NULL) on bad fread");
   ASSERT_EQ_MSG(Ntagidx, 0, "expect (Ntagidx == 0) on bad fread");

   /* check tag_find() fails with unable to build tagidx */
   remove("ledger.dat");
   ASSERT_NE(tag_find(le[4].addr, NULL, NULL, TXTAGLEN), VEOK);
   ASSERT_EQ_MSG(Tagidx, NULL, "expect (Tagidx == NULL) on no ledger/Tagidx");
   ASSERT_EQ_MSG(Ntagidx, 0, "expect (Ntagidx == 0) on no ledger/Tagidx");
   /* check tag_find() fails with unable to open ledger */
   make_Tagidx(1);
   ASSERT_NE(tag_find(le[4].addr, le[4].addr, NULL, TXTAGLEN), VEOK);
   ASSERT_EQ_MSG(Tagidx, NULL, "expect (Tagidx == NULL) no ledger");
   ASSERT_EQ_MSG(Ntagidx, 0, "expect (Ntagidx == 0) no ledger");

   /* cleanup */
   remove("ledger.dat");
}
