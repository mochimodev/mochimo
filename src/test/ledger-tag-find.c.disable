/* testtag.c  Test tag index functions

   See LICENSE.PDF

   Date: 15 December 2019
*/

#include <stdlib.h>
#include "_assert.h"

#define EXCLUDE_NODES
#include "../tag.c"

#define GOODADDRSTR  "goodaddr"
#define GOODTAGSTR   "goodtag"
#define BADTAGSTR    "badtag"

void make_Tagidx(int withN)
{
   int n = withN + 1;
   size_t size = (size_t) (n * ADDR_TAG_LEN);
   Ntagidx = (word32) n;
   Tagidx = (word8 *) malloc(size);
   memset(Tagidx, 0, size);
   strcpy((char *) &Tagidx[withN * ADDR_TAG_LEN], GOODTAGSTR);
}

int main()
{
   FILE *fp;
   LENTRY le[6];

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
   ASSERT_EQ(tag_find(le[0].addr, NULL, NULL, ADDR_TAG_LEN), VERROR);

   /* clear LENTRY and search for tag GOODTAGSTR, expecting tag found (VEOK) */
   ASSERT_EQ(tag_find(le[4].addr, le[4].addr, le[4].balance, ADDR_TAG_LEN), 0);
   /* check partial search of 7 bytes, expecting tag found (VEOK) */
   ASSERT_EQ(tag_find(le[4].addr, le[4].addr, le[4].balance, 7), 0);
   /* check partial search of 4 bytes, expecting tag found (VEOK) */
   ASSERT_EQ(tag_find(le[4].addr, le[4].addr, le[4].balance, 4), 0);
   /* check tag_find() built Tagidx & Ntagidx as well*/
   ASSERT_NE_MSG(Tagidx, NULL, "expect (Tagidx != NULL) when tag_find()= 0");
   ASSERT_GT_MSG(Ntagidx, 0, "expect (Ntagidx > 0) when tag_find()= 0");
   /* check tag address is as expected */
   ASSERT_STR_MSG((char *) le[4].addr, GOODADDRSTR, strlen(GOODADDRSTR),
      "should be = " GOODADDRSTR);
   /* check tag balance is as expected */
   ASSERT_EQ_MSG(get32(le[4].balance), 16, "should be = 16");
   /* check tag_free() free's Tagidx & resets Ntagidx */
   tag_free();
   ASSERT_EQ_MSG(Tagidx, NULL, "expect (Tagidx == NULL) when tag_free()= 0");
   ASSERT_EQ_MSG(Ntagidx, 0, "expect (Ntagidx == 0) when tag_free()= 0");

   /* check tag_find() fails on IO error, ecode= 4 */
   make_Tagidx(8);
   ASSERT_EQ(tag_find(le[4].addr, le[4].addr, NULL, ADDR_TAG_LEN), 4);
   ASSERT_EQ_MSG(Tagidx, NULL, "expect (Tagidx == NULL) when tag_find()= 4");
   ASSERT_EQ_MSG(Ntagidx, 0, "expect (Ntagidx == 0) when tag_find()= 4");
   /* check tag_find() fails on bad fread, ecode= 5 */
   make_Tagidx(3);
   ASSERT_EQ(tag_find(le[4].addr, le[4].addr, NULL, ADDR_TAG_LEN), 5);
   ASSERT_EQ_MSG(Tagidx, NULL, "expect (Tagidx == NULL) when tag_find()= 5");
   ASSERT_EQ_MSG(Ntagidx, 0, "expect (Ntagidx == 0) when tag_find()= 5");

   /* check tag_find() fails with unable to build tagidx, ecode=2 */
   remove("ledger.dat");
   ASSERT_EQ(tag_find(le[4].addr, NULL, NULL, ADDR_TAG_LEN), 2);
   ASSERT_EQ_MSG(Tagidx, NULL, "expect (Tagidx == NULL) when tag_find()= 2");
   ASSERT_EQ_MSG(Ntagidx, 0, "expect (Ntagidx == 0) when tag_find()= 2");
   /* check tag_find() fails with unable to open ledger, ecode= 3 */
   make_Tagidx(1);
   ASSERT_EQ(tag_find(le[4].addr, le[4].addr, NULL, ADDR_TAG_LEN), 3);
   ASSERT_EQ_MSG(Tagidx, NULL, "expect (Tagidx == NULL) when tag_find()= 3");
   ASSERT_EQ_MSG(Ntagidx, 0, "expect (Ntagidx == 0) when tag_find()= 3");

   /* cleanup */
   remove("ledger.dat");
}
