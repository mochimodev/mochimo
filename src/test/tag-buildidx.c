
#include <stdlib.h>
#include "_assert.h"
#include "tag.c"

int main()
{
   FILE *fp;
   LENTRY le;

   set_print_level(0);

   /* write dummy ledger to disk */
   ASSERT_NE_MSG((fp = fopen("ledger.dat", "wb")), NULL, "expect dummy ledger");
   ASSERT_EQ_MSG(fwrite(&le, sizeof(LENTRY), 1, fp), 1, "expect ledger entry");
   fclose(fp);

   /* check tag_buildidx() returns VEOK if Tagidx has pointer */
   Tagidx = (word8 *) malloc(1);
   ASSERT_EQ_MSG(tag_buildidx(), 0, "expect tag_buildidx()= 0");
   /* tag_free() x2 to check Tagidx is NOT free()'d twice */
   tag_free(); tag_free();
   /* build Tagidx from ledger file, expecting success (VEOK) */
   ASSERT_EQ_MSG(tag_buildidx(), 0, "expect tag_buildidx()= 0");
   /* check repeat call to tag_buildidx(), expecting success (VEOK) */
   ASSERT_EQ_MSG(tag_buildidx(), 0, "expect repeat tag_buildidx()= 0");
   /* check tag_buildidx() built Tagidx & Ntagidx as well*/
   ASSERT_NE_MSG(Tagidx, NULL,
      "expect (Tagidx != NULL) when tag_buildidx()= 0");
   ASSERT_GT_MSG(Ntagidx, 0, "expect (Ntagidx > 0) when tag_buildidx()= 0");
   /* check tag_buildidx() fails to rebuild tagidx, ecode=1 */
   remove("ledger.dat");
   tag_free();
   ASSERT_EQ_MSG(tag_buildidx(), 1, "expect tag_buildidx() fail: no ledger");
   ASSERT_EQ_MSG(Tagidx, NULL,
      "expect (Tagidx == NULL) when tag_buildidx()= 1");
   ASSERT_EQ_MSG(Ntagidx, 0, "expect (Ntagidx == 0) when tag_buildidx()= 1");
   /* check tag_buildidx() fails on malloc(TOOMUCHMEMORY), ecode= 2...
    * no way to reliably test tag_buildidx()= 2 condition
   ASSERT_EQ_MSG(tag_buildidx(), 2, "expect tag_buildidx() fail: malloc()");
   ASSERT_EQ_MSG(Tagidx, NULL,
      "expect (Tagidx == NULL) when tag_buildidx()= 2");
   ASSERT_EQ_MSG(Ntagidx, 0, "expect (Ntagidx == 0) when tag_buildidx()= 2"); */
   /* check tag_buildidx() fails on IO error, ecode= 3...
    * no way to reliably test tag_buildidx()= 3 condition
   ASSERT_EQ_MSG(tag_buildidx(), 3, "expect tag_buildidx() fail: IO error");
   ASSERT_EQ_MSG(Tagidx, NULL,
      "expect (Tagidx == NULL) when tag_buildidx()= 3");
   ASSERT_EQ_MSG(Ntagidx, 0, "expect (Ntagidx == 0) when tag_buildidx()= 3"); */

   /* cleanup */
   remove("ledger.dat");
}
