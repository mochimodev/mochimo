
#include <stdlib.h>
#include "_assert.h"

#define EXCLUDE_NODES
#include "../tag.c"

#define GOODADDRSTR  "goodaddr"
#define GOODTAGSTR   "goodtag"

int main()
{
   FILE *fp;
   LENTRY le;
   TX tx;

   /* init tx request */
   memset(&tx, 0, sizeof(tx));
   strcpy((char *) ADDR_TAG_PTR(tx.dst_addr), GOODTAGSTR);

   /* test conditions where tag_resolve()= 1 */
   ASSERT_EQ_MSG(tag_resolve(&tx), 1, "expect ecode=1, for failure to resolve");
   ASSERT_EQ_MSG(get32(tx.send_total), 0, "expect tag found flag to be false");
   ASSERT_EQ_MSG(get32(tx.change_total), 0, "expect balance to be zero 0");

   /* write ledger entry to test ledger */
   memset(&le, 0, sizeof(le));
   strcpy((char *) le.addr, GOODADDRSTR);
   strcpy((char *) ADDR_TAG_PTR(le.addr), GOODTAGSTR);
   put32(le.balance, 16);
   ASSERT_NE_MSG((fp = fopen("ledger.dat", "wb")), NULL, "expect ledger.dat");
   ASSERT_EQ_MSG(fwrite(&le, sizeof(le), 1, fp), 1,
      "expect write ledger entry to ledger.dat");
   fclose(fp);

   /* test conditions where tag_resolve()= 0 */
   ASSERT_EQ_MSG(tag_resolve(&tx), 0, "expect ecode=0, for resolve success");
   ASSERT_EQ_MSG(get32(tx.send_total), 1, "expect tag found flag to be true");
   ASSERT_EQ_MSG(get32(tx.change_total), 16,
      "expect resolved balance to be placed in change_total");
   ASSERT_STR_MSG((char *) tx.dst_addr, GOODADDRSTR, strlen(GOODADDRSTR),
      "expect resolved address to be placed in dst_addr");

   /* cleanup */
   remove("ledger.dat");
}
