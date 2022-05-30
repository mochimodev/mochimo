
#include <stdio.h>
#include <stdlib.h>
#include "extprint.h"
#include "_assert.h"
#include "ledger.h"

#define LEDGER "ledger.dat"
#define LEDGER_BAD "ledger.bad.dat"
#define LEDGER_EMPTY "ledger.empty.dat"

int main()
{
   FILE *fp;
   LENTRY le;

   set_print_level(0);

   /* fill dummy ledger entry with data */
   strcpy((char *) le.addr, "abcdef");
   strcpy((char *) ADDR_TAG_PTR(le.addr), "123456");
   memset(le.balance, 1, 6);

   /* create ledgers for test */
   ASSERT_NE((fp = fopen(LEDGER, "wb")), NULL);
   ASSERT_EQ(fwrite(&le, sizeof(le), 1, fp), 1);
   fclose(fp);
   ASSERT_NE((fp = fopen(LEDGER_BAD, "wb")), NULL);
   ASSERT_EQ(fwrite(&le, sizeof(le), 1, fp), 1);
   ASSERT_EQ(fwrite(le.addr, sizeof(le.addr), 1, fp), 1);
   fclose(fp);
   ASSERT_NE((fp = fopen(LEDGER_EMPTY, "wb")), NULL);
   fclose(fp);

   /* check le_open() VEOK */
   ASSERT_EQ_MSG(le_open(LEDGER, "rb"), VEOK, "initial le_open failed");
   ASSERT_EQ_MSG(le_open(LEDGER, "rb"), VEOK, "successive le_open failed");
   le_close();

   /* check le_open() VERROR */
   ASSERT_EQ_MSG(le_open(LEDGER_BAD, "rb"), VERROR,
      "le_open didn't fail on malformed ledger");
   ASSERT_EQ_MSG(le_open(LEDGER_EMPTY, "rb"), VERROR,
      "le_open didn't fail on empty ledger");

   /* remove all ledger files */
   remove(LEDGER);
   remove(LEDGER_BAD);
   remove(LEDGER_EMPTY);

   /* check le_open VERRO (missing) */
   ASSERT_EQ_MSG(le_open(LEDGER, "rb"), VERROR,
      "le_open didn't fail on missing ledger");
}
