
#include <stdio.h>
#include <stdlib.h>
#include "extprint.h"
#include "_assert.h"
#include "ledger.h"

#include "_testutils.h"

#define LEDGER "ledger.dat"

int main()
{
   FILE *fp;
   LENTRY le;
   long position;
   word8 addr[TXADDRLEN];

   set_print_level(0);
   memset(addr, 1, TXADDRLEN);

   /* write dummy ledger entry/s to test ledger */
   ASSERT_NE((fp = fopen(LEDGER, "wb")), NULL);
   ASSERT_EQ(fwrite(ledgerdata, sizeof(ledgerdata), 1, fp), 1);
   fclose(fp);

   /* check ledger function called before le_open() */
   ASSERT_EQ_MSG(le_find(ledgerdata[2].addr, &le, NULL, TXADDRLEN), FALSE,
      "le_find didn't fail before le_open() was called");
   le_open(LEDGER, "rb");

   /* search for address that doesn't exist { 1, 1, 1, ... } */
   ASSERT_EQ_MSG(le_find(addr, &le, NULL, TXADDRLEN), FALSE,
      "le_find didn't fail on address that doesn't exist");

   ASSERT_EQ_MSG(le_find(ledgerdata[1].addr, &le, &position, TXADDRLEN), TRUE,
      "le_find failed to find ledgerdata[1]");
   ASSERT_EQ_MSG(position, 1, "le_find didn't return position 1");
   ASSERT_CMP_MSG(&ledgerdata[1], &le, sizeof(le),
      "le_find returned incorrect ledger entry");
   ASSERT_EQ_MSG(le_find(ledgerdata[3].addr, &le, &position, TXADDRLEN), TRUE,
      "le_find failed to find ledgerdata[3]");
   ASSERT_EQ_MSG(position, 3, "le_find didn't return position 3");
   ASSERT_CMP_MSG(&ledgerdata[3], &le, sizeof(le),
      "le_find returned incorrect ledger entry");
   ASSERT_EQ_MSG(le_find(ledgerdata[5].addr, &le, &position, TXADDRLEN), TRUE,
      "le_find failed to find ledgerdata[5]");
   ASSERT_EQ_MSG(position, 5, "le_find didn't return position 5");
   ASSERT_CMP_MSG(&ledgerdata[5], &le, sizeof(le),
      "le_find returned incorrect ledger entry");

   /* check partial address search */
   ASSERT_EQ_MSG(le_find(ledgerdata[4].addr, &le, &position, 2), TRUE,
      "le_find failed to find ledgerdata[4] with (partial)");
   ASSERT_EQ_MSG(position, 4, "le_find didn't return position 4 (partial)");
   ASSERT_CMP_MSG(&ledgerdata[4], &le, sizeof(le),
      "le_find returned incorrect ledger entry (partial)");
   /* ensure length 1 searches all */
   ASSERT_EQ_MSG(le_find(ledgerdata[7].addr, &le, &position, 1), TRUE,
      "le_find failed to find ledgerdata[7] (len=1)");
   ASSERT_EQ_MSG(position, 7, "le_find didn't return position 7 (len=1)");
   ASSERT_CMP_MSG(&ledgerdata[7], &le, sizeof(le),
      "le_find returned incorrect ledger entry (len=1)");

   /* cleanup */
   le_close();
   remove(LEDGER);
}
