/* testtag.c  Test tag index functions

   See LICENSE.PDF

   Date: 15 December 2019
*/

#include "extmath.h"    /* 64-bit math support */
#include "extprint.h"   /* print/logging support */

#include "../config.h"
#include "../mochimo.h"
#include "../proto.h"

#include "../data.c"
#include "../crypto/crc16.c"
#include "../rand.c"
#include "../util.c"
#include "../ledger.c"
#define EXCLUDE_RESOLVE
#include "../tag.c"

/* kill mirror() children and grandchildren */
void stop_mirror(void)
{
   if(Mqpid) {
      if(Trace) plog("   Reaping mirror() zombies...");
      kill(Mqpid, SIGTERM);
      waitpid(Mqpid, NULL, 0);
      Mqpid = 0;
   }
}  /* end stop_mirror() */


int main()
{
   static LENTRY le;
   int status, j;

   Trace = 1;

   printf("sizeof(LENTRY) = %u\n", (int) sizeof(LENTRY));
   strcpy((char *) ADDR_TAG_PTR(le.addr), "badtag");
   printf("tag_find('%-12.12s')\n", (char *) ADDR_TAG_PTR(le.addr));
   status = tag_find(le.addr, le.addr, le.balance, ADDR_TAG_LEN);
   printf("tag_find() returned %d  S/B 1\n", status);

   printf("Tags:\n");
   for(j = 0; j < Ntagidx; j++) {
      printf("%d:  %-12.12s\n", j, (char *) &Tagidx[j * ADDR_TAG_LEN]);
   }

   memset(ADDR_TAG_PTR(le.addr), 0, ADDR_TAG_LEN);
   strcpy((char *) ADDR_TAG_PTR(le.addr), "123");
   printf("tag_find('%-12.12s')\n", (char *) ADDR_TAG_PTR(le.addr));
   status = tag_find(le.addr, le.addr, le.balance, ADDR_TAG_LEN);
   printf("tag_find() returned %d  S/B 0\n", status);
   printf("le.addr = %-20.20s...\n", le.addr);
   printf("le.balance = %u\n", get32(le.balance));

   tag_free();
   tag_free();
   printf("tag_find('%-12.12s')\n", (char *) ADDR_TAG_PTR(le.addr));
   status = tag_find(le.addr, le.addr, le.balance, ADDR_TAG_LEN);
   printf("tag_find() returned %d  S/B 0\n", status);
   tag_buildidx();

   return 0;
}
