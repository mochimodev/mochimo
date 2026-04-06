/**
 * @file peer-provisional.c
 * @brief Unit tests for provisional peer management.
 *
 * Tests addprovisional(), harvest_provisional(), source reputation,
 * thread lifecycle, capacity limits, and deduplication.
 */

#include "_assert.h"
#include "peer.h"
#include "global.h"
#include "extthrd.h"
#include "exttime.h"
#include "extlib.h"

#include <stdio.h>
#include <string.h>
#include <time.h>

int main()
{
   word32 i;
   int rc, promoted;

   /* init globals needed by peer functions */
   memset(Rplist, 0, sizeof(Rplist));
   Rplistidx = 0;
   Running = 1;
   Dstport = 2095;
   srand16(12345, 67890, 11111);

   /* ---- basic add and dedup ---- */

   purge_provisional();

   rc = addprovisional(0x01020304, 0x0A0B0C0D);
   ASSERT_EQ(rc, 0);

   rc = addprovisional(0x05060708, 0x0A0B0C0D);
   ASSERT_EQ(rc, 0);

   /* duplicate should be deduplicated (returns 0) */
   rc = addprovisional(0x01020304, 0x0A0B0C0D);
   ASSERT_EQ(rc, 0);

   /* zero IP should be silently dropped */
   rc = addprovisional(0, 0x0A0B0C0D);
   ASSERT_EQ(rc, 0);

   /* ---- dedup against Rplist ---- */

   purge_provisional();
   memset(Rplist, 0, sizeof(Rplist));
   Rplistidx = 0;
   addrecent(0xAABBCCDD);

   /* IP already in Rplist should be dropped */
   rc = addprovisional(0xAABBCCDD, 0x01010101);
   ASSERT_EQ(rc, 0);

   /* different IP should succeed */
   rc = addprovisional(0x11223344, 0x01010101);
   ASSERT_EQ(rc, 0);

   memset(Rplist, 0, sizeof(Rplist));
   Rplistidx = 0;

   /* ---- harvest with no verified entries ---- */

   purge_provisional();
   addprovisional(0x01010101, 0xAAAAAAAA);
   addprovisional(0x02020202, 0xAAAAAAAA);
   addprovisional(0x03030303, 0xAAAAAAAA);

   promoted = harvest_provisional();
   ASSERT_EQ(promoted, 0);

   /* Rplist should still be empty */
   ASSERT_EQ_MSG(search32(0x01010101, Rplist, RPLISTLEN), NULL,
      "peer should not be in Rplist before verification");

   /* ---- capacity limit ---- */

   purge_provisional();
   for (i = 0; i < PROVPEERSLEN; i++) {
      rc = addprovisional(i + 1, 0xAAAAAAAA);
      ASSERT_EQ(rc, 0);
   }

   /* next add should fail */
   rc = addprovisional(PROVPEERSLEN + 1, 0xAAAAAAAA);
   ASSERT_EQ(rc, -1);

   /* ---- purge clears everything ---- */

   purge_provisional();
   rc = addprovisional(0x11111111, 0xAAAAAAAA);
   ASSERT_EQ(rc, 0);

   /* ---- multiple sources with cross-source dedup ---- */

   purge_provisional();
   for (i = 1; i <= 10; i++) {
      addprovisional(0x0A000000 | i, 0xAAAA0001);
   }
   for (i = 1; i <= 10; i++) {
      addprovisional(0x0B000000 | i, 0xBBBB0001);
   }

   /* cross-source duplicate (same IP, different source) */
   rc = addprovisional(0x0A000001, 0xCCCC0001);
   ASSERT_EQ(rc, 0);

   /* unique IP from third source */
   rc = addprovisional(0x0C000001, 0xCCCC0001);
   ASSERT_EQ(rc, 0);

   /* ---- harvest compaction ---- */

   purge_provisional();
   for (i = 1; i <= 10; i++) {
      addprovisional(i, 0xAAAAAAAA);
   }
   promoted = harvest_provisional();
   ASSERT_EQ(promoted, 0);

   /* list should still accept new entries */
   rc = addprovisional(0xFF000001, 0xBBBBBBBB);
   ASSERT_EQ(rc, 0);

   /* ---- rapid add/harvest cycles ---- */

   purge_provisional();
   memset(Rplist, 0, sizeof(Rplist));
   Rplistidx = 0;
   for (i = 0; i < 100; i++) {
      addprovisional((word32)(i + 1), 0xAAAAAAAA);
      if (i % 10 == 9) harvest_provisional();
   }
   memset(Rplist, 0, sizeof(Rplist));
   Rplistidx = 0;

   /* ---- thread start and stop lifecycle ---- */

   purge_provisional();
   rc = start_provisional_verifier();
   ASSERT_EQ(rc, 0);

   /* double start should return 0 (already running) */
   rc = start_provisional_verifier();
   ASSERT_EQ(rc, 0);

   millisleep(500);
   stop_provisional_verifier();

   /* double stop should be safe */
   stop_provisional_verifier();

   /* ---- cleanup ---- */

   purge_provisional();
   memset(Rplist, 0, sizeof(Rplist));
   Rplistidx = 0;
   Running = 0;
}
