
#include "_assert.h"
#include "extmath.h"
#include "peer.h"
#include "network.h"
#include <stdlib.h>

#include "_testutils.h"

int main()
{  /* check scan_nettwork() returns non-Zero parameters */
   word32 qplist[RPLISTLEN] = { 0 };
   word8 hash[HASHLEN] = { 0 };
   word8 weight[32] = { 0 };
   word8 bnum[8] = { 0 };

   Running = 1;
   sock_startup();  /* enable socket support */

   /* download starting peers (use corephosts as fallback) */
   http_get("https://mochimo.org/peers/start", "start.lst", STD_TIMEOUT);
   read_ipl("start.lst", Rplist, RPLISTLEN, &Rplistidx);
   while(*hostsp && aton(*hostsp)) {
      addpeer(aton(*hostsp++), Rplist, RPLISTLEN, &Rplistidx);
   }

   /* check peers and scan network */
   ASSERT_NE_MSG(Rplistidx, 0, "Recent peers must have been filled");
   ASSERT_GT_MSG(scan_network(NULL, 0, NULL, NULL, NULL), 0,
      "scan_network() should return greater than 0 consensus peers");
   ASSERT_GT_MSG(scan_network(qplist, RPLISTLEN, hash, weight, bnum), 0,
      "scan_network() should return greater than 0 quorum peers");
   ASSERT_EQ_MSG(iszero(hash, 32), 0, "hash should not be Zero");
   ASSERT_EQ_MSG(iszero(weight, 32), 0, "weight should not be Zero");
   ASSERT_EQ_MSG(iszero(bnum, 8), 0, "bnum should not be Zero");

   /* cleanup */
   remove("start.lst");
   sock_cleanup();
}
