
#include "_assert.h"
#include "extmath.h"
#include "network.h"

word8 Running = 1;

int main()
{  /* check scan_nettwork() returns non-Zero parameters */
   word32 qplist[RPLISTLEN] = { 0 };
   word8 hash[HASHLEN] = { 0 };
   word8 weight[32] = { 0 };
   word8 bnum[8] = { 0 };

   sock_startup();  /* enable socket support */

   /* initialize peers and scan network */
   init_peers();
   ASSERT_NE_MSG(Rplistidx, 0, "Recent peers must have been filled");
   ASSERT_GT_MSG(scan_network(NULL, 0, NULL, NULL, NULL), 0,
      "scan_network() should return greater than 0 consensus peers");
   ASSERT_GT_MSG(scan_network(qplist, RPLISTLEN, hash, weight, bnum), 0,
      "scan_network() should return greater than 0 quorum peers");
   ASSERT_EQ_MSG(iszero(hash, 32), 0, "hash should not be Zero");
   ASSERT_EQ_MSG(iszero(weight, 32), 0, "weight should not be Zero");
   ASSERT_EQ_MSG(iszero(bnum, 8), 0, "bnum should not be Zero");
   sock_cleanup();
}
