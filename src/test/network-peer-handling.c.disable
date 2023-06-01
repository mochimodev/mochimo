
#include "_assert.h"
#include "network.h"
#include <stdlib.h>

#include "_testutils.h"

int main()
{  /* check init_peers() fills both Tplist[] and Rplist[] */
   word32 peer = 0;

   while(*hostsp && peer == 0) {
      peer = aton(*hostsp++);
   }
   ASSERT_NE_MSG(peer, 0, "peer should have a value for test to work");

   /* test include32() */
   ASSERT_EQ_MSG(Rplistidx, 0, "Rplistidx should initialize 0");
   ASSERT_EQ_MSG(include32(1, Rplist, RPLISTLEN, NULL), 0,
      "List index NULL should return Zero (0)");
   ASSERT_EQ_MSG(include32(0, Rplist, RPLISTLEN, &Rplistidx), 0,
      "Only non-zero values should be included in lists");
   ASSERT_EQ_MSG(include32(peer, Rplist, RPLISTLEN, &Rplistidx), peer,
      "Should return peer value on success");
   ASSERT_EQ_MSG(Rplistidx, 1, "Rplistidx should increment");
   ASSERT_NE_MSG(Rplist[0], 0, "Rplist[] should have a peer");
   /* test remove32() */
   ASSERT_EQ_MSG(0, remove32(1, Rplist, RPLISTLEN, &Rplistidx),
      "Should return 0 if peer is not in list");
   ASSERT_EQ_MSG(peer, remove32(peer, Rplist, RPLISTLEN, &Rplistidx),
      "Should return peer val on success");
   ASSERT_EQ_MSG(Rplistidx, 0, "Rplistidx should decrement");
   ASSERT_EQ_MSG(Rplist[0], 0, "Rplist[] should have be empty");
}
