
#include "_assert.h"
#include "network.h"
#include "extprint.h"

#include "_testutils.h"

int main()
{
   int status = VERROR;
   NODE node;

   sock_startup();  /* enable socket support */
   ASSERT_EQ_MSG(Dstport, 2095, "Dstport should default to port 2095");
   /* try communicate with invalid node on default port */
   ASSERT_NE(callserver(&node, aton("example.com")), VEOK);
   /* try communicate with invalid node on port 80 */
   Dstport = 80;
   ASSERT_NE(callserver(&node, aton("example.com")), VEOK);
   Dstport = 2095;  /* reset Dstport for remaining tests */
   /* try communicate with one of the valid mochimap nodes */
   do {
      status = callserver(&node, aton(*(hostsp++)));
   } while(status && **hostsp);
   ASSERT_EQ(status, VEOK);
   sock_cleanup();
}
