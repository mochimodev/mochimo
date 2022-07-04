
#include "_assert.h"
#include "network.h"

#include "_testutils.h"

int main()
{
   int status = VERROR;
   NODE node;

   Running = 1;
   sock_startup();  /* enable socket support */
   /* try communicate with an invalid nodes */
   ASSERT_NE(get_ipl(&node, aton("example.com")), VEOK);
   /* try communicate with one of the valid mochimap nodes */
   do {
      status = get_ipl(&node, aton(*(hostsp++)));
   } while(status && **hostsp);
   ASSERT_EQ_MSG(status, VEOK, "failed to communicate with mochimap nodes");
   sock_cleanup();
}
