
#include "_assert.h"
#include "network.h"

char *Corephosts[] = {
   "usw-node.mochimap.com",
   "use-node.mochimap.com",
   "usc-node.mochimap.com",
   "sgp-node.mochimap.com",
   "deu-node.mochimap.com",
   ""
};
char **hostsp = Corephosts;

int main()
{
   int status = VERROR;
   NODE node;

   sock_startup();  /* enable socket support */
   /* try communicate with an invalid nodes */
   ASSERT_NE(get_tx(&node, aton("example.com"), OP_GET_IPL), VEOK);
   /* try communicate with one of the valid mochimap nodes */
   do {
      status = get_tx(&node, aton(*(hostsp++)), OP_GET_IPL);
   } while(status && **hostsp);
   ASSERT_EQ_MSG(status, VEOK, "failed to communicate with mochimap nodes");
   sock_cleanup();
}
