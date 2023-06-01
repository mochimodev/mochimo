
#include "_assert.h"
#include "_testutils.h"
#include "protocol.h"
#include "error.h"

int main()
{
   NODE node = { 0 };
   word32 ip;
   char **cpp;

   sock_startup();

   for (cpp = Corephosts; (ip = aton(*cpp)); cpp++) {
      for (
         node_init(&node, INVALID_SOCKET, ip, PORT1, OP_GET_IPL, NULL);
         node_request_operation(&node) == VEWAITING;
         millisleep(1)
      );
      node_cleanup(&node);
      if (node.status == VEOK) break;
   }

   ASSERT_EQ_MSG(node.status, VEOK, "no successful request attempts");
}
