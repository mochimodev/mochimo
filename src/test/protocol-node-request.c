
#include "_assert.h"
#include "_testutils.h"
#include "protocol.h"
#include "error.h"

int main()
{
   SNODE node = { 0 };
   char **cpp;

   sock_startup();
   set_print_level(PLEVEL_LOG);

   for (cpp = Corephosts; aton(*cpp); cpp++) {
      for (
         prep_request(&node, aton(*cpp), PORT1, OP_GET_IPL, NULL);
         node_request(&node) == VEWAITING;
         millisleep(1)
      );
      node_cleanup(&node);
      if (node.status == VEOK) break;
   }

   ASSERT_EQ_MSG(node.status, VEOK, "no successful request attempts");
}
