
#include "_assert.h"
#include "network.h"

int main()
{  /* check init_peers() fills both Tplist[] and Rplist[] */

   /* initialize peers */
   init_peers();
   /* perform checks */
   ASSERT_NE_MSG(Rplist[0], 0, "Rplist[] should have at least one valid peer");
   save_ipl("coreip.lst", Rplist, RPLISTLEN);
   save_ipl("tplist.lst", Rplist, RPLISTLEN);
   save_ipl("rplist.lst", Rplist, RPLISTLEN);
   /* clear recent peers list */
   while(*Rplist) remove32(*Rplist, Rplist, RPLISTLEN, &Rplistidx);
   /* re-initialize peers */
   init_peers();
   /* all peer lists should contain peers  */
   ASSERT_NE_MSG(*Rplist, 0, "Rplist[] should have at least one valid peer");
   ASSERT_NE_MSG(*Tplist, 0, "Tplist[] should have at least one valid peer");
   /* cleanup */
   remove("rplist.lst");
   remove("tplist.lst");
   remove("coreip.lst");
}
