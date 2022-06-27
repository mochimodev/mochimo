
#include "_assert.h"
#include "network.h"
#include <stdlib.h>

#include "_testutils.h"

int main()
{
   char *fname = "genesis.bc";
   word8 bnum[8] = { 0 };
   int status = VERROR;

   sock_startup();  /* enable socket support */
   do {  /* retrieve genesis block from one of the mochimap nodes */
      status = get_file(aton(*(hostsp++)), bnum, fname);
   } while(status && **hostsp);
   ASSERT_EQ_MSG(status, VEOK, "failed to obtain genesis block");
   remove(fname);
   sock_cleanup();
}
