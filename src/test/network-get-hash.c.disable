
#include "_assert.h"
#include "network.h"
#include <string.h>

#include "_testutils.h"

int main()
{
   word8 zero[HASHLEN] = { 0, };
   word8 hash[HASHLEN] = { 0, };
   word8 hash0[HASHLEN] = { 0, };
   word8 bnum0[8] = { 0, };
   int status = VERROR;
   NODE node;

   Running = 1;
   sock_startup();  /* enable socket support */
   /* try communicate with an invalid nodes */
   ASSERT_NE(get_hash(&node, aton("example.com"), NULL, NULL), VEOK);
   /* try communicate with one of the valid mochimap nodes */
   while (**hostsp) {
      status = get_hash(&node, aton(*hostsp), bnum0, hash0);
      if (status == VEOK) break;
      hostsp++;
   }
   ASSERT_EQ_MSG(status, VEOK, "failed to obtain genesis hash");
   ASSERT_NE_MSG(memcmp(hash0, zero, HASHLEN), 0,
      "genesis hash should not be zeros");
   while (**hostsp) {
      status = get_hash(&node, aton(*hostsp), NULL, hash);
      if (status == VEOK) break;
      hostsp++;
   }
   ASSERT_EQ_MSG(status, VEOK, "failed to obtain latest hash");
   ASSERT_NE_MSG(memcmp(hash, zero, HASHLEN), 0,
      "latest hash is not likely zeros");
   sock_cleanup();
}
