
#include "_assert.h"
#include "network.h"

int main()
{  /* check isprivate() filters private IPs */
   char **pp, *private[] = {
      "10.0.0.0", "10.255.255.255",  /* class A */
      "172.16.0.0", "172.31.255.255",  /* class B */
      "192.168.0.0", "192.168.255.255",  /* class C */
      "169.254.0.0", "169.254.255.255",  /* auto */
      ""
   };
   /* check all private IPs */
   pp = private;
   while(**pp) ASSERT_GT(isprivate(aton(*(pp++))), 0);
   ASSERT_EQ(isprivate(aton("123.123.123")), 0);
}
