
#include "_assert.h"
#include "extint.h"
#include "extlib.h"
#include "extmath.h"
#include "util.h"

/* ==================== */
/* THE OG get_mreward() */
void og_get_mreward(word32 *reward, word32 *bnum)
{
   word8 bnum2[8];
   static word32 delta[2] = { 0xDAC0, 0 };      /* reward delta 56000 */
   static word32 base1[2] = { 0x2A05F200, 1 };  /* base 5000000000 */
   static word32 base2[2] = { 0x60b43c80, 1 };  /* base 5917392000 */
   static word32 base3[2] = { 0xdbe74670, 0x0d };  /* base 59523942000 */
   static word32 t1[2] =  { 17185, 0 };         /* new reward trigger block */
   static word32 t2[2] =  { 373761, 0 };        /* mid block */
   static word32 t3[2] =  { 2097152, 0 };       /* final reward block */
   static word32 delta2[2] = { 150000, 0 };     /* increment */
   static word32 delta3[2] = { 28488, 0 };      /* decrement */

   if(cmp64(bnum, t1) < 0) {
      /* bnum < 17185 */
      if(sub64(bnum, One, bnum2)) goto noreward;
      mult64(delta, bnum2, reward);
      add64(reward, base1, reward);
      return;
   }
   if(cmp64(bnum, t3) > 0) {
noreward:
      reward[0] = reward[1] = 0;  /* after t3, reward is zero */
      return;
   }
   if(cmp64(bnum, t2) < 0) {
      /* first 4 years */
      sub64(bnum, t1, bnum2);
      mult64(delta2, bnum2, reward);
      add64(reward, base2, reward);
      return;
   } else {
      /* last 18 years */
      sub64(bnum, t2, bnum2);
      mult64(delta3, bnum2, reward);
      if(sub64(base3, reward, reward)) goto noreward;
   }
}  /* end og_get_mreward() */
/* ======================= */

int main()
{
   word32 test_reward[2];
   word32 og_reward[2];
   word32 bnum[2];
   int i;

   put64(bnum, One);

   /* iterate every block reward and ensure match */
   for (i = 0; i < 0x200000; i++, add64(bnum, One, bnum)) {
      get_mreward(test_reward, bnum);
      og_get_mreward(og_reward, bnum);
      /* test every single reward */
      ASSERT_CMP_MSG(test_reward, og_reward, 8, "rewards must match");
   }
}