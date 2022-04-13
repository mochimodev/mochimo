
#include "_assert.h"
#include "extint.h"
#include "trigg.h"

int main()
{  /* check trigg_generate() produces expected syntax */
   word8 halfnonce[16];
   int j;

   /* Perform 40 Million iterations... yes, 40Million.
    * Why? So the 0.00000264% chance haiku frame is covered.
   */

   for (j = 0; j < 40000000; j++) {
      trigg_generate_fast(halfnonce);
      ASSERT_EQ(trigg_syntax(halfnonce), VEOK);
   }
}
