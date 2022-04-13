
#include "_assert.h"
#include "extint.h"
#include "trigg.h"

int main()
{  /* check trigg_generate() produces expected syntax */
   word8 halfnonce[16];
   int j;

   for (j = 0; j < 100000; j++) {
      trigg_generate(halfnonce);
      ASSERT_EQ(trigg_syntax(halfnonce), VEOK);
   }
}
