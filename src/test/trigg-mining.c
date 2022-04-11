
#include "_assert.h"
#include "extint.h"
#include "trigg.h"
#include <string.h>
#include <time.h>
#include <math.h>

#define MINDIFF     16
#define MAXDIFF     20
#define MAXDELTA    10.0f

/* Metric prefix array */
static char Metric[9][3] = { "", "K", "M", "G", "T", "P", "E", "Z", "Y" };

/* Block 0x1 trailer data taken directly from the Mochimo Blockchain Tfile */
static word8 Block1[BTSIZE] = {
   0x00, 0x17, 0x0c, 0x67, 0x11, 0xb9, 0xdc, 0x3c, 0xa7, 0x46,
   0xc4, 0x6c, 0xc2, 0x81, 0xbc, 0x69, 0xe3, 0x03, 0xdf, 0xad,
   0x2f, 0x33, 0x3b, 0xa3, 0x97, 0xba, 0x06, 0x1e, 0xcc, 0xef,
   0xde, 0x03, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   0xf4, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
   0xf7, 0x2d, 0x1f, 0xae, 0xa8, 0x7f, 0x5b, 0x8f, 0x3c, 0xa9,
   0xce, 0x6c, 0xdd, 0x5a, 0xe6, 0xf1, 0xb0, 0x81, 0xe5, 0x70,
   0xc1, 0xf8, 0xe9, 0x63, 0x90, 0xb1, 0x25, 0x38, 0x8e, 0x48,
   0x46, 0x73, 0x10, 0xf9, 0x01, 0x05, 0xf1, 0x01, 0x26, 0x00,
   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x56, 0xdf,
   0x01, 0x11, 0x05, 0x4b, 0xb7, 0x03, 0x01, 0x56, 0x00, 0x00,
   0x00, 0x00, 0x00, 0x00, 0xb1, 0x0d, 0x31, 0x5b, 0x78, 0x49,
   0x1f, 0x37, 0xaa, 0xa7, 0x54, 0xef, 0x7d, 0xb8, 0x1a, 0x96,
   0x42, 0xd4, 0xba, 0x1c, 0xf7, 0x2f, 0x6e, 0x37, 0xff, 0x92,
   0x99, 0x9a, 0xa0, 0x32, 0x55, 0x51, 0xbc, 0xf1, 0x5f, 0x69
};

int main()
{
   BTRAILER bt;
   clock_t solve;
   word8 diff, digest[SHA256LEN];
   float delta, hps;
   int n;

   delta = hps = n = 0;
   srand16((word32) time(NULL), 0, 0);
   memcpy(&bt, Block1, BTSIZE);
   /* increment difficulty until solve time hits 1 second */
   for (diff = 1; diff < MAXDIFF && delta < MAXDELTA; diff++) {
      bt.difficulty[0] = diff; /* update block trailer with diff */
      solve = clock(); /* record solve timestamp */
      /* initialize Trigg context, adjust diff; solve Trigg; increment hash */
      for(; trigg_solve(&bt, diff, bt.nonce); n++);
      /* calculate time taken to produce solve */
      delta = (float) (clock() - solve) / (float) CLOCKS_PER_SEC;
      /* calculate performance of algorithm */
      if (delta > 0) {
         hps = (float) n / delta;
         n = hps ? (log10f(hps) / 3) : 0;
         if (n > 0) hps /= powf(2, 10) * n;
         ASSERT_DEBUG("Diff(%d) perf: ~%.02f %sH/s\n", diff, hps, Metric[n]);
      }
      /* ensure solution is correct */
      ASSERT_EQ(trigg_checkhash(&bt, diff, digest), 0);
   }
   /* check difficulty met requirement */
   ASSERT_GE_MSG(diff, MINDIFF, "should meet minimum diff requirement");
   /* output final performance on success */
   printf("Trigg mining performance: ~%.02f %sH/s\n", hps, Metric[n]);
}