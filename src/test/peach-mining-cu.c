
/* must be declared before includes */
#ifndef CUDA
   #define CUDA
#endif

#include <string.h>
#include <time.h>
#include <math.h>

#include "_assert.h"
#include "extint.h"
#include "extprint.h"
#include "exttime.h"
#include "peach.h"

#define MINDIFF     18
#define MAXDIFF     22
#define MAXDELTA    10.0f
#define GPUMAX      16

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
   BTRAILER bt, btout;
   word8 diff, digest[SHA256LEN];
   DEVICE_CTX D[GPUMAX] = { 0 };
   float delta, hps;
   int m, n, count;
   // time_t now = 0;

   set_print_level(PLEVEL_DEBUG);
   count = peach_init_cuda(D, GPUMAX);

   delta = hps = n = 0;
   srand16((word32) time(NULL), (word32) time(NULL), (word32) time(NULL));
   memcpy(&bt, Block1, BTSIZE);
   /* increment difficulty until solve time hits 1 second */
   for (diff = MINDIFF, m = 0; diff < MAXDIFF && delta < MAXDELTA; diff++, m=0) {
      /* update block trailer with diff */
      bt.difficulty[0] = diff;
      bt.phash[0] = diff;
      /* initialize Peach context, adjust diff; solve Peach; increment hash */
      while(peach_solve_cuda(&D[m], &bt, diff, &btout)) {
         if (++m >= count) m = 0;
         millisleep(1); /*
         psticky("CUDA#%d: status: %d, progress: %" P64u ", "
            "hps: %g H/s, "
            "fan/pow/temp/util: %u/%u/%u/%u, "
            "grid/block/threads: %d/%d/%d",
            D->status, 0, (double) D->work / difftime(time(NULL), D->last_work),
            D->work, D->fan, D->pow, D->temp, D->util,
            D->grid, D->block, D->threads); */
      }
      /* calculate performance of algorithm */
      for(hps = n = 0; n < count; n++) {
         delta = difftime(time(NULL), D[n].last_work);
         if (delta == 0) hps += (float) D->work;
         else hps += (float) D->work / difftime(time(NULL), D[n].last_work);
      }
      n = hps ? (log10f(hps) / 3) : 0;
      hps /= powf(1000, n);
      ASSERT_DEBUG("Diff(%d) perf: ~%g %sH/s", diff, hps, Metric[n]);
      /* ensure solution is correct */
      ASSERT_EQ(peach_checkhash(&btout, btout.difficulty[0], digest), VEOK);
      /*
      psticky("CUDA#%d: progress: %" P64u ", "
         "fan/pow/temp/util: %u/%u/%u/%u, "
         "grid/block/threads: %d/%d/%d",
         0, D->work, D->fan, D->pow, D->temp, D->util,
         D->grid, D->block, D->threads); */
   } /*
   plog("CUDA#%d: progress: %" P64u ", "
      "fan/pow/temp/util: %u/%u/%u/%u, "
      "grid/block/threads: %d/%d/%d",
      0, D->work, D->fan, D->pow, D->temp, D->util,
      D->grid, D->block, D->threads); */
   /* check difficulty met requirement */
   ASSERT_GE_MSG(diff, MINDIFF, "should meet minimum diff requirement");
   /* output final performance on success */
   printf("Peach CUDA mining performance: ~%.02f %sH/s\n", hps, Metric[n]);
}
