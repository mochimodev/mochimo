
#include <string.h>
#include <time.h>
#include <math.h>

#include "_assert.h"
#include "extint.h"
#include "exttime.h"

#include "error.h"
#include "peach.h"
#include "device.h"

#define MINDIFF     18
#define MAXDIFF     28
#define MAXDELTA    10.0f
#define GPUMAX      16

/* Block 0x1 trailer data taken directly from the Mochimo Blockchain Tfile */
static word8 Block1[sizeof(BTRAILER)] = {
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
   DEVICE_CTX D[GPUMAX] = { 0 };
   BTRAILER bt, btout;
   double delta, hps;
   int n, count;
   word32 seed;
   word8 diff, digest[SHA256LEN];
   char *m;

   setploglevel(PLOG_DEBUG);

   count = init_cuda_devices(D, GPUMAX);
   plog("Cuda Devices (%d)...", count);
   for (int idx = 0; idx < count; idx++) {
      plog(" - %s", D[idx].info);
      pdebug("initilizing device...");
      if (peach_init_cuda_device(&D[idx]) != VEOK) {
         perrno("peach initialization FAILURE");
         pwarn("%s will not be utilized...", D[idx].info);
      }
   }

   m = "";
   delta = hps = 0.0;
   time((time_t *) &seed);
   srand16(seed, seed, seed);
   memcpy(&bt, Block1, sizeof(bt));
   memset(&btout, 0, sizeof(btout));
   /* increment difficulty until solve time hits MAXDELTA */
   for (diff = MINDIFF; diff < MAXDIFF && delta < MAXDELTA; diff++) {
      /* update block trailer */
      bt.difficulty[0] = diff;
      bt.phash[0]++;
      bt.bnum[0]++;
      put32(bt.time0, (word32) time(NULL));
      /* solve Peach algorithm */
      for (n = 0; ; ) {
         int ecode = peach_solve_cuda(&D[n], &bt, 0, &btout);
         if (ecode == VEOK) break;
         if (++n >= count) n = 0;
         millisleep(1);
      }
      /* calculate performance of algorithm */
      for(hps = 0.0, n = 0; n < count; n++) {
         delta = difftime(time(NULL), D[n].last);
         if (delta == 0) delta = 1.0;
         hps += (double) D->work / delta;
      }
      m = metric_reduce(&hps);
      ASSERT_DEBUG("Diff(%d) perf: ~%.2lf %sH/s\n", diff, hps, m);
      /* ensure solution is correct */
      ASSERT_EQ(peach_checkhash(&btout, btout.difficulty[0], digest), 0);
   }
   /* check difficulty met requirement */
   ASSERT_GE_MSG(diff, 2, "should meet minimum diff requirement");
   /* output final performance on success */
   printf("Peach CUDA mining performance: ~%.2lf %sH/s\n", hps, m);
}
