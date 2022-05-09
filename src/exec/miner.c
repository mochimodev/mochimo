/* miner.c  The Block Miner  -- Child Process
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * The Mochimo Project System Software
 *
 * Date: 13 January 2018
 *
 * Expect this file to be if-def'd up for various miners.
 *
 */

#include <inttypes.h>
#include "exttime.h"
#include "extprint.h"
#include "peach.h"

#define GPUMAX 64

uint8_t nvml_init = 0;

/* miner blockin blockout -- child process */
int miner(char *blockin, char *blockout)
{
   BTRAILER bt;
   FILE *fp;
   SHA256_CTX bctx;  /* to resume entire block hash after bcon.c */

   char phaiku[256];
   time_t start;
   word32 temp[3] /* , hps, n */;

#ifdef CUDA
   DEVICE_CTX D[GPUMAX] = { 0 };
   char gpustats[BUFSIZ] = { 0 };
   char *sp;
   int m, count;
   double htime;
   time_t poll = time(NULL);

#else
   int n;

#endif

   /* Keep a separate rand16() sequence for miner child */
   if(read_data(&temp, 12, "mseed.dat") == 12)
      srand16(temp[0], temp[1], temp[2]);

   for(time(&start);; sleep(10)) {
      /* Running is set to 0 on SIGTERM */
      if(!Running) break;
      if(!fexists(blockin)) break;
      if(read_data(&bctx, sizeof(bctx), "bctx.dat") != sizeof(bctx)) {
         perr("miner: cannot read bctx.dat");
         break;
      }
      unlink("bctx.dat");
      if((fp = fopen(blockin, "rb")) == NULL) {
         perr("miner: cannot open %s", blockin);
         break;
      }
      if(fseek(fp, -(sizeof(BTRAILER)), SEEK_END) != 0) {
         fclose(fp);
         perr("miner: seek error");
         break;
      }
      if(fread(&bt, 1, sizeof(bt), fp) != sizeof(bt)) {
         perr("miner: read error");
         fclose(fp);
         break;
      }
      fclose(fp);
      unlink("miner.tmp");
      if(rename(blockin, "miner.tmp") != 0) {
         perr("miner: cannot rename %s", blockin);
         break;
      }

      show("solving");
      pdebug("miner: beginning solve: %s block: 0x%s", blockin,
              bnum2hex(bt.bnum));

#ifdef CUDA
         /* Allocate and initialize necessary memory on CUDA devices */
         for (m = 0, count = peach_init_cuda(D, GPUMAX); Running && count &&
               peach_solve_cuda(&D[m], &bt, Difficulty, &bt);
               millisleep(1))
         {
            if (++m >= count) {
               if (difftime(time(NULL), poll) && !Monitor) {
                  time(&poll);
                  sp = gpustats;
                  memset(gpustats, 0, BUFSIZ);
                  for (m = 0; m < count; m++) {
                     if ((sp - gpustats) >= (BUFSIZ - 1)) break;
                     htime = difftime(time(NULL), D[m].last_work);
                     if (sp != gpustats) *(sp++) = '\n';
                     snprintf(sp, (size_t) (BUFSIZ - (sp - gpustats)),
                        "%s [%uW:%u°C] %g H/s",
                        D[m].nameId, D[m].pow, D[m].temp,
                        (double) D[m].work / htime);
                     sp += strlen(sp);
                  }
                  psticky("%s", gpustats);
               }
               m = 0;
            }
         }
         /* Free allocated memory on CUDA devices */
         /* free_cuda_peach(); */
         /* K all g... */

#else  /* CPU */
         /* initialize Peach context, adjust diff; solve Peach; increment hash */
         for(peach_init(&bt);
            Running && peach_solve(&bt, Difficulty, bt.nonce);
            n++);
         /* Calculate and write Haiku/s to disk 
         htime = difftime(time(NULL), start);
         if(htime == 0) htime = 1;
         hps = n / htime; */

#endif

         /* Block validation check */
         if (!Running) break;
         else if (peach_check(&bt) == VEOK) {
            /* Print Haiku */
            trigg_expand(bt.nonce, phaiku);
            if(!Bgflag) plog("\nSOLVED!!!\n%s\n\n", phaiku);
         } else {
            perr("Invalid solve!\n");
            break;
         }

      /* Everything below this line is shared code.  */
      show("solved");

      /* solved block! */
      sleep(1);  /* make sure that stime is not to early */
      put32(bt.stime, time(NULL));  /* put solve time in trailer */
      /* hash-in nonce and solved time to hash
       * context begun by bcon.
       */
      sha256_update(&bctx, bt.nonce, HASHLEN + 4);
      sha256_final(&bctx, bt.bhash);  /* put hash in block trailer */
      fp = fopen("miner.tmp", "r+b");
      if(fp == NULL) {
         pdebug("miner: cannot re-open miner.tmp");
         break;
      }
      if(fseek(fp, -(sizeof(BTRAILER)), SEEK_END) != 0) {
         fclose(fp);
         perr("miner: cannot fseek(trailer) miner.tmp");
         break;
      }
      if(fwrite(&bt, 1, sizeof(bt), fp) != sizeof(bt)) {
         fclose(fp);
         perr("miner: cannot fwrite(trailer) miner.tmp");
         break;
      }
      fclose(fp);
      unlink(blockout);
      if(rename("miner.tmp", blockout) != 0) {
         perr("miner: cannot rename miner.tmp");
         break;
      }

      pdebug("miner: solved block 0x%s is now: %s",
              bnum2hex(bt.bnum), blockout);
      break;
   }  /* end for(;;) exit miner  */

   get_rand16(temp, &temp[1], &temp[2]);
   write_data(&temp, 12, "mseed.dat");    /* maintain rand16() sequence */
   return 0;
}  /* end miner() */


/* Start the miner as a child process */
int start_miner(void)
{
   pid_t pid;

   if(Mpid) return VEOK;
   pid = fork();
   if(pid < 0) return VERROR;
   if(pid) { Mpid = pid; return VEOK; }  /* parent */
   /* child */
   miner("cblock.dat", "mblock.dat");
   exit(0);
}  /* end start_miner() */
