/* miner.c  The Block Miner  -- Child Process
 *
 * Copyright (c) 2018 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * The Mochimo Project System Software
 *
 * Date: 13 January 2018
 *
 */

#include <inttypes.h>

extern int trigg_init_cuda(byte difficulty, byte *blockNumber);
extern void trigg_free_cuda();
extern char *trigg_generate_cuda(byte *mroot, unsigned long long *nHaiku);

/* miner blockin blockout -- child process */
int miner(char *blockin, char *blockout)
{
   BTRAILER bt;
   FILE *fp;
   byte *ptr;
   SHA256_CTX bctx;  /* to resume entire block hash after bcon.c */
   char *haiku;
   time_t htime;
   unsigned long long hcount, hps, total_hcount;
   word32 temp[3];
   int initGPU;
   struct timespec chill = {0,Dynasleep*1000L};

   /* Keep a separate rand2() sequence for miner child */
   if(read_data(&temp, 12, "mseed.dat") == 12)
      srand2(temp[0], temp[1], temp[2]);

   for( ;; sleep(10)) {
      /* Running is set to 0 on SIGTERM */
      if(!Running) break;
      if(!exists(blockin)) break;
      if(read_data(&bctx, sizeof(bctx), "bctx.dat") != sizeof(bctx)) {
         error("miner: cannot read bctx.dat");
         break;
      }
      unlink("bctx.dat");
      if((fp = fopen(blockin, "rb")) == NULL) {
         error("miner: cannot open %s", blockin);
         break;
      }
      if(fseek(fp, -(sizeof(BTRAILER)), SEEK_END) != 0) {
         fclose(fp);
         error("miner: seek error");
         break;
      }
      if(fread(&bt, 1, sizeof(bt), fp) != sizeof(bt)) {
         error("miner: read error");
         fclose(fp);
         break;
      }
      fclose(fp);
      unlink("miner.tmp");
      if(rename(blockin, "miner.tmp") != 0) {
         error("miner: cannot rename %s", blockin);
         break;
      }

      show("solving");
      if(Trace)
         plog("miner: beginning solve: %s block: 0x%s", blockin,
              bnum2hex(bt.bnum));

      /* Create the solution state-space beginning with 
       * the first plausible link on the TRIGG chain.
       */
      trigg_solve(bt.mroot, bt.difficulty[0], bt.bnum);

      /* Initialize CUDA specific memory allocations
       * and check for obvious errors
       */
      initGPU = -1;
      initGPU = trigg_init_cuda(bt.difficulty[0], bt.bnum);
      if(initGPU==-1) {
         error("miner: cuda initialization failed. Check GPUs");
         trigg_free_cuda();
         break;
      }
      if(initGPU<1 || initGPU>64) {
         error("miner: unsupported number of GPUs detected -> %d",initGPU);
         trigg_free_cuda();
         break;
      }

      /* Traverse all TRIGG links to build the
       * solution chain with trigg_generate()...
       */

      for(haiku = NULL, htime = time(NULL), hcount = 0; ; ) {
         if(!Running) break;
         if(haiku != NULL) break;
         haiku = trigg_generate_cuda(bt.mroot, &hcount);
         if(total_hcount == hcount) nanosleep(&chill, NULL);
         else total_hcount = hcount;
      }

      /* Free CUDA specific memory allocations */
      trigg_free_cuda();

      /* Calculate and write Haiku/s to disk */
      htime = time(NULL) - htime;
      if(htime == 0) htime = 1;
      hps = hcount / htime;
      write_data(&hps, 8, "hps.dat");  /* unsigned long haiku per second */
      if(!Running) break;

      /* Block validation (double)check */
      if (!trigg_check(bt.mroot, bt.difficulty[0], bt.bnum)) {
         printf("ERROR - Block is not valid\n");
         break;
      }

      show("solved");

      /* solved block! */
      sleep(2);  /* make sure that stime is not to early */
      put32(bt.stime, time(NULL));  /* put solve time in trailer */
      /* hash-in nonce and solved time to hash 
       * context begun by bcon.
       */
      sha256_update(&bctx, bt.nonce, HASHLEN + 4);
      sha256_final(&bctx, bt.bhash);  /* put hash in block trailer */
      fp = fopen("miner.tmp", "r+b");
      if(fp == NULL) {
         if(Trace) plog("miner: cannot re-open miner.tmp");
         break;
      }
      if(fseek(fp, -(sizeof(BTRAILER)), SEEK_END) != 0) {
         fclose(fp);
         error("miner: cannot fseek(trailer) miner.tmp");
         break;
      }
      if(fwrite(&bt, 1, sizeof(bt), fp) != sizeof(bt)) {
         fclose(fp);
         error("miner: cannot fwrite(trailer) miner.tmp");
         break;
      }
      fclose(fp);
      unlink(blockout);
      if(rename("miner.tmp", blockout) != 0) {
         error("miner: cannot rename miner.tmp");
         break;
      }

      if(Trace)
         plog("miner: solved block 0x%s is now: %s",
              bnum2hex(bt.bnum), blockout);

      printf("\n%s\n\n", haiku);

      break;
   }  /* end for  */
done:
   getrand2(temp, &temp[1], &temp[2]);
   write_data(&temp, 12, "mseed.dat");   /* maintain rand2() sequence */
   if(Trace) plog("Miner exiting...");
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
