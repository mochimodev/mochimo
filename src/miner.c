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

extern int gpu_count_cuda();
extern char *trigg_generate_cuda(byte *mroot, byte difficulty, byte *blockNumber, uint32_t threads, int gpucount);

/* miner blockin blockout -- child process */
int miner(char *blockin, char *blockout)
{
   BTRAILER bt;
   FILE *fp;
   byte *ptr;
   SHA256_CTX bctx;  /* to resume entire block hash after bcon.c */
   char *haiku;
   time_t htime;
   unsigned long hcount, hps, total_hcount;
   word32 temp[3];
   int gpucount;

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

      /*
       * First GPU - Do this once per thread
       */

      gpucount = gpu_count_cuda();

      /* Traverse all TRIGG links to build the
       * solution chain with trigg_generate()...
       */

      uint32_t threads = 600047615;
      uint64_t total_hcount = 0;

      for(;;){
         if(threads % gpucount == 0) break;
         threads--;
      }
      threads = threads / gpucount;
      for(htime = time(NULL), hcount = 0; ; hcount++) {
         if(!Running) break;
         haiku = trigg_generate_cuda(bt.mroot, bt.difficulty[0], bt.bnum, threads, gpucount);
         total_hcount += threads * gpucount;
         if(haiku != NULL) break;
      }
      htime = time(NULL) - htime;
      if(htime == 0) htime = 1;
      hps = total_hcount / htime;
      write_data(&hps, 8, "hps.dat");  /* unsigned long haiku per second */
      if(!Running) break;

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
