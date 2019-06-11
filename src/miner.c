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
#include "algo/peach/peach.c"

#ifdef CUDANODE
char *cuda_peach(byte *bt, char *haiku, uint32_t *hps, byte *runflag);
int init_cuda_peach(byte difficulty, byte *prevhash, byte *blocknumber);
void free_cuda_peach();
#endif

/* miner blockin blockout -- child process */
int miner(char *blockin, char *blockout)
{
   BTRAILER bt;
   FILE *fp;
   SHA256_CTX bctx;  /* to resume entire block hash after bcon.c */
   char *haiku;
   time_t htime;
   word32 temp[3], hcount, hps;
   static word32 v24trigger[2] = { V24TRIGGER, 0 };

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
#ifdef CUDANODE
      if (!(init_cuda_peach(Difficulty, bt.phash, bt.bnum) & 0x3f))
      {
          error("Miner failed to initilize CUDA devices\n");
          break;
      }
#endif

      if(cmp64(bt.bnum, v24trigger) > 0)
      { /* v2.4 and later */

#ifdef CUDANODE
         cuda_peach((byte *) &bt, haiku, &hps, &Running);
          /* ... better double check */
          if(peach(&bt, Difficulty, haiku, NULL, 1)) {
              printf("ERROR - Solved block is not valid\n");
              error("!!!!!Peach solved block is not valid!!!!!");
             sleep(5);
              break;;
          }
          /* K all g... */
#endif
#ifdef CPUNODE
         if(peach(&bt, Difficulty, haiku, &hps, 0)) break;
#endif

         write_data(&hps, sizeof(word32), "hps.dat");
      }


/* Legacy handler is CPU Only for all v2.3 and earlier blocks *

      if(cmp64(bt.bnum, v24trigger) <= 0)
      {
         /* Create the solution state-space beginning with
          * the first plausible link on the TRIGG chain.
          *
         trigg_solve(bt.mroot, bt.difficulty[0], bt.bnum);

         /* Traverse all TRIGG links to build the
          * solution chain with trigg_generate()...
          *

         for(haiku = NULL, htime = time(NULL), hcount = 0; ; ) {
            if(!Running) break;
            if(haiku != NULL) break;
            haiku = trigg_generate(bt.mroot, bt.difficulty[0]);
            hcount++;
         }
         /* Calculate and write Haiku/s to disk *
         htime = time(NULL) - htime;
         if(htime == 0) htime = 1;
         hps = hcount / htime;
         write_data(&hps, sizeof(hps), "hps.dat");  /* unsigned int haiku per second *
         if(!Running) break;

         /* Block validation check *
         if (!trigg_check(bt.mroot, bt.difficulty[0], bt.bnum)) {
            printf("ERROR - Block is not valid\n");
            break;
         }
      } /* end legacy handler */

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

      if(!Bgflag) printf("\nM:%s\n\n", haiku);

      break;
   }  /* end for(;;) exit miner  */

#ifdef CUDANODE
   free_cuda_peach();
#endif
   getrand2(temp, &temp[1], &temp[2]);
   write_data(&temp, 12, "mseed.dat");    /* maintain rand2() sequence */
   printf("Miner exiting...\n");
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
