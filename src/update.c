/* update.c  Block Update
 *
 * Copyright (c) 2018 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 25 April 2018
*/

/* Creates child to send OP_FOUND to all recent peers */
int send_found(void)
{
   word32 *ipp;
   NODE node;
   BTRAILER bt;
   char fname[128];
   int ecode;

   if(Sendfound_pid)
      return error("send_found() already running!");

   Sendfound_pid = fork();
   if(Sendfound_pid == -1) {
      Sendfound_pid = 0;
      return VERROR;  /* fork() failed */
   }
   if(Sendfound_pid) return VEOK;          /* parent returns */

   /* in child */
   show("found_child");

   /* Check if "found" NG block v.23 */
   if(Cblocknum[0] == 0) {
      ecode = 1;
      /* Back up our Cblocknum in child only to 0x...ff block. */
      if(sub64(Cblocknum, One, Cblocknum)) goto bad;
      sprintf(fname, "%s/b%s.bc", Bcdir, bnum2hex(Cblocknum));
      ecode = 2;
      if(readtrailer(&bt, fname) != VEOK
         || cmp64(Cblocknum, bt.bnum) != 0) {
bad:
         error("send_found(): ecode: %d", ecode);
         exit(1);
      }
      ecode = 3;
      if(memcmp(Prevhash, bt.bhash, HASHLEN)) goto bad;
      memcpy(Cblockhash, bt.bhash, HASHLEN);
      memcpy(Prevhash, bt.phash, HASHLEN);
   }  /* end if NG block v.23 */

   if(Trace)
      plog("send_found(0x%s)", bnum2hex(Cblocknum));

   /* Send found message to recent peers */
   shuffle32(Rplist, RPLISTLEN);
   for(ipp = Rplist; ipp < &Rplist[RPLISTLEN] && Running; ipp++) {
      if(*ipp == 0) continue;
      if(callserver(&node, *ipp) != VEOK) continue;
      send_op(&node, OP_FOUND);
      closesocket(node.sd);
   }
   exit(0);
}  /* end send_found() */


/* Return child status of pid.
 * Add peer ip to lists if needed.
 * Increment counts.
 * Returns 1 if child did not call exit(1),
 * else the exit() code (which can also be 1).
 */
int child_status(NODE *np, pid_t pid, int status)
{
   if(Trace) plog("child_status(): pid = %d  status = 0x%x", pid, status);
   if(pid > 0) {  /* child existed and called exit() */
      if(WIFEXITED(status)) {
         status = WEXITSTATUS(status);
         if(status != 0) {
            if(status >= 2) pinklist(np->src_ip);
            if(status >= 3) epinklist(np->src_ip);
            return status;
         }
      } else return 1;  /* error if not exited */
      return status;  /* 0 */
   }  /* end if child exit()'ed */
   return 1;  /* error if child caught signal */
}  /* end child_status() */


/* validate and update from fname = rblock.dat or vblock.dat
 * mode: 0 = their block
 *       1 = our block
 */
int update(char *fname, int mode)
{
   char cmd[100];

   if(Trace) plog("Entering update()");
   if(!exists(fname)) return VERROR;
   show("update");

   if(Bcpid) {
      if(Trace) plog("   Waiting for bcon to exit...");
      waitpid(Bcpid, NULL, 0);  /* wait for bcon to exit */
      /* Make miner idle during update. */
      unlink("cblock.dat");
      unlink("miner.tmp");
      unlink("ublock.tmp");
      if(mode == 0) unlink("mblock.dat");
      Bcpid = 0;
   }
   #ifndef NO_CUDA
   stop_miner();
   #endif

   /* wait for send_found() to exit */
   if(Sendfound_pid) {
      if(Trace) plog("   Waiting for send_found() to exit");
      kill(Sendfound_pid, SIGTERM);
      waitpid(Sendfound_pid, NULL, 0);
      Sendfound_pid = 0;
   }

   write_global();  /* gift bval with Peerip and other globals */
   le_close();      /* close server ledger reference */

   if(Trace) plog("   About to call bval and bup...");

   sprintf(cmd, "../bval %s", fname);  /* call validator on fname */
   system(cmd);
   if(!exists("vblock.dat")) {      /* validation failed */
      if(mode == 0) {   /* their block -- bad validation */
         pinklist(Peerip);             /* she was naughty */
         epinklist(Peerip);            /* she was a bad girl! */
      }
      system("../txclean txclean.dat");  /* prune missing src_addr's */
      le_open("ledger.dat", "rb");  /* re-open ledger */
      return VERROR;
   }
   /* update vblock.dat */
   system("../bup vblock.dat ublock.dat");

   system("../txclean txclean.dat");  /* prune missing src_addr's */
   le_open("ledger.dat", "rb");  /* re-open new ledger.dat */
   if(!exists("ublock.dat")) {
      if(mode == 0) {
         pinklist(Peerip);   /* peer was bad */
         epinklist(Peerip);  /* peer was evil */
      } else unlink("mblock.dat");
      return VERROR;
   }

   /* Everything below this line has to succeed, or else
    * we restart() with an update error.
    * -----------------------------------------------------*/

   /* Update:
    * Cblocknum, Cblockhash, Prevhash, Difficulty, Time0,
    * and tfile.dat.
    */
   if(bupdata() != VEOK) goto err;  /* calls add_weight() */
   if(append_tfile("ublock.dat", "tfile.dat") != VEOK) goto err;
   if((Cblocknum[0] & EPOCHMASK) == 0)  /* pink list epoch counter */
      purge_epoch();
   /* rename and move ublock.dat to the bc/ directory */
   if(moveublock("ublock.dat", Cblocknum) != VEOK) goto err;
   mergepinklists();
   if(write_global() != VEOK) goto err;     /* for miner */
   if(Cblocknum[0] == 0xff) {
      if(do_neogen() != VEOK) goto err;
      if(Trace) {
         plog("neo Cblocknum: 0x%s", bnum2hex(Cblocknum));
         plog("Cblockhash: %s for block: 0x%s", hash2str(Cblockhash),
              bnum2hex(Cblocknum));
      }
   }
   if(mode == 1) {
      Nsolved++;  /* our block */
      write_data(&Nsolved, 4, "solved.dat");
   }
   Nupdated++;
   memset(Crclist, 0, CRCLISTLEN*4);  /* clear recent crc list */
   Crclistidx = 0;
   Stime = Ltime + 20;  /* hold status display */
   if(!Ininit) {
      if(exists("../update.sh")) system("../update.sh");  /* synchronous */
   }
   plog("Block %s: 0x%s", mode ? "solved" : "updated", bnum2hex(Cblocknum));
   if(!Ininit)
      printf("Solved: %u  Haiku/second: %lu  Difficulty: %d\n",
             Nsolved, Hps, Difficulty);
   return VEOK;
err:
   restart("update error!");
   return VERROR;  /* never gets here */
}  /* end update() */
