/* update.c  Block Update
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 25 April 2018
 * Updated: 15 December 2019
*/

/* Creates child to send OP_FOUND to all recent peers */
int send_found(void)
{
   word32 *ipp;
   NODE node;
   BTRAILER bt;
   char fname[128];
   int ecode;
   TX tx;

   if(Sendfound_pid) {
      plog("send_found() is already running -- rerun it.");
      kill(Sendfound_pid, SIGTERM);
      waitpid(Sendfound_pid, NULL, 0);
      Sendfound_pid = 0;
   }

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
         perr("send_found(): ecode: %d", ecode);
         exit(1);
      }
      ecode = 3;
      if(memcmp(Prevhash, bt.bhash, HASHLEN)) goto bad;
      memcpy(Cblockhash, bt.bhash, HASHLEN);
      memcpy(Prevhash, bt.phash, HASHLEN);
   }  /* end if NG block v.23 */

   pdebug("send_found(0x%s)", bnum2hex(Cblocknum));

   loadproof(&tx);  /* get proof from tfile.dat */
   /* Send found message to recent peers */
   shuffle32(Rplist, RPLISTLEN);
   for(ipp = Rplist; ipp < &Rplist[RPLISTLEN] && Running; ipp++) {
      if(*ipp == 0) continue;
      if(callserver(&node, *ipp) != VEOK) continue;
      memcpy(&node.tx, &tx, sizeof(TX));  /* copy in tfile proof */
      send_op(&node, OP_FOUND);
      sock_close(node.sd);
   }
   /* Send found message to local peers */
   for(ipp = Tplist; ipp < &Tplist[TPLISTLEN] && Running; ipp++) {
      if(*ipp == 0) continue;
      if(callserver(&node, *ipp) != VEOK) continue;
      memcpy(&node.tx, &tx, sizeof(TX));  /* copy in tfile proof */
      send_op(&node, OP_FOUND);
      sock_close(node.sd);
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
   pdebug("child_status(): pid = %d  status = 0x%x", pid, status);
   if(pid > 0) {  /* child existed and called exit() */
      if(WIFEXITED(status)) {
         status = WEXITSTATUS(status);
         if(status != 0) {
            if(status >= 2) pinklist(np->ip);
            if(status >= 3) epinklist(np->ip);
            return status;
         }
      } else return 1;  /* error if not exited */
      return status;  /* 0 */
   }  /* end if child exit()'ed */
   return 1;  /* error if child caught signal */
}  /* end child_status() */


/* Reap cblock and mblock-push children... */
void reaper2(void)
{
   NODE *np;
   word16 opcode;

   /* Don't fear the Reaper, baby...It won't hurt... */
   for(np = Nodes; np < Hi_node; np++) {
      if(np->pid == 0) continue;
      opcode = get16(np->tx.opcode);
      if(opcode == OP_GET_CBLOCK || opcode == OP_MBLOCK) {
         kill(np->pid, SIGTERM);
         waitpid(np->pid, NULL, 0);
         freeslot(np);
      }
   }
}  /* end reaper2() */


/* Return block header length, or 0 if not found. */
word32 gethdrlen(char *fname)
{
   FILE *fp;
   word32 len;

   fp = fopen(fname, "rb");
   if(fp == NULL) return 0;
   if(fread(&len, 1, 4, fp) !=  4) { fclose(fp); return 0; }
   fclose(fp);
   return len;
}


#include "txclean.c"  /* internal txclean() function */

/* validate and update from fname = rblock.dat or vblock.dat
 * mode: 0 = their block
 *       1 = our block
 *       2 = pseudo-block
 */
int update(char *fname, int mode)
{
   char cmd[100];
   char *solvestr;

   pdebug("Entering update()");
   if(!fexists(fname)) return VERROR;
   show("update");
   solvestr = NULL;

   if(Bcpid) {
      pdebug("   Waiting for bcon to exit...");
      waitpid(Bcpid, NULL, 0);  /* wait for bcon to exit */
      /* Make miner idle during update. */
      unlink("cblock.dat");
      unlink("miner.tmp");
      unlink("ublock.tmp");
      if(mode == 0) unlink("mblock.dat");
      Bcpid = 0;
   }
   stop_miner();

   /* wait for send_found() to exit */
   if(Sendfound_pid) {
      pdebug("   Waiting for send_found() to exit");
      kill(Sendfound_pid, SIGTERM);
      waitpid(Sendfound_pid, NULL, 0);
      Sendfound_pid = 0;
   }

   if(!Ininit && Allowpush) reaper2();

   write_global();  /* gift bval with Peerip and other globals */

   /* Check for pseudo-block */
   if(mode == 2 || gethdrlen(fname) == 4) {
      mode = 2;
      if(pval(fname) != VEOK) return VERROR;  /* renames to ublock.dat */
      goto after_bup;
   }

   le_close();      /* close server ledger reference */

   pdebug("   About to call bval and bup...");

   /* Hotfix for critical bug identified on 09/26/19 */
   if(fexists("cblock.lck")) {
      unlink("cblock.lck");
      solvestr = "pushed";
   }

   tag_free(); /* Erase Tagidx[] to be rebuilt on next tag_find() call. */
   sprintf(cmd, "../bval %s", fname);  /* call validator on fname */
   system(cmd);
   if(!fexists("vblock.dat")) {      /* validation failed */
      txclean();  /* clean the queue */
      le_open("ledger.dat", "rb");  /* re-open ledger */
      return VERROR;
   }
   /* update vblock.dat */
   system("../bup vblock.dat ublock.dat");

   txclean();  /* clean the queue */
   le_open("ledger.dat", "rb");  /* re-open new ledger.dat */
   if(!fexists("ublock.dat")) {
      if(mode != 0) unlink("mblock.dat");
      return VERROR;
   }

after_bup:

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
      if(CAROUSEL(Cblocknum)) {
         tag_free();  /* Erase old in-memory Tagidx[] */
         if(renew()) goto err;
         txclean();  /* clean the tx queue */
         if(le_open("ledger.dat", "rb") != VEOK) goto err;  /* reopen */
      }
   }
   if(mode == 1 && Insyncup == 0 && solvestr == NULL) { /* not "pushed" */
      solvestr = "solved";
      Nsolved++;  /* our block */
      write_data(&Nsolved, 4, "solved.dat");
   }
   Stime = Ltime + 20;  /* hold status display */
   if(!Ininit) {
      /* synchronous */
      if(fexists("../update-external.sh")) system("../update-external.sh");
   }
   if(mode != 2) {  /* not a pseudo-block */
      if(!Ininit) {
         if(Insyncup) plog("Syncing Block: 0x%s", bnum2hex(Cblocknum));
         else {
            if(solvestr == NULL) solvestr = "updated";
            plog("Block %s: 0x%s", solvestr, bnum2hex(Cblocknum));
            if(!Bgflag)
               printf("Solved: %u  Haiku/second: %lu  Difficulty: %d\n",
                      Nsolved, (unsigned long) Hps, Difficulty);
            Nupdated++;
         }  /* end if !Insyncup */
         Utime = time(NULL);  /* update time for watchdog */
      }  /* end if !Ininit */
   }  /* end if not-pseudo-block */
   Bridgetime = Time0 + BRIDGE;  /* advance pseudo-block timer */
   return VEOK;
err:
   restart("update error!");
   return VERROR;  /* never gets here */
}  /* end update() */
