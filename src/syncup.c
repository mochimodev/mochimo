/* syncup.c Recover from chain split
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 10 November 2019
 *
*/

/* Pull a divergent block chain and merge it into ours
 * rather than bailing out to contention!
 * Always returns VEOK to ignore contention.
 * splitblock is where the two chains diverge.
 * txcblock is the advertised block of peer,
 * and peerip is its ip address.
 */
int syncup(word32 splitblock, byte *txcblock, word32 peerip)
{
   byte bnum[8], *tfweight, saveweight[HASHLEN];
   static word32 lastneo[2], sblock[2];
   char buff[256];
   int j, result;
   NODE *np2;
   time_t lasttime;

   show("syncup");
   if(Bcpid) { /* Wait for block constructor to exit... */
      if(Trace) plog("syncup(): Waiting for bcon to exit...");
      waitpid(Bcpid, NULL, 0);
      Bcpid = 0;
   }
		
   /* Kill the miner process */
   if(Trace) plog("syncup(): Stopping the miner.");
   stop_miner();

   /* Stop sending update blocks, since we're behind */
   if(Sendfound_pid) {
      if(Trace) plog("syncup(): Killing send_found()...");
      kill(Sendfound_pid, SIGTERM);
      waitpid(Sendfound_pid, NULL, 0);
      Sendfound_pid = 0;
   }

   /* Stop block transfer children and others */
   for(np2 = Nodes; np2 < Hi_node; np2++) {
      if(np2->pid == 0) continue;
      kill(np2->pid, SIGTERM);
      waitpid(np2->pid, NULL, 0);
      freeslot(np2);
   }

   /* Close server ledger */	
   if(Trace) plog("syncup(): beginning state save...");
   le_close(); 

   /* Backup TFILE, Ledger, and blocks to split-tree directory. */
   /* system("mkdir split"); * already exists */
   if(Trace) plog("syncup(): Backing up TFILE, ledger.dat, and blocks...");
   system("rm split/*");
   system("cp tfile.dat split");
   system("cp ledger.dat split");
   system("mv bc/*.bc split");
   memcpy(saveweight, Weight, HASHLEN);

   sblock[0] = splitblock;
   /* Compute first previous NG block */
   lastneo[0] = (get32(Cblocknum) & 0xffffff00) - 256;
   if(Trace) plog("syncup(): Identified first previous NG block as %s",
                  bnum2hex((byte *) &lastneo));

   /* Delete Ledger and trim T-File */
   if(unlink("ledger.dat") != 0) {
      if(Trace) plog("syncup() failed!  Unable to delete ledger.dat");
      goto badsyncup;
   }
   if(trim_tfile((byte *) &lastneo) != VEOK) {
      if(Trace) plog("syncup(): T-File trim failed!");
      goto badsyncup;
   }

   /* Extract first previous Neogenesis Block to ledger.dat */
   if(Trace) plog("syncup(): Expanding Neo-genesis block to ledger.dat...");
   sprintf(buff, "cp split/b%s.bc bc/b%s.bc", bnum2hex((byte *) &lastneo), 
           bnum2hex((byte *) &lastneo));
   system(buff);
   sprintf(buff, "bc/b%s.bc", bnum2hex((byte *) &lastneo));
   if(extract(buff, "ledger.dat") != VEOK) {
      if(Trace) plog("syncup(): failed!  Unable to extract ledger!");
      goto badsyncup;
   }

   /* setup Difficulty and globals, based on neogenesis block */
   if(reset_difficulty(NULL, Bcdir) != VEOK) {
      if(Trace) plog("syncup(): failed!  reset_difficulty() failed!");
      goto badsyncup;
   }
   le_open("ledger.dat", "rb");

   if(Trace) plog("Split point is block %s", bnum2hex((byte *) &sblock));
   add64(lastneo, One, bnum);
   for( ;cmp64(bnum, sblock) < 0; ) {
      if(Trace) plog("syncup(): Copying split/b%s.bc to spblock.tmp",
                     bnum2hex(bnum));
      sprintf(buff, "cp split/b%s.bc spblock.tmp", bnum2hex(bnum));
      system(buff);
      if(update("spblock.tmp", 1) != VEOK) {
         if(Trace) plog("syncup(): failed to update our own block.");
         goto badsyncup;
      }
      add64(bnum, One, bnum);
      if(bnum[0] == 0) add64(bnum, One, bnum);  /* skip NG blocks */
   }

   /* Download missing blocks from peer. */
   if(Trace) plog("Download and update missing blocks from peer...");
   put64(bnum, sblock);
   for(j = 0; ; ) {
      if(bnum[0] == 0) add64(bnum, One, bnum); /* don't get NG block */
      sprintf(buff, "b%s.bc", bnum2hex(bnum));
      if(j == 60) {
         if(Trace) plog("syncup(): failed while downloading %s from %s",
                        buff, ntoa((byte *) &peerip));
         goto badsyncup;
      }
      lasttime = time(NULL);
      if(get_block2(peerip, bnum, buff, OP_GETBLOCK) != VEOK) {
         if(cmp64(bnum, txcblock) >= 0) break;  /* success */
         if(time(NULL) == lasttime) sleep(1);
         j++;  /* retry counter */
         continue;
      }
      if(update(buff, 0) != VEOK) {
         if(Trace) plog("syncup(): cannot update peer's block.");
         goto badsyncup;
      }
      add64(bnum, One, bnum);
   }
   system("cp split/b0000000000000000.bc bc");
   system("rm split/*");
   /* re-compute tfile weight */
   tfweight = tfval("tfile.dat", bnum, 1, &result);
   if(result) plog("syncup(): tfval() error: %d", result);
   memcpy(Weight, tfweight, HASHLEN);
   if(result == 0) plog("syncup() is good!");
   return VEOK;

badsyncup:
   /* Restore block chain from saved state after a bad re-sync attempt. */
   if(Trace) plog("syncup(): bad sync: restoring saved state...");
   le_close();
   system("mv split/tfile.dat .");
   system("mv split/ledger.dat .");
   system("rm *.bc bc/*");
   system("mv split/* bc");
   reset_difficulty(NULL, Bcdir);  /* reset Difficulty and others */
   memcpy(Weight, saveweight, HASHLEN);
   le_open("ledger.dat", "rb");
   return VEOK;
}  /* end syncup() */
