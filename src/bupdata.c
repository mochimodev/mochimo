/* bupdata.c  Update Globals
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * The Mochimo Project System Software
 *
 * Date: 6 February 2018
 *
*/


/* seconds is 32-bit signed, stime and bnum are from block trailer.
 * NOTE: hash is set to 0 for old algorithm.
 * If used and integrating into an old chain,
 * change DTRIGGER31 to a non-NG block number on which to
 * trigger new algorithm.
 */
word32 set_difficulty(word32 difficulty, int seconds, word32 stime, byte *bnum)
{
   word32 hash;
   int highsolve = 284;
   int lowsolve = 143;

   /* Change DTRIGGER31 to a non-NG block number trigger for new algorithm. */
   static word32 trigger_block[2] = { DTRIGGER31, 0 };
   static word32 fix_trigger[2] = { FIXTRIGGER, 0 };
   if(seconds < 0) return difficulty;
   if(cmp64(bnum, trigger_block) < 0){
      hash = 0;
      highsolve = 506;
      lowsolve = 253;
   }
   else
      hash = (stime >> 6) ^ stime;
   if(cmp64(bnum, fix_trigger) > 0) hash = 0;
   if(seconds > highsolve) {
      if(difficulty > 0) difficulty--;
      if(difficulty > 0 && (hash & 1)) difficulty--;
   } else if(seconds < lowsolve) {
      if((hash & 3) == 0  && difficulty < 255)
         difficulty++;
   }
   return difficulty;
}

/* Called from server.c to update globals */
int bupdata(void)
{
   int ecode = VEOK;
   word32 time1;
   BTRAILER bt;

   if(add64(Cblocknum, One, Cblocknum)) /* increment block number */
      ecode = error("new blocknum overflow");

   /* Update block hashes */
   memcpy(Prevhash, Cblockhash, HASHLEN);
   if(readtrailer(&bt, "ublock.dat") != VEOK)
      ecode = error("bupdata(): cannot read new ublock.dat hash");
   memcpy(Cblockhash, bt.bhash, HASHLEN);
   Difficulty = get32(bt.difficulty);
   Time0 = get32(bt.time0);
   time1 = get32(bt.stime);
   add_weight(Weight, Difficulty, bt.bnum);
   /* Update block difficulty */
   Difficulty = set_difficulty(Difficulty, time1 - Time0, time1, bt.bnum);
   if(Trace) {
      plog("new: Difficulty = %d  seconds = %d",
           Difficulty, time1 - Time0);
      plog("Cblockhash: %s for block: 0x%s", hash2str(Cblockhash),
           bnum2hex(Cblocknum));
   }
   Time0 = time1;
   return ecode;
}  /* end bupdata() */


/* Build a neo-genesis block -- called from server.c */
int do_neogen(void)
{
   char cmd[1024];
   word32 newnum[2];
   char *cp;
   int ecode;
   BTRAILER bt;

   unlink("neofail.lck");
   cp = bnum2hex(Cblocknum);
   sprintf(cmd, "../neogen %s/b%s.bc ngblock.dat", Bcdir, cp);
   if(Trace) plog("Creating neo-genesis block:\n '%s'", cmd);
   ecode = system(cmd);
   if(Trace) plog("do_neogen(): system():  ecode = %d", ecode);
   if(exists("neofail.lck"))
      return error("do_neogen failed");
   add64(Cblocknum, One, newnum);
   cp = bnum2hex((byte *) newnum);
   sprintf(cmd, "%s/b%s.bc", Bcdir, cp);
   if(rename("ngblock.dat", cmd) != 0)
      return error("do_neogen failed to rename ngblock.dat to %s", cmd);

   add64(Cblocknum, One, Cblocknum);
   /* Update block hashes */
   memcpy(Prevhash, Cblockhash, HASHLEN);
   if(readtrailer(&bt, cmd) != VEOK)
      return error("do_neogen(): cannot read NG block hash");
   memcpy(Cblockhash, bt.bhash, HASHLEN);
   Eon++;
   return VEOK;
}
