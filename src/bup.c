/**
 * @private
 * @headerfile bup.h <bup.h>
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_BUP_C
#define MOCHIMO_BUP_C


#include "bup.h"

/* internal support */
#include "util.h"
#include "txclean.h"
#include "tag.h"
#include "peer.h"
#include "peach.h"
#include "ledger.h"
#include "global.h"
#include "bval.h"
#include "bcon.h"

/* external support */
#include <string.h>
#include "extmath.h"
#include "extlib.h"

/**
 * mode: 0 = their block
 * nonzero = our block
 * specifically: 1 = normal block
 *               2 = pseudo-block
 * Perform a block update with a blockchain file. Removes TX's from
 * "txclean.dat", and updates the ledger by applying ledger transaction
 * deltas, "ltran.dat", generated during b_val(). Uses "ledger.tmp" as
 * temporary (new) ledger file. Ledger file is kept sorted on addr. Ledger
 * transaction file is sorted by sortlt() on  addr+trancode: '-' before 'A'.
 * @param fname File name of validated block: "vblock.dat"
 * @returns VEOK on success, else error code
*/
int b_update(char *fname, int mode)
{
   BTRAILER bt;
   word32 time1, btxs, btime, bdiff;
   char haiku[256], *haiku1, *haiku2, *haiku3;
   char bnumstr[24], *solvestr;
   int ecode;

   /* init */
   solvestr = NULL;

   pdebug("b_update(): updating block...");

   /* check block file exists */
   if (!fexists(fname)) return perr("b_update(): %s missing...", fname);

   /* check for pseudo-block */
   if (gethdrlen(fname) == 4) mode = 2;

   /* separate validation process for pseudo-block */
   if (mode != 2) {
      if (mode == 0) remove("mblock.dat");
      tag_free(); /* Erase Tagidx[] to be rebuilt on next tag_find() */
      /* Hotfix for critical bug identified on 09/26/19 */
      if (fexists("cblock.lck")) {
         remove("cblock.lck");
         solvestr = "pushed";
      }
      /* perform block validation, clean and ledger update chain */
      ecode = b_val(fname) || txclean_bc(fname) || le_update();
      /* ... NOTE: le_update() closes reference to the ledger */
      /* (re)open ledger and clean the queue, regardless of above result */
      le_open("ledger.dat", "rb");
      txclean_le();
      /* check chain ecode result */
      if (ecode != VEOK) {
         if (mode != 0) {
            rename(fname, "mblock.dat.fail");
            rename("ltran.dat.last", "ltran.dat.fail");
            remove("mblock.dat");
         }
         return perr("b_update(): validate-clean-ledger chain failure");
      }
   } else if (p_val(fname) != VEOK) {
      return perr("b_update(): failed to validate pseudo-block");
   }

   /* Everything below this line has to succeed, or else
    * we restart() with an update error.
    * -----------------------------------------------------*/

   /* Update:
    * Cblockhash, Cblocknum, Prevhash, Difficulty, Time0, and tfile.dat
    */
   if (add64(Cblocknum, One, Cblocknum)) {
      restart("b_update(): new blocknum overflow");
   } else if (readtrailer(&bt, fname) != VEOK) {
      restart("b_update(): failed to readtrailer()");
   }
   memcpy(Prevhash, Cblockhash, HASHLEN);
   memcpy(Cblockhash, bt.bhash, HASHLEN);
   Difficulty = get32(bt.difficulty);
   Time0 = get32(bt.time0);
   time1 = get32(bt.stime);
   add_weight(Weight, bt.difficulty[0], bt.bnum);
   /* Update block difficulty */
   Difficulty = set_difficulty(&bt);
   Time0 = time1;
   /* add block trailer to tfile and accept block */
   if (append_tfile(fname, "tfile.dat") != VEOK) {
      restart("b_update(): failed to append_tfile()");
   } else if (accept_block(fname, Cblocknum) != VEOK){
      restart("b_update(): failed to accept_block()");
   }

   /* update server data */
   if (write_global() != VEOK) {
      restart("b_update(): failed to write_global()");
   } else if (Ininit == 0) {
      if (Insyncup == 0) {
         if (mode == 1 && solvestr == NULL) {  /* not "pushed" */
            solvestr = "solved";
            Nsolved++;  /* our block */
            write_data(&Nsolved, 4, "solved.dat");
         }
         Nupdated++;  /* block update counter */
      } else solvestr = "synced";
      Utime = time(NULL);  /* update time for watchdog */
   }  /* end if not-Ininit */

   /* print update log */
   if(!Bgflag) {
      /* prepare block stats */
      if (solvestr == NULL) solvestr = "updated";
      btxs = (unsigned) get32(bt.tcount);
      btime = (unsigned) get32(bt.stime) - get32(bt.time0);
      bdiff = (unsigned) get32(bt.difficulty);
      /* print haiku if non-pseudo block */
      if (mode != 2) {
         /* expand and split haiku into lines for printing */
         trigg_expand(bt.nonce, haiku);
         haiku1 = strtok(haiku, "\n");
         haiku2 = strtok(&haiku1[strlen(haiku1) + 1], "\n");
         haiku3 = strtok(&haiku2[strlen(haiku2) + 1], "\n");
         print("  __/)  %s\n", haiku1);
         print(".(__(=:  %s\n", haiku2);
         print("│   \\)    %s\n", haiku3);
      } else print("<{ pseudo-block }>\n");
      /* print block update and stats */
      print("└┬ Block %s: 0x%s (%" P32u ")\n", solvestr,
         val2hex(bt.bnum, 8, bnumstr, 24), get32(bt.bnum));
      print(" └─ Diff: %u, Time: %us, Txs: %u\n", bdiff, btime, btxs);
      print("\n");  /* padding*/
   } else pdebug("Block %s: 0x%s", solvestr, bnum2hex(bt.bnum));

   /* perform neogenesis block update -- as necessary */
   if (Cblocknum[0] == 0xff) {
      if (neogen() != VEOK) restart("b_update(): failed to neogen()");
      /* Neogenensis Block Update:
       * Cblockhash, Cblocknum, Prevhash, and tfile.dat
       */
      if (add64(Cblocknum, One, Cblocknum)) {
         restart("b_update(): neogenesis blocknum overflow");
      } else if (readtrailer(&bt, "ngblock.dat") != VEOK) {
         restart("b_update(): failed to readtrailer(ngblock.dat)");
      }
      memcpy(Prevhash, Cblockhash, HASHLEN);
      memcpy(Cblockhash, bt.bhash, HASHLEN);
      Eon++;
      /* add neogenesis block trailer to tfile and accept block */
      if (append_tfile("ngblock.dat", "tfile.dat") != VEOK) {
         restart("b_update(): failed to append_tfile(ngblock.dat)");
      } else if (accept_block("ngblock.dat", Cblocknum) != VEOK){
         restart("b_update(): failed to accept_block(ngblock.dat)");
      }
      /* check CAROUSEL() */
      if (get32(Cblocknum) == Lastday) {
         tag_free();  /* Erase old in-memory Tagidx[] */
         if (le_renew()) restart("b_update(): failed to le_renew()");
         txclean_le();  /* clean the tx queue */
         if (le_open("ledger.dat", "rb") != VEOK) {
             restart("b_update(): failed to re-open ledger");
         }
      }
      /* print block update */
      if(!Bgflag) {
         print("<{ neogenesis-block }>\n");
         print("└┬ Block generated: 0x%s (%" P32u ")\n",
            bnum2hex(bt.bnum), get32(bt.bnum));
         print(" └─ %s...\n", addr2str(Cblockhash));
         print("\n");  /* padding*/
      } else pdebug("Block generated: 0x%s", bnum2hex(bt.bnum));
   }

   /* update pinklists */
   if ((Cblocknum[0] & EPOCHMASK) == 0) purge_epoch();
   mergepinklists();
   /* trigger synchronous external update - if available */
   if (Ininit == 0 && fexists("../update-external.sh")) {
      system("../update-external.sh");
   }

   return VEOK;
}  /* end b_update() */

/* end include guard */
#endif
