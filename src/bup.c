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
#include "tx.h"
#include "tfile.h"
#include "peer.h"
#include "peach.h"
#include "ledger.h"
#include "global.h"
#include "error.h"
#include "bval.h"
#include "bcon.h"

/* external support */
#include <string.h>
#include <stdlib.h>
#include "extmath.h"
#include "extlib.h"

void print_bup(BTRAILER *bt)
{
   word32 bnum, btxs, btime, bdiff;
   char haiku[256], hash[10], *haiku1, *haiku2, *haiku3, *cp;
   char *type = "Update";

   /* determine alternate update type */
   if (get32(bt->tcount) == 0) type = "Pseudo";
   if (bt->bnum[0] == 0) type = "Neogen";

   /* prepare block stats */
   bnum = get32(bt->bnum);
   btxs = get32(bt->tcount);
   btime = get32(bt->stime) - get32(bt->time0);
   bdiff = get32(bt->difficulty);
   /* print haiku if normal block */
   if (!Insyncup && get32(bt->tcount) && bt->bnum[0]) {
      trigg_expand(bt->nonce, haiku);
      /* remove backspace char -- causes issues in journalctl logs */
      while (( cp = strchr(haiku, '\b') )) {
         if (cp == haiku) memmove(cp, cp + 1, strlen(cp));
         else memmove(cp - 1, cp + 1, strlen(cp));
      }
      /* split haiku into lines for printing */
      haiku1 = strtok(haiku, "\n");
      haiku2 = strtok(&haiku1[strlen(haiku1) + 1], "\n");
      haiku3 = strtok(&haiku2[strlen(haiku2) + 1], "\n");
      plog("\n/) %s\n(=: %s\n\\) %s\n", haiku1, haiku2, haiku3);
      /* print block update and details */
      plog("Time: %" P32u "s, Diff: %" P32u ", Txs: %" P32u,
         btime, bdiff, btxs);
   }
   /* print block identification */
   hash2hex32(bt->bhash, hash);
   plog("%s-block: 0x%" P32x " #%s...", type, bnum, hash);
}  /* end print_bup() */

/**
 * Perform a block validate and update with a blockchain file. Performs
 * txclean() on successfully updated blocks. If block validation fails,
 * or the block does not contain transactions, txclean() is performed
 * without a blockchain file. Ledger updates are performed by taking
 * the ledger transaction file, generated by b_val(), and applying it
 * to the ledger. The ledger file is kept sorted on address.
 * @param fname File name of block to validate/update
 * @returns VEOK on success, else error code
*/
int b_update(char *fname)
{
   BTRAILER bt;
   FILENAME fpath;
   FILENAME cleanfile;
   int ecode;
   char bcfname[21];

   pdebug("updating block...");

   /* check block file exists */
   if (!fexists(fname)) {
      pdebug("%s missing...", fname);
      return VERROR;
   }

   /* Hotfix for critical bug identified on 09/26/19 */
   if (fexists("cblock.lck")) {
      remove("cblock.lck");
   }

   /* validate block (compatible with pseudoblocks) */
   ecode = b_val(fname, "ltran.dat");
   if (ecode != VEOK) {
      perrno("block -> ltran.dat validation FAILURE");
      rename(fname, "block.last.fail");
      fname = NULL;
      goto CLEANUP;
   }

   /* update ledger with (ledger) transactions */
   ecode = le_update("ltran.dat");
   if (ecode != VEOK) {
      perrno("ledger update FAILURE");
      rename(fname, "ltran.last.fail");
      fname = NULL;
      goto CLEANUP;
   }

   /* Everything below this line has to succeed, or else
    * we restart() with an update error.
    * -----------------------------------------------------*/

   /* Update:
    * Cblockhash, Cblocknum, Prevhash, Difficulty, Time0, and tfile.dat
    */
   if (add64(Cblocknum, One, Cblocknum)) {
      restart("new blocknum overflow");
   } else if (read_trailer(&bt, fname) != VEOK) {
      restart("failed to read_trailer()");
   }
   memcpy(Prevhash, Cblockhash, HASHLEN);
   memcpy(Cblockhash, bt.bhash, HASHLEN);
   add_weight(Weight, bt.difficulty[0]);
   /* Update block difficulty */
   Difficulty = next_difficulty(&bt);
   Time0 = get32(bt.stime);
   /* add block trailer to tfile and accept block */
   bnum2fname(Cblocknum, bcfname);
   path_join(fpath, Bcdir, bcfname);
   if (append_tfile(&bt, 1, "tfile.dat") != VEOK) {
      restart("failed to append_tfile()");
   } else if (rename(fname, fpath) != 0) {
      perrno("failed on rename() %s to %s", fname, fpath);
      restart("failed to accept block");
   }

   /* set cleanfile for (final) txclean requirement */
   strncpy(cleanfile, fpath, FILENAME_MAX);
   fname = cleanfile;

   /* update server data */
   remove("cblock.dat");
   remove("mblock.dat");
   if (Ininit == 0) {
      if (Insyncup == 0) {
         Nupdated++;  /* block update counter */
      }
      Utime = time(NULL);  /* update time for watchdog */
   }  /* end if not-Ininit */

   /* print update log */
   if (!Bgflag) print_bup(&bt);

   /* perform neogenesis block update -- as necessary */
   if (Cblocknum[0] == 0xff) {
      /* Neogenensis Block Update:
       * Determine input block b...ff.bc file with Cblocknum.
       * Update Cblockhash, Cblocknum, Prevhash, Eon and tfile.dat
       */
      bnum2fname(Cblocknum, bcfname);
      path_join(fpath, Bcdir, bcfname);
      if (neogen(&bt, "ledger.dat", "ngblock.dat") != VEOK) {
         perrno("neogen() FAILURE");
         restart("failed to neogen()");
      } else if (add64(Cblocknum, One, Cblocknum)) {
         restart("neogenesis blocknum overflow");
      } else if (read_trailer(&bt, "ngblock.dat") != VEOK) {
         restart("failed to read_trailer(ngblock.dat)");
      }
      memcpy(Prevhash, Cblockhash, HASHLEN);
      memcpy(Cblockhash, bt.bhash, HASHLEN);
      Eon++;
      /* add neogenesis block trailer to tfile and accept block */
      bnum2fname(Cblocknum, bcfname);
      path_join(fpath, Bcdir, bcfname);
      if (append_tfile(&bt, 1, "tfile.dat") != VEOK) {
         restart("failed to append_tfile(ngblock.dat)");
      } else if (rename("ngblock.dat", fpath) != 0) {
         perrno("failed on rename() ngblock.dat to %s", fpath);
         restart("failed to accept block");
      }
      /* check CAROUSEL() -- REMOVE SANCTUARY TRIGGER FOR NOW
      if (get32(Cblocknum) == Lastday) {
         plog("Lastday 0x%x.  Carousel begins...", Lastday);
         if (le_renew() != VEOK) {
            perrno("Carousel failure");
            restart("failed to le_renew()");
         }
         if (txclean("txclean.dat", NULL) != VEOK) {
            pwarn("forcing clean TX queue...");
            remove("txclean.dat");
         }
      } */
      /* print neogen update */
      if(!Bgflag) print_bup(&bt);
   }

   /* update pinklists */
   if ((Cblocknum[0] & EPOCHMASK) == 0) purge_epoch();
   mergepinklists();
   /* trigger synchronous external update - if available */
   if (Ininit == 0 && fexists("../update-external.sh")) {
      system("../update-external.sh");
   }

CLEANUP:
   /* NOTE: txclean() should occur on ALL update results, both as a form
    * of removing any issues withe txclean.dat AND to update tx_nonce
    */

   /* ... combine transaction queues before a clean */
   if (fexists("txq1.dat")) {
      system("cat txq1.dat >>txclean.dat 2>/dev/null");
      remove("txq1.dat");
      /* txq1.dat is empty now */
      Txcount = 0;
   }
   if (fexistsnz("txclean.dat")) {
      /* fname was set to cleanfile for transaction blocks, or
       * NULL for non-transaction blocks and update failures
       */
      if (txclean("txclean.dat", fname) != VEOK) {
         perrno("post-update txclean() FAILURE");
         pwarn("txclean.dat integrity unknown, deleting...");
         remove("txclean.dat");
      }
   }

   return VEOK;
}  /* end b_update() */

/* end include guard */
#endif
