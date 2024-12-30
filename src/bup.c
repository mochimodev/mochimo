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

void print_bup(BTRAILER *bt, char *solvestr)
{
   const char *bsdd_haiku_tail = " \b-- ";
   word32 bnum, btxs, btime, bdiff;
   char haiku[256], *haiku1, *haiku2, *haiku3;
   char hash[10];
   char *cp;

   /* prepare block stats */
   bnum = get32(bt->bnum);
   btxs = get32(bt->tcount);
   btime = get32(bt->stime) - get32(bt->time0);
   bdiff = get32(bt->difficulty);
   /* print haiku if non-pseudo block */
   if (!Insyncup && btxs) {
      /* expand and split haiku into lines for printing */
      trigg_expand(bt->nonce, haiku);
      haiku1 = strtok(haiku, "\n");
      haiku2 = strtok(&haiku1[strlen(haiku1) + 1], "\n");
      haiku3 = strtok(&haiku2[strlen(haiku2) + 1], "\n");
      /* remove backspace char -- causes issues in journalctl logs */
      cp = strstr(haiku2, bsdd_haiku_tail);
      if (cp != NULL) memmove(cp, cp + 2, 4);
      printf("\n/) %s\n(=: %s\n\\)   %s\n", haiku1, haiku2, haiku3);
      /* print block update and details */
      plog("Time: %" P32u "s, Diff: %" P32u ", Txs: %" P32u,
         btime, bdiff, btxs);
   }
   /* print block identification */
   hash2hex32(bt->bhash, hash);
   plog("%s-block: 0x%" P32x " #%s...", solvestr, bnum, hash);
}  /* end print_bup() */

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
   word32 bnum, len;
   int ecode;
   char fpath[FILENAME_MAX], *solvestr;
   char bcfname[21];
   char bhash[10];
   char *bcfile_clean;
   FILE *fp;

   /* init */
   solvestr = NULL;
   bcfile_clean = NULL;

   pdebug("updating block...");

   /* check block file exists */
   if (!fexists(fname)) {
      pdebug("%s missing...", fname);
      return VERROR;
   }

   /* check "their" blocks (mode == 0) for pseudo-block */
   if (mode == 0) {
      fp = fopen(fname, "rb");
      if (fp == NULL) return VERROR;
      ecode = fread(&len, 4, 1, fp);
      if (feof(fp)) set_errno(EMCM_EOF);
      fclose(fp);
      if (ecode != 1) return VERROR;
      if (len == 4) mode = 2;
   }

   /* separate validation process for pseudo-block */
   if (mode != 2) {
      if (mode == 0) remove("mblock.dat");
      tag_free(); /* Erase Tagidx[] to be rebuilt on next tag_find() */
      /* Hotfix for critical bug identified on 09/26/19 */
      if (fexists("cblock.lck")) {
         remove("cblock.lck");
         solvestr = "Pushed";
      }
      /* perform block validation, update and tx queue clean... */
      ecode = b_val(fname, "ltran.dat");
      if (ecode == VEOK) {
         ecode = le_update("ltran.dat");
         if (ecode != VEOK) perrno("ledger update FAILURE");
         else bcfile_clean = fname;
         /* ... txclean() with block file only occurs on success */
      } else perrno("block -> ltran.dat validation FAILURE");
   } else {
      ecode = p_val(fname);
      if (ecode != VEOK) perrno("pseudoblock validation FAILURE");
      /* ... nothing to update within pseudoblocks */
   }

   /* check ecode result of block validation and update */
   if (ecode != VEOK) {
      pdebug("(validate -> update) failure");
      if (mode != 0) {
         rename(fname, "mblock.dat.fail");
         rename("ltran.dat.last", "ltran.dat.fail");
         remove("mblock.dat");
      }
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
   } else if (bcfile_clean) bcfile_clean = fpath;

   /* update server data */
   remove("cblock.dat");
   if (Ininit == 0) {
      if (Insyncup == 0) {
         Nupdated++;  /* block update counter */
      } else solvestr = "Synced";
      Utime = time(NULL);  /* update time for watchdog */
   }  /* end if not-Ininit */

   /* print update log */
   if (mode == 2) solvestr = "Pseudo";
   if (solvestr == NULL) solvestr = "Update";
   if (!Bgflag) print_bup(&bt, solvestr);

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
      /* check CAROUSEL() */
      if (get32(Cblocknum) == Lastday) {
         tag_free();  /* Erase old in-memory Tagidx[] */
         plog("Lastday 0x%x.  Carousel begins...", Lastday);
         if (le_renew() != VEOK) {
            perrno("Carousel failure");
            restart("failed to le_renew()");
         }
         /* clean the tx queue (again), no bc file */
         if (txclean("txclean.dat", NULL) != VEOK) {
            pwarn("forcing clean TX queue...");
            remove("txclean.dat");
         }
      }
      /* print block update */
      if(!Bgflag) {
         bnum = get32(bt.bnum);
         hash2hex32(bt.bhash, bhash);
         plog("Neogenesis: 0x%" P32x " #%s...", bnum, bhash);
      }
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

   /* combine transaction queues before a clean */
   if (fexists("txq1.dat")) {
      system("cat txq1.dat >>txclean.dat 2>/dev/null");
      remove("txq1.dat");
      /* txq1.dat is empty now */
      Txcount = 0;
   }

   /* reconstruct candidate block if transactions exist in "clean" queue */
   if (fexistsnz("txclean.dat")) {
      if (txclean("txclean.dat", bcfile_clean) != VEOK) {
         perrno("post-update txclean() FAILURE");
         pwarn("txclean.dat integrity unknown, deleting...");
         remove("txclean.dat");
      }
      /* check txclean.dat contains transactions */
      if (fexistsnz("txclean.dat")) {
         if (b_con("txclean.dat") != VEOK) {
            perrno("post-update b_con() FAILURE");
         }
      }
   }

   return ecode;
}  /* end b_update() */

/* end include guard */
#endif
