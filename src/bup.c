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
#include "tag.h"
#include "sort.h"
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
   word32 bnum, btxs, btime, bdiff;
   char haiku[256], *haiku1, *haiku2, *haiku3;
   char hash[10];

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
      printf("\n/)  %s\n(=:  %s\n\\)    %s\n", haiku1, haiku2, haiku3);
      /* print block update and details */
      plog("Time: %" P32u "s, Diff: %" P32u ", Txs: %" P32u,
         btime, bdiff, btxs);
   }
   /* print block identification */
   hash2hex32(bt->bhash, hash);
   plog("%s-block: 0x%" P32x " #%s...", solvestr, bnum, hash);
   /* print miner data if enabled */
   if (!Ininit && !Insyncup && !Nominer) {
      read_data(&Hps, sizeof(Hps), "hps.dat");
      printf("Solved: %" P32u "  Hps: %" P32u "\n", Nsolved, Hps);
   }
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
   FILE *fp;

   /* init */
   solvestr = NULL;

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
      /* ... NOTE: le_update() closes server ledger reference */
      ecode = b_val(fname);
      if (ecode == VEOK) ecode = le_update("ledger.dat", "ltran.dat");
      /* ... NOTE: transaction queues should be combined before clean */
      if (fexists("txq1.dat")) {
         system("cat txq1.dat >>txclean.dat 2>/dev/null");
         remove("txq1.dat");
         /* txq1.dat is empty now */
         Txcount = 0;
      }
      /* clean the transaction queue, with the block and the ledger.
       * clean with blockchain file only occurs on successful update.
       * NOTE: txclean() opens server ledger reference.
       */
      if (txclean("txclean.dat", ecode == VEOK ? fname : NULL) != VEOK) {
         perrno("txclean failure");
         pwarn("forcing clean...");
         remove("txclean.dat");
      }
      /* (re)open the ledger, regardless of above results */
      if (le_open("ledger.dat", "rb") != VEOK) {
         restart("failed to reopen ledger after update");
      }
      /* check chain ecode result */
      if (ecode != VEOK) {
         pdebug("(validate -> update) failure");
         if (mode != 0) {
            rename(fname, "mblock.dat.fail");
            rename("ltran.dat.last", "ltran.dat.fail");
            remove("mblock.dat");
         }
         return VERROR;
      }
   } else if (p_val(fname) != VEOK) {
      perr("failed to validate pseudo-block");
      return VERROR;
   }

   /* Everything below this line has to succeed, or else
    * we restart() with an update error.
    * -----------------------------------------------------*/

   /* Update:
    * Cblockhash, Cblocknum, Prevhash, Difficulty, Time0, and tfile.dat
    */
   if (add64(Cblocknum, One, Cblocknum)) {
      restart("new blocknum overflow");
   } else if (readtrailer(&bt, fname) != VEOK) {
      restart("failed to readtrailer()");
   }
   memcpy(Prevhash, Cblockhash, HASHLEN);
   memcpy(Cblockhash, bt.bhash, HASHLEN);
   add_weight(Weight, bt.difficulty[0], bt.bnum);
   /* Update block difficulty */
   Difficulty = set_difficulty(&bt);
   Time0 = get32(bt.stime);
   /* add block trailer to tfile and accept block */
   bnum2fname(Cblocknum, bcfname);
   path_join(fpath, Bcdir, bcfname);
   if (append_tfile(fname, "tfile.dat") != VEOK) {
      restart("failed to append_tfile()");
   } else if (rename(fname, fpath) != 0) {
      perrno("failed on rename() %s to %s", fname, fpath);
      restart("failed to accept block");
   }

   /* update server data */
   remove("cblock.dat");
   if (write_global() != VEOK) {
      restart("failed to write_global()");
   } else if (Ininit == 0) {
      if (Insyncup == 0) {
         if (mode == 1 && solvestr == NULL) {  /* not "pushed" */
            solvestr = "Solved";
            Nsolved++;  /* our block */
            write_data(&Nsolved, 4, "solved.dat");
         }
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
      if (neogen(fpath, "ngblock.dat") != VEOK) {
         restart("failed to neogen()");
      } else if (add64(Cblocknum, One, Cblocknum)) {
         restart("neogenesis blocknum overflow");
      } else if (readtrailer(&bt, "ngblock.dat") != VEOK) {
         restart("failed to readtrailer(ngblock.dat)");
      }
      memcpy(Prevhash, Cblockhash, HASHLEN);
      memcpy(Cblockhash, bt.bhash, HASHLEN);
      Eon++;
      /* add neogenesis block trailer to tfile and accept block */
      bnum2fname(Cblocknum, bcfname);
      path_join(fpath, Bcdir, bcfname);
      if (append_tfile("ngblock.dat", "tfile.dat") != VEOK) {
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
         /* (re)open the ledger, regardless of above results */
         if (le_open("ledger.dat", "rb") != VEOK) {
            restart("failed to reopen ledger after update");
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

   return VEOK;
}  /* end b_update() */

/* end include guard */
#endif
