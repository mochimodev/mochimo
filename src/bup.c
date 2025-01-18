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
   const char *blob_trigger = " \b-- ";
   const char *blob_fix = "--";
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
      cp = strstr(haiku2, blob_trigger);
      if (cp != NULL) strncpy(cp, blob_fix, strlen(blob_fix) + 1);
      printf("\n/) %s\n(=: %s\n\\) %s\n", haiku1, haiku2, haiku3);
      /* print block update and details */
      plog("Time: %" P32u "s, Diff: %" P32u ", Txs: %" P32u,
         btime, bdiff, btxs);
   }
   /* print block identification */
   hash2hex32(bt->bhash, hash);
   plog("%s-block: 0x%" P32x " #%s...", solvestr, bnum, hash);
}  /* end print_bup() */

/**
 * Adjust a block to accommodate a new mining address. Uses existing state
 * and updates the necessary components to reflect the new mining address.
 * @param fp File pointer to candidate block file
 * @param maddr New mining address to update block with
 * @return (int) value representing operation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int b_adjust_maddr_fp(FILE *fp, void *maddr)
{
   TXENTRY txc;   /* for holding transaction data */
   BTRAILER bt;   /* block trailers are fixed length */
   BHEADER bh;    /* the minimal length block header */
   word8 *mtree;  /* malloc'd merkle tree list */
   size_t tcount; /* malloc'd transaction count */
   size_t j;      /* loop counter and txclean count */

   /* init pointers */
   mtree = NULL;

   /* read block trailer */
   if (fseek(fp, (long) -(sizeof(BTRAILER)), SEEK_END) != 0) goto ERROR_CLEANUP;
   if (fread(&bt, sizeof(BTRAILER), 1, fp) != 1) goto RDERR_CLEANUP;
   tcount = get32(bt.tcount);

   /* read/update block header */
   if (fseek(fp, 0L, SEEK_SET) != 0) goto ERROR_CLEANUP;
   if (fread(&bh, sizeof(BHEADER), 1, fp) != 1) goto RDERR_CLEANUP;
   memcpy(bh.maddr, maddr, sizeof(bh.maddr));
   if (fseek(fp, 0L, SEEK_SET) != 0) goto ERROR_CLEANUP;
   if (fwrite(&bh, sizeof(BHEADER), 1, fp) != 1) goto ERROR_CLEANUP;

   /* ... fp left at start of TXENTRY[] ...*/

   /* malloc merkle tree (+1 for header data) */
   mtree = malloc((tcount + 1) * HASHLEN);
   if (mtree == NULL) goto ERROR_CLEANUP;

   /* begin merkel hash with mining address + reward */
   sha256(bh.maddr /* + bh.mreward */, sizeof(bh.maddr) + 8, mtree);

   /* read transactions from txclean.dat using sorted TXPOS array */
   for (j = 1; j < (tcount + 1); j++) {
      /* add transaction id to merkel tree (++ prefix for miner) */
      if (tx_fread(&txc, fp) != VEOK) goto RDERR_CLEANUP;
      memcpy(&mtree[j * HASHLEN], txc.tx_id, HASHLEN);
   }  /* end for() */

   /* (re)compute merkel root hash straight into the trailer */
   merkle_root(mtree, tcount + 1, bt.mroot);
   /* ... bt.nonce left zero'd (not known) */
   /* ... bt.stime left zero'd (not known) */
   /* ... bt.bhash left zero'd (not known) */

   /* write trailer to file */
   if (fwrite(&bt, sizeof(BTRAILER), 1, fp) != 1) goto ERROR_CLEANUP;

   /* cleanup */
   free(mtree);

   /* success */
   return VEOK;

   /* cleanup / error handling */
RDERR_CLEANUP:
   if (!ferror(fp)) set_errno(EMCM_EOF);
ERROR_CLEANUP:
   if (mtree) free(mtree);

   return VERROR;
}  /* end b_adjust_maddr_fp() */

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

   /* ... combine transaction queues before a clean */
   if (fexists("txq1.dat")) {
      system("cat txq1.dat >>txclean.dat 2>/dev/null");
      remove("txq1.dat");
      /* txq1.dat is empty now */
      Txcount = 0;
   }
   if (fexistsnz("txclean.dat")) {
      if (txclean("txclean.dat", bcfile_clean) != VEOK) {
         perrno("post-update txclean() FAILURE");
         pwarn("txclean.dat integrity unknown, deleting...");
         remove("txclean.dat");
      }
   }

   return ecode;
}  /* end b_update() */

/* end include guard */
#endif
