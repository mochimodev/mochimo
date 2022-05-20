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
#include "tag.h"
#include "sort.h"
#include "peer.h"
#include "peach.h"
#include "ledger.h"
#include "global.h"
#include "bval.h"
#include "bcon.h"

/* external support */
#include <string.h>
#include <stdlib.h>
#include "extmath.h"
#include "extlib.h"

/**
 * Remove bad TX's from a txclean file based on a blockchain file.
 * Uses "ledger.dat" as (input) ledger file, "txq.tmp" as temporary (output)
 * txclean file and renames to "txclean.dat" on success.
 * @param bcfname Filename of blockchain file to check against (optional)
 * @returns VEOK on success, else VERROR
 * @note Nothing else should be using the ledger.
 * @note Leaves ledger.dat open on return.
*/
int b_txclean(char *bcfname)
{
   TXQENTRY txc;           /* passes transactions from input to output */
   TXQENTRY tx;            /* Holds one transaction in the array */
   BTRAILER bt;            /* holds block trailer data of bcfname */
   FILE *fp, *fpout;       /* input/output txclean file pointers */
   FILE *bfp;              /* input block file pointer */
   word32 *idx;
   word32 j;               /* unsigned iteration and comparison */
   word32 hdrlen;          /* for block header length */
   word32 diff[2];
   word32 btx, nout;       /* transaction record counters */
   clock_t ticks;
   int cond, ecode;

   void *ap, *bp;    /* comparison pointers */

   /* init */
   ticks = clock();
   ecode = VEOK;

   /* check txclean exists AND has transactions to clean*/
   if (!fexists("txclean.dat")) {
      pdebug("b_txclean(): nothing to clean, done...");
      return VEOK;
   }

   /* build sorted index Txidx[] from txclean.dat */
   if (sorttx("txclean.dat") != VEOK) {
      mError(FAIL, "b_txclean(): bad sorttx(txclean.dat)");
   }
   /* open validated block file, read fixed length header and check */
   bfp = fopen(bcfname, "rb");
   if (bfp == NULL) {
      mErrno(FAIL_IN, "b_txclean(): failed to fopen(%s)", bcfname);
   } else if (fread(&hdrlen, 4, 1, bfp) != 1) {
      mError(FAIL_IO, "b_txclean(): failed to fread(hdrlen)");
   } else if (hdrlen != sizeof(BHEADER)) {
      mError(FAIL_IO, "b_txclean(): bad hdrlen");
   }
   /* seek to and read block trailer */
   if (fseek(bfp, -(sizeof(BTRAILER)), SEEK_END) != 0) {
      mErrno(FAIL_IO, "b_txclean(): failed to fseek(END-BTRAILER)");
   } else if (fread(&bt, sizeof(BTRAILER), 1, bfp) != 1) {
      mError(FAIL_IO, "b_txclean(): failed to fread(bt)");
   }
   /* check Cblocknum alignment with block number */
   if (sub64(bt.bnum, Cblocknum, diff) || diff[0] != 1 || diff[1] != 0) {
      mError(FAIL_IO, "b_txclean(): bt.bnum - Cblocknum != 1");
   }
   /* re-open the clean TX queue to read */
   fp = fopen("txclean.dat", "rb");
   if (fp == NULL) {
      mErrno(FAIL_IO, "b_txclean(): failed to fopen(txclean.dat)");
   }
   /* create new clean TX queue */
   fpout = fopen("txq.tmp", "wb");
   if (fpout == NULL) {
      mErrno(FAIL_OUT, "b_txclean(): failed to fopen(txq.tmp)");
   }

   /***** Read Merkel Block Array from new block *****/
   if (fseek(bfp, hdrlen, SEEK_SET) != 0) {
      mErrno(FAIL_IO2, "b_txclean(): failed to fseek(bfp, SET)");
   }

   /* Remove TX_ID's from clean TX queue that are in the new block.
   * Merkel Array in new block is already sorted on TX_ID;
   * bval checks this in foreign blocks.
   * Above we sorted clean queue, txclean.dat, with sorttx() call.
   *
   * NOTE: end of file check on Merkel Block (bfp) depends on block
   *       trailer being shorter than a TXQENTRY !
   */

   nout = 0;    /* output counter */
   btx = 0;  /* block counter */
   idx = Txidx;  /* *idx is its index */
   /* read transactions from the block array */
   for (j = 0; j < Ntx && fread(&tx, sizeof(TXQENTRY), 1, bfp); ) {
      btx++;  /* count transactions in block */
      do {  /* check if block tx matches clean tx... */
         cond = memcmp(tx.tx_id, &Tx_ids[*idx * HASHLEN], HASHLEN);
         /* if Merkel Block TX_ID is higher, pass clean TX to temp */
         if (cond > 0) {
            if (fseek(fp, *idx * sizeof(TXQENTRY), SEEK_SET) != 0) {
               mErrno(FAIL_IO2, "b_txclean(): failed to fseek(fp, SET)");
            } else if (fread(&txc, sizeof(TXQENTRY), 1, fp) != 1) {
               mError(FAIL_IO2, "b_txclean(): failed to fread(tx)");
            } else if (fwrite(&txc, sizeof(TXQENTRY), 1, fpout) != 1) {
               mError(FAIL_IO2, "b_txclean(): failed to fwrite(tx)");
            } else {
               pdebug("b_txclean(): keep tx_id %s...",
                  addr2str(&Tx_ids[*idx * HASHLEN]));
            }
            nout++;  /* count output records to temp file -- new txclean */
         }
         /* skip dup transaction ids */
         if (cond >= 0) {
            do {  /* break on end of clean TX file or non-dup tx_id */
               pdebug("b_txclean(): drop tx_id %s...",
                  addr2str(&Tx_ids[*idx * HASHLEN]));
               j++;
               ap = (void *) &Tx_ids[*(idx++) * HASHLEN];
               bp = (void *) &Tx_ids[*idx * HASHLEN];
            } while (j < Ntx && memcmp(ap, bp, HASHLEN) == 0);
         }
         /* if (cond <= 0) the block transaction is not in txclean.dat
         * (Maybe the block is foreign, maybe we're done) */
      } while (j < Ntx && cond > 0);
   }  /* end for(j = 0... */

   /* At end of Merkel Block, and if there are remaining transactions,
   * copy the remaining transactions from txclean.dat to temp file */
   for( ; j < Ntx; j++, idx++) {
      if (j > 0) {
         /* Check for dups in txclean.dat */
         ap = (void *) &Tx_ids[idx[-1] * HASHLEN];
         bp = (void *) &Tx_ids[*idx * HASHLEN];
         if (memcmp(ap, bp, HASHLEN) == 0) {
            pdebug("b_txclean(): drop dup tx_id, drop tx_id %s...",
               addr2str(&Tx_ids[*idx * HASHLEN]));
            continue;
         }
      }
      /* Read clean TX in sorted order using index. */
      if (fseek(fp, *idx * sizeof(TXQENTRY), SEEK_SET) != 0) {
         mErrno(FAIL_IO2, "b_txclean(): failed to (re)fseek(fp, SET)");
      } else if (fread(&txc, sizeof(TXQENTRY), 1, fp) != 1) {
         mError(FAIL_IO2, "b_txclean(): failed to (re)fread(tx)");
      } else if (fwrite(&txc, sizeof(TXQENTRY), 1, fpout) != 1) {
         mError(FAIL_IO2, "b_txclean(): failed to (re)fwrite(tx)");
      } else {
         pdebug("b_txclean(): keep remaining tx_id %s...",
            addr2str(txc.tx_id));
      }
      nout++;
   }  /* end for j */

   if (btx > get32(bt.tcount)) {  /* should never happen! */
      mError(FAIL_IO2, "b_txclean(): bad tcount in new block");
   }

   /* cleanup / error handling */
FAIL_IO2:
   fclose(fpout);
FAIL_OUT:
   fclose(fp);
FAIL_IO:
   fclose(bfp);
FAIL_IN:
   free(Tx_ids);
   free(Txidx);
   Tx_ids = NULL;
   Txidx = NULL;
FAIL:

   /* if no failures */
   if (ecode == VEOK) {
      remove("txclean.dat");
      if (nout == 0) pdebug("b_txclean(): txclean.dat is empty");
      else if (rename("txq.tmp", "txclean.dat") != VEOK) {
         mError(FAIL, "b_txclean(): failed to move txq to txclean");
      }

      /* clean TX queue is updated */
      pdebug("b_txclean(): wrote %u/%u entries to txclean", nout, Ntx);
      pdebug("b_txclean(): done in %gs", diffclocktime(clock(), ticks));
   }

   /* final cleanup */
   remove("txq.tmp");

   return ecode;
}  /* end b_txclean() */

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
   word32 bnum, btxs, btime, bdiff;
   char haiku[256], *haiku1, *haiku2, *haiku3, *solvestr;
   char bcfname[FILENAME_MAX];
   int ecode;

   /* init */
   solvestr = NULL;

   pdebug("b_update(): updating block...");

   /* check block file exists */
   if (!fexists(fname)) {
      pdebug("b_update(): %s missing...", fname);
      return VERROR;
   }

   /* check for pseudo-block */
   if (gethdrlen(fname) == 4) mode = 2;

   /* separate validation process for pseudo-block */
   if (mode != 2) {
      if (mode == 0) remove("mblock.dat");
      tag_free(); /* Erase Tagidx[] to be rebuilt on next tag_find() */
      /* Hotfix for critical bug identified on 09/26/19 */
      if (fexists("cblock.lck")) {
         remove("cblock.lck");
         solvestr = "Pushed";
      }
      /* perform block validation and update... then clean tx queue */
      /* ... NOTE: le_update() closes server ledger reference */
      ecode = b_val(fname) || le_update();
      /* ... NOTE: transaction queues should be combined before clean */
      if (fexists("txq1.dat")) {
         system("cat txq1.dat >>txclean.dat 2>/dev/null");
         remove("txq1.dat");
      }
      /* ... NOTE: le_txclean() opens server ledger reference */
      if ((b_txclean(fname) | le_txclean()) != VEOK) {
         pwarn("b_update(): txclean failure, forcing clean TX queue...");
         remove("txclean.dat");
      }
      /* (re)open the ledger, regardless of above results */
      if (le_open("ledger.dat", "rb") != VEOK) {
         restart("b_update(): failed to reopen ledger after update");
      }
      /* check chain ecode result */
      if (ecode != VEOK) {
         if (mode != 0) {
            rename(fname, "mblock.dat.fail");
            rename("ltran.dat.last", "ltran.dat.fail");
            remove("mblock.dat");
         }
         return perr("b_update(): (validate -> update) failure");
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
   add_weight(Weight, bt.difficulty[0], bt.bnum);
   /* Update block difficulty */
   Difficulty = set_difficulty(&bt);
   Time0 = get32(bt.stime);
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
            solvestr = "Solved";
            Nsolved++;  /* our block */
            write_data(&Nsolved, 4, "solved.dat");
         }
         Nupdated++;  /* block update counter */
      } else solvestr = "Synced";
      Utime = time(NULL);  /* update time for watchdog */
   }  /* end if not-Ininit */

   /* print update log */
   if(!Bgflag) {
      if (solvestr == NULL) solvestr = "Update";
      /* prepare block stats */
      bnum = get32(bt.bnum);
      btxs = get32(bt.tcount);
      btime = get32(bt.stime) - get32(bt.time0);
      bdiff = get32(bt.difficulty);
      /* print haiku if non-pseudo block */
      if (!Insyncup && mode != 2) {
         /* expand and split haiku into lines for printing */
         trigg_expand(bt.nonce, haiku);
         haiku1 = strtok(haiku, "\n");
         haiku2 = strtok(&haiku1[strlen(haiku1) + 1], "\n");
         haiku3 = strtok(&haiku2[strlen(haiku2) + 1], "\n");
         print("\n/)  %s\n(=:  %s\n\\)    %s\n", haiku1, haiku2, haiku3);
         /* print block update and details */
         plog("Time: %" P32u "s, Diff: %" P32u ", Txs: %" P32u,
            btime, bdiff, btxs);
      }
      if (mode == 2) solvestr = "Pseudo";
      plog("%s-block: 0x%" P32x " #%s...", solvestr, bnum, addr2str(bt.bhash));
   }

   /* perform neogenesis block update -- as necessary */
   if (Cblocknum[0] == 0xff) {
      /* Neogenensis Block Update:
       * Determine input block b...ff.bc file with Cblocknum.
       * Update Cblockhash, Cblocknum, Prevhash, Eon and tfile.dat
       */
      snprintf(bcfname, FILENAME_MAX, "%s/b%s.bc", Bcdir, bnum2hex(Cblocknum));
      if (neogen(bcfname, "ngblock.dat") != VEOK) {
         restart("b_update(): failed to neogen()");
      } else if (add64(Cblocknum, One, Cblocknum)) {
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
         /* clean the tx queue (again), no bc file */
         if (le_txclean() != VEOK) {
            pwarn("b_update(): forcing clean TX queue...");
            remove("txclean.dat");
         }
         /* (re)open the ledger, regardless of above results */
         if (le_open("ledger.dat", "rb") != VEOK) {
            restart("b_update(): failed to reopen ledger after update");
         }
      }
      /* print block update */
      if(!Bgflag) {
         bnum = get32(bt.bnum);
         plog("Neogenesis: 0x%" P32x " #%s...", bnum, addr2str(bt.bhash));
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
