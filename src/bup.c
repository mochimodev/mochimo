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

#define restart(msg) { palert(msg); kill_services_exit(1); }

int accept_block(char *ublock, word8 *newnum)
{
   char buff[256];
   char cmd[288];
   char bnum[18];

   bnum2hex(newnum, bnum);
   sprintf(buff, "b%s.bc", bnum);
   sprintf(cmd, "%s/b%s.bc", Bcdir, bnum);
   if(fexists(buff) || fexists(cmd)) {
      perr("failed: %s already exists!", buff);
      return VERROR;
   }
   if(rename(ublock, buff) != 0) {
      perrno("failed on rename() %s to %s", ublock, buff);
      return VERROR;
   }
   sprintf(cmd, "mv %s %s", buff, Bcdir);
   if (system(cmd)) return VERROR;
   sprintf(buff, "%s/b%s.bc", Bcdir, bnum);
   if(!fexists(buff)) {
      perr("failed on system(%s): %s missing", cmd, buff);
      return VERROR;
   }
   return VEOK;
}  /* end accept_block() */

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
   plog("%s-block: 0x%" P32x " #%s...",
      solvestr, bnum, hash2hex(bt->bhash, 4, hash));
   /* print miner data if enabled */
   if (!Ininit && !Insyncup && !Nominer) {
      read_data(&Hps, sizeof(Hps), "hps.dat");
      printf("Solved: %" P32u "  Hps: %" P32u "\n", Nsolved, Hps);
   }
}  /* end print_bup() */

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
   int cond;
   char addrhash[10];

   void *ap, *bp;    /* comparison pointers */

   /* init */
   ticks = clock();
   nout = 0;

   /* check txclean exists AND has transactions to clean*/
   if (!fexists("txclean.dat")) {
      pdebug("nothing to clean, done...");
      return VEOK;
   }

   /* build sorted index Txidx[] from txclean.dat */
   if (sorttx("txclean.dat") != VEOK) {
      perr("bad sorttx(txclean.dat)");
      return VERROR;
   }
   /* open validated block file, read fixed length header and check */
   bfp = fopen(bcfname, "rb");
   if (bfp == NULL) {
      perrno("failed to fopen(%s)", bcfname);
      goto CLEANUP;
   } else if (fread(&hdrlen, 4, 1, bfp) != 1) {
      perr("failed to fread(hdrlen)");
      goto CLEANUP_BLK;
   } else if (hdrlen != sizeof(BHEADER)) {
      perr("bad hdrlen");
      goto CLEANUP_BLK;
   }
   /* seek to and read block trailer */
   if (fseek(bfp, -(sizeof(BTRAILER)), SEEK_END) != 0) {
      perrno("failed to fseek(END-BTRAILER)");
      goto CLEANUP_BLK;
   } else if (fread(&bt, sizeof(BTRAILER), 1, bfp) != 1) {
      perr("failed to fread(bt)");
      goto CLEANUP_BLK;
   }
   /* check Cblocknum alignment with block number */
   if (sub64(bt.bnum, Cblocknum, diff) || diff[0] != 1 || diff[1] != 0) {
      perr("bt.bnum - Cblocknum != 1");
      goto CLEANUP_BLK;
   }
   /* re-open the clean TX queue to read */
   fp = fopen("txclean.dat", "rb");
   if (fp == NULL) {
      perrno("failed to fopen(txclean.dat)");
      goto CLEANUP_BLK;
   }
   /* create new clean TX queue */
   fpout = fopen("txq.tmp", "wb");
   if (fpout == NULL) {
      perrno("failed to fopen(txq.tmp)");
      goto CLEANUP_TXC;
   }

   /***** Read Merkel Block Array from new block *****/
   if (fseek(bfp, hdrlen, SEEK_SET) != 0) {
      perrno("failed to fseek(bfp, SET)");
      goto CLEANUP_TXQ;
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
               perrno("failed to fseek(fp, SET)");
               goto CLEANUP_TXQ;
            } else if (fread(&txc, sizeof(TXQENTRY), 1, fp) != 1) {
               perr("failed to fread(tx)");
               goto CLEANUP_TXQ;
            } else if (fwrite(&txc, sizeof(TXQENTRY), 1, fpout) != 1) {
               perr("failed to fwrite(tx)");
               goto CLEANUP_TXQ;
            } else {
               hash2hex(&Tx_ids[*idx * HASHLEN], 4, addrhash);
               pdebug("keep tx_id %s...", addrhash);
            }
            nout++;  /* count output records to temp file -- new txclean */
         }
         /* skip dup transaction ids */
         if (cond >= 0) {
            do {  /* break on end of clean TX file or non-dup tx_id */
               hash2hex(&Tx_ids[*idx * HASHLEN], 4, addrhash);
               pdebug("drop tx_id %s...", addrhash);
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
            hash2hex(&Tx_ids[*idx * HASHLEN], 4, addrhash);
            pdebug("drop dup tx_id, drop tx_id %s...", addrhash);
            continue;
         }
      }
      /* Read clean TX in sorted order using index. */
      if (fseek(fp, *idx * sizeof(TXQENTRY), SEEK_SET) != 0) {
         perrno("failed to (re)fseek(fp, SET)");
         goto CLEANUP_TXQ;
      } else if (fread(&txc, sizeof(TXQENTRY), 1, fp) != 1) {
         perr("failed to (re)fread(tx)");
         goto CLEANUP_TXQ;
      } else if (fwrite(&txc, sizeof(TXQENTRY), 1, fpout) != 1) {
         perr("failed to (re)fwrite(tx)");
         goto CLEANUP_TXQ;
      } else {
         hash2hex(txc.tx_id, 4, addrhash);
         pdebug("keep remaining tx_id %s...", addrhash);
      }
      nout++;
   }  /* end for j */

   if (btx > get32(bt.tcount)) {  /* should never happen! */
      perr("bad tcount in new block");
      goto CLEANUP_TXQ;
   }

   /* cleanup */
   fclose(fpout);
   fclose(fp);
   fclose(bfp);
   sorttx_free();

   remove("txclean.dat");
   if (nout == 0) pdebug("txclean.dat is empty");
   else if (rename("txq.tmp", "txclean.dat") != VEOK) {
      perr("failed to move txq to txclean");
      remove("txq.tmp");
      return VERROR;
   }

   /* clean TX queue is updated */
   pdebug("wrote %u/%u entries to txclean", nout, Ntx);
   pdebug("block level txclean done in %gs", diffclocktime(ticks));

   /* success */
   return VEOK;

   /* cleanup / error handling */
CLEANUP_TXQ:
   fclose(fpout);
   remove("txq.tmp");
CLEANUP_TXC:
   fclose(fp);
CLEANUP_BLK:
   fclose(bfp);
CLEANUP:
   sorttx_free();
   return VERROR;
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
   word32 bnum, len;
   int ecode;
   char bcfname[FILENAME_MAX], *solvestr;
   char bnumhex[18];
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
      if (ecode == VEOK) ecode = le_update();
      /* ... NOTE: transaction queues should be combined before clean */
      if (fexists("txq1.dat")) {
         system("cat txq1.dat >>txclean.dat 2>/dev/null");
         remove("txq1.dat");
         /* txq1.dat is empty now */
         Txcount = 0;
      }
      /* clean the transaction queue, with the block and the ledger */
      /* ... NOTE: blockchain clean only occurs on successful update */
      if (ecode == VEOK && b_txclean(fname) != VEOK) {
         pwarn("b_txclean() failure, forcing clean...");
         remove("txclean.dat");
      }
      /* ... NOTE: le_txclean() opens server ledger reference */
      if (le_txclean() != VEOK) {
         pwarn("le_txclean failure, forcing clean...");
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
   if (append_tfile(fname, "tfile.dat") != VEOK) {
      restart("failed to append_tfile()");
   } else if (accept_block(fname, Cblocknum) != VEOK) {
      restart("failed to accept_block()");
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
      bnum2hex(Cblocknum, bnumhex);
      snprintf(bcfname, FILENAME_MAX, "%s/b%s.bc", Bcdir, bnumhex);
      if (neogen(bcfname, "ngblock.dat") != VEOK) {
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
      if (append_tfile("ngblock.dat", "tfile.dat") != VEOK) {
         restart("failed to append_tfile(ngblock.dat)");
      } else if (accept_block("ngblock.dat", Cblocknum) != VEOK){
         restart("failed to accept_block(ngblock.dat)");
      }
      /* check CAROUSEL() */
      if (get32(Cblocknum) == Lastday) {
         tag_free();  /* Erase old in-memory Tagidx[] */
         if (le_renew()) restart("failed to le_renew()");
         /* clean the tx queue (again), no bc file */
         if (le_txclean() != VEOK) {
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
         hash2hex(bt.bhash, 4, bhash);
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
