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

#define DEBUG_LE(...)   \
   { pdebug("txclean(): %s, drop %s...", __VA_ARGS__); continue; }

/**
 * Remove bad TX's from a txclean file, optionally based on a blockchain
 * file, and the ledger file. Uses "ledger.dat" as (input) ledger file,
 * "txq.tmp" as temporary (output)
 * txclean file and renames to "txclean.dat" on success.
 * @param bcfname Filename of blockchain file to check against (optional)
 * @returns VEOK on success, else VERROR
 * @note Nothing else should be using the ledger.
 * @note Leaves ledger.dat open on return.
*/
int txclean(char *bcfname)
{
   TXQENTRY txc;           /* passes transactions from input to output */
   TXQENTRY tx;            /* Holds one transaction in the array */
   BTRAILER bt;            /* holds block trailer data of bcfname */
   LENTRY src_le;          /* for le_find() */
   FILE *fp, *fpout;       /* input/output txclean file pointers */
   FILE *bfp;              /* input block file pointer */
   MTX *mtx;               /* for MTX checks */
   word32 *idx;
   word32 j;               /* unsigned iteration and comparison */
   word32 hdrlen;          /* for block header length */
   word32 diff[2];
   word32 total[2];
   word32 btx, nout, tnum; /* transaction record counters */
   word8 addr[TXADDRLEN];  /* for tag_find() (MTX checks) */
   clock_t ticks;
   int cond, ecode;

   void *ap, *bp;    /* comparison pointers */

   /* init */
   ticks = clock();
   ecode = VEOK;

   /* check txclean exists AND has transactions to clean*/
   if (!fexists("txclean.dat")) {
      pdebug("txclean(): nothing to clean, done...");
      return VEOK;
   }

   /* check blockchain filename is supplied */
   if (bcfname == NULL) {
      pdebug("txclean(): no bc file, skipping...");
   } else {
      /* build sorted index Txidx[] from txclean.dat */
      if (sorttx("txclean.dat") != VEOK) {
         mError(BCFAIL, "txclean(bc): bad sorttx(txclean.dat)");
      }
      /* open validated block file, read fixed length header and check */
      bfp = fopen(bcfname, "rb");
      if (bfp == NULL) {
         mErrno(BCFAIL_IN, "txclean(bc): failed to fopen(%s)", bcfname);
      } else if (fread(&hdrlen, 4, 1, bfp) != 1) {
         mError(BCFAIL_IO, "txclean(bc): failed to fread(hdrlen)");
      } else if (hdrlen != sizeof(BHEADER)) {
         mError(BCFAIL_IO, "txclean(bc): bad hdrlen");
      }
      /* seek to and read block trailer */
      if (fseek(bfp, -(sizeof(BTRAILER)), SEEK_END) != 0) {
         mErrno(BCFAIL_IO, "txclean(bc): failed to fseek(END-BTRAILER)");
      } else if (fread(&bt, sizeof(BTRAILER), 1, bfp) != 1) {
         mError(BCFAIL_IO, "txclean(bc): failed to fread(bt)");
      }
      /* check Cblocknum alignment with block number */
      if (sub64(bt.bnum, Cblocknum, diff) || diff[0] != 1 || diff[1] != 0) {
         mError(BCFAIL_IO, "txclean(bc): bt.bnum - Cblocknum != 1");
      }
      /* re-open the clean TX queue to read */
      fp = fopen("txclean.dat", "rb");
      if (fp == NULL) {
         mErrno(BCFAIL_IO, "txclean(bc): failed to fopen(txclean.dat)");
      }
      /* create new clean TX queue */
      fpout = fopen("txq.tmp", "wb");
      if (fpout == NULL) {
         mErrno(BCFAIL_OUT, "txclean(bc): failed to fopen(txq.tmp)");
      }

      /***** Read Merkel Block Array from new block *****/
      if (fseek(bfp, hdrlen, SEEK_SET) != 0) {
         mErrno(BCFAIL_IO2, "txclean(bc): failed to fseek(bfp, SET)");
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
                  mErrno(BCFAIL_IO2, "txclean(bc): failed to fseek(fp, SET)");
               } else if (fread(&txc, sizeof(TXQENTRY), 1, fp) != 1) {
                  mError(BCFAIL_IO2, "txclean(bc): failed to fread(tx)");
               } else if (fwrite(&txc, sizeof(TXQENTRY), 1, fpout) != 1) {
                  mError(BCFAIL_IO2, "txclean(bc): failed to fwrite(tx)");
               } else pdebug("txclean(bc): keep %s...", addr2str(txc.src_addr));
               nout++;  /* count output records to temp file -- new txclean */
            }
            /* skip dup transaction ids */
            if (cond >= 0) {
               if (fseek(fp, *idx * sizeof(TXQENTRY), SEEK_SET) == 0 &&
                     fread(&tx, sizeof(TXQENTRY), 1, fp)) {
                  pdebug("txclean(bc): drop %s...", addr2str(tx.src_addr));
               } else {
                  pdebug("txclean(bc): drop tx_id %s...",
                     addr2str(&Tx_ids[*idx * HASHLEN]));
               }
               do {  /* break on end of clean TX file or non-dup tx */
                  j++;
                  idx++;
                  ap = (void *) &Tx_ids[idx[-1] * HASHLEN];
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
         /* Check for dups in txclean.dat */
         ap = (void *) &Tx_ids[idx[-1] * HASHLEN];
         bp = (void *) &Tx_ids[*idx * HASHLEN];
         if (j > 0 && memcmp(ap, bp, HASHLEN) == 0) {
            if (fseek(fp, *idx * sizeof(TXQENTRY), SEEK_SET) == 0 &&
                  fread(&tx, sizeof(TXQENTRY), 1, fp)) {
               pdebug("txclean(bc): dup tx_id, drop %s...",
                  addr2str(&Tx_ids[*idx * HASHLEN]));
            } else {
               pdebug("txclean(bc): dup tx_id, drop tx_id %s...",
                  addr2str(&Tx_ids[*idx * HASHLEN]));
            }
            continue;
         }
         /* Read clean TX in sorted order using index. */
         if (fseek(fp, *idx * sizeof(TXQENTRY), SEEK_SET) != 0) {
            mErrno(BCFAIL_IO2, "txclean(bc): failed to (re)fseek(fp, SET)");
         } else if (fread(&tx, sizeof(TXQENTRY), 1, fp) != 1) {
            mError(BCFAIL_IO2, "txclean(bc): failed to (re)fread(tx)");
         } else if (fwrite(&tx, sizeof(TXQENTRY), 1, fpout) != 1) {
            mError(BCFAIL_IO2, "txclean(bc): failed to (re)fwrite(tx)");
         } else pdebug("txclean(bc): keep(2) %s...", addr2str(tx.src_addr));
         nout++;
      }  /* end for j */

      if (btx > get32(bt.tcount)) {  /* should never happen! */
         mError(BCFAIL_IO2, "txclean(bc): bad tcount in new block");
      }

      /* cleanup / error handling */
BCFAIL_IO2:
      fclose(fpout);
BCFAIL_OUT:
      fclose(fp);
BCFAIL_IO:
      fclose(bfp);
BCFAIL_IN:
      free(Tx_ids);
      free(Txidx);
      Tx_ids = NULL;
      Txidx = NULL;
BCFAIL:

      /* check for failures */
      if (ecode) remove("txq.dat");
      else {
         /* perform txclean replacement */
         remove("txclean.dat");
         if (rename("txq.tmp", "txclean.dat") != 0) {
            perr("txclean(bc): failed to move txq.dat to txclean.dat");
         } else pdebug("txclean(bc): wrote %u/%u txs", nout, Ntx);
      }
   }

   /* ensure ledger is open */
   if (le_open("ledger.dat", "rb") != VEOK) {
      mError(LEFAIL, "txclean(): failed to le_open(ledger.dat)");
   }

   /* open clean TX queue and new (temp) clean TX queue */
   fp = fopen("txclean.dat", "rb");
   if (fp == NULL) mErrno(LEFAIL, "txclean(): cannot open txclean");
   fpout = fopen("txq.tmp", "wb");
   if (fpout == NULL) mErrno(LEFAIL2, "txclean(): cannot open txq");

   /* read TX from txclean.dat and process */
   for(nout = tnum = 0; fread(&tx, sizeof(TXQENTRY), 1, fp); tnum++) {
      /* check src in ledger, balances and amounts are good */
      if (le_find(tx.src_addr, &src_le, NULL, TXADDRLEN) == FALSE) {
         DEBUG_LE("bad le_find", addr2str(tx.src_addr));
      } else if (cmp64(tx.tx_fee, Myfee) < 0) {  /* bad tx fee */
         DEBUG_LE("bad tx_fee", addr2str(tx.src_addr));
      } else if (add64(tx.send_total, tx.change_total, total)) {  /* bad amounts */
         DEBUG_LE("bad amounts", addr2str(tx.src_addr));
      } else if (add64(tx.tx_fee, total, total)) {  /* bad total */
         DEBUG_LE("bad total", addr2str(tx.src_addr));
      } else if (cmp64(src_le.balance, total) != 0) {  /* bad balance */
         DEBUG_LE("bad balance", addr2str(tx.src_addr));
      } else if (TX_IS_MTX(&tx) && get32(Cblocknum) >= MTXTRIGGER) {
         pdebug("txclean(): MTX detected...");
         mtx = (MTX *) &tx;
         for(j = 0; j < MDST_NUM_DST; j++) {
            if (iszero(mtx->dst[j].tag, TXTAGLEN)) break;
            memcpy(ADDR_TAG_PTR(addr), mtx->dst[j].tag, TXTAGLEN);
            mtx->zeros[j] = 0;
            /* If dst[j] tag not found, put error code in zeros[] array. */
            if (tag_find(addr, NULL, NULL, TXTAGLEN) != VEOK) {
               mtx->zeros[j] = 1;
            }
         }
      } else if (tag_valid(tx.src_addr, tx.chg_addr, tx.dst_addr,
            NULL) != VEOK) {
         DEBUG_LE("invalidated tags", addr2str(tx.src_addr));
      }
      /* write TX to new queue */
      if (fwrite(&tx, sizeof(TXQENTRY), 1, fpout) != 1) {
         mError(LEFAIL_TX, "txclean(): failed to fwrite(tx): TX#%u", tnum);
      }
      nout++;
   }  /* end for (nout = tnum = 0... */

   /* cleanup / error handling */
LEFAIL_TX:
   fclose(fpout);
LEFAIL2:
   fclose(fp);
LEFAIL:

   /* if no failures */
   if (ecode == VEOK) {
      if (nout == 0) pdebug("txclean(): txclean.dat is empty");
      else if (rename("txq.tmp", "txclean.dat") != VEOK) {
         mError(LEFAIL, "txclean(): failed to move txq.tmp to txclean.dat");
      }

      /* clean TX queue is updated */
      pdebug("txclean(): wrote %u/%u entries to txclean.dat", nout, tnum);
      pdebug("txclean(): completed in %gs", diffclocktime(clock(), ticks));
   }

   /* final cleanup */
   remove("txq.tmp");

   return ecode;
}  /* end txclean() */

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
      ecode = b_val(fname) || le_update();
      /* ... NOTE: le_update() closes reference to the ledger */
      /* clean the queue, regardless of the above result */
      if (txclean(fname) != VEOK) {
         pwarn("b_update(): forcing clean TX queue...");
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
         return perr("b_update(): validate -> ledger update, failure");
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
      plog("└┬ Block %s: 0x%s (%" P32u ")", solvestr,
         val2hex(bt.bnum, 8, bnumstr, 24), get32(bt.bnum));
      plog(" └─ Diff: %u, Time: %us, Txs: %u", bdiff, btime, btxs);
      print("\n");  /* padding*/
   }

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
         /* clean the tx queue (again), no bc file */
         if (txclean(NULL) != VEOK) {
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
         print("<{ neogenesis-block }>\n");
         plog("└┬ Block generated: 0x%s (%" P32u ")",
            bnum2hex(bt.bnum), get32(bt.bnum));
         plog(" └─ %s...", addr2str(Cblockhash));
         print("\n");  /* padding*/
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
