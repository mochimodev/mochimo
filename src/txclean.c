/**
 * @private
 * @headerfile txclean.h <txclean.h>
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_TXCLEAN_C
#define MOCHIMO_TXCLEAN_C


#include "txclean.h"

/* internal support */
#include "util.h"
#include "types.h"
#include "tag.h"
#include "sort.h"
#include "ledger.h"
#include "global.h"

/* external support */
#include <string.h>
#include <stdlib.h>
#include "extmath.h"
#include "extlib.h"

#define DEBUG_LE(...)   \
   { pdebug("txclean_le(): %s, drop %s...", __VA_ARGS__); continue; }

/**
 * Remove TX's from "txclean.dat", that are in a blockchain file.
 * @param fname File name of validated block
 * @returns VEOK on success or if no txclean file, else VERROR
*/
int txclean_bc(char *fname)
{
   TXQENTRY tx, txc;            /* Holds one transaction in the array */
   BTRAILER bt;
   BHEADER bh;
   FILE *fp, *fpout;       /* input txclean and output txq file pointers */
   FILE *bfp;              /* input block file pointer */
   clock_t ticks;
   word32 hdrlen;          /* for block header length */
   word32 *idx;
   word32 j;
   word32 diff[2];
   word32 bcount, nout;    /* block input and temp file output counter */
   int cond, ecode;

   void *ap, *bp;    /* comparison pointers */

   /* init */
   ticks = clock();

   /* check txclean exists */
   if (!fexists("txclean.dat")) {
      pdebug("txclean_bc(): no txclean file, txclean.dat, skipping...");
      return VEOK;
   }

   /* build sorted index Txidx[] from txclean.dat */
   if (sorttx("txclean.dat") != VEOK) {
      mError(FAIL, "txclean_bc(): bad sorttx(txclean.dat)");
   }

   /* open validated block file, read fixed length header and check */
   bfp = fopen(fname, "rb");
   if (bfp == NULL) {
      mErrno(FAIL_IN, "txclean_bc(): failed to fopen(%s)", fname);
   } else if (fread(&hdrlen, 4, 1, bfp) != 1) {
      mError(FAIL_IO, "txclean_bc(): failed to fread(hdrlen)");
   } else if (hdrlen != sizeof(bh)) {
      mError(FAIL_IO, "txclean_bc(): bad hdrlen");
   }
   /* read block header and trailer */
   if (fseek(bfp, 0, SEEK_SET) != 0) {
      mErrno(FAIL_IO, "txclean_bc(): failed to fseek(SET)");
   } else if (fread(&bh, sizeof(BHEADER), 1, bfp) != 1) {
      mError(FAIL_IO, "txclean_bc(): failed to fread(bh)");
   } else if (fseek(bfp, -(sizeof(BTRAILER)), SEEK_END) != 0) {
      mErrno(FAIL_IO, "txclean_bc(): failed to fseek(END-BTRAILER)");
   } else if (fread(&bt, sizeof(BTRAILER), 1, bfp) != 1) {
      mError(FAIL_IO, "txclean_bc(): failed to fread(bt)");
   }

   /* check Cblocknum alignment with block number */
   if (sub64(bt.bnum, Cblocknum, diff) || diff[0] != 1 || diff[1] != 0) {
      mError(FAIL_IO, "txclean_bc(): bt.bnum - Cblocknum != 1");
   }

   /* re-open the clean TX queue to read */
   fp = fopen("txclean.dat", "rb");
   if (fp == NULL) {
      mErrno(FAIL_IO, "txclean_bc(): failed to fopen(txclean.dat)");
   }
   /* create new clean TX queue */
   fpout = fopen("txq.tmp", "wb");
   if (fpout == NULL) {
      mErrno(FAIL_OUT, "txclean_bc(): failed to fopen(txq.tmp)");
   }

   /***** Read Merkel Block Array from new block *****/
   if (fseek(bfp, hdrlen, SEEK_SET) != 0) {
      mErrno(FAIL_IO2, "txclean_bc(): failed to fseek(bfp, SET)");
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
   bcount = 0;  /* block counter */
   idx = Txidx;  /* *idx is its index */
   /* read transactions from the block array */
   for (j = 0; j < Ntx && fread(&tx, sizeof(TXQENTRY), 1, bfp); ) {
      bcount++;  /* count transactions in block */
      do {  /* check if block tx matches clean tx... */
         cond = memcmp(tx.tx_id, &Tx_ids[*idx * HASHLEN], HASHLEN);
         /* if Merkel Block TX_ID is higher, pass clean TX to temp file */
         if (cond > 0) {
            if (fseek(fp, *idx * sizeof(TXQENTRY), SEEK_SET) != 0) {
               mErrno(FAIL_IO2, "txclean_bc(): failed to fseek(fp, SET)");
            } else if (fread(&txc, sizeof(TXQENTRY), 1, fp) != 1) {
               mError(FAIL_IO2, "txclean_bc(): failed to fread(tx)");
            } else if (fwrite(&txc, sizeof(TXQENTRY), 1, fpout) != 1) {
               mError(FAIL_IO2, "txclean_bc(): failed to fwrite(tx)");
            } else pdebug("txclean_bc(): pass %s...", addr2str(txc.src_addr));
            nout++;  /* count output records to temp file -- new txclean */
         }
         /* skip dup transaction ids */
         if (cond >= 0) {
            if (fseek(fp, *idx * sizeof(TXQENTRY), SEEK_SET) == 0 &&
                  fread(&txc, sizeof(TXQENTRY), 1, fp)) {
               pdebug("txclean_bc(): drop %s...", addr2str(tx.src_addr));
            } else {
               pdebug("txclean_bc(): drop tx_id %s...",
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
            pdebug("txclean_bc(): dup tx_id, drop %s...",
               addr2str(&Tx_ids[*idx * HASHLEN]));
         } else {
            pdebug("txclean_bc(): dup tx_id, drop tx_id %s...",
               addr2str(&Tx_ids[*idx * HASHLEN]));
         }
         continue;
      }
      /* Read clean TX in sorted order using index. */
      if (fseek(fp, *idx * sizeof(TXQENTRY), SEEK_SET) != 0) {
         mErrno(FAIL_IO2, "txclean_bc(): failed to (re)fseek(fp, SET)");
      } else if (fread(&tx, sizeof(TXQENTRY), 1, fp) != 1) {
         mError(FAIL_IO2, "txclean_bc(): failed to (re)fread(tx)");
      } else if (fwrite(&tx, sizeof(TXQENTRY), 1, fpout) != 1) {
         mError(FAIL_IO2, "txclean_bc(): failed to (re)fwrite(tx)");
      } else pdebug("txclean_bc(): keep %s...", addr2str(tx.src_addr));
      nout++;
   }  /* end for j */

   if (bcount > get32(bt.tcount)) {  /* should never happen! */
      mError(FAIL_IO2, "txclean_bc(): bad tcount in new block");
   }

   /* cleanup */
   fclose(fpout);   /* txq.tmp temp file */
   fclose(fp);      /* txclean.dat */
   fclose(bfp);     /* block */
   free(Tx_ids);
   free(Txidx);
   Tx_ids = NULL;
   Txidx = NULL;

   remove("txclean.dat");
   if (rename("txq.tmp", "txclean.dat") != 0) {
      mErrno(FAIL, "txclean_bc(): failed to move txq.dat to txclean.dat");
   }

   /* clean TX queue is updated */
   pdebug("txclean_bc(): wrote %u/%u entries to txclean.dat", nout, Ntx);
   pdebug("txclean_bc(): completed in %gs", diffclocktime(clock(), ticks));

   /* success */
   return VEOK;

   /* failure / error handling */
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

   return ecode;
}  /* end txclean_bc() */



/**
 * Remove bad TX's from a txclean file based on a ledger file. Uses
 * "ledger.dat" as (input) ledger file, "txq.tmp" as temporary (output)
 * txclean file and renames to "txclean.dat" on success.
 * @returns VEOK on success, else VERROR
 * @note Nothing else should be using the ledger.
 * @note Leaves ledger.dat open on return.
*/
int txclean_le(void)
{
   LENTRY src_le;       /* for le_find() */
   TXQENTRY tx;         /* Holds one transaction in the array */
   MTX *mtx;
   FILE *fp, *fpout;    /* txclean.dat and txq.tmp */
   clock_t ticks;
   word32 nout, tnum;   /* temp file output record counter */
   word32 total[2];
   word8 addr[TXADDRLEN];
   int ecode, j;

   /* init */
   ticks = clock();

   /* check txclean exists */
   if (!fexists("txclean.dat")) {
      pdebug("txclean_bc(): no txclean file txclean.dat, ignoring...");
      return VEOK;
   }

   /* open clean TX queue, new (temp) clean TX queue, and ledger */
   fp = fopen("txclean.dat", "rb");
   if (fp == NULL) mErrno(FAIL, "txclean_le(): cannot open txclean");
   fpout = fopen("txq.tmp", "wb");
   if (fpout == NULL) mErrno(FAIL2, "txclean_le(): cannot open txq");
   if (le_open("ledger.dat", "rb") != VEOK) {
      mError(FAIL3, "txclean_le(): failed to le_open(ledger.dat)");
   }

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
         pdebug("txclean_le(): MTX detected...");
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
      }
      /* write TX to new queue */
      if (fwrite(&tx, sizeof(TXQENTRY), 1, fpout) != 1) {
         mError(FAIL3, "txclean_le(): failed to fwrite(tx): TX#%u", tnum);
      }
      nout++;
   }  /* end for (nout = tnum = 0... */

   fclose(fp);
   fclose(fpout);
   remove("txclean.dat");
   if (nout == 0) {
      remove("txq.tmp");  /* remove empty temp file */
      pdebug("txclean_le(): %s is empty", "txclean.dat");
   } else if (rename("txq.tmp", "txclean.dat") != VEOK) {
      mError(FAIL, "txclean_le(): failed to move txq.dat to txclean.dat");
   }

   pdebug("txclean_le(): wrote %u/%u entries to txclean.dat", nout, tnum);
   pdebug("txclean_le(): completed in %gs", diffclocktime(clock(), ticks));

   /* success */
   return VEOK;

   /* failure / error handling */
FAIL3:
   fclose(fpout);
   remove("txq.tmp");
FAIL2:
   fclose(fp);
FAIL:

   return ecode;
}  /* end txclean_le() */

/* end include guard */
#endif
