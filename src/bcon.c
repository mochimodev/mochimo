/**
 * @private
 * @headerfile bcon.h <bcon.h>
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_BCON_C
#define MOCHIMO_BCON_C


#include "bcon.h"

/* internal support */
#include "tx.h"
#include "tfile.h"
#include "sort.h"
#include "global.h"
#include "error.h"

/* external support */
#include <string.h>
#include "sha256.h"
#include "extmath.h"
#include "extlib.h"

/**
 * @private Transaction Position structure.
 * Contains an ID and file position type pair.
*/
typedef struct {
   word8 id[HASHLEN];
   fpos_t pos;
} TXPOS;

/**
 * @private
 * Comparison function to sort TXPOS objects by id.
*/
static int txpos_compare(const void *va, const void *vb)
{
   TXPOS *a = (TXPOS *) va;
   TXPOS *b = (TXPOS *) vb;

   return memcmp(a->id, b->id, sizeof(a->id));
}

/**
 * Generate a pseudo-block with bnum = Cblocknum + 1. Uses node state
 * (Cblockhash, Cblocknum, Time0, and Difficulty) to generate block data.
 * @param output Filename of output block (typically "pblock.dat")
 * @returns VEOK on success, else error code
*/
int pseudo(const char *output)
{
   const word32 hdrlen = 4;

   SHA256_CTX ctx;
   BTRAILER bt;
   FILE *fp;

   /* open pseudo-block file and write hdrlen */
   fp = fopen(output, "wb");
   if (fp == NULL) return VERROR;
   if (fwrite(&hdrlen, 4, 1, fp) != 1) goto CLEANUP;

   /* fill block trailer with appropriate pseudo-data */
   memset(&bt, 0, sizeof(bt));
   memcpy(bt.phash, Cblockhash, HASHLEN);
   add64(Cblocknum, One, bt.bnum);
   put32(bt.time0, Time0);
   put32(bt.difficulty, Difficulty);
   put32(bt.stime, Time0 + BRIDGE);

   /* compute pseudo-block hash directly into block trailer */
   sha256_init(&ctx);
   sha256_update(&ctx, &hdrlen, 4);
   sha256_update(&ctx, &bt, sizeof(bt) - HASHLEN);
   sha256_final(&ctx, bt.bhash);

   /* write block trailer to pseudo-block file */
   if (fwrite(&bt, sizeof(bt), 1, fp) != 1) goto CLEANUP;

   /* cleanup */
   fclose(fp);

   /* success */
   return VEOK;

   /* cleanup / error handling */
CLEANUP:
   fclose(fp);
   remove(output);
   return VERROR;
}  /* end pseudo() */

/**
 * Generate a neogenesis block. Uses a block trailer (MUST BE 0x..ff) and
 * a ledger file as input to create a output neogenesis file (0x..00).
 * @param bt Pointer to block trailer data
 * @param lefile Filename of ledger to convert to neogenesis block
 * @param output Filename of output block (typically "ngblock.dat")
 * @returns VEOK on success, else VERROR
*/
int neogen(const BTRAILER *bt, const char *lefile, const char *output)
{
   SHA256_CTX bctx;     /* (entire) block hash */
   BTRAILER bt_out;     /* output block trailers */
   NGHEADER ngh;        /* neogenesis header data */
   FILE *nfp, *lfp;
   size_t count;        /* size counters */
   long long llen;      /* ledger length */
   word8 buff[BUFSIZ];

   /* calculate neogensis block number */
   if (add64(bt->bnum, ONE64, bt_out.bnum)) {
      set_errno(EMCM_MATH64_OVERFLOW);
      return VERROR;
   }
   /* block number MUST be 0x...00 */
   if (bt_out.bnum[0] != 0x00) {
      set_errno(EMCM_BNUM);
      return VERROR;
   }

   /* open ledger read-only */
   lfp = fopen(lefile, "rb");
   if (lfp == NULL) return VERROR;
   /* fseek() to compute ledger length and check */
   if (fseek64(lfp, 0LL, SEEK_END) != 0) goto FAIL_LE;
   llen = ftell64(lfp);
   if (llen == (-1)) goto FAIL_LE;
   if (llen == 0 || (llen % sizeof(LENTRY)) != 0) {
      set_errno(EMCM_FILELEN);
      goto FAIL_LE;
   }

   /* build neogensis header data */
   put32(ngh.hdrlen, sizeof(NGHEADER));
   put64(ngh.lbytes, &llen);

   /* open neogenesis output file for writing */
   nfp = fopen(output, "wb");
   if (nfp == NULL) goto FAIL_LE;
   /* Begin the Neo-Genesis block by writing the header */
   if (fwrite(&ngh, sizeof(NGHEADER), 1, nfp) != 1) goto FAIL_ALL;

   /* initialize and begin block hash */
   sha256_init(&bctx);
   sha256_update(&bctx, &ngh, sizeof(NGHEADER));

   /* Cue ledger.dat to beginning and copy it to neo-gen block
    * header whilst hashing it into bctx.
    */
   for (rewind(lfp); llen > 0; llen -= (long long) count) {
      /* read ledger data from ledger file */
      count = fread(buff, 1, BUFSIZ, lfp);
      if (count < BUFSIZ) {
         if (ferror(lfp)) goto FAIL_ALL;
         /* for() SHOULD break at EOF... */
         if (count <= 0) {
            /* ... else, UNEXPECTED EOF */
            set_errno(EMCM_EOF);
            goto FAIL_ALL;
         }
      }
      /* write to neogenesis file and update hash */
      if (fwrite(buff, count, 1, nfp) != 1) goto FAIL_ALL;
      sha256_update(&bctx, buff, count);
   }

   /* Fix-up block trailer and write to neogenesis-block */
   memcpy(bt_out.phash, bt->bhash, HASHLEN);
   put32(bt_out.stime, get32(bt->stime));
   put32(bt_out.time0, get32(bt->time0));
   put32(bt_out.difficulty, get32(bt->difficulty));
   sha256_update(&bctx, &bt_out, sizeof(BTRAILER) - HASHLEN);
   sha256_final(&bctx, bt_out.bhash);
   if (fwrite(&bt_out, sizeof(BTRAILER), 1, nfp) != 1) {
      goto FAIL_ALL;
   }

   /* cleanup */
   fclose(nfp);
   fclose(lfp);

   /* success */
   return VEOK;

   /* cleanup / error handling */
FAIL_ALL:
   fclose(nfp);
   remove(output);
FAIL_LE:
   fclose(lfp);

   return VERROR;
}  /* end neogen() */

/**
 * Construct a candidate block from "txclean.dat". Uses node state
 * (Cblocknum, Cblockhash, Mfee, Difficulty, Time0) for block data.
 * @param output Filename of output block (typically "cblock.dat")
 * 
 * @returns VEOK on success, or VERROR on error; check errno for details
 * 
 * @return (int) Mochimo return code
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int b_con(const char *output)
{
   SHA256_CTX mctx;        /* merkel array hash */
   TXQENTRY txc;           /* for holding transaction data */
   XDATA xdata;            /* for holding eXtended transaction data */
   BTRAILER bt;            /* block trailers are fixed length */
   BHEADER bh;             /* the minimal length block header */
   TXPOS *tx;              /* malloc'd transaction positions */
   FILE *fp, *fpout;       /* to read txclean file and write cblock */
   fpos_t pos;             /* file position offset indicator */
   long long offset;       /* file position offset value (ftell) */
   size_t len, tcount;     /* malloc'd space and transaction count */
   size_t j;               /* loop counter */
   int count;              /* read_data() return value */

   /* init pointers */
   fpout = fp = NULL;
   tx = NULL;

   /* get mining address */
   count = read_data(bh.maddr, TXADDRLEN, "maddr.dat");
   if (count != TXADDRLEN || ADDR_HAS_TAG(bh.maddr)) {
      /* tagged addresses are NOT mining addresses -- for now... */
      set_errno(EMCM_MADDR);
      return VERROR;
   }

   /* open the clean TX queue (txclean.dat) and create a sorted index */
   fp = fopen("txclean.dat", "rb");
   if (fp == NULL) return VERROR;

   /* obtain EOF offset */
   if (fseek64(fp, 0LL, SEEK_END) != 0) goto ERROR_CLEANUP;
   offset = ftell64(fp);
   if (offset == (-1)) goto ERROR_CLEANUP;
   if (offset == 0LL) {
      /* no transactions? */
      set_errno(EMCM_TX0);
      goto ERROR_CLEANUP;
   }
   if ((size_t) offset < sizeof(TXQENTRY)) {
      /* file contains unknown data */
      set_errno(EMCM_FILEDATA);
      goto ERROR_CLEANUP;
   }

   /* malloc required space (approximate) */
   len = (size_t) offset / sizeof(TXQENTRY);
   tx = malloc(len * sizeof(TXPOS));
   if (tx == NULL) goto ERROR_CLEANUP;

   /* store txid and associated fpos_t value in arrays */
   for (rewind(fp), tcount = 0; tcount < len; tcount++) {
      /* store position for later use (if tx is read) */
      if (fgetpos(fp, &pos) != 0) goto ERROR_CLEANUP;
      if (tx_fread(&txc, NULL, fp) != VEOK) {
         if (ferror(fp)) goto ERROR_CLEANUP;
         /* EOF -- check fridge for leftovers */
         if (ftell64(fp) < offset) {
            set_errno(EMCM_FILEDATA);
            goto ERROR_CLEANUP;
         }
         break;
      }
      /* set txid reference data */
      memcpy(&(tx[tcount].id), txc.tx_id, HASHLEN);
      tx[tcount].pos = pos;
   }
   /* sort the txid reference array */
   qsort(tx, tcount, sizeof(TXPOS), txpos_compare);

   /* open output for writing */
   fpout = fopen("cblock.tmp", "wb");
   if (fpout == NULL) goto ERROR_CLEANUP;

   /* compute new block number */
   if (add64(Cblocknum, ONE64, bt.bnum)) {
      set_errno(EMCM_MATH64_OVERFLOW);
      goto ERROR_CLEANUP;
   }

   /* finalize block header data (compute reward) */
   put32(bh.hdrlen, sizeof(BHEADER));
   /* ... bh.maddr assigned earlier */
   get_mreward(bh.mreward, bt.bnum);

   /* write header to disk */
   if (fwrite(&bh, sizeof(BHEADER), 1, fpout) != 1) {
      goto ERROR_CLEANUP;
   }

   /* begin merkel array hash with mining address */
   sha256_init(&mctx);
   sha256_update(&mctx, bh.maddr, TXADDRLEN);

   /* read transactions from txclean.dat using sorted TXPOS array */
   for (j = 0; j < tcount; j++) {
      /** @todo: duplicate transaction checks should really be done before
       * we get here (i.e. when we are cleaning the queue, or validating)
      */
      if (j > 0) {
         /* check previous transaction id for duplicates */
         if (memcmp(tx[j].id, tx[j-1].id, HASHLEN) == 0) {
         /* pwarn("duplicate transaction in clean queue"); */
            /* ignore duplicate transaction */
            continue;
         }
      }
      /* seek to transaction position */
      if (fsetpos(fp, &tx[j].pos) != 0) {
         goto ERROR_CLEANUP;
      }
      /* read transaction */
      if (tx_fread(&txc, &xdata, fp) != VEOK) {
         if (!ferror(fp)) set_errno(EMCM_EOF);
         goto ERROR_CLEANUP;
      }
      /* add transaction id to merkel array */
      sha256_update(&mctx, txc.tx_id, HASHLEN);
      /* write transaction to block */
      if (tx_fwrite(&txc, &xdata, fpout) != 1) {
         goto ERROR_CLEANUP;
      }
   }  /* end for() */

   /* finalize block trailer data */
   memcpy(bt.phash, Cblockhash, HASHLEN);
   /* ... bt.bnum assigned earlier */
   put64(bt.mfee, Mfee);
   put32(bt.tcount, tcount);
   put32(bt.time0, Time0);
   put32(bt.difficulty, Difficulty);
   /* finalize merkel array hash straight into the trailer */
   sha256_final(&mctx, bt.mroot);
   /* remaining data is set zero (not known yet) */
   memset(bt.nonce, 0, HASHLEN);
   memset(bt.stime, 0, 4);
   memset(bt.bhash, 0, HASHLEN);

   /* write trailer to file */
   if (fwrite(&bt, sizeof(BTRAILER), 1, fpout) != 1) {
      goto ERROR_CLEANUP;
   }

   /* cleanup */
   fclose(fpout);
   fclose(fp);
   free(tx);

   /* move temporary output (*.tmp) to working output (*.dat) */
   remove(output);
   if (rename("cblock.tmp", output) != 0) {
      return VERROR;
   }

   /* success */
   return VEOK;

   /* cleanup / error handling */
ERROR_CLEANUP:
   if (fpout != NULL) {
      fclose(fpout);
      remove("cblock.tmp");
   }
   if (fp != NULL) fclose(fp);
   if (tx != NULL) free(tx);

   return VERROR;
}  /* end b_con() */

/* end include guard */
#endif
