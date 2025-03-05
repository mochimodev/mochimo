/**
 * @private
 * @headerfile bcon.h <bcon.h>
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
 * @todo This unit contains duplicated code from src/tx.c involving
 * TXPOS and txpos_compare(). This code has not been refactored into
 * a common header file or source file due to future planned deprecation.
*/

/* include guard */
#ifndef MOCHIMO_BCON_C
#define MOCHIMO_BCON_C


#include "bcon.h"

/* internal support */
#include "tx.h"
#include "tfile.h"
#include "ledger.h"
#include "global.h"
#include "error.h"

/* external support */
#include <string.h>
#include "sha256.h"
#include "extmath.h"
#include "extlib.h"

/* Pseudoblock mining address tag (hexadecimal encoding) */
static const word8 Maddr_pseudo[ADDR_TAG_LEN] = {
   0x12, 0x60, 0x88, 0x7b, 0x0d, 0x36, 0x53, 0xf8, 0xac, 0x25,
   0x59, 0x69, 0xcb, 0x4a, 0x60, 0x7f, 0x00, 0x6c, 0xf6, 0x73
};

/* Standard block mining address tag (hexadecimal encoding) */
static word8 Maddr[ADDR_TAG_LEN] = {
   0x3f, 0x1f, 0xba, 0x70, 0x25, 0xc7, 0xd3, 0x74, 0x70, 0xe7,
   0x26, 0x01, 0x17, 0xa7, 0x2b, 0x7d, 0xe9, 0xf5, 0xca, 0x59
};

/**
 * @private Transaction Position structure.
 * Contains a source and file position type pair.
*/
typedef struct {
   word8 src[ADDR_LEN];
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

   return memcmp(a->src, b->src, sizeof(a->src));
}

/**
 * Get the mining address for pseudo-blocks.
 * @param maddr Pointer to place pseudo-block mining address
 */
void get_pseudo_maddr(void *maddr)
{
   memcpy(maddr, Maddr_pseudo, ADDR_TAG_LEN);
}

/**
 * Set the mining address for all standard blocks.
 * @param maddr Pointer to mining address to set
 */
void set_maddr(const void *maddr)
{
   memcpy(Maddr, maddr, ADDR_TAG_LEN);
}

/**
 * Generate a pseudo-block with bnum = Cblocknum + 1. Uses node state
 * (Cblockhash, Cblocknum, Time0, and Difficulty) to generate block data.
 * @param output Filename of output block (typically "pblock.dat")
 * @return (int) value representing operation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int pseudo(const char *output)
{
   BTRAILER bt;
   BHEADER bh;
   FILE *fp;

   /* open pseudo-block file */
   fp = fopen(output, "wb");
   if (fp == NULL) return VERROR;

   /* init block header with pseudo_maddr */
   put32(bh.hdrlen, sizeof(BHEADER));
   memcpy(bh.maddr, Maddr_pseudo, sizeof(bh.maddr));

   /* init block trailer (zero) and compute bnum */
   memset(&bt, 0, sizeof(BTRAILER));
   if (add64(Cblocknum, ONE64, bt.bnum)) {
      set_errno(EMCM_MATH64_OVERFLOW);
      goto ERROR_CLEANUP;
   }

   /* place mreward in block header */
   get_mreward(bh.mreward, bt.bnum);

   /* fill block trailer with remaining data */
   memcpy(bt.phash, Cblockhash, HASHLEN);
   /* ... bt.bnum set earlier via add64() */
   /* ... bt.mfee left zero'd (no transactions) */
   /* ... bt.tcount left zero'd (no transactions) */
   put32(bt.time0, Time0);
   put32(bt.difficulty, Difficulty);
   sha256(bh.maddr, sizeof(bh.maddr) + sizeof(bh.mreward), bt.mroot);
   /* ... bt.nonce left zero'd (solve abandoned) */
   put32(bt.stime, Time0 + BRIDGEv3);
   /* compute pseudo-block hash directly into block trailer */
   sha256(&bt, sizeof(BTRAILER) - HASHLEN, bt.bhash);

   /* write pseudo-block file and close */
   if (fwrite(&bh, sizeof(BHEADER), 1, fp) != 1) goto ERROR_CLEANUP;
   if (fwrite(&bt, sizeof(BTRAILER), 1, fp) != 1) goto ERROR_CLEANUP;
   fclose(fp);

   return VEOK;

   /* cleanup / error handling */
ERROR_CLEANUP:
   fclose(fp);
   remove(output);

   return VERROR;
}  /* end pseudo() */

/**
 * Generate a neogenesis block. Uses a block trailer (MUST BE 0x..ff) and
 * a ledger file as input to create a output neogenesis file (0x..00).
 * @param prev_bt Pointer to previous block trailer data
 * @param lefile Filename of ledger to convert to neogenesis block
 * @param output Filename of output block (typically "ngblock.dat")
 * @return (int) value representing operation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int neogen(const BTRAILER *prev_bt, const char *lefile, const char *output)
{
   LENTRY le;           /* ledger entry */
   BTRAILER bt;         /* block trailer */
   NGHEADER ngh;        /* neogenesis header data */
   FILE *fp, *lfp;
   word8 *mtree;        /* malloc'd merkle tree */
   size_t mcount;       /* merkle tree count */
   size_t j;            /* loop counter */
   long long llen;      /* ledger length */

   /* init */
   fp = lfp = NULL;
   mtree = NULL;
   mcount = 0;

   /* init block trailer (zero) and compute bnum */
   memset(&bt, 0, sizeof(BTRAILER));
   if (add64(prev_bt->bnum, ONE64, bt.bnum)) {
      set_errno(EMCM_MATH64_OVERFLOW);
      return VERROR;
   }
   /* block number MUST be 0x...00 */
   if (bt.bnum[0] != 0x00) {
      set_errno(EMCM_BNUM);
      return VERROR;
   }

   /* open ledger read-only */
   lfp = fopen(lefile, "rb");
   if (lfp == NULL) return VERROR;
   /* fseek() to compute ledger length and check */
   if (fseek64(lfp, 0LL, SEEK_END) != 0) goto ERROR_CLEANUP;
   llen = ftell64(lfp);
   if (llen == (-1)) goto ERROR_CLEANUP;
   if (llen < (long long) sizeof(LENTRY)) {
      /* unexpected ledger data */
      set_errno(EMCM_FILEDATA);
      goto ERROR_CLEANUP;
   } else if (llen % sizeof(LENTRY) != 0) {
      /* unexpected ledger len */
      set_errno(EMCM_FILELEN);
      goto ERROR_CLEANUP;
   }

   /* build neogensis header data */
   put32(ngh.hdrlen, sizeof(NGHEADER));
   put64(ngh.lbytes, &llen);

   /* open neogenesis output file for writing */
   fp = fopen(output, "wb");
   if (fp == NULL) goto ERROR_CLEANUP;
   /* Begin the Neo-Genesis block by writing the header */
   if (fwrite(&ngh, sizeof(NGHEADER), 1, fp) != 1) goto ERROR_CLEANUP;

   /* get merkle tree count and malloc */
   mcount = (size_t) llen / sizeof(LENTRY);
   mtree = malloc(mcount * HASHLEN);
   if (mtree == NULL) goto ERROR_CLEANUP;

   /** @todo switch to a progressive merkle hash function that
    * doesn't require the entire list to be in memory at once.
    */

   /* Cue ledger.dat to beginning and copy it to neo-gen block
    * header whilst collecting merkle tree nodes.
    */
   for (rewind(lfp), j = 0; j < mcount; j++) {
      /* read individual ledger entries for processing */
      if (fread(&le, sizeof(LENTRY), 1, lfp) != 1) {
         /* check file error, else unexpected EOF */
         if (ferror(lfp)) goto ERROR_CLEANUP;
         set_errno(EMCM_EOF);
         goto ERROR_CLEANUP;
      }
      /* write to neogenesis file and update merkle list */
      if (fwrite(&le, sizeof(LENTRY), 1, fp) != 1) goto ERROR_CLEANUP;
      sha256(&le, sizeof(LENTRY), mtree + (j * HASHLEN));
   }

   /* fill block trailer with remaining data */
   memcpy(bt.phash, prev_bt->bhash, HASHLEN);
   /* ... bt.bnum set earlier via add64() */
   /* ... bt.mfee left zero'd (no transactions) */
   /* ... bt.tcount left zero'd (no transactions) */
   put32(bt.time0, get32(prev_bt->time0));
   put32(bt.difficulty, get32(prev_bt->difficulty));
   merkle_root(mtree, mcount, bt.mroot);
   /* ... bt.nonce left zero'd (not required) */
   put32(bt.stime, get32(prev_bt->stime));
   /* compute neogenesis block hash directly into block trailer */
   sha256(&bt, sizeof(BTRAILER) - HASHLEN, bt.bhash);

   /* write block trailer to neogenesis block */
   if (fwrite(&bt, sizeof(BTRAILER), 1, fp) != 1) {
      goto ERROR_CLEANUP;
   }

   /* cleanup */
   fclose(fp);
   fclose(lfp);
   free(mtree);

   return VEOK;

   /* cleanup / error handling */
ERROR_CLEANUP:
   if (mtree) free(mtree);
   if (lfp) fclose(lfp);
   if (fp) {
      fclose(fp);
      remove(output);
   }

   return VERROR;
}  /* end neogen() */

/**
 * Adjust a block to accommodate a new mining address. Uses existing state
 * and updates the necessary components to reflect the new mining address.
 * @param fp File pointer to candidate block file
 * @param maddr New mining address to update block with
 * @return (int) value representing operation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int b_adjust_maddr_fp(FILE *fp)
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
   if (fseek(fp, -((long) sizeof(BTRAILER)), SEEK_END) != 0) goto ERROR_CLEANUP;
   if (fread(&bt, sizeof(BTRAILER), 1, fp) != 1) goto RDERR_CLEANUP;
   tcount = get32(bt.tcount);

   /* read/update block header */
   if (fseek(fp, 0L, SEEK_SET) != 0) goto ERROR_CLEANUP;
   if (fread(&bh, sizeof(BHEADER), 1, fp) != 1) goto RDERR_CLEANUP;
   memcpy(bh.maddr, Maddr, sizeof(bh.maddr));
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

   /* explicit seek is required for switching between input and output */
   if (fseek(fp, -((long) sizeof(BTRAILER)), SEEK_END) != 0) goto ERROR_CLEANUP;
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
 * Construct a candidate block from "txclean.dat". Uses node state
 * (Cblocknum, Cblockhash, Mfee, Difficulty, Time0) for block data.
 * @param output Filename of output block (typically "cblock.dat")
 * @return (int) value representing operation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int b_con(const char *output)
{
   TXENTRY txc;            /* for holding transaction data */
   BTRAILER bt;            /* block trailers are fixed length */
   BHEADER bh;             /* the minimal length block header */
   TXPOS *tx;              /* malloc'd transaction positions */
   FILE *fp, *fpout;       /* to read txclean file and write cblock */
   void *ptr;              /* realloc pointer */
   word8 *mtree;           /* malloc'd merkle tree list */
   fpos_t pos;             /* file position offset indicator */
   long long offset;       /* file position offset value (ftell) */
   word32 count, tcount;   /* malloc'd space and transaction count */
   word32 j, actual;       /* loop counter and txclean count */
   int cond;               /* loop condition */

   /* init pointers */
   fpout = fp = NULL;
   mtree = NULL;
   tx = NULL;

   /* get mining address tag */
   memcpy(bh.maddr, Maddr, sizeof(bh.maddr));

   /* BEGIN TRANSACTION SORT */

   /* open the clean TX queue (txclean.dat) and create a sorted index */
   fp = fopen("txclean.dat", "rb");
   if (fp == NULL) return VERROR;

   /* obtain EOF offset and check */
   if (fseek64(fp, 0LL, SEEK_END) != 0) goto ERROR_CLEANUP;
   offset = ftell64(fp);
   if (offset == (-1)) goto ERROR_CLEANUP;
   /* NOTE: transaction sizes can differ */
   if (offset == 0LL) {
      /* no transactions? */
      set_errno(EMCM_TX0);
      goto ERROR_CLEANUP;
   } else if ((size_t) offset < TXLEN_MIN) {
      /* file contains unknown data */
      set_errno(EMCM_FILEDATA);
      goto ERROR_CLEANUP;
   }

   /* reset file position indicator */
   if (fseek64(fp, 0LL, SEEK_SET) != 0) goto ERROR_CLEANUP;
   /* loop to check allocated space is sufficient (+32 TXs/loop) */
   for (actual = 0, cond = 1, count = 32; cond; count += 32) {
      /* (re)allocate memory space for 32 TXs at a time */
      ptr = realloc(tx, count * sizeof(TXPOS));
      if (ptr == NULL) goto ERROR_CLEANUP;
      tx = ptr;
      /* loop to store source and associated fpos_t value in array */
      while (actual < count) {
         /* store position for later use (if tx is read) */
         if (fgetpos(fp, &pos) != 0) goto ERROR_CLEANUP;
         if (tx_fread(&txc, fp) != VEOK) {
            if (ferror(fp)) goto ERROR_CLEANUP;
            /* set EOF condition */
            cond = 0;
            break;
         }
         /* set source reference data */
         memcpy(&(tx[actual].src), txc.src_addr, ADDR_LEN);
         tx[actual++].pos = pos;
      }  /* end while() */
   }  /* end for() */
   /* check for leftover data */
   if (ftell64(fp) < offset) {
      set_errno(EMCM_FILEDATA);
      goto ERROR_CLEANUP;
   }
   /* sort the txid reference array */
   qsort(tx, actual, sizeof(TXPOS), txpos_compare);

   /* BEGIN BLOCK CONSTRUCTION */

   /* open output for writing */
   fpout = fopen("cblock.tmp", "wb");
   if (fpout == NULL) goto ERROR_CLEANUP;

   /* init block trailer (zero) and compute bnum */
   memset(&bt, 0, sizeof(BTRAILER));
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

   /* malloc merkle tree (+1 for header data) */
   mtree = malloc((actual + 1) * HASHLEN);
   if (mtree == NULL) goto ERROR_CLEANUP;

   /* begin merkel hash with mining address + reward */
   sha256(bh.maddr /* + bh.mreward */, sizeof(bh.maddr) + 8, mtree);

   /** @todo switch to a progressive merkle hash function that
    * doesn't require the entire list to be in memory at once.
    */

   /* read transactions from txclean.dat using sorted TXPOS array */
   for (j = tcount = 0; j < actual; j++) {
      /* seek to transaction position */
      if (fsetpos(fp, &tx[j].pos) != 0) {
         goto ERROR_CLEANUP;
      }
      /* read transaction */
      if (tx_fread(&txc, fp) != VEOK) {
         if (!ferror(fp)) set_errno(EMCM_EOF);
         goto ERROR_CLEANUP;
      }
      /* skip duplicate source address */
      if (j > 0) {
         if (memcmp(txc.src_addr, tx[j - 1].src, HASHLEN) == 0) {
            continue;
         }
      }
      /* set appropriate nonce and hash */
      memset(txc.tlr->nonce, 0, sizeof(txc.tlr->nonce));
      tx_hash(&txc, TX_HASH_ID, txc.tx_id);
      /* add transaction id to merkel tree (++ prefix for miner) */
      memcpy(&mtree[(++tcount) * HASHLEN], txc.tx_id, HASHLEN);
      /* write transaction to block */
      if (tx_fwrite(&txc, fpout) != VEOK) {
         goto ERROR_CLEANUP;
      }
   }  /* end for() */

   /* finalize block trailer data */
   memcpy(bt.phash, Cblockhash, HASHLEN);
   /* ... bt.bnum set earlier via add64() */
   put64(bt.mfee, Mfee);
   put32(bt.tcount, tcount);
   put32(bt.time0, Time0);
   put32(bt.difficulty, Difficulty);
   /* compute merkel root hash straight into the trailer */
   merkle_root(mtree, tcount + 1, bt.mroot);
   /* ... bt.nonce left zero'd (not known) */
   /* ... bt.stime left zero'd (not known) */
   /* ... bt.bhash left zero'd (not known) */

   /* write trailer to file */
   if (fwrite(&bt, sizeof(BTRAILER), 1, fpout) != 1) {
      goto ERROR_CLEANUP;
   }

   /* cleanup */
   fclose(fpout);
   fclose(fp);
   free(mtree);
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
   if (fpout) {
      fclose(fpout);
      remove("cblock.tmp");
   }
   if (fp) fclose(fp);
   if (mtree) free(mtree);
   if (tx) free(tx);

   return VERROR;
}  /* end b_con() */

/* end include guard */
#endif
