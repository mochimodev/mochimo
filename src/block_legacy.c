/**
 * @private
 * @headerfile block.h <block.h>
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_BLOCK_LEGACY_C
#define MOCHIMO_BLOCK_LEGACY_C


#include "block.h"

/* internal support */
#include "transaction.h"
#include "ledger.h"
#include "error.h"
#include "chain.h"

/* external support */
#include <string.h>
#include "sha256.h"
#include "extmath.h"
#include "extlib.h"

/* block number feature test MACRO */
#define NEWYEAR(bnum)   ( get32(bnum) >= V23TRIGGER || get32(bnum + 4) )

/**
 * Validate an open WOTS+ blockchain file.
 * @param fp Opened blockchain FILE pointer to validate
 * @param btp Pointer to previous block trailer data
 * @return (int) value representing operation result
 * @retval VEOK on success
 * @retval VERROR on internal error; check errno for details
 * @retval VEBAD on invalid block; check errno for details
*/
int blockw_val_fp(FILE *fp, BTRAILER *btp)
{
   TXW txw;       /* buffer to read-in WOTS+ transactions */
   BHEADER_W bh;  /* buffer for block header */
   BTRAILER bt;   /* buffer for block trailer */
   SHA256_CTX bctx, mctx;
   long long blocklen, chklen;
   word32 mreward[2];
   word32 hdrlen, tcount, j;
   word8 tx_id[HASHLEN], prev_tx_id[HASHLEN];
   word8 mroot[HASHLEN], bhash[HASHLEN];
   int res;

   /* sanity check */
   if (fp == NULL || btp == NULL) goto FAIL_INVAL;

   /* ensure blockchain file is at start */
   rewind(fp);

   /* read and check regular fixed size block header */
   if (fread(&hdrlen, sizeof(hdrlen), 1, fp) != 1) return VERROR;
   if (hdrlen != sizeof(bh)) goto BAD_HDRLEN;

   /* read block trailer */
   if (fseek64(fp, -(sizeof(bt)), SEEK_END) != 0) return VERROR;
   if (fread(&bt, sizeof(bt), 1, fp) != 1) return VERROR;

   /* validate block trailer sequence and Proof of Work */
   res = validate_trailer(&bt, btp);
   if (res == VEOK) res = validate_pow(&bt);
   if (res == VERROR) goto FAIL;
   if (res >= VEBAD) goto BAD;

   /* check transaction count */
   tcount = get32(bt.tcount);
   if (tcount == 0 || tcount > MAXBLTX) goto BAD_TCOUNT;

   /* read and check total block file length */
   chklen = (long long) hdrlen + (tcount * sizeof(txw)) + sizeof(bt);
   blocklen = ftell64(fp);
   if (blocklen == EOF) return VERROR;
   if (blocklen != chklen) goto BAD_FILELEN;

   /* read entire block header */
   if (fseek64(fp, 0LL, SEEK_SET) != 0) return VERROR;
   if (fread(&bh, sizeof(bh), 1, fp) != 1) goto FAIL;

   /* check mining reward/address/no-tag */
   get_mreward(mreward, bt.bnum);
   if (memcmp(bh.mreward, mreward, 8) != 0) goto BAD_MREWARD;
   if (WOTS_HAS_TAG(bh.maddr)) goto BAD_MADDR;

   /* fp left at offset of Merkel Block Array -- ready to fread() */

   /* begin hashing contexts */
   sha256_init(&bctx);
   sha256_update(&bctx, &bh, hdrlen);

   /* read/validate/store each transaction */
   for (j = 0; j < tcount; j++) {
      /* read next transaction into memory */
      if (fread(&txw, sizeof(txw), 1, fp) != 1) goto FAIL;

      /* running block/merkel hash */
      sha256_update(&bctx, txw.src_addr, TXWOTSLEN);
      sha256_update(&bctx, txw.dst_addr, TXWOTSLEN);
      sha256_update(&bctx, txw.chg_addr, TXWOTSLEN);
      sha256_update(&bctx, txw.send_total, TXAMOUNTLEN);
      sha256_update(&bctx, txw.change_total, TXAMOUNTLEN);
      sha256_update(&bctx, txw.tx_fee, TXAMOUNTLEN);
      sha256_update(&bctx, txw.tx_sig, TXSIGLEN);
      sha256_update(&bctx, txw.tx_id, HASHLEN);
      /* check tx_id matches computed hash */
      sha256(txw.src_addr, TXWOTSLEN, tx_id);
      if (memcmp(tx_id, txw.tx_id, HASHLEN) != 0) goto BAD_TX_ID;
      /* check that tx_id is sorted (skip first) */
      if (j != 0) {
         res = memcmp(tx_id, prev_tx_id, HASHLEN);
         if (res <= 0) {
            if (res < 0) goto BAD_TX_SORT;
            if (res == 0) goto BAD_TX_DUP;
         }
      }
      /* remember this tx_id for next time */
      memcpy(prev_tx_id, tx_id, HASHLEN);

      /* validate transaction data */
      /* NOTE: transaction may be modified by TXW_MDST validation */
      res = txw_val((TXW *) &txw, bt.mfee);
      if (res == VERROR) goto FAIL;
      if (res >= VEBAD) goto BAD;
   }  /* end for j */

   /* finalize Merkel Root - phash, bnum, mfee, tcount, time0, difficulty */
   sha256_update(&bctx, &bt, (HASHLEN + 8 + 8 + 4 + 4 + 4));
   memcpy(&mctx, &bctx, sizeof(mctx));
   sha256_final(&mctx, mroot);
   if (memcmp(bt.mroot, mroot, HASHLEN) != 0) goto BAD_MROOT;

   /* finalize block hash - Block trailer (- block hash) */
   sha256_update(&bctx, &bt, sizeof(BTRAILER) - HASHLEN);
   sha256_final(&bctx, bhash);
   if (memcmp(bt.bhash, bhash, HASHLEN) != 0) goto BAD_BHASH;

   /* block is valid */
   return VEOK;

FAIL_INVAL: set_errno(EINVAL);
FAIL:
   if (feof(fp)) set_errno(EMCM_EOF);
   return VERROR;

BAD_HDRLEN: set_errno(EMCM_HDRLEN); return VEBAD;
BAD_TCOUNT: set_errno(EMCM_TCOUNT); return VEBAD;
BAD_FILELEN: set_errno(EMCM_FILELEN); return VEBAD;
BAD_MREWARD: set_errno(EMCM_MREWARD); return VEBAD;
BAD_MADDR: set_errno(EMCM_MADDR); return VEBAD;
BAD_TX_ID: set_errno(EMCM_TX_ID); return VEBAD;
BAD_TX_SORT: set_errno(EMCM_TX_SORT); return VEBAD;
BAD_TX_DUP: set_errno(EMCM_TX_DUP); return VEBAD;
BAD_MROOT: set_errno(EMCM_MROOT); return VEBAD;
BAD_BHASH: set_errno(EMCM_BHASH);
BAD:
   return VEBAD;
}  /* end blockw_val_fp() */

/**
 * Validate a WOTS+ blockchain file.
 * @param fname Filename of blockchain file to validate
 * @param tfname Filename of Tfile to validate against
 * @return (int) value representing operation result
 * @retval VEBAD on block format violation; check errno for details
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int blockw_val(char *fname, char *tfname)
{
   BTRAILER prev_bt;
   FILE *fp;
   int ecode;

   /* read last Tfile trailer for validation against */
   if (read_trailer(&prev_bt, tfname) != VEOK) return VERROR;

   /* open pseudo-block file for validation */
   fp = fopen(fname, "rb");
   if (fp == NULL) return VERROR;
   ecode = blockw_val_fp(fp, &prev_bt);
   fclose(fp);

   return ecode;
}  /* end blockw_val() */

/**
 * Generate a WOTS+ blockchain file with transactions.
 * Transaction list must be sorted in ascending order.
 * Uses the last trailer in the Tfile as state.
 * @param fname Filename of output block (typically "cblock.dat")
 * @param txw_clean Pointer to a list of WOTS+ transactions
 * @param count Number of transaction in @a txw_clean
 * @param tfname Filename of Tfile to use as state
 * @return (int) value representing operation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int blockw(char *fname, TXW *txw_clean, size_t count, char *tfname)
{
   static word8 one[8] = { 1, 0 };

   SHA256_CTX mctx;        /* buffer for merkel array hash */
   BTRAILER bt, prev_bt;   /* buffers for block trailers */
   BHEADER_W bhw;          /* buffer for WOTS+ block header */
   TXW *txp;               /* pointer to next transaction */
   FILE *fp;               /* file read/write handling */
   size_t ntx;
   word32 mfee[2];         /* mining fee tracking */
   word8 maddr[TXWOTSLEN]; /* to read mining address for block */
   word8 prev_tx_id[HASHLEN];  /* to check for duplicate transactions */
   int cond;

   /* get previous trailer */
   if (read_trailer(&prev_bt, tfname) != VEOK) return VERROR;

   /* get mining address */
   fp = fopen(Maddr_opt, "rb");
   if (fp == NULL) return VERROR;
   if (fread(maddr, sizeof(maddr), 1, fp) != 1) goto FAIL_IO;
   fclose(fp);

   /* compute new block number */
   memset(&bt, 0, sizeof(bt));
   add64(prev_bt.bnum, one, bt.bnum);

   /* prepare block header */
   put32(bhw.hdrlen, sizeof(bhw));
   memcpy(bhw.maddr, maddr, sizeof(maddr));
   get_mreward(bhw.mreward, bt.bnum);

   /* create cblock.tmp */
   fp = fopen(fname, "wb");
   if (fp == NULL) return VERROR;

   /* prepare hashing states (with block header after NEWYEAR) */
   sha256_init(&mctx);
   if (NEWYEAR(bt.bnum)) sha256_update(&mctx, &bhw, sizeof(bhw));
   /* write block header to disk */
   if (fwrite(&bhw, sizeof(bhw), 1, fp) != 1) goto FAIL_IO;

   /* hash and write transactions into block */
   for (ntx = 0; ntx < count && ntx < MAXBLTX; ntx++) {
      txp = &txw_clean[ntx];
      /* update mfee on first transaction... */
      if (ntx == 0) memcpy(mfee, txp->tx_fee, sizeof(mfee));
      if (ntx != 0) {
         /* compare/update mfee */
         if (cmp64(txp->tx_fee, mfee) < 0) {
            memcpy(mfee, txp->tx_fee, sizeof(mfee));
         }
         /* compare transaction ID sort */
         cond = memcmp(txp->tx_id, prev_tx_id, HASHLEN);
         if (cond < 0) goto FAIL_IO_SORT;
         if (cond == 0) goto FAIL_IO_DUP;
      }
      /* remember tx_id for next iteration */
      memcpy(prev_tx_id, txp->tx_fee, HASHLEN);
      /* add transaction to block hash (and merkel array) */
      sha256_update(&mctx, txp, sizeof(TXW));
      /* write transaction to block */
      if (fwrite(txp, sizeof(TXW), 1, fp) != 1) goto FAIL_IO;
   }  /* end for ntx */
   if (ntx == 0) goto FAIL_IO_NOTX;

   /* finish preparing block trailer */
   memcpy(bt.phash, prev_bt.bhash, HASHLEN);
   put64(bt.mfee, mfee);
   put32(bt.tcount, (word32) ntx);
   put32(bt.time0, get32(prev_bt.stime));
   put32(bt.difficulty, next_difficulty(&prev_bt));
   /* finalize merkel array into block trailer */
   sha256_update(&mctx, &bt, (HASHLEN + 8 + 8 + 4 + 4 + 4));
   sha256_final(&mctx, bt.mroot);
   /* write block trailer */
   if (fwrite(&bt, sizeof(bt), 1, fp) != 1) goto FAIL_IO;
   /* finished with block file */
   fclose(fp);

   /* success */
   return VEOK;

/* error handling */
FAIL_IO_SORT: set_errno(EMCM_TX_SORT); goto FAIL_IO;
FAIL_IO_DUP: set_errno(EMCM_TX_DUP); goto FAIL_IO;
FAIL_IO_NOTX: set_errno(EMCMNOTXS);
FAIL_IO:
   if (feof(fp)) set_errno(EMCM_EOF);
   return VERROR;
}  /* end blockw() */

/**
 * Validate an open WOTS+ neo-genesis block.
 * Checks ledger entries are in ascending sort.
 * Checks block hash matches calculated hash.
 * Checks block size matches neo-genesis format.
 * Checks block trailer matches Tfile entry.
 * Checks sum of amounts do not exceed "expected" rewards.
 * NOTE: Tfile should have been verified before neo-genesis validation.
 * @param ngfp Open neo-genesis FILE pointer to validate
 * @param tfname Filename of Tfile to validate against
 * @return (int) value representing operation result
 * @retval VEBAD on block format violation; check errno for details
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int neogenw_val_fp(FILE *fp, char *tfname)
{
   LENTRY_W le, prev_le;
   SHA256_CTX cctx;
   BTRAILER bt, tft;
   size_t chklen, count;
   word32 hdrlen, first;
   word8 chash[HASHLEN];
   word8 amounts[8];
   word8 rewards[8];

   /* compute block hash, sum ledger amounts and check length */
   chklen = 0;
   sha256_init(&cctx);
   memset(amounts, 0, 8);

   /* read header length */
   if (fseek64(fp, 0LL, SEEK_SET) != 0) return VERROR;
   if (fread(&hdrlen, sizeof(hdrlen), 1, fp) != 1) return VERROR;

   /* update data from headerlen */
   chklen = sizeof(hdrlen);
   sha256_update(&cctx, &hdrlen, sizeof(hdrlen));

   /* read remaining block data */
   for (first = 1; ; first = 0, chklen += count) {
      /* perform read into ledger entry -- check read count */
      if ((count = fread(&le, sizeof(le), 1, fp)) != 1) {
         if (ferror(fp)) return VERROR;
         /* EOF -- read block trailer */
         /* NOTE: this logic relies on sizeof(BTRAILER) < sizeof(LENTRY) */
         if (fread(&bt, sizeof(bt), 1, fp) != 1) goto FAIL_IO;
         break;
      }
      /* check ledger sort -- skip on first read */
      if (!first && memcmp(le.addr, prev_le.addr, sizeof(le.addr)) <= 0) {
         goto BAD_SORT;
      }
      /* update amounts sum, ensure no overflow */
      if (add64(amounts, le.balance, amounts)) goto BAD_OVERFLOW;
      /* update data from neogenesis */
      memcpy(&prev_le, &le, sizeof(le));
      sha256_update(&cctx, &le, count);
   }  /* end for() */

   /* add trailer data -(HASHLEN) to computed hash -- finalize */
   sha256_update(&cctx, &bt, sizeof(bt) - HASHLEN);
   sha256_final(&cctx, chash);

   /* check block hash, headerlen and modulus */
   if (memcmp(chash, bt.bhash, HASHLEN) != 0) goto BAD_BHASH;
   if (chklen != hdrlen) goto BAD_HDRLEN;
   if ((chklen % sizeof(le)) != sizeof(hdrlen)) goto BAD_HDRLEN;

   /* compare Neogenesis block trailer to Tfile trailer */
   if (read_tfile(&tft, bt.bnum, 1, tfname) != 1) return VERROR;
   if (memcmp(&bt, &tft, sizeof(bt)) != 0) goto BAD_TRAILER;

   /* obtain accurate sum of rewards with Tfile -- check ledger amounts */
   /* NOTE: get_tfrewards() cannot calculate wiped account balances < MFEE,
    * so we only check the Neogenesis amounts do not exceed expected */
   if (get_tfrewards(tfname, rewards, bt.bnum) != VEOK) return VERROR;

   /* check calculated rewards against ledger amounts */
   if (cmp64(amounts, rewards) > 0) goto BAD_AMOUNTS;

   /* neogenesis is valid */
   return VEOK;

/* error handling */
FAIL_IO:
   if (feof(fp)) set_errno(EMCM_EOF);
   return VERROR;

/* block format violation handling */
BAD_AMOUNTS: set_errno(EMCM_LE_AMOUNTS_SUM); return VEBAD;
BAD_BHASH: set_errno(EMCM_BHASH); return VEBAD;
BAD_HDRLEN: set_errno(EMCM_HDRLEN); return VEBAD;
BAD_TRAILER: set_errno(EMCM_TRAILER); return VEBAD;
BAD_OVERFLOW: set_errno(EMCM_LE_AMOUNTS_OVERFLOW); return VEBAD;
BAD_SORT: set_errno(EMCM_LE_SORT); return VEBAD;
}  /* end neogenw_val_fp() */

/**
 * Validate a WOTS+ neo-genesis block.
 * Checks ledger entries are in ascending sort.
 * Checks block hash matches calculated hash.
 * Checks block size matches neo-genesis format.
 * Checks block trailer matches Tfile entry.
 * Checks sum of amounts do not exceed "expected" rewards.
 * NOTE: Tfile should have been verified before neo-genesis validation.
 * @param fname Filename of neo-genesis file to validate
 * @param tfname Filename of Tfile to validate against
 * @return (int) value representing operation result
 * @retval VEBAD on block format violation; check errno for details
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int neogenw_val(char *fname, char *tfname)
{
   FILE *fp;
   int ecode;

   /* open pseudo-block file for validation */
   fp = fopen(fname, "rb");
   if (fp == NULL) return VERROR;
   ecode = neogenw_val_fp(fp, tfname);
   fclose(fp);

   return ecode;
}  /* end neogenw_val() */

/* end include guard */
#endif
