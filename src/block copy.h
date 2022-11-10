/**
 * @private
 * @headerfile block.h <block.h>
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_BLOCK_C
#define MOCHIMO_BLOCK_C


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

int sort_pptxqe_by_src_tag(const void *a, const void *b)
{
   const void *const va = WOTS_TAGp(*((TXW **) a));
   const void *const vb = WOTS_TAGp(*((TXW **) b));

   return memcmp(va, vb, TXTAGLEN);
}

int search_tag_in_txsa_by_src_tag(const void *key, const void *item)
{
   return memcmp(key, WOTS_TAGp((*((TXW **) item))->src_addr), TXTAGLEN);
}

/**
 * Validate a WOTS+ blockchain file data.
 * NOTE: requires "opened" file pointers in "rb" mode
 * @param bcfp Opened blockchain FILE pointer to validate
 * @param btp Pointer to previous block trailer data
 * @retval VEOK on success
 * @retval VERROR on internal error; check errno for details
 * @retval VEBAD on invalid block; check errno for details
*/
int blockw_val_data(FILE *bcfp, BTRAILER *btp)
{
   TXW txw;       /* buffer to read-in WOTS+ transactions */
   TXW_MDST *mtx; /* pointer to reference multi-destination transactions */
   BHEADER_W bh;  /* buffer for block header */
   BTRAILER bt;   /* buffer for block trailer */
   LENTRY *lep;   /* pointer to ledger entry */
   struct {
      word8 src_addr[TXADDRLEN];
      word8 chg_addr[TXADDRLEN];
   } *cbp;                    /* malloc()'d change buffer array */
   LTRAN *ltp;                /* malloc()'d ledger transactions array */
   size_t cbp_idx, cbp_len;   /* change buffer index and length */
   size_t ltp_idx, ltp_len;   /* ledger transactions index and length */
   size_t ltp_mtx;            /* additional ltrans space for mtx */
   SHA256_CTX bctx, mctx;
   long long blocklen, chklen;
   word32 total[2], mfees[2], mreward[2];
   word32 hdrlen, count, tcount, j, k;
   word8 tx_id[HASHLEN], prev_tx_id[HASHLEN];
   word8 mroot[HASHLEN], bhash[HASHLEN];
   word8 src_addr[TXADDRLEN];
   word8 dst_addr[TXADDRLEN];
   word8 chg_addr[TXADDRLEN];
   word8 *credit;
   int res;

   /* sanity check */
   if (bcfp == NULL || btp == NULL) goto FAIL_INVAL;

   /* ensure blockchain file is at start */
   rewind(bcfp);

   /* read and check regular fixed size block header */
   if (fread(&hdrlen, sizeof(hdrlen), 1, bcfp) != 1) return VERROR;
   if (hdrlen != sizeof(bh)) goto BAD_HDRLEN;

   /* read block trailer */
   if (fseek64(bcfp, -(sizeof(bt)), SEEK_END) != 0) return VERROR;
   if (fread(&bt, sizeof(bt), 1, bcfp) != 1) return VERROR;

   /* validate block trailer sequence and Proof of Work */
   res = validate_trailer(&bt, btp);
   if (res == VEOK) res = validate_pow(&bt);
   if (res != VEOK) return res;

   /* check transaction count */
   tcount = get32(bt.tcount);
   if (tcount == 0 || tcount > MAXBLTX) goto BAD_TCOUNT;

   /* read and check total block file length */
   chklen = (long long) hdrlen + (tcount * sizeof(txw)) + sizeof(bt);
   blocklen = ftell64(bcfp);
   if (blocklen == EOF) return VERROR;
   if (blocklen != chklen) goto BAD_FILELEN;

   /* read entire block header */
   if (fseek64(bcfp, 0LL, SEEK_SET) != 0) return VERROR;
   if (fread(&bh, sizeof(bh), 1, bcfp) != 1) return VERROR;

   /* check mining reward/address/no-tag */
   get_mreward(mreward, bt.bnum);
   if (memcmp(bh.mreward, mreward, 8) != 0) goto BAD_MREWARD;
   if (WOTS_HAS_TAG(bh.maddr)) goto BAD_MADDR;

   /* bfp left at offset of Merkel Block Array -- ready to fread() */

   /* begin hashing contexts */
   sha256_init(&bctx);
   sha256_update(&bctx, &bh, hdrlen);

   /* IMPORTANT NOTE FOR LEDGER TRANSACTION CREATION:
    * Any transactions to dst_tag's, where the dst_tag also exists as
    * a src_tag in another transaction within the same block, MUST HAVE
    * the dst_addr replaced with the chg_addr of the transaction from
    * which the dst_tag was spent as a src_tag.
    * Example (not necessarily in this order):
    * TX#1: src(A) -> dst(B) -> chg(C);
    * TX#2: src(D) -> dst(A) -> chg(E);
    * ... TX#2's dst(A) needs to be changed to TX#1's chg(C);
    *
    * To address this issue without introducing multiple O(n^2) loops,
    * we store affected addresses in a separate buffer during validation.
    * This buffer is then sorted by source address tag, and used to check
    * the ledger transactions list, after it is built.
   */
   cbp_idx = 0;
   cbp_len = tcount;
   cbp = malloc(cbp_len * sizeof(*cbp));
   ltp_idx = 0;
   ltp_len = 3 * tcount;
   ltp = malloc(ltp_len * sizeof(*ltp));
   /* NOTE: ltp may be reallocated more space as necessary */
   if (cbp == NULL || ltp == NULL) goto FAIL_IO;

   /* read/validate/store each transaction */
   for (j = 0; j < tcount; j++) {
      /* read next transaction into memory */
      if (fread(&txw, sizeof(txw), 1, bcfp) != 1) goto FAIL_IO;

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
      if (memcmp(tx_id, txw.tx_id, HASHLEN) != 0) goto BAD_IO_TX_ID;
      /* check that tx_id is sorted (skip first) */
      if (j != 0) {
         res = memcmp(tx_id, prev_tx_id, HASHLEN);
         if (res <= 0) {
            if (res < 0) goto BAD_IO_TX_SORT;
            if (res == 0) goto BAD_IO_TX_DUP;
         }
      }
      /* remember this tx_id for next time */
      memcpy(prev_tx_id, tx_id, HASHLEN);

      /* validate transaction data */
      /* NOTE: transaction may be modified by TXW_MDST validation */
      res = txw_val((TXW *) &txw, bt.mfee);
      if (res == VERROR) goto FAIL_IO;
      if (res >= VEBAD) goto BAD_IO;

      /* convert WOTS+ transaction addresses to Hashed addresses */
      le_convert(src_addr, txw.src_addr);
      le_convert(chg_addr, txw.chg_addr);

      /* if src is tagged && src_tag == chg_tag, add to change buffer */
      if (ADDR_HAS_TAG(src_addr) && \
            tag_equal(ADDR_TAGp(src_addr), ADDR_TAGp(chg_addr))) {
         memcpy(cbp[cbp_idx].src_addr, src_addr, TXADDRLEN);
         memcpy(cbp[cbp_idx].chg_addr, chg_addr, TXADDRLEN);
         cbp_idx++;
      }

      /* add source address debit by total */
      memcpy(ltp[ltp_idx].addr, src_addr, TXADDRLEN);
      ltp[ltp_idx].trancode[0] = (word8) '-';
      /* sum spend amount -- overflow checked during validation */
      memset(ltp[ltp_idx].amount, 0, TXAMOUNTLEN);
      add64(txw.send_total, txw.change_total, ltp[ltp_idx].amount);
      add64(txw.tx_fee, ltp[ltp_idx].amount, ltp[ltp_idx].amount);
      ltp_idx++;
      /* credit destination address' -- expand for TXW_MDST */
      if (TXW_IS_XTX(&txw) && TXW_XTYPE(&txw) == XTYPE_MTX) {
         mtx = (TXW_MDST *) &txw;
         /* For each dst[] tag... */
         for (k = 0; k < MDST_NUM_DST; k++) {
            /* zero tag marks the end of dst[] */
            if (iszero(mtx->dst[k].tag, TXTAGLEN)) break;
            /* search for tagged ledger entry */
            lep = tag_find(mtx->dst[k].tag);
            if (lep == NULL) {
               /* ... if not found, refund change address */
               credit = chg_addr;
            } else credit = lep->addr;
            /* copy the ledger transaction for mtx->dst[k] */
            memcpy(ltp[ltp_idx].addr, chg_addr, TXADDRLEN);
            ltp[ltp_idx].trancode[0] = (word8) '-';

            if (fwrite(credit,     TXWOTSLEN, 1, ltfp) != 1) goto FAIL_IO;
            if (fwrite("A",                1, 1, ltfp) != 1) goto FAIL_IO;
            if (fwrite(mtx->dst[k].amount, 8, 1, ltfp) != 1) goto FAIL_IO;

            /* search for spent tags... */
            txp = bsearch(mtx->dst[k].tag, txsa, count, sizeof(*txsa),
               search_tag_in_txsa_by_src_tag);
            if (txp == NULL) {
               /* ... if no spent tag, search for ledger address */
               lewp = le_tagfindw(mtx->dst[k].tag, TXTAGLEN);
               if (lewp == NULL) {
                  /* ... if no ledger address, refund change address */
                  credit = mtx->chg_addr;
               } else credit = lewp->addr;
            } else credit = txp->chg_addr;
            /* write out the ledger transaction for mtx->dst[k] */
            if (fwrite(credit,     TXWOTSLEN, 1, ltfp) != 1) goto FAIL_IO;
            if (fwrite("A",                1, 1, ltfp) != 1) goto FAIL_IO;
            if (fwrite(mtx->dst[k].amount, 8, 1, ltfp) != 1) goto FAIL_IO;
         }
      } else if (!iszero(txw.send_total, 8)) {
         /* convert destination address */
         le_convert(dst_addr, txw.dst_addr);
         /* search for spent tags */
         txp = bsearch(WOTS_TAGp(txs[j].dst_addr),
            txsa, count, sizeof(*txsa), search_tag_in_txsa_by_src_tag);
         if (txp) credit = txp->chg_addr;
         else credit = txs[j].dst_addr;
         if (fwrite(credit,    TXWOTSLEN, 1, ltfp) != 1) goto FAIL_IO;
         if (fwrite("A",               1, 1, ltfp) != 1) goto FAIL_IO;
         if (fwrite(txs[j].send_total, 8, 1, ltfp) != 1) goto FAIL_IO;
      }
      /* add to or create change address */
      if(!iszero(txs[j].change_total, 8)) {
         credit = txs[j].chg_addr;
         if (fwrite(credit,      TXWOTSLEN, 1, ltfp) != 1) goto FAIL_IO;
         if (fwrite("A",                 1, 1, ltfp) != 1) goto FAIL_IO;
         if (fwrite(txs[j].change_total, 8, 1, ltfp) != 1) goto FAIL_IO;
      }
      /* additionally, sum fees for miner credit */
      if (add64(mfees, tx.tx_fee, mfees)) goto BAD_IO_FEES_OVERFLOW;
   }  /* end for j */

   /* finalize Merkel Root - phash, bnum, mfee, tcount, time0, difficulty */
   sha256_update(&bctx, &bt, (HASHLEN + 8 + 8 + 4 + 4 + 4));
   memcpy(&mctx, &bctx, sizeof(mctx));
   sha256_final(&mctx, mroot);
   if (memcmp(bt.mroot, mroot, HASHLEN) != 0) goto BAD_IO_MROOT;

   /* finalize block hash - Block trailer (- block hash) */
   sha256_update(&bctx, &bt, sizeof(BTRAILER) - HASHLEN);
   sha256_final(&bctx, bhash);
   if (memcmp(bt.bhash, bhash, HASHLEN) != 0) goto BAD_IO_BHASH;







   /* sort transaction pointer array, by src_tag */
   qsort(txsa, count, sizeof(*txsa), sort_pptxqe_by_src_tag);

   /* create ledger transactions */
   for (j = 0; j < tcount; j++) {
      /* sum spend amount -- overflow checked during validation */
      total[0] = total[1] = 0;
      add64(txs[j].send_total, txs[j].change_total, total);
      add64(txs[j].tx_fee, total, total);
      /* debit source address by total */
      if (fwrite(txs[j].src_addr, TXWOTSLEN, 1, ltfp) != 0) goto FAIL_IO;
      if (fwrite("-",                     1, 1, ltfp) != 0) goto FAIL_IO;
      if (fwrite(total,                   8, 1, ltfp) != 0) goto FAIL_IO;
      /* credit destination address' -- expand for TXW_MDST */
      if (TXW_IS_XTX(&txs[j]) && TXW_XTYPE(&txs[j]) == XTYPE_MTX) {
         mtx = (TXW_MDST *) &(txs[j]);
         /* For each dst[] tag... */
         for (j = 0; j < MDST_NUM_DST; j++) {
            /* zero tag marks the end of dst[] */
            if (iszero(mtx->dst[j].tag, TXTAGLEN)) break;
            /* search for spent tags... */
            txp = bsearch(mtx->dst[j].tag, txsa, count, sizeof(*txsa),
               search_tag_in_txsa_by_src_tag);
            if (txp == NULL) {
               /* ... if no spent tag, search for ledger address */
               lewp = le_tagfindw(mtx->dst[j].tag, TXTAGLEN);
               if (lewp == NULL) {
                  /* ... if no ledger address, refund change address */
                  credit = mtx->chg_addr;
               } else credit = lewp->addr;
            } else credit = txp->chg_addr;
            /* write out the ledger transaction for mtx->dst[j] */
            if (fwrite(credit,     TXWOTSLEN, 1, ltfp) != 1) goto FAIL_IO;
            if (fwrite("A",                1, 1, ltfp) != 1) goto FAIL_IO;
            if (fwrite(mtx->dst[j].amount, 8, 1, ltfp) != 1) goto FAIL_IO;
         }
      } else if (!iszero(txs[j].send_total, 8)) {
         /* search for spent tags */
         txp = bsearch(WOTS_TAGp(txs[j].dst_addr),
            txsa, count, sizeof(*txsa), search_tag_in_txsa_by_src_tag);
         if (txp) credit = txp->chg_addr;
         else credit = txs[j].dst_addr;
         if (fwrite(credit,    TXWOTSLEN, 1, ltfp) != 1) goto FAIL_IO;
         if (fwrite("A",               1, 1, ltfp) != 1) goto FAIL_IO;
         if (fwrite(txs[j].send_total, 8, 1, ltfp) != 1) goto FAIL_IO;
      }
      /* add to or create change address */
      if(!iszero(txs[j].change_total, 8)) {
         credit = txs[j].chg_addr;
         if (fwrite(credit,      TXWOTSLEN, 1, ltfp) != 1) goto FAIL_IO;
         if (fwrite("A",                 1, 1, ltfp) != 1) goto FAIL_IO;
         if (fwrite(txs[j].change_total, 8, 1, ltfp) != 1) goto FAIL_IO;
      }
      /* additionally, sum fees for miner credit */
      if (add64(mfees, tx.tx_fee, mfees)) goto BAD_IO_FEES_OVERFLOW;
   }

   /* cleanup malloc()'d resources */
   free(txs);
   free(txsa);

   /* Create a transaction amount = mreward + mfees
   * address = bh.maddr
   */
   if (add64(mreward, mfees, mreward)) goto BAD_MREWARDS_OVERFLOW;
   /* Make ledger tran to add to or create mining address.
   * '...Money from nothing...'
   */
   if (fwrite(bh.maddr, TXWOTSLEN, 1, ltfp) != 1) return VERROR;
   if (fwrite("A",              1, 1, ltfp) != 1) return VERROR;
   if (fwrite(mreward,          8, 1, ltfp) != 1) return VERROR;

DONE:
   /* block is valid */
   return VEOK;

FAIL_INVAL: set_errno(EINVAL); return VERROR;
FAIL_IO:
   free(ltp);
   free(cbp);
   return res;

BAD_HDRLEN: set_errno(EMCM_HDRLEN); return VEBAD;
BAD_TCOUNT: set_errno(EMCM_TCOUNT); return VEBAD;
BAD_FILELEN: set_errno(EMCM_FILELEN); return VEBAD;
BAD_MREWARD: set_errno(EMCM_MREWARD); return VEBAD;
BAD_MADDR: set_errno(EMCM_MADDR); return VEBAD;
BAD_MREWARDS_OVERFLOW: set_errno(EMCM_MREWARDS_OVERFLOW); return VEBAD;
BAD_IO_TX_ID: set_errno(EMCM_TX_ID); goto BAD_IO;
BAD_IO_TX_SORT: set_errno(EMCM_TX_SORT); goto BAD_IO;
BAD_IO_TX_DUP: set_errno(EMCM_TX_DUP); goto BAD_IO;
BAD_IO_MROOT: set_errno(EMCM_MROOT); goto BAD_IO;
BAD_IO_BHASH: set_errno(EMCM_BHASH); goto BAD_IO;
BAD_IO_FEES_OVERFLOW: set_errno(EMCM_MFEES_OVERFLOW);
BAD_IO:
   free(ltp);
   free(cbp);
   return VEBAD;
}  /* end bcw_val() */

/**
 * Validate a WOTS+ neogenesis blockchain file.
 * Checks ledger entries are in ascending sort.
 * Checks block hash matches calculated hash.
 * Checks block size matches Neogenesis format.
 * Checks block trailer matches Tfile entry.
 * Checks sum of amounts do not exceed "expected" rewards.
 * NOTE: Tfile should have been verified before neogenesis validation.
 * @param bfile Filename of Neogenesis block to validate
 * @param tfile Filename of Tfile to validate against
 * @returns VEOK on valid Neogenesis block, VEBAD on invalid block, or
 * VERROR on internal error. Check errno for more details.
*/
int ngw_val(char *bfile, char *tfile)
{
   LENTRY_W le, ple;
   SHA256_CTX cctx;
   BTRAILER bt, tft;
   long long tfoffset;
   FILE *fp;
   size_t chklen, count;
   word32 hdrlen, first;
   word8 chash[HASHLEN];
   word8 amounts[8];
   word8 rewards[8];

   /* read trailer of neogenesis block and rewind for compute */
   if ((fp = fopen(bfile, "rb")) == NULL) return VERROR;
   if (fseek(fp, -(sizeof(bt)), SEEK_END) != 0) goto FAIL_IO;
   if (fread(&bt, sizeof(bt), 1, fp) != 1) goto FAIL_IO;
   rewind(fp);

   /* compute block hash, sum ledger amounts and check length */
   chklen = 0;
   sha256_init(&cctx);
   memset(amounts, 0, 8);

   /* read headerlen */
   if (fread(&hdrlen, sizeof(hdrlen), 1, fp) != 1) goto FAIL_IO;

   /* update data from headerlen */
   chklen = sizeof(hdrlen);
   sha256_update(&cctx, &hdrlen, sizeof(hdrlen));

   /* read remaining block data */
   for (first = 1; ; first = 0, chklen += count) {
      /* perform read into ledger entry -- check read count */
      if ((count = fread(&le, 1, sizeof(le), fp)) != sizeof(le)) {
         /* ensure final read is block trailer */
         if (count != sizeof(bt)) goto FAIL_IO_TRAILER;
         break;
      }
      /* check ledger sort -- skip on first read */
      if (!first && memcmp(le.addr, ple.addr, sizeof(le.addr)) <= 0) {
         goto BAD_IO_SORT;
      }
      /* update amounts sum, ensure no overflow */
      if (add64(amounts, le.balance, amounts)) goto BAD_IO_OVERFLOW;
      /* update data from neogenesis */
      memcpy(&ple, &le, sizeof(le));
      sha256_update(&cctx, &le, count);
   }

   /* close file */
   fclose(fp);

   /* add trailer data -(HASHLEN) to computed hash -- finalize */
   sha256_update(&cctx, &bt, sizeof(bt) - HASHLEN);
   sha256_final(&cctx, chash);

   /* check block hash, headerlen and modulus */
   if (memcmp(chash, bt.bhash, HASHLEN) != 0) goto BAD_BHASH;
   if (chklen != hdrlen) goto BAD_HDRLEN;
   if ((chklen % sizeof(le)) != sizeof(hdrlen)) goto BAD_HDRLEN;

   /* calculate Tfile offset for trailer compare */
   tfoffset = sizeof(BTRAILER);
   if (mult64(bt.bnum, &tfoffset, &tfoffset)) goto FAIL_OVERFLOW;
   /* read (tfoffset) trailer of Tfile for trailer compare */
   if ((fp = fopen(tfile, "rb")) == NULL) return VERROR;
   if (fseek64(fp, tfoffset, SEEK_SET) != 0) goto FAIL_IO;
   if (fread(&tft, sizeof(tft), 1, fp) != 1) goto FAIL_IO;
   fclose(fp);

   /* compare Neogenesis block trailer to Tfile trailer */
   if (memcmp(&bt, &tft, sizeof(bt)) != 0) goto BAD_TRAILER;

   /* obtain accurate sum of rewards with Tfile -- check ledger amounts */
   /* NOTE: get_tfrewards() cannot calculate wiped account balances < MFEE,
    * so we only check the Neogenesis amounts do not exceed expected */
   if (get_tfrewards(tfile, rewards, bt.bnum) != VEOK) return VERROR;

   /* check calculated rewards against ledger amounts */
   if (cmp64(amounts, rewards) > 0) goto BAD_AMOUNTS;

   /* neogenesis is valid */
   return VEOK;

/* error handling */
FAIL_OVERFLOW: set_errno(EMCM_MATH64_OVERFLOW); return VERROR;

FAIL_IO_TRAILER: set_errno(EMCM_TLRLEN);
FAIL_IO: fclose(fp);
   return VERROR;

/* invalid handling */
BAD_AMOUNTS: set_errno(EMCM_LE_AMOUNTS_SUM); return VEBAD;
BAD_BHASH: set_errno(EMCM_BHASH); return VEBAD;
BAD_HDRLEN: set_errno(EMCM_HDRLEN); return VEBAD;
BAD_TRAILER: set_errno(EMCM_TRAILER); return VEBAD;

BAD_IO_OVERFLOW: set_errno(EMCM_LE_AMOUNTS_OVERFLOW); goto BAD_IO;
BAD_IO_SORT: set_errno(EMCM_LE_SORT);
BAD_IO: fclose(fp);
   return VEBAD;
}  /* end ngw_val() */

int pseudow_val(char *bfile, char *tfile)
{
   static word8 one[8] = { 1, 0 };

   BTRAILER bt, pbt;
   SHA256_CTX ctx;
   FILE *fp;
   long blocklen;
   word32 hdrlen;
   word8 chash[HASHLEN];
   word8 bnum[8];

   /* read block header, trailer and file length */
   if ((fp = fopen(bfile, "rb")) == NULL) return VERROR;
   if (fread(&hdrlen, sizeof(hdrlen), 1, fp) != 1) goto FAIL_IO;
   if (fread(&bt, sizeof(bt), 1, fp) != 1) goto FAIL_IO;
   if (fseek(fp, 0L, SEEK_END)) goto FAIL_IO;
   if ((blocklen = ftell(fp)) == EOF) goto FAIL_IO;
   fclose(fp);

   /* compute and check block hash */
   sha256_init(&ctx);
   sha256_update(&ctx, &hdrlen, 4);
   sha256_update(&ctx, &bt, sizeof(bt) - HASHLEN);
   sha256_final(&ctx, chash);
   if (memcmp(bt.bhash, chash, HASHLEN) != 0) goto BAD_BHASH;

   /* check header/trailer lengths */
   if (hdrlen != sizeof(hdrlen)) goto BAD_HDRLEN;
   if (blocklen != sizeof(hdrlen) + sizeof(bt)) goto BAD_TLRLEN;
   /* check zeros in block trailer*/
   if (get32(bt.tcount) != 0) goto BAD_TCOUNT;
   if (!iszero(bt.mroot, 32)) goto BAD_MROOT;
   if (!iszero(bt.nonce, 32)) goto BAD_NONCE;

   /* read (tfoffset) trailer of Tfile for trailer compare */
   if (read_trailer(&pbt, tfile) != VEOK) return VERROR;

   /* check block num, hash, and difficulty */
   add64(pbt.bnum, one, bnum);
   if (cmp64(bt.bnum, bnum) != 0) goto BAD_BNUM;
   if (memcmp(bt.phash, pbt.bhash, HASHLEN)) goto BAD_PHASH;
   if (get32(bt.difficulty) != next_difficulty(&pbt)) goto BAD_DIFF;

   /* check block times */
   if (get32(bt.time0) != get32(pbt.stime)) goto BAD_TIME0;
   if (get32(bt.stime) != get32(pbt.stime) + BRIDGE) goto BAD_STIME;
   if (!iszero(bt.mfee, 8)) goto BAD_MFEE;

   /* pseudo-block is valid */
   return VEOK;

/* error handling */
FAIL_IO: fclose(fp);
   return VERROR;

/* invalid handling */
BAD_BHASH: set_errno(EMCM_BHASH); return VEBAD;
BAD_HDRLEN: set_errno(EMCM_HDRLEN); return VEBAD;
BAD_TLRLEN: set_errno(EMCM_TLRLEN); return VEBAD;
BAD_TCOUNT: set_errno(EMCM_TCOUNT); return VEBAD;
BAD_MROOT: set_errno(EMCM_MROOT); return VEBAD;
BAD_NONCE: set_errno(EMCM_NONCE); return VEBAD;
BAD_BNUM: set_errno(EMCM_BNUM); return VEBAD;
BAD_PHASH: set_errno(EMCM_PHASH); return VEBAD;
BAD_DIFF: set_errno(EMCM_DIFF); return VEBAD;
BAD_TIME0: set_errno(EMCM_TIME0); return VEBAD;
BAD_STIME: set_errno(EMCM_STIME); return VEBAD;
BAD_MFEE: set_errno(EMCM_MFEE); return VEBAD;
}  /* end validate_pseudo() */

/* end include guard */
#endif
