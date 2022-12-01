/**
 * @private
 * @headerfile transaction.h <transaction.h>
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_TRANSACTION_C
#define MOCHIMO_TRANSACTION_C


#include "transaction.h"

/* internal support */
#include "wots.h"
#include "ledger.h"
#include "error.h"

/* external support */
#include <string.h>
#include "sha256.h"
#include "extmath.h"
#include <ctype.h>   /* for isprint() */

#define TXERR(e)    { set_errno(e); return VERROR; }
#define TXBAD(e)    { set_errno(e); return VEBAD;  }
#define TXBAD2(e)   { set_errno(e); return VEBAD2; }

/**
 * Validate a MEMO Hashed transaction.
 * Includes tag checking. Fee is set by caller:
 * - Transaction validator sets fee parameter to Myfee.
 * - Blockchain validator sets fee to trailer Mfee.
 * @param tx Pointer to a MEMO Hashed transaction to validate
 * @param fee Pointer to fee to validate against
 * @return (int) value representing operation result
 * @retval VEOK on success, multi-destination transaction is valid
 * @retval VERROR on error, check errno for details
*/
int tx_memo_val(TX_MEMO *mtx, void *fee)
{
   word8 *src_tag, *chg_tag;
   int j;

   /* init */
   src_tag = ADDR_TAGp(mtx->src_addr);
   chg_tag = ADDR_TAGp(mtx->chg_addr);

   /* Transaction validator has already checked...
    * src != chg, src exists, sig is good, and totals are good.
    */

   /* check src is tagged, matches chg, and chg tag will NOT dissolve */
   if (!ADDR_HAS_TAG(mtx->src_addr)) TXERR(EMCMTXSRCNOTAG);
   if (!tag_equal(src_tag, chg_tag)) TXERR(EMCMXTXTAGMISMATCH);
   if (cmp64(mtx->change_total, fee) <= 0) TXERR(EMCMXTXCHGTOTAL);
   /* check dst tag is not src tag, exists in ledger and non-zero send */
   if (tag_equal(mtx->dst_tag, src_tag)) TXERR(EMCMXTXTAGMATCH);
   if (tag_find(mtx->dst_tag) == NULL) TXERR(EMCMXTXTAGNOLE);
   if (iszero(mtx->send_total, 8)) TXERR(EMCMXTXSENDTOTAL);
   /* check MEMO is null terminated */
   if (mtx->dst_memo[TXMEMOLEN - 1] != 0) TXERR(EMCMXTXNOTERM);
   /* check MEMO consist of only printable, non-punctuation characters */
   for (j = 0; j < TXMEMOLEN; j++) {
      /* Zero marks the end of the MEMO */
      if (mtx->dst_memo[j] == 0) break;
      if (ispunct(mtx->dst_memo[j])) TXERR(EMCMXTXHASPUNCT);
      if (!isprint(mtx->dst_memo[j])) TXERR(EMCMXTXNONPRINT);
   }
   /* check MEMO zero padding */
   if (!iszero(mtx->dst_memo + j, TXMEMOLEN - j)) TXERR(EMCMXTXNZTPADDING);

   /* TX_MEMO is valid */
   return VEOK;
}  /* end tx_memo_val() */

/**
 * Validate a Hashed transaction. Requires an open ledger.
 * @param tx Pointer to a Hashed transaction to validate
 * @param fee Pointer to fee to validate against
 * @return (int) value representing operation result
 * @retval VEOK on success, transaction is valid
 * @retval VERROR on internal error, check errno for details
 * @retval VEBAD on protocol violation, bad transaction data
 * @retval VEBAD2 on malicious violation, invalid WOTS+ signature
*/
int tx_val(TX *tx, void *fee)
{
   SHA256_CTX mctx;
   LENTRY *le;             /* ledger entry */
   word32 total[2];        /* for 64-bit maths */
   word32 ADRS[8];         /* for WOTS+, ADRS[] */
   word8 MESSAGE[HASHLEN]; /* for WOTS+, transaction hash */
   word8 PUBKEY[TXSIGLEN]; /* for WOTS+, public_key[] */
   word8 *PUBSEEDp;        /* for WOTS+, public_seed pointer */
   word8 *src_addr, *dst_addr, *chg_addr, *src_tag, *chg_tag;
   int overflow, is_xtx, xtype;

   /* init */
   is_xtx = TX_IS_XTX(tx);
   xtype = TX_XTYPE(tx);
   src_addr = tx->src_addr;
   dst_addr = tx->dst_addr;
   chg_addr = tx->chg_addr;

   /* validate transaction fixed fee and src != chg */
   if (cmp64(tx->tx_fee, fee) < 0) TXBAD(EMCMTXFEE);
   if (memcmp(src_addr, chg_addr, TXADDRLEN) == 0) TXBAD(EMCMTXSRCISCHG);
   /* for non-xtx transactions check src != dst and validate tags */
   if (!is_xtx) {
      if (memcmp(src_addr, dst_addr, TXADDRLEN) == 0) TXBAD(EMCMTXSRCISDST);
      /* If dst_addr is tagged... */
      if (ADDR_HAS_TAG(dst_addr)) {
         /* check full dst_addr is in ledger, else invalid */
         if (le_find(dst_addr) == NULL) TXERR(EMCMTXDSTNOLE);
      }
      src_tag = ADDR_TAGp(src_addr);
      chg_tag = ADDR_TAGp(chg_addr);
      /* If change tag exists and src_tag != chg_tag (transfer), check... */
      if (ADDR_HAS_TAG(chg_addr) && !tag_equal(src_tag, chg_tag)) {
         /* ... if src is not default, tx is invalid. */
         if (ADDR_HAS_TAG(src_addr)) TXERR(EMCMTXTAGSRC);
         /* ... if change tag is in ledger.dat, tx is invalid. */
         if (tag_find(chg_tag)) TXERR(EMCMTXTAGCHG);
      }
   }

   /* look up source address in ledger */
   le = le_find(src_addr);
   if (le == NULL) TXERR(EMCMTXSRCNOLE);
   /* use add64() to prepare totals and check overflow */
   total[0] = total[1] = 0;
   overflow = add64(tx->send_total, tx->change_total, total);
   overflow += add64(tx->tx_fee, total, total);
   if (overflow) TXBAD(EMCMTXOVERFLOW);
   /* check totals match ledger balance */
   if (cmp64(le->balance, total) != 0) TXERR(EMCMTXTOTAL);

   /* MDST HASHED TRANSACTION LOGIC UNDECIDED...
   if (is_xtx && xtype == XTYPE_MTX) {
      mtx = (TX_MDST *) tx;
      memset(mtx->zeros, 0, MDST_NUM_DZEROS);
   } */

   /* check WOTS+ Signature against transaction hash (message) */
   sha256_init(&mctx);
   sha256_update(&mctx, tx->src_addr, TXADDRLEN);
   sha256_update(&mctx, tx->dst_addr, TXADDRLEN);
   sha256_update(&mctx, tx->chg_addr, TXADDRLEN);
   sha256_update(&mctx, tx->send_total, TXAMOUNTLEN);
   sha256_update(&mctx, tx->change_total, TXAMOUNTLEN);
   sha256_update(&mctx, tx->tx_fee, TXAMOUNTLEN);
   sha256_update(&mctx, tx->tx_ttl, TXAMOUNTLEN);
   sha256_update(&mctx, tx->tx_spk, HASHLEN);
   sha256_final(&mctx, MESSAGE);
   PUBSEEDp = src_addr + TXSIGLEN;
   memcpy(ADRS, PUBSEEDp + 32, 32);  /* copy WOTS ADRS[] */
   wots_pk_from_sig(PUBKEY, tx->tx_sig, MESSAGE, PUBSEEDp, ADRS);
   if (memcmp(PUBKEY, tx->src_addr, TXSIGLEN) != 0) TXBAD2(EMCMTXWOTS);
   /* check for eXtended TX transaction type */
   if (is_xtx) {
      /* eXtended TX transaction type validation methods */
      switch (xtype) {
         /* MDST HASHED TRANSACTION LOGIC UNDECIDED...
         case XTYPE_MTX: {
            mtx = (TXW_MDST *) tx;
            return txw_mdst_val(mtx, fee);
         } */
         case XTYPE_MEMO: return tx_memo_val((TX_MEMO *) tx, fee);
         default: TXBAD(EMCMXTXNODEF);
      }  /* end switch (xtype) */
   }  /* end if (is_xtx) */

   /* TX is valid */
   return VEOK;
}  /* end tx_val() */

/**
 * Validate a multi-destination WOTS+ transaction.
 * Includes tag checking. Fee is set by caller:
 * - Transaction validator sets fee parameter to Myfee.
 * - Blockchain validator sets fee to trailer Mfee.
 * @param tx Pointer to a MDST WOTS+ transaction to validate
 * @param fee Pointer to fee to validate against
 * @return (int) value representing operation result
 * @retval VEOK on success, multi-destination transaction is valid
 * @retval VERROR on error, check errno for details
*/
int txw_mdst_val(TXW_MDST *mtx, void *fee)
{
   word8 *bp, *limit;
   word32 total[2], mfees[2];
   word8 *src_tag, *chg_tag;
   int j;

   /* init */
   limit = &mtx->zeros[0];
   src_tag = WOTS_TAGp(mtx->src_addr);
   chg_tag = WOTS_TAGp(mtx->chg_addr);

   /* Transaction validator has already checked...
    * src != chg, src exists, sig is good, and (non-mtx) totals are good.
    */

   /* check src is tagged, matches chg, and chg tag will NOT dissolve */
   if (!WOTS_HAS_TAG(mtx->src_addr)) TXERR(EMCMTXSRCNOTAG);
   if (!tag_equal(src_tag, chg_tag)) TXERR(EMCMXTXTAGMISMATCH);
   if (cmp64(mtx->change_total, fee) <= 0) TXERR(EMCMXTXCHGTOTAL);

   total[0] = total[1] = 0;
   mfees[0] = mfees[1] = 0;
   /* Tally each dst[] amount and mfees... */
   for (j = 0; j < MDST_NUM_DST; j++) {
      /* zero dst[] tag marks end of list */
      if (iszero(mtx->dst[j].tag, TXTAGLEN)) {
         bp = mtx->dst[j].amount;
         if (!iszero(bp, limit - bp)) TXBAD(EMCMXTXNZTPADDING);
         break;
      }
      /* check dst tag is not src tag and non-zero send amount */
      if (tag_equal(mtx->dst[j].tag, src_tag)) TXERR(EMCMXTXTAGMATCH);
      if (iszero(mtx->dst[j].amount, 8)) TXBAD(EMCMXTXDSTAMOUNT);
      /* tally fees and totals */
      if (add64(mfees, fee, mfees)) TXERR(EMCMTXFEEOVERFLOW);
      if (add64(total, mtx->dst[j].amount, total)) TXERR(EMCMTXOVERFLOW);
      /* If dst[j] tag not found, put error code in zeros[] array. */
      mtx->zeros[j] = tag_find(mtx->dst[j].tag) ? 1 : 0;
   }  /* end for j */
   /* Check tallies... */
   if (cmp64(total, mtx->send_total) != 0) TXERR(EMCMXTXTOTALS);
   if (cmp64(mtx->tx_fee, mfees) < 0) TXERR(EMCMXTXFEES);

   /* TXW_MDST is valid */
   return VEOK;
}  /* end txw_mdst_val() */

/**
 * Validate a WOTS+ transaction. Requires an open ledger.
 * @param tx Pointer to a WOTS+ transaction to validate
 * @param fee Pointer to fee to validate against
 * @return (int) value representing operation result
 * @retval VEOK on success, transaction is valid
 * @retval VERROR on internal error, check errno for details
 * @retval VEBAD on protocol violation, bad transaction data
 * @retval VEBAD2 on malicious violation, invalid WOTS+ signature
*/
int txw_val(TXW *tx, void *fee)
{
   SHA256_CTX mctx;
   LENTRY *le;             /* ledger entry */
   TXW_MDST *mtx;          /* for mtx specific WOTS+ sig check */
   word32 total[2];        /* for 64-bit maths */
   word32 ADRS[8];         /* for WOTS+, ADRS[] */
   word8 MESSAGE[HASHLEN]; /* for WOTS+, transaction hash */
   word8 PUBKEY[TXSIGLEN]; /* for WOTS+, public_key[] */
   word8 *PUBSEEDp;        /* for WOTS+, public_seed pointer */
   word8 *src_addr, *dst_addr, *chg_addr, *src_tag, *chg_tag;
   int overflow, is_xtx, xtype;

   /* init */
   is_xtx = TXW_IS_XTX(tx);
   xtype = TXW_XTYPE(tx);
   src_addr = tx->src_addr;
   dst_addr = tx->dst_addr;
   chg_addr = tx->chg_addr;

   /* validate transaction fixed fee and src != chg */
   if (cmp64(tx->tx_fee, fee) < 0) TXBAD(EMCMTXFEE);
   if (memcmp(src_addr, chg_addr, TXWOTSLEN) == 0) TXBAD(EMCMTXSRCISCHG);
   /* for non-xtx transactions check src != dst and validate tags */
   if (!is_xtx) {
      if (memcmp(src_addr, dst_addr, TXWOTSLEN) == 0) TXBAD(EMCMTXSRCISDST);
      /* If dst_addr is tagged... */
      if (WOTS_HAS_TAG(dst_addr)) {
         /* check full dst_addr is in ledger, else invalid */
         if (le_findw(dst_addr) == NULL) TXERR(EMCMTXDSTNOLE);
      }
      src_tag = WOTS_TAGp(src_addr);
      chg_tag = WOTS_TAGp(chg_addr);
      /* If change tag exists and src_tag != chg_tag (transfer), check... */
      if (WOTS_HAS_TAG(chg_addr) && !tag_equal(src_tag, chg_tag)) {
         /* ... if src is not default, tx is invalid. */
         if (WOTS_HAS_TAG(src_addr)) TXERR(EMCMTXTAGSRC);
         /* ... if change tag is in ledger.dat, tx is invalid. */
         if (tag_find(chg_tag)) TXERR(EMCMTXTAGCHG);
      }
   }

   /* look up source address in ledger */
   le = le_findw(src_addr);
   if (le == NULL) TXERR(EMCMTXSRCNOLE);
   /* use add64() to prepare totals and check overflow */
   total[0] = total[1] = 0;
   overflow = add64(tx->send_total, tx->change_total, total);
   overflow += add64(tx->tx_fee, total, total);
   if (overflow) TXERR(EMCMTXOVERFLOW);
   /* check totals match ledger balance */
   if (cmp64(le->balance, total) != 0) TXERR(EMCMTXTOTAL);

   /* TXW_MDST transactions are always signed with trailing zeros */
   if (is_xtx && xtype == XTYPE_MTX) {
      mtx = (TXW_MDST *) tx->src_addr;  /* poor man's union */
      memset(mtx->zeros, 0, MDST_NUM_DZEROS);
   }
   /* check WOTS+ Signature against transaction hash (message) */
   sha256_init(&mctx);
   sha256_update(&mctx, tx->src_addr, TXWOTSLEN);
   sha256_update(&mctx, tx->dst_addr, TXWOTSLEN);
   sha256_update(&mctx, tx->chg_addr, TXWOTSLEN);
   sha256_update(&mctx, tx->send_total, TXAMOUNTLEN);
   sha256_update(&mctx, tx->change_total, TXAMOUNTLEN);
   sha256_update(&mctx, tx->tx_fee, TXAMOUNTLEN);
   sha256_final(&mctx, MESSAGE);
   PUBSEEDp = src_addr + TXSIGLEN;
   memcpy(ADRS, PUBSEEDp + 32, 32);  /* copy WOTS ADRS[] */
   wots_pk_from_sig(PUBKEY, tx->tx_sig, MESSAGE, PUBSEEDp, ADRS);
   if (memcmp(PUBKEY, tx->src_addr, TXSIGLEN) != 0) TXBAD(EMCMTXWOTS);
   /* check for eXtended TX transaction type */
   if (is_xtx) {
      /* eXtended TX transaction type validation methods */
      switch (xtype) {
         case XTYPE_MTX: {
            mtx = (TXW_MDST *) tx->src_addr;  /* poor man's union */
            return txw_mdst_val(mtx, fee);
         }
         case XTYPE_MEMO: /* fallthrough -- for now */
         default: TXBAD(EMCMXTXNODEF);
      }  /* end switch (xtype) */
   }  /* end if (is_xtx) */

   /* TX is valid */
   return VEOK;
}  /* end txw_val() */

/* end include guard */
#endif
