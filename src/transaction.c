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

/**
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
   word8 addr[TXWOTSLEN];
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
   if (!WOTS_HAS_TAG(mtx->src_addr)) goto FAIL_SRC_TAG;
   if (!tag_equal(src_tag, chg_tag)) goto FAIL_SRC_NOT_CHG;
   if (cmp64(mtx->change_total, fee) <= 0) goto FAIL_CHG_DISSOLVE;

   total[0] = total[1] = 0;
   mfees[0] = mfees[1] = 0;
   /* Tally each dst[] amount and mfees... */
   for (j = 0; j < MDST_NUM_DST; j++) {
      /* zero dst[] tag marks end of list */
      if (iszero(mtx->dst[j].tag, TXTAGLEN)) {
         bp = mtx->dst[j].amount;
         if (!iszero(bp, limit - bp)) goto FAIL_NZTPADDING;
         break;
      }
      /* check dst tag is not src tag and non-zero send amount */
      if (tag_equal(mtx->dst[j].tag, src_tag)) goto FAIL_DST_IS_SRC;
      if (iszero(mtx->dst[j].amount, 8)) goto FAIL_DST_AMOUNT;
      /* tally fees and totals */
      if (add64(mfees, fee, mfees)) goto FAIL_FEES_OVERFLOW;
      if (add64(total, mtx->dst[j].amount, total)) goto FAIL_OVERFLOW;
      /* If dst[j] tag not found, put error code in zeros[] array. */
      memcpy(WOTS_TAGp(addr), mtx->dst[j].tag, TXTAGLEN);
      mtx->zeros[j] = tag_find(mtx->dst[j].tag) ? 1 : 0;
   }  /* end for j */
   /* Check tallies... */
   if (cmp64(total, mtx->send_total) != 0) goto FAIL_AMOUNTS;
   if (cmp64(mtx->tx_fee, mfees) < 0) goto FAIL_FEES;

   /* TXW_MDST is valid */
   return VEOK;

FAIL_SRC_TAG: set_errno(EMCM_TXMDST_SRC_TAG); return VERROR;
FAIL_SRC_NOT_CHG: set_errno(EMCM_TXMDST_SRC_NOT_CHG); return VERROR;
FAIL_CHG_DISSOLVE: set_errno(EMCM_TXMDST_CHG_DISSOLVE); return VERROR;
FAIL_NZTPADDING: set_errno(EMCM_XTX_NZTPADDING); return VERROR;
FAIL_DST_IS_SRC: set_errno(EMCM_TXMDST_DST_IS_SRC); return VERROR;
FAIL_DST_AMOUNT: set_errno(EMCM_TXMDST_DST_AMOUNT); return VERROR;
FAIL_FEES_OVERFLOW: set_errno(EMCM_TXMDST_FEES_OVERFLOW); return VERROR;
FAIL_OVERFLOW: set_errno(EMCM_TXMDST_AMOUNTS_OVERFLOW); return VERROR;
FAIL_AMOUNTS: set_errno(EMCM_TXMDST_AMOUNTS); return VERROR;
FAIL_FEES: set_errno(EMCM_TXMDST_FEES); return VERROR;
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
   if (cmp64(tx->tx_fee, fee) < 0) goto BAD_TX_FEE;
   if (memcmp(src_addr, chg_addr, TXWOTSLEN) == 0) goto BAD_TX_CHG_ADDR;
   /* for non-xtx transactions check src != dst and validate tags */
   if (!is_xtx) {
      if (memcmp(src_addr, dst_addr, TXWOTSLEN) == 0) goto BAD_TX_DST_ADDR;
      /* If dst_addr is tagged... */
      if (WOTS_HAS_TAG(dst_addr)) {
         /* check full dst_addr is in ledger, else invalid */
         if (le_findw(dst_addr) == NULL) goto FAIL_ADDRNOTAVAIL;
      }
      src_tag = WOTS_TAGp(src_addr);
      chg_tag = WOTS_TAGp(chg_addr);
      /* If change tag exists and src_tag != chg_tag (transfer), check... */
      if (WOTS_HAS_TAG(chg_addr) && !tag_equal(src_tag, chg_tag)) {
         /* ... if src is not default, tx is invalid. */
         if (WOTS_HAS_TAG(src_addr)) goto FAIL_TX_SRC_TAGGED;
         /* ... if change tag is in ledger.dat, tx is invalid. */
         if (tag_find(chg_tag)) goto FAIL_TX_CHG_TAG;
      }
   }

   /* look up source address in ledger */
   le = le_findw(src_addr);
   if (le == NULL) goto FAIL_ADDRNOTAVAIL;
   /* use add64() to prepare totals and check overflow */
   total[0] = total[1] = 0;
   overflow = add64(tx->send_total, tx->change_total, total);
   overflow += add64(tx->tx_fee, total, total);
   if (overflow) goto BAD_TX_AMOUNTS_OVERFLOW;
   /* check totals match ledger balance */
   if (cmp64(le->balance, total) != 0) goto FAIL_TX_SRC_LE_BALANCE;

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
   if (memcmp(PUBKEY, tx->src_addr, TXSIGLEN) != 0) goto BAD2_TXWOTS_SIG;
   /* check for eXtended TX transaction type */
   if (is_xtx) {
      /* eXtended TX transaction type validation methods */
      switch (xtype) {
         case XTYPE_MTX: {
            mtx = (TXW_MDST *) tx->src_addr;  /* poor man's union */
            return txw_mdst_val(mtx, fee);
         }
         case XTYPE_MEMO: /* fallthrough -- for now */
         default: goto FAIL_XTX_UNDEFINED;
      }  /* end switch (xtype) */
   }  /* end if (is_xtx) */

   /* TX is valid */
   return VEOK;

/* internal error handling */
FAIL_ADDRNOTAVAIL: set_errno(EADDRNOTAVAIL); return VERROR;
FAIL_TX_SRC_TAGGED: set_errno(EMCM_TX_SRC_TAGGED); return VERROR;
FAIL_TX_CHG_TAG: set_errno(EMCM_TX_CHG_TAG); return VERROR;
FAIL_TX_SRC_LE_BALANCE: set_errno(EMCM_TX_SRC_LE_BALANCE); return VERROR;
FAIL_XTX_UNDEFINED: set_errno(EMCM_XTX_UNDEFINED); return VERROR;

/* protocol violation handling */
BAD_TX_FEE: set_errno(EMCM_TX_FEE); return VEBAD;
BAD_TX_CHG_ADDR: set_errno(EMCM_TX_CHG_ADDR); return VEBAD;
BAD_TX_DST_ADDR: set_errno(EMCM_TX_DST_ADDR); return VEBAD;
BAD_TX_AMOUNTS_OVERFLOW: set_errno(EMCM_TX_AMOUNTS_OVERFLOW); return VEBAD;

/* malicious violation handling */
BAD2_TXWOTS_SIG: set_errno(EMCM_TXWOTS_SIG); return VEBAD2;
}  /* end txw_val() */

/* end include guard */
#endif
