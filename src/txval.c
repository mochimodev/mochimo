/* txval.c  Transaction Validator
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * The Mochimo Project System Software
 *
 * Date: 8 January 2018
 *
 * NOTE: called from process_tx() in mirror.c
 *       le_open() has been called.
 *
 * Returns exit code 0 on valid TX,
 *                   1 on server errors, or
 *                   2 or 3 if peer is evil.
 *
 * Inputs:  tx parameter points to the TX struct to validate.
 *
 * Requires legder.c, mtxval.c
 *
*/

#include "extint.h"

#include "config.h"
#include "data.c"
#include "types.h"

#define BAIL(m) { message = m; goto bail; }

/* Validates a multi-dst transaction MTX.
 * (Does all tag checking as well.)
 * tx->src_addr is already checked in ledger.dat and totals tally.
 * tx_val() sets fee parameter to Myfee and bval.c sets fee to Mfee.
 * Returns 0 on valid, else error code.
 */
int mtx_val(MTX *mtx, word32 *fee)
{
   int j, message;
   word8 total[8], mfees[8], *bp, *limit;
   static word8 addr[TXADDRLEN];

   limit = &mtx->zeros[0];

   /* Check that src and chg have tags.
    * Check that src and chg have same tag.
    * tx_val() or bval.c has already checked src != chg, src exists,
    *   sig is good, and totals are good.
    */
   if(!ADDR_HAS_TAG(mtx->src_addr)) BAIL(1);
   if(memcmp(ADDR_TAG_PTR(mtx->src_addr),
             ADDR_TAG_PTR(mtx->chg_addr), TXTAGLEN) != 0) BAIL(2);
   if(cmp64(mtx->change_total, Mfee) <= 0) BAIL(3);

   memset(total, 0, 8);
   memset(mfees, 0, 8);
   /* Tally each dst[] amount and mfees... */
   for(j = 0; j < MDST_NUM_DST; j++) {
      /* zero dst[] tag marks end of list.  */
      if(iszero(mtx->dst[j].tag, TXTAGLEN)) {
         for(bp = mtx->dst[j].amount; bp < limit; bp++) {
            if(*bp) BAIL(4);  /* Check that rest of dst[] list is zeros. */
         }
         break;
      }
      if(iszero(mtx->dst[j].amount, 8)) BAIL(5);  /* bad send amount */
      /* no dst to src */
      if(memcmp(mtx->dst[j].tag,
                ADDR_TAG_PTR(mtx->src_addr), TXTAGLEN) == 0) BAIL(6);
      /* tally fees and send_total */
      if(add64(total, mtx->dst[j].amount, total)) BAIL(7);
      if(add64(mfees, fee, mfees)) BAIL(8);  /* Mfee or Myfee */
      if(get32(Cblocknum) >= MTXTRIGGER) {
         memcpy(ADDR_TAG_PTR(addr), mtx->dst[j].tag, TXTAGLEN);
         mtx->zeros[j] = 0;
         /* If dst[j] tag not found, put error code in zeros[] array. */
         if(tag_find(addr, NULL, NULL, TXTAGLEN) != VEOK) mtx->zeros[j] = 1;
      }
   }  /* end for j */
   /* Check tallies... */
   if(cmp64(total, mtx->send_total) != 0) BAIL(9);
   if(cmp64(mtx->tx_fee, mfees) < 0) BAIL(10);
   return 0;  /* valid */
bail:
   if(message && Trace) plog("mtx_val(): %d", message);
   return message;  /* bad */
}  /* end mtx_val() */


/* Validate TX address tags.
 * If called from tx_val(), bnum is NULL in order to check
 * queues, txq1.dat and txclean.dat, and always do dst check.
 * When called from bval.c, bnum is not NULL and is checked
 * against tagval_trigger in order to do dst check.
 * Return VEOK if tags are valid, else VERROR to reject TX.
 */
int tag_valid(word8 *src_addr, word8 *chg_addr, word8 *dst_addr, word8 *bnum)
{
   LENTRY le;
   static word32 tagval_trigger[2] = { RTRIGGER31, 0 };  /* For v2.0 */

   if(bnum == NULL || cmp64(bnum, tagval_trigger) >= 0) {
      /* Do below check on or after block 17185 when called
       * from bval().  If called from tx_val(), always perform
       * check.  src_addr was already found in ledger.dat and dup
       * already checked by txval or bval.
       *
       * Check dst_addr.  If no dst_tag, dst_addr is valid:
       */

      if(ADDR_HAS_TAG(dst_addr)) {
         /* If there is a dst_tag, and its full address is not
          * already in ledger.dat, tx is not valid.
          */
         if(le_find(dst_addr, &le, NULL, TXADDRLEN) == FALSE) {
            plog("DST_ADDR Tagged, but Tag is not in ledger!");
            goto bad;
         }
      }
   }  /* end if dst tag check */
   /* If no change tag, tx is valid. */
   if(!ADDR_HAS_TAG(chg_addr)) return VEOK;
   /* If src and chg tags are the same, tx is valid (transfer). */
   if(memcmp(ADDR_TAG_PTR(src_addr),
             ADDR_TAG_PTR(chg_addr), TXTAGLEN) == 0) goto good;

   /* If tags are not the same and the src is not default, tx invalid. */
   if(ADDR_HAS_TAG(src_addr)) {
      plog("SRC_TAG != CHG_TAG, and SRC_TAG is Non-Default!");
      goto bad;
   }
   /* Otherwise, check all queues and ledger.dat for change tag.
    * First, if change tag is in ledger.dat, tx is invalid.
    */
   if(tag_find(chg_addr, NULL, NULL, TXTAGLEN) == VEOK) {
      plog("New CHG_TAG Already Exists in Ledger!");
      goto bad;
   }
   if(bnum == NULL) {
      /* If called from tx_val(),
       * and if tag is in txq1.dat or txclean.dat, tx is invalid.
       */
      if(tag_qfind(chg_addr) == VEOK) {
         plog("Tag is already in queue");
         goto bad;
      }
   }
   if(Trace) plog("Tag created");
   return VEOK;  /* If we get here, a new TX change tag gets created. */
good:
   if(Trace) plog("Tag moved");
   return VEOK;
bad:
   if(Trace) plog("Tag rejected");
   return VERROR;
}  /* end tag_valid() */



/* Validate a transaction against ledger
 *
 * Returns: 0 if vaild (accept)
 *          1 if server error (drop)
 *          2 or 3 if evil    (drop)
 */
int tx_val(TX *tx)
{
   int cond;
   static LENTRY src_le;            /* source ledger entry */
   word32 total[2];                 /* for 64-bit maths */
   static word8 message[HASHLEN];    /* transaction hash for WOTS */
   static word8 pk2[TXSIGLEN];       /* more WOTS */
   static word8 rnd2[32];            /* for WOTS addr[] */
   MTX *mtx;
   static TX txs;

   if(memcmp(tx->src_addr, tx->chg_addr, TXADDRLEN) == 0) {
      if(Trace) plog("tx_val(): src == chg");  /* also mtx */
      return 2;
   }

   if(!TX_IS_MTX(tx) && memcmp(tx->src_addr, tx->dst_addr, TXADDRLEN) == 0) {
      if(Trace) plog("tx_val(): src == dst");
      return 2;
   }

   /* validate transaction fixed fee */
   if(cmp64(tx->tx_fee, Mfee) < 0) {
      if(Trace) plog("tx_val(): bad mining fee");
      return 2;
   }
   /* validate my fee */
   if(cmp64(tx->tx_fee, Myfee) < 0) {
      if(Trace) plog("tx_val(): fee < %u", Myfee[0]);
      return 1;
   }

   /* check WTOS signature */
   if(TX_IS_MTX(tx) && get32(Cblocknum) >= MTXTRIGGER) {
      memcpy(&txs, tx, sizeof(txs));
      mtx = (MTX *) TRANBUFF(&txs);  /* poor man's union */
      memset(mtx->zeros, 0, MDST_NUM_DZEROS);
      sha256(txs.src_addr, TRANSIGHASHLEN, message);
   } else {
      sha256(tx->src_addr, TRANSIGHASHLEN, message);
   }

   memcpy(rnd2, &tx->src_addr[TXSIGLEN+32], 32);  /* copy WOTS addr[] */
   wots_pk_from_sig(pk2, tx->tx_sig, message, &tx->src_addr[TXSIGLEN],
                    (word32 *) rnd2);
   if(memcmp(pk2, tx->src_addr, TXSIGLEN) != 0) {
      plog("tx_val(): WOTS signature failed!");
      return 3;
   }

   /* look up source address in ledger */
   if(le_find(tx->src_addr, &src_le, NULL, TXADDRLEN) == FALSE) {
      if(Trace) plog("tx_val(): src_addr not in ledger");
      return 1;
   }
   total[0] = total[1] = 0;
   /* use add64() to check for overflow */
   cond =  add64(tx->send_total, tx->change_total, total);
   cond += add64(tx->tx_fee, total, total);
   if(cond) {
      plog("tx_val(): TX amount overflow");
      return 2;
   }
   if(cmp64(src_le.balance, total) != 0) {
      if(Trace) plog("tx_val(): bad transaction total != src_le.balance");
      return 1;
   }
   if(TX_IS_MTX(tx)) {
      mtx = (MTX *) TRANBUFF(tx);  /* poor man's union */
      if(mtx_val(mtx, Myfee)) return 1;  /* bad mtx */
   } else {
      if(tag_valid(tx->src_addr, tx->chg_addr, tx->dst_addr,
                   NULL) != VEOK) return 1;  /* bad tag */
   }
   return 0;  /* tx valid */
}  /* end tx_val() */
