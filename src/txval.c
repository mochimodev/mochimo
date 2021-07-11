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

#include "mtxval.c"  /* multi-dst validator */


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
   static byte message[HASHLEN];    /* transaction hash for WOTS */
   static byte pk2[TXSIGLEN];       /* more WOTS */
   static byte rnd2[32];            /* for WOTS addr[] */
   MTX *mtx;
   static TX txs;

   if(memcmp(tx->src_addr, tx->chg_addr, TXADDRLEN) == 0) {
      if(Trace) plog("tx_val(): src == chg");  /* also mtx */
      return 2;
   }

   if(!ismtx(tx) && memcmp(tx->src_addr, tx->dst_addr, TXADDRLEN) == 0) {
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
   if(ismtx(tx) && get32(Cblocknum) >= MTXTRIGGER) {
      memcpy(&txs, tx, sizeof(txs));
      mtx = (MTX *) TRANBUFF(&txs);  /* poor man's union */
      memset(mtx->zeros, 0, NR_DZEROS);
      sha256(txs.src_addr, SIG_HASH_COUNT, message);
   } else {
      sha256(tx->src_addr, SIG_HASH_COUNT, message);
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
   if(ismtx(tx)) {
      mtx = (MTX *) TRANBUFF(tx);  /* poor man's union */
      if(mtx_val(mtx, Myfee)) return 1;  /* bad mtx */
   } else {
      if(tag_valid(tx->src_addr, tx->chg_addr, tx->dst_addr,
                   NULL) != VEOK) return 1;  /* bad tag */
   }
   return 0;  /* tx valid */
}  /* end tx_val() */
