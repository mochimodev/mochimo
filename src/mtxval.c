/* mtxval.c  Multi-dst transaction validator
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * **** NO WARRANTY ****
 *
 * Date: 6 May 2019
*/


#define BAIL(m) { message = m; goto bail; }

/* Validates a multi-dst transaction MTX.
 * (Does all tag checking as well.)
 * tx->src_addr is already checked in ledger.dat and totals tally.
 * tx_val() sets fee parameter to Myfee and bval.c sets fee to Mfee.
 * Returns 0 on valid, else error code.
 */
int mtx_val(MTX *mtx, word32 *fee)
{
   byte addr[TXADDRLEN];
   int j, message;
   byte total[8], mfees[8], *bp, *limit;

   limit = &mtx->zeros[0];

   /* Check that src and chg have tags.
    * Check that src and chg have same tag.
    * tx_val() or bval.c has already checked src != chg, src exists, 
    *   sig is good, and totals are good.
    */
   if(!HAS_TAG(mtx->src_addr)) BAIL(1);
   if(memcmp(ADDR_TAG_PTR(mtx->src_addr),
             ADDR_TAG_PTR(mtx->chg_addr), ADDR_TAG_LEN) != 0) BAIL(2);
   if(cmp64(mtx->change_total, Mfee) <= 0) BAIL(3);

   if(!iszero(mtx->zeros, 208)) BAIL(4);  /* reserved with ismtx() tag */
   memset(total, 0, 8);
   memset(mfees, 0, 8);
   /* Tally each dst[] amount and mfees... */
   for(j = 0; j < 100; j++) {
      /* zero dst[] tag marks end of list.  */
      if(iszero(mtx->dst[j].tag, ADDR_TAG_LEN)) {
         for(bp = mtx->dst[j].amount; bp < limit; bp++) {
            if(*bp) BAIL(5);  /* Check that rest of dst[] list is zeros. */
         }
         break;
      }
      if(iszero(mtx->dst[j].amount, 8)) BAIL(6);  /* bad send amount */
      /* no dst to src */
      if(memcmp(mtx->dst[j].tag,
                ADDR_TAG_PTR(mtx->src_addr), ADDR_TAG_LEN) == 0) BAIL(7);
      /* tally fees and send_total */
      if(add64(total, mtx->dst[j].amount, total)) BAIL(8);
      if(add64(total, fee, mfees)) BAIL(9);  /* Mfee or Myfee */
   }  /* end for j */
   /* Check tallies... */
   if(cmp64(total, mtx->send_total) != 0) BAIL(10);
   if(cmp64(mtx->tx_fee, mfees) < 0) BAIL(11);
   return 0;  /* valid */
bail:
   if(message && Trace) plog("mtx_val(): %d", message);
   return message;  /* bad */
}  /* end mtx_val() */
