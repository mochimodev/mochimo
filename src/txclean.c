/* txclean.c  Remove bad TX's from txclean.dat
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * The Mochimo Project System Software
 *
 * Date: 2 April 2018
 * Revised: 10 May 2019
 *
 * NOTE: Called by update() after bval and bup.
 *
 * Inputs:  ledger.dat   NO-ONE ELSE is using this file!
 *          txclean.dat
 *
 * Outputs: txclean.dat without bad TX's.
*/


/* Return 0 on success, else error code.
 * Leaves ledger.dat open on return.
 */
int txclean(void)
{
   static TXQENTRY tx;     /* Holds one transaction in the array */
   FILE *fp, *fpout;       /* txclean.dat */
   static LENTRY src_le;   /* for le_find() */
   int count, message, tnum, j;
   word32 nout;            /* temp file output record counter */
   word32 total[2];
   static byte addr[TXADDRLEN];
   MTX *mtx;

   /* open the clean TX queue (txclean.dat) to read */
   fp = fpout = NULL;
   fp = fopen("txclean.dat", "rb");
   if(!fp) BAIL(1);

   /* create new clean TX queue */
   fpout = fopen("txq.tmp", "wb");
   if(!fpout) BAIL(2);

   /* open ledger read-only */
   if(le_open("ledger.dat", "rb") != VEOK)
      BAIL(3);

   nout = 0;    /* output counter */

   for(tnum = 0; ; tnum++) {
      /* read TX from txclean.dat */
      count = fread(&tx, 1, sizeof(TXQENTRY), fp);
      if(count != sizeof(TXQENTRY)) break;  /* EOF */
      /* if src not in ledger continue; */
      if(le_find(tx.src_addr, &src_le, NULL, 0) == FALSE) continue;
      if(cmp64(tx.tx_fee, Myfee) < 0) continue;  /* bad tx fee */
      /* check total overflow and balance */
      if(add64(tx.send_total, tx.change_total, total)) continue;
      if(add64(tx.tx_fee, total, total)) continue;
      if(cmp64(src_le.balance, total) != 0) continue;  /* bad balance */
      if(ismtx(&tx) && get32(Cblocknum) >= MTXTRIGGER) {
         mtx = (MTX *) &tx;
         for(j = 0; j < NR_DST; j++) {
            if(iszero(mtx->dst[j].tag, ADDR_TAG_LEN)) break;
            memcpy(ADDR_TAG_PTR(addr), mtx->dst[j].tag, ADDR_TAG_LEN);
            mtx->zeros[j] = 0;
            /* If dst[j] tag not found, put error code in zeros[] array. */
            if(tag_find(addr, addr, NULL) != VEOK) mtx->zeros[j] = 1;
         }
      }
      count = fwrite(&tx, 1, sizeof(TXQENTRY), fpout);
      if(count != sizeof(TXQENTRY)) BAIL(4);
      nout++;
   }  /* end for */

   fclose(fp);
   fclose(fpout);
   fp = fpout = NULL;

   unlink("txclean.dat");
   if(nout) {
      /* if there are entries in txq.tmp */
      if(rename("txq.tmp", "txclean.dat")) BAIL(5);
   } else {
      unlink("txq.tmp");  /* remove empty temp file */
      if(Trace) plog("txclean.dat is empty.");
   }

   if(Trace && nout) plog("txclean.c: wrote %u entries from %u"
                          " to new txclean.dat", nout, tnum);
   return 0;        /* success */

bail:
   if(fp) fclose(fp);
   if(fpout) fclose(fpout);
   unlink("txq.tmp");
   if(Trace) plog("txclean(): %d", message);
   return message;
}  /* end txclean() */
