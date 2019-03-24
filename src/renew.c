/* renew.c  Spin the Carousel on Lastday
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 14 March 2019
 *
*/


#define CAROUSEL(bnum) (get32(bnum) == Lastday)


/* Returns 0 on success, else error code. */
int renew(void)
{
   FILE *fp, *fpout;
   LENTRY le;
   int message = 0;
   word32 n, m;
   static word32 sanctuary[2];

   if(Sanctuary == 0) return 0;  /* success */
   le_close();  /* make sure ledger.dat is closed */
   plog("Lastday 0x%0x.  Carousel begins...", Lastday);
   n = m = 0;
   fp = fpout = NULL;
   sanctuary[0] = Sanctuary;

   fp = fopen("ledger.dat", "rb");
   if(fp == NULL) BAIL(1);
   fpout = fopen("ledger.tmp", "wb");
   if(fpout == NULL) BAIL(2);
   for(;;) {
      if(fread(&le, sizeof(le), 1, fp) != 1) break;
      n++;
      if(sub64(le.balance, sanctuary, le.balance)) continue;
      if(cmp64(le.balance, Mfee) <= 0) continue;
      if(fwrite(&le, sizeof(le), 1, fpout) != 1) BAIL(3);
      m++;
   }
   fclose(fp);
   fclose(fpout);
   fp = fpout = NULL;
   unlink("ledger.dat");
   if(rename("ledger.tmp", "ledger.dat")) BAIL(4);
   plog("%u citizens renewed out of %u", n - m, n);
   return 0;  /* success */
bail:
   if(fp != NULL) fclose(fp);
   if(fpout != NULL) fclose(fpout);
   error("Carousel renewal code: %d (%u,%u)", message, n - m, n);
   return message;
}  /* end renew() */


/* Refresh the ip list and catch up if needed.
 * Called from server().
 * Returns result code.
 */
int refresh_ipl(void)
{
   NODE node;
   int j, message = 0;
   word32 ip;
   byte bnum[8];

   for(j = ip = 0; j < 1000 && ip == 0; j++)
      ip = Rplist[rand16() % RPLISTLEN];
   if(ip == 0) BAIL(1);
   if(get_ipl(&node, ip) != VEOK) BAIL(2);
   /* ignore low block num */
   if(cmp64(node.tx.cblock, Cblocknum) <= 0) BAIL(3);
   /* ignore low weight */
   if(cmp_weight(node.tx.weight, Weight) <= 0) BAIL(4);

   /* catchup loop */
   put64(bnum, Cblocknum);
   for( ; Running; ) {
      add64(bnum, One, bnum);
      if(bnum[0] == 0) continue;  /* do not fetch NG blocks */
      if(get_block2(ip, bnum, "refresh.tmp", OP_GETBLOCK) != VEOK) BAIL(5);
      if(update("refresh.tmp", 0) != VEOK) BAIL(6);
   }
bail:
   if(Trace) plog("refresh_ipl(): %d", message);
   return message;
}  /* end refresh_ipl() */
