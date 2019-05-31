/* tag.c  Tag a big Mochimo address with a little name
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 17 May 2018
 * Revised: 2 Sep 2018
 *
*/

/****  Tested with wallet31.c  ****/

/*
   [<---2196 Bytes Address--->][<--12 Bytes Tag-->]

   12-byte tag:  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                 ^
              type byte
*/


#define ADDR_TAG_PTR(addr) (((byte *) (addr)) + 2196)
#define ADDR_TAG_LEN 12
#define HAS_TAG(addr) \
   (((byte *) (addr))[2196] != 0x42 && ((byte *) (addr))[2196] != 0x00)


/* Search txq1.dat and txclean.dat for a tag matching tag of addr in
 * some pending TX's change address.
 * Return VEOK if the tag is found in a chg_addr, otherwise VERROR.
 */
int tag_qfind(byte *addr)
{
   FILE *fp;
   TXQENTRY tx;
   byte *tag, *txtag;

   tag = ADDR_TAG_PTR(addr);
   txtag = ADDR_TAG_PTR(tx.chg_addr);

   fp = fopen("txq1.dat", "rb");
   if(fp != NULL) {
      for(;;) {
         if(fread(&tx, 1, sizeof(TXQENTRY), fp) != sizeof(TXQENTRY)) break;
         if(memcmp(tag, txtag, ADDR_TAG_LEN) == 0) {
            fclose(fp);
            return VEOK;  /* found */
         }
      }  /* end for */
      fclose(fp);
   }  /* end if fp */

   /* Stop Block Constructor who uses txclean.dat. */
   if(Bcpid) {
      kill(Bcpid, SIGTERM);
      waitpid(Bcpid, NULL, 0);
      Bcpid = 0;
   }

   fp = fopen("txclean.dat", "rb");
   if(fp != NULL) {
      for(;;) {
         if(fread(&tx, 1, sizeof(TXQENTRY), fp) != sizeof(TXQENTRY)) break;
         if(memcmp(tag, txtag, ADDR_TAG_LEN) == 0) {
            fclose(fp);
            return VEOK;  /* found */
         }
      }  /* end for */
      fclose(fp);
   }  /* end if fp */
   return VERROR;  /* tag not found */
}  /* end tag_qfind() */


/* Find the tag of addr in ledger.dat and copy the
 * full address to foundaddr.
 * Return VEOK if tag found, else VERROR.
 */
int tag_find(byte *addr, byte *foundaddr, byte *balance)
{
   FILE *fp;
   byte *tag;
   LENTRY le;

   fp = fopen("ledger.dat", "rb");
   if(fp == NULL) return error("tag_find(): Cannot open ledger.dat");
   tag = ADDR_TAG_PTR(addr);
   for(;;) {
      if(fread(&le, 1, sizeof(LENTRY), fp) != sizeof(LENTRY)) break;
      if(memcmp(tag, ADDR_TAG_PTR(le.addr), ADDR_TAG_LEN) == 0)
      {
    	  memcpy(foundaddr, le.addr, TXADDRLEN);
         if(balance != NULL)
         {
        	 memcpy(balance, le.balance, TXAMOUNT);
         }

         fclose(fp);
         return VEOK;  /* found */
      }
   }
   fclose(fp);
   return VERROR;  /* not found */
}  /* end tag_find() */


/* Validate TX address tags.
 * If called from tx_val(), checkq is non-zero in
 * order to check queues, txq1.dat and txclean.dat.
 * Return VEOK if tags are valid, else VERROR to reject TX.
 */
int tag_valid(byte *src_addr, byte *chg_addr, byte *dst_addr, int checkq, byte *bnum)
{
   LENTRY le;
   word32 tagval_trigger[2];

   tagval_trigger[0] = tagval_trigger[1] = 0;
   if(checkq == 0 && bnum != NULL) {
      tagval_trigger[0] = RTRIGGER31; /* For v2.0 */
   }
   if(cmp64(bnum, tagval_trigger) >= 0) {
   /* Ignore the below check prior to block 17185...
    * src_addr was already found in ledger.dat and dup checked
    * by txval or bval.
    *
    * Check dst_addr.  If no dst_tag, dst_addr is valid: */

      if(HAS_TAG(dst_addr)) {
         /* If there is a dst_tag, and its full address is not
          * already in ledger.dat, tx is not valid.
          */
         if(le_find(dst_addr, &le, NULL, 0) == FALSE) {
            plog("DST_ADDR Tagged, but Tag is not in ledger!");
            goto bad;
         }
      }
   }
   /* If no change tag, tx is valid. */
   if(!HAS_TAG(chg_addr)) return VEOK;
   /* If src and chg tags are the same, tx is valid (transfer). */
   if(memcmp(ADDR_TAG_PTR(src_addr),
             ADDR_TAG_PTR(chg_addr), ADDR_TAG_LEN) == 0) goto good;

   /* If tags are not the same and the src is not default, tx invalid. */
   if(HAS_TAG(src_addr)) {
      plog("SRC_TAG != CHG_TAG, and SRC_TAG is Non-Default!"); 
      goto bad;
   }
   /* Otherwise, check all queues and ledger.dat for change tag.
    * First, if change tag is in ledger.dat, tx is invalid.
    */
   if(tag_find(chg_addr, le.addr, NULL) == VEOK) {
      plog("New CHG_TAG Already Exists in Ledger!");
      goto bad;
   }
   if(checkq) {
      /* If called from tx_val(),
       * and if tag is in txq1.dat or txclean.dat, tx is invalid.
       */
      if(tag_qfind(chg_addr) == VEOK) {
          plog("tag_qfind() returned VEOK");
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


#ifndef EXCLUDE_RESOLVE

/* Look-up and return an address tag to np.
 * Called from gettx() opcode == OP_RESOLVE
 *
 * on entry:
 *     tag string at ADDR_TAG_PTR(np->tx.dst_addr)    tag to query
 * on return:
 *     np->tx.send_total = 1 if found, or 0 if not found.
 *     np->tx.dst_addr has full found address with tag.
 *
 * Returns VEOK if found, else VERROR.
*/
int tag_resolve(NODE *np)
{
   byte foundaddr[TXADDRLEN];
   static byte zeros[8];
   byte balance[TXAMOUNT];
   int status, ecode = VERROR;

   put64(np->tx.send_total, zeros);
   status = tag_find(np->tx.dst_addr, foundaddr, balance);  /* in legger.dat */
   if(status == VEOK) {
      memcpy(np->tx.dst_addr, foundaddr, TXADDRLEN);
      memcpy(np->tx.change_total, balance, TXAMOUNT);
      put64(np->tx.send_total, One);

      ecode = VEOK;
   }
   send_op(np, OP_RESOLVE);
   return ecode;
}  /* end tag_resolve() */

#endif /* EXCLUDE_RESOLVE */
