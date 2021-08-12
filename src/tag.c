/* tag.c  Tag a big Mochimo address with a little name
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 17 May 2018
 * Revised: 15 December 2019
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

byte *Tagidx;    /* array of all 12-byte tags in ledger order */
word32 Ntagidx;  /* number of tags in Tagidx[] */
#define BAIL(m) { message = m; goto bail; }


/* Release tag index */
void tag_free(void)
{
   if(Tagidx != NULL) free(Tagidx);
   Tagidx = NULL;
   Ntagidx = 0;
}


/* Build the tag index, Tagidx[].
 * Return VEOK if success, else error code.
 */
int tag_buildidx(void)
{
   FILE *fp;
   LENTRY le;
   int message;
   word32 n;
   byte *tp;

   if(Trace) plog("tag_buildidx()");
   if(Tagidx != NULL) return VEOK;  /* index already made */

   fp = fopen("ledger.dat", "rb");
   if(fp == NULL) BAIL(1);
   fseek(fp, 0L, SEEK_END);
   Ntagidx = ftell(fp) / sizeof(le);
   Tagidx = (byte *) malloc(Ntagidx * ADDR_TAG_LEN);
   if(Tagidx == NULL) BAIL(2);  /* no memory */
   fseek(fp, 0L, SEEK_SET);
   for(tp = Tagidx, n = 0; n < Ntagidx; n++, tp += ADDR_TAG_LEN) {
      if(fread(&le, sizeof(le), 1, fp) != 1) break;   /* EOF */
      memcpy(tp, ADDR_TAG_PTR(le.addr), ADDR_TAG_LEN);
   }
   if(n != Ntagidx) BAIL(3);  /* I/O error likely */
   fclose(fp);
   if(Trace) plog("tag_buildidx() success: Ntagidx = %u", Ntagidx);
   return VEOK;  /* index built */
bail:
   if(fp != NULL) fclose(fp);
   if(Tagidx != NULL) free(Tagidx);
   Tagidx = NULL;
   Ntagidx = 0;
   error("tag_buildidx(): BAIL(%d)\007", message);  /* should not happen */
   return message;
}  /* end tag_buildidx() */


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


#if ADDR_TAG_LEN != 12
   ADDR_TAG_LEN must be 12 for tag code in tag.c tag_find()
#endif

/* Find the tag of addr in Tagidx[].
 * If foundaddr or balance is not NULL, copy the
 * full fields from ledger.dat to foundaddr and/or balance.
 * Return VEOK if tag found, VERROR if not found, or
 * some other internal error code.
 */
int tag_find(byte *addr, byte *foundaddr, byte *balance, size_t len)
{
   FILE *fp;
   byte *tag, *tp;
   LENTRY le;
   word32 n;
   int message;

   fp = NULL;  /* for bail */
   if(Tagidx == NULL) tag_buildidx();
   if(Tagidx == NULL) BAIL(2);  /* 2 > VERROR */

   tag = ADDR_TAG_PTR(addr);
   /* Search tag index, Tagidx[] for tag. */
   for(tp = Tagidx, n = 0; n < Ntagidx; n++, tp += ADDR_TAG_LEN) {
      /* compare tag in Tagidx[] to tag */
      if((len != 0 && memcmp(tp, tag, len) == 0) /* partial tag len search */
         || (len == 0 /* full tag len search (about 9 instructions in asm) */
         && *((word32 *) tp)       == *((word32 *) tag)
         && *((word32 *) (tp + 4)) == *((word32 *) (tag + 4))
         && *((word32 *) (tp + 8)) == *((word32 *) (tag + 8))) ) {
         /* tag found */
         if(foundaddr != NULL || balance != NULL) {
            /* and caller wants ledger entry... */
            fp = fopen("ledger.dat", "rb");
            if(fp == NULL) BAIL(3);
            /* n is record number in ledger.dat */
            if(fseek(fp, n * sizeof(le), SEEK_SET)) BAIL(4);
            if(fread(&le, sizeof(le), 1, fp) != 1) BAIL(5);
            if(memcmp(ADDR_TAG_PTR(le.addr), tag, len)) BAIL(6);
            fclose(fp);
            if(foundaddr != NULL) memcpy(foundaddr, le.addr, TXADDRLEN);
            if(balance != NULL) memcpy(balance, le.balance, TXAMOUNT);
         }  /* end if copy entry */
         return VEOK;  /* found tag! */
      }  /* end if memcmp found */
   }  /* end for tp -- search for tag */
   return VERROR;  /* tag not found */
bail:
   if(fp != NULL) fclose(fp);
   tag_free();  /* Erase the bad index */
   error("tag_find(): BAIL(%d)\007", message);  /* should not happen */
   return message;
}  /* end tag_find() */


/* Validate TX address tags.
 * If called from tx_val(), bnum is NULL in order to check
 * queues, txq1.dat and txclean.dat, and always do dst check.
 * When called from bval.c, bnum is not NULL and is checked
 * against tagval_trigger in order to do dst check.
 * Return VEOK if tags are valid, else VERROR to reject TX.
 */
int tag_valid(byte *src_addr, byte *chg_addr, byte *dst_addr, byte *bnum)
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

      if(HAS_TAG(dst_addr)) {
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
   if(tag_find(chg_addr, NULL, NULL, ADDR_TAG_LEN) == VEOK) {
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


#ifndef EXCLUDE_RESOLVE

/* Look-up and return an address tag to np.
 * Called from gettx() opcode == OP_RESOLVE
 *
 * on entry:
 *     tag string at ADDR_TAG_PTR(np->tx.dst_addr)    tag to query
 * on return:
 *     np->tx.send_total = 1 if found, or 0 if not found.
 *     if found: np->tx.dst_addr has full found address with tag.
 *               np->tx.change_total has balance.
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
   put64(np->tx.change_total, zeros);
   /* find tag in leger.dat */
   status = tag_find(np->tx.dst_addr, foundaddr, balance, get16(np->tx.len));
   if(status == VEOK) {
      memcpy(np->tx.dst_addr, foundaddr, TXADDRLEN);
      memcpy(np->tx.change_total, balance, TXAMOUNT);
      put64(np->tx.send_total, One);
      ecode = VEOK;
   }
   send_op(np, OP_RESOLVE);
   return ecode;
}  /* end tag_resolve() */

#endif /* !EXCLUDE_RESOLVE */
