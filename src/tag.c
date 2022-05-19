/**
 * @private
 * @headerfile tag.h <tag.h>
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_tag_C
#define MOCHIMO_tag_C


#include "tag.h"

/* internal support */
#include "wots.h"
#include "util.h"
#include "trigg.h"
#include "peach.h"
#include "ledger.h"
#include "global.h"

/* external support */
#include "extlib.h"
#include "extmath.h"
#include <errno.h>
#include "crc16.h"

word8 *Tagidx;    /* array of all 12-word8 tags in ledger order */
word32 Ntagidx;   /* number of tags in Tagidx[] */

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
   word8 *tp;

   pdebug("tag_buildidx()");
   if(Tagidx != NULL) return VEOK;  /* index already made */

   fp = fopen("ledger.dat", "rb");
   if(fp == NULL) BAIL(1);
   fseek(fp, 0L, SEEK_END);
   Ntagidx = ftell(fp) / sizeof(le);
   Tagidx = (word8 *) malloc(Ntagidx * TXTAGLEN);
   if(Tagidx == NULL) BAIL(2);  /* no memory */
   fseek(fp, 0L, SEEK_SET);
   for(tp = Tagidx, n = 0; n < Ntagidx; n++, tp += TXTAGLEN) {
      if(fread(&le, sizeof(le), 1, fp) != 1) break;   /* EOF */
      memcpy(tp, ADDR_TAG_PTR(le.addr), TXTAGLEN);
   }
   if(n != Ntagidx) BAIL(3);  /* I/O error likely */
   fclose(fp);
   pdebug("tag_buildidx() success: Ntagidx = %u", Ntagidx);
   return VEOK;  /* index built */
bail:
   if(fp != NULL) fclose(fp);
   if(Tagidx != NULL) free(Tagidx);
   Tagidx = NULL;
   Ntagidx = 0;
   perr("tag_buildidx(): BAIL(%d)\007", message);  /* should not happen */
   return message;
}  /* end tag_buildidx() */


/* Search txq1.dat and txclean.dat for a tag matching tag of addr in
 * some pending TX's change address.
 * Return VEOK if the tag is found in a chg_addr, otherwise VERROR.
 */
int tag_qfind(word8 *addr)
{
   FILE *fp;
   TXQENTRY tx;
   word8 *tag, *txtag;

   tag = ADDR_TAG_PTR(addr);
   txtag = ADDR_TAG_PTR(tx.chg_addr);

   fp = fopen("txq1.dat", "rb");
   if(fp != NULL) {
      for(;;) {
         if(fread(&tx, 1, sizeof(TXQENTRY), fp) != sizeof(TXQENTRY)) break;
         if(memcmp(tag, txtag, TXTAGLEN) == 0) {
            fclose(fp);
            return VEOK;  /* found */
         }
      }  /* end for */
      fclose(fp);
   }  /* end if fp */

   /* Stop Block Constructor who uses txclean.dat. */
   stop_bcon();

   fp = fopen("txclean.dat", "rb");
   if(fp != NULL) {
      for(;;) {
         if(fread(&tx, 1, sizeof(TXQENTRY), fp) != sizeof(TXQENTRY)) break;
         if(memcmp(tag, txtag, TXTAGLEN) == 0) {
            fclose(fp);
            return VEOK;  /* found */
         }
      }  /* end for */
      fclose(fp);
   }  /* end if fp */
   return VERROR;  /* tag not found */
}  /* end tag_qfind() */

/* Find the tag of addr in Tagidx[].
 * If foundaddr or balance is not NULL, copy the
 * full fields from ledger.dat to foundaddr and/or balance.
 * Return VEOK if tag found, VERROR if not found, or
 * some other internal error code.
 */
int tag_find(word8 *addr, word8 *foundaddr, word8 *balance, size_t len)
{
   FILE *fp;
   word8 *tag, *tp;
   LENTRY le;
   word32 n;
   int message;

   fp = NULL;  /* for bail */
   if(Tagidx == NULL) tag_buildidx();
   if(Tagidx == NULL) BAIL(2);  /* 2 > VERROR */

   tag = ADDR_TAG_PTR(addr);
   /* Search tag index, Tagidx[] for tag. */
   for(tp = Tagidx, n = 0; n < Ntagidx; n++, tp += TXTAGLEN) {
      /* compare tag in Tagidx[] to tag */
      if(  /* partial tag len search */
         (len > 1 && len < TXTAGLEN && memcmp(tp, tag, len) == 0)
         || (  /* full tag len search (about 9 instructions in asm) */
            *((word32 *) tp)       == *((word32 *) tag)
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
   perr("tag_find(): BAIL(%d)\007", message);  /* should not happen */
   return message;
}  /* end tag_find() */

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
            pdebug("DST_ADDR Tagged, but Tag is not in ledger!");
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
      pdebug("SRC_TAG != CHG_TAG, and SRC_TAG is Non-Default!");
      goto bad;
   }
   /* Otherwise, check all queues and ledger.dat for change tag.
    * First, if change tag is in ledger.dat, tx is invalid.
    */
   if(tag_find(chg_addr, NULL, NULL, TXTAGLEN) == VEOK) {
      pdebug("New CHG_TAG Already Exists in Ledger!");
      goto bad;
   }
   if(bnum == NULL) {
      /* If called from tx_val(),
       * and if tag is in txq1.dat or txclean.dat, tx is invalid.
       */
      if(tag_qfind(chg_addr) == VEOK) {
         pdebug("Tag is already in queue");
         goto bad;
      }
   }
   pdebug("Tag created");
   return VEOK;  /* If we get here, a new TX change tag gets created. */
good:
   pdebug("Tag moved");
   return VEOK;
bad:
   pdebug("Tag rejected");
   return VERROR;
}  /* end tag_valid() */

/* end include guard */
#endif
