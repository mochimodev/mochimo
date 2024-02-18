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
#include "ledger.h"
#include "global.h"
#include "error.h"

/* external support */
#include <string.h>
#include <stdlib.h>
#include "extmath.h"

word8 *Tagidx;    /* array of all 12-word8 tags in ledger order */
word32 Ntagidx;   /* number of tags in Tagidx[] */

/**
 * @private
 * Efficient 12-byte Address Tag comparison.
 * @param a Pointer to tag
 * @param b Pointer to tag
 * @returns 1 if tags are equal, else 0
*/
static inline int tag_equal(word8 *a, word8 *b)
{
   return (
      *((word32 *) a) == *((word32 *) b)
      && *((word32 *) (a + 4)) == *((word32 *) (b + 4))
      && *((word32 *) (a + 8)) == *((word32 *) (b + 8))
   );
}

/**
 * Clear the global tag index.
 * Free's memory allocated to Tagidx and sets Ntagidx to 0.
*/
void tag_free(void)
{
   pdebug("entered...");

   Ntagidx = 0;
   if (Tagidx != NULL) {
      free(Tagidx);
      Tagidx = NULL;
      pdebug("Tagidx free'd");
   }
}

/**
 * Build the global tag index, Tagidx[]. Uses "ledger.dat" as ledger file.
 * @returns VEOK if success, else error code.
 */
int tag_buildidx(void)
{
   LENTRY le;
   FILE *fp;
   word8 *tp;
   size_t len;
   word32 n;

   pdebug("building...");

   /* check Tagidx for existing build */
   if (Tagidx != NULL) {
      pdebug("tag index already made");
      return VEOK;
   }

   /* open ledger file for binary reading */
   fp = fopen("ledger.dat", "rb");
   if (fp == NULL) {
      perrno("cannot open ledger.dat");
      goto FAIL;
   }
   /* obtain ledger sizeand idx count */
   if (fseek(fp, 0L, SEEK_END) != 0) {
      perrno("fseek(END)");
      goto FAIL_IO;
   }
   Ntagidx = ftell(fp) / sizeof(le);
   /* malloc space for tags */
   Tagidx = (word8 *) malloc((len = Ntagidx * TXTAGLEN));
   if (Tagidx == NULL) {
      perrno("malloc(%zu)", len);
      goto FAIL_IO;
   }
   /* return ledger file pointer to beginning */
   if (fseek(fp, 0L, SEEK_SET) != 0) {
      perrno("fseek(SET)");
      goto FAIL_IO;
   }
   /* read ledger entry tags into Tagidx */
   for (tp = Tagidx, n = 0; n < Ntagidx; n++, tp += TXTAGLEN) {
      if (fread(&le, sizeof(le), 1, fp) != 1) {
         if (ferror(fp)) {
            perrno("file error");
            goto FAIL_IO;
         }
         break;  /* EOF */
      }
      /* copy current ledger entry tag into Tagidx via tp */
      memcpy(tp, ADDR_TAG_PTR(le.addr), TXTAGLEN);
   }
   fclose(fp);

   /* success */
   pdebug("success, Ntagidx = %" P32u, Ntagidx);
   return VEOK;

   /* failure / error handling */
FAIL_IO:
   fclose(fp);
FAIL:
   tag_free();
   return VERROR;
}  /* end tag_buildidx() */

/**
 * @private
 * Search txq1.dat and txclean.dat change addresses for the tag in @a addr.
 * @param addr Pointer to address containing tag to search for
 * @returns VEOK if the tag is found in a chg_addr, otherwise VERROR
 */
int tag_qfind(word8 *addr)
{
   TXQENTRY tx;
   FILE *fp;
   word8 *tag, *txtag;

   /* init */
   pdebug("searching queues...");
   tag = ADDR_TAG_PTR(addr);
   txtag = ADDR_TAG_PTR(tx.chg_addr);

   /* check the (dirty) transaction queue */
   fp = fopen("txq1.dat", "rb");
   if (fp != NULL) {  /* search for tag in txq1.dat */
      while (fread(&tx, sizeof(tx), 1, fp) && !tag_equal(tag, txtag));
      if (ferror(fp)) {
         perrno("fread(txq1.dat)");
         goto FAIL_IO;
      }
      if (feof(fp)) fclose(fp);  /* tag not found */
      else {  /* tag found */
         fclose(fp);
         return VEOK;
      }
   }

   /* Stop Block Constructor who uses txclean.dat. */
   stop_bcon();

   /* check the (clean) transaction queue */
   fp = fopen("txclean.dat", "rb");
   if (fp != NULL) {  /* search for tag in txclean.dat */
      while (fread(&tx, sizeof(tx), 1, fp) && !tag_equal(tag, txtag));
      if (ferror(fp)) {
         perrno("fread(txclean.dat)");
         goto FAIL_IO;
      }
      if (feof(fp)) fclose(fp);  /* tag not found */
      else {  /* tag found */
         fclose(fp);
         return VEOK;
      }
   }

   /* tag not found */
   return VERROR;

   /* failure / error handling */
FAIL_IO:
   fclose(fp);
   return VERROR;
}  /* end tag_qfind() */

/**
 * Find the tag of addr in Tagidx[]. If foundaddr or balance is not NULL,
 * copy the full fields from ledger.dat to foundaddr and/or balance.
 * @param addr Pointer to address containing tag to search for
 * @param foundaddr Pointer to place "full" found address
 * @param balance Pointer to place balance of found address
 * @param len Tag search length
 * @returns VEOK if tag found, VERROR if not found or error encountered.
 * @note The @a len parameter must be a value between (but not including)
 * One (1) and TXTAGLEN (12) to search for a "partial" tag match. All other
 * values will search for "full" tag.
*/
int tag_find(word8 *addr, word8 *foundaddr, word8 *balance, size_t len)
{
   LENTRY le;
   FILE *fp;
   word8 *tag, *tp;
   word32 n;

   /* init */
   pdebug("searching Tagidx...");

   /* check/build Tagidx */
   if (Tagidx == NULL && tag_buildidx() != VEOK) {
      perr("failed to tag_buildidx()");
      goto FAIL;
   }

   /* init */
   tag = ADDR_TAG_PTR(addr);
   /* determine "search type" via requested length */
   if (len > 1 && len < TXTAGLEN) {
      /* Search tag index, Tagidx[] for "partial" tag */
      for (tp = Tagidx, n = 0; n < Ntagidx; n++, tp += TXTAGLEN) {
         if (memcmp(tp, tag, len) == 0) break;
      }
   } else {
      /* Search tag index, Tagidx[] for "full" tag */
      for (tp = Tagidx, n = 0; n < Ntagidx; n++, tp += TXTAGLEN) {
         if (tag_equal(tp, tag)) break;
      }
   }
   /* check search result */
   if (n != Ntagidx) {
      if (foundaddr != NULL || balance != NULL) {
         /* and caller wants ledger entry... */
         fp = fopen("ledger.dat", "rb");
         if (fp == NULL) {
            perrno("cannot open ledger.dat");
            goto FAIL;
         }
         /* n is record number in ledger.dat */
         if (fseek(fp, n * sizeof(le), SEEK_SET) != 0) {
            perrno("fseek(SET)");
            goto FAIL_IO;
         } else if (fread(&le, sizeof(le), 1, fp) != 1) {
            perrno("fread(le)");
            goto FAIL_IO;
         } else if (memcmp(ADDR_TAG_PTR(le.addr), tag, len)) {
            perrno("memcmp(SET)");
            goto FAIL_IO;
         } else fclose(fp);
         /* copy address/balance to available pointers */
         if (foundaddr != NULL) memcpy(foundaddr, le.addr, TXADDRLEN);
         if (balance != NULL) memcpy(balance, le.balance, TXAMOUNT);
      }  /* end if (foundaddr... || balance... */
      /* success -- tag found */
      return VEOK;
   }

   /* success -- tag not found */
   return VERROR;

   /* failure / error handling */
FAIL_IO:
   fclose(fp);
FAIL:
   tag_free();
   return VERROR;
}  /* end tag_find() */

/**
 * Validate tags of a transaction.
 * If called from tx_val(), bnum is NULL in order to check
 * queues, txq1.dat and txclean.dat, and always do dst check.
 * When called from bval.c, bnum is not NULL and is checked
 * against tagval_trigger in order to do dst check.
 * @param src_addr Pointer to source address of transaction
 * @param chg_addr Pointer to change address of transaction
 * @param dst_addr Pointer to destination address of transaction
 * @param bnum Pointer to block number, or NULL
 * @returns VEOK if tags are valid, else VERROR to reject TX.
*/
int tag_valid(word8 *src_addr, word8 *chg_addr, word8 *dst_addr, word8 *bnum)
{
   static word32 tagval_trigger[2] = { RTRIGGER31, 0 };  /* For v2.0 */
   LENTRY le;

   if (bnum == NULL || cmp64(bnum, tagval_trigger) >= 0) {
      /* Do below check on or after block 17185 when called from bval().
       * If called from tx_val(), always perform check.
       * Check dst_addr.  If no dst_tag, dst_addr is valid:
      */
      if (ADDR_HAS_TAG(dst_addr)) {
         /* If there is a dst_tag, and its full address is not
          * already in ledger.dat, tx is not valid.
          */
         if (le_find(dst_addr, &le, NULL, TXADDRLEN) == FALSE) {
            pdebug("DST_ADDR Tagged, but Tag is not in ledger!");
            return VERROR;
         }
      }
   }  /* end if dst tag check */
   /* If no change tag, tx is valid. */
   if (!ADDR_HAS_TAG(chg_addr)) return VEOK;
   /* If src and chg tags are the same, tx is valid (transfer). */
   if (tag_equal(ADDR_TAG_PTR(src_addr), ADDR_TAG_PTR(chg_addr))) {
      return VEOK;
   }
   /* If tags are not the same and the src is not default, tx invalid. */
   if (ADDR_HAS_TAG(src_addr)) {
      pdebug("SRC_TAG != CHG_TAG, and SRC_TAG is Non-Default!");
      return VERROR;
   }
   /* If change tag is in ledger.dat, tx is invalid. */
   if (tag_find(chg_addr, NULL, NULL, TXTAGLEN) == VEOK) {
      pdebug("New CHG_TAG Already Exists in Ledger!");
      return VERROR;
   }
   /* If called from tx_val() and chg tag is in queue, tx is invalid. */
   if (bnum == NULL && tag_qfind(chg_addr) == VEOK) {
      pdebug("Tag is already in queue");
      return VERROR;
   }
   /* If we get here, a new TX change tag gets created, tx is valid. */
   pdebug("New CHG_TAG created");
   return VEOK;
}  /* end tag_valid() */

/* end include guard */
#endif
