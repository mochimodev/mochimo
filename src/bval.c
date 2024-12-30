/**
 * @private
 * @headerfile bval.h <bval.h>
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_BVAL_C
#define MOCHIMO_BVAL_C


#include "bval.h"

/* internal support */
#include "wots.h"
#include "tx.h"
#include "trigg.h"
#include "tfile.h"
#include "tag.h"
#include "peach.h"
#include "ledger.h"
#include "global.h"
#include "error.h"

/* external support */
#include <string.h>
#include <stdlib.h>
#include "sha256.h"
#include "extmath.h"

/**
 * Validate a pseudo-block against current node state. Uses node state
 * (Cblocknum, Cblockhash, Difficulty, Time0).
 * @param pfile Filename of pseudo-block to validate
 * @return (int) value representing operation result
 * @retval VEBAD2 on malicious block; check errno for details
 * @retval VEBAD on block format violation; check errno for details
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int p_val(const char *pfile)
{
   BTRAILER bt, tft;
   FILE *fp;
   long long len;
   word32 hdrlen;

   /* open the pseudo-block to validate */
   fp = fopen(pfile, "rb");
   if (fp == NULL) return VERROR;
   /* read pseudo-block data and jumpt to EOF for file length */
   if (fread(&hdrlen, 4, 1, fp) != 1) goto RDERR_CLEANUP;
   if (hdrlen != 4) {
      set_errno(EMCM_HDRLEN);
      goto ERROR_CLEANUP;
   }

   /* fseek to check pseudo-block file length */
   if (fseek64(fp, 0LL, SEEK_END) != 0) goto ERROR_CLEANUP;
   len = ftell(fp);
   if (len == (-1)) goto ERROR_CLEANUP;
   if (len != sizeof(BTRAILER) + 4) {
      set_errno(EMCM_FILELEN);
      goto ERROR_CLEANUP;
   }

   /* read trailer */
   if (fseek64(fp, -(sizeof(BTRAILER)), SEEK_END) != 0) goto ERROR_CLEANUP;
   if (fread(&bt, sizeof(BTRAILER), 1, fp) != 1) goto RDERR_CLEANUP;
   /* cleanup (early) */
   fclose(fp);

   /* validate block trailer against tfile trailer */
   if (read_trailer(&tft, "tfile.dat") != VEOK) return VERROR;
   if (validate_trailer(&bt, &tft) != VEOK) return VEBAD2;

   /* tcount cannot reliably be validated by (the current routines of)
    * validate_trailer(), so we must ENSURE the validity of tcount here
    */
   if (get32(bt.tcount) != 0) {
      set_errno(EMCM_TCOUNT);
      return VEBAD2;
   }

   /* success */
   return VEOK;

   /* cleanup / error handling */
RDERR_CLEANUP:
   if (!ferror(fp)) {
      set_errno(EMCM_EOF);
   }
ERROR_CLEANUP:
   fclose(fp);

   return VERROR;
}  /* end p_val() */

/**
 * Validate a neogenesis-block containing a hash-based ledger.
 * Checks ledger entries are in ascending sort.
 * Checks block hash matches calculated hash.
 * Checks block size matches neogenesis format.
 * Checks block trailer matches Tfile entry.
 * Checks sum of amounts do not exceed "expected" rewards.
 * NOTE: Tfile should have been verified before neogenesis validation.
 * @param ngfile Filename of neogenesis block to validate
 * @param bnum Pointer to expected block number, or NULL to ignore
 * @return (int) value representing operation result
 * @retval VEBAD2 on malicious block; check errno for details
 * @retval VEBAD on block format violation; check errno for details
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int ng_val(const char *ngfile, const word8 bnum[8])
{
   LENTRY le;
   NGHEADER ngh;
   BTRAILER bt, tft;
   long long len;
   word64 lbytes;
   size_t j, lcount;
   word8 prev_addr[ADDR_LEN];
   word8 mroot[HASHLEN];
   word8 amounts[8];
   word8 rewards[8];
   word8 *mtree;
   FILE *fp;
   int ecode;

   /* init */
   mtree = NULL;

   /* open file for validation */
   fp = fopen(ngfile, "rb");
   if (fp == NULL) return VERROR;
   /* read block trailer (fp left at EOF) */
   if (fseek64(fp, -(sizeof(BTRAILER)), SEEK_END) != 0) goto ERROR_CLEANUP;
   if (fread(&bt, sizeof(BTRAILER), 1, fp) != 1) goto RDERR_CLEANUP;
   /* read EOF file offset as file length */
   len = ftell64(fp);
   if (len == (-1)) goto ERROR_CLEANUP;
   /* read and check neogenesis header data */
   if (fseek64(fp, 0LL, SEEK_SET) != 0) goto ERROR_CLEANUP;
   if (fread(&ngh, sizeof(NGHEADER), 1, fp) != 1) goto RDERR_CLEANUP;
   if (get32(ngh.hdrlen) != sizeof(NGHEADER)) {
      set_errno(EMCM_HDRLEN);
      goto DROP_CLEANUP;
   }
   put64(&lbytes, ngh.lbytes);
   if (lbytes < sizeof(LENTRY) || (lbytes % sizeof(LENTRY)) != 0) {
      set_errno(EMCM_FILEDATA);
      goto DROP_CLEANUP;
   }
   /* check file length against header data */
   if (len != (long long) (sizeof(NGHEADER) + lbytes + sizeof(BTRAILER))) {
      set_errno(EMCM_FILELEN);
      goto DROP_CLEANUP;
   }

   /* ... fp is left at beginning of ledger entries ... */

   /* compare block trailer against tfile trailer */
   if (read_tfile(&tft, bt.bnum, 1, "tfile.dat") != 1) {
      if (errno != EMCM_EOF) goto ERROR_CLEANUP;
      /* ... we don't have the trailer, validate against last trailer */
      if (read_trailer(&tft, "tfile.dat") != VEOK) goto ERROR_CLEANUP;
      if (validate_trailer(&bt, &tft) != VEOK) goto DROP_CLEANUP;
      /* ... else compare tfile trailer against block trailer */
   } else if (memcmp(&tft, &bt, sizeof(BTRAILER)) != 0) {
      set_errno(EMCM_TRAILER);
      goto DROP_CLEANUP;
   }

   /* tcount cannot reliably be validated by (the current routines of)
    * validate_trailer(), so we must ENSURE the validity of tcount here
    */
   if (get32(bt.tcount) != 0) {
      set_errno(EMCM_TCOUNT);
      goto DROP_CLEANUP;
   }

   /* additional bnum validation from calling parent...
    * probably should be done in calling parent, but until routines
    * focus on deduplication of file freads, probably best to stay here
    */
   if (bnum) {
      if (cmp64(bt.bnum, bnum) != 0) {
         set_errno(EMCM_BNUM);
         goto DROP_CLEANUP;
      }
   }

   /* malloc merkle tree */
   lcount = lbytes / sizeof(LENTRY);
   mtree = malloc(lcount * HASHLEN);
   if (mtree == NULL) goto ERROR_CLEANUP;

   /* init amounts before summing */
   memset(amounts, 0, 8);

   /* read neogenesis ledger data... */
   for (j = 0; j < lcount; j++) {
      if (fread(&le, sizeof(LENTRY), 1, fp) != 1) goto RDERR_CLEANUP;
      /* check ledger sort -- skip on first read */
      if (j > 0 && memcmp(le.addr, prev_addr, ADDR_LEN) <= 0) {
         set_errno(EMCM_LESORT);
         goto DROP_CLEANUP;
      }
      /* update amounts sum, ensure no overflow */
      if (add64(amounts, le.balance, amounts)) {
         set_errno(EMCM_MATH64_OVERFLOW);
         goto DROP_CLEANUP;
      }
      /* hash ledger entry directly into merkel tree -- store prev addr */
      sha256(&le, sizeof(LENTRY), mtree + (j * HASHLEN));
      memcpy(prev_addr, le.addr, ADDR_LEN);
   }

   /* compute and validate Merkel Root */
   merkle_root(mtree, lcount, mroot);
   if (memcmp(bt.mroot, mroot, HASHLEN) != 0) {
      set_errno(EMCM_MROOT);
      goto DROP_CLEANUP;
   }

   /* cleanup */
   free(mtree);
   fclose(fp);

   /* check accurate sum of Tfile rewards against ledger amounts */
   if (get_tfrewards("tfile.dat", rewards, bt.bnum) != VEOK) return VERROR;
   /* ... get_tfile_rewards() cannot calculate supply burn where a
    * balance is less than the transaction fee, so we only check the
    * Neogenesis amounts do not exceed our expected rewards...
    */

   /* check calculated rewards against ledger amounts */
   if (cmp64(amounts, rewards) > 0) {
      set_errno(EMCM_LESUM);
      return VEBAD2;
   }

   return VEOK;

   /* cleanup / error handling */
RDERR_CLEANUP:
   if (!ferror(fp)) {
      set_errno(EMCM_EOF);
   }
ERROR_CLEANUP:
   ecode = VERROR;
   goto CLEANUP;
DROP_CLEANUP:
   ecode = VEBAD2;
CLEANUP:
   if (mtree) free(mtree);
   fclose(fp);

   return ecode;
}  /* end ng_val() */

/**
 * Validate a transaction block file and create ledger transaction file.
 * @param bcfile Filename of block file to validate
 * @param ltfile Filename of ledger transactions file to write
 * @return (int) value representing operation result
 * @retval VEBAD2 on malicious block; check errno for details
 * @retval VEBAD on invalid block; check errno for details
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 */
int b_val(const char *bcfile, const char *ltfile)
{
   TXQENTRY tx;            /* Holds one transaction in the array */
   XDATA xdata;
   BTRAILER tft;           /* fixed length block trailer (tfile) */
   BTRAILER bt;            /* fixed length block trailer */
   BHEADER bh;             /* fixed length block header */
   long long len, min;
   size_t ltcount, ltidx, refidx;
   void *ptr;
   LTRAN *ltran;
   word8 *mtree;
   word8 *raddr;           /* reference address list of spent tags */
   FILE *fp, *ltfp;        /* input fname, output file ltran.tmp */
   word8 mroot[HASHLEN];   /* computed Merkel root */
   word8 addr[TXADDRLEN];  /* for tag_find() */
   word8 mreward[8];
   word32 mdstlen, tcount; /* multi-destination and transaction count */
   word32 j, k;            /* loop counters */
   int ecode;

   /* init NULL for error handling */
   mtree = raddr = NULL;
   fp = ltfp = NULL;
   ltran = NULL;

   /* open block file and extract metadata */
   fp = fopen(bcfile, "rb");
   if (fp == NULL) goto ERROR_CLEANUP;
   /* read block trailer (fp left at EOF) */
   if (fseek64(fp, -(sizeof(BTRAILER)), SEEK_END) != 0) return VERROR;
   if (fread(&bt, sizeof(BTRAILER), 1, fp) != 1) goto RDERR_CLEANUP;
   /* read EOF file offset as file length */
   len = ftell64(fp);
   if (len == (-1)) return VERROR;
   /* ensure file contains the minimum amount of data */
   min = sizeof(BHEADER) + sizeof(TXQENTRY) + sizeof(BTRAILER);
   if (len < min) {
      set_errno(EMCM_FILEDATA);
      goto ERROR_CLEANUP;
   }
   /* read and check regular fixed size block header */
   if (fseek64(fp, 0LL, SEEK_SET) != 0) return VERROR;
   if (fread(&bh, sizeof(BHEADER), 1, fp) != 1) goto RDERR_CLEANUP;
   if (get32(bh.hdrlen) != sizeof(BHEADER)) {
      set_errno(EMCM_HDRLEN);
      goto DROP_CLEANUP;
   }

   /* ... fp is left at beginning of transactions ... */

   /* check mining reward/address */
   get_mreward(mreward, bt.bnum);
   if (memcmp(bh.mreward, mreward, 8) != 0) {
      set_errno(EMCM_MREWARD);
      goto DROP_CLEANUP;
   }
   if (ADDR_HAS_TAG(bh.maddr)) {
      set_errno(EMCM_MADDR);
      goto DROP_CLEANUP;
   }

   /* validate block trailer (incl. PoW) against tfile trailer */
   if (read_trailer(&tft, "tfile.dat") != VEOK) goto ERROR_CLEANUP;
   if (validate_trailer(&bt, &tft) != VEOK) goto DROP_CLEANUP;
   if (validate_pow(&bt) != VEOK) goto DROP_CLEANUP;

   /* tcount cannot reliably be validated by (the current routines of)
    * validate_trailer(), so we must ENSURE the validity of tcount here
    */
   tcount = get32(bt.tcount);
   if (tcount == 0 || tcount > MAXBLTX) {
      set_errno(EMCM_TCOUNT);
      goto DROP_CLEANUP;
   }

   /* malloc merkle tree (+1 for miner) */
   mtree = malloc((tcount + 1) * HASHLEN);
   if (mtree == NULL) goto ERROR_CLEANUP;

   /* begin merkel hash with mining address + reward */
   sha256(bh.maddr /* + bh.mreward */, TXADDRLEN + 8, mtree);

   /* Any transactions to dst_tag's, where the dst_tag also exists as
    * src_tag in another transaction within the same block, MUST have
    * the dst_addr replaced with the chg_addr of the transaction from
    * which the dst_tag was spent as a src_tag. This only applies to
    * tagged addreses; untagged (mining) addresses are unaffected.
    *
    * Example (not dependant on order):
    * TX#1: src(A) -> dst(B) -> chg(C);
    * TX#2: src(D) -> dst(A) -> chg(E);
    * ... TX#2's dst(A) needs to be changed to TX#1's chg(C);
    *
    * To address this issue without introducing multiple O(n^2) loops,
    * we store the chg_addr of afflicted transactions in a list of
    * reference addresses (*raddr). This list is then sorted by tag
    * for binary searching, and used to modify destinations during
    * the final (write) pass of ledger transaction creation.
    *
    * Addresses are place in the reference list on the following criterea:
    * (src_addr) is a tagged address  AND  (src_tag) is equal to (chg_tag)
    */

   /* ltran space is initially estimated (+1 for miner) */
   ltcount = tcount * 3 + 1;
   /* malloc space for ltran list (estimated) */
   ltran = malloc(sizeof(LTRAN) * ltcount);
   if (ltran == NULL) goto ERROR_CLEANUP;
   /* malloc space for reference list (assumes worst case) */
   raddr = malloc(TXADDRLEN * tcount);
   if (raddr == NULL) goto ERROR_CLEANUP;
   /* zero indexes */
   ltidx = refidx = 0;

   /* Validate each transaction */
   for (j = 0; j < tcount; j++) {
      /* read transaction data for validation */
      if (tx_fread(&tx, &xdata, fp) != VEOK) goto RDERR_CLEANUP;

      /* ... TRANSACTION PROCESSING ... */

      /* check tx_id is sorted (skip first) -- mtree holds prev tx_id */
      if (j > 0 && memcmp(tx.tx_id, mtree + (j * HASHLEN), HASHLEN) <= 0) {
         set_errno(EMCM_TXSORT);
         goto DROP_CLEANUP;
      }
      /* validate transaction */
      ecode = txqe_val(&tx, &xdata, bt.bnum);
      if (ecode != VEOK) goto CLEANUP;

      /* add transaction id to merkel tree (infer previous tx_id) */
      memcpy(&mtree[(j + 1) * HASHLEN], tx.tx_id, HASHLEN);

      /* ... LTRAN PROCESSING ... */

      /* save afflicted change address to reference list */
      if (ADDR_HAS_TAG(tx.src_addr) &&
            addr_tag_equal(tx.src_addr, tx.chg_addr)) {
         memcpy(raddr + (refidx * TXADDRLEN), tx.chg_addr, TXADDRLEN);
         refidx++;
      }
      /* ltran debit source address */
      memcpy(ltran[ltidx].addr, tx.src_addr, TXADDRLEN);
      ltran[ltidx].trancode[0] = '-';
      add64(tx.send_total, tx.change_total, ltran[ltidx].amount);
      add64(tx.tx_fee, ltran[ltidx].amount, ltran[ltidx].amount);
      ltidx++;
      /* check for eXtended TX type transaction */
      if (!IS_XTX(&tx)) {
         /* ltran credit destination address */
         memcpy(ltran[ltidx].addr, tx.dst_addr, TXADDRLEN);
         ltran[ltidx].trancode[0] = 'D';
         /* ... 'D' is changed to 'A' in the final (write) pass */
         put64(ltran[ltidx].amount, tx.send_total);
         ltidx++;
      } else if (XTX_TYPE(&tx) == XTX_MDST) {
         /* handle Multi-destination Transaction */
         mdstlen = XTX_COUNT(&tx) + 1;
         if (mdstlen > 1) {
            /* realloc space to account for additional destinations */
            ltcount += mdstlen - 1;
            ptr = realloc(ltran, sizeof(LTRAN) * ltcount);
            if (ptr == NULL) goto ERROR_CLEANUP;
            ltran = ptr;
         }
         /* iterate multi-destinations */
         for (k = 0; k < mdstlen; k++) {
            /* find address associated with tag */
            memcpy(ADDR_TAG_PTR(addr), xdata.mdst[k].tag, TXTAGLEN);
            if (tag_find(addr, addr, NULL, TXTAGLEN) != VEOK) {
               set_errno(EMCM_LETAG);
               goto ERROR_CLEANUP;
            }
            /* ltran credit multi-destination address */
            memcpy(ltran[ltidx].addr, addr, TXADDRLEN);
            ltran[ltidx].trancode[0] = 'D';
            /* ... 'D' is changed to 'A' in the final (write) pass */
            put64(ltran[ltidx].amount, xdata.mdst[k].amount);
            ltidx++;
         }
      }  /* end else if (XTX_TYPE(&tx) == XTX_MDST) */
      /* ltran credit change address */
      memcpy(ltran[ltidx].addr, tx.chg_addr, TXADDRLEN);
      ltran[ltidx].trancode[0] = 'A';
      put64(ltran[ltidx].amount, tx.change_total);
      ltidx++;
      /* additionally, sum fees for miner credit */
      if (add64(mreward, tx.tx_fee, mreward)) {
         set_errno(EMCM_MFEES_OVERFLOW);
         goto ERROR_CLEANUP;
      }
   }  /* end for j */

   /* transactions are variable length, so to check file length we must
    * check the current offset (offset at which transactions end), plus
    * the size of a block trailer is equal to the EOF offset
    */
   len = ftell64(fp);
   if (len == (-1)) goto ERROR_CLEANUP;
   if (fseek64(fp, 0LL, SEEK_END) != 0) goto ERROR_CLEANUP;
   if (ftell64(fp) != len + (long long) sizeof(BTRAILER)) {
      set_errno(EMCM_FILELEN);
      goto DROP_CLEANUP;
   }

   /* compute and validate Merkel Root (+1 for miner) */
   merkle_root(mtree, tcount + 1, mroot);
   if (memcmp(bt.mroot, mroot, HASHLEN) != 0) {
      set_errno(EMCM_MROOT);
      goto DROP_CLEANUP;
   }

   /* Make ledger tran to add to or create mining address.
    * '...Money from nothing...'
    */
   memcpy(ltran[ltidx].addr, bh.maddr, TXADDRLEN);
   ltran[ltidx].trancode[0] = 'A';
   put64(ltran[ltidx].amount, mreward);
   ltidx++;

   /* sort reference address list by tag */
   qsort(raddr, refidx, TXADDRLEN, addr_tag_compare);

   /* open ltran file (if provided) */
   ltfp = fopen(ltfile, "wb");
   if (ltfp == NULL) goto ERROR_CLEANUP;
   /* write ltrans to file, check zero value credits and references */
   for (j = 0; j < ltidx; j++) {
      /* skip zero value credits */
      if (ltran[j].trancode[0] == 'A' || ltran[j].trancode[0] == 'D') {
         if (iszero(ltran[j].amount, 8)) continue;
      }
      /* raddrect destinations of spent tags */
      if (ltran[j].trancode[0] == 'D') {
         ltran[j].trancode[0] = 'A';
         /* replace with reference address if tag matches */
         ptr = bsearch(ltran[j].addr, raddr, refidx, TXADDRLEN, addr_tag_compare);
         if (ptr) memcpy(ltran[j].addr, ptr, TXADDRLEN);
      }
      /* write ltran entry -- sorted by le_update() */
      if (fwrite(&ltran[j], sizeof(LTRAN), 1, ltfp) != 1) goto ERROR_CLEANUP;
   }

   /* cleanup */
   free(ltran);
   free(raddr);
   free(mtree);
   fclose(fp);
   fclose(ltfp);

   /* success */
   return VEOK;

   /* cleanup / error handling */
RDERR_CLEANUP:
   if (!ferror(fp)) {
      set_errno(EMCM_EOF);
   }
ERROR_CLEANUP:
   ecode = VERROR;
   goto CLEANUP;
DROP_CLEANUP:
   ecode = VEBAD2;
CLEANUP:
   if (raddr) free(raddr);
   if (ltran) free(ltran);
   if (mtree) free(mtree);
   if (fp) fclose(fp);
   if (ltfp) {
      fclose(ltfp);
      remove(ltfile);
   }

   return ecode;
}  /* end b_val() */

/* end include guard */
#endif
