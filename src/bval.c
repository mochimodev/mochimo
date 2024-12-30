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
   TXENTRY txe;            /* holds one transaction entry from block */
   BTRAILER tft;           /* fixed length block trailer (tfile) */
   BTRAILER bt;            /* fixed length block trailer */
   BHEADER bh;             /* fixed length block header */
   LTRAN lt;               /* ledger transaction */
   long long len, min;
   word8 *mtree;
   FILE *fp, *ltfp;        /* input fname, output file ltran.tmp */
   word8 prev_src_addr[ADDR_LEN];   /* source address sort check */
   word8 mroot[HASHLEN];   /* computed Merkel root */
   word8 mreward[8];
   word32 mdstlen, tcount; /* multi-destination and transaction count */
   word32 j, k;            /* loop counters */
   int ecode, overflow;

   /* init NULL for error handling */
   fp = ltfp = NULL;
   mtree = NULL;

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
   min = sizeof(BHEADER) + TXLEN_MIN + sizeof(BTRAILER);
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
   sha256(bh.maddr /* + bh.mreward */, sizeof(bh.maddr) + 8, mtree);

   /* open ltran file for writing */
   ltfp = fopen(ltfile, "wb");
   if (ltfp == NULL) goto ERROR_CLEANUP;

   /* Validate each transaction */
   for (j = 0; j < tcount; j++) {
      /* read transaction data for validation */
      if (tx_fread(&txe, fp) != VEOK) goto RDERR_CLEANUP;

      /* ... TRANSACTION PROCESSING ... */

      /* skip first src_addr check */
      if (j > 0) {
         /* check src_addr is sorted, NO DUPLICATES */
         if (memcmp(txe.src_addr, prev_src_addr, ADDR_LEN) <= 0) {
            set_errno(EMCM_TXSORT);
            goto DROP_CLEANUP;
         }
      }
      /* validate transaction */
      ecode = txe_val(&txe, bt.bnum, bt.mfee);
      if (ecode != VEOK) goto CLEANUP;

      /* add transaction id to merkel tree, store src_addr */
      memcpy(&mtree[(j + 1) * HASHLEN], txe.tx_id, HASHLEN);
      memcpy(prev_src_addr, txe.src_addr, ADDR_LEN);

      /* sum fees for (additional) miner credit */
      if (add64(mreward, txe.tx_fee, mreward)) {
         set_errno(EMCM_MFEES_OVERFLOW);
         goto ERROR_CLEANUP;
      }

      /* ... LTRAN PROCESSING ... */

      /* ltran DEBIT source address */
      memcpy(lt.addr, txe.src_addr, ADDR_LEN);
      lt.trancode[0] = '-';
      overflow  = add64(txe.send_total, txe.change_total, lt.amount);
      overflow += add64(lt.amount, txe.tx_fee, lt.amount);
      if (overflow) {
         set_errno(EMCM_TXOVERFLOW);
         goto DROP_CLEANUP;
      }
      if (fwrite(&lt, sizeof(LTRAN), 1, ltfp) != 1) {
         goto ERROR_CLEANUP;
      }

      /* ltran REHASH src_tag->change address */
      memcpy(lt.addr, txe.chg_addr, ADDR_LEN);
      lt.trancode[0] = 'H';
      put64(lt.amount, txe.change_total);
      if (fwrite(&lt, sizeof(LTRAN), 1, ltfp) != 1) {
         goto ERROR_CLEANUP;
      }

      /* process transaction per transaction type */
      switch (TXDAT_TYPE(&txe)) {
         case TXDAT_MDST:
            /* ltran credit every destination address */
            mdstlen = MDST_COUNT(&txe);
            for (k = 0; k < mdstlen; k++) {
               /* copy tag and zero address "hash" */
               memcpy(ADDR_TAG_PTR(lt.addr), txe.mdst[k].tag, ADDR_TAG_LEN);
               memset(ADDR_HASH_PTR(lt.addr), 0, ADDR_HASH_LEN);
               lt.trancode[0] = 'A';
               put64(lt.amount, txe.mdst[k].amount);
               if (fwrite(&lt, sizeof(LTRAN), 1, ltfp) != 1) {
                  goto ERROR_CLEANUP;
               }
            }
            break;
         default:
            set_errno(EMCM_XTXUNDEF);
            goto DROP_CLEANUP;
      }  /* end switch TX_TYPE */
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

   /* copy "mining" tag and zero address "hash" */
   memcpy(ADDR_TAG_PTR(lt.addr), bh.maddr, ADDR_TAG_LEN);
   memset(ADDR_HASH_PTR(lt.addr), 0, ADDR_HASH_LEN);
   lt.trancode[0] = 'A';
   put64(lt.amount, mreward);
   if (fwrite(&lt, sizeof(LTRAN), 1, ltfp) != 1) {
      goto ERROR_CLEANUP;
   }

   /* cleanup */
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
