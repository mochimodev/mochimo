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

static int fwrite_hashed(void *wots, const char *code, void *bal, FILE *fp)
{
   LTRAN lt;

   /* build ledger transaction */
   hash_wots_addr(lt.addr, wots);
   memcpy(lt.trancode, code, 1);
   memcpy(lt.amount, bal, 8);

   /* queue and return result of single write operation */
   return fwrite(&lt, sizeof(lt), 1, fp);
}

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
   word8 prev_addr[TXADDRLEN];
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

   /* validate block trailer against tfile trailer */
   if (read_trailer(&tft, "tfile.dat") != VEOK) goto ERROR_CLEANUP;
   if (validate_trailer(&bt, &tft) != VEOK) goto DROP_CLEANUP;

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
      if (j > 0 && memcmp(le.addr, prev_addr, TXADDRLEN) <= 0) {
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
      memcpy(prev_addr, le.addr, TXADDRLEN);
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
 * Validate a blockchain file and rename to "vblock.dat" on success. Also
 * creates ledger transaction deltas file, "ltran.dat", on success. Uses
 * node state (Mfee, Difficulty, Time0, Cblocknum, Cblockhash) to verify
 * the blockchain file against.
 * @param fname Name of blockchain file to validate: "rblock.dat"
 * @returns VEOK on success, else VERROR
*/
int b_val(char *fname)
{
   /* fork trigger blocks */
   static word32 tot_trigger[2] = { V23TRIGGER, 0 };
   static word32 v24_trigger[2] = { V24TRIGGER, 0 };
   /* Adding constants to skip validation on BoxingDay corrupt block
    * provided the blockhash matches.  See "Boxing Day Anomaly" write
    * up on the Wiki or on [ REDACTED ] for more details. */
   static word32 boxingday[2] = { 0x52d3c, 0 };
   static char boxdayhash[32] = {
      0x2f, 0xfa, 0xb9, 0xb9, 0x00, 0xe1, 0xbc, 0xa8,
      0x25, 0x19, 0x20, 0xc2, 0xdd, 0xf0, 0x46, 0xb8,
      0x07, 0x44, 0x2a, 0xbb, 0xfa, 0x5e, 0x94, 0x51,
      0xb0, 0x60, 0x03, 0xcc, 0x82, 0x2d, 0xb1, 0x12
   };

   SHA256_CTX bctx, mctx;  /* (entire) block hash and merkel array */
   LENTRY src_le;          /* source and change ledger entries */
   TXQENTRY tx;            /* Holds one transaction in the array */
   BTRAILER bt;            /* fixed length block trailer */
   BHEADER bh;             /* fixed length block header */
   TXQENTRY txs;           /* for mtx sig check */
   TXQENTRY *Q2, *qlimit;  /* tag mods */
   TXQENTRY *qp1, *qp2;    /* tag mods */
   MTX *mtx;
   FILE *fp, *ltfp;        /* input fname, output file ltran.tmp */
   word8 mroot[HASHLEN];   /* computed Merkel root */
   word8 bhash[HASHLEN];   /* computed block hash */
   word8 tx_id[HASHLEN];   /* hash of transaction and signature */
   word8 prev_tx_id[HASHLEN]; /* to check sort */
   word8 addr[TXWOTSLEN];     /* for mtx scan 4 */
   word8 pk2[WOTSSIGBYTES];   /* for WOTS */
   word8 msg[32];             /* for WOTS */
   word32 rnd2[8];            /* for WOTS */
   word32 bnum[2], stemp;
   word32 mfees[2], mreward[2];
   word32 total[2];        /* for 64-bit maths */
   clock_t ticks;          /* for function execution time */
   size_t len;             /* for malloc lengths */
   word32 hdrlen, tcount;  /* header length and transaction count */
   long blocklen;
   unsigned int j;
   int cond, count;
   char addrhash[10];

   /* init */
   ticks = clock();
   memset(mfees, 0, sizeof(mfees));

   pdebug("validating blockchain file %s...", fname);

   /* open ledger read-only */
   if (le_open("ledger.dat", "rb") != VEOK) {
      perrno("failed to le_open(ledger.dat)");
      return VERROR;
   }
   /* create ledger transaction temp file */
   ltfp = fopen("ltran.tmp", "wb");
   if (ltfp == NULL) {
      perrno("failed to fopen(ltran.tmp)");
      return VERROR;
   }
   /* open the block to validate */
   fp = fopen(fname, "rb");
   if (fp == NULL) {
      perrno("failed to fopen(%s)", fname);
      goto CLEANUP_LT;
   }
   /* read and check regular fixed size block header */
   if (fread(&hdrlen, 4, 1, fp) != 1) {
      perr("failed to fread(hdrlen)");
      goto CLEANUP_BLK;
   } else if (hdrlen != sizeof(BHEADER)) {
      perr("bad hdrlen size: %" P32u, hdrlen);
      goto CLEANUP_BLK_DROP;
   }

   /* compute block file length */
   if (fseek(fp, 0, SEEK_END) != 0) {
      perrno("failed to fseek(END)");
      goto CLEANUP_BLK;
   }
   blocklen = ftell(fp);
   if (blocklen == EOF) {
      perr("failed on ftell(fp)");
      goto CLEANUP_BLK;
   }

   /* Read block trailer:
    * Check phash, bnum,
    * difficulty, Merkel Root, nonce, solve time, and block hash.
    */
   if (fseek(fp, -(sizeof(BTRAILER)), SEEK_END) != 0) {
      perrno("failed on fseek(-BTRAILER)");
      goto CLEANUP_BLK;
   } else if (fread(&bt, sizeof(BTRAILER), 1, fp) != 1) {
      perr("failed on fread(bt)");
      goto CLEANUP_BLK;
   }
   /* check block number */
   add64(Cblocknum, One, bnum);
   if (memcmp(bnum, bt.bnum, 8) != 0) {
      pdebug("bad block number");
      goto CLEANUP_BLK_BAD;
   }
   /* check block times */
   stemp = get32(bt.stime);
   if (stemp <= Time0) {
      perr("early block time");
      goto CLEANUP_BLK_DROP;
   } else if (stemp > (time(NULL) + BCONFREQ)) {
      perr("time travel?");
      goto CLEANUP_BLK_DROP;
   } else if (cmp64(bnum, tot_trigger) > 0 && Cblocknum[0] != 0xfe) {
      if ((stemp - get32(bt.time0)) > BRIDGE) {
         perr("invalid TOT trigger");
         goto CLEANUP_BLK_DROP;
      }
   } else if (cmp64(bt.mfee, Mfee) < 0) {
      perr("bad mining fee");
      goto CLEANUP_BLK_DROP;
   } else if (get32(bt.difficulty) != Difficulty) {
      perr("difficulty mismatch");
      goto CLEANUP_BLK_DROP;
   }

   /* check previous block hash */
   if (memcmp(Cblockhash, bt.phash, HASHLEN) != 0) {
      perr("previous block hash mismatch");
      goto CLEANUP_BLK_DROP;
   }
   /* check transaction count */
   tcount = get32(bt.tcount);
   if (tcount == 0 || tcount > MAXBLTX) {
      perr("bad tcount");
      goto CLEANUP_BLK_DROP;
   }
   /* check total block length */
   if ((hdrlen + sizeof(BTRAILER) + (tcount * sizeof(TXQENTRY)))
         != (word32) blocklen) {
      perr("invalid block length");
      goto CLEANUP_BLK_DROP;
   }

   /* check enforced delay, collect haiku from block */
   if (cmp64(bnum, v24_trigger) > 0) {
      /* Boxing Day Anomaly -- Bugfix */
      if (cmp64(bt.bnum, boxingday) == 0) {
         if (memcmp(bt.bhash, boxdayhash, 32) != 0) {
            perr("bad boxingday anomaly bhash");
            goto CLEANUP_BLK_DROP;
         }
      } else if (peach_check(&bt)) {
         perr("bad Peach");
         goto CLEANUP_BLK_DROP;
      }
   } else if (trigg_check(&bt)) {
      perr("bad Trigg");
      goto CLEANUP_BLK_DROP;
   }

   /* Read block header */
   if (fseek(fp, 0, SEEK_SET) != 0) {
      perrno("failed on fseek(SET)");
      goto CLEANUP_BLK;
   } else if (fread(&bh, hdrlen, 1, fp) != 1) {
      perr("failed on fread(bh)");
      goto CLEANUP_BLK;
   }
   /* check mining reward/address */
   get_mreward(mreward, bnum);
   if (memcmp(bh.mreward, mreward, 8) != 0) {
      perr("bad mining reward");
      goto CLEANUP_BLK_DROP;
   } else if (ADDR_HAS_TAG(bh.maddr)) {
      perr("maddr has tag");
      goto CLEANUP_BLK_DROP;
   }

   /* fp left at offset of Merkel Block Array--ready to fread() */

   /* begin hashing contexts */
   sha256_init(&bctx);   /* begin entire block hash */
   sha256_update(&bctx, &bh, hdrlen);  /* ... with the header */
   if (!NEWYEAR(bt.bnum)) sha256_init(&mctx); /* begin Merkel Array hash */
   else memcpy(&mctx, &bctx, sizeof(mctx));  /* ... or copy bctx state */

   /* temp TX tag processing queue */
   Q2 = malloc((len = tcount * sizeof(TXQENTRY)));
   if (Q2 == NULL) {
      perr("failed to malloc(%zu) Q2", len);
      goto CLEANUP_BLK;
   }

   /* Validate each transaction */
   for (j = 0; j < tcount; j++) {
      if (j >= MAXBLTX) {
         perr("too many transactions");
         goto CLEANUP_TX;
      }
      if (fread(&tx, sizeof(TXQENTRY), 1, fp) != 1) {
         perr("failed on fread(TX): TX#%" P32u, j);
         goto CLEANUP_TX;
      } else if (cmp64(tx.tx_fee, Mfee) < 0) {
         perr("bad tx_fee: TX#%" P32u, j);
         goto CLEANUP_TX_DROP;
      } else if (memcmp(tx.src_addr, tx.chg_addr, TXWOTSLEN) == 0) {
         perr("(src == chg): TX#%" P32u, j);
         goto CLEANUP_TX_DROP;
      } else if (!TX_IS_MTX(&tx)) {
         if (memcmp(tx.src_addr, tx.dst_addr, TXWOTSLEN) == 0) {
            perr("(src == dst): TX#%" P32u, j);
            goto CLEANUP_TX_DROP;
         }
      }

      /* running block/merkel hash */
      sha256_update(&bctx, &tx, sizeof(TXQENTRY));
      sha256_update(&mctx, &tx, sizeof(TXQENTRY));
      /* tx_id is hash of tx.src_addr */
      sha256(tx.src_addr, TXWOTSLEN, tx_id);
      if (memcmp(tx_id, tx.tx_id, HASHLEN) != 0) {
         perr("bad tx_id: TX#%" P32u, j);
         goto CLEANUP_TX_DROP;
      }

      /* Check that tx_id is sorted. */
      if (j != 0) {
         cond = memcmp(tx_id, prev_tx_id, HASHLEN);
         if (cond <= 0) {
            if (cond == 0) {
               perr("duplicate tx_id: TX#%" P32u, j);
               goto CLEANUP_TX_DROP;
            } else {
               perr("unsorted tx_id: TX#%" P32u, j);
               goto CLEANUP_TX_DROP;
            }
         }
      }
      /* remember this tx_id for next time */
      memcpy(prev_tx_id, tx_id, HASHLEN);

      /* check WTOS signature */
      if (TX_IS_MTX(&tx) && get32(Cblocknum) >= MTXTRIGGER) {
         memcpy(&txs, &tx, sizeof(txs));
         mtx = (MTX *) &txs;
         /* mtx->zeros is always signed when zero */
         memset(mtx->zeros, 0, MDST_NUM_DZEROS);
         sha256(txs.src_addr, TXSIG_INLEN, msg);
      } else sha256(tx.src_addr, TXSIG_INLEN, msg);
      memcpy(rnd2, &tx.src_addr[TXSIGLEN + 32], 32);  /* copy WOTS addr[] */
      wots_pk_from_sig(pk2, tx.tx_sig, msg, &tx.src_addr[TXSIGLEN], rnd2);
      if (memcmp(pk2, tx.src_addr, TXSIGLEN) != 0) {
         perr("WOTS signature failed: TX#%" P32u, j);
         goto CLEANUP_TX_DROP;
      }

      /* look up source address in ledger */
      if (le_find(tx.src_addr, &src_le, TXWOTSLEN) == 0) {
         hash2hex32(tx.src_addr, addrhash);
         pdebug("error address %s...", addrhash);
         perr("src_addr not in ledger: TX#%" P32u, j);
         goto CLEANUP_TX_DROP;
      }

      total[0] = total[1] = 0;
      /* use add64() to check for carry out, total, and fees */
      cond =  add64(tx.send_total, tx.change_total, total);
      cond += add64(tx.tx_fee, total, total);
      if (cond) {
         perr("total overflow: TX#%" P32u, j);
         goto CLEANUP_TX_DROP;
      } else if (cmp64(src_le.balance, total) != 0) {
         perr("bad transaction total: TX#%" P32u, j);
         goto CLEANUP_TX_DROP;
      } else if (add64(mfees, tx.tx_fee, mfees)) {
         perr("mfees overflow: TX#%" P32u, j);
         goto CLEANUP_TX;
      }
      /* check mtx/tag_valid */
      if (!TX_IS_MTX(&tx)) {
         if(tag_valid(tx.src_addr, tx.chg_addr, tx.dst_addr, bt.bnum)) {
            perr("tag not valid: TX#%" P32u, j);
            goto CLEANUP_TX_DROP;
         }
      } else if(mtx_val((MTX *) &tx, Mfee)) {
         perr("bad mtx_val: TX#%" P32u, j);
         goto CLEANUP_TX_DROP;
      }

      /* copy TX to tag queue */
      memcpy(&Q2[j], &tx, sizeof(TXQENTRY));
   }  /* end for j */

   /* finalize Merkel Root - phash, bnum, mfee, tcount, time0, difficulty */
   if (NEWYEAR(bt.bnum)) sha256_update(&mctx, &bt, (HASHLEN+8+8+4+4+4));
   sha256_final(&mctx, mroot);
   if (memcmp(bt.mroot, mroot, HASHLEN) != 0) {
      perr("bad merkel root");
      goto CLEANUP_TX_DROP;
   }

   /* finalize block hash - Block trailer (- block hash) */
   sha256_update(&bctx, &bt, sizeof(BTRAILER) - HASHLEN);
   sha256_final(&bctx, bhash);
   if (memcmp(bt.bhash, bhash, HASHLEN) != 0) {
      perr("bad block hash");
      goto CLEANUP_TX_DROP;
   }

   /* When spending from a tag, the address associated with said tag will
    * change. Any transactions to dst_tags that are also spent from as
    * src_tags within the same block, MUST have their dst_addr replaced
    * with the chg_addr of said src_tag transactions. */

   /* tag search  Begin ... */
   qlimit = &Q2[tcount];
   for(qp1 = Q2; qp1 < qlimit; qp1++) {
      if(!ADDR_HAS_TAG(qp1->src_addr)
         || memcmp(ADDR_TAG_PTR(qp1->src_addr), ADDR_TAG_PTR(qp1->chg_addr),
                   TXTAGLEN) != 0) continue;
      /* Step 2: Start another big-O n squared, nested loop here... */
      for(qp2 = Q2; qp2 < qlimit; qp2++) {
         if(qp1 == qp2) continue;  /* added -trg */
         if(TX_IS_MTX(qp2)) continue;  /* skip multi-dst's for now */
         /* if src1 == dst2, then copy chg1 to dst2 -- 32-bit for DSL -trg */
         if(   *((word32 *) ADDR_TAG_PTR(qp1->src_addr))
            == *((word32 *) ADDR_TAG_PTR(qp2->dst_addr))
            && *((word32 *) (ADDR_TAG_PTR(qp1->src_addr) + 4))
            == *((word32 *) (ADDR_TAG_PTR(qp2->dst_addr) + 4))
            && *((word32 *) (ADDR_TAG_PTR(qp1->src_addr) + 8))
            == *((word32 *) (ADDR_TAG_PTR(qp2->dst_addr) + 8)))
                   memcpy(qp2->dst_addr, qp1->chg_addr, TXWOTSLEN);
      }  /* end for qp2 */
   }  /* end for qp1 */

   /*
    * Three times is the charm...
    */
   for(j = 0, qp1 = Q2; qp1 < qlimit; qp1++, j++) {
      /* Re-do all the maths again... */
      total[0] = total[1] = 0;
      cond =  add64(qp1->send_total, qp1->change_total, total);
      cond += add64(qp1->tx_fee, total, total);
      if (cond) {
         perr("scan3 total overflow");
         goto CLEANUP_TX;
      }

      /* Write ledger transactions to ltran.tmp for all src and chg,
       * but only non-mtx dst
       * that will have to be sorted, read again, and applied by bup...
       */
      fwrite_hashed(qp1->src_addr, "-", total, ltfp);  /* debit src addr */
      /* add to or create non-multi dst address */
      if(!TX_IS_MTX(qp1) && !iszero(qp1->send_total, 8)) {
         fwrite_hashed(qp1->dst_addr, "A", qp1->send_total, ltfp);
      }
      /* add to or create change address */
      if(!iszero(qp1->change_total, 8)) {
         fwrite_hashed(qp1->chg_addr, "A", qp1->change_total, ltfp);
      }
   }  /* end for j -- scan 3 */
   if(j != tcount) {
      perr("scan3 tcount mismatch: %" P32u, j);
      goto CLEANUP_TX;
   }

   /* mtx tag search  Begin scan 4 ...
    *
    * Write out the multi-dst trans using tag scan logic @
    * that more or less repeats the above big-O n-squared loops, and
    * expands the tags, and copies addresses around.
    */
   for(qp1 = Q2; qp1 < qlimit; qp1++) {
      if(!TX_IS_MTX(qp1)) continue;  /* only multi-dst's this time */
      mtx = (MTX *) qp1;  /* poor man's union */
      /* For each dst[] tag... */
      for(j = 0; j < MDST_NUM_DST; j++) {
         if(iszero(mtx->dst[j].tag, TXTAGLEN)) break; /* end of dst[] */
         memcpy(ADDR_TAG_PTR(addr), mtx->dst[j].tag, TXTAGLEN);
         /* If dst[j] tag not found, write money back to chg addr. */
         if(tag_find(addr, addr, NULL, TXTAGLEN) != VEOK) {
            count = fwrite_hashed(mtx->chg_addr, "A", mtx->dst[j].amount, ltfp);
            if (count == 1) continue;  /* next dst[j] */
            perr("bad I/O dst-->chg write");
            goto CLEANUP_TX;
         }
         /* Start another big-O n-squared, nested loop here... scan 5 */
         for(qp2 = Q2; qp2 < qlimit; qp2++) {
            if(qp1 == qp2) continue;
            /* if dst[j] tag == any other src addr tag and chg addr tag,
             * copy other chg addr to dst[] addr.
             */
            if(!ADDR_HAS_TAG(qp2->src_addr)) continue;
            if(memcmp(ADDR_TAG_PTR(qp2->src_addr),
                      ADDR_TAG_PTR(qp2->chg_addr), TXTAGLEN) != 0)
                         continue;
            if(memcmp(ADDR_TAG_PTR(qp2->src_addr), ADDR_TAG_PTR(addr),
                      TXTAGLEN) == 0) {
                         memcpy(addr, qp2->chg_addr, TXWOTSLEN);
                         break;
            }
         }  /* end for qp2 scan 5 */
         /* write out the dst transaction */
         count = fwrite_hashed(addr, "A", mtx->dst[j].amount, ltfp);
         if (count != 1) {
            perr("bad I/O scan 4");
            goto CLEANUP_TX;
         }
      }  /* end for j */
   }  /* end for qp1 */
   /* end mtx scan 4 */

   /* Create a transaction amount = mreward + mfees
    * address = bh.maddr
    */
   if (add64(mfees, mreward, mfees)) {
      perr("mfees overflow");
      goto CLEANUP_TX;
   }
   /* Make ledger tran to add to or create mining address.
    * '...Money from nothing...'
    */
   count = fwrite_hashed(bh.maddr, "A", mfees, ltfp);
   if (count != 1 || ferror(ltfp)) {
      perr("ltfp IO error");
      goto CLEANUP_TX;
   } else {
      hash2hex32(bh.maddr, addrhash);
      pdebug("wrote reward (%08x%08x) to %s...",
         mreward[1], mreward[0], addrhash);
   }

   /* cleanup */
   free(Q2);
   fclose(fp);
   fclose(ltfp);
   /* promote ltran.tmp to *.dat file and rename blockchain file */
   remove("ltran.dat");
   if (rename("ltran.tmp", "ltran.dat") != 0) {
      perr("failed to move ltran.tmp to ltran.dat");
      remove("ltran.tmp");
      return VERROR;
   }

   pdebug("%s validated in %gs", fname, diffclocktime(ticks));

   /* success */
   return VEOK;

   /* failure / error handling */
CLEANUP_TX_DROP:
   free(Q2);
CLEANUP_BLK_DROP:
   fclose(fp);
   fclose(ltfp);
   remove("ltran.tmp");
   /* epoch pinklist */
   return VEBAD2;
CLEANUP_BLK_BAD:
   fclose(fp);
   fclose(ltfp);
   remove("ltran.tmp");
   /* current pinklist */
   return VEBAD;
CLEANUP_TX:
   free(Q2);
CLEANUP_BLK:
   fclose(fp);
CLEANUP_LT:
   fclose(ltfp);
   remove("ltran.tmp");
   return VERROR;
}  /* end b_val() */

/* end include guard */
#endif
