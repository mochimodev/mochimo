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
#include "util.h"
#include "tx.h"
#include "trigg.h"
#include "tfile.h"
#include "tag.h"
#include "peach.h"
#include "ledger.h"
#include "global.h"

/* external support */
#include <string.h>
#include <stdlib.h>
#include "sha256.h"
#include "extprint.h"
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
 * @returns VEOK on success, else error code
*/
int p_val(char *fname)
{
   BTRAILER bt;
   SHA256_CTX ctx;
   FILE *fp;
   clock_t ticks;
   long plen;
   word32 hdrlen, bnum[2];
   word8 hash[HASHLEN];
   int ecode;

   /* init */
   ticks = clock();

   pdebug("p_val(): validating pseudo-block...");

   /* open the pseudo-block to validate */
   fp = fopen(fname, "rb");
   if (fp == NULL) mErrno(FAIL, "p_val(): failed to fopen(%s)", fname);
   /* read and check regular fixed size block header */
   if (fread(&hdrlen, 4, 1, fp) != 1) {
      mError(FAIL_IO, "p_val(): failed to fread(hdrlen)");
   } else if (hdrlen != 4) {
      mEdrop(FAIL_IO, "p_val(): bad hdrlen size: %" P32u, hdrlen);
   }

   /* fseek to check pseudo-block file length */
   if (fseek(fp, 0, SEEK_END) != 0) {
      mErrno(FAIL_IO, "p_val(): failed to fseek(END)");
   }
   plen = ftell(fp);
   if (plen == EOF) mErrno(FAIL_IO, "p_val(): failed to ftell(fp)");
   if (plen != sizeof(BTRAILER) + 4) {
      mError(FAIL_IO, "p_val(): invalid pseudo-block length: %ld", plen);
   }

   /* read trailer */
   if (fseek(fp, -(sizeof(BTRAILER)), SEEK_END) != 0) {
      mErrno(FAIL_IO, "p_val(): failed on fseek(END-BTRAILER)");
   } else if (fread(&bt, sizeof(BTRAILER), 1, fp) != 1) {
      mError(FAIL_IO, "p_val(): failed to fread(bt)");
   }

   /* check zeros */
   if (get32(bt.tcount) != 0) mEdrop(FAIL_IO, "p_val(): bad tcount");
   if (!iszero(bt.mroot, 32)) mEdrop(FAIL_IO, "p_val(): bad merkel array");
   if (!iszero(bt.nonce, 32)) mEdrop(FAIL_IO, "p_val(): bad nonce");

   /* check block num, hash, and difficulty */
   add64(Cblocknum, One, bnum);
   if (cmp64(bt.bnum, bnum) != 0) {
      mEdrop(FAIL_IO, "p_val(): bad block number");
   } else if (memcmp(bt.phash, Cblockhash, HASHLEN) != 0) {
      mEdrop(FAIL_IO, "p_val(): previous block hash mismatch");
   } else if (get32(bt.difficulty) != Difficulty) {
      mEdrop(FAIL_IO, "p_val(): bad difficulty");
   }

   /* check block times */
   if (get32(bt.time0) != Time0) {
      mEdrop(FAIL_IO, "p_val(): bad start time (time0)");
   } else if (get32(bt.stime) != Time0 + BRIDGE) {
      mEdrop(FAIL_IO, "p_val(): bad bridge time (stime)");
   } else if (!iszero(bt.mfee, 8)) {
      mEdrop(FAIL_IO, "p_val(): bad mining fee");
   }

   /* compute and check block hash */
   sha256_init(&ctx);
   sha256_update(&ctx, &hdrlen, 4);
   sha256_update(&ctx, &bt, sizeof(bt) - HASHLEN);
   sha256_final(&ctx, hash);
   if (memcmp(bt.bhash, hash, HASHLEN) != 0) {
      mEdrop(FAIL_IO, "p_val(): bad block hash");
   }

   pdebug("p_val(): completed in %gs", diffclocktime(clock(), ticks));

   ecode = VEOK;  /* success */

   /* cleanup / error handling */
FAIL_IO:
   fclose(fp);
FAIL:

   return ecode;
}  /* end p_val() */

/**
 * Validate a WOTS+ neo-genesis block.
 * Checks ledger entries are in ascending sort.
 * Checks block hash matches calculated hash.
 * Checks block size matches neo-genesis format.
 * Checks block trailer matches Tfile entry.
 * Checks sum of amounts do not exceed "expected" rewards.
 * NOTE: Tfile should have been verified before neo-genesis validation.
 * @param fname Filename of neo-genesis file to validate
 * @param tfname Filename of Tfile to validate against
 * @param bnum Pointer to expected block number, or NULL to ignore
 * @return (int) value representing operation result
 * @retval VEBAD on block format violation; check errno for details
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int ng_val(const char *fname, const char *tfname, void *bnum)
{
   LENTRY_W le, prev_le;
   SHA256_CTX cctx;
   BTRAILER bt, tft;
   size_t remain, count;
   word32 hdrlen, first;
   word8 chash[HASHLEN];
   word8 amounts[8];
   word8 rewards[8];
   FILE *fp;

   /* open file for validation */
   fp = fopen(fname, "rb");
   if (fp == NULL) return VERROR;

   /* read header length */
   if (fread(&hdrlen, sizeof(hdrlen), 1, fp) != 1) goto FAIL_ERR;

   /* check data from headerlen */
   remain = hdrlen;
   if (remain < (sizeof(hdrlen) + sizeof(le))) goto BAD_HDRLEN;
   if ((remain % sizeof(le)) != sizeof(hdrlen)) goto BAD_HDRLEN;

   /* init and begin computed hash */
   sha256_init(&cctx);
   sha256_update(&cctx, &hdrlen, sizeof(hdrlen));

   /* init amounts before summing */
   memset(amounts, 0, 8);

   /* read remaining block data */
   for (first = 1, remain -= sizeof(hdrlen); remain; remain -= sizeof(le)) {
      /* perform read into ledger entry -- check read count */
      if ((count = fread(&le, sizeof(le), 1, fp)) != 1) goto FAIL_IO;
      /* check ledger sort -- skip on first read */
      if (first) first = 0;
      else if (memcmp(le.addr, prev_le.addr, sizeof(le.addr)) <= 0) {
         goto BAD_SORT;
      }
      /* update amounts sum, ensure no overflow */
      if (add64(amounts, le.balance, amounts)) goto BAD_OVERFLOW;
      /* update data from neogenesis */
      memcpy(&prev_le, &le, sizeof(le));
      sha256_update(&cctx, &le, sizeof(le));
   }  /* end for() */

   /* read block trailer and check block number (where supplied) */
   if (fread(&bt, sizeof(bt), 1, fp) != 1) goto FAIL_IO;
   else if (bnum && cmp64(bt.bnum, bnum) != 0) goto BAD_BNUM;

   /* add trailer data -(HASHLEN) to computed hash -- finalize */
   sha256_update(&cctx, &bt, sizeof(bt) - HASHLEN);
   sha256_final(&cctx, chash);

   /* check block hash */
   if (memcmp(chash, bt.bhash, HASHLEN) != 0) goto BAD_BHASH;

   /* compare Neogenesis block trailer to Tfile trailer */
   if (read_tfile(&tft, bt.bnum, 1, tfname) != 1) {
      /* .. or the previous trailer where Tfile does not contain it */
      if (read_trailer(&tft, tfname) != VEOK) goto FAIL_ERR;
      if (validate_trailer(&bt, &tft) != VEOK) goto FAIL_ERR;
   } else if (memcmp(&bt, &tft, sizeof(bt)) != 0) goto BAD_TRAILER;

   /* check accurate sum of Tfile rewards against ledger amounts */
   /* NOTE: get_tfile_rewards() cannot caluclate balance < MFEE supply loss,
    * so we only check the Neogenesis amounts do not exceed expected */
   if (get_tfile_rewards(tfname, rewards, bt.bnum) != VEOK) goto FAIL_ERR;

   /* check calculated rewards against ledger amounts */
   if (cmp64(amounts, rewards) > 0) goto BAD_AMOUNTS;

   /* neogenesis is valid */
   fclose(fp);
   return VEOK;

/* error handling */
FAIL_IO:
   if (feof(fp)) {
      /* set_errno(EMCM_EOF); */
      pdebug("ng_val(): EOF");
   }
FAIL_ERR: fclose(fp); return VERROR;
FAIL_BAD: fclose(fp); return VEBAD;

/* block format violation handling */
BAD_AMOUNTS: set_errno(EMCM_LE_AMOUNTS_SUM); goto FAIL_BAD;
BAD_BHASH: set_errno(EMCM_BHASH); goto FAIL_BAD;
BAD_BNUM: set_errno(EMCM_BNUM); goto FAIL_BAD;
BAD_HDRLEN: set_errno(EMCM_HDRLEN); goto FAIL_BAD;
BAD_TRAILER: set_errno(EMCM_TRAILER); goto FAIL_BAD;
BAD_OVERFLOW: set_errno(EMCM_LE_AMOUNTS_OVERFLOW); goto FAIL_BAD;
BAD_SORT: set_errno(EMCM_LE_SORT); goto FAIL_BAD;
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
   int cond, count, ecode;

   /* init */
   ticks = clock();
   memset(mfees, 0, sizeof(mfees));

   pdebug("b_val(): validating blockchain file %s...", fname);

   /* open ledger read-only */
   if (le_open("ledger.dat", "rb") != VEOK) {
      mErrno(FAIL, "b_val(): failed to le_open(ledger.dat)");
   }
   /* create ledger transaction temp file */
   ltfp = fopen("ltran.tmp", "wb");
   if (ltfp == NULL) mErrno(FAIL, "b_val(): failed to fopen(ltran.tmp)");
   /* open the block to validate */
   fp = fopen(fname, "rb");
   if (fp == NULL) mErrno(FAIL_IN, "b_val(): failed to fopen(%s)", fname);
   /* read and check regular fixed size block header */
   if (fread(&hdrlen, 4, 1, fp) != 1) {
      mError(FAIL_IO, "b_val(): failed to fread(hdrlen)");
   } else if (hdrlen != sizeof(BHEADER)) {
      mEdrop(FAIL_IO, "b_val(): bad hdrlen size: %" P32u, hdrlen);
   }

   /* compute block file length */
   if (fseek(fp, 0, SEEK_END) != 0) {
      mErrno(FAIL_IO, "b_val(): failed to fseek(END)");
   }
   blocklen = ftell(fp);
   if (blocklen == EOF) mError(FAIL_IO, "b_val(): failed on ftell(fp)");

   /* Read block trailer:
    * Check phash, bnum,
    * difficulty, Merkel Root, nonce, solve time, and block hash.
    */
   if (fseek(fp, -(sizeof(BTRAILER)), SEEK_END) != 0) {
      mErrno(FAIL_IO, "b_val(): failed on fseek(-BTRAILER)");
   } else if (fread(&bt, sizeof(BTRAILER), 1, fp) != 1) {
      mError(FAIL_IO, "b_val(): failed on fread(bt)");
   }
   /* check block number */
   add64(Cblocknum, One, bnum);
   if (memcmp(bnum, bt.bnum, 8) != 0) {
      pdebug("b_val(): bad block number");
      mEcode(FAIL_IO, VEBAD);
   }
   /* check block times */
   stemp = get32(bt.stime);
   if (stemp <= Time0) mEdrop(FAIL_IO, "b_val(): early block time");
   if (stemp > (time(NULL) + BCONFREQ)) {
      mEdrop(FAIL_IO, "b_val(): time travel?");
   }
   if (cmp64(bnum, tot_trigger) > 0 && Cblocknum[0] != 0xfe) {
      if ((stemp - get32(bt.time0)) > BRIDGE) {
         mEdrop(FAIL_IO, "b_val(): invalid TOT trigger");
      }
   } else if (cmp64(bt.mfee, Mfee) < 0) {
      mEdrop(FAIL_IO, "b_val(): bad mining fee");
   } else if (get32(bt.difficulty) != Difficulty) {
      mEdrop(FAIL_IO, "b_val(): difficulty mismatch");
   }

   /* check previous block hash */
   if (memcmp(Cblockhash, bt.phash, HASHLEN) != 0) {
      mEdrop(FAIL_IO, "b_val(): previous block hash mismatch");
   }
   /* check transaction count */
   tcount = get32(bt.tcount);
   if (tcount == 0 || tcount > MAXBLTX) {
      mEdrop(FAIL_IO, "b_val(): bad tcount");
   }
   /* check total block length */
   if ((hdrlen + sizeof(BTRAILER) + (tcount * sizeof(TXQENTRY)))
         != (word32) blocklen) {
      mEdrop(FAIL_IO, "b_val(): invalid block length");
   }

   /* check enforced delay, collect haiku from block */
   if (cmp64(bnum, v24_trigger) > 0) {
      /* Boxing Day Anomaly -- Bugfix */
      if (cmp64(bt.bnum, boxingday) == 0) {
         if (memcmp(bt.bhash, boxdayhash, 32) != 0) {
            mEdrop(FAIL_IO, "b_val(): bad boxingday anomaly bhash");
         }
      } else if (peach_check(&bt)) mEdrop(FAIL_IO, "b_val(): bad Peach");
   } else if (trigg_check(&bt)) mEdrop(FAIL_IO, "b_val(): bad Trigg");

   /* Read block header */
   if (fseek(fp, 0, SEEK_SET) != 0) {
      mErrno(FAIL_IO, "b_val(): failed on fseek(SET)");
   } else if (fread(&bh, hdrlen, 1, fp) != 1) {
      mError(FAIL_IO, "b_val(): failed on fread(bh)");
   }
   /* check mining reward/address */
   get_mreward(mreward, bnum);
   if (memcmp(bh.mreward, mreward, 8) != 0) {
      mEdrop(FAIL_IO, "b_val(): bad mining reward");
   } else if (ADDR_HAS_TAG(bh.maddr)) {
      mEdrop(FAIL_IO, "b_val(): maddr has tag");
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
      mError(FAIL_IO, "b_val(): failed to malloc(%zu) Q2", len);
   }

   /* Validate each transaction */
   for (j = 0; j < tcount; j++) {
      if (j >= MAXBLTX) mError(FAIL_TX, "b_val(): too many transactions");
      if (fread(&tx, sizeof(TXQENTRY), 1, fp) != 1) {
         mError(FAIL_TX, "b_val(): failed on fread(TX): TX#%" P32u, j);
      } else if (cmp64(tx.tx_fee, Mfee) < 0) {
         mEdrop(FAIL_TX, "b_val(): bad tx_fee: TX#%" P32u, j);
      } else if (memcmp(tx.src_addr, tx.chg_addr, TXWOTSLEN) == 0) {
         mEdrop(FAIL_TX, "b_val(): (src == chg): TX#%" P32u, j);
      } else if (!TX_IS_MTX(&tx)) {
         if (memcmp(tx.src_addr, tx.dst_addr, TXWOTSLEN) == 0) {
            mEdrop(FAIL_TX, "b_val(): (src == dst): TX#%" P32u, j);
         }
      }

      /* running block/merkel hash */
      sha256_update(&bctx, &tx, sizeof(TXQENTRY));
      sha256_update(&mctx, &tx, sizeof(TXQENTRY));
      /* tx_id is hash of tx.src_addr */
      sha256(tx.src_addr, TXWOTSLEN, tx_id);
      if (memcmp(tx_id, tx.tx_id, HASHLEN) != 0) {
         mEdrop(FAIL_TX, "b_val(): bad tx_id: TX#%" P32u, j);
      }

      /* Check that tx_id is sorted. */
      if (j != 0) {
         cond = memcmp(tx_id, prev_tx_id, HASHLEN);
         if (cond <= 0) {
            if (cond == 0) {
               mEdrop(FAIL_TX, "b_val(): duplicate tx_id: TX#%" P32u, j);
            } else mEdrop(FAIL_TX, "b_val(): unsorted tx_id: TX#%" P32u, j);
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
         mEdrop(FAIL_TX, "b_val(): WOTS signature failed: TX#%" P32u, j);
      }

      /* look up source address in ledger */
      if (le_find(tx.src_addr, &src_le, TXWOTSLEN) == 0) {
         pdebug("b_val(): error address %s...", addr2str(tx.src_addr));
         mEdrop(FAIL_TX, "b_val(): src_addr not in ledger: TX#%" P32u, j);
      }

      total[0] = total[1] = 0;
      /* use add64() to check for carry out, total, and fees */
      cond =  add64(tx.send_total, tx.change_total, total);
      cond += add64(tx.tx_fee, total, total);
      if (cond) {
         mEdrop(FAIL_TX, "b_val(): total overflow: TX#%" P32u, j);
      } else if (cmp64(src_le.balance, total) != 0) {
         mEdrop(FAIL_TX, "b_val(): bad transaction total: TX#%" P32u, j);
      } else if (add64(mfees, tx.tx_fee, mfees)) {
         mError(FAIL_TX, "b_val(): mfees overflow: TX#%" P32u, j);
      }
      /* check mtx/tag_valid */
      if (!TX_IS_MTX(&tx)) {
         if(tag_valid(tx.src_addr, tx.chg_addr, tx.dst_addr, bt.bnum)) {
            mEdrop(FAIL_TX, "b_val(): tag not valid: TX#%" P32u, j);
         }
      } else if(mtx_val((MTX *) &tx, Mfee)) {
         mEdrop(FAIL_TX, "b_val(): bad mtx_val: TX#%" P32u, j);
      }

      /* copy TX to tag queue */
      memcpy(&Q2[j], &tx, sizeof(TXQENTRY));
   }  /* end for j */

   /* finalize Merkel Root - phash, bnum, mfee, tcount, time0, difficulty */
   if (NEWYEAR(bt.bnum)) sha256_update(&mctx, &bt, (HASHLEN+8+8+4+4+4));
   sha256_final(&mctx, mroot);
   if (memcmp(bt.mroot, mroot, HASHLEN) != 0) {
      mEdrop(FAIL_TX, "b_val(): bad merkel root");
   }

   /* finalize block hash - Block trailer (- block hash) */
   sha256_update(&bctx, &bt, sizeof(BTRAILER) - HASHLEN);
   sha256_final(&bctx, bhash);
   if (memcmp(bt.bhash, bhash, HASHLEN) != 0) {
      mEdrop(FAIL_TX, "b_val(): bad block hash");
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
      if (cond) mError(FAIL_TX, "b_val(): scan3 total overflow");

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
      mError(FAIL_TX, "b_val(): scan3 tcount mismatch: %" P32u, j);
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
            mError(FAIL_TX, "b_val(): bad I/O dst-->chg write");
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
         if (count != 1) mError(FAIL_TX, "b_val(): bad I/O scan 4");
      }  /* end for j */
   }  /* end for qp1 */
   /* end mtx scan 4 */

   /* Create a transaction amount = mreward + mfees
    * address = bh.maddr
    */
   if (add64(mfees, mreward, mfees)) {
      mError(FAIL_TX, "b_val(): mfees overflow");
   }
   /* Make ledger tran to add to or create mining address.
    * '...Money from nothing...'
    */
   count = fwrite_hashed(bh.maddr, "A", mfees, ltfp);
   if (count != 1 || ferror(ltfp)) {
      mError(FAIL_TX, "b_val(): ltfp IO error");
   } else {
      pdebug("b_val(): wrote reward (%08x%08x) to %s...",
         mreward[1], mreward[0], addr2str(bh.maddr));
   }

   /* cleanup */
   free(Q2);
   fclose(fp);
   fclose(ltfp);  /* revert failure mode to (FAIL) */
   /* promote ltran.tmp to *.dat file and rename blockchain file */
   remove("ltran.dat");
   if (rename("ltran.tmp", "ltran.dat") != 0) {
      mError(FAIL, "b_val(): failed to move ltran.tmp to ltran.dat");
   }

   pdebug("b_val(): %s validated", fname);
   pdebug("b_val(): completed in %gs", diffclocktime(clock(), ticks));

   /* success */
   return VEOK;

   /* failure / error handling */
FAIL_TX:
   free(Q2);
FAIL_IO:
   fclose(fp);
FAIL_IN:
   fclose(ltfp);
   remove("ltran.tmp");
FAIL:

   return ecode;
}  /* end b_val() */

/* end include guard */
#endif
