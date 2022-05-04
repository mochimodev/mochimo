/**
 * @private
 * @headerfile validate.h <validate.h>
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_VALIDATE_C
#define MOCHIMO_VALIDATE_C


#include "validate.h"

/* system support */
#include <errno.h>

/* extended-c support */
#include "extlib.h"     /* general support */
#include "extmath.h"    /* 64-bit math support */

/* crypto support */
#include "crc16.h"

/* algo support */
#include "peach.h"
#include "trigg.h"
#include "wots.h"

/* mochimo support */
#include "config.h"
#include "data.c"
#include "util.c"
#include "ledger.c"

/* validation error MACROs */

/**
 * Validation error code. Sets @a ecode to given value and jumps to label.
 * Example: @code vEcode(FAIL_LABEL, VETIMEOUT); @endcode
 * @param _lbl Label to jump to
 * @param _e Error code to set ecode to
*/
#define vEcode(_lbl, _e)   { ecode = _e; goto _lbl; }

/**
 * Validation protocol violation. Calls perr(...) with variable arguments,
 * sets @a ecode to VEBAD2 (indicating that a peer is in violation of
 * protocol and may need pinklisting), and jumps to label.
 * Example: @code vEdrop(FAIL_LABEL, "Violation of protocol"); @endcode
 * @param _lbl Label to jump to
 * @param ... arguments passed to perr()
*/
#define vEdrop(_lbl, ...)  { perr(__VA_ARGS__); vEcode(_lbl, VEBAD2); }

/**
 * Validation error w/ error number. Calls perrno(...) with variable
 * arguments, sets @a ecode to VERROR, and jumps to label.
 * Example: @code vErrno(FAIL_LABEL, errno, "Failure message"); @endcode
 * @param _lbl Label to jump to
 * @param ... arguments passed to perr()
*/
#define vErrno(_lbl, ...)  vEcode(_lbl, perrno(__VA_ARGS__));

/**
 * Validation error. Calls perr(...) with variable arguments, sets
 * @a ecode to VERROR, and jumps to label.
 * Example: @code vError(FAIL_LABEL, "Failure message"); @endcode
 * @param _lbl Label to jump to
 * @param ... arguments passed to perr()
*/
#define vError(_lbl, ...)  vEcode(_lbl, perr(__VA_ARGS__));


#define BAIL(m) { message = m; goto bail; }

/* Validates a multi-dst transaction MTX.
 * (Does all tag checking as well.)
 * tx->src_addr is already checked in ledger.dat and totals tally.
 * tx_val() sets fee parameter to Myfee and bval.c sets fee to Mfee.
 * Returns 0 on valid, else error code.
 */
int mtx_val(MTX *mtx, word32 *fee)
{
   int j, message;
   word8 total[8], mfees[8], *bp, *limit;
   static word8 addr[TXADDRLEN];

   limit = &mtx->zeros[0];

   /* Check that src and chg have tags.
    * Check that src and chg have same tag.
    * tx_val() or bval.c has already checked src != chg, src exists,
    *   sig is good, and totals are good.
    */
   if(!ADDR_HAS_TAG(mtx->src_addr)) BAIL(1);
   if(memcmp(ADDR_TAG_PTR(mtx->src_addr),
             ADDR_TAG_PTR(mtx->chg_addr), TXTAGLEN) != 0) BAIL(2);
   if(cmp64(mtx->change_total, Mfee) <= 0) BAIL(3);

   memset(total, 0, 8);
   memset(mfees, 0, 8);
   /* Tally each dst[] amount and mfees... */
   for(j = 0; j < MDST_NUM_DST; j++) {
      /* zero dst[] tag marks end of list.  */
      if(iszero(mtx->dst[j].tag, TXTAGLEN)) {
         for(bp = mtx->dst[j].amount; bp < limit; bp++) {
            if(*bp) BAIL(4);  /* Check that rest of dst[] list is zeros. */
         }
         break;
      }
      if(iszero(mtx->dst[j].amount, 8)) BAIL(5);  /* bad send amount */
      /* no dst to src */
      if(memcmp(mtx->dst[j].tag,
                ADDR_TAG_PTR(mtx->src_addr), TXTAGLEN) == 0) BAIL(6);
      /* tally fees and send_total */
      if(add64(total, mtx->dst[j].amount, total)) BAIL(7);
      if(add64(mfees, fee, mfees)) BAIL(8);  /* Mfee or Myfee */
      if(get32(Cblocknum) >= MTXTRIGGER) {
         memcpy(ADDR_TAG_PTR(addr), mtx->dst[j].tag, TXTAGLEN);
         mtx->zeros[j] = 0;
         /* If dst[j] tag not found, put error code in zeros[] array. */
         if(tag_find(addr, NULL, NULL, TXTAGLEN) != VEOK) mtx->zeros[j] = 1;
      }
   }  /* end for j */
   /* Check tallies... */
   if(cmp64(total, mtx->send_total) != 0) BAIL(9);
   if(cmp64(mtx->tx_fee, mfees) < 0) BAIL(10);
   return 0;  /* valid */
bail:
   if(message && Trace) plog("mtx_val(): %d", message);
   return message;  /* bad */
}  /* end mtx_val() */


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
            plog("DST_ADDR Tagged, but Tag is not in ledger!");
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
      plog("SRC_TAG != CHG_TAG, and SRC_TAG is Non-Default!");
      goto bad;
   }
   /* Otherwise, check all queues and ledger.dat for change tag.
    * First, if change tag is in ledger.dat, tx is invalid.
    */
   if(tag_find(chg_addr, NULL, NULL, TXTAGLEN) == VEOK) {
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
   pdebug("Tag created");
   return VEOK;  /* If we get here, a new TX change tag gets created. */
good:
   pdebug("Tag moved");
   return VEOK;
bad:
   pdebug("Tag rejected");
   return VERROR;
}  /* end tag_valid() */

/* Validate a transaction against ledger
 *
 * Returns: 0 if vaild (accept)
 *          1 if server error (drop)
 *          2 or 3 if evil    (drop)
 */
int tx_val(TX *tx)
{
   int cond;
   static LENTRY src_le;            /* source ledger entry */
   word32 total[2];                 /* for 64-bit maths */
   static word8 message[HASHLEN];    /* transaction hash for WOTS */
   static word8 pk2[TXSIGLEN];       /* more WOTS */
   static word8 rnd2[32];            /* for WOTS addr[] */
   MTX *mtx;
   static TX txs;

   if(memcmp(tx->src_addr, tx->chg_addr, TXADDRLEN) == 0) {
      pdebug("tx_val(): src == chg");  /* also mtx */
      return 2;
   }

   if(!TX_IS_MTX(tx) && memcmp(tx->src_addr, tx->dst_addr, TXADDRLEN) == 0) {
      pdebug("tx_val(): src == dst");
      return 2;
   }

   /* validate transaction fixed fee */
   if(cmp64(tx->tx_fee, Mfee) < 0) {
      pdebug("tx_val(): bad mining fee");
      return 2;
   }
   /* validate my fee */
   if(cmp64(tx->tx_fee, Myfee) < 0) {
      pdebug("tx_val(): fee < %u", Myfee[0]);
      return 1;
   }

   /* check WTOS signature */
   if(TX_IS_MTX(tx) && get32(Cblocknum) >= MTXTRIGGER) {
      memcpy(&txs, tx, sizeof(txs));
      mtx = (MTX *) TRANBUFF(&txs);  /* poor man's union */
      memset(mtx->zeros, 0, MDST_NUM_DZEROS);
      sha256(txs.src_addr, TXSIGHASH_COUNT, message);
   } else {
      sha256(tx->src_addr, TXSIGHASH_COUNT, message);
   }

   memcpy(rnd2, &tx->src_addr[TXSIGLEN+32], 32);  /* copy WOTS addr[] */
   wots_pk_from_sig(pk2, tx->tx_sig, message, &tx->src_addr[TXSIGLEN],
                    (word32 *) rnd2);
   if(memcmp(pk2, tx->src_addr, TXSIGLEN) != 0) {
      plog("tx_val(): WOTS signature failed!");
      return 3;
   }

   /* look up source address in ledger */
   if(le_find(tx->src_addr, &src_le, NULL, TXADDRLEN) == FALSE) {
      pdebug("tx_val(): src_addr not in ledger");
      return 1;
   }
   total[0] = total[1] = 0;
   /* use add64() to check for overflow */
   cond =  add64(tx->send_total, tx->change_total, total);
   cond += add64(tx->tx_fee, total, total);
   if(cond) {
      plog("tx_val(): TX amount overflow");
      return 2;
   }
   if(cmp64(src_le.balance, total) != 0) {
      pdebug("tx_val(): bad transaction total != src_le.balance");
      return 1;
   }
   if(TX_IS_MTX(tx)) {
      mtx = (MTX *) TRANBUFF(tx);  /* poor man's union */
      if(mtx_val(mtx, Myfee)) return 1;  /* bad mtx */
   } else {
      if(tag_valid(tx->src_addr, tx->chg_addr, tx->dst_addr,
                   NULL) != VEOK) return 1;  /* bad tag */
   }
   return 0;  /* tx valid */
}  /* end tx_val() */

/* Validate a pseudo-block -- Called from update() */
int p_val(char *fname)
{
   BTRAILER bt;
   SHA256_CTX ctx;
   word8 hash[HASHLEN];
   word32 hdrlen, bnum[2];
   int ecode;
   FILE *fp;
   clock_t ticks; /* for function execution time */

   /* init */
   ticks = clock();
   pdebug("p_val(%s): entering...", fname);

   /* open the pseudo-block to validate */
   fp = fopen(fname, "rb");
   if (fp == NULL) vError(FAIL, "p_val(): cannot open %s", fname);
   /* read and check regular fixed size block header */
   if (fread(&hdrlen, 4, 1, fp) != 1) {
      vError(FAIL_IO, "p_val(): failed to fread(hdrlen)");
   } else if (hdrlen != 4) {
      vEdrop(FAIL_IO, "p_val(): bad hdrlen size: %" P32u, hdrlen);
   }

   /* compute block file length */
   ecode = fseek(fp, 0, SEEK_END);
   if (ecode) vErrno(FAIL_IO, ecode, "p_val(): failed to fseek(END)");
   if (ftell(fp) != sizeof(BTRAILER) + 4) {
      vError(FAIL_IO, "p_val(): failed on ftell(fp)");
   }

   /* read trailer */
   ecode = fseek(fp, -(sizeof(BTRAILER)), SEEK_END);
   if (ecode) vErrno(FAIL_IO, ecode, "p_val(): failed on fseek(-BTRAILER)");
   if (fread(&bt, sizeof(BTRAILER), 1, fp) != 1) {
      vError(FAIL_IO, "p_val(): failed to fread(bt)");
   }

   /* check zeros */
   if (get32(bt.tcount) != 0) vEdrop(FAIL_IO, "p_val(): bad tcount");
   if (!iszero(bt.mroot, 32)) vEdrop(FAIL_IO, "p_val(): bad merkel array");
   if (!iszero(bt.nonce, 32)) vEdrop(FAIL_IO, "p_val(): bad nonce");

   /* check block num, hash, and difficulty */
   add64(Cblocknum, One, bnum);
   if (cmp64(bt.bnum, bnum) != 0) {
      vEdrop(FAIL_IO, "p_val(): bad block number");
   } else if (memcmp(bt.phash, Cblockhash, HASHLEN) != 0) {
      vEdrop(FAIL_IO, "p_val(): previous block hash mismatch");
   } else if (get32(bt.difficulty) != Difficulty) {
      vEdrop(FAIL_IO, "p_val(): bad difficulty");
   } 

   /* check block times */
   if (get32(bt.time0) != Time0) {
      vEdrop(FAIL_IO, "p_val(): bad start time (time0)");
   } else if (get32(bt.stime) != Time0 + BRIDGE) {
      vEdrop(FAIL_IO, "p_val(): bad bridge time (stime)");
   } else if (!iszero(bt.mfee, 8)) {
      vEdrop(FAIL_IO, "p_val(): bad mining fee");
   }

   /* compute and check block hash */
   sha256_init(&ctx);
   sha256_update(&ctx, &hdrlen, 4);
   sha256_update(&ctx, &bt, sizeof(bt) - HASHLEN);
   sha256_final(&ctx, hash);
   if (memcmp(bt.bhash, hash, HASHLEN) != 0) {
      vEdrop(FAIL_IO, "p_val(): bad block hash");
   }

   remove("ublock.dat");
   if (rename(fname, "ublock.dat") != 0) {
      vError(FAIL_IO, "p_val(): failed to move %s to ublock.dat", fname);
   }

   ecode = VEOK; /* success */
   pdebug("p_val(): completed in %u ticks.", (word32) (clock() - ticks));

   /* cleanup - error handling */
FAIL_IO:
   fclose(fp);
FAIL:

   return ecode;
}  /* end p_val() */

/**
 * Validate a blockchain file, as the next block, according to node state.
 * @param fname Name of blockchain file to validate
 * @param vfname If not NULL, renames @a fname to this filename on success
 * @returns VEOK on success, else VERROR
 * @note Creates ltran.dat on success.
 * @note Uses the folowing globals:<br/>
 * Mfee, Difficulty, Time0, Cblocknum, Cblockhash
*/
int b_val(char *fname, char *vfname)
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
   word8 addr[TXADDRLEN];     /* for mtx scan 4 */
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
   unsigned j;
   int cond, count;
   int ecode;

   /* init */
   ticks = clock();
   pdebug("b_val(%s, %s): entering...", fname, vfname);

   /* open ledger read-only */
   if (le_open("ledger.dat", "rb") != VEOK) {
      vErrno(FAIL_LE, errno, "b_val(): cannot open ledger.dat");
   }
   /* create ledger transaction temp file */
   ltfp = fopen("ltran.tmp", "wb");
   if (ltfp == NULL) {
      vErrno(FAIL_LT, errno, "b_val(): cannot open ltran.tmp");
   }
   /* open the block to validate */
   fp = fopen(fname, "rb");
   if (fp == NULL) vErrno(FAIL_FP, errno, "b_val(): cannot open %s", fname);
   /* read and check regular fixed size block header */
   if (fread(&hdrlen, 1, 4, fp) != 4) {
      vError(FAIL_IO, "b_val(): failed to fread(hdrlen)");
   } else if (hdrlen != sizeof(BHEADER)) {
      vEdrop(FAIL_IO, "b_val(): bad hdrlen size: %" P32u, hdrlen);
   }

   /* compute block file length */
   ecode = fseek(fp, 0, SEEK_END);
   if (ecode) vErrno(FAIL_IO, ecode, "b_val(): failed to fseek(END)");
   blocklen = ftell(fp);
   if (blocklen == EOF) vError(FAIL_IO, "b_val(): failed on ftell(fp)");

   /* Read block trailer:
    * Check phash, bnum,
    * difficulty, Merkel Root, nonce, solve time, and block hash.
    */
   ecode = fseek(fp, -(sizeof(BTRAILER)), SEEK_END);
   if (ecode) vErrno(FAIL_IO, ecode, "b_val(): failed on fseek(-BTRAILER)");
   if (fread(&bt, 1, sizeof(BTRAILER), fp) != sizeof(BTRAILER)) {
      vError(FAIL_IO, "b_val(): failed on fread(bt)");
   } else if (cmp64(bt.mfee, Mfee) < 0) {
      vEdrop(FAIL_IO, "b_val(): bad mining fee");
   } else if (get32(bt.difficulty) != Difficulty) {
      vEdrop(FAIL_IO, "b_val(): difficulty mismatch");
   }

   /* check block times */
   stemp = get32(bt.stime);
   if (stemp <= Time0) vEdrop(FAIL_IO, "b_val(): early block time");
   if (stemp > (time(NULL) + BCONFREQ)) {
      vEdrop(FAIL_IO, "b_val(): time travel?");
   }
   /* check block number */
   add64(Cblocknum, One, bnum);
   if (memcmp(bnum, bt.bnum, 8) != 0) {
      vEdrop(FAIL_IO, "b_val(): bad block number");
   } else if (cmp64(bnum, tot_trigger) > 0 && Cblocknum[0] != 0xfe) {
      if ((word32) (stemp - get32(bt.time0)) > BRIDGE) {
         vEdrop(FAIL_IO, "b_val(): invalid TOT trigger");
      }
   }
   /* check previous block hash */
   if (memcmp(Cblockhash, bt.phash, HASHLEN) != 0) {
      vEdrop(FAIL_IO, "b_val(): previous block hash mismatch");
   }
   /* check transaction count */
   tcount = get32(bt.tcount);
   if(tcount == 0 || tcount > MAXBLTX) {
      vEdrop(FAIL_IO, "b_val(): bad tcount");
   }
   /* check total block length */
   if((hdrlen + sizeof(BTRAILER) + (tcount * sizeof(TXQENTRY)))
         != (word32) blocklen) {
      vEdrop(FAIL_IO, "b_val(): invalid block length");
   }

   /* check enforced delay, collect haiku from block */
   if (cmp64(bnum, v24_trigger) > 0) {
      /* Boxing Day Anomaly -- Bugfix */
      if (cmp64(bt.bnum, boxingday) == 0) {
         if (memcmp(bt.bhash, boxdayhash, 32) != 0) {
            vEdrop(FAIL_IO, "b_val(): bad boxingday anomaly bhash");
         }
      } else if (peach_check(&bt)) vEdrop(FAIL_IO, "b_val(): bad Peach");
   } else if (trigg_check(&bt)) vEdrop(FAIL_IO, "b_val(): bad Trigg");

   /* Read block header */
   ecode = fseek(fp, 0, SEEK_SET);
   if (ecode) vErrno(FAIL_IO, ecode, "b_val(): failed on fseek(SET)");
   if (fread(&bh, 1, hdrlen, fp) != hdrlen) {
      vError(FAIL_IO, "b_val(): failed on fread(bh)");
   }
   /* check mining reward/address */
   get_mreward(mreward, bnum);
   if (memcmp(bh.mreward, mreward, 8) != 0) {
      vEdrop(FAIL_IO, "b_val(): bad mining reward");
   } else if (ADDR_HAS_TAG(bh.maddr)) {
      vEdrop(FAIL_IO, "b_val(): maddr has tag");
   }

   /* fp left at offset of Merkel Block Array--ready to fread() */

   /* begin hashing contexts */
   sha256_init(&bctx);   /* begin entire block hash */
   sha256_update(&bctx, &bh, hdrlen);  /* ... with the header */
   if (!NEWYEAR(bt.bnum)) sha256_init(&mctx); /* begin Merkel Array hash */
   else memcpy(&mctx, &bctx, sizeof(mctx));  /* ... or copy bctx state */

   /* temp TX tag processing queue */
   len = tcount * sizeof(TXQENTRY);
   Q2 = malloc(len);
   if (Q2 == NULL) {
      vError(FAIL_Q2, "b_val(): failed to malloc(%zu) for Q2", len);
   }

   /* Validate each transaction */
   for (j = 0; j < tcount; j++) {
      if (j >= MAXBLTX) vError(FAIL_TX, "b_val(): too many transactions");
      if (fread(&tx, sizeof(TXQENTRY), 1, fp) != 1) {
         vError(FAIL_TX, "b_val(): failed on fread(TX): TX#%" P32u, j);
      } else if (cmp64(tx.tx_fee, Mfee) < 0) {
         vEdrop(FAIL_TX, "b_val(): bad tx_fee: TX#%" P32u, j);
      } else if (memcmp(tx.src_addr, tx.chg_addr, TXADDRLEN) == 0) {
         vEdrop(FAIL_TX, "b_val(): (src == chg): TX#%" P32u, j);
      } else if (!TX_IS_MTX(&tx)) {
         if (memcmp(tx.src_addr, tx.dst_addr, TXADDRLEN) == 0) {
            vEdrop(FAIL_TX, "b_val(): (src == dst): TX#%" P32u, j);
         }
      }

      /* running block/merkel hash */
      sha256_update(&bctx, &tx, sizeof(TXQENTRY));
      sha256_update(&mctx, &tx, sizeof(TXQENTRY));
      /* tx_id is hash of tx.src_addr */
      sha256(tx.src_addr, TXADDRLEN, tx_id);
      if (memcmp(tx_id, tx.tx_id, HASHLEN) != 0) {
         vEdrop(FAIL_TX, "b_val(): bad tx_id: TX#%" P32u, j);
      }

      /* Check that tx_id is sorted. */
      if (j != 0) {
         cond = memcmp(tx_id, prev_tx_id, HASHLEN);
         if (cond <= 0) {
            if (cond == 0) {
               vEdrop(FAIL_TX, "b_val(): duplicate tx_id: TX#%" P32u, j);
            } else vEdrop(FAIL_TX, "b_val(): unsorted tx_id: TX#%" P32u, j);
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
         sha256(txs.src_addr, TXSIGHASH_COUNT, msg);
      } else sha256(tx.src_addr, TXSIGHASH_COUNT, msg);
      memcpy(rnd2, &tx.src_addr[TXSIGLEN + 32], 32);  /* copy WOTS addr[] */
      wots_pk_from_sig(pk2, tx.tx_sig, msg, &tx.src_addr[TXSIGLEN], rnd2);
      if (memcmp(pk2, tx.src_addr, TXSIGLEN) != 0) {
         vEdrop(FAIL_TX, "b_val(): WOTS signature failed: TX#%" P32u, j);
      }

      /* look up source address in ledger */
      if (le_find(tx.src_addr, &src_le, NULL, TXADDRLEN) == 0) {
         vEdrop(FAIL_TX, "b_val(): src_addr not in ledger: TX#%" P32u, j);
      }

      total[0] = total[1] = 0;
      /* use add64() to check for carry out, total, and fees */
      cond =  add64(tx.send_total, tx.change_total, total);
      cond += add64(tx.tx_fee, total, total);
      if (cond) {
         vEdrop(FAIL_TX, "b_val(): total overflow: TX#%" P32u, j);
      } else if (cmp64(src_le.balance, total) != 0) {
         vEdrop(FAIL_TX, "b_val(): bad transaction total: TX#%" P32u, j);
      } else if (add64(mfees, tx.tx_fee, mfees)) {
         vError(FAIL_TX, "b_val(): mfees overflow: TX#%" P32u, j);
      }
      /* check mtx/tag_valid */
      if (!TX_IS_MTX(&tx)) {
         if(tag_valid(tx.src_addr, tx.chg_addr, tx.dst_addr, bt.bnum)) {
            vEdrop(FAIL_TX, "b_val(): tag not valid: TX#%" P32u, j);
         }
      } else if(mtx_val((MTX *) &tx, Mfee)) {
         vEdrop(FAIL_TX, "b_val(): bad mtx_val: TX#%" P32u, j);
      }

      /* copy TX to tag queue */
      memcpy(&Q2[j], &tx, sizeof(TXQENTRY));
   }  /* end for j */

   /* finalize Merkel Root - phash, bnum, mfee, tcount, time0, difficulty */
   if (NEWYEAR(bt.bnum)) sha256_update(&mctx, &bt, (HASHLEN+8+8+4+4+4));
   sha256_final(&mctx, mroot);
   if (memcmp(bt.mroot, mroot, HASHLEN) != 0) {
      vEdrop(FAIL_FINAL, "b_val(): bad merkel root");
   }

   /* finalize block hash - Block trailer (- block hash) */
   sha256_update(&bctx, &bt, sizeof(BTRAILER) - HASHLEN);
   sha256_final(&bctx, bhash);
   if (memcmp(bt.bhash, bhash, HASHLEN) != 0) {
      vEdrop(FAIL_FINAL, "b_val(): bad block hash");
   }

   /* When spending from a tag, the address associated with said tag will
    * change. Any transactions to dst_tags that also exist as the src_tags
    * for other transactions within the same block, MUST have their dst_addr
    * replaced with the chg_addr of said src_tag transactions. */

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
                   memcpy(qp2->dst_addr, qp1->chg_addr, TXADDRLEN);
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
      if (cond) vError(FAIL_FINAL, "b_val(): scan3 total overflow");

      /* Write ledger transactions to ltran.tmp for all src and chg,
       * but only non-mtx dst
       * that will have to be sorted, read again, and applied by bup...
       */
      fwrite(qp1->src_addr,  1, TXADDRLEN, ltfp);
      fwrite("-",            1,         1, ltfp);  /* debit src addr */
      fwrite(total,          1,         8, ltfp);
      /* add to or create non-multi dst address */
      if(!TX_IS_MTX(qp1) && !iszero(qp1->send_total, 8)) {
         fwrite(qp1->dst_addr,   1, TXADDRLEN, ltfp);
         fwrite("A",             1,         1, ltfp);
         fwrite(qp1->send_total, 1,         8, ltfp);
      }
      /* add to or create change address */
      if(!iszero(qp1->change_total, 8)) {
         fwrite(qp1->chg_addr,     1, TXADDRLEN, ltfp);
         fwrite("A",               1,         1, ltfp);
         fwrite(qp1->change_total, 1,         8, ltfp);
      }
   }  /* end for j -- scan 3 */
   if(j != tcount) {
      vError(FAIL_FINAL, "b_val(): scan3 tcount mismatch: %" P32u, j);
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
            count =  fwrite(mtx->chg_addr, TXADDRLEN, 1, ltfp);
            count += fwrite("A", 1, 1, ltfp);
            count += fwrite(mtx->dst[j].amount, 8, 1, ltfp);
            if (count == 3) continue;  /* next dst[j] */
            vError(FAIL_FINAL, "b_val(): bad I/O dst-->chg write");
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
                         memcpy(addr, qp2->chg_addr, TXADDRLEN);
                         break;
            }
         }  /* end for qp2 scan 5 */
         /* write out the dst transaction */
         count =  fwrite(addr, TXADDRLEN, 1, ltfp);
         count += fwrite("A", 1, 1, ltfp);
         count += fwrite(mtx->dst[j].amount, 8, 1, ltfp);
         if (count != 3) vError(FAIL_FINAL, "b_val(): bad I/O scan 4");
      }  /* end for j */
   }  /* end for qp1 */
   /* end mtx scan 4 */

   /* Create a transaction amount = mreward + mfees
    * address = bh.maddr
    */
   if (add64(mfees, mreward, mfees)) {
      vError(FAIL_FINAL, "b_val(): mfees overflow");
   }
   /* Make ledger tran to add to or create mining address.
    * '...Money from nothing...'
    */
   count =  fwrite(bh.maddr, 1, TXADDRLEN, ltfp);
   count += fwrite("A",      1,         1, ltfp);
   count += fwrite(mfees,    1,         8, ltfp);
   if(count != (TXADDRLEN+1+8) || ferror(ltfp)) {
      vError(FAIL_FINAL, "b_val(): ltfp IO error");
   }

   /* promote ltran.tmp to *.dat file */
   remove("ltran.dat");
   if (rename("ltran.tmp", "ltran.dat") != 0) {
      vError(FAIL_FINAL, "b_val(): rename ltran.tmp to ltran.dat");
   }
   /* rename if specified */
   if (vfname) {
      remove(vfname);
      if (rename(fname, vfname) != 0) {
         vError(FAIL_FINAL, "b_val(): rename %s to %s", fname, vfname);
      }
      pdebug("b_val(): %s validated to %s\n", fname, vfname);
   } else pdebug("b_val(): %s validated\n", fname);

   ecode = VEOK; /* success */
   pdebug("b_val(): completed in %u ticks.", (word32) (clock() - ticks));

   /* cleanup - error handling */
FAIL_FINAL:
FAIL_TX:
   free(Q2);
FAIL_Q2:
FAIL_IO:
   fclose(fp);
FAIL_FP:
   fclose(ltfp);
   remove("ltran.tmp");
FAIL_LT:
   le_close();
FAIL_LE:

   return ecode;
}  /* end bc_val() */

#ifdef BCVALMAIN

int main(int argc, char **argv)
{
   char *vfname = NULL;
   char *fname = NULL;
   int result = VERROR;

   /* sanity checks */
   if (sizeof(MTX) != sizeof(TXQENTRY)) {
      printf("struct size error: MTX != TXQENTRY\n");
      goto FAIL;
   }

   /* check usage/options */
   if (argc < 2) {
      printf("\n"
         "bval: performs bval(input_file, output_file)\n\n"
         "usage: bval <input_file> [output_file]\n\n");
      goto FAIL;
   }
   if (argc > 2) {
      vfname = argv[2];
   }
   fname = argv[1];

   result = bc_val(fname, vfname);
   printf("b_val(): bc_val(%s, %s) = %d\n\n",
      fname, vfname, result);

FAIL:
   return result;
}  /*end b_val() */

/* end #ifdef BVALMAIN... */
#endif

/* end include guard */
#endif
