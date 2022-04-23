/* bval.c  Block Validator
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * The Mochimo Project System Software
 *
 * Date: 8 January 2018
 *
 * NOTE: Invoked by server.c update() by wait on system()
 *
 * Returns exit code 0 on successful validation,
 *                   1 I/O errors, or
 *                   >= 2 Peerip needs pinklist().
 *
 * Inputs:  argv[1],    rblock.dat, the block to validate
 *          ledger.dat  the ledger of address balances
 *
 * Outputs: ltran.dat  transaction file to post against ledger.dat
 *          exit status 0=valid or non-zero=not valid.
 *          renames argv[1] to "vblock.dat" on good validation.
*/

/* include guard */
#ifndef MOCHIMO_BVAL_C
#define MOCHIMO_BVAL_C


/* system support */
#include <errno.h>

/* extended-c support */
#include "extint.h"     /* integer support */
#include "extlib.h"     /* general support */
#include "extmath.h"    /* 64-bit math support */
#include "extprint.h"   /* print/logging support */

/* crypto support */
#include "crc16.h"

/* algo support */
#include "peach.h"
#include "trigg.h"
#include "wots.h"

/* mochimo support */
#include "config.h"
#include "data.c"
#include "daemon.c"
#include "ledger.c"
#include "txval.c"  /* for mtx */
#include "util.c"

/* Invocation: bval file_to_validate */
int main(int argc, char **argv)
{
   BHEADER bh;             /* fixed length block header */
   static BTRAILER bt;     /* block trailer */
   static TXQENTRY tx;     /* Holds one transaction in the array */
   FILE *fp;               /* to read block file */
   FILE *ltfp;             /* ledger transaction output file ltran.tmp */
   word32 hdrlen, tcount;  /* header length and transaction count */
   int cond;
   static LENTRY src_le;            /* source and change ledger entries */
   word32 total[2];                 /* for 64-bit maths */
   static word8 mroot[HASHLEN];      /* computed Merkel root */
   static word8 bhash[HASHLEN];      /* computed block hash */
   static word8 tx_id[HASHLEN];      /* hash of transaction and signature */
   static word8 prev_tx_id[HASHLEN]; /* to check sort */
   static SHA256_CTX bctx;  /* to hash entire block */
   static SHA256_CTX mctx;  /* to hash transaction array */
   word32 bnum[2], stemp;
   static word32 mfees[2], mreward[2];
   long blocklen;
   int count;
   static word8 do_rename = 1;
   static word8 pk2[WOTSSIGBYTES], msg[32];  /* for WOTS */
   static word32 rnd2[8];  /* for WOTS */
   TXQENTRY *qp1, *qp2, *qlimit;   /* tag mods */
   clock_t ticks;
   static word32 tottrigger[2] = { V23TRIGGER, 0 };
   static word32 v24trigger[2] = { V24TRIGGER, 0 };
   MTX *mtx;
   static word8 addr[TXADDRLEN];  /* for mtx scan 4 */
   int j;  /* mtx */
   static TXQENTRY txs;     /* for mtx sig check */
   TXQENTRY *Q2;        /* tag mods */
   size_t len;
   word32 Tnum = -1;    /* transaction sequence number */
   int ecode;

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

   /* init */
   ticks = clock();
   ecode = VEOK;
   fix_signals();
   close_extra();

   /* check usage/options */
   if (argc < 2) {
      printf("\nusage: bval {rblock.dat | file_to_validate} [-n]\n"
             "  -n no rename, just create ltran.dat\n"
             "This program is spawned from server.c\n\n");
      exit(VERROR);
   } else if (argc > 2 && argv[2][0] == '-') {
      if (argv[2][1] == 'n') do_rename = 0;
   }

   /* enable logging */
   set_output_file(LOGFNAME, "a");

   /* sanity checks */
   if (sizeof(MTX) != sizeof(TXQENTRY)) {
      ecode = perr("bval: bad MTX size");
      goto FAIL;
   }

   /* get global block number, peer ip, etc. */
   if (read_global() != VEOK) {
      ecode = perr("bval: cannot read_global()");
      goto FAIL;
   }
   /* open ledger read-only */
   if (le_open("ledger.dat", "rb") != VEOK) {
      ecode = perr("bval: cannot open ledger.dat");
      goto FAIL;
   }

   /* create ledger transaction temp file */
   ltfp = fopen("ltran.tmp", "wb");
   if (ltfp == NULL) {
      ecode = perrno(errno, "bval: cannot fopen(ltran.tmp, wb)");
      goto FAIL_LE;
   }
   /* open the block to validate */
   fp = fopen(argv[1], "rb");
   if (fp == NULL) {
      ecode = perrno(errno, "bval: cannot fopen(%s, rb)", argv[1]);
      goto FAIL_FP;
   }
   /* read and check regular fixed size block header */
   if (fread(&hdrlen, 1, 4, fp) != 4) {
      ecode = perr("bval: failed on fread(hdrlen)");
      goto FAIL_IO;
   } else if (hdrlen != sizeof(BHEADER)) {
      ecode = perr("bval: failed on hdrlen size: %" P32u " bytes", hdrlen);
      goto FAIL_IO;
   }

   /* compute block file length */
   ecode = fseek(fp, 0, SEEK_END);
   if (ecode) {
      ecode = perrno(ecode, "bval: failed on fseek(END)");
      goto FAIL_IO;
   }
   blocklen = ftell(fp);
   if (blocklen == EOF) {
      ecode = perr("bval: failed on ftell(fp)");
      goto FAIL_IO;
   }

   /* Read block trailer:
    * Check phash, bnum,
    * difficulty, Merkel Root, nonce, solve time, and block hash.
    */
   ecode = fseek(fp, -(sizeof(BTRAILER)), SEEK_END);
   if (ecode) {
      ecode = perrno(ecode, "bval: failed on fseek(END - BTRAILER)");
      goto FAIL_IO;
   } else if (fread(&bt, 1, sizeof(BTRAILER), fp) != sizeof(BTRAILER)) {
      ecode = perr("bval: failed on fread(bt)");
      goto FAIL_IO;
   } else if (cmp64(bt.mfee, Mfee) < 0) {
      ecode = perr("bval: bad mining fee");
      goto FAIL_IO;
   } else if (get32(bt.difficulty) != Difficulty) {
      ecode = perr("bval: difficulty missmatch");
      goto FAIL_IO;
   }

   /* check block times */
   stemp = get32(bt.stime);
   if(stemp <= Time0)  {
      ecode = perrno(ecode, "bval: early block time");
      goto FAIL_IO;
   } else if(stemp > (time(NULL) + BCONFREQ)) {
      ecode = perrno(ecode, "bval: time travel?");
      goto FAIL_IO;
   }
   /* check block number */
   add64(Cblocknum, One, bnum);
   if (memcmp(bnum, bt.bnum, 8) != 0) {
      ecode = perr("bval: bad block number");
      goto FAIL_IO;
   } else if (cmp64(bnum, tottrigger) > 0 && Cblocknum[0] != 0xfe) {
      if ((word32) (stemp - get32(bt.time0)) > BRIDGE) {
         ecode = perr("bval: invalid TOT trigger");
         goto FAIL_IO;
      }
   }
   /* check previous block hash */
   if (memcmp(Cblockhash, bt.phash, HASHLEN) != 0) {
      ecode = perr("bval: previous block hash mismatch");
      goto FAIL_IO;
   }
   /* check transaction count */
   tcount = get32(bt.tcount);
   if(tcount == 0 || tcount > MAXBLTX) {
      perr("bval: bad tcount");
      ecode = VEBAD2; /* pinklist */
      goto FAIL_IO;
   }
   /* check total block length */
   if((hdrlen + sizeof(BTRAILER) + (tcount * sizeof(TXQENTRY)))
         != (word32) blocklen) {
      ecode = perr("bval: invalid block length");
      goto FAIL_IO;
   }

   /* check enforced delay, collect haiku from block */
   if (cmp64(bnum, v24trigger) > 0) {
      /* Boxing Day Anomaly -- Bugfix */
      if (cmp64(bt.bnum, boxingday) == 0) {
         if (memcmp(bt.bhash, boxdayhash, 32) != 0) {
            ecode = perr("bval: bad boxing day anomaly bhash");
            goto FAIL_IO;
         }
      } else if (peach_check(&bt)) {
         ecode = perr("bval: bad POW(peach)");
         goto FAIL_IO;
      }
   } else if (trigg_check(&bt)) {
      ecode = perr("bval: bad POW(trigg)");
      goto FAIL_IO;
   }

   /* Read block header */
   ecode = fseek(fp, 0, SEEK_SET);
   if (ecode) {
      ecode = perrno(ecode, "bval: failed on fseek(SET)");
      goto FAIL_IO;
   } else if (fread(&bh, 1, hdrlen, fp) != hdrlen) {
      ecode = perr("bval: failed on fread(bh)");
      goto FAIL_IO;
   }
   /* check mining reward/address */
   get_mreward(mreward, bnum);
   if (memcmp(bh.mreward, mreward, 8) != 0) {
      ecode = perr("bval: bad mining reward");
      goto FAIL_IO;
   } else if (ADDR_HAS_TAG(bh.maddr)) {
      ecode = perr("bval: maddr has tag");
      goto FAIL_IO;
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
      ecode = perr("bval: failed to malloc(Q2): %zu bytes", len);
      goto FAIL_Q2;
   }

   /* Validate each transaction */
   for (Tnum = 0; Tnum < tcount; Tnum++) {
      if (Tnum >= MAXBLTX) {
         ecode = perr("bval: too many transactions");
         goto FAIL_TX;
      } else if (fread(&tx, 1, sizeof(TXQENTRY), fp) != sizeof(TXQENTRY)) {
         ecode = perr("bval: failed on fread(TX): TX#%" P32u, Tnum);
         goto FAIL_TX;
      } else if (cmp64(tx.tx_fee, Mfee) < 0) {
         ecode = perr("bval: bad tx_fee: TX#%" P32u, Tnum);
         goto FAIL_TX;
      } else if (memcmp(tx.src_addr, tx.chg_addr, TXADDRLEN) == 0) {
         ecode = perr("bval: (src == chg): TX#%" P32u, Tnum);
         goto FAIL_TX;
      } else if (!TX_IS_MTX(&tx)) {
         if (memcmp(tx.src_addr, tx.dst_addr, TXADDRLEN) == 0) {
            ecode = perr("bval: (src == dst): TX#%" P32u, Tnum);
            goto FAIL_TX;
         }
      }

      /* running block/merkel hash */
      sha256_update(&bctx, &tx, sizeof(TXQENTRY));
      sha256_update(&mctx, &tx, sizeof(TXQENTRY));
      /* tx_id is hash of tx.src_addr */
      sha256(tx.src_addr, TXADDRLEN, tx_id);
      if (memcmp(tx_id, tx.tx_id, HASHLEN) != 0) {
         ecode = perr("bval: bad tx_id: TX#%" P32u, Tnum);
         goto FAIL_TX;
      }

      /* Check that tx_id is sorted. */
      if (Tnum != 0) {
         cond = memcmp(tx_id, prev_tx_id, HASHLEN);
         if (cond <= 0) {
            ecode = cond
               ? perr("bval: unsorted tx_id: TX#%" P32u, Tnum)
               : perr("bval: duplicate tx_id: TX#%" P32u, Tnum);
            goto FAIL_TX;
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
         perr("bval: WOTS signature failed: TX#%" P32u, Tnum);
         ecode = VEBAD; /* pinklist */
         goto FAIL_TX;
      }

      /* look up source address in ledger */
      if (le_find(tx.src_addr, &src_le, NULL, TXADDRLEN) == 0) {
         ecode = perr("bval: src_addr not in ledger: TX#%" P32u, Tnum);
         goto FAIL_TX;
      }

      total[0] = total[1] = 0;
      /* use add64() to check for carry out, total, and fees */
      cond =  add64(tx.send_total, tx.change_total, total);
      cond += add64(tx.tx_fee, total, total);
      if (cond) {
         ecode = perr("bval: total overflow: TX#%" P32u, Tnum);
         goto FAIL_TX;
      } else if (cmp64(src_le.balance, total) != 0) {
         ecode = perr("bval: bad transaction total: TX#%" P32u, Tnum);
         goto FAIL_TX;
      } else if (add64(mfees, tx.tx_fee, mfees)) {
         ecode = perr("bval: mfees overflow: TX#%" P32u, Tnum);
         goto FAIL_TX;
      }
      /* check mtx/tag_valid */
      if (!TX_IS_MTX(&tx)) {
         if(tag_valid(tx.src_addr, tx.chg_addr, tx.dst_addr, bt.bnum)) {
            ecode = perr("bval: tag not valid: TX#%" P32u, Tnum);
            goto FAIL_TX;
         }
      } else if(mtx_val((MTX *) &tx, Mfee)) {
         ecode = perr("bval: bad mtx_val: TX#%" P32u, Tnum);
         goto FAIL_TX;
      }

      /* copy TX to tag queue */
      memcpy(&Q2[Tnum], &tx, sizeof(TXQENTRY));
   }  /* end for Tnum */

   /* finalize Merkel Root - phash, bnum, mfee, tcount, time0, difficulty */
   if (NEWYEAR(bt.bnum)) sha256_update(&mctx, &bt, (HASHLEN+8+8+4+4+4));
   sha256_final(&mctx, mroot);
   if (memcmp(bt.mroot, mroot, HASHLEN) != 0) {
      perr("bval: bad merkel root");
      ecode = VEBAD2; /* pinklist */
      goto FAIL_FINAL;
   }

   /* finalize block hash - Block trailer (- block hash) */
   sha256_update(&bctx, &bt, sizeof(BTRAILER) - HASHLEN);
   sha256_final(&bctx, bhash);
   if (memcmp(bt.bhash, bhash, HASHLEN) != 0) {
      ecode = perr("bval: bad block hash");
      goto FAIL_FINAL;
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
   for(Tnum = 0, qp1 = Q2; qp1 < qlimit; qp1++, Tnum++) {
      /* Re-do all the maths again... */
      total[0] = total[1] = 0;
      cond =  add64(qp1->send_total, qp1->change_total, total);
      cond += add64(qp1->tx_fee, total, total);
      if (cond) {
         ecode = perr("bval: scan3 total overflow");
         goto FAIL_FINAL;
      }

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
   }  /* end for Tnum -- scan 3 */
   if(Tnum != tcount)  {
      ecode = perr("bval: scan3 tcount mismatch: %" P32u, Tnum);
      goto FAIL_FINAL;
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
            if (count != 3) {
               ecode = perr("bval: bad I/O dst-->chg write");
               goto FAIL_FINAL;
            }
            continue;  /* next dst[j] */
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
         if (count != 3) {
            ecode = perr("bval: bad I/O scan 4");
            goto FAIL_FINAL;
         }
      }  /* end for j */
   }  /* end for qp1 */
   /* end mtx scan 4 */

   /* Create a transaction amount = mreward + mfees
    * address = bh.maddr
    */
   if (add64(mfees, mreward, mfees)) {
      ecode = perr("bval: mfees overflow");
      goto FAIL_FINAL;
   }
   /* Make ledger tran to add to or create mining address.
    * '...Money from nothing...'
    */
   count =  fwrite(bh.maddr, 1, TXADDRLEN, ltfp);
   count += fwrite("A",      1,         1, ltfp);
   count += fwrite(mfees,    1,         8, ltfp);
   if(count != (TXADDRLEN+1+8) || ferror(ltfp)) {
      ecode = perr("bval: ltfp IO error");
      goto FAIL_FINAL;
   }

   /* promote ltran tmp to dat file */
   remove("ltran.dat");
   rename("ltran.tmp", "ltran.dat");
   /* rename if specified */
   if (do_rename) {
      remove("vblock.dat");
      rename(argv[1], "vblock.dat");
      pdebug("bval: %s validated to vblock.data\n", argv[1]);
   } else pdebug("bval: %s validated\n", argv[1]);
   ecode = VEOK; /* success */

   pdebug("bval: completed in %u ticks.", (word32) (clock() - ticks));

   /* cleanup - error handling */
FAIL_FINAL:
FAIL_TX:
   free(Q2);
   Q2 = NULL;
FAIL_Q2:
FAIL_IO:
   fclose(fp);
FAIL_FP:
   fclose(ltfp);
   remove("ltran.tmp");
FAIL_LE:
   le_close();
FAIL:
   /* remove input file on failure */
   if (ecode && argv[1]) remove(argv[1]);

   return ecode;
}  /* end main() */

/* end include guard */
#endif
