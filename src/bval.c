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


#include "config.h"
#include "mochimo.h"
#define closesocket(_sd) close(_sd)
char *trigg_check(byte *in, byte d, byte *bnum);
void trigg_expand2(byte *in, char *out);

#define EXCLUDE_NODES   /* exclude Nodes[], ip, and socket data */
#include "data.c"

#include "error.c"
#include "rand.c"
#include "crypto/crc16.c"
#include "add64.c"
#include "util.c"
#include "daemon.c"
#include "ledger.c"

#define EXCLUDE_RESOLVE
#include "tag.c"
#include "algo/peach/peach.c"
#include "mtxval.c"  /* for mtx */

word32 Tnum = -1;    /* transaction sequence number */
char *Bvaldelfname;  /* set == argv[1] to delete input file on failure */
TXQENTRY *Q2;        /* tag mods */


void cleanup(int ecode)
{
   if(Q2 != NULL) free(Q2);
   unlink("ltran.tmp");
   if(Bvaldelfname) unlink(Bvaldelfname);
   if(Trace) plog("cleanup() with ecode %i", ecode);
   exit(1);  /* no pink-list */
}

void drop(char *message)
{
   if(Trace && message)
      plog("bval: drop(): %s TX index = %d", message, Tnum);
   cleanup(3);
}


void baddrop(char *message)
{
   if(Trace && message)
      plog("bval: baddrop(): %s from: %s  TX index = %d",
           message, ntoa((byte *) &Peerip), Tnum);
   /* add Peerip to epoch pink list */
   cleanup(3);  /* put on epink.lst */
}


void bail(char *message)
{
   if(message) error("bval: %s", message);
   cleanup(1);
}


#if ADDR_TAG_LEN != 12
   ADDR_TAG_LEN must be 12 for tag code in bval.c
#endif


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
   static byte mroot[HASHLEN];      /* computed Merkel root */
   static byte bhash[HASHLEN];      /* computed block hash */
   static byte tx_id[HASHLEN];      /* hash of transaction and signature */
   static byte prev_tx_id[HASHLEN]; /* to check sort */
   static SHA256_CTX bctx;  /* to hash entire block */
   static SHA256_CTX mctx;  /* to hash transaction array */
   word32 bnum[2], stemp;
   static word32 mfees[2], mreward[2];
   unsigned long blocklen;
   int count;
   static byte do_rename = 1;
   static byte pk2[WOTSSIGBYTES], message[32], rnd2[32];  /* for WOTS */
   static char *haiku;
   static char haikufull[256];
   word32 now;
   TXQENTRY *qp1, *qp2, *qlimit;   /* tag mods */
   clock_t ticks;
   static word32 tottrigger[2] = { V23TRIGGER, 0 };
   static word32 v24trigger[2] = { V24TRIGGER, 0 };
   MTX *mtx;
   static byte addr[TXADDRLEN];  /* for mtx scan 4 */
   int j;  /* mtx */

   
   
   ticks = clock();
   fix_signals();
   close_extra();

   if(argc < 2) {
      printf("\nusage: bval {rblock.dat | file_to_validate} [-n]\n"
             "  -n no rename, just create ltran.dat\n"
             "This program is spawned from server.c\n\n");
      exit(1);
   }

   if(sizeof(MTX) != sizeof(TXQENTRY)) bail("bad MTX size");

   if(argc > 2 && argv[2][0] == '-') {
      if(argv[2][1] == 'n') do_rename = 0;
   }

   if(strcmp(argv[1], "rblock.dat") == 0) Bvaldelfname = argv[1];
   unlink("vblock.dat");
   unlink("ltran.dat");

   /* get global block number, peer ip, etc. */
   if(read_global() != VEOK)
      bail("Cannot read_global()");

   if(Trace) Logfp = fopen(LOGFNAME, "a");

   /* open ledger read-only */
   if(le_open("ledger.dat", "rb") != VEOK)
      bail("Cannot open ledger.dat");

   /* create ledger transaction temp file */
   ltfp = fopen("ltran.tmp", "wb");
   if(ltfp == NULL) bail("Cannot create ltran.tmp");

   /* open the block to validate */
   fp = fopen(argv[1], "rb");
   if(!fp) {
badread:
      bail("Cannot read input rblock.dat");
   }
   if(fread(&hdrlen, 1, 4, fp) != 4) goto badread;  /* read header length */
   /* regular fixed size block header */
   if(hdrlen != sizeof(BHEADER))
      drop("bad hdrlen");

   /* compute block file length */
   if(fseek(fp, 0, SEEK_END)) goto badread;
   blocklen = ftell(fp);

   /* Read block trailer:
    * Check phash, bnum,
    * difficulty, Merkel Root, nonce, solve time, and block hash.
    */
   if(fseek(fp, -(sizeof(BTRAILER)), SEEK_END)) goto badread;
   if(fread(&bt, 1, sizeof(BTRAILER), fp) != sizeof(BTRAILER))
      drop("bad trailer read");
   if(cmp64(bt.mfee, Mfee) < 0)
      drop("bad mining fee");
   if(get32(bt.difficulty) != Difficulty)
      drop("difficulty mismatch");

   /* Check block times and block number. */
   stemp = get32(bt.stime);
   /* check for early block time */
   if(stemp <= Time0) drop("E");  /* unsigned time here */
   now = time(NULL);
   if(stemp > (now + BCONFREQ)) drop("F");
   add64(Cblocknum, One, bnum);
   if(memcmp(bnum, bt.bnum, 8) != 0) drop("bad block number");
   if(cmp64(bnum, tottrigger) > 0 && Cblocknum[0] != 0xfe) {
      if((word32) (stemp - get32(bt.time0)) > BRIDGE) drop("TOT");
   }

   if(memcmp(Cblockhash, bt.phash, HASHLEN) != 0)
      drop("previous hash mismatch");

   /* check enforced delay, collect haiku from block */
   if(cmp64(bnum, v24trigger) > 0) {
      if(peach(&bt, get32(bt.difficulty), NULL, 1)){
         drop("peach validation failed!");
      }

      trigg_expand2(bt.nonce, haikufull);
      if(!Bgflag) printf("\n%s\n\n", haikufull);
   }
   if(cmp64(bnum, v24trigger) <= 0) {
      if((haiku = trigg_check(bt.mroot, bt.difficulty[0], bt.bnum)) == NULL) {
      drop("trigg_check() failed!");
      }
      if(!Bgflag) printf("\n%s\n\n", haiku);
   }

   /* Read block header */
   if(fseek(fp, 0, SEEK_SET)) goto badread;
   if(fread(&bh, 1, hdrlen, fp) != hdrlen)
      drop("short header read");
   get_mreward(mreward, bnum);
   if(memcmp(bh.mreward, mreward, 8) != 0)
      drop("bad mining reward");
   if(HAS_TAG(bh.maddr))
      drop("bh.maddr has tag!");

   /* fp left at offset of Merkel Block Array--ready to fread() */

   sha256_init(&bctx);   /* begin entire block hash */
   sha256_update(&bctx, (byte *) &bh, hdrlen);  /* ... with the header */

   if(NEWYEAR(bt.bnum)) memcpy(&mctx, &bctx, sizeof(mctx));

   /*
    * Copy transaction count from block trailer and check.
    */
   tcount = get32(bt.tcount);
   if(tcount == 0 || tcount > MAXBLTX)
      baddrop("bad bt.tcount");
   if((hdrlen + sizeof(BTRAILER) + (tcount * sizeof(TXQENTRY))) != blocklen)
      drop("bad block length");

   /* temp TX tag processing queue */
   Q2 = malloc(tcount * sizeof(TXQENTRY));
   if(Q2 == NULL) bail("no memory!");

   /* Now ready to read transactions */
   if(!NEWYEAR(bt.bnum)) sha256_init(&mctx);   /* begin Merkel Array hash */

   /* Validate each transaction */
   for(Tnum = 0; Tnum < tcount; Tnum++) {
      if(Tnum >= MAXBLTX)
         drop("too many TX's");
      if(fread(&tx, 1, sizeof(TXQENTRY), fp) != sizeof(TXQENTRY))
         drop("bad TX read");
      if(memcmp(tx.src_addr, tx.chg_addr, TXADDRLEN) == 0)
         drop("src == chg");
      if(!ismtx(&tx) && memcmp(tx.src_addr, tx.dst_addr, TXADDRLEN) == 0)
         drop("src == dst");

      if(cmp64(tx.tx_fee, Mfee) < 0) drop("tx_fee is bad");

      /* running block hash */
      sha256_update(&bctx, (byte *) &tx, sizeof(TXQENTRY));
      /* running Merkel hash */
      sha256_update(&mctx, (byte *) &tx, sizeof(TXQENTRY));
      /* tx_id is hash of tx.src_add */
      sha256(tx.src_addr, TXADDRLEN, tx_id);
      if(memcmp(tx_id, tx.tx_id, HASHLEN) != 0)
         drop("bad TX_ID");

      /* Check that tx_id is sorted. */
      if(Tnum != 0) {
         cond = memcmp(tx_id, prev_tx_id, HASHLEN);
         if(cond < 0)  drop("TX_ID unsorted");
         if(cond == 0) drop("duplicate TX_ID");
      }
      /* remember this tx_id for next time */
      memcpy(prev_tx_id, tx_id, HASHLEN);

      /* check WTOS signature */
      sha256(tx.src_addr, SIG_HASH_COUNT, message);
      memcpy(rnd2, &tx.src_addr[TXSIGLEN+32], 32);  /* copy WOTS addr[] */
      wots_pk_from_sig(pk2, tx.tx_sig, message, &tx.src_addr[TXSIGLEN],
                       (word32 *) rnd2);
      if(memcmp(pk2, tx.src_addr, TXSIGLEN) != 0)
         baddrop("WOTS signature failed!");

      /* look up source address in ledger */
      if(le_find(tx.src_addr, &src_le, NULL, 0) == FALSE)
         drop("src_addr not in ledger");

      total[0] = total[1] = 0;
      /* use add64() to check for carry out */
      cond =  add64(tx.send_total, tx.change_total, total);
      cond += add64(tx.tx_fee, total, total);
      if(cond) drop("total overflow");

      if(cmp64(src_le.balance, total) != 0)
         drop("bad transaction total");
      if(!ismtx(&tx)) {
         if(tag_valid(tx.src_addr, tx.chg_addr, tx.dst_addr, 0, bt.bnum)
            != VEOK) drop("tag not valid");
      } else {
         if(mtx_val((MTX *) &tx, Mfee) != 0) drop("bad mtx_val()");
      }

      memcpy(&Q2[Tnum], &tx, sizeof(TXQENTRY));  /* copy TX to tag queue */

      if(add64(mfees, tx.tx_fee, mfees)) {
fee_overflow:
         bail("mfees overflow");
      }
   }  /* end for Tnum */
   if(NEWYEAR(bt.bnum))
      /* phash, bnum, mfee, tcount, time0, difficulty */
      sha256_update(&mctx, (byte *) &bt, (HASHLEN+8+8+4+4+4));

   sha256_final(&mctx, mroot);  /* compute Merkel Root */
   if(memcmp(bt.mroot, mroot, HASHLEN) != 0)
      baddrop("bad Merkle root");

   sha256_update(&bctx, (byte *) &bt, sizeof(BTRAILER) - HASHLEN);
   sha256_final(&bctx, bhash);
   if(memcmp(bt.bhash, bhash, HASHLEN) != 0)
      drop("bad block hash");

   /* tag search  Begin ... */
   qlimit = &Q2[tcount];
   for(qp1 = Q2; qp1 < qlimit; qp1++) {
      if(!HAS_TAG(qp1->src_addr)
         || memcmp(ADDR_TAG_PTR(qp1->src_addr), ADDR_TAG_PTR(qp1->chg_addr),
                   ADDR_TAG_LEN) != 0) continue;
      /* Step 2: Start another big-O n squared, nested loop here... */
      for(qp2 = Q2; qp2 < qlimit; qp2++) {
         if(qp1 == qp2) continue;  /* added -trg */
         if(ismtx(qp2)) continue;  /* skip multi-dst's for now */
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
      if(cond) bail("scan3 total overflow");

      /* Write ledger transactions to ltran.tmp for all src and chg,
       * but only non-mtx dst
       * that will have to be sorted, read again, and applied by bup...
       */
      fwrite(qp1->src_addr,  1, TXADDRLEN, ltfp);
      fwrite("-",            1,         1, ltfp);  /* debit src addr */
      fwrite(total,          1,         8, ltfp);
      /* add to or create non-multi dst address */
      if(!ismtx(qp1) && !iszero(qp1->send_total, 8)) {
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

   
   if(Tnum != tcount) bail("scan 3");
   /* mtx tag search  Begin scan 4 ...
    *
    * Write out the multi-dst trans using tag scan logic @
    * that more or less repeats the above big-O n-squared loops, and
    * expands the tags, and copies addresses around.
    */
   for(qp1 = Q2; qp1 < qlimit; qp1++) {
      if(!ismtx(qp1)) continue;  /* only multi-dst's this time */
      mtx = (MTX *) qp1;  /* poor man's union */
      /* For each dst[] tag... */
      for(j = 0; j < 100; j++) {
         if(iszero(mtx->dst[j].tag, ADDR_TAG_LEN)) break; /* end of dst[] */
         memcpy(ADDR_TAG_PTR(addr), mtx->dst[j].tag, ADDR_TAG_LEN);
         /* If dst[j] tag not found, write money back to chg addr. */
         if(tag_find(addr, addr, NULL) != VEOK) {
            count =  fwrite(mtx->chg_addr, TXADDRLEN, 1, ltfp);
            count += fwrite("A", 1, 1, ltfp);
            count += fwrite(mtx->dst[j].amount, 8, 1, ltfp);
            if(count != 3) bail("bad I/O dst-->chg write");
            continue;  /* next dst[j] */
         }
         /* Start another big-O n-squared, nested loop here... scan 5 */
         for(qp2 = Q2; qp2 < qlimit; qp2++) {
            if(qp1 == qp2) continue;
            /* if dst[j] tag == any other src addr tag and chg addr tag,
             * copy other chg addr to dst[] addr.
             */
            if(!HAS_TAG(qp2->src_addr)) continue;
            if(memcmp(ADDR_TAG_PTR(qp2->src_addr),
                      ADDR_TAG_PTR(qp2->chg_addr), ADDR_TAG_LEN) != 0)
                         continue;
            if(memcmp(ADDR_TAG_PTR(qp2->src_addr), ADDR_TAG_PTR(addr), 
                      ADDR_TAG_LEN) == 0) {
                         memcpy(addr, qp2->chg_addr, TXADDRLEN);
                         break;
            }
         }  /* end for qp2 scan 5 */
         /* write out the dst transaction */
         count =  fwrite(addr, TXADDRLEN, 1, ltfp);
         count += fwrite("A", 1, 1, ltfp);
         count += fwrite(mtx->dst[j].amount, 8, 1, ltfp);
         if(count != 3) bail("bad I/O scan 4");
      }  /* end for j */
   }  /* end for qp1 */
   /* end mtx scan 4 */

   /* Create a transaction amount = mreward + mfees
    * address = bh.maddr
    */
   if(add64(mfees, mreward, mfees)) goto fee_overflow;
   /* Make ledger tran to add to or create mining address.
    * '...Money from nothing...'
    */
   count =  fwrite(bh.maddr, 1, TXADDRLEN, ltfp);
   count += fwrite("A",      1,         1, ltfp);
   count += fwrite(mfees,    1,         8, ltfp);
   if(count != (TXADDRLEN+1+8) || ferror(ltfp))
      bail("ltfp I/O error");

   free(Q2);  Q2 = NULL;
   le_close();
   fclose(ltfp);
   fclose(fp);
   rename("ltran.tmp", "ltran.dat");
   unlink("vblock.dat");
   if(do_rename)
      rename(argv[1], "vblock.dat");
   if(Trace)
      plog("bval: block validated to vblock.dat (%u usec.)",
           (word32) (clock() - ticks));
   if(argc > 2) printf("Validated\n");
   return 0;  /* success */
}  /* end main() */
