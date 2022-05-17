/**
 * @private
 * @headerfile bcon.h <bcon.h>
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_BCON_C
#define MOCHIMO_BCON_C


#include "bcon.h"

/* internal support */
#include "util.h"
#include "sort.h"
#include "global.h"

/* external support */
#include <string.h>
#include "sha256.h"
#include "extmath.h"
#include "extlib.h"

/**
 * Generate a pseudo-block with bnum = Cblocknum + 1. Uses node state
 * (Cblockhash, Cblocknum, Time0, and Difficulty) to generate block data.
 * @param output Filename of output block (typically "pblock.dat")
 * @returns VEOK on success, else error code
*/
int pseudo(char *output)
{
   static const word32 pseudo_hdrlen = 4;

   SHA256_CTX ctx;
   BTRAILER bt;
   clock_t ticks;
   int ecode;
   FILE *fp;

   /* init */
   ticks = clock();

   pdebug("pseudo(): generating pseudo-block at %s...", output);

   /* open pseudo-block file and write hdrlen */
   fp = fopen("pblock.dat", "wb");
   if (fp == NULL) mErrno(FAIL, "pseudo(): failed to fopen(%s)", output);
   if (fwrite(&pseudo_hdrlen, 4, 1, fp) != 1) {
      mError(FAIL_IO, "pseudo(): failed to fwrite(pseudo_hdrlen)");
   }

   /* fill block trailer with appropriate pseudo-data */
   memset(&bt, 0, sizeof(bt));
   memcpy(bt.phash, Cblockhash, HASHLEN);
   add64(Cblocknum, One, bt.bnum);
   put32(bt.time0, Time0);
   put32(bt.difficulty, Difficulty);
   put32(bt.stime, Time0 + BRIDGE);

   /* compute pseudo-block hash directly into block trailer */
   sha256_init(&ctx);
   sha256_update(&ctx, &pseudo_hdrlen, 4);
   sha256_update(&ctx, &bt, sizeof(bt) - HASHLEN);
   sha256_final(&ctx, bt.bhash);

   /* write block trailer to pseudo-block file */
   if (fwrite(&bt, sizeof(bt), 1, fp) != 1) {
      mError(FAIL_IO, "pseudo(): failed to fwrite(bt)");
   }

   pdebug("pseudo(): completed in %gs", diffclocktime(clock(), ticks));

   /* success */
   ecode = VEOK;

   /* cleanup / error handling */
FAIL_IO:
   fclose(fp);
FAIL:

   /* remove pblock on failure */
   if (ecode) remove("pblock.dat");

   return ecode;
}  /* end pseudo() */

/**
 * Generate a neogenesis block.
 * Uses input bc file (0x..ff) to create a output neogen-bc file (0x..00).
 * Requires Cblocknum to equal bnum of input block.
 * @param input Filename of input block (matching bnum 0x..ff)
 * @param output Filename of output block (typically "ngblock.dat")
 * @returns VEOK on success, else VERROR
*/
int neogen(char *input, char *output)
{
   SHA256_CTX bctx;     /* (entire) block hash */
   BTRAILER bt, nbt;    /* input and output block trailers */
   FILE *nfp, *lfp;
   clock_t ticks;
   size_t total, count; /* size counters */
   long llen;           /* ledger length */
   word32 hdrlen;       /* header length for neo block */
   word8 neobnum[8];
   word8 buff[BUFSIZ];
   int ecode;

   /* init */
   ticks = clock();

   pdebug("neogen(): generating neogenesis-block...");

   /* read and check trailer from 0x..ff block */
   if (readtrailer(&bt, input) != VEOK) {
      mError(FAIL, "neogen(): failed to read_trailer()");
   } else if (bt.bnum[0] != 0xff) {
      mError(FAIL, "neogen(): bad modulus on bt.bnum");
   } else if(cmp64(bt.bnum, Cblocknum) != 0) {
      mError(FAIL, "neogen(): bt.bnum != Cblocknum");
   }

   /* calculate neogensis block number */
   add64(Cblocknum, One, neobnum);
   if (neobnum[0] != 0) {
      mError(FAIL, "neogen(): bad modulus on Cblocknum");
   }

   /* open ledger read-only */
   lfp = fopen("ledger.dat", "rb");
   if (lfp == NULL) mErrno(FAIL, "neogen(): failed to fopen(ledger.dat)");
   /* fseek() to compute ledger length and check */
   if (fseek(lfp, 0, SEEK_END) != 0) {
      mErrno(FAIL_IO, "neogen(): failed to fseek(END)");
   }
   llen = ftell(lfp);
   if (llen == EOF) mErrno(FAIL_IO, "neogen(): failed to ftell(lfp)");
   if (llen == 0 || (llen % sizeof(LENTRY)) != 0) {
      mError(FAIL_IO, "neogen(): invalid ledger length: %ld", llen);
   }

   /* open neogenesis output file for writing */
   nfp = fopen(output, "wb");
   if(nfp == NULL) {
      mErrno(FAIL_IO, "neogen(): failed to fopen(%s)", output);
   }

   /* Add length of ledger.dat to length of header length field. */
   hdrlen = (word32) llen + 4;
   /* Begin the Neo-Genesis block by writing the header length to it. */
   if (fwrite(&hdrlen, 4, 1, nfp) != 1) {
      mError(FAIL_IO2, "neogen(): failed to fwrite(hdrlen)");
   }

   sha256_init(&bctx);  /* begin entire block hash */
   sha256_update(&bctx, &hdrlen, 4); /* ... with the header length field. */

   /* Cue ledger.dat to beginning and copy it to neo-gen block
    * header whilst hashing it into bctx.
    */
   if (fseek(lfp, 0, SEEK_SET) != 0) {
      mErrno(FAIL_IO2, "neogen(): failed to fseek(lfp, SET)");
   }
   for (total = 0; (count = fread(buff, 1, BUFSIZ, lfp)); total += count) {
      sha256_update(&bctx, buff, count);
      if (fwrite(buff, count, 1, nfp) != 1) {
         mError(FAIL_IO2, "neogen(): failed to fwrite(buff)");
      }
   }
   /* check that everything got copied, and no file errors */
   if (ferror(lfp)) mErrno(FAIL_IO2, "neogen(): ferror(lfp)");
   if (total != (size_t) llen) {
      mError(FAIL_IO2, "neogen(): failed to copy all data");
   }

   /* Fix-up block trailer and write to neogenesis-block */
   memset(&nbt, 0, sizeof(nbt));  /* first clear trailer */
   memcpy(nbt.phash, bt.bhash, HASHLEN);
   put64(nbt.bnum, neobnum);
   put32(nbt.stime, get32(bt.stime));
   put32(nbt.time0, get32(bt.time0));
   put32(nbt.difficulty, get32(bt.difficulty));
   sha256_update(&bctx, &nbt, sizeof(nbt) - HASHLEN);
   sha256_final(&bctx, nbt.bhash);
   if (fwrite(&nbt, sizeof(BTRAILER), 1, nfp) != 1) {
      mError(FAIL_IO2, "neogen(): failed to fwrite(nbt)");
   } else if (ferror(nfp)) mErrno(FAIL_IO2, "neogen(): ferror(nfp)");

   pdebug("neogen(): completed in %gs", diffclocktime(clock(), ticks));

   /* success */
   ecode = VEOK;

   /* cleanup / error handling */
FAIL_IO2:
   fclose(nfp);
FAIL_IO:
   fclose(lfp);
FAIL:

   /* remove output file on failure */
   if (ecode) remove(output);

   return ecode;
}  /* end neogen() */

/**
 * Construct a candidate block from "txclean.dat". Uses node state
 * (Cblocknum, Cblockhash, Mfee, Difficulty, Time0) for block data.
 * @param fname Candidate block (output) file name: "cblock.dat"
 * @returns VEOK on success, else error code
*/
int b_con(char *fname)
{
   SHA256_CTX bctx, mctx;  /* (entire) block hash and merkel array */
   TXQENTRY tx;            /* Holds one transaction in the array */
   BTRAILER bt;            /* block trailers are fixed length */
   BHEADER bh;             /* the minimal length block header */
   FILE *fp, *fpout;       /* to read txclean file and write cblock */
   clock_t ticks;
   word32 mreward[2];      /* mining reward */
   word32 bnum[2];         /* new block num */
   word32 *idx, ntx, num;
   word8 maddr[TXADDRLEN]; /* to read mining address for block */
   word8 prev_tx_id[HASHLEN];  /* to check for duplicate transactions */
   int cond, ecode;

   /* init */
   ticks = clock();

   pdebug("b_con(): constructing candidate-block...");

   /* get mining address and build sorted Txidx[]... */
   if (read_data(maddr, TXADDRLEN, "maddr.dat") != TXADDRLEN) {
      mError(FAIL, "b_con(): failed to read_data(maddr)");
   } else if (sorttx("txclean.dat") != VEOK) {
      mError(FAIL, "b_con(): bad sorttx(txclean.dat)");
   }

   /* re-open the clean TX queue (txclean.dat) to read */
   fp = fopen("txclean.dat", "rb");
   if (fp == NULL) {
      mErrno(FAIL_IN, "b_con(): failed to fopen(txclean.dat)");
   }

   /* create cblock.tmp */
   fpout = fopen("cblock.tmp", "wb");
   if (fpout == NULL) {
      mErrno(FAIL_OUT, "b_con(): failed to fopen(cblock.tmp)");
   }

   /* compute new block number, mining reward */
   add64(Cblocknum, One, bnum);
   get_mreward(mreward, bnum);

   /* prepare new block header... */
   put32(bh.hdrlen, sizeof(bh));
   memcpy(bh.maddr, maddr, TXADDRLEN);
   put64(bh.mreward, mreward);
   /* ... and trailer */
   memcpy(bt.phash, Cblockhash, HASHLEN);
   put64(bt.bnum, bnum);
   put64(bt.mfee, Mfee);
   put32(bt.difficulty, Difficulty);
   put32(bt.time0, Time0);

   /* prepare hashing states */
   sha256_init(&bctx);   /* begin entire block hash */
   sha256_update(&bctx, &bh, sizeof(bh));  /* ... with the header */
   if (!NEWYEAR(bt.bnum)) sha256_init(&mctx); /* begin Merkel Array hash */
   else memcpy(&mctx, &bctx, sizeof(mctx));  /* ... or copy bctx state */

   /* write header to disk */
   if (fwrite(&bh, sizeof(bh), 1, fpout) != 1) {
      mError(FAIL_IO, "b_con(): failed to fwrite(bh)");
   }

   ntx = 0;
   idx = Txidx;
   /* Read transactions from txclean.dat in sort order using Txidx[] */
   for (num = 0; num < Ntx && ntx < MAXBLTX; num++, idx++) {
      if (num != 0) {
         cond = memcmp(&Tx_ids[*idx * HASHLEN], prev_tx_id, HASHLEN);
         if (cond <= 0) {
            if (cond == 0) continue;  /* ignore duplicate transaction */
            mError(FAIL_IO, "b_con(): txclean sort error: TX#%" P32u, num);
         }
      }
      /* remember tx_id for next iteration */
      memcpy(prev_tx_id, &Tx_ids[*idx * HASHLEN], HASHLEN);
      /* seek to and read TXQENTRY */
      if (fseek(fp, *idx * sizeof(TXQENTRY), SEEK_SET) != 0) {
         mErrno(FAIL_IO, "b_con(): bad fseek(TX): TX#%" P32u, num);
      } else if (fread(&tx, sizeof(TXQENTRY), 1, fp) != 1) {
         mError(FAIL_IO, "b_con(): bad fread(TX): TX#%" P32u, num);
      }
      /* add transaction to block hash and merkel array */
      sha256_update(&bctx, &tx, sizeof(TXQENTRY));
      sha256_update(&mctx, &tx, sizeof(TXQENTRY));
      /* write transaction to block */
      if (fwrite(&tx, sizeof(TXQENTRY), 1, fpout) != 1) {
         mError(FAIL_IO, "b_con(): bad fwrite(TX): TX#%" P32u, num);
      }
      /* increment actual transactions for block */
      ntx++;
   }  /* end for num */

   /* Put tran count in trailer */
   if (ntx) put32(bt.tcount, ntx);
   else mError(FAIL_IO, "b_con(): no good transactions");

   /* finalize merkel array - (phash+bnum+mfee+tcount+time0+difficulty)*/
   if (NEWYEAR(bt.bnum)) sha256_update(&mctx, &bt, (HASHLEN+8+8+4+4+4));
   sha256_final(&mctx, bt.mroot);  /* put the Merkel root in trailer */
   /* Hash in the trailer leaving out: nonce[32], stime[4], and bhash[32] */
   sha256_update(&bctx, &bt, (sizeof(BTRAILER) - (2 * HASHLEN) - 4));

   /* Let the miner put final hash[] and stime[] at end of BTRAILER struct
    * with the calls to sha256_final() and put32().
    * Gift bctx to miner using write_data().
    */

   /* write trailer to disk */
   if (fwrite(&bt, sizeof(BTRAILER), 1, fpout) != 1) {
      mError(FAIL_IO, "b_con(): failed to fwrite(bt)");
   }

   /* save bctx to disk for miner */
   remove("bctx.dat");
   if (write_data(&bctx, sizeof(bctx), "bctx.dat") != sizeof(bctx)) {
      mError(FAIL_IO, "b_con(): failed to write_data(bctx)");
   }

   remove(fname);
   if (rename("cblock.tmp", fname)) {
      mErrno(FAIL_IO, "b_con(): rename cblock.tmp to %s", fname);
   }

   pdebug("b_con(): completed in %gs", diffclocktime(clock(), ticks));

   /* success */
   ecode = VEOK;

   /* cleanup / error handling */
FAIL_IO:
   fclose(fpout);
FAIL_OUT:
   fclose(fp);
FAIL_IN:
   /* sorttx() allocated these two */
   free(Tx_ids);
   free(Txidx);
   Tx_ids = NULL;
   Txidx = NULL;
FAIL:

   return ecode;
}  /* end b_con() */

/* end include guard */
#endif
