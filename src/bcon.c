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
#include "tfile.h"
#include "sort.h"
#include "global.h"
#include "error.h"

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
   FILE *fp;

   /* init */
   ticks = clock();

   pdebug("generating pseudo-block at %s...", output);

   /* open pseudo-block file and write hdrlen */
   fp = fopen(output, "wb");
   if (fp == NULL) {
      perrno("failed to fopen(%s)", output);
      return VERROR;
   }
   if (fwrite(&pseudo_hdrlen, 4, 1, fp) != 1) {
      perr("failed to fwrite(pseudo_hdrlen)");
      goto CLEANUP;
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
      perr("failed to fwrite(bt)");
      goto CLEANUP;
   }

   /* cleanup */
   fclose(fp);

   pdebug("completed in %gs", diffclocktime(ticks));

   /* success */
   return VEOK;

   /* cleanup / error handling */
CLEANUP:
   fclose(fp);
   remove(output);
   return VERROR;
}  /* end pseudo() */

/**
 * Generate a neogenesis block.
 * Uses input bc file (0x..ff) to create a output neogen-bc file (0x..00).
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

   /* init */
   ticks = clock();

   pdebug("generating neogenesis-block...");

   /* read and check trailer from 0x..ff block */
   if (readtrailer(&bt, input) != VEOK) {
      perr("failed to read_trailer()");
      return VERROR;
   } else if (bt.bnum[0] != 0xff) {
      perr("bad modulus on bt.bnum");
      return VERROR;
   }

   /* calculate neogensis block number */
   add64(bt.bnum, One, neobnum);
   if (neobnum[0] != 0) {
      perr("bad modulus on Cblocknum");
      return VERROR;
   }

   /* open ledger read-only */
   lfp = fopen("ledger.dat", "rb");
   if (lfp == NULL) {
      perrno("failed to fopen(ledger.dat)");
      return VERROR;
   }
   /* fseek() to compute ledger length and check */
   if (fseek(lfp, 0, SEEK_END) != 0) {
      perrno("failed to fseek(END)");
      goto CLEANUP_LE;
   }
   llen = ftell(lfp);
   if (llen == EOF) {
      perrno("failed to ftell(lfp)");
      goto CLEANUP_LE;
   }
   if (llen == 0 || (llen % sizeof(LENTRY)) != 0) {
      perr("invalid ledger length: %ld", llen);
      goto CLEANUP_LE;
   }

   /* open neogenesis output file for writing */
   nfp = fopen(output, "wb");
   if(nfp == NULL) {
      perrno("failed to fopen(%s)", output);
      goto CLEANUP_LE;
   }

   /* Add length of ledger.dat to length of header length field. */
   hdrlen = (word32) llen + 4;
   /* Begin the Neo-Genesis block by writing the header length to it. */
   if (fwrite(&hdrlen, 4, 1, nfp) != 1) {
      perr("failed to fwrite(hdrlen)");
      goto CLEANUP_NEO;
   }

   sha256_init(&bctx);  /* begin entire block hash */
   sha256_update(&bctx, &hdrlen, 4); /* ... with the header length field. */

   /* Cue ledger.dat to beginning and copy it to neo-gen block
    * header whilst hashing it into bctx.
    */
   if (fseek(lfp, 0, SEEK_SET) != 0) {
      perrno("failed to fseek(lfp, SET)");
      goto CLEANUP_NEO;
   }
   for (total = 0; (count = fread(buff, 1, BUFSIZ, lfp)); total += count) {
      sha256_update(&bctx, buff, count);
      if (fwrite(buff, count, 1, nfp) != 1) {
         perr("failed to fwrite(buff)");
         goto CLEANUP_NEO;
      }
   }
   /* check that everything got copied, and no file errors */
   if (ferror(lfp)) {
      perrno("ferror(lfp)");
      goto CLEANUP_NEO;
   }
   if (total != (size_t) llen) {
      perr("failed to copy all data");
      goto CLEANUP_NEO;
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
      perr("failed to fwrite(nbt)");
      goto CLEANUP_NEO;
   } else if (ferror(nfp)) {
      perrno("ferror(nfp)");
      goto CLEANUP_NEO;
   }

   /* cleanup */
   fclose(nfp);
   fclose(lfp);

   pdebug("completed in %gs", diffclocktime(ticks));

   /* success */
   return VEOK;

   /* cleanup / error handling */
CLEANUP_NEO:
   fclose(nfp);
   remove(output);
CLEANUP_LE:
   fclose(lfp);
   return VERROR;
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
   int cond;

   /* init */
   ticks = clock();

   pdebug("constructing candidate-block...");

   /* get mining address and build sorted Txidx[]... */
   if (read_data(maddr, TXADDRLEN, "maddr.dat") != TXADDRLEN) {
      perr("failed to read_data(maddr)");
      return VERROR;
   } else if (sorttx("txclean.dat") != VEOK) {
      perr("bad sorttx(txclean.dat)");
      return VERROR;
   }

   /* re-open the clean TX queue (txclean.dat) to read */
   fp = fopen("txclean.dat", "rb");
   if (fp == NULL) {
      perrno("failed to fopen(txclean.dat)");
      goto CLEANUP;
   }

   /* create cblock.tmp */
   fpout = fopen("cblock.tmp", "wb");
   if (fpout == NULL) {
      perrno("failed to fopen(cblock.tmp)");
      goto CLEANUP_TXC;
   }

   /* compute new block number, mining reward */
   add64(Cblocknum, One, bnum);
   get_mreward(mreward, bnum);

   /* prepare new block header... */
   put32(bh.hdrlen, sizeof(bh));
   memcpy(bh.maddr, maddr, TXADDRLEN);
   put64(bh.mreward, mreward);
   /* ... and trailer */
   memset(&bt, 0, sizeof(bt));
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
      perr("failed to fwrite(bh)");
      goto CLEANUP_CBLK;
   }

   ntx = 0;
   idx = Txidx;
   /* Read transactions from txclean.dat in sort order using Txidx[] */
   for (num = 0; num < Ntx && ntx < MAXBLTX; num++, idx++) {
      if (num != 0) {
         cond = memcmp(&Tx_ids[*idx * HASHLEN], prev_tx_id, HASHLEN);
         if (cond <= 0) {
            if (cond == 0) continue;  /* ignore duplicate transaction */
            perr("txclean sort error: TX#%" P32u, num);
            goto CLEANUP_CBLK;
         }
      }
      /* remember tx_id for next iteration */
      memcpy(prev_tx_id, &Tx_ids[*idx * HASHLEN], HASHLEN);
      /* seek to and read TXQENTRY */
      if (fseek(fp, *idx * sizeof(TXQENTRY), SEEK_SET) != 0) {
         perrno("bad fseek(TX): TX#%" P32u, num);
         goto CLEANUP_CBLK;
      } else if (fread(&tx, sizeof(TXQENTRY), 1, fp) != 1) {
         perr("bad fread(TX): TX#%" P32u, num);
         goto CLEANUP_CBLK;
      }
      /* add transaction to block hash and merkel array */
      sha256_update(&bctx, &tx, sizeof(TXQENTRY));
      sha256_update(&mctx, &tx, sizeof(TXQENTRY));
      /* write transaction to block */
      if (fwrite(&tx, sizeof(TXQENTRY), 1, fpout) != 1) {
         perr("bad fwrite(TX): TX#%" P32u, num);
         goto CLEANUP_CBLK;
      }
      /* increment actual transactions for block */
      ntx++;
   }  /* end for num */

   /* Put tran count in trailer */
   if (ntx) put32(bt.tcount, ntx);
   else {
      perr("no good transactions");
      goto CLEANUP_CBLK;
   }

   /* finalize merkel array - (phash+bnum+mfee+tcount+time0+difficulty)*/
   if (NEWYEAR(bt.bnum)) sha256_update(&mctx, &bt, (HASHLEN+8+8+4+4+4));
   sha256_final(&mctx, bt.mroot);  /* put the Merkel root in trailer */
   /* Hash in the trailer leaving out: nonce[32], stime[4], and bhash[32] */
   sha256_update(&bctx, &bt, (sizeof(BTRAILER) - (2 * HASHLEN) - 4));

   /* Let the miner put final hash[] and stime[] at end of BTRAILER struct
    * with the calls to sha256_final() and put32().
    * Gift bctx to miner using write_data() + rename().
    */

   /* write trailer to disk */
   if (fwrite(&bt, sizeof(BTRAILER), 1, fpout) != 1) {
      perr("failed to fwrite(bt)");
      goto CLEANUP_CBLK;
   }

   /* cleanup */
   fclose(fpout);
   fclose(fp);

   /* move temporary output (*.tmp) to working output (*.dat) */
   remove(fname);
   if (rename("cblock.tmp", fname)) {
      perrno("rename cblock.tmp to %s", fname);
      goto CLEANUP;
   } else if (!Nominer) {
      /* save bctx to disk for miner */
      remove("bctx.tmp");
      remove("bctx.dat");
      if (write_data(&bctx, sizeof(bctx), "bctx.tmp") != sizeof(bctx)) {
         perr("failed to write_data(bctx)");
      } else if (rename("bctx.tmp", "bctx.dat") != 0) {
         perr("rename bctx");
      }
   }

   /* cleanup */
   sorttx_free();

   pdebug("completed in %gs", diffclocktime(ticks));

   /* success */
   return VEOK;

   /* cleanup / error handling */
CLEANUP_CBLK:
   fclose(fpout);
CLEANUP_TXC:
   fclose(fp);
CLEANUP:
   sorttx_free();
   return VERROR;
}  /* end b_con() */

/* end include guard */
#endif
