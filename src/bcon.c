/* bcon.c  Block Constructor
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * The Mochimo Project System Software
 *
 * Date: 10 January 2018
 *
 * NOTE: Invoked by server.c by fork() and execl()
 *
 * Inputs:  argv[1],    txclean.dat
 *
 * Outputs: argv[2]     candidate block cblock.dat
 *          exit status 0=block make, or non-zero=no block.
*/

/* include guard */
#ifndef MOCHIMO_BCON_C
#define MOCHIMO_BCON_C


/* system support */
#include <errno.h>

/* extended-c support */
#include "extlib.h"     /* general support */
#include "extmath.h"    /* 64-bit math support */
#include "extprint.h"   /* print/logging support */

/* crypto support */
#include "crc16.h"
#include "sha256.h"

/* mochimo support */
#include "config.h"
#include "data.c"
#include "daemon.c"
#include "sort.c"
#include "util.c"

/*
 * Clean-up on SIGTERM
 */
void sigterm2(int sig)
{
   unlink("cblock.tmp");
   unlink("cblock.dat");
   unlink("bctx.dat");
   pdebug("sigterm() received signal %i", sig);
   exit(1);
}

/* Invocation: bcon txclean.dat cblock.dat */
int main(int argc, char **argv)
{
   word32 Tnum = -1;       /* transaction sequence number */
   static TXQENTRY tx;     /* Holds one transaction in the array */
   FILE *fp;               /* to read txclean.dat file */
   FILE *fpout;            /* for cblock.dat */
   word32 bnum[2];         /* new block num */
   SHA256_CTX mctx;     /* to hash transaction array */
   SHA256_CTX bctx;     /* to hash entire block */
   static BHEADER bh;   /* the minimal length block header */
   static BTRAILER bt;  /* block trailers are fixed length */
   word32 *idx;
   word8 prev_tx_id[HASHLEN];  /* to check for duplicate transactions */
   int cond;
   word32 ntx;
   static word32 mreward[2];
   clock_t ticks;
   int ecode;

   /* init */
   ticks = clock();
   ecode = VEOK;
   fix_signals();
   signal(SIGTERM, sigterm2);  /* server() may kill us. */

   /* check usage/options */
   if (argc != 3) {
      printf("\nusage: bcon txclean.dat cblock.dat\n"
             "This program is spawned from server.c\n\n");
      exit(VERROR);
   }

   /* enable logging */
   set_output_file(LOGFNAME, "a");

   /* get global data, mining address and build sorted Txidx[]... */
   if (read_global() != VEOK) {
      ecode = perr("bcon: failed to read_global()");
      goto FAIL;
   } else if (read_data(Maddr, TXADDRLEN, "maddr.dat") != TXADDRLEN) {
      ecode = perr("bcon: failed to read_data(maddr.dat)");
      goto FAIL;
   } else if (sorttx(argv[1]) != VEOK) {
      ecode = perr("bcon: bad sorttx()");
      goto FAIL;
   }

   /* re-open the clean TX queue (txclean.dat) to read */
   fp = fopen(argv[1], "rb");
   if (fp == NULL) {
      ecode = perrno(errno, "bcon: failed to fopen(%s)", argv[1]);
      goto FAIL_FP;
   }

   /* create cblock.dat */
   fpout = fopen("cblock.tmp", "wb");
   if (fpout == NULL) {
      ecode = perrno(errno, "bcon: failed to fopen(cblock.tmp, wb)");
      goto FAIL_FPOUT;
   }

   /* compute new block number, mining reward */
   add64(Cblocknum, One, bnum);
   get_mreward(mreward, bnum);

   /* prepare new block header... */
   put32(bh.hdrlen, sizeof(bh));
   memcpy(bh.maddr, Maddr, TXADDRLEN);
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
   if (fwrite(&bh, 1, sizeof(bh), fpout) != sizeof(bh)) {
      ecode = perr("bcon: failed to fwrite(bh)");
      goto FAIL_IO;
   }

   ntx = 0;
   idx = Txidx;
   /* Read transactions from txclean.dat in sort order using Txidx[] */
   for (Tnum = 0; Tnum < Ntx && ntx < MAXBLTX; Tnum++, ntx++, idx++) {
      if (Tnum != 0) {
         cond = memcmp(&Tx_ids[*idx * HASHLEN], prev_tx_id, HASHLEN);
         if (cond < 0) {
            ecode = perr("bcon: txclean sort error: TX#%" P32u, Tnum);
            goto FAIL_IO;
         } else if(cond == 0) continue;  /* ignore duplicate transaction */
      }
      /* remember tx_id for next iteration */
      memcpy(prev_tx_id, &Tx_ids[*idx * HASHLEN], HASHLEN);

      /* seek to and read TXQENTRY */
      ecode = fseek(fp, *idx * sizeof(TXQENTRY), SEEK_SET);
      if (ecode) {
         ecode = perrno(ecode, "bcon: bad fseek(TX): TX#%" P32u, Tnum);
         goto FAIL_IO;
      } else if (fread(&tx, sizeof(TXQENTRY), 1, fp) != 1) {
         ecode = perr("bcon: bad fread(TX): TX#%" P32u, Tnum);
         goto FAIL_IO;
      }

      /* add transaction to block hash and merkel array */
      sha256_update(&bctx, &tx, sizeof(TXQENTRY));
      sha256_update(&mctx, &tx, sizeof(TXQENTRY));

      /* write transaction to block */
      if (fwrite(&tx, sizeof(TXQENTRY), 1, fpout) != 1) {
         ecode = perr("bcon: bad fwrite(TX): TX#%" P32u, Tnum);
         goto FAIL_IO;
      }
   }  /* end for Tnum */

   /* Put tran count in trailer */
   if (ntx) put32(bt.tcount, ntx);
   else {
      ecode = perr("bcon: no good transactions");
      goto FAIL_IO;
   }

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
      ecode = perr("bcon: failed to fwrite(bt)");
      goto FAIL_IO;
   }

   /* save bctx to disk for miner */
   remove("bctx.dat");
   if (write_data(&bctx, sizeof(bctx), "bctx.dat") != sizeof(bctx)) {
      ecode = perr("bcon: failed to write_data(bctx)");
      goto FAIL_IO;
   }

   remove(argv[2]);
   if(rename("cblock.tmp", argv[2])) {
      ecode = perrno(errno, "bcon: failed to move cblock.tmp to %s", argv[2]);
      goto FAIL_IO;
   }

   ecode = VEOK; /* success */

   pdebug("bcon: completed in %u ticks.", (word32) (clock() - ticks));

   /* cleanup - error handling */
FAIL_IO:
   fclose(fpout);
FAIL_FPOUT:
   fclose(fp);
FAIL_FP:
   /* sorttx() allocated these two */
   if (Tx_ids) free(Tx_ids);
   if (Txidx) free(Txidx);
FAIL:

   return ecode;
}  /* end main() */

/* end include guard */
#endif
