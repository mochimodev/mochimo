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

#include "extmath.h"    /* 64-bit math support */

#include "config.h"
#include "mochimo.h"
#include "errno.h"
#define closesocket(_sd) close(_sd)

#define EXCLUDE_NODES   /* exclude Nodes[], ip, and socket data */
#include "data.c"

#include "error.c"
#include "crypto/crc16.c"
#include "rand.c"
#include "util.c"
#include "daemon.c"

#include "sorttx.c"

word32 Tnum = -1;  /* transaction sequence number */

void bail(char *message)
{
   if(message) error("bcon: bailing out: %s (%d)", message, Tnum);
   exit(1);
}

/*
 * Clean-up on SIGTERM
 */
void sigterm2(int sig)
{
   unlink("cblock.tmp");
   unlink("cblock.dat");
   unlink("bctx.dat");
   if(Trace) plog("sigterm() received signal %i", sig);
   exit(1);
}


/* Invocation: bcon txclean.dat cblock.dat */
int main(int argc, char **argv)
{
   static TXQENTRY tx;     /* Holds one transaction in the array */
   FILE *fp;               /* to read txclean.dat file */
   FILE *fpout;            /* for cblock.dat */
   word32 bnum[2];         /* new block num */
   int count;
   SHA256_CTX mctx;     /* to hash transaction array */
   SHA256_CTX bctx;     /* to hash entire block */
   static BHEADER bh;   /* the minimal length block header */
   static BTRAILER bt;  /* block trailers are fixed length */
   word32 *idx;
   byte prev_tx_id[HASHLEN];  /* to check for duplicate transactions */
   int cond;
   word32 ntx;
   static word32 mreward[2];

   fix_signals();
   signal(SIGTERM, sigterm2);  /* server() may kill us. */

   if(argc != 3) {
      printf("\nusage: bcon txclean.dat cblock.dat\n"
             "This program is spawned from server.c\n\n");
      exit(1);
   }

   unlink("bctx.dat");

   /* get global block number, peer ip, etc. */
   if(read_global() != VEOK)
      bail("no global.dat");

   /* read mining address */
   if(read_data(Maddr, TXADDRLEN, "maddr.dat") != TXADDRLEN)
      bail("no maddr.dat");

   if(Trace) {
      Logfp = fopen(LOGFNAME, "a");
      plog("Entering bcon...");
   }

   /* build sorted index Txidx[] from txclean.dat */
   if(sorttx(argv[1]) != VEOK) bail("bad sorttx()");

   /* re-open the clean TX queue (txclean.dat) to read */
   fp = fopen(argv[1], "rb");
   if(!fp) {
badread:
      bail("Cannot open [txclean.dat]");
   }

   /* create cblock.dat */
   fpout = fopen("cblock.tmp", "wb");
   if(!fpout) {
badwrite:
      bail("Cannot write [cblock.tmp]");
   }

   sha256_init(&mctx);  /* for Merkel array */
   sha256_init(&bctx);  /* for entire block */

   /* trailer */
   memcpy(bt.phash, Cblockhash, HASHLEN);  /* hash of previous to new block */
   add64(Cblocknum, One, bnum);   /* Compute the new block num */
   put64(bt.bnum, bnum);          /*   and put in trailer. */
   if(Trace) plog("bcon: put 0x%s in trailer", bnum2hex(bt.bnum));
   put64(bt.mfee, Mfee);
   put32(bt.difficulty, Difficulty);
   put32(bt.time0, Time0);

   /* Prepare new block header and trailer */
   put32(bh.hdrlen, sizeof(bh));
   memcpy(bh.maddr, Maddr, TXADDRLEN);
   get_mreward(mreward, bnum);
   put64(bh.mreward, mreward);

   /* begin hash of entire block */
   sha256_update(&bctx, (byte *) &bh, sizeof(BHEADER));
   if(NEWYEAR(bt.bnum)) memcpy(&mctx, &bctx, sizeof(mctx));

   /* write header to disk */
   count = fwrite(&bh, 1, sizeof(BHEADER), fpout);
   if(count != sizeof(BHEADER)) goto badwrite;

   /* Read transactions from txclean.dat in sort order
    * using Txidx[].
    */
   ntx = 0;
   for(idx = Txidx, Tnum = 0; Tnum < Ntx && ntx < MAXBLTX; Tnum++, idx++) {
      if(Tnum != 0) {
         cond = memcmp(&Tx_ids[*idx * HASHLEN], prev_tx_id, HASHLEN);
         if(cond < 0)
            bail("internal txclean.dat sort error");
         if(cond == 0) continue;  /* ignore duplicate transaction */
      }
      memcpy(prev_tx_id, &Tx_ids[*idx * HASHLEN], HASHLEN);

      if(fseek(fp, *idx * sizeof(TXQENTRY), SEEK_SET) != 0)
         bail("bad seek on txclean.dat");

      count = fread(&tx, 1, sizeof(TXQENTRY), fp);
      if(count != sizeof(TXQENTRY)) goto badread;
      ntx++;  /* actual transactions for block */
      sha256_update(&bctx, (byte *) &tx, sizeof(TXQENTRY));  /* entire block */
      sha256_update(&mctx, (byte *) &tx, sizeof(TXQENTRY));  /* Merkel Array */
      count = fwrite(&tx, 1, sizeof(TXQENTRY), fpout);
      if(count != sizeof(TXQENTRY)) goto badwrite;
   }  /* end for Tnum */

   /* Put tran count in trailer */
   if(ntx == 0) {
      if(Trace) plog("bcon: no good transactions");
      bail(NULL);
   }
   put32(bt.tcount, ntx);

   if(NEWYEAR(bt.bnum))
      sha256_update(&mctx, (byte *) &bt, (HASHLEN+8+8+4+4+4));

   sha256_final(&mctx, bt.mroot);  /* put the Merkel root in trailer */


   /* Hash in the trailer leaving out:
    * nonce[32], stime[4], and bhash[32].
    */
   sha256_update(&bctx, (byte *) &bt, (sizeof(BTRAILER) - (2*HASHLEN) - 4));

   /* Let the miner put final hash[] and stime[] at end of BTRAILER struct
    * with the calls to sha256_final() and put32().
    * Gift bctx to miner using write_data().
    */

   /* write trailer to disk */
   count = fwrite(&bt, 1, sizeof(BTRAILER), fpout);
   if(count != sizeof(BTRAILER)) goto badwrite;

   if(Tx_ids) free(Tx_ids);    /* sorttx() allocated these two */
   if(Txidx) free(Txidx);
   fclose(fp);      /* txclean.dat */
   fclose(fpout);   /* cblock.dat */

   /* save bctx to disk for miner */
   if(write_data(&bctx, sizeof(bctx), "bctx.dat") != VEOK)
      bail("bctx.dat");

   unlink(argv[2]);
   if(rename("cblock.tmp", argv[2])) {
            error("bcon: rename cblock.tmp (%d)", errno);
            bail(NULL);
   }

   return 0;
}  /* end main() */
