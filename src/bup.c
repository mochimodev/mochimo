/* bup.c  Block Updater
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * The Mochimo Project System Software
 *
 * Date: 10 January 2018
 *
 * NOTE: Invoked by server.c update() by wait on system()
 *
 * Inputs:  argv[1],    mined block or valid received block
 *          ledger.dat  sorted
 *          ltran.dat   pre-sorted by sortlt.exe
 *
 * Outputs: if argv[2] != NULL, rename(argv[1], argv[2]) on success.
 *          updates ledger.dat by applying ltran.dat deltas
 *          removes transactions from txclean.dat
 *          exit status 0=block update, or non-zero=error.
*/

#include "extmath.h"    /* 64-bit math support */

#include "config.h"
#include "mochimo.h"
#define closesocket(_sd) close(_sd)

#define EXCLUDE_NODES   /* exclude Nodes[], ip, and socket data */
#include "data.c"

#include "error.c"
#include "crypto/crc16.c"
#include "rand.c"
#include "util.c"
#include "sorttx.c"
#include "daemon.c"

word32 Tnum = -1;  /* transaction sequence number */

void cleanup(int ecode)
{
   unlink("ledger.tmp");
   unlink("txq.tmp");
   unlink("ltran.dat");
   if(Trace) plog("cleanup() exiting with ecode %i", ecode);
   exit(1);
}


void bail(char *message)
{
   if(message) error("bup.c bailing out: %s (%d)", message, Tnum);
   cleanup(1);
}


void badtran(char *message)
{
   if(Trace && message)
      plog("bup.c bailing out: %s (%d)", message, Tnum);
   cleanup(3);
}


/* Invocation: bup mblock.dat ublock.bc */
int main(int argc, char **argv)
{
   static TXQENTRY tx;     /* Holds one transaction in the array */
   FILE *fp;
   FILE *fpout;
   FILE *bfp;              /* to read the new block */
   FILE *lfp;              /* ledger.dat */
   word32 hdrlen;          /* for block header length */
   unsigned int count;
   word32 *idx;
   int cond;
   LENTRY oldle;     /* input ledger entry  */
   LENTRY newle;     /* output ledger entry */
   LTRAN  lt;        /* ledger transaction  */
   byte taddr[TXADDRLEN];  /* transaction address hold */
   byte leof, teof;  /* end of file flags   */
   byte hold;        /* hold ledger entry for next loop */
   word32 nout;      /* temp file output record counter */
   word32 j, bcount;
   static BHEADER bh;
   static BTRAILER bt;
   word32 diff[2];
   static byte le_prev[TXADDRLEN];  /* for ledger sequence check */
   static byte lt_prev[TXADDRLEN];  /* for tran delta sequence check */

   fix_signals();
   close_extra();   /* close files > 2 */

   if(argc != 3) {
      printf("\nusage: bup ublock.tmp ublock.dat\n"
             "This program is spawned from server.c\n\n");
      exit(1);
   }

   /* get global block number, peer ip, etc. */
   if(read_global() != VEOK)
      bail("no global.dat");

   if(Trace) Logfp = fopen(LOGFNAME, "a");

   SORTLTCMD();            /* sort the ledger transaction file -- wait */
   /* build sorted index Txidx[] from txclean.dat */
   if(exists("txclean.dat")) {
      if(sorttx("txclean.dat") != VEOK)
         bail("sorttx('txclean.dat') failed!");
   }

   /***** Open the block file. *****
    *  It has already been validated.
    */
   bfp = fopen(argv[1], "rb");
   if(!bfp) {
badblock:
      error("Cannot read %s", argv[1]);
      bail("");
   }
   /* read block header */
   if(fread(&hdrlen, 1, 4, bfp) != 4) goto badblock;
   /* fixed length regular block header */
   if(hdrlen != sizeof(bh)) bail("bad hdrlen");
   if(fseek(bfp, 0, SEEK_SET)) goto badblock;
   if(fread(&bh, 1, sizeof(BHEADER), bfp) != sizeof(BHEADER)) goto badblock;
   /* read block trailer */
   if(fseek(bfp, -(sizeof(BTRAILER)), SEEK_END)) goto badblock;
   if(fread(&bt, 1, sizeof(BTRAILER), bfp) != sizeof(BTRAILER)) goto badblock;

   if(sub64(bt.bnum, Cblocknum, diff) || diff[0] != 1 || diff[1] != 0)
      bail("bt.bnum - Cblocknum != 1");

   /* re-open the clean TX queue (txclean.dat) to read */
   fp = fopen("txclean.dat", "rb");
   if(!fp) {
      fclose(bfp);     /* block */
      goto noclean;
   }

   /* create new clean TX queue */
   fpout = fopen("txq.tmp", "wb");
   if(!fpout) {
badtemp:
      bail("Cannot write txq.tmp");
   }

   /***** Read Merkel Block Array from new block *****/
   if(fseek(bfp, hdrlen, SEEK_SET)) goto badblock;

   /* Remove TX_ID's from clean TX queue that are in the new block. 
    * Merkel Array in new block is already sorted on TX_ID;
    * bval checks this in foreign blocks.
    * Above we sorted clean queue, txclean.dat, with sorttx() call.
    *
    * NOTE: end of file check on Merkel Block (bfp) depends on block
    *       trailer being shorter than a TXQENTRY !
    */

   nout = 0;    /* output counter */
   bcount = 0;  /* block counter */
   j = 0;       /* counter for block transactions */
   idx = Txidx;  /* *idx is its index */

   for( ; j < Ntx; ) {
      /* read the next tran. from the block array */
      count = fread(&tx, 1, sizeof(TXQENTRY), bfp);
      /* At end of Merkel Block, copy rest of txclean.dat to temp file */
      if(count < sizeof(TXQENTRY)) {  /* EOF */
         for( ; j < Ntx; j++, idx++) {
            /* Check for dups in txclean.dat */
            if(j > 0
               && memcmp(&Tx_ids[idx[-1] * HASHLEN], 
                         &Tx_ids[*idx * HASHLEN], HASHLEN) == 0) continue;
            /* Read clean TX in sorted order using index. */
            if(fseek(fp, *idx * sizeof(TXQENTRY), SEEK_SET) != 0) {
badclean:
               bail("Cannot read txclean.dat");
               goto badclean;
            }
            count = fread(&tx, 1, sizeof(TXQENTRY), fp);
            if(count != sizeof(TXQENTRY)) goto badclean;
            count = fwrite(&tx, 1, sizeof(TXQENTRY), fpout); 
            if(count != sizeof(TXQENTRY)) goto badtemp;
            nout++;
         }  /* end for j */
         break;  /* done */
      }  /* end if block EOF */

      bcount++;  /* count transactions in block */

nextclean:
      /* Otherwise check if block tx matches clean tx... */
      cond = memcmp(tx.tx_id, &Tx_ids[*idx * HASHLEN], HASHLEN);
      if(cond == 0) goto next2;  /* skip dup tran. */
      /* ... or if the Merkel Block has a higher TX_ID
       * copy the clean TX to temp file.
       */
      if(cond > 0) {
         if(fseek(fp, *idx * sizeof(TXQENTRY), SEEK_SET) != 0)
            goto badclean;
         count = fread(&tx, 1, sizeof(TXQENTRY), fp);
         if(count != sizeof(TXQENTRY)) goto badclean;
         count = fwrite(&tx, 1, sizeof(TXQENTRY), fpout);
         if(count != sizeof(TXQENTRY)) goto badtemp;
         nout++;  /* count output records to temp file -- new txclean */
next2:   /* examine next clean TX */
         j++;
         idx++;
         if(j >= Ntx) break;  /* done -- end of clean TX file */
         /* skip dups in txclean.dat */
         if(memcmp(&Tx_ids[idx[-1] * HASHLEN], 
                   &Tx_ids[*idx * HASHLEN], HASHLEN) == 0) goto next2;
         goto nextclean;
      }
      /* Otherwise the block transaction is not in txclean.dat.
       * (Maybe the block is foreign.)
       */
   }  /* end for j < Ntx */
   fclose(fp);      /* txclean.dat */
   fclose(fpout);   /* txq.tmp temp file */
   fclose(bfp);     /* block */
   if(bcount > get32(bt.tcount))
      bail("Bad tcount in new block");  /* should never happen! */
   unlink("txclean.dat");
   rename("txq.tmp", "txclean.dat");    /* clean TX queue is updated */
   if(Trace) plog("bup.c: wrote %u entries to new txclean.dat", nout);

noclean:

   /***** Update ledger by applying ltran.dat to ledger.dat *****
    *
    * ledger.dat is kept sorted on addr.
    * ltran.dat sorted by sortlt on addr+trancode: '-' then 'A'
    */
   leof = teof = 0;  /* end of file flags for ledger and transactions */
   nout = 0;         /* output record counter */
   hold = 0;         /* hold ledger flag */

#ifndef DEBUG_LEDGER
   lfp   = fopen("ledger.dat", "rb");
   if(lfp == NULL) bail("Cannot open ledger.dat");
   fp    = fopen("ltran.dat", "rb");
   if(fp == NULL) bail("Cannot open ltran.dat");
   fpout = fopen("ledger.tmp", "wb");
   if(fpout == NULL) bail("Cannot open ledger.tmp");

   count = fread(&lt, 1, sizeof(LTRAN), fp);  /* read a transaction */
   if(count != sizeof(LTRAN)) teof = 1;

read_ledger:
   count  = fread(&oldle, 1, sizeof(LENTRY), lfp);  /* read ledger */
   if(count != sizeof(LENTRY)) leof = 1;
      /* Sequence check on oldle.addr as else clause */
      else if(memcmp(oldle.addr, le_prev, TXADDRLEN) < 0)
              bail("bad ledger.dat sort");
   memcpy(le_prev, oldle.addr, TXADDRLEN);

   /* while one of the files is still open */
   while(leof == 0 || teof == 0) {
      /* compare ledger address to transaction address */
      cond = memcmp(oldle.addr, lt.addr, TXADDRLEN);

      if(cond == 0 && teof == 0 && leof == 0) {
         /* If ledger and transaction addr match, 
          * and both files not at end...
          */
         debug("bup: ledger<-->tran addr match");  /* debug */
         /* copy the old ledger entry to a new struct for editing */
         memcpy(&newle, &oldle, sizeof(LENTRY));
apply_tran:
         memcpy(taddr, lt.addr, TXADDRLEN);  /* save tran address */
apply2:
         if(Trace > 1) plog("bup: Applying '%c' to %s...", lt.trancode[0],
                            addr2str(lt.addr));
         /* '-' transaction sorts before 'A' */
         if(lt.trancode[0] == 'A') {
            cond = add64(newle.balance, lt.amount, newle.balance);
            if(cond) memset(newle.balance, 0, 8);
         } else if(lt.trancode[0] == '-') {
            if(cmp64(newle.balance, lt.amount) != 0)
               badtran("'-' balance != transaction amount");
            memset(newle.balance, 0, 8);
         } else bail("bad trancode");  /* should never happen! */
         /* read next transaction */
         debug("apply -- reading transaction");  /* debug */
         if(fread(&lt, 1, sizeof(LTRAN), fp) != sizeof(LTRAN)) {
            debug("eof on tran");  /* debug */
            teof = 1;
            goto write2;
         }
         /* Sequence check on lt.addr */
         if(memcmp(lt.addr, lt_prev, TXADDRLEN) < 0)
            bail("bad ltran.dat sort");
         memcpy(lt_prev, lt.addr, TXADDRLEN);

         /* Check for multiple transactions on a single address:
          * '-' must come before 'A'
          * (Transaction file did not run out and its addr matches
          *  the previous transaction...)
          */
         if(memcmp(lt.addr, taddr, TXADDRLEN) == 0) goto apply2;
write2:
         /* Only balances > Mfee are written to updated ledger. */
         if(cmp64(newle.balance, Mfee) > 0) {
            if(Trace > 1) plog("bup.c: Writing new balance to %s...",
                               addr2str(newle.addr));   /* debug */
            /* write new balance to temp file */
            count  = fwrite(&newle, 1, sizeof(LENTRY), fpout);
            if(count != sizeof(LENTRY)) bail("bad write on temp file 2");
            nout++;  /* count output records */
         } else {
            if(Trace > 1) plog("   new balance <= Mfee is not written");
         }
         if(hold) {
            debug("hold ledger");  /* debug */
            hold = 0;
            continue;  /* ...with eof checks and address compare */
         }
         goto read_ledger;
      } else if((cond < 0 || teof) && leof == 0) {
         if(Trace > 1) plog("l < t: write old ledger 1");
         /* write the old ledger entry to temp file */
         count  = fwrite(&oldle, 1, sizeof(LENTRY), fpout);
         if(count != sizeof(LENTRY)) bail("bad write on temp file 1");
         nout++;  /* count records in temp file */
         goto read_ledger;  /* read next ledger entry */
      } else if((cond > 0 || leof) && teof == 0) {
         if(lt.trancode[0] != 'A') badtran("create tran not 'A'");
         if(Trace > 1)
            plog("bup: Creating address %s...", addr2str(lt.addr));
         /* CREATE NEW ADDR
          * Copy address from transaction to new ledger entry.
          */
         memcpy(&newle, lt.addr, TXADDRLEN);
         memset(newle.balance, 0 , 8);  /* but zero balance for apply_tran */
         /* Hold old ledger entry to insert before this addition. */
         hold = 1;
         goto apply_tran;
      }
   }  /* end while not both on EOF  -- updating ledger */

   fclose(fp);
   fclose(fpout);
   fclose(lfp);
   if(nout) {
      /* if there are entries in ledger.tmp */
      unlink("ledger.dat");
      rename("ledger.tmp", "ledger.dat");
      unlink("ltran.dat");   /* may need to archive this */
   } else {
      unlink("ledger.tmp");  /* remove empty temp file */
      bail("The ledger.dat is empty!");
   }

#endif  /* !DEBUG_LEDGER */

   if(Trace) plog("bup.c: wrote %u entries to new ledger.dat", nout);

   if(rename(argv[1], argv[2]) != 0) bail("rename failed");  /* fail */

   /* malloc'd indexes for sorttx() freed on exit */

   return 0;        /* success */
}  /* end main() */
