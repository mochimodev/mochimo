/* txclean.c  Remove missing src_addr's from txclean.dat when bval fails
 *
 * Copyright (c) 2018 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * The Mochimo Project System Software
 *
 * Date: 2 April 2018
 *
 * NOTE: Invoked by server.c update() after bval and bup.
 *
 * Inputs:  ledger.dat   NO-ONE ELSE is using this file!
 *          txclean.dat
 *
 * Outputs: txclean.dat without unfound src_addr's
*/


#include "config.h"
#include "mochimo.h"
#define closesocket(_sd) close(_sd)

#define EXCLUDE_NODES   /* exclude Nodes[], ip, and socket data */
#include "data.c"

#include "error.c"
#include "crypto/crc16.c"
#include "rand.c"
#include "add64.c"
#include "util.c"
#include "daemon.c"
#include "ledger.c"

int Tnum = -1;  /* transaction sequence number */

void cleanup(int ecode)
{
   unlink("txq.tmp");
   exit(ecode);
}


void bail(char *message)
{
   if(Trace > 1)
      plog("txclean: bailing out: %s (%d)", message, Tnum);
   cleanup(0);
}

void badbail(char *message)
{
   error("txclean: %s (%d)", message, Tnum);
   cleanup(1);
}


/* Invocation: txclean txclean.dat */
int main(int argc, char **argv)
{
   static TXQENTRY tx;     /* Holds one transaction in the array */
   FILE *fp;               /* txclean.dat */
   FILE *fpout;            /* txq.tmp */
   static LENTRY src_le;   /* for le_find() */
   int count;
   word32 nout;            /* temp file output record counter */

   fix_signals();
   close_extra();   /* close files > 2 */

   if(argc != 2) {
      printf("\nusage: txclean txclean.dat\n\n");
      exit(1);
   }

   /* get global block number, peer ip, etc. */
   if(read_global() != VEOK)
      badbail("no global.dat");

   if(Trace) Logfp = fopen(LOGFNAME, "a");

   /* open the clean TX queue (txclean.dat) to read */
   fp = fopen(argv[1], "rb");
   if(!fp)
      bail("no 'txclean.dat'");

   /* create new clean TX queue */
   fpout = fopen("txq.tmp", "wb");
   if(!fpout) {
badtemp:
      badbail("Cannot write txq.tmp");
   }

   /* open ledger read-only */
   if(le_open("ledger.dat", "rb") != VEOK)
      badbail("Cannot open ledger.dat");

   nout = 0;    /* output counter */

   for(Tnum = 0; ; Tnum++) {
      /* read TX from txclean.dat */
      count = fread(&tx, 1, sizeof(TXQENTRY), fp);
      if(count != sizeof(TXQENTRY)) break;  /* EOF */
      /* if src not in ledger continue; */
      if(le_find(tx.src_addr, &src_le, NULL, 0) == FALSE) continue;
      count = fwrite(&tx, 1, sizeof(TXQENTRY), fpout); 
      if(count != sizeof(TXQENTRY)) goto badtemp;
      nout++;
   }  /* end for */

   le_close();
   fclose(fp);
   fclose(fpout);

   unlink(argv[1]);  /* txclean.dat */
   if(nout) {
      /* if there are entries in txq.tmp */
      if(rename("txq.tmp", argv[1]))
         badbail("cannot rename txq.tmp");
   } else {
      unlink("txq.tmp");  /* remove empty temp file */
      if(Trace) plog("txclean.dat is empty.");
   }

   if(Trace && nout) plog("txclean.c: wrote %u entries from %u"
                          " to new txclean.dat", nout, Tnum);
   return 0;        /* success */
}  /* end main() */

