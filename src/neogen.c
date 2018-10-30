/* neogen.c  Neo-Genesis Block Generator
 *
 * Copyright (c) 2018 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * The Mochimo Project System Software
 *
 * Date: 16 February 2018
 *
*/

#include "config.h"
#include "mochimo.h"

#define EXCLUDE_NODES   /* exclude Nodes[], ip, and socket data */
#include "data.c"

#include "error.c"
#include "crypto/crc16.c"
#include "rand.c"
#include "add64.c"
#include "util.c"
#include "daemon.c"


void bail(char *message)
{
   if(message) error("neogen: bailing out: %s", message);
   write_data("fail", 4, "neofail.lck");
   exit(1);
}

#define IOBUFFLEN (32*1024)

/* neogen b00...ff b00...100 */
int main(int argc, char **argv)
{
   static BTRAILER bt, nbt;
   word32 hdrlen;      /* header length for neo block */
   word32 llen;        /* ledger length */
   SHA256_CTX bctx;
   FILE *nfp, *lfp;
   word32 neobnum[2];
   static byte buff[IOBUFFLEN];
   int count;
   word32 total;

   fix_signals();

   if(argc != 3) {
      printf("\nusage: neogen b00...ff b00...100\n"
             "This program is spawned from server.c\n\n");
      exit(1);
   }

   /* get global block number */
   if(read_global() != VEOK)
      bail("no global.dat");

   if(Cblocknum[0] != 0xff)
      bail("Cblocknum has bad modulus");

   /* read trailer from  b...ff block */
   if(readtrailer(&bt, argv[1]) != VEOK)
      bail("bad trailer read");
   if(bt.bnum[0] != 0xff)
      bail("bt.bnum has bad modulus");
   if(memcmp(bt.bnum, Cblocknum, 8) != 0)
      bail("bt.bnum != Cblocknum");
   add64(Cblocknum, One, neobnum);

   /* open ledger read-only */
   if((lfp = fopen("ledger.dat", "rb")) == NULL)
      bail("Cannot open ledger.dat");
   if(fseek(lfp, 0, SEEK_END) != 0) {
badledger:
      bail("ledger I/O error");
   }
   /* Compute ledger length and check */
   llen = ftell(lfp);
   if(llen == 0)
      bail("ledger length is zero");
   if((llen % sizeof(LENTRY)) != 0)   /* byte alignment */
      bail("invalid ledger length");

   nfp = fopen(argv[2], "wb");
   if(nfp == NULL)
      bail("Cannot create Neo-Genesis block");

   /* Add length of ledger.dat to length of header length field. */
   hdrlen = llen + 4;
   /* Begin the Neo-Genesis block by writing the header length to it. */
   if(fwrite(&hdrlen, 1, 4, nfp) != 4) {
badneo:
      bail("Neo-Genesis block I/O error");
   }

   sha256_init(&bctx);   /* begin entire block hash */
   /* ... with the header length field. */
   sha256_update(&bctx, (byte *) &hdrlen, 4);

   /* Cue ledger.dat to beginning and copy it to neo-gen block
    * header whilst hashing it into bctx.
    */
   if(fseek(lfp, 0, SEEK_SET) != 0) goto badledger;
   for(total = 0; ; ) {
      count = fread(buff, 1, IOBUFFLEN, lfp);
      if(count < 1) break;
      if(fwrite(buff, 1, count, nfp) != count) goto badneo;
      sha256_update(&bctx, buff, count);
      total += count;
   }
   if(total != llen) goto badneo;  /* check that everything got copied */
   if(ferror(lfp)) goto badneo;
   fclose(lfp);

   /* Fix-up block trailer and write to neo-block */
   memcpy(nbt.phash, bt.bhash, HASHLEN);
   put64(nbt.bnum, neobnum);
   put32(nbt.stime, get32(bt.stime));
   put32(nbt.time0, get32(bt.time0));
   put32(nbt.difficulty, get32(bt.difficulty));
   sha256_update(&bctx, (byte *) &nbt, sizeof(BTRAILER) - HASHLEN);
   sha256_final(&bctx, nbt.bhash);
   if(fwrite(&nbt, 1, sizeof(BTRAILER), nfp) != sizeof(BTRAILER))
      goto badneo;
   if(ferror(nfp)) goto badneo;
   fclose(nfp);
   if(append_tfile(argv[2], "tfile.dat") != VEOK) goto badneo;
   return 0;  /* success */
}
