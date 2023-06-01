
#include "_assert.h"
#include "extmath.h"
#include "extprint.h"
#include "sha256.h"
#include <stdio.h>
#include <stdlib.h>

#include "global.h"
#include "ledger.h"
#include "bcon.h"

#include "_testutils.h"

#define LEFILE "ledger.dat"
#define BCFILE "block.dat"
#define NEOFILE "ngblock.dat"
#define BADFILE "bad.file"

int main()
{
   FILE *fp;
   LENTRY le;
   BTRAILER *bt, nbt;
   SHA256_CTX ctx;
   word8 expect_hash[SHA256LEN];
   word8 expect_bnum[8];
   word32 hdrlen;
   long len;
   int j;

   /* suppress terminal logs */
   set_print_level(PLEVEL_NONE);

   /* CONFIGURE SYSTEM FOR SUCCESSFUL TEST FIRST */
   bt = (BTRAILER *) (b5c4ff + 4);

   /* write dummy ledger and blockchain file */
   ASSERT_EQ(write2file(LEFILE, ledgerdata, sizeof(ledgerdata)), VEOK);
   ASSERT_EQ(write2file(BCFILE, b5c4ff, sizeof(b5c4ff)), VEOK);

   /* PERFORM TEST UNDER SUCCESS CONFIGURATION */

   /* generate neogenesis block */
   ASSERT_EQ_MSG(neogen(BCFILE, NEOFILE), VEOK,
      "neogen() didn't return VEOK");

   /* initialize block hash */
   sha256_init(&ctx);
   /* verify validity of output file... */
   ASSERT_NE_MSG((fp = fopen(NEOFILE, "rb")), NULL, "cannot open " NEOFILE);
   /* ... get file length and check against hdrlen */
   ASSERT_EQ(fseek(fp, 0, SEEK_END), 0);
   ASSERT_NE((len = ftell(fp)), EOF);
   ASSERT_EQ(fseek(fp, 0, SEEK_SET), 0);
   ASSERT_EQ_MSG(fread(&hdrlen, 4, 1, fp), 1, "cannot read hdrlen");
   ASSERT_EQ_MSG((len - sizeof(BTRAILER)), hdrlen,
      "hdrlen value does not match ledger length");
   /* ...( hash hdrlen )... */
   sha256_update(&ctx, &hdrlen, 4);
   /* ... check ledger entries */
   for (j = 0; fread(&le, sizeof(le), 1, fp); j++) {
      ASSERT_CMP_MSG(&le, &ledgerdata[j], sizeof(le), "le mismatch");
      sha256_update(&ctx, &le, sizeof(le));
   }
   ASSERT_EQ_MSG(j, 10, "ledger entries in neogenesis block != 10");
   /* ... check block trailer data */
   ASSERT_EQ(fseek(fp, -(sizeof(nbt)), SEEK_END), 0);
   ASSERT_EQ(fread(&nbt, sizeof(nbt), 1, fp), 1);
   fclose(fp);
   ASSERT_CMP_MSG(nbt.phash, bt->bhash, sizeof(nbt.phash),
      "neogen phash did not compare equal to previous bhash");
   add64(bt->bnum, One, expect_bnum);
   ASSERT_CMP_MSG(nbt.bnum, expect_bnum, sizeof(nbt.bnum),
      "neogen bnum did not compare equal to previous bnum + one");
   ASSERT_CMP_MSG(nbt.mfee, Zeros, sizeof(nbt.mfee),
      "neogen mfee did not compare equal to zero");
   ASSERT_CMP_MSG(nbt.tcount, Zeros, sizeof(nbt.tcount),
      "neogen tcount did not compare equal to zero");
   ASSERT_CMP_MSG(nbt.time0, bt->time0, sizeof(nbt.time0),
      "neogen time0 did not compare equal to previous time0");
   ASSERT_CMP_MSG(nbt.difficulty, bt->difficulty, sizeof(nbt.difficulty),
      "neogen difficulty did not compare equal to previous diff");
   ASSERT_CMP_MSG(nbt.mroot, Zeros, sizeof(nbt.mroot),
      "neogen mroot did not compare equal to zero");
   ASSERT_CMP_MSG(nbt.nonce, Zeros, sizeof(nbt.nonce),
      "neogen nonce did not compare equal to zero");
   ASSERT_CMP_MSG(nbt.stime, bt->stime, sizeof(nbt.stime),
      "neogen stime did not compare equal to previous stime");
   /* hash block trailer */
   sha256_update(&ctx, &nbt, sizeof(nbt) - HASHLEN);
   /* check final hash */
   sha256_final(&ctx, expect_hash);
   ASSERT_CMP_MSG(nbt.bhash, expect_hash, sizeof(nbt.bhash),
      "neogen bhash did not compare equal to expected hash");

   /* CONFIGURE SYSTEM FOR FAILURE TESTS */

   ASSERT_NE_MSG(neogen(BADFILE, NEOFILE), VEOK,
      "neogen() returned VEOK with invalid input file name");
   sub64(bt->bnum, One, bt->bnum);
   ASSERT_EQ(write2file(BADFILE, b5c4ff, sizeof(b5c4ff)), VEOK);
   add64(bt->bnum, One, bt->bnum);
   ASSERT_NE_MSG(neogen(BADFILE, NEOFILE), VEOK,
      "neogen() returned VEOK with bad block number modulo in input file");
   ASSERT_EQ(write2file(LEFILE, ledgerdata, 1), VEOK);
   ASSERT_NE_MSG(neogen(BCFILE, NEOFILE), VEOK,
      "neogen() returned VEOK with bad ledger file size (1 byte)");
   ASSERT_EQ(write2file(LEFILE, ledgerdata, 0), VEOK);
   ASSERT_NE_MSG(neogen(BCFILE, NEOFILE), VEOK,
      "neogen() returned VEOK with empty ledger file (0 bytes)");
   remove(LEFILE);
   ASSERT_NE_MSG(neogen(BCFILE, NEOFILE), VEOK,
      "neogen() returned VEOK with missing ledger file");

   /* cleanup */
   remove(LEFILE);
   remove(BCFILE);
   remove(NEOFILE);
   remove(BADFILE);
}
