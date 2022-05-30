
#include "_assert.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "bcon.c"

#include "_testutils.h"

int main()
{
   static word8 expect_hash[SHA256LEN];
   static word8 expect_zero[32] = { 0 };
   static word8 expected_bnum[8] = {
      0x60, 0x28, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00
   };

   SHA256_CTX ctx;   /* for hashing pseudoblock */
   BTRAILER bt, pbt; /* pseudo-block trailer */
   word32 hdrlen;    /* pseudo-block header */
   size_t len;
   FILE *fp;
   int ecode;

   /* init block trailer */
   memcpy(&bt, b1285f, sizeof(bt));

   /* initialize globals for pseudo */
   memcpy(Cblocknum, bt.bnum, 8);
   memcpy(Cblockhash, bt.bhash, 32);
   memcpy(&Difficulty, bt.difficulty, 4);
   memcpy(&Time0, bt.time0, 4);

   /* generate pblock.dat */
   ASSERT_EQ_MSG(pseudo("pblock.dat"), VEOK,
      "pseudo() failed to generate pseudo-block");

   /* begin checks */
   fp = fopen("pblock.dat", "rb");
   ASSERT_NE_MSG(fp, NULL, "file pointer to pblock should not be NULL");
   len = fread(&hdrlen, sizeof(hdrlen), 1, fp);
   ASSERT_EQ_MSG(len, 1, "fread(hdrlen) should be 1, success");
   ASSERT_EQ_MSG(hdrlen, 4, "pblock hdrlen should be 4");
   len = fread(&pbt, sizeof(pbt), 1, fp);
   ASSERT_EQ_MSG(len, 1, "fread(btrailer) should be 1, success");
   ecode = fseek(fp, 0, SEEK_END);
   ASSERT_EQ_MSG(ecode, 0, "fseek(END) should be 0, success");
   len = (size_t) ftell(fp);
   ASSERT_EQ_MSG(len, (size_t) (hdrlen + sizeof(pbt)),
      "pbt file size should be hdrlen + sizeof(pbt) = 164 bytes");
   fclose(fp);

   /* btrailer checks */
   ASSERT_CMP_MSG(pbt.phash, Cblockhash, sizeof(pbt.phash),
      "pseudo-block phash should compare equal to Cblockhash");
   ASSERT_CMP_MSG(pbt.bnum, expected_bnum, sizeof(pbt.bnum),
      "pseudo-block bnum should compare equal to expected bnum");
   ASSERT_CMP_MSG(pbt.mfee, expect_zero, sizeof(pbt.mfee),
      "pseudo-block mfee should compare equal to zero");
   ASSERT_CMP_MSG(pbt.tcount, expect_zero, sizeof(pbt.tcount),
      "pseudo-block tcount should compare equal to zero");
   ASSERT_CMP_MSG(pbt.time0, &Time0, sizeof(pbt.time0),
      "pseudo-block time0 should compare equal to Time0");
   ASSERT_CMP_MSG(pbt.difficulty, &Difficulty, sizeof(pbt.difficulty),
      "pseudo-block difficulty should compare equal to Difficulty");
   ASSERT_CMP_MSG(pbt.mroot, expect_zero, sizeof(pbt.mroot),
      "pseudo-block mroot should compare equal to zero");
   ASSERT_CMP_MSG(pbt.nonce, expect_zero, sizeof(pbt.nonce),
      "pseudo-block nonce should compare equal to zero");
   Time0 += BRIDGE;
   ASSERT_CMP_MSG(pbt.stime, &Time0, sizeof(pbt.stime),
      "pseudo-block stime should compare equal to Time0 + BRIDGE");

   /* check final hash */
   sha256_init(&ctx);
   sha256_update(&ctx, &hdrlen, 4);
   sha256_update(&ctx, &pbt, sizeof(pbt) - HASHLEN);
   sha256_final(&ctx, expect_hash);

   ASSERT_CMP_MSG(pbt.bhash, expect_hash, sizeof(pbt.bhash),
      "pseudo-block bhash should compare equal to expected hash");

   /* cleanup */
   remove("pblock.dat");
}