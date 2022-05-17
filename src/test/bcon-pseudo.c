
#include "_assert.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "bcon.c"

/* initial block trailer data */
BTRAILER BT;
word8 BTDATA[sizeof(BT)] = {  /* Block 0x1285f (75871) */
   0xb0, 0xdc, 0x58, 0xa1, 0x2e, 0x99, 0xdd, 0xd1, 0x01, 0xa9,
   0x5e, 0x4f, 0xf8, 0x20, 0xaf, 0x60, 0x6d, 0x0b, 0xe3, 0x99,
   0x1d, 0xe2, 0xb0, 0x15, 0xd8, 0xd7, 0x0b, 0xd2, 0xd6, 0x53,
   0x6a, 0x81, 0x5f, 0x28, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   0x00, 0x00, 0xdb, 0x0a, 0x1c, 0x5d, 0x21, 0x00, 0x00, 0x00,
   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   0x00, 0x00, 0x00, 0x00, 0x90, 0x0e, 0x1c, 0x5d, 0x41, 0x85,
   0xf2, 0x88, 0x09, 0x44, 0x32, 0x7f, 0xfb, 0x76, 0x1c, 0x32,
   0xc3, 0x12, 0x8e, 0xf1, 0xbf, 0xe2, 0xc0, 0x97, 0xfd, 0xc9,
   0xd3, 0x87, 0xc3, 0xf7, 0x0b, 0xe6, 0xe5, 0x66, 0x5e, 0xae
};

int main()
{
   static word8 expect_hash[SHA256LEN];
   static word8 expect_zero[32] = { 0 };
   static word8 expected_bnum[8] = {
      0x60, 0x28, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00
   };

   SHA256_CTX ctx;   /* for hashing pseudoblock */
   BTRAILER pbt;     /* pseudo-block trailer */
   word32 hdrlen;    /* pseudo-block header */
   size_t len;
   FILE *fp;
   int ecode;

   /* init BT */
   memcpy(&BT, BTDATA, sizeof(BT));

   /* initialize globals for pseudo */
   memcpy(Cblocknum, BT.bnum, 8);
   memcpy(Cblockhash, BT.bhash, 32);
   memcpy(&Difficulty, BT.difficulty, 4);
   memcpy(&Time0, BT.time0, 4);

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