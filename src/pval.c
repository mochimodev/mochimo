/* pval.c Times of Trouble
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.TXT   **** NO WARRANTY ****
 *
 * Date: 14 February 2019
 *
 */

#define BAIL(m) { message = m; goto bail; }


/* Validate a pseudo-block -- Called from update() */
int pval(char *fname)
{
   SHA256_CTX ctx;
   BTRAILER bt;
   word32 hdrlen, temp[2];
   int message;
   byte h[HASHLEN];
   FILE *fp;

   if(Trace) plog("Checking %s", fname);

   fp = fopen(fname, "rb");
   if(fp == NULL) BAIL(1);
   if(fread(&hdrlen, 4, 1, fp) != 1) BAIL(2);
   if(hdrlen != 4) BAIL(3);

   /* compute block file length */
   if(fseek(fp, 0, SEEK_END)) BAIL(4);
   if(ftell(fp) != sizeof(bt) + 4) BAIL(5);

   /* read trailer */
   if(fseek(fp, -(sizeof(BTRAILER)), SEEK_END)) BAIL(6);
   if(fread(&bt, sizeof(BTRAILER), 1, fp) != 1) BAIL(7);
   fclose(fp);  /* was opened by bval() */
   fp = NULL;

   /* check zeros */
   if(get32(bt.tcount) != 0) BAIL(8);
   if(!iszero(bt.mroot, 32)) BAIL(9);
   if(!iszero(bt.nonce, 32)) BAIL(10);

   /* check block num, hash, and difficulty */
   add64(Cblocknum, One, temp);
   if(cmp64(bt.bnum, temp) != 0) BAIL(11);
   if(memcmp(bt.phash, Cblockhash, HASHLEN) != 0) BAIL(12);
   if(get32(bt.difficulty) != Difficulty) BAIL(13);

   /* check block times */
   if(get32(bt.time0) != Time0) BAIL(14);
   if(get32(bt.stime) != Time0 + BRIDGE) BAIL(15);
   if(!iszero(bt.mfee, 8)) BAIL(16);

   /* compute and check block hash */
   sha256_init(&ctx);
   sha256_update(&ctx, (byte *) &hdrlen, 4);
   sha256_update(&ctx, (byte *) &bt, sizeof(bt) - HASHLEN);
   sha256_final(&ctx, h);
   if(memcmp(bt.bhash, h, HASHLEN) != 0) BAIL(17);
   unlink("ublock.dat");
   if(rename(fname, "ublock.dat") != 0) BAIL(18);
   return VEOK;

bail:
   if(fp != NULL) fclose(fp);
   if(Trace) plog("pval(): ecode = %d", message);
   return VERROR;
}  /* end pval() */


/* Make a pseudo-block with bnum = Cblocknum + 1 */
int bridge(void)
{
   static word32 hdrlen = 4;
   SHA256_CTX ctx;
   BTRAILER bt;
   word32 temp;
   FILE *fp = NULL;
   int message;

   if(Cblocknum[0] == 0xfe) BAIL(1);  /* internal error */

   if(Trace) plog("Making pblock.dat...");

   fp = fopen("pblock.dat", "wb");
   if(fp == NULL) BAIL(2);

   if(fwrite(&hdrlen, 4, 1, fp) != 1) BAIL(3);
   memset(&bt, 0, sizeof(bt));
   memcpy(bt.phash, Cblockhash, HASHLEN);
   add64(Cblocknum, One, bt.bnum);
   put32(bt.time0, Time0);
   put32(bt.difficulty, Difficulty);
   temp = Time0 + BRIDGE;
   put32(bt.stime, temp);
   
   /* compute pseudo-block hash */
   sha256_init(&ctx);
   sha256_update(&ctx, (byte *) &hdrlen, 4);
   sha256_update(&ctx, (byte *) &bt, sizeof(bt) - HASHLEN);
   sha256_final(&ctx, bt.bhash);
   if(fwrite(&bt, sizeof(bt), 1, fp) != 1) BAIL(4);
   fclose(fp);
   return VEOK;

bail:
   if(fp != NULL) fclose(fp);
   error("bridge(): ecode = %d", message);
   unlink("pblock.dat");
   return VERROR;
}  /* end bridge() */
