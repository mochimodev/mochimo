/* proof.c  Get proof for OP_FOUND from tfile.dat
 * Date: 10 November 2019
 * See LICENSE.PDF
 */

/* Count of trailers that fit in a TX: */
#define NTFTX (TRANLEN / sizeof(BTRAILER))

/* Return number of records read from tfile.dat. */
int readtf(void *buff, word32 bnum, word32 count)
{
   FILE *fp;

   fp = fopen("tfile.dat", "rb");
   if(fp == NULL) return 0;
   if(fseek(fp, bnum * sizeof(BTRAILER), SEEK_SET)) {
      fclose(fp);
      return 0;
   }
   count = fread(buff, sizeof(BTRAILER), count, fp);
   if(Trace) plog("readtf() read %u trailers", count);
   fclose(fp);
   return count;
}  /* end readtf() */


/* Load proof from tfile.dat into TX prior to sending OP_FOUND */
int loadproof(TX *tx)
{
   word32 tnum;

   memset(TRANBUFF(tx), 0, TRANLEN);
   tnum = get32(Cblocknum);
   if(tnum > NTFTX) tnum = tnum - NTFTX + 1; else tnum = 1;
   return readtf(TRANBUFF(tx), tnum, NTFTX);
}


/* Reduce weight based on difficulty
 * Note: Only works above WTRIGGER31.
 */
int sub_weight(byte *weight, byte difficulty)
{
   byte temp[32];

   memset(temp, 0, 32);
   /* temp = 2**difficulty */
   temp[difficulty / 8] = 1 << (difficulty % 8);
   return multi_sub(weight, temp, weight, 32);
}  /* end sub_weight() */


/* Compute our weight at lownum and return in weight[]
 * Return VEOK on success, else error code.
 */
int past_weight(byte *weight, word32 lownum)
{
   word32 cbnum;
   int message;
   BTRAILER bts;

   cbnum = get32(Cblocknum);
   if(lownum >= cbnum) BAIL(1);
   memcpy(weight, Weight, 32);
   for( ; cbnum > lownum; cbnum--) {
      if((cbnum & 0xff) == 0) continue;  /* skip NG blocks */
      if(readtf(&bts, cbnum, 1) != 1) BAIL(2);
      sub_weight(weight, bts.difficulty[0]);
   }
   return VEOK;
bail:
   memset(weight, 0, 32);
   if(Trace) plog("past_weight(): bail: %d", message);
   return message;
}  /* end past_weight() */


#define INVALID_DIFF 256

/* Check the proof given from peer's tfile.dat in an OP_FOUND message.
 * Return VEOK to run syncup(), else error code to ignore peer.
 * On VEOK, splitblock is set to first block number where peer chain
 * splits from our chain.
 */
int checkproof(TX *tx, word32 *splitblock)
{
   int j, count, message;
   BTRAILER *bt, bts;
   word32 diff, stime, s, time0, now, difficulty, highblock, prevnum;
   static word32 tnum[2];
   static word32 v24trigger[2] = { V24TRIGGER, 0 };
   byte weight[32];

   /* Check preconditions for proof scan: */
   *splitblock = 0;  /* invalid syncup() block */
   if(get32(Cblocknum) < V23TRIGGER) goto allow;
   if(get32(Cblocknum+4)) goto allow;  /* if more than 4G blocks */
   if(get32(Cblocknum) <= NTFTX) goto allow;  /* not enough proof */
   highblock = get32(tx->cblock);
   if(highblock <= NTFTX) goto allow;   /* not enough proof */
   highblock = highblock - NTFTX + 1;

   /*** Compute peer's past weight at their low block in proof. ***/
   bt = (BTRAILER *) TRANBUFF(tx);  /* top of proof trailer array */
   tnum[0] = get32(bt->bnum);  /* their low block number */
   /* The first proof trailer must match us, */
   count = readtf(&bts, tnum[0], 1);  /* so read our tfile */
   /* and compare it to theirs. */
   if(count != 1 || memcmp(bt, &bts, sizeof(BTRAILER)) != 0) BAIL(1);
   /* If we get here, our weights must also match at the first trailer. */
   /* Compute our weight at their low block number less one. */
   if(past_weight(weight, tnum[0] - 1) != VEOK) BAIL(2);

   /* Verify peer's proof trailers in OP_FOUND TX. */
   diff = INVALID_DIFF;
   now = time(NULL);
   prevnum = highblock - 1;
   bt = (BTRAILER *) TRANBUFF(tx);
   for(j = 0; j < NTFTX; j++, bt++) {
      tnum[0] = get32(bt->bnum);  /* get trailer block number */
      /* check tfile bnum sequence */
      if(tnum[0] != prevnum + 1) BAIL(3);
      prevnum = tnum[0];
      stime = get32(bt->stime);
      time0 = get32(bt->time0);
      difficulty = get32(bt->difficulty);
      if(difficulty > 255) BAIL(4);
      if(stime <= time0) BAIL(5);  /* bad solve time sequence */
      if(stime > (now + BCONFREQ)) BAIL(6);  /* a future block is bad */
      if(j != 0 && memcmp(bt->phash, (bt - 1)->bhash, HASHLEN)) BAIL(7);
      if(bt->bnum[0] == 0) continue;  /* skip NG block */
      if(diff != INVALID_DIFF) {
         if(difficulty != diff) BAIL(8);  /* bad difficulty sequence */
      }
      if(j != 0) {
         /* stime must increase */
         if(stime <= (s = get32((bt - 1)->stime))) BAIL(9);
         if(time0 != s) BAIL(10);  /* time0 must == the previous stime */
      }
      if(get32(bt->tcount) != 0) {
         /* bt is not a pseudoblock so check work: */
         if(cmp64(bt->bnum, v24trigger) > 0) {  /* v2.4 */
            if(peach(bt, difficulty, NULL, 1)) BAIL(11);
         } else {  /* v2.3 and prior */
            if(trigg_check(bt->mroot, difficulty, bt->bnum) == NULL) BAIL(12);
         }
      }
      add_weight2(weight, difficulty);  /* tally peer's chain weight */
      /* Compute diff = next difficulty to check next peer trailer. */
      diff = set_difficulty(difficulty, stime - time0, stime,
                            (byte *) tnum);
      if(!Running) BAIL(13);
   }  /* end for j, bt -- proof trailers check */

   if(memcmp(weight, tx->weight, 32)) BAIL(14);  /* their weight is bad */

   /* Scan through trailer array to find where chain splits: splitblock */
   bt = (BTRAILER *) TRANBUFF(tx);
   for(j = 0; j < NTFTX; j++, bt++) {
      tnum[0] = get32(bt->bnum);
      /* get our matching trailer from local tfile.dat */
      count = readtf(&bts, tnum[0], 1);
      if(count != 1 || memcmp(bt, &bts, sizeof(BTRAILER)) != 0) {
         /* Our trailers do not match (or end of our tfile) */
         *splitblock = tnum[0];  /* return first non-matching block number */
         break;
      }
      if(!Running) BAIL(15);
      /* trailers match -- continue scan */
   }  /* end for j, bt -- split detection */
   if(j == 0) BAIL(16);  /* never matched -- should not happen */
   if(j >= NTFTX) BAIL(17);  /* no split found -- should not happen */
allow:
   if(Trace) plog("checkproof() splitblock = 0x%x", *splitblock);
   return VEOK;  /* allow syncup() to run */
bail:
   if(Trace) plog("checkproof() ignore peer (%d)", message);
   return message;  /* ignore contention */
}  /* end checkproof() */
