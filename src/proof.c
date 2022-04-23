/* proof.c  Get proof for OP_FOUND from tfile.dat
 *
 * Copyright (c) 2019-2021 Adequate Systems, LLC. All Rights Reserved.
 * For more information, please refer to ../LICENSE
 *
 * Date: 10 November 2019
 * Revised: 18 October 2021
 *
*/

#ifndef _MOCHIMO_PROOF_C_
#define _MOCHIMO_PROOF_C_  /* include guard */


#include "extint.h"
#include "extio.h"

#include "config.h"
#include "types.h"
#include "data.c"

#include "trigg.h"
#include "peach.h"

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
   pdebug("readtf() read %u trailers", count);
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

/* Compute our weight at lownum and return in weight[]
 * Return VEOK on success, else VERROR.
 */
int past_weight(word8 *weight, word32 lownum)
{
   BTRAILER bt;
   word32 cbnum;
   word8 temp[32];

   cbnum = get32(Cblocknum);
   if(lownum >= cbnum) perr("past_weight() failed on insufficient cbnum");
   else {
      memcpy(weight, Weight, 32);
      for( ; cbnum > lownum; cbnum--) {
         if((cbnum & 0xff) == 0) continue;  /* skip NG blocks */
         if(readtf(&bt, cbnum, 1) != 1) {
            perr("past_weight() failed on readtf()");
            break;
         }
         /* Reduce weight based on difficulty
          * Note: Only works above WTRIGGER31. */
         memset(temp, 0, 32);
         /* temp = 2**bt.difficulty[0] */
         temp[bt.difficulty[0] / 8] = 1 << (bt.difficulty[0] % 8);
         multi_sub(weight, temp, weight, 32);
      }
      if (cbnum == lownum) return VEOK;
   }
   memset(weight, 0, 32);
   return VERROR;
}  /* end past_weight() */


#define INVALID_DIFF 256

/* Check the proof given from peer's tfile.dat in an OP_FOUND message.
 * Return VEOK to run syncup(), else error code to ignore peer.
 * On VEOK, splitblock is set to first block number where peer chain
 * splits from our chain.
 */
int checkproof(TX *tx, word32 *splitblock)
{
   unsigned j;
   int count, message;
   BTRAILER *bt, bts;
   word32 diff, stime, s, time0, now, difficulty, highblock, prevnum;
   static word32 tnum[2];
   static word32 v24trigger[2] = { V24TRIGGER, 0 };
   word8 weight[32];

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
            if (peach_check(bt) != VEOK) BAIL(11);
         } else {  /* v2.3 and prior */
            if (trigg_check(bt) != VEOK) BAIL(12);
         }
      }
      add_weight(weight, difficulty, bt->bnum);  /* tally peer's chain weight */
      /* Compute diff = next difficulty to check next peer trailer. */
      diff = set_difficulty(bt);
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


#endif  /* end _MOCHIMO_PROOF_C_ */
