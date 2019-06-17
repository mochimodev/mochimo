/* proof.c  Get proof for OP_FOUND from tfile.dat
 * Date: 12 March 2019
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


/* Check the proof given from tfile.dat in an OP_FOUND message.
 * Return VEOK to ignore, or VERROR for contention.
 */
int checkproof(TX *tx)
{
   unsigned int j; 
   int count, match, message = 0;
   BTRAILER *bt, bts;
   word32 diff = 200;  /* big number */
   word32 stime, time0, now, difficulty, highblock, prevnum = 0;
   static word32 tnum[2];
   static word32 v24trigger[2] = { V24TRIGGER, 0 };

   if(get32(Cblocknum) < V23TRIGGER) return VERROR;
   /* Un-comment next line to disable function. */
   /* return VERROR; */

   /* Check preconditions */
   match = 0;
   if(get32(Cblocknum+4)) goto allow;  /* if more than 4G blocks */
   if(get32(Cblocknum) <= NTFTX) goto allow;
   highblock = get32(tx->cblock);
   if(highblock <= NTFTX) goto allow;
   highblock = highblock - NTFTX + 1;
   now = time(NULL);

   /* Scan through trailer array in OP_FOUND TX. */
   prevnum = highblock - 1;
   bt = (BTRAILER *) TRANBUFF(tx);
   for(j = 0; j < NTFTX; j++, bt++) {
      tnum[0] = get32(bt->bnum);
      /* check tfile bnum sequence */
      if(tnum[0] != prevnum + 1) BAIL(1);
      prevnum = tnum[0];
      /* get our matching trailer from local tfile.dat */
      count = readtf(&bts, tnum[0], 1);
      stime = get32(bt->stime);
      time0 = get32(bt->time0);
      difficulty = get32(bt->difficulty);
      if(count == 1 && memcmp(bt, &bts, sizeof(BTRAILER)) == 0) {
         /* trailers match */
         match++;
         goto setdiff;
      }
      if(j == 0) BAIL(2);  /* first trailer did not match */
      /* trailers did not match so check proof trailer */
      if(stime <= time0) BAIL(3);
      if(stime > (now + BCONFREQ)) BAIL(4);
      if(difficulty != diff) BAIL(5);
      if(bt->bnum[0] == 0) continue;  /* skip NG block */
      /* stime must increase */
      if(stime <= get32((bt - 1)->stime)) BAIL(6);
      if(!get32(bt->tcount)) continue;  /* skip p-block */
      if(cmp64(bt->bnum, v24trigger) > 0) { /* v2.4 */
         if(peach(bt, diff, NULL, 1)){
            BAIL(7);
         }
      }
      if(cmp64(bt->bnum, v24trigger) <= 0) { /* v2.3 and prior */           
         if(trigg_check(bt->mroot, diff, bt->bnum) == NULL) {
            BAIL(8);
         }
      }
setdiff:
      /* update difficulty from proof and get next trailer */
      diff = set_difficulty(difficulty, stime - time0, stime,
                            (byte *) tnum);
      if(!Running) BAIL(9);
   }  /* end for j, bt */
   /* We were on the same chain, but not now... */
allow:
   /* If match == 0, preconditions were not met. */
   if(Trace) plog("checkproof() %u matches -- contention!", match);
   return VERROR;  /* contention */
bail:
   if(Trace) plog("checkproof() ignore peer (%d)", message);
   return VEOK;  /* ignore */
}  /* end checkproof() */
