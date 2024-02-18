/**
 * @private
 * @file tfile.c
 * @copyright © Adequate Systems LLC, 2018-2021. All Rights Reserved.
 * <br />For more information, please refer to ../LICENSE
*/

/* include guard */
#ifndef MOCHIMO_TFILE_C
#define MOCHIMO_TFILE_C


#include "tfile.h"

/* internal support */
#include "types.h"
#include "trigg.h"
#include "peach.h"
#include "network.h"
#include "global.h"
#include "error.h"

/* external support */
#include <string.h>
#include "extthrd.h"
#include "extmath.h"
#include "extlib.h"
#include "extio.h"

#define INVALID_DIFF 256
#define THREADS_MAX  64

typedef struct {
   volatile int tr;  /* thread function result -- set by thread */
   BTRAILER bt;      /* blocktrailer for validation */
} TF_VAL_THREAD_ARGS;

ThreadProc thread_pow_val(void *arg)
{
   static word32 v24trigger[2] = { V24TRIGGER, 0 };
   TF_VAL_THREAD_ARGS *argp = (TF_VAL_THREAD_ARGS *) arg;

   /* Adding constants to skip validation on BoxingDay corrupt block
    * provided the blockhash matches.  See "Boxing Day Anomaly" write
    * up on the Wiki or on [ REDACTED ] for more details. */
   static word32 boxingday[2] = { 0x52d3c, 0 };
   static word8 boxdayhash[32] = {
      0x2f, 0xfa, 0xb9, 0xb9, 0x00, 0xe1, 0xbc, 0xa8,
      0x25, 0x19, 0x20, 0xc2, 0xdd, 0xf0, 0x46, 0xb8,
      0x07, 0x44, 0x2a, 0xbb, 0xfa, 0x5e, 0x94, 0x51,
      0xb0, 0x60, 0x03, 0xcc, 0x82, 0x2d, 0xb1, 0x12
   };

   /* v2.4 onwards uses peach, else trigg */
   if (cmp64(argp->bt.bnum, v24trigger) > 0) {
      /* Boxing Day Anomaly -- Bugfix */
      if (cmp64(argp->bt.bnum, boxingday) == 0) {
         if (memcmp(argp->bt.bhash, boxdayhash, 32) != 0) {
            pdebug("Boxing Day Anomaly Bhash Failure");
             argp->tr = 0x0101; /* fail */
         } else argp->tr = 0x0001; /* pass */
      } else
      if (peach_check(&(argp->bt))) argp->tr = 0x0101; /* fail */
      else argp->tr = 0x0001; /* pass */
   } else if (trigg_check(&(argp->bt))) argp->tr = 0x0101; /* fail */
   else argp->tr = 0x0001; /* pass */

   Unthread;
}

/* Accumulate weight based on difficulty */
void add_weight(word8 *weight, word8 difficulty, word8 *bnum)
{
   static word32 trigger[2] = { WTRIGGER31, 0 };
   word8 add256[32] = { 0 };

   /* trigger block shifts weight increment from linear to exponential */
   if(bnum && cmp64(bnum, trigger) < 0) add256[0] = difficulty;
   else add256[difficulty / 8] = 1 << (difficulty % 8);  /* 2 ** difficulty */
   multi_add(weight, add256, weight, 32);
}  /* end add_weight() */

int append_tfile(char *fname, char *tfile)
{
   BTRAILER bt;
   FILE *fp;
   size_t count;

   if(readtrailer(&bt, fname) != VEOK) {
      perr("Cannot append_tfile()");
      return VERROR;
   }
   fp = fopen(tfile, "ab");
   if (fp == NULL) {
      perrno("failed on fopen() for %s", tfile);
      return VERROR;
   } else {
      count = fwrite(&bt, 1, sizeof(BTRAILER), fp);
      fclose(fp);
   }
   if(count != sizeof(BTRAILER)) {
      perr("failed on fwrite(): wrote %zu/%zu bytes to %s",
         count, sizeof(BTRAILER), tfile);
      return VERROR;
   }
   return VEOK;
}

/* Compute mining reward and copy to reward
 * It is a function of block number:
 *
 * Starting Reward: 0x12A05F200
 * Premine: 20800000037927936
 * Mining Distribution: 71778872624714400  (blocks 1-2097152) less NG blocks.
 * NOTE: Calculated for RTRIGGER31 = 16383
 *
 */
void get_mreward(word32 *reward, word32 *bnum)
{
   word8 bnum2[8];
   static word32 delta[2] = { 0xDAC0, 0 };      /* reward delta 56000 */
   static word32 base1[2] = { 0x2A05F200, 1 };  /* base 5000000000 */
   static word32 base2[2] = { 0x60b43c80, 1 };  /* base 5917392000 */
   static word32 base3[2] = { 0xdbe74670, 0x0d };  /* base 59523942000 */
   static word32 t1[2] =  { RTRIGGER31, 0 };    /* new reward trigger block */
   static word32 t2[2] =  { 373761, 0 };        /* mid block */
   static word32 t3[2] =  { 2097152, 0 };       /* final reward block */
   static word32 delta2[2] = { 150000, 0 };     /* increment */
   static word32 delta3[2] = { 28488, 0 };      /* decrement */

   if(cmp64(bnum, t1) < 0) {
      /* bnum < 17185 */
      if(sub64(bnum, One, bnum2)) {
         /* underflow, no reward */
         reward[0] = reward[1] = 0;
      } else {
         mult64(delta, bnum2, reward);
         add64(reward, base1, reward);
      }
   } else if(cmp64(bnum, t2) < 0) {
      /* first 4 years (excl. bnum[0... 17184]) */
      sub64(bnum, t1, bnum2);
      mult64(delta2, bnum2, reward);
      add64(reward, base2, reward);
   } else if(cmp64(bnum, t3) <= 0) {
      /* last 18 years */
      sub64(bnum, t2, bnum2);
      mult64(delta3, bnum2, reward);
      if(sub64(base3, reward, reward)) {
         /* underflow, no reward */
         reward[0] = reward[1] = 0;
      }
   } else reward[0] = reward[1] = 0;
}  /* end get_mreward() */

/* Seek to end of fname and read block trailer.
 * Return VEOK on success, else error code.
 */
int readtrailer(BTRAILER *trailer, char *fname)
{
   FILE *fp;
   size_t count;
   int seekerr;

   fp = fopen(fname, "rb");
   if (fp == NULL) {
      perrno("failed on fopen() for %s", fname);
      return VERROR;
   } else {
      seekerr = fseek(fp, -(sizeof(BTRAILER)), SEEK_END);
      if (seekerr == 0) count = fread(trailer, 1, sizeof(BTRAILER), fp);
      fclose(fp);
   }
   if(seekerr) {
      perr("failed on fseek() for %s: ecode=%d", fname, seekerr);
      return VERROR;
   }
   if(count != sizeof(BTRAILER)) {
      perr("failed on fread() for %s: read %zu/%zu bytes",
         fname, count, sizeof(BTRAILER));
      return VERROR;
   }
   return VEOK;
}

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
   pdebug("read %u trailers", count);
   fclose(fp);
   return count;
}  /* end readtf() */

/* seconds is 32-bit signed, stime and bnum are from block trailer.
 * NOTE: hash is set to 0 for old algorithm.
 * If used and integrating into an old chain,
 * change DTRIGGER31 to a non-NG block number on which to
 * trigger new algorithm.
 */
word32 set_difficulty(BTRAILER *btp)
{
   word32 hash;
   word32 stime = get32(btp->stime);
   word32 difficulty = get32(btp->difficulty);
   int seconds = stime - get32(btp->time0);
   int highsolve = 284;
   int lowsolve = 143;

   /* Change DTRIGGER31 to a non-NG block number trigger for new algorithm. */
   static word32 trigger_block[2] = { DTRIGGER31, 0 };
   static word32 fix_trigger[2] = { FIXTRIGGER, 0 };
   if(seconds < 0) return difficulty;
   if(cmp64(btp->bnum, trigger_block) < 0){
      hash = 0;
      highsolve = 506;
      lowsolve = 253;
   }
   else
      hash = (stime >> 6) ^ stime;
   if(cmp64(btp->bnum, fix_trigger) > 0) hash = 0;
   if(seconds > highsolve) {
      if(difficulty > 0) difficulty--;
      if(difficulty > 0 && (hash & 1)) difficulty--;
   } else if(seconds < lowsolve) {
      if((hash & 3) == 0  && difficulty < 255)
         difficulty++;
   }
   return difficulty;
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
   if(lownum >= cbnum) perr("failed on insufficient cbnum");
   else {
      memcpy(weight, Weight, 32);
      for( ; cbnum > lownum; cbnum--) {
         if((cbnum & 0xff) == 0) continue;  /* skip NG blocks */
         if(readtf(&bt, cbnum, 1) != 1) {
            perr("failed on readtf()");
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


/* Load proof from tfile.dat into TX prior to sending OP_FOUND */
int loadproof(TX *tx)
{
   word32 tnum;

   memset(TRANBUFF(tx), 0, TRANLEN);
   tnum = get32(Cblocknum);
   if(tnum > NTFTX) tnum = tnum - NTFTX + 1; else tnum = 1;
   return readtf(TRANBUFF(tx), tnum, NTFTX);
}

/* Check the proof given from peer's tfile.dat in an OP_FOUND message.
 * Return VEOK to run syncup(), else error code to ignore peer.
 * On VEOK, splitblock is set to first block number where peer chain
 * splits from our chain.
 */
int checkproof(TX *tx, word32 *splitblock)
{
   unsigned j;
   int count;
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
   if(count != 1 || memcmp(bt, &bts, sizeof(BTRAILER)) != 0) goto bail;
   /* If we get here, our weights must also match at the first trailer. */
   /* Compute our weight at their low block number less one. */
   if(past_weight(weight, tnum[0] - 1) != VEOK) goto bail;

   /* Verify peer's proof trailers in OP_FOUND TX. */
   diff = INVALID_DIFF;
   now = time(NULL);
   prevnum = highblock - 1;
   bt = (BTRAILER *) TRANBUFF(tx);
   for(j = 0; j < NTFTX; j++, bt++) {
      tnum[0] = get32(bt->bnum);  /* get trailer block number */
      /* check tfile bnum sequence */
      if(tnum[0] != prevnum + 1) goto bail;
      prevnum = tnum[0];
      stime = get32(bt->stime);
      time0 = get32(bt->time0);
      difficulty = get32(bt->difficulty);
      if(difficulty > 255) goto bail;
      if(stime <= time0) goto bail;  /* bad solve time sequence */
      if(stime > (now + BCONFREQ)) goto bail;  /* a future block is bad */
      if(j != 0 && memcmp(bt->phash, (bt - 1)->bhash, HASHLEN)) goto bail;
      if(bt->bnum[0] == 0) continue;  /* skip NG block */
      if(diff != INVALID_DIFF) {
         if(difficulty != diff) goto bail;  /* bad difficulty sequence */
      }
      if(j != 0) {
         /* stime must increase */
         if(stime <= (s = get32((bt - 1)->stime))) goto bail;
         if(time0 != s) goto bail;  /* time0 must == the previous stime */
      }
      if(get32(bt->tcount) != 0) {
         /* bt is not a pseudoblock so check work: */
         if(cmp64(bt->bnum, v24trigger) > 0) {  /* v2.4 */
            if (peach_check(bt) != VEOK) goto bail;
         } else {  /* v2.3 and prior */
            if (trigg_check(bt) != VEOK) goto bail;
         }
      }
      add_weight(weight, difficulty, bt->bnum);  /* tally peer's chain weight */
      /* Compute diff = next difficulty to check next peer trailer. */
      diff = set_difficulty(bt);
      if(!Running) goto bail;
   }  /* end for j, bt -- proof trailers check */

   if(memcmp(weight, tx->weight, 32)) goto bail;  /* their weight is bad */

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
      if(!Running) goto bail;
      /* trailers match -- continue scan */
   }  /* end for j, bt -- split detection */
   if(j == 0) goto bail;  /* never matched -- should not happen */
   if(j >= NTFTX) goto bail;  /* no split found -- should not happen */
allow:
   pdebug("splitblock = 0x%x", *splitblock);
   return VEOK;  /* allow syncup() to run */
bail:
   pdebug("ignore peer");
   return VERROR;  /* ignore contention */
}  /* end checkproof() */

/**
 * @brief Validate a Trailer file.
 *
 * Validates a Trailer file at @a fname and returns the Trailer
 * file block number and accumulated weight values to the
 * @a *bnum and @a weight pointers, respectively.
 * @param fname filename of the trailer file
 * @param bnum pointer to 64-bit block number value, or `NULL`
 * @param weight pointer to a 256-bit weight value, or `NULL`
 * @param weight_only when set non-zero, indicates a "weight only"
 * trailer file validation (where POW validation is not required)
 * @returns Integer representing the result of the tfile validation.
 * @retval VEOK Tfile is valid
 * @retval 1-10 Tfile is invalid
 * @retval 100-199 I/O errors
 * @retval 200+ (200 + errno)
 * @note Additional error details are logged with perr().
 */
int tf_val(char *fname, void *bnum, void *weight, int weight_only)
{
   ThreadId tid[THREADS_MAX] = { 0 };
   TF_VAL_THREAD_ARGS targp[THREADS_MAX] = { 0 };
   int tlen, tidx, tactive, tres;
   float percent;
   time_t start;
   FILE *fp;
   BTRAILER bt = { 0 };
   word32 difficulty = 0;
   word32 time1 = 0;
   word32 stime;
   word8 prevhash[HASHLEN];
   word8 highweight[HASHLEN];  /* return value */
   word8 highblock[8];         /* return value */
   long filelen;
   unsigned endblock;
   int i, ecode, gblock;
   char genfile[100], bnumhex[17];
   word32 tcount;
   static word32 tottrigger[2] = { V23TRIGGER, 0 };
   static word32 v24trigger[2] = { V24TRIGGER, 0 };

   /* Adding constants to skip validation on BoxingDay corrupt block
    * provided the blockhash matches.  See "Boxing Day Anomaly" write
    * up on the Wiki or on [ REDACTED ] for more details. */
   static word32 boxingday[2] = { 0x52d3c, 0 };
   static word8 boxdayhash[32] = {
      0x2f, 0xfa, 0xb9, 0xb9, 0x00, 0xe1, 0xbc, 0xa8,
      0x25, 0x19, 0x20, 0xc2, 0xdd, 0xf0, 0x46, 0xb8,
      0x07, 0x44, 0x2a, 0xbb, 0xfa, 0x5e, 0x94, 0x51,
      0xb0, 0x60, 0x03, 0xcc, 0x82, 0x2d, 0xb1, 0x12
   };

   ecode = 100;                 /* I/O high error code */
   tidx = tactive = 0;
   memset(highblock, 0, 8);       /* start from genesis block */
   memset(highweight, 0, HASHLEN);
   /* set ideal thread count for pow validation */
   tlen = weight_only ? 0 : cpu_cores();

   pdebug("Entering tf_val()");
   show("tfval");

   sprintf(genfile, "%s/b0000000000000000.bc", Bcdir);
   /* get trailer from our Genesis Block */
   if(readtrailer(&bt, genfile) != VEOK) goto tfval_end;  /* error 100 */
   memcpy(prevhash, bt.bhash, HASHLEN);

   fp = fopen(fname, "rb");
   if(!fp) {
      perr("Cannot open %s", fname);
      ecode = 101;
      goto tfval_end;
   }

   fseek(fp, 0, SEEK_END);
   filelen = ftell(fp);
   if((filelen % sizeof(BTRAILER)) != 0) {
      fclose(fp);
      ecode = 102;
      goto tfval_end;
   } else endblock = (unsigned) (filelen / sizeof(BTRAILER));
   fseek(fp, 0, SEEK_SET);

   /* Validate every block trailer in tfile and compute weight. */
   for(time(&start), gblock = 1, ecode = 0; Running; gblock = ecode = 0) {
      if(fread(&bt, 1, sizeof(BTRAILER), fp) != sizeof(BTRAILER)) {
         /* check for I/O error */
         if (ferror(fp)) break;
      }

      if (tlen > 0) {
         do {  /* check threads ready for join */
            for (tidx = 0; tidx < tlen && tidx < THREADS_MAX; tidx++) {
               if (tid[tidx] > 0 && targp[tidx].tr) {
                  tres = thread_join(tid[tidx]);
                  if (tres != VEOK) {
                     perrno("failed to wait for thread");
                  }  /* check thread status */
                  if ((targp[tidx].tr >> 8) != VEOK) {
                     bnum2hex(targp[tidx].bt.bnum, bnumhex);
                     perr("(0x%s) failed on POW validation", bnumhex);
                     /* wait for all threads to finish */
                     for (i = 0; i < tlen; i++) {
                        if (tid[i] == 0) continue;
                        if (thread_cancel(tid[i]) != 0) {
                           perrno("thread_cancel()");
                        }
                     }
                     ecode = 9;
                     break;
                  }
                  /* clear thread result/id, and reduce thread count */
                  targp[tidx].tr = 0;
                  tid[tidx] = 0;
                  tactive--;
               }
            }
            if (!Running || ecode) break;
            /* wait for at least ONE (1) free thread, or all threads on EOF */
         } while(tactive == tlen || (tactive && feof(fp)));
      }

      /* check for EOF or error */
      if (feof(fp) || ecode) break;

      if (!weight_only && bt.bnum[1] == 0 && endblock) {
         percent = 100.0 * get32(bt.bnum) / endblock;
         plog("Validating Tfile %.2f%% (0x%" P32x ")",
            percent, get32(bt.bnum));
      }

      tcount = get32(bt.tcount);

      ecode++;
      /* The Genesis Block is very special. 1 */
      if(gblock) {
         if(!iszero(&bt, (sizeof(BTRAILER) - HASHLEN))) break;
         ecode++;  /* 2 */
         if(memcmp(prevhash, bt.bhash, HASHLEN) != 0) break;
         difficulty = 1;  /* difficulty of block one. */
         goto next;
      }
      if(weight_only) goto skipval;

      ecode = 3;
      /* validate block trailer -- Mfee: 3 */
      if(highblock[0] && tcount) {
         if(cmp64(bt.mfee, Mfee) < 0) break;
      } else if(!iszero(bt.mfee, 8)) break;  /* for NG block or P-block */

      ecode++;  /* difficulty ecode = 4 */
      if(get32(bt.difficulty) != difficulty) break;

      ecode++;
      /* check for early block time 5 */
      stime = get32(bt.stime);
      if(highblock[0]) {
         if(stime <= time1) break;  /* unsigned time here */
         ecode++;  /* future block time 6 */
         if(stime > start && (stime - start) > BCONFREQ) break;
      }
      else if(stime != time1) break;  /* bad time for NG block */
      ecode = 7;
      /* bad block number 7 */
      if(cmp64(highblock, bt.bnum) != 0) break;
      ecode++;
      /* bad previous hash 8 */
      if(memcmp(prevhash, bt.phash, HASHLEN) != 0) break;
      ecode++;
      /* check enforced delay 9 */
      if (get32(bt.bnum) > Trustblock) {
         if(highblock[0] && tcount) {
            if (tlen > 0) {  /* find free thread slot, allocate and continue */
               for(tidx = 0; tidx < tlen && tidx < THREADS_MAX; tidx++) {
                  if (!Running) break;
                  if (tid[tidx]) continue;
                  /* copy block trailer to thread arguments */
                  memcpy(&(targp[tidx].bt), &bt, sizeof(bt));
                  tres = thread_create(&tid[tidx], &thread_pow_val, &targp[tidx]);
                  if (tres != VEOK) {
                     tid[tidx--] = 0;
                     if (Dynasleep) sleep(Dynasleep);
                     perrno("failed to create thread");
                  } else {
                     tactive++;
                     break;
                  }
               }
            } else {
               /* v2.4 onwards uses peach, else trigg */
               if (cmp64(bt.bnum, v24trigger) > 0) {
                  /* Boxing Day Anomaly -- Bugfix */
                  if(cmp64(bt.bnum, boxingday) == 0) {
                     if(memcmp(bt.bhash, boxdayhash, 32) != 0) {
                        pdebug("Boxing Day Anomaly Bhash Failure");
                        break;
                     }
                  } else
                  /* check POW */
                  if (peach_check(&bt)) break;
               } else if(trigg_check(&bt)) break;
            }
         }
      }
      ecode = 10;
      if(cmp64(highblock, tottrigger) > 0 &&
        (highblock[0] != 0xfe && highblock[0] != 0xff && highblock[0] != 0)) {
         if((word32) (stime - get32(bt.time0)) > BRIDGE) break;
      }

skipval:
      /* update for next loop 11 */
      time1 = get32(bt.stime);

      /*
       * Let the neo-genesis (not the 0xff) block change the
       * difficulty for the next 0x01 block.
       */
      if(highblock[0] != 0xff) {
         add_weight(highweight, difficulty, bt.bnum);
         difficulty = set_difficulty(&bt);
      }
next:
      /* set previous hash for next iteration */
      memcpy(prevhash, bt.bhash, HASHLEN);
      add64(highblock, One, highblock);  /* bnum in next trailer */
   }  /* end for */
   /* ensure all threads are finished */
   if (tactive) {
      /* wait for all threads to finish */
      for (i = 0; i < tlen; i++) {
         if (tid[i] == 0) continue;
         if (thread_cancel(tid[i]) != 0) {
            perrno("thread_cancel()");
         }
      }
   }
   sub64(highblock, One, bnum);     /* fix high block number */
   memcpy(weight, highweight, HASHLEN);
   fclose(fp);
   pdebug("ecode = %d  bnum = 0x%s  weight = 0x...%x",
      ecode, bnum2hex(highblock, bnumhex), highweight[0]);
tfval_end:
   return ecode;
}  /* end tf_val() */


int trim_tfile(void *highbnum)
{
   FILE *fp, *fpout;
   BTRAILER bt;
   word8 bnum[8], flag;
   char bnumhex[17];

   fp = fopen("tfile.dat", "rb");
   if(!fp) return VERROR;
   fpout = fopen("tfile.tmp", "wb");
   if(!fpout) { fclose(fp);  return VERROR; }

   put64(bnum, highbnum);
   for(flag = 0; ; ) {
      if(fread(&bt, 1, sizeof(BTRAILER), fp) != sizeof(BTRAILER)) break;
      if(fwrite(&bt, 1, sizeof(BTRAILER), fpout) != sizeof(BTRAILER)) break;
      flag = 1;
      if(iszero(bnum, 8)) break;
      sub64(bnum, One, bnum);
   }
   fclose(fpout);
   fclose(fp);
   if(iszero(bnum, 8) && flag != 0) {
      remove("tfile.dat");
      return rename("tfile.tmp", "tfile.dat");  /* VEOK (0) on success */
   }
   perr("flag = %d  bnum = 0x%s", flag, bnum2hex(bnum, bnumhex));
   return VERROR;  /* non-zero -- fail */
}  /* end trim_tfile() */

/* end include guard */
#endif
