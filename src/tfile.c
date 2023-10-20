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
#include "util.h"
#include "types.h"
#include "trigg.h"
#include "peach.h"
#include "network.h"
#include "global.h"

/* external support */
#include <string.h>
#include "extthread.h"
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
            pdebug("init(): Boxing Day Anomaly Bhash Failure");
             argp->tr = 0x0101; /* fail */
         } else argp->tr = 0x0001; /* pass */
      } else
      if (peach_check(&(argp->bt))) argp->tr = 0x0101; /* fail */
      else argp->tr = 0x0001; /* pass */
   } else if (trigg_check(&(argp->bt))) argp->tr = 0x0101; /* fail */
   else argp->tr = 0x0001; /* pass */

   Unthread;
}

/**
 * Get the sum of block rewards represented by a Tfile.
 * @param fname Filename of Tfile to count rewards from
 * @param rewards Pointer to place sum of block rewards
 * @param bnum Pointer to block number of last reward or NULL for no limit
 * @return (int) value representing operation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 */
int get_tfile_rewards(const char *fname, void *rewards, void *bnum)
{
   /* premine value = 4757066000000000 */
   static word32 premine[2] = { 0xbd1a6400, 0x0010e686 };

   BTRAILER bt;
   FILE *fp;
   word32 reward[2];

   /* sanity check */
   if (fname == NULL || rewards == NULL) goto FAIL_INVAL;

   /* initialize rewards with premine */
   put64(rewards, premine);

   /* open Tfile for reading */
   fp = fopen(fname, "rb");
   if (fp == NULL) return VERROR;

   /* read trailer data and calculate rewards */
   while (!feof(fp)) {
      if (fread(&bt, sizeof(bt), 1, fp) != 1 && ferror(fp)) goto FAIL_IO;
      /* check block reward limit */
      if (bnum && cmp64(bt.bnum, bnum) > 0) break;
      /* no block reward if no transactions */
      if (get32(bt.tcount)) {
         get_mreward(reward, (word32 *) bt.bnum);
         if (add64(rewards, reward, rewards)) goto FAIL_IO_OVERFLOW;
      }
   }

   /* close Tfile */
   fclose(fp);

   /* success */
   return VEOK;

/* error handling */
FAIL_INVAL: set_errno(EINVAL); return VERROR;
FAIL_IO_OVERFLOW: set_errno(EMCM_MREWARDS_OVERFLOW);
FAIL_IO: fclose(fp); return VERROR;
}  /* end get_tfile_rewards() */

/**
 * Read Tfile data into a buffer.
 * @param buffer Pointer to buffer to read Tfile data into
 * @param bnum Start block number to read from Tfile
 * @param count Number of trailers to read from Tfile
 * @return (int) number of records read from Tfile, which may be less
 * than count if an error ocurrs. Check errno for details.
*/
int read_tfile(void *buffer, void *bnum, int count, const char *tfname)
{
   long long offset;
   FILE *fp;

   fp = fopen(tfname, "rb");
   if (fp == NULL) return VERROR;
   put64(&offset, bnum);
   offset *= sizeof(BTRAILER);
   if (fseek64(fp, offset, SEEK_SET) != 0) {
      fclose(fp);
      return 0;
   }
   /* perform read into buffer */
   count = fread(buffer, sizeof(BTRAILER), (size_t) count, fp);
   fclose(fp);
   return count;
}  /* end read_tfile() */

/**
 * Read the Block Trailer of a blockchain file.
 * May also be used on the Tfile to get the last trailer entry.
 * @param btp Pointer to place Block Trailer data
 * @param fname Filename of blockchain file to read
 * @return (int) value representing operation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 */
int read_trailer(BTRAILER *btp, const char *fname)
{
   FILE *fp;

   /* read Block Trailer data */
   if ((fp = fopen(fname, "rb")) == NULL) return VERROR;
   if (fseek64(fp, -(sizeof(BTRAILER)), SEEK_END)) goto FAIL_IO;
   if (fread(btp, sizeof(BTRAILER), 1, fp) != 1) goto FAIL_IO;
   fclose(fp);

   /* success */
   return VEOK;

/* error handling */
FAIL_IO:
   fclose(fp);
   return VERROR;
}  /* end read_trailer() */

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
   pdebug("checkproof() splitblock = 0x%x", *splitblock);
   return VEOK;  /* allow syncup() to run */
bail:
   pdebug("checkproof() ignore peer (%d)", message);
   return message;  /* ignore contention */
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
   int ecode, gblock;
   char genfile[100];
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
      perr("tf_val(): Cannot open %s", fname);
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
                     perrno(tres, "tf_val() failed to wait for thread");
                  }  /* check thread status */
                  if ((targp[tidx].tr >> 8) != VEOK) {
                     perr("tf_val(0x%s) failed on POW validation",
                        bnum2hex(targp[tidx].bt.bnum));
                     /* wait for all threads to finish */
                     thread_join_list(tid, tlen);
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

      if (bt.bnum[0] == 0 && endblock) {
         percent = 100.0 * get32(bt.bnum) / endblock;
         if (weight_only) {
            psticky("Tfile check %.02f%% (0x%" P32x ")",
               percent, get32(bt.bnum));
         } else {
            psticky("Validating Tfile %.2f%% (0x%" P32x ")",
               percent, get32(bt.bnum));
         }
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
                     perrno(tres, "tf_val() failed to create thread");
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
                        pdebug("init(): Boxing Day Anomaly Bhash Failure");
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
   if (tactive) thread_join_list(tid, tlen);
   sub64(highblock, One, bnum);     /* fix high block number */
   memcpy(weight, highweight, HASHLEN);
   fclose(fp);
   pdebug("tf_val(): ecode = %d  bnum = 0x%s  weight = 0x...%x",
                  ecode, bnum2hex(highblock), highweight[0]);
tfval_end:
   psticky("");
   return ecode;
}  /* end tf_val() */


int trim_tfile(void *highbnum)
{
   FILE *fp, *fpout;
   BTRAILER bt;
   word8 bnum[8], flag;

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
      unlink("tfile.dat");
      return rename("tfile.tmp", "tfile.dat");  /* VEOK (0) on success */
   }
   perr("tfile(): flag = %d  bnum = 0x%s", flag, bnum2hex(bnum));
   return VERROR;  /* non-zero -- fail */
}  /* end trim_tfile() */

/**
 * Validate a Block Trailer against a previous (excludes PoW).
 * @param btp Pointer to Block Trailer to validate
 * @param pbtp Pointer to previous Block Trailer to validate against
 * @return (int) value representing operation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int validate_trailer(BTRAILER *btp, BTRAILER *pbtp)
{
   static word32 one[2] = { 1, 0 };
   static word32 mfee[2] = { MFEE, 0 };
   static word32 tottrigger[2] = { V23TRIGGER, 0 };
   static word8 GenesisHash[32] = {
      0x00, 0x17, 0x0c, 0x67, 0x11, 0xb9, 0xdc, 0x3c,
      0xa7, 0x46, 0xc4, 0x6c, 0xc2, 0x81, 0xbc, 0x69,
      0xe3, 0x03, 0xdf, 0xad, 0x2f, 0x33, 0x3b, 0xa3,
      0x97, 0xba, 0x06, 0x1e, 0xcc, 0xef, 0xde, 0x03
   };

   time_t start;
   word32 next_block[2], stime;

   /* init */
   time(&start);

   /* if previous Block Trailer NULL, perform genesis checks */
   if (pbtp == NULL) {
      /* check block trailer data is empty (exc. block hash) */
      if (!iszero(btp, sizeof(BTRAILER) - 32)) goto BAD_NZGEN;
      if (memcmp(btp->bhash, GenesisHash, 32) != 0) goto BAD_GENHASH;

      /* genesis ok */
      return VEOK;
   }

   /* check Mfee */
   if (btp->bnum[0] && get32(btp->tcount)) {
      if (cmp64(btp->mfee, mfee) < 0) goto BAD_MFEE;
   } else if(!iszero(btp->mfee, 8)) goto BAD_MFEE;

   /* store solve time for multiple checks */
   stime = get32(btp->stime);

   /* check diff and block times */
   if (btp->bnum[0]) {
      /* check difficulty (non-NG blocks) */
      if (get32(btp->difficulty) != set_difficulty(pbtp)) goto BAD_DIFF;
      /* check early solve time (non-NG blocks) */
      if (stime <= get32(pbtp->stime)) {
         /* discern failure type */
         if (stime == get32(pbtp->stime)) goto BAD_STIME;
         /* allow stime anomaly ONLY for the Epochalypse, Y2K38 */
         if ((word32) (stime - get32(pbtp->stime)) > BRIDGE) goto BAD_STIME;
         /* reduce "start" time to 32-bit for future solve time check */
         start &= (time_t) WORD32_C(0xffffffff);
      }
      /* check future solve time */
      if (stime > start && (stime - start) > BCONFREQ) goto BAD_STIME;
   } else {
      /* check difficulty matches previous (NG blocks) */
      if (get32(btp->difficulty) != get32(pbtp->difficulty)) goto BAD_DIFF;
      /* check solve time matches previous (NG blocks) */
      if (stime != get32(pbtp->stime)) goto BAD_STIME;
   }
   /* check for times of trouble...
    * I can't figure out the "why" of this bnum complexity...
    * so it remains, in a modified but functionally exact state... */
   if (cmp64(btp->bnum, tottrigger) > 0 /* && btp->bnum[0] != 0xfe && */
      /* btp->bnum[0] != 0xff && btp->bnum[0] != 0 */) {
      if (btp->bnum[0] > 0 && btp->bnum[0] < 0xfe) {
         if ((word32) (stime - get32(btp->time0)) > BRIDGE) goto BAD_STIME;
      }
   }
   /* check block number increment */
   add64(pbtp->bnum, one, next_block);
   if (cmp64(btp->bnum, next_block) != 0) goto BAD_BNUM;
   /* check previous hash */
   if (memcmp(pbtp->bhash, btp->phash, HASHLEN) != 0) goto BAD_PHASH;

   /* trailer is valid */
   return VEOK;

BAD_NZGEN: set_errno(EMCM_NZGEN); return VERROR;
BAD_GENHASH: set_errno(EMCM_GENHASH); return VERROR;
BAD_MFEE: set_errno(EMCM_MFEE); return VERROR;
BAD_DIFF: set_errno(EMCM_DIFF); return VERROR;
BAD_STIME: set_errno(EMCM_STIME); return VERROR;
BAD_BNUM: set_errno(EMCM_BNUM); return VERROR;
BAD_PHASH: set_errno(EMCM_PHASH); return VERROR;
}  /* end validate_trailer() */

/* end include guard */
#endif
