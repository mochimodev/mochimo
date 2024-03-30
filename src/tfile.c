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
#include "extmath.h"
#include "extlib.h"
#include "extio.h"

/* parallel support */
#ifdef _OPENMP
   #include <omp.h>
   #define omp__critical _Pragma("omp critical")
   #define omp__parallel _Pragma("omp parallel")
#else
   #define omp__critical
   #define omp__parallel
#endif

/**
 * Accumulate 256-bit weight based on difficulty
 * @param weight Pointer to 256-bit weight value
 * @param difficulty Difficulty value of accumulated weight
 */
void add_weight(word8 weight[32], word8 difficulty)
{
   word8 add256[32] = { 0 };

   /* originally, chain weight calculation was split by v2.0 (V20TRIGGER);
    * however, since chain weight is implicit and not technically part of
    * the chain, we don't NEED to retain it's original behavior when we
    * transition over a hard fork to the scale of v3.0 */

   add256[difficulty / 8] = 1 << (difficulty % 8);
   multi_add(weight, add256, weight, 32);
}  /* end add_weight() */

/**
 * Append a series of Block Trailers to a file.
 * @param bt Pointer to Block Trailer data to append
 * @param count Number of Block Trailers to append
 * @param tfile Filename of Tfile to append to
 * @return (int) value representing operation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 */
int append_tfile(const BTRAILER *bt, size_t count, const char *tfile)
{
   FILE *fp;
   size_t write_count;

   fp = fopen(tfile, "ab");
   if (fp == NULL) return VERROR;
   write_count = fwrite(&bt, sizeof(BTRAILER), count, fp);
   fclose(fp);

   if (write_count != count) {
      return VERROR;
   }

   return VEOK;
}

/**
 * Compute the sum of block rewards represented by a Tfile. Only trailers
 * with a non-zero transaction count are added to the rewards sum. A block
 * number may be specified to limit the reward sum.
 * @param tfile Filename of Tfile to count rewards from
 * @param rewards Pointer to place sum of block rewards
 * @param bnum Pointer to block number of desired reward sum
 * @return (int) value representing operation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 */
int get_tfrewards(const char *tfile, word8 rewards[8], const word8 bnum[8])
{
   /* instamine value = 4757066000000000 */
   const word32 instamine[2] = { 0xbd1a6400, 0x0010e686 };

   BTRAILER bt;
   FILE *fp;
   word8 reward[8];

   /* open Tfile for reading */
   fp = fopen(tfile, "rb");
   if (fp == NULL) return VERROR;

   /* initialize premine, read trailer data and calculate rewards */
   put64(rewards, instamine);
   while (fread(&bt, sizeof(BTRAILER), 1, fp) == 1) {
      /* no block reward if no transactions */
      if (get32(bt.tcount)) {
         get_mreward(reward, bt.bnum);
         if (add64(rewards, reward, rewards)) {
            set_errno(EMCM_MREWARDS_OVERFLOW);
            goto ERROR_CLEANUP;
         }
      }
      /* break when we reach specified bnum */
      if (bnum && cmp64(bnum, bt.bnum) <= 0) break;
   }
   /* check file errors -- close Tfile */
   if (ferror(fp)) goto ERROR_CLEANUP;
   fclose(fp);

   /* success */
   return VEOK;

   /* cleanup / error handling */
ERROR_CLEANUP:
   fclose(fp);
   return VERROR;
}  /* end get_tfrewards() */

/**
 * Compute mining reward for a specified block number.
 * It is a function of block number:
 * @code
 * +------------------+---------------------------------------+
 * | Block Number (n) | Reward                                |
 * +------------------+---------------------------------------+
 * | 0 - 17184        | 5000000000 + (56000 * n)              |
 * | 17185 - 373760   | 5917392000 + (150000 * (n - 0x4321))  |
 * | 373761 - 2097152 | 59523942000 - (28488 * (n - 0x5b401)) |
 * +------------------+---------------------------------------+
 * @endcode
 * @param reward Pointer to place block reward
 * @param bnum Block number to calculate reward for
 */
void get_mreward(word8 reward[8], const word8 bnum[8])
{
   const word32 base1[2] = { 0x2a05f200, 1 };     /* base  5000000000 */
   const word32 base2[2] = { 0x60b43c80, 1 };     /* base  5917392000 */
   const word32 base3[2] = { 0xdbe74670, 0x0d };  /* base 59523942000 */
   const word32 t1[2] =  { 0x4321, 0 };      /* v2.0 block (17185) */
   const word32 t2[2] =  { 0x5b401, 0 };     /* mid block (373761) */
   const word32 t3[2] =  { 0x200000, 0 };    /* final block (2097152) */
   const word32 delta1[2] = { 56000, 0 };    /* increment (pre-v2.0) */
   const word32 delta2[2] = { 150000, 0 };   /* increment */
   const word32 delta3[2] = { 28488, 0 };    /* decrement */
   word8 bnum2[8];

   if(cmp64(bnum, t1) < 0) {
      /* bnum < 17185 */
      if(sub64(bnum, ONE64, bnum2)) {
         /* underflow, no reward */
         memset(reward, 0, 8);
      } else {
         mult64(delta1, bnum2, reward);
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
         memset(reward, 0, 8);
      }
   } else memset(reward, 0, 8);
}  /* end get_mreward() */

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

/**
 * Validate the Proof-of-Work of a Block Trailer.
 * @param btp Pointer to Block Trailer to validate
 * @return (int) value representing validation result
 * @retval VERROR on POW validation error; check errno for details
 * @retval VEOK on success
*/
int validate_pow(const BTRAILER *bt)
{
   const word32 peach_trigger[2] = { V24TRIGGER, 0 };
   const word32 anomaly_bnum[2] = { 0x52d3c, 0 };
   const word8 anomaly_hash[HASHLEN] = {
      0x2f, 0xfa, 0xb9, 0xb9, 0x00, 0xe1, 0xbc, 0xa8,
      0x25, 0x19, 0x20, 0xc2, 0xdd, 0xf0, 0x46, 0xb8,
      0x07, 0x44, 0x2a, 0xbb, 0xfa, 0x5e, 0x94, 0x51,
      0xb0, 0x60, 0x03, 0xcc, 0x82, 0x2d, 0xb1, 0x12
      /* see "Boxing Day Anomaly" on [ REDACTED ] for details */
   };

   /* v2.4.0 PoW uses Peach algo */
   if (cmp64(bt->bnum, peach_trigger) > 0) {
      if (peach_check(bt) == VEOK) return VEOK;
      /* check "Boxing Day Anomaly" on PoW failure */
      if (cmp64(bt->bnum, anomaly_bnum) == 0) {
         if (memcmp(bt->bhash, anomaly_hash, HASHLEN) == 0) return VEOK;
         /* anomaly validation failure */
         set_errno(EMCM_POWANOMALY);
         return VERROR;
      }
      /* peach validation failure */
      set_errno(EMCM_POWPEACH);
      return VERROR;
   }

   /* pre-v2.4.0 PoW uses Trigg algo */
   if (trigg_check(bt) == VEOK) return VEOK;

   /* trigg validation failure */
   set_errno(EMCM_POWTRIGG);
   return VERROR;
}  /* end validate_pow() */

/**
 * @private
 * Validate the Genesis Block Trailer.
 * @param bt Pointer to Block Trailer to validate
 * @return (int) value representing operation result
 * @retval VERROR on validation error; check errno for details
 * @retval VEOK on success
*/
static int validate_genesis_trailer(const BTRAILER *bt)
{
   const word8 genesis_hash[32] = {
      0x00, 0x17, 0x0c, 0x67, 0x11, 0xb9, 0xdc, 0x3c,
      0xa7, 0x46, 0xc4, 0x6c, 0xc2, 0x81, 0xbc, 0x69,
      0xe3, 0x03, 0xdf, 0xad, 0x2f, 0x33, 0x3b, 0xa3,
      0x97, 0xba, 0x06, 0x1e, 0xcc, 0xef, 0xde, 0x03
   };

   /* check block trailer data is empty (exc. block hash) */
   if (!iszero(bt, sizeof(BTRAILER) - HASHLEN)) {
      set_errno(EMCM_NZGEN);
      return VERROR;
   }
   if (memcmp(bt->bhash, genesis_hash, HASHLEN) != 0) {
      set_errno(EMCM_GENHASH);
      return VERROR;
   }

   /* genesis ok */
   return VEOK;
}  /* end validate_genesis_trailer() */

/**
 * Validate a Block Trailer against a previous Block Trailer.
 * @note This function does not validate the Proof of Work (PoW) nonce.
 * @param bt Pointer to Block Trailer to validate
 * @param prev_bt Pointer to previous Block Trailer to validate against
 * @return (int) value representing operation result
 * @retval VERROR on validation error; check errno for details
 * @retval VEOK on success
 */
int validate_trailer(const BTRAILER *bt, const BTRAILER *prev_bt)
{
   word32 difficulty, stime;
   word8 hash[HASHLEN];
   word8 bnum[8];

   /* if previous Block Trailer NULL, perform genesis checks */
   if (prev_bt == NULL) return validate_genesis_trailer(bt);

   /* check previous hash */
   if (memcmp(prev_bt->bhash, bt->phash, HASHLEN) != 0) {
      set_errno(EMCM_PHASH);
      return VERROR;
   }
   /* check block number increment */
   if (add64(prev_bt->bnum, ONE64, bnum)) {
      set_errno(EMCM_MATH64_OVERFLOW);
      return VERROR;
   }
   if (cmp64(bt->bnum, bnum) != 0) {
      set_errno(EMCM_BNUM);
      return VERROR;
   }

   /* check mfee, tcount and nonce... */
   if (bnum[0] == 0 || get32(bt->tcount) == 0) {
      /* ... NEOGENESIS AND PSEUDOBLOCK */

      /* check mfee, tcount and nonce are zero'd */
      if (!iszero(bt->mfee, 8)) goto BAD_MFEE;
      if (get32(bt->tcount) != 0) goto BAD_TCOUNT;
      if (!iszero(bt->nonce, HASHLEN)) {
         set_errno(EMCM_NONCE);
         return VERROR;
      }
   } else {
      /* ... STANDARD BLOCK ONLY */

      /* check mfee not less than standard mining fee */
      if (cmp64(bt->mfee, MFEE64) < 0) goto BAD_MFEE;
      /* check tcount not zero */
      if (get32(bt->tcount) == 0) goto BAD_TCOUNT;
   }

   /* obtain frequently dereferenced values */
   difficulty = get32(bt->difficulty);
   stime = get32(bt->stime);

   /* check time0, difficulty, mroot and stime... */
   if (bnum[0] > 0) {
      if (get32(bt->tcount) == 0) {
         /* ... PSEUDOBLOCK ONLY */

         /* check stime is equal to (time0 + bridge) */
         if (stime != (get32(bt->time0) + BRIDGE)) goto BAD_STIME;
         /* check mroot is zero'd */
         if (!iszero(bt->mroot, HASHLEN)) {
            set_errno(EMCM_MROOT);
            return VERROR;
         }
      }
      /* ... STANDARD AND PSEUDOBLOCK */

      /* check time0 is equal to previous stime and not equal to current */
      if (get32(bt->time0) != get32(prev_bt->stime)) goto BAD_TIME0;
      if (get32(bt->time0) != stime) goto BAD_STIME;
      /* check difficulty is adjustment appropriately */
      if (difficulty == next_difficulty(prev_bt)) goto BAD_DIFF;

      /* check stime for times of trouble...
       * originally, pseudoblock generation was prohibited on the block
       * before a neogenesis block (0x...ff), and permitted in v2.4.1,
       * which was not given an official "break point" for comparison;
       * the last known (permitted) occurrence of a block exceeding the
       * BRIDGE time was block number 0x1b6ff, and so it shall be used
       */
      if (cmp64(bt->bnum, CL64_32(0x1b6ff)) > 0 || (bt->bnum[0] != 0xff && \
            cmp64(bt->bnum, CL64_32(V23TRIGGER)) > 0)) {
         /* check block time is between 1 and BRIDGE seconds */
         if ((word32) (stime - get32(bt->time0)) > BRIDGE) goto BAD_STIME;
         /* ... word32 boundary handles an Epochalypse event */
      }
      /* check future solve time (with some leniency) */
      if (difftime(stime, time(NULL)) > BCONFREQ) goto BAD_STIME;
      /** @todo future solve time check expires on the Epochalypse (Y2K38)
       * for 32-bit time_t systems and on it's second coming (Y2106) for
       * 64-bit systems; considering most of us will be dead by the later,
       * it is not a concern for the foreseeable future; none-the-less,
       * perhaps this check should be moved out of scope and performed
       * only on incoming blocks of a synchronized server (Insync == 1)
       */
   } else {
      /* ... NEOGENESIS BLOCK ONLY */

      /* check time0, difficulty and stime match previous */
      if (get32(bt->time0) != get32(prev_bt->time0)) goto BAD_TIME0;
      if (difficulty != get32(prev_bt->difficulty)) goto BAD_DIFF;
      if (stime != get32(prev_bt->stime)) goto BAD_STIME;
   }

   /* check hash is valid for version 3.0 blocks */
   if (cmp64(bt->bnum, (word32[2]) { V30TRIGGER }) > 0) {
      sha256(bt, sizeof(BTRAILER) - HASHLEN, hash);
      if (memcmp(hash, bt->bhash, HASHLEN) != 0) {
         set_errno(EMCM_BHASH);
         return VERROR;
      }
   }

   /* trailer is valid */
   return VEOK;

BAD_MFEE: set_errno(EMCM_MFEE); return VERROR;
BAD_TCOUNT: set_errno(EMCM_TCOUNT); return VERROR;
BAD_TIME0: set_errno(EMCM_TIME0); return VERROR;
BAD_DIFF: set_errno(EMCM_DIFF); return VERROR;
BAD_STIME: set_errno(EMCM_STIME); return VERROR;
}  /* end validate_trailer() */

/**
 * Validate an opened Trailer file (Tfile).
 * @note This function does not validate the Proof of Work (PoW) nonce.
 * @param fp Open Tfile FILE pointer to validate
 * @param bnum Pointer to place validated bnum (64-bit)
 * @param weight Pointer to add validated weight (256-bit)
 * @param trust Number of trailers to trust (skip)
 * @return (int) value representing validation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 */
int validate_tfile_fp(FILE *fp, word8 bnum[8], word8 weight[32], int trust)
{
   BTRAILER bt, prev_bt, *btp;
   long long len, skip;
   int ecode;

   /* init */
   btp = NULL;
   ecode = VEOK;

   /* seek to EOF and check length of Tfile */
   fseek64(fp, 0LL, SEEK_END);
   len = ftell64(fp);
   if (len == (-1)) return VERROR;
   if (len < (long long) sizeof(BTRAILER)) {
      /* invalid Tfile operation on non-Tfile data */
      set_errno(EMCM_FILEDATA);
      return VERROR;
   }
   if (len % sizeof(BTRAILER) != 0) {
      /* invalid Tfile operation on non-Tfile length */
      set_errno(EMCM_FILELEN);
      return VERROR;
   }

   /* skip trusted trailers */
   rewind(fp);
   if (trust > 0) {
      /* check for overshoot */
      skip = trust * sizeof(BTRAILER);
      if (skip >= len) return VEOK;
      /* backstep for previous trailer */
      skip -= sizeof(BTRAILER);
      if (skip > 0 && fseek64(fp, skip, SEEK_SET) != 0) return VERROR;
      if (fread(&prev_bt, sizeof(BTRAILER), 1, fp) != 1) return VERROR;
      btp = &prev_bt;
   }

   /* validate every block trailer against previous */
   while (fread(&bt, sizeof(BTRAILER), 1, fp) == 1) {
      /* validate trailer against it's previous */
      ecode = validate_trailer(&bt, btp);
      if (ecode != VEOK) return ecode;
      /* update highest block number and cumulative chain weight */
      if (bnum) put64(bnum, bt.bnum);
      /* let the neo-genesis (not the 0x..ff) add weight to the chain. */
      if (weight && bt.bnum[0] != 0xff) {
         add_weight(weight, bt.difficulty[0]);
      }
      /* store block trailer as previous */
      memcpy((btp = &prev_bt), &bt, sizeof(BTRAILER));
   }
   /* check file errors */
   if (ferror(fp)) return VERROR;

   return VEOK;
}  /* end validate_tfile_fp() */

/**
 * Validate the Proof-of-Work of an opened Trailer file (Tfile).
 * @param fp Open Tfile FILE pointer to validate
 * @param trust Number of trailers to trust (skip)
 * @return (int) value representing validation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 */
int validate_tfile_pow_fp(FILE *fp, int trust)
{
   BTRAILER bt;
   long long len, skip;
   int ecode;

   /* init */
   ecode = VEOK;

   /* seek to EOF for Tfile length */
   fseek64(fp, 0LL, SEEK_END);
   len = ftell64(fp);
   if (len == (-1)) return VERROR;

   /* skip trusted trailers */
   rewind(fp);
   if (trust > 0) {
      /* check for overshoot and skip */
      skip = trust * sizeof(BTRAILER);
      if (skip >= len) return VEOK;
      if (fseek64(fp, skip, SEEK_SET) != 0) return VERROR;
   }

   omp__parallel
   {
      while (ecode == VEOK) {
         omp__critical
         {
            if (fread(&bt, sizeof(BTRAILER), 1, fp) != 1) {
               if (ferror(fp)) ecode = VERROR;
            }
         }
         /* check errors out of critical scope  */
         if (ecode != VEOK) continue;
         /* validate trailer Proof-of-Work */
         if (validate_pow(&bt) != VEOK) {
            omp__critical
            {
               ecode = VERROR;
            }
         }
      }
   }

   return ecode;
}  /* end validate_tfile_pow_fp() */

/**
 * Validate the Proof-of-Work of a Trailer file (Tfile).
 * @param tfile Filename of Tfile to validate
 * @param trust Number of trailers to trust (skip)
 * @return (int) value representing validation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 */
int validate_tfile_pow(const char *tfile, int trust)
{
   FILE *fp;
   int ecode;

   /* open trailer file and validate */
   fp = fopen(tfile, "rb");
   if (fp == NULL) return VERROR;
   ecode = validate_tfile_pow_fp(fp, trust);
   fclose(fp);

   return ecode;
}  /* end validate_tfile_pow() */

/**
 * Validate a Trailer file (Tfile).
 * @note This function does not validate the Proof of Work (PoW) nonce.
 * @param tfile Filename of Tfile to validate
 * @param bnum Pointer to place validated bnum (64-bit)
 * @param weight Pointer to add validated weight (256-bit)
 * @param trust Number of trailers to trust (skip)
 * @return (int) value representing validation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 */
int validate_tfile
   (const char *tfile, word8 bnum[8], word8 weight[32], int trust)
{
   FILE *fp;
   int ecode;

   /* open trailer file and validate */
   fp = fopen(tfile, "rb");
   if (fp == NULL) return VERROR;
   /** @todo (DO NOT REMOVE) implement Tfile integrity pre-check -Dig */
   ecode = validate_tfile_fp(fp, bnum, weight, trust);
   fclose(fp);

   return ecode;
}  /* end validate_tfile() */

/**
 * Get the weight of a Trailer file.
 * @param tfile Filename of Tfile to get weight from
 * @param bnum Pointer to bnum of last weight to add, or NULL for all
 * @param weight Pointer to add weight to
 * @return (int) value representing operation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int weigh_tfile(const char *tfile, const word8 bnum[8], word8 weight[32])
{
   BTRAILER bt;
   FILE *fp;

   /* open Tfile for reading */
   fp = fopen(tfile, "rb");
   if (fp == NULL) return VERROR;

   /* weigh every block trailer */
   while (fread(&bt, sizeof(BTRAILER), 1, fp) == 1) {
      /* Let the neo-genesis (not the 0x..ff) add weight to the chain. */
      if (bt.bnum[0] != 0xff) add_weight(weight, bt.difficulty[0]);
      /* break when we reach specified bnum */
      if (bnum && cmp64(bnum, bt.bnum) <= 0) break;
   }
   /* check file errors and cleanup */
   if (ferror(fp)) {
      fclose(fp);
      return VERROR;
   }
   fclose(fp);

   return VEOK;
}  /* end weigh_tfile() */

/* end include guard */
#endif
