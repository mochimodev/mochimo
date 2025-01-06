/**
 * @private
 * @headerfile tfile.h <tfile.h>
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
#endif

/** Global shutdown flag for long running functions of tfile.h */
word8 TfileShutdown = 0;

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
   write_count = fwrite(bt, sizeof(BTRAILER), count, fp);
   fclose(fp);

   if (write_count != count) {
      return VERROR;
   }

   return VEOK;
}

/**
 * Compute the bridge time for a specified block number.
 * @param bnum Block number to calculate bridge time for
 * @return (word32) value representing bridge time
 */
word32 get_bridge(const void *bnum)
{
   /* pre-V30TRIGGER BRIDGE time */
   if (bnum && cmp64(bnum, CL64_32(V30TRIGGER)) < 0) {
      return 949;
   }
   /* post-V30TRIGGER BRIDGE time */
   return 238;
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
 * As of version 3, this function targets a total supply of
 * 2 ^ 56 (~72M) coins at block number 6422560, where the reward
 * is fixed to 1000000000 for all proceeding rewards.
 * The reward distribution is as follows:
 * @code
 * +------------------+---------------------------------------+
 * | Block Number (n) | Reward                                |
 * +------------------+---------------------------------------+
 * | 0 - 17184        | 5000000000 + (56000 * n)              |
 * | 17185 - 373760   | 5917392000 + (150000 * (n - 0x4321))  |
 * | 373761 - 655360  | 59523942000 - (28488 * (n - 0x5b401)) |
 * | 655361 - 6422560 | 12938103399 - (2070 * (n - 0xa0000))  |
 * | 6422560 - ...    | 1000000000                            |
 * +------------------+---------------------------------------+
 * @endcode
 * @note While in some form or another the target supply has always
 * been 2^56 (~72M) coins, initially this included the premine, but
 * for various changes over the history of the network, it did not.
 * Version 3.x of the network, reverts the reward distribution back
 * to the intended target supply of 2^56 coins, including premine.
 * The process of determining the parameters to achieve this target
 * supply is no more than simple algebraic manipulation of the Sum
 * of an Arithmetic Sequence formula... followed by some dark arts.
 * The formula for the sum of an arithmetic sequence is:
 *    Sn = n(A1 + An)/2
 * where Sn is the sum of the sequence,
 *       n is the number of rewards for the sequence,
 *       A1 is the first reward in the sequence, and
 *       An is the n-th reward in the sequence.
 * Using the existing parameters, one can determine the total supply
 * at any given block number, subtract this supply from the desired
 * total supply, and use the result as the desired Sum (Sn). Assuming
 * the n-th reward (An) and distribution length (n) are known, one
 * can solve for the first reward (A1) and subsequent reward deltas
 * for a "close enough" parameter set to achieve the desired total
 * supply. The "dark arts" come in when one realizes that neogenesis
 * blocks negatively affect the accuracy of this method. By removing
 * the neogenesis blocks from the n parameter, one can achieve a more
 * accurate result, but still non-exact result, which is then further
 * adjusted manually to achieve the desired total supply. It's likely
 * there is an exact solution to this problem that does not require
 * such "dark arts", but it is beyond the scope of this documentation.
 * @param reward Pointer to place block reward
 * @param bnum Block number to calculate reward for
 */
void get_mreward(word8 reward[8], const word8 bnum[8])
{
   const word32 final[2] = { 0x3b9aca00, 0 };     /* final 1000000000 */
   const word32 base1[2] = { 0x2a05f200, 1 };     /* base  5000000000 */
   const word32 base2[2] = { 0x60b43c80, 1 };     /* base  5917392000 */
   const word32 base3[2] = { 0xdbe74670, 0x0d };  /* base 59523942000 */
   const word32 base4[2] = { 0x032bca67, 0x03 };  /* base 12938103399 */
   const word32 t1[2] =  { V20TRIGGER, 0 };  /* v2.0 block (17185) */
   const word32 t2[2] =  { MIDTRIGGER, 0 };  /* mid block (373761) */
   const word32 t3[2] =  { V30TRIGGER, 0 };  /* v3.0 block (655360) */
   const word32 delta1[2] = { 56000, 0 };    /* increment (pre-v2.0) */
   const word32 delta2[2] = { 150000, 0 };   /* increment */
   const word32 delta3[2] = { 28488, 0 };    /* decrement */
   const word32 delta4[2] = { 2070 , 0 };    /* decrement */
   word8 bnum2[8];
   int fix = 0;

   if (cmp64(bnum, t1) < 0) {
      /* v1.x.x reward incrementing */
      if(sub64(bnum, ONE64, bnum2)) {
         /* underflow, no reward */
         memset(reward, 0, 8);
      } else {
         mult64(delta1, bnum2, reward);
         add64(reward, base1, reward);
      }
   } else if(cmp64(bnum, t2) < 0) {
      /* v2.x.x reward incrementing */
      sub64(bnum, t1, bnum2);
      mult64(delta2, bnum2, reward);
      add64(reward, base2, reward);
   } else if(cmp64(bnum, t3) <= 0) {
      /* v2.x reward decrementing */
      sub64(bnum, t2, bnum2);
      mult64(delta3, bnum2, reward);
      if(sub64(base3, reward, reward)) {
         /* underflow, no reward */
         memset(reward, 0, 8);
      }
   } else {
      /* v3.x.x reward decrementing */
      sub64(bnum, t3, bnum2);
      if (mult64(delta4, bnum2, reward)) fix = 1;
      else if (sub64(base4, reward, reward)) fix = 1;
      if (fix || cmp64(reward, final) < 0) {
         memcpy(reward, final, 8);
      }
   }
}  /* end get_mreward() */

/**
 * Compute the Merkle Root of a list of hashes. Assumes HASHLEN byte hashes.
 * @note This function is recursive with an integral depth of 1 + log2(n).
 * @param hashlist Pointer to list of hashes
 * @param count Number of hashes in list
 * @param root Pointer to place Merkle Root hash
 */
void merkle_root(const word8 *hashlist, size_t count, word8 *root)
{
   word8 merkle[HASHLEN * 2];
   word8 *splitlist;
   size_t split;

   switch (count) {
      case 0: return;
      case 1:
         /* transfer hash to root */
         memcpy(root, hashlist, HASHLEN);
         return;
      case 2:
         /* hash items directly into root */
         sha256(hashlist, HASHLEN * 2, root);
         return;
      default:
         /* split list in half and recurse */
         split = count / 2;
         count = count - split;
         splitlist = ((word8 *) hashlist) + (count * HASHLEN);
         merkle_root(hashlist, count, merkle);
         merkle_root(splitlist, split, merkle + HASHLEN);
         /* hash merkle node hashes into root */
         sha256(merkle, HASHLEN * 2, root);
   }
}  /* end merkle_root() */

/**
 * Read Trailers from a Tfile into a buffer.
 * @param buffer Pointer to buffer to read Tfile data into
 * @param bnum Start block number to read from Tfile
 * @param count Number of trailers to read from Tfile
 * @param tfile Filename of Tfile to read from
 * @return (int) number of records read from Tfile, which may be less
 * than count if an error ocurrs; check errno for details
*/
size_t read_tfile
   (void *buffer, const word8 bnum[8], size_t count, const char *tfile)
{
   long long offset;
   size_t n = 0;
   FILE *fp;

   /* open Tfile and read trailer from offset */
   fp = fopen(tfile, "rb");
   if (fp == NULL) return VERROR;
   /* seek to read offset for bnum */
   put64(&offset, bnum);
   offset *= sizeof(BTRAILER);
   if (fseek64(fp, offset, SEEK_SET) == 0) {
      /* perform read into buffer and cleanup -- check for EOF */
      n = fread(buffer, sizeof(BTRAILER), (size_t) count, fp);
      if (n != count && !ferror(fp)) set_errno(EMCM_EOF);
   }

   fclose(fp);

   return n;
}  /* end read_tfile() */

/**
 * Read a Block Trailer from a file.
 * May also be used on the Tfile to get the last trailer entry.
 * @param bt Pointer to place Block Trailer data
 * @param file Filename of blockchain file to read
 * @return (int) value representing operation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 */
int read_trailer(BTRAILER *bt, const char *file)
{
   FILE *fp;

   /* open file and read Trailer */
   fp = fopen(file, "rb");
   if (fp == NULL) return VERROR;
   if (fseek64(fp, -(sizeof(BTRAILER)), SEEK_END) != 0) goto ERROR_CLEANUP;
   if (fread(bt, sizeof(BTRAILER), 1, fp) != 1) {
      if (ferror(fp)) goto ERROR_CLEANUP;
   }
   /* cleanup */
   fclose(fp);

   return VEOK;

/* cleanup / error handling */
ERROR_CLEANUP:
   fclose(fp);

   return VERROR;
}  /* end read_trailer() */

/**
 * Compute the difficulty of the next block, given the previous trailer.
 * NOTE: hash is set to 0 for old algorithm.
 * NOTE: seconds is intentionally 32-bit signed.
 * @param bt Pointer to previous Block Trailer
 * @return (word32) value representing the next difficulty
 */
word32 next_difficulty(const BTRAILER *bt)
{
   word32 hash = 0;
   word32 stime = get32(bt->stime);
   word32 difficulty = get32(bt->difficulty);
   int seconds = (int) stime - get32(bt->time0);
   int highsolve = 284;
   int lowsolve = 143;

/* LEGACY ALGORITHM (reference only)

   const word32 trigger_block[2] = { V20TRIGGER };
   const word32 fix_trigger[2] = { V2001PATCH };

      I fear no man. But that thing...

   if(seconds < 0) return difficulty;
   if(cmp64(bt->bnum, trigger_block) < 0){
      hash = 0;
      highsolve = 506;
      lowsolve = 253;
   }
   else
      hash = (stime >> 6) ^ stime;
   if(cmp64(bt->bnum, fix_trigger) > 0) hash = 0;
   if(seconds > highsolve) {
      if(difficulty > 0) difficulty--;
      if(difficulty > 0 && (hash & 1)) difficulty--;
   } else if(seconds < lowsolve) {
      if((hash & 3) == 0  && difficulty < 255)
         difficulty++;
   }

      ... it scares me.

   */

   /* EPOCHALYPSE HANDLER */
   if (seconds < 0) return difficulty;

   if (cmp64(bt->bnum, CL64_32(V30TRIGGER)) >= 0) {
      highsolve = 138;
      lowsolve = 69;
   } else if (cmp64(bt->bnum, CL64_32(V20TRIGGER)) >= 0) {
      /* ... fear makes strangers of people who would be friends ... */
      if (cmp64(bt->bnum, CL64_32(V2001PATCH)) <= 0) {
         hash = (stime >> 6) ^ stime;
      }
      highsolve = 284;
      lowsolve = 143;
   } else {
      highsolve = 506;
      lowsolve = 253;
   }

   if (seconds > highsolve) {
      if (difficulty > 0) {
         if (hash & 1) difficulty--;
         difficulty--;
      }
   } else if(seconds < lowsolve) {
      if ((hash & 3) == 0 && difficulty < 255) {
         difficulty++;
      }
   }

   return difficulty;
}  /* next_difficulty() */

/**
 * Compute our weight at a lower block number of a given Tfile. Current
 * weight should be provided as if accurate to the latest Tfile trailer.
 * Serves as a better alternative to re-weighing the entire Tfile.
 * @param tfile Filename of Tfile to get weight from
 * @param bnum Pointer to bnum of desired past weight
 * @param weight Pointer to weight to subtract weight from
 * @return (int) value representing operation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 */
int past_weight(const char *tfile, const word8 bnum[8], word8 weight[32])
{
   FILE *fp;
   BTRAILER bt;
   long long seek;
   word8 subweight[32] = { 0 };

   /* open Tfile for reading */
   fp = fopen(tfile, "rb");
   if (fp == NULL) return VERROR;

   /* determine seek position */
   put64(&seek, bnum);
   seek = seek * sizeof(BTRAILER);
   /* seek to position of desired weight */
   if (fseek64(fp, seek, SEEK_SET) != 0) goto ERROR_CLEANUP;
   /* verify we are in the right place */
   if (fread(&bt, sizeof(BTRAILER), 1, fp) != 1) {
      if (ferror(fp)) goto ERROR_CLEANUP;
      /* Unexpected EOF if not error */
      set_errno(EMCM_EOF);
      goto ERROR_CLEANUP;
   } else if (cmp64(bt.bnum, bnum) != 0) {
      set_errno(EMCM_BNUM);
      goto ERROR_CLEANUP;
   }

   /* weigh every block trailer to EOF */
   while (fread(&bt, sizeof(BTRAILER), 1, fp) == 1) {
      /* Let the neo-genesis (not the 0x..ff) add weight to the chain. */
      if (bt.bnum[0] != 0xff) add_weight(subweight, bt.difficulty[0]);
   }
   /* check file errors and cleanup */
   if (ferror(fp)) goto ERROR_CLEANUP;
   fclose(fp);

   /* subtract accumulated past weight */
   if (multi_sub(weight, subweight, weight, 32)) {
      set_errno(EMCM_MATH64_OVERFLOW);
      return VERROR;
   }

   return VEOK;

   /* cleanup / error handling */
ERROR_CLEANUP:
   fclose(fp);

   return VERROR;
}  /* end past_weight() */

/**
 * Trim the provided Tfile to a specified block number.
 * @param highbnum Pointer to block number to trim Tfile to
 * @return (int) value representing operation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 * @todo migrate the embedded _WIN32 compatible ftruncate() MACRO to
 * a common library (e.g. extended-c -> extio.c)
 */
int trim_tfile(const char* tfile, const word8 highbnum[8])
{
   FILE *fp;
   BTRAILER bt;
   long long seek;

   fp = fopen(tfile, "r+b");
   if (fp == NULL) return VERROR;

   /* determine seek position */
   put64(&seek, highbnum);
   seek = seek * sizeof(BTRAILER);
   /* seek to position and read trailer (verify bnum) */
   if (fseek64(fp, seek, SEEK_SET) != 0) goto ERROR_CLEANUP;
   if (fread(&bt, sizeof(BTRAILER), 1, fp) != 1) {
      if (ferror(fp)) set_errno(EMCM_EOF);
      goto ERROR_CLEANUP;
   }
   /* verify trim position (highbnum) */
   if (cmp64(bt.bnum, highbnum) != 0) {
      set_errno(EMCM_BNUM);
      goto ERROR_CLEANUP;
   }

#ifdef _WIN32
   #define ftruncate(fd, len) _chsize_s(fd, len)
#endif

   /* truncate file at current position */
   if (ftruncate(fileno(fp), ftell64(fp)) != 0) goto ERROR_CLEANUP;

   /* cleanup */
   fclose(fp);

   return VEOK;

   /* cleanup / error handling */
ERROR_CLEANUP:
   fclose(fp);

   return VERROR;
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
   word32 difficulty, stime, time0;
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
      /* check tcount is within approved range */
      if (get32(bt->tcount) == 0) goto BAD_TCOUNT;
      if (get32(bt->tcount) > MAXBLTX) goto BAD_TCOUNT;
   }

   /* obtain frequently dereferenced values */
   difficulty = get32(bt->difficulty);
   stime = get32(bt->stime);
   time0 = get32(bt->time0);

   /* check time0, difficulty, mroot and stime... */
   if (bnum[0] > 0) {
      if (get32(bt->tcount) == 0) {
         /* ... PSEUDOBLOCK ONLY */

         /* check times of trouble... must equal BRIDGE seconds */
         if ((word32) (stime - time0) != get_bridge(bnum)) goto BAD_STIME;
         /* ... word32 boundary handles an Epochalypse event */
         /* check mroot is zero'd */
         if (!iszero(bt->mroot, HASHLEN)) {
            set_errno(EMCM_MROOT);
            return VERROR;
         }
      } else {
         /* ... STANDARD BLOCK ONLY */

         /* check time0 is not equal to stime */
         if (time0 == stime) goto BAD_STIME;
         /* check stime for times of trouble...
          * originally, pseudoblock generation was prohibited on the block
          * before a neogenesis block (0x...ff), and permitted in v2.4.1,
          * which was not given an official "break point" for comparison;
          * the last known (permitted) occurrence of a block exceeding the
          * BRIDGE time was block number 0x1b6ff, and so it shall be used
          */
         if (cmp64(bnum, CL64_32(0x1b6ff)) > 0 || (bnum[0] != 0xff
               && cmp64(bnum, CL64_32(V23TRIGGER)) > 0)) {
            /* check block time does not exceed BRIDGE seconds */
            if ((word32) (stime - time0) > get_bridge(bnum)) goto BAD_STIME;
            /* ... word32 boundary handles an Epochalypse event */
         }
      }
      /* ... STANDARD AND PSEUDOBLOCK */

      /* check stime rollover... */
      if (cmp64(bnum, CL64_32(V20TRIGGER)) > 0) {
         /* ... patched in v2.0 (V20TRIGGER) */
         if (time0 != get32(prev_bt->stime)) goto BAD_TIME0;
      }
      /* check difficulty is adjustment appropriately */
      if (difficulty != next_difficulty(prev_bt)) goto BAD_DIFF;
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
      if (time0 != get32(prev_bt->time0)) goto BAD_TIME0;
      if (difficulty != get32(prev_bt->difficulty)) goto BAD_DIFF;
      if (stime != get32(prev_bt->stime)) goto BAD_STIME;
   }

   /* check hash is valid for version 3.0 blocks */
   if (cmp64(bnum, CL64_32(V30TRIGGER)) >= 0) {
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
   int ecode, errnum;

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

#ifdef _OPENMP
   #pragma omp parallel private(bt)
#endif
   {
      while (!TfileShutdown && ecode == VEOK) {
      #ifdef _OPENMP
         #pragma omp critical
      #endif
         {
            if (fread(&bt, sizeof(BTRAILER), 1, fp) != 1) {
               ecode = VEWAITING;
               if (ferror(fp)) {
                  errnum = errno;
                  ecode = VERROR;
               }
            }
         }
         /* check for end of file */
         if (ecode != VEOK) continue;
         /* check for parameters not requiring PoW validation */
         if (bt.bnum[0] == 0 || get32(bt.tcount) == 0) continue;
         /* validate trailer Proof-of-Work */
         if (validate_pow(&bt) != VEOK) {
         #ifdef _OPENMP
            #pragma omp critical
         #endif
            {
               errnum = errno;
               ecode = VERROR;
               pdebug("PoW verification FAILURE on block %08x%08x",
                  get32(bt.bnum + 4), get32(bt.bnum));
            }
         }
      }
   }

   /* ensure errno integrity through parallel processing */
   if (ecode == VEWAITING) ecode = VEOK;
   if (ecode != VEOK) set_errno(errnum);

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
   /** @todo (DO NOT REMOVE) implement Tfile integrity pre-check to
    * verify the integrity of the Tfile up to v3.0 (sha256) -Dig
    */
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

   /* zero weight before weighing a whole file */
   memset(weight, 0, 32);

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
