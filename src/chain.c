/**
 * @private
 * @headerfile chain.h <chain.h>
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_CHAIN_C
#define MOCHIMO_CHAIN_C


#include "chain.h"

/* internal support */
#include "trigg.h"
#include "peach.h"
#include "error.h"

/* external support */
#include <string.h>
#include "extmath.h"
#include "extlib.h"

/* Accumulate weight based on difficulty */
void add_weight(word8 *weight, word8 difficulty, word8 *bnum)
{
   static word32 trigger[2] = { V20TRIGGER, 0 };
   word8 add256[32] = { 0 };

   /* trigger block shifts weight increment from linear to exponential */
   if(bnum && cmp64(bnum, trigger) < 0) add256[0] = difficulty;
   else add256[difficulty / 8] = 1 << (difficulty % 8);  /* 2 ** difficulty */
   multi_add(weight, add256, weight, 32);
}  /* end add_weight() */

/**
 * Append FILE pointer data to a Tfile.
 * @param fp FILE pointer containing data to append
 * @param tfilename Filename of a Trailer file
 * @return (int) value representing operation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int append_tfile_fp(FILE *fp, char *tfilename)
{
   BTRAILER bt;
   FILE *tfp;

   tfp = fopen(tfilename, "ab");
   if (tfp == NULL) return VERROR;
   while (fread(&bt, sizeof(bt), 1, fp) == 1) {
      if (fwrite(&bt, sizeof(bt), 1, tfp) != 1) goto FAIL;
   }
   fclose(tfp);

   return VEOK;

/* error handling */
FAIL: fclose(tfp); return VERROR;
}  /* end append_tfile_fp() */

/**
 * Append the (block) Trailer of a specified filename, to a Tfile.
 * @param filename Filename of a Blockchain file
 * @param tfilename Filename of a Trailer file
 * @return (int) value representing operation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int append_tfile(char *filename, char *tfilename)
{
   FILE *fp;
   int ecode;

   fp = fopen(filename, "rb");
   if (fp == NULL) return VERROR;
   ecode = fseek64(fp, -(sizeof(BTRAILER)), SEEK_END) == 0 ? VEOK : VERROR;
   if (ecode == VEOK) ecode = append_tfile_fp(fp, tfilename);
   fclose(fp);

   /* return ecode */
   return ecode;
}  /* end append_tfile() */

/**
 * Compute the mining reward for a specified block number.
 * @param reward Pointer to place 64-bit reward value into
 * @param bnum Pointer to 64-bit block number of reward
 */
void get_mreward(void *reward, void *bnum)
{
   static word32 one[2] = { 1, 0 };
   static word32 base1[2] = { 0x2A05F200, 1 };     /* base 5000000000 */
   static word32 base2[2] = { 0x60b43c80, 1 };     /* base 5917392000 */
   static word32 base3[2] = { 0xdbe74670, 0x0d };  /* base 59523942000 */
   static word32 delta1[2] = { 0xDAC0, 0 };        /* reward delta 56000 */
   static word32 delta2[2] = { 150000, 0 };        /* increment */
   static word32 delta3[2] = { 28488, 0 };         /* decrement */
   static word32 t1[2] =  { V20TRIGGER, 0 };       /* new reward block */
   static word32 t2[2] =  { 373761, 0 };           /* mid block */
   static word32 t3[2] =  { 2097152, 0 };          /* final reward block */
   word8 bnum2[8];

   if (cmp64(bnum, t1) < 0) {
      /* bnum < 17185 */
      if (sub64(bnum, one, bnum2)) {
         /* underflow, no reward */
         memset(reward, 0, 8);
      } else {
         mult64(delta1, bnum2, reward);
         add64(reward, base1, reward);
      }
   } else if (cmp64(bnum, t2) < 0) {
      /* first 4 years (excl. bnum[0... 17184]) */
      sub64(bnum, t1, bnum2);
      mult64(delta2, bnum2, reward);
      add64(reward, base2, reward);
   } else if (cmp64(bnum, t3) <= 0) {
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
 * Calculate the sum of block rewards from a Tfile.
 * @param fname Filename of Tfile to calculate rewards from
 * @param rewards Pointer to place sum of block rewards
 * @param bnum Pointer to block number of last reward or NULL for no limit
 * @return (int) value representing operation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 */
int get_tfrewards(char *fname, void *rewards, void *bnum)
{
   /* premine value = 4757066000000000 */
   static word32 premine[2] = { 0xbd1a6400, 0x0010e686 };

   BTRAILER bt;
   FILE *fp;
   word8 reward[8];

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
         get_mreward(reward, bt.bnum);
         if (add64(rewards, reward, rewards)) goto FAIL_IO_OVERFLOW;
      }
   }

   /* close Tfile */
   fclose(fp);

   /* success */
   return VEOK;

FAIL_INVAL: set_errno(EINVAL); return VERROR;
FAIL_IO_OVERFLOW: set_errno(EMCM_MREWARDS_OVERFLOW);
FAIL_IO:
   fclose(fp);
   return VERROR;
}  /* end get_tfrewards() */

/**
 * Compute the next difficulty based on data from a Block Trailer.
 * @param btp Pointer to Block Trailer for calculating next difficulty
 * @return (word32) 32-bit unsigned value representing the next difficulty
 */
word32 next_difficulty(BTRAILER *btp)
{
   static word32 trigger_block[2] = { V20TRIGGER, 0 };
   static word32 fix_trigger[2] = { P20TRIGGER, 0 };

   word32 hash = 0;
   word32 stime = get32(btp->stime);
   word32 difficulty = get32(btp->difficulty);
   int seconds = stime - get32(btp->time0);
   int highsolve = 284;  /* post v2.0 */
   int lowsolve = 143;  /* post v2.0 */

   /* no difficulty change during an Epochalypse, Y2K38 */
   if (seconds < 0) return difficulty;

   /* check trigger blocks for parameter changes */
   if (cmp64(btp->bnum, trigger_block) < 0) {
      /* pre-v2.0 high/low thresholds */
      highsolve = 506;
      lowsolve = 253;
   } else if (cmp64(btp->bnum, fix_trigger) <= 0) {
      /* anomalous modifier -- exists only as a [ REDACTED ] */
      hash = (stime >> 6) ^ stime;
   }

   /* compare solve seconds against appropriate thresholds */
   if (seconds > highsolve) {
      if (difficulty > 0) {
         if (hash & 1) difficulty--;
         difficulty--;
      }
   } else if (seconds < lowsolve) {
      if ((hash & 3) == 0 && difficulty < 255) difficulty++;
   }

   /* return modified difficulty value */
   return difficulty;
} /* end next_difficulty() */

/**
 * Print trailer information (typically on block update).
 * @param btp Pointer to block trailer with data to print
 */
void ptrailer(BTRAILER *btp)
{
   long long blocknumber;
   word32 btxs, btime, bdiff;
   char str[256], *haiku1, *haiku2, *haiku3;

   /* prepare block stats */
   blocknumber = 0;
   put64(&blocknumber, btp->bnum);
   btxs = get32(btp->tcount);
   bdiff = get32(btp->difficulty);
   btime = get32(btp->stime) - get32(btp->time0);
   /* print haiku on solved block, else  print block type */
   if (btxs) {
      /* expand and split haiku into lines for printing */
      trigg_expand(btp->nonce, str);
      haiku1 = strtok(str, "\n");
      haiku2 = strtok(&haiku1[strlen(haiku1) + 1], "\n");
      haiku3 = strtok(&haiku2[strlen(haiku2) + 1], "\n");
      print("\n/) %s\n(=: %s\n\\) %s\n", haiku1, haiku2, haiku3);
   } else if (btp->bnum[0]) print("\n(=: pseudo-block :=)\n");
   else if (!iszero(btp->bnum, 8)) print("\n(=: neogenesis-block :=)\n");
   else print("\n(=: genesis-block :=)\n");
   /* print block identification adn details */
   print("0x%s(%lld)#%s\n", bnum2hex(btp->bnum, str), blocknumber,
      addr2hex(btp->bhash, &str[18]));
   print("Time: %us, Diff: %u, Txs: %u\n", btime, bdiff, btxs);
}  /* end ptrailer() */

/**
 * Read the block number value from the trailer of a blockchain file.
 * @param bnum Pointer to buffer to place block number value
 * @param filename Filename of block to read from
 * @return (int) value representing operation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 */
int read_bnum(void *bnum, char *filename)
{
   FILE *fp;

   /* open file and read bnum in BTRAILER */
   fp = fopen(filename, "rb");
   if (fp == NULL) return VERROR;
   if (fseek64(fp, -(sizeof(BTRAILER) - HASHLEN), SEEK_END)) goto FAIL_IO;
   if (fread(bnum, 8, 1, fp) != 1) goto FAIL_IO;
   fclose(fp);

   /* success */
   return VEOK;

/* error handling */
FAIL_IO:
   fclose(fp);
   return VERROR;
}  /* end read_bnum() */

/**
 * Read the header length value from a blockchain file.
 * @param hdrlen Pointer to buffer to place headerlen value
 * @param filename Filename of block to read from
 * @return (int) value representing operation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 */
int read_hdrlen(void *hdrlen, char *filename)
{
   FILE *fp;

   /* open file and read bnum in BTRAILER */
   fp = fopen(filename, "rb");
   if (fp == NULL) return VERROR;
   if (fread(hdrlen, sizeof(word32), 1, fp) != 1) {
      if (feof(fp)) set_errno(EMCM_EOF);
      fclose(fp);
      return VERROR;
   }
   fclose(fp);

   /* success */
   return VEOK;
}  /* end read_hdrlen() */

/**
 * Read Tfile data into a buffer.
 * @param buffer Pointer to buffer to read Tfile data into
 * @param bnum Block number at which to start reading from Tfile
 * @param count Number of trailers to read from Tfile
 * @return (int) number of records read from Tfile, which may be less
 * than count if an error ocurrs. Check errno for details.
*/
int read_tfile(void *buffer, void *bnum, int count, char *tfname)
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
int read_trailer(BTRAILER *btp, char *fname)
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

/**
 * Trim the Tfile to a specified block number (inclusive).
 * @param highbnum Pointer to 64-bit block number to trim Tfile to
 * @param weight Pointer to place weight of trimmed Tfile
 * @return (int) value representing operation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 */
int trim_tfile(char *tfname, void *highbnum, void *weight)
{
   static word32 one[2] = { 1, 0 };

   BTRAILER bt;
   FILE *fp, *fpout;
   word8 bnum[8];
   char tmpfname[FILENAME_MAX];

   /* init */
   put64(bnum, highbnum);
   snprintf(tmpfname, FILENAME_MAX, "%s.tmp", tfname);

   /* open source file */
   fp = fopen(tfname, "rb");
   if (fp == NULL) return VERROR;

   /* open (tmp) destination file */
   fpout = fopen(tmpfname, "wb");
   if (fpout == NULL) goto FAIL_IO;
   /* perform Tfile read and rewrite */
   while (fread(&bt, sizeof(bt), 1, fp)) {
      if (weight && bt.bnum[0] != 0xff) {
         add_weight(weight, bt.difficulty[0], bt.bnum);
      }
      if (fwrite(&bt, sizeof(bt), 1, fpout) != 1) break;
      if (iszero(bnum, 8) || sub64(bnum, one, bnum)) break;
   }
   /* check file errors -- close */
   if (ferror(fpout) || ferror(fpout)) goto FAIL_IO2;
   fclose(fpout);
   fclose(fp);

   /* check read/write iteration completed without errors */
   if (iszero(bnum, 8)) {
      if (remove(tfname) || rename(tmpfname, tfname)) return VERROR;
   }

   /* delete temp files */
   remove(tmpfname);

   /* success */
   return VEOK;

FAIL_IO2:
   fclose(fpout);
FAIL_IO:
   fclose(fp);
   return VERROR;
}  /* end trim_tfile() */

/**
 * Validate the Proof of Work of a Block Trailer.
 * @param btp Pointer to Block Trailer to validate
 * @returns VEOK on success, VERROR on PoW failure, or VEBAD on
 * "Boxing Day Anomaly" blockhash failure.
*/
int validate_pow(BTRAILER *btp)
{
   static word32 v24trigger[2] = { V24TRIGGER, 0 };
   /* Adding constants to allow skipping PoW validation of Boxing Day's
    * corrupted block, provided the blockhash matches said constants.
    * See "Boxing Day Anomaly" on [ REDACTED ] for more details. */
   static word32 boxingday[2] = { 0x52d3c, 0 };
   static word8 boxdayhash[32] = {
      0x2f, 0xfa, 0xb9, 0xb9, 0x00, 0xe1, 0xbc, 0xa8,
      0x25, 0x19, 0x20, 0xc2, 0xdd, 0xf0, 0x46, 0xb8,
      0x07, 0x44, 0x2a, 0xbb, 0xfa, 0x5e, 0x94, 0x51,
      0xb0, 0x60, 0x03, 0xcc, 0x82, 0x2d, 0xb1, 0x12
   };

   /* skip where PoW N/A */
   if (get32(btp->tcount) == 0) return VEOK;

   if (cmp64(btp->bnum, v24trigger) > 0) {
      /* v2.4 PoW uses Peach algo... */
      if (peach_check(btp) == VEOK) return VEOK;
      /* check Boxing Day Anomaly on PoW failure -- Bugfix */
      if (cmp64(btp->bnum, boxingday) == 0) {
         if (memcmp(btp->bhash, boxdayhash, 32) == 0) return VEOK;
         return VEBAD;
      }
      /* ... else, pre-v2.4 PoW uses Trigg algo */
   } else return trigg_check(btp);

   /* PoW failure */
   return VERROR;
}  /* end validate_pow() */

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
      if (get32(btp->difficulty) != next_difficulty(pbtp)) goto BAD_DIFF;
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
      if (stime > start && (stime - start) > STIME_VARIANCE) goto BAD_STIME;
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

/**
 * Validate an opened Trailer file (excludes PoW validation).
 * @param tfp Open Tfile FILE pointer to validate
 * @param bnum Pointer to place validated bnum
 * @param weight Pointer to add validated weight
 * @param part Flag indicating validation of a partial Tfile.
 * When set, skips validation of the first trailer entry.
 * @return (int) value representing operation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int validate_tfile_fp(FILE *tfp, void *bnum, void *weight, int part)
{
   BTRAILER bt, bt_prev, *btp;
   long long filelen;
   int ecode;

   /* init */
   btp = NULL;
   ecode = VEOK;

   /* check length of Tfile */
   fseek64(tfp, 0LL, SEEK_END);
   filelen = ftell64(tfp);
   if ((filelen % sizeof(BTRAILER)) != 0) {
      set_errno(EMCM_FILELEN);
      return VERROR;
   }
   rewind(tfp);

   /* skip Genesis validation on partial Tfile */
   if (part) fread((btp = &bt_prev), sizeof(bt), 1, tfp);

   /* validate every block trailer against previous
    * NOTE: Genesis block trailer is validated by itself */
   while (fread(&bt, sizeof(bt), 1, tfp)) {
      ecode = validate_trailer(&bt, btp);
      if (ecode) return ecode;
      /* update highest block number and cumulative chain weight */
      if (bnum) put64(bnum, bt.bnum);
      /* Let the neo-genesis (not the 0x..ff) add weight to the chain. */
      if (weight && bt.bnum[0] != 0xff) {
         add_weight(weight, bt.difficulty[0], bt.bnum);
      }
      /* shift block trailer (sets btp) */
      memcpy((btp = &bt_prev), &bt, sizeof(bt));
   }

   /* check for IO errors */
   if (ferror(tfp)) return VERROR;
   /* else EOF */

   /* tfile valid */
   return VEOK;
}  /* end validate_tfile_fp() */

/**
 * Validate a Trailer file (excludes PoW validation).
 * @param tfname Filename of Tfile to validate
 * @param bnum Pointer to place validated bnum
 * @param weight Pointer to add validated weight
 * @return (int) value representing operation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int validate_tfile(char *tfname, void *bnum, void *weight)
{
   FILE *tfp;
   int ecode;

   tfp = fopen(tfname, "rb");
   if (tfp == NULL) return VERROR;
   ecode = validate_tfile_fp(tfp, bnum, weight, 0);
   fclose(tfp);

   return ecode;
}  /* end validate_tfile() */

/**
 * Get the weight of a Trailer file. Trailer file is assumed Valid.
 * @param tfname Filename of Tfile to get weight from
 * @param bnum Pointer to bnum of last weight to add, or NULL
 * @param weight Pointer to add weight
 * @return (int) value representing operation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int weigh_tfile(char *tfname, void *bnum, void *weight)
{
   BTRAILER bt;
   FILE *tfp;

   tfp = fopen(tfname, "rb");
   if (tfp == NULL) return VERROR;

   /* weigh every block trailer */
   while (fread(&bt, sizeof(bt), 1, tfp)) {
      /* Let the neo-genesis (not the 0x..ff) add weight to the chain. */
      if (bt.bnum[0] != 0xff) {
         add_weight(weight, bt.difficulty[0], bt.bnum);
      }
      /* break when we reach specified bnum */
      if (bnum && cmp64(bnum, bt.bnum) <= 0) break;
   }
   fclose(tfp);

   /* tfile weighed */
   return VEOK;
}  /* end weigh_tfile() */

/* end include guard */
#endif
