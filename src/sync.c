/**
 * @file syncup.c
 * @date 11 February 2018 (Revised 2 Dec 2021)
 * @brief Blockchain initialization and synchronization support.
 * @copyright © Adequate Systems LLC, 2018-2021. All Rights Reserved.
 * <br />For more information, please refer to ../LICENSE
*/

#ifndef MOCHIMO_SYNCUP_C
#define MOCHIMO_SYNCUP_C  /* include guard */

#include "extio.h"
#include "extint.h"
#include "extthread.h"
#include <sys/types.h>
#include <dirent.h>  /* for scanning directories for files */

#include "config.h"
#include "network.h"
#include "types.h"
#include "data.c"

#define THREADS_MAX  64

typedef struct {
   volatile int tr;  /* thread function result -- set by thread */
   word8 bnum[8];    /* blockchain file to download */
   word32 ip;        /* source ip */
} BIP_THREAD_ARGS;

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
 * Reset chain data from local directory.
 * Find last block in `Bcdir` directory, and set Cblocknum, Eon,
 * Time0, Difficulty, Cblockhash and Prevhash from block trailer.
 * @return VEOK on success, else VERROR.
 */
int reset_chain(void)
{
   BTRAILER bt;
   word32 bnum[2], bchk[2];
   struct dirent *ep;
   DIR *dp;
   char *ext;
   char fname[FILENAME_MAX];
   char bcfname[FILENAME_MAX / 2] = "";

   /* find highest named blockchain file in Bcdir */
   dp = opendir(Bcdir);
   if (dp == NULL) return perrno(errno, "failed to open Bcdir...");
   else while((ep = readdir(dp))) {
      /* ensure valid blockchain file format (strlen("b*.bc") == 20) */
      if (ep->d_name[0] != 'b' || strlen(ep->d_name) != 20) continue;
      if ((ext = strrchr(ep->d_name, '.')) == NULL) continue;
      if (strncmp(ext, ".bc", FILENAME_MAX) != 0) continue;
      /* check if filename compares greater */
      if (strncmp(ep->d_name, bcfname, FILENAME_MAX) > 0) {
         /* ensure filename hexadecimal is exposed */
         if (sscanf(ep->d_name, "b%08x%08x", &bchk[1], &bchk[0]) == 2) {
            strncpy(bcfname, ep->d_name, FILENAME_MAX);
            bcfname[FILENAME_MAX - 1] = '\0';
            put64(bnum, bchk);
         }
      }
   }
   closedir(dp);

   /* read block trailer of file and ensure block numbers match */
   snprintf(fname, FILENAME_MAX, "%s/%s", Bcdir, bcfname);
   if (readtrailer(&bt, fname)) {
      return perr("failed to read block trailer, %s", fname);
   } else if (cmp64(bt.bnum, bnum)) return perr("%s bnum mismatch!", fname);

   /* initialize chain data from block trailer */
   put64(Cblocknum, bnum);
   Eon = get32(bnum) >> 8;
   Time0 = get32(bt.stime);
   Difficulty = set_difficulty(&bt);
   memcpy(Prevhash, bt.phash, HASHLEN);
   memcpy(Cblockhash, bt.bhash, HASHLEN);

   return VEOK;
}  /* end reset_chain() */

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
                     thread_multijoin(tid, tlen);
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
            psticky("Tfile check %.02f%% (0x%08x)", percent, get32(bt.bnum));
         } else {
            psticky("Validating Tfile %.2f%% (0x%08x) | Elapsed %gs",
               percent, get32(bt.bnum), difftime(time(NULL), start));
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
   if (tactive) thread_multijoin(tid, tlen);
   sub64(highblock, One, bnum);     /* fix high block number */
   memcpy(weight, highweight, HASHLEN);
   fclose(fp);
   pdebug("tf_val(): ecode = %d  bnum = 0x%s  weight = 0x...%x",
                  ecode, bnum2hex(highblock), highweight[0]);
tfval_end:
   psticky("");
   return ecode;
}  /* end tf_val() */


/* Delete all blocks above bc/matchblock.
 * Returns number of blocks deleted.
 */
int delete_blocks(word8 *matchblock)
{
   char fname[128];
   int j;
   word8 bnum[8];

   put64(bnum, matchblock);
   if(iszero(bnum,8)) add64(bnum, One, bnum);
   for(j = 0; ; j++) {
      sprintf(fname, "%s/b%s.bc", Bcdir, bnum2hex(bnum));
      if(unlink(fname) != 0) break;
      add64(bnum, One, bnum);
   }
   return j;
}


int trim_tfile(word8 *highbnum)
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


/* Extract Genesis Block to ledger.dat */
int extract_gen(char *lfile)
{
   char fname[128];

   sprintf(fname, "%s/b0000000000000000.bc", Bcdir);
   /* extract the ledger from our Genesis Block */
   return le_extract(fname, lfile);
}


/* Integrate reward function from block 0 to block bnum.
 * Return result in sum.
 */
word8 *get_treward(void *sum, void *bnum)
{
   word32 reward[2], bnum2[2];

   put64(bnum2, bnum);
   if(!iszero(bnum, 8)) {
      for(memset(sum, 0, 8); ;) {
         if(((word8 *) bnum2)[0]) {
            get_mreward(reward, bnum2);
            add64(sum, reward, sum);
         }
         if(sub64(bnum2, One, bnum2)) break;
      }
   }
   return sum;
}


#define NGBUFFLEN (16*1024)
#define NGERROR(e) { ecode = e; goto err; }

/* Check NG block:
 * 1. check hash is good and == Cblockhash
 * 2. not too much in amounts
 * 3. block hash is in tfile.dat
 *
 * Return 0 if NG is good, else error code.
 * (reset_chain() has already been called to set Cblockhash.)
 */
int check_ng(char *fname, word8 *bnum)
{
   static word32 premine[2]
      = { 0xbd1a6400, 0x0010e686 };  /* 4757066000000000 */
   static word32 tlen[2] = { sizeof(BTRAILER), 0 };
   word8 sum[8], sum2[8], temp[8];
   LENTRY le;
   BTRAILER bt;
   long toffset;
   word8 chash[HASHLEN];
   word8 bhash[HASHLEN];
   word8 buff[NGBUFFLEN];
   FILE *fp;
   unsigned long len;
   unsigned count, n;
   SHA256_CTX cctx;
   int ecode = 2;

   fp = fopen(fname, "rb");
   if(fp == NULL) return 1;
   if(fseek(fp, 0, SEEK_END)) {
err:
      fclose(fp);
      return ecode;
   }
   /* Read hash value in NG trailer */
   len = ftell(fp);
   if(len < (sizeof(BTRAILER) + sizeof(LENTRY))) NGERROR(3);
   if(fseek(fp, -(HASHLEN), SEEK_END)) NGERROR(4);
   if(fread(bhash, 1, HASHLEN, fp) != HASHLEN) NGERROR(5);
   /* Compute NG block hash */
   if(fseek(fp, 0, SEEK_SET)) NGERROR(6);
   sha256_init(&cctx);
   len -= HASHLEN;
   n = NGBUFFLEN;
   for( ; len; len -= count) {
      if(len < NGBUFFLEN) n = len;
      count = fread(buff, 1, n, fp);
      if(count < 1) break;
      sha256_update(&cctx, buff, count);
   }
   if(len) NGERROR(7);
   sha256_final(&cctx, chash);
   /* Check computed hash, chash, against hash from trailer, bhash. */
   if(memcmp(chash, bhash, HASHLEN) != 0) NGERROR(8);
   /* and the hash set by reset_chain(). */
   if(memcmp(chash, Cblockhash, HASHLEN) != 0) NGERROR(9);

   /* Compute total reward + premine into sum. */
   get_treward(sum, bnum);
   add64(premine, sum, sum);
   pdebug("premine: %lu  0x%lx\n", *((long *) premine), *((long *) premine));
   pdebug("sum:  %lu  0x%lx\n", *((long *) sum), *((long *) sum));
   /* Check sum of amounts in NG ledger. */
   fseek(fp, 4, SEEK_SET);
   for(memset(sum2, 0, 8); ; ) {
      if(fread(&le, 1, sizeof(LENTRY), fp) != sizeof(LENTRY)) break;
      /* add64(sum2, le.balance, sum2);
      if(cmp64(sum2, sum) > 0) NGERROR(10); */
   }
   pdebug("sum2: %lu  0x%lx\n", *((long *) sum2), *((long *) sum2));
   fclose(fp);

   /* Now check bnum's hash in trailer in tfile.dat */
   fp = fopen("tfile.dat", "rb");
   if(fp == NULL) return 11;
   put64(temp, bnum);
   mult64(temp, tlen, temp);
   if(sizeof(toffset) == 8) put64(&toffset, temp);
   if(sizeof(toffset) != 8) *((word32 *) &toffset) = *((word32 *) temp);
   if(fseek(fp, toffset, SEEK_SET)) NGERROR(12);
   if(fread(&bt, 1, sizeof(BTRAILER), fp) != sizeof(BTRAILER))
      NGERROR(13);
   if(memcmp(bt.bhash, Cblockhash, HASHLEN) != 0) NGERROR(14);
   fclose(fp);

   return 0;  /* success */
}  /* end check_ng() */

ThreadProc thread_get_block(void *arg)
{
   BIP_THREAD_ARGS *args = (BIP_THREAD_ARGS *) arg;
   char fname[FILENAME_MAX], fname2[FILENAME_MAX], bnumstr[24];
   int res;

   /* initialize */
   sprintf(fname, "b%.16s.tmp", val2hex64(args->bnum, bnumstr));
   sprintf(fname2, "b%.16s.dat", val2hex64(args->bnum, bnumstr));
   res = get_file(args->ip, args->bnum, fname);
   if (res == VEOK) {
      res = rename(fname, fname2);
      if (res != VEOK) {
         perrno(res, "catchup(): failed to move %s -> %s", fname, fname2);
         res = VERROR;
      }
   }

   remove(fname);
   args->tr = (res << 8) | 1;
   Unthread;
}

/**
 * Catch up by getting blocks from peers in plist[count].
 * Returns VEOK if updates made, else update() error code. */
int catchup(word32 plist[], word32 count)
{
   char fname[FILENAME_MAX], fname2[FILENAME_MAX], bnumstr[24];
   ThreadId tid[MAXQUORUM] = { 0 };
   BIP_THREAD_ARGS args[MAXQUORUM] = { 0 };
   word8 bnum[8], bclear[8];
   word32 i, n, done;
   FILE *fp;
   int res;

   /* initialize... */
   show("getblock");  /* get blockchain files */
   pdebug("catchup(%" P32u " peers): begin...", count);
   if ((res = mkdir_p(Bcdir))) {  /* ensure Bcdir is ready */
      perrno(res, "catchup(): failed to verify %s/ directory", Bcdir);
      return VERROR;
   }  /* fill args with peer ips */
   for(done = n = 0; n < MAXQUORUM && n < count; n++) args[n].ip = plist[n];

   /* download/validate/update blocks from args */
   put64(bclear, Cblocknum);
   while(Running && done < n) {
      for(put64(bnum, Cblocknum), done = i = 0; i < n; i++) {
         if (args[i].ip == 0) done++;
         else if (tid[i] > 0 && args[i].tr) {  /* thread finished */
            res = thread_join(tid[i]);
            if (res != VEOK) perrno(res, "catchup(): thread_join");
            if ((args[i].tr >> 8) != VEOK) args[i].ip = 0;  /* kick */
            args[i].tr = 0;
            tid[i] = 0;
         } else if (tid[i] == 0) {
            do {  /* determine next required block - skip neogenesis */
               add64(bnum, One, bnum);
               if (bnum[0] == 0) add64(bnum, One, bnum);
               sprintf(fname, "b%.16s.tmp", val2hex64(bnum, bnumstr));
               sprintf(fname2, "b%.16s.dat", val2hex64(bnum, bnumstr));
               if (cmp64(bnum, bclear) > 0) {
                  /* clear a safe path for the incoming blocks */
                  put64(bclear, bnum);
                  remove(fname2);
                  remove(fname);
               }  /* ... path is clear */
            } while(fexists(fname) || fexists(fname2));
            /* create file for child, so the children don't fight */
            fp = fopen(fname, "w");
            if (fp == NULL) perrno(errno, "catchup(): fopen(%s) failed", fname);
            else {
               fclose(fp);
               put64(args[i].bnum, bnum);
               res = thread_create(&(tid[i]), &thread_get_block, &args[i]);
               if (res != VEOK) {
                  perrno(res, "catchup(): thread_create");
                  args[i].tr = 0;
                  tid[i] = 0;
                  remove(fname);
               }
            }
         }
      }
      do {
         add64(Cblocknum, One, bnum);
         sprintf(fname2, "b%.16s.dat", val2hex64(bnum, bnumstr));
         if (fexists(fname2)) {
            res = update(fname2, 0);
            if (res != VEOK) {
               perr("catchup(): failed to update block file %s", fname2);
               /* wait for all threads to finish and return res */
               thread_multijoin(tid, n);
               return res;
            }
         }
      } while(Running && cmp64(Cblocknum, bnum) == 0);
      if(Dynasleep) usleep(Dynasleep);  /* small rest */
   }  /* end while(Running && done < n... download blocks */

   return VEOK;
}  /* end catchup() */

/**
 * Resynchronize blockchain up to network weight/bnum using quorum[qidx].
 * Returns VEOK on success, else restarts. */
int resync(word32 quorum[], word32 *qidx, void *highweight, void *highbnum)
{
   static word8 num256[8] = { 0, 1, };
   char ipaddr[16], fname[FILENAME_MAX];
   word8 bnum[8], weight[HASHLEN];
   int result;

   show("gettfile");  /* get tfile */
   pdebug("resync(): fetching tfile.dat from %s", ntoa(&quorum[0], ipaddr));
   while(Running && *quorum) {
      remove("tfile.dat");
      if(get_file(*quorum, NULL, "tfile.tmp") == VEOK) {
         if (rename("tfile.tmp", "tfile.dat") == 0) break;
         perrno(errno, "resync(): failed to rename tfile.dat");
      }
      /* remove quorum member, and try again */
      remove32(*quorum, quorum, *qidx, qidx);
   }
   if (!(*quorum)) restart("gettfile no quorum");
   if (!Running) resign("gettfile exiting");

   show("tfval");  /* validate tfile */
   if (tf_val("tfile.dat", bnum, weight, 0)) return VERROR;
   else pdebug("resync(): tfile.dat is valid.");
   if (cmp256(weight, highweight) >= 0 && cmp64(bnum, highbnum) >= 0) {
      pdebug("resync(): tfile.dat matches advertised bnum and weight.");
   } else return VERROR;
   if (!(*quorum)) restart("tfval no quorum");
   if (!Running) resign("tfval exiting");

   show("getneo");  /* get neo-genesis block */
   /* determine starting neo-genesis block */
   put64(bnum, highbnum); bnum[0] = 0;
   if (sub64(bnum, num256, bnum)) memset(bnum, 0, 8);
   pdebug("resync(): neo-genesis block 0x%s", bnum2hex(bnum));
   /* clean bc/ directory of block >= ngnum */
   delete_blocks(bnum);
   /* trim the tfile back to the neo-genesis block and close the ledger */
   if (trim_tfile(bnum) != VEOK) restart("getneo tfile_trim()");  /* panic */
   le_close();  /* close ledger, we're gonna grab a new one... */
   /* download neo-genesis block if no backup */
   if(!iszero(bnum, 8)) {  /* ... no need to download genesis block */
      plog("init(): downloading neo-genesis block 0x%s", bnum2hex(bnum));
      while(Running && *quorum) {
         remove("ngblock.dat");
         if(get_file(*quorum, bnum, "ngblock.dat") == VEOK) break;
         /* remove quorum member, and try again */
         remove32(*quorum, quorum, *qidx, qidx);
      }
      if (!(*quorum)) restart("getneo no quorum");
      if (!Running) resign("getneo exiting");
      /* transfer neo-genesis block to bcdir */
      sprintf(fname, "%.106s/b%.16s.bc", Bcdir, bnum2hex(bnum));
      if(rename("ngblock.dat", fname) != 0) {
         perrno(errno, "init(): cannot move neo-genesis to %s", fname);
         return VERROR;
      }
      /* extract ledger from neo-genesis block... */
      if(le_extract(fname, "ledger.dat") != VEOK) {
         restart("getneo ledger extraction");
      }  /* ... or from genesis block */
   } else extract_gen("ledger.dat");

   show("setdiff");  /* setup difficulty, based on [neo]genesis block */
   if(reset_chain() != VEOK) restart("setdiff reset");
   le_open("ledger.dat", "rb");

   show("checkneo");  /* check neo-genesis hash against Cblockhash */
   if(!iszero(bnum, 8)) {  /* Cblockhash was set by reset_chain() */
      result = check_ng(fname, bnum);
      if(result != 0) {
         plog("init(): Bad NG block! ecode: %d", result);
         remove(fname);
         return VERROR;
      }
   }

   /* get blockchain */
   if (catchup(quorum, *qidx) != VEOK) {
      plog("resync(): catchup() encountered an error, restarting...");
      restart("catchup error");
   }

   /* Post-sync hook for external SQL database export */
   /* Shell script in /bin directory */
   if(Exportflag && fexists("../init-external.sh")) {
     plog("Calling ../init-external.sh\n");  /* first time call */
     system("../init-external.sh");
   }

   if(!Running) resign("quorum update");

   /* Re-compute Weight[].
    * Check tf_val() set bnum to high block number on chain */
   tf_val("tfile.dat", bnum, weight, 1);
   memcpy(Weight, weight, HASHLEN);
   if(cmp64(bnum, Cblocknum) != 0) {
      perr("init(): block number mismatch!");  /* should not happen */
      restart("tfval_last error");
   }

   pdebug("re-computed Weight = 0x...%x", Weight[0]);
   plog("\nVeronica says, 'You're done!'");

   /* Done! */
   return VEOK;
}

/* Pull a divergent block chain and merge it into ours
 * rather than bailing out to contention!
 * Always returns VEOK to ignore contention.
 * splitblock is where the two chains diverge.
 * txcblock is the advertised block of peer,
 * and peerip is its ip address.
 */
int syncup(word32 splitblock, word8 *txcblock, word32 peerip)
{
   word8 bnum[8], tfweight[HASHLEN], saveweight[HASHLEN];
   static word32 lastneo[2], sblock[2];
   char buff[256];
   int j;
   NODE *np2;
   time_t lasttime;

   Insyncup = 1;
   show("syncup");
   if(Bcpid) { /* Wait for block constructor to exit... */
      pdebug("syncup(): Waiting for bcon to exit...");
      kill(Bcpid, SIGTERM);
      waitpid(Bcpid, NULL, 0);
      Bcpid = 0;
   }

   /* Stop sending update blocks, since we're behind */
   if(Sendfound_pid) {
      pdebug("syncup(): Killing send_found()...");
      kill(Sendfound_pid, SIGTERM);
      waitpid(Sendfound_pid, NULL, 0);
      Sendfound_pid = 0;
   }

   /* Stop block transfer children and others */
   for(np2 = Nodes; np2 < Hi_node; np2++) {
      if(np2->pid == 0) continue;
      kill(np2->pid, SIGTERM);
      waitpid(np2->pid, NULL, 0);
      freeslot(np2);
   }

   /* Close server ledger */	
   pdebug("syncup(): beginning state save...");
   le_close(); 

   /* Backup TFILE, Ledger, and blocks to split-tree directory. */
   /* system("mkdir split"); * already exists */
   pdebug("syncup(): Backing up TFILE, ledger.dat, and blocks...");
   system("rm split/*");
   system("cp tfile.dat split");
   system("cp ledger.dat split");
   system("mv bc/*.bc split");
   memcpy(saveweight, Weight, HASHLEN);

   sblock[0] = splitblock;
   /* Compute first previous NG block */
   lastneo[0] = (get32(Cblocknum) & 0xffffff00) - 256;
   pdebug("syncup(): Identified first previous NG block as %s",
                  bnum2hex((word8 *) &lastneo));

   /* Delete Ledger and trim T-File */
   if(unlink("ledger.dat") != 0) {
      pdebug("syncup() failed!  Unable to delete ledger.dat");
      goto badsyncup;
   }
   if(trim_tfile((word8 *) &lastneo) != VEOK) {
      pdebug("syncup(): T-File trim failed!");
      goto badsyncup;
   }

   /* Extract first previous Neogenesis Block to ledger.dat */
   pdebug("syncup(): Expanding Neo-genesis block to ledger.dat...");
   sprintf(buff, "cp split/b%s.bc bc/b%s.bc", bnum2hex((word8 *) &lastneo), 
           bnum2hex((word8 *) &lastneo));
   system(buff);
   sprintf(buff, "bc/b%s.bc", bnum2hex((word8 *) &lastneo));
   if(le_extract(buff, "ledger.dat") != VEOK) {
      pdebug("syncup(): failed!  Unable to extract ledger!");
      goto badsyncup;
   }

   /* setup Difficulty and globals, based on neogenesis block */
   if(reset_chain() != VEOK) {
      pdebug("syncup(): failed!  reset_chain() failed!");
      goto badsyncup;
   }
   le_open("ledger.dat", "rb");

   pdebug("Split point is block %s", bnum2hex((word8 *) &sblock));
   add64(lastneo, One, bnum);
   for( ;cmp64(bnum, sblock) < 0; ) {
      pdebug("syncup(): Copying split/b%s.bc to spblock.tmp",
                     bnum2hex(bnum));
      sprintf(buff, "cp split/b%s.bc spblock.tmp", bnum2hex(bnum));
      system(buff);
      if(update("spblock.tmp", 1) != VEOK) {
         pdebug("syncup(): failed to update our own block.");
         goto badsyncup;
      }
      add64(bnum, One, bnum);
      if(bnum[0] == 0) add64(bnum, One, bnum);  /* skip NG blocks */
   }

   /* Download missing blocks from peer. */
   pdebug("Download and update missing blocks from peer...");
   put64(bnum, sblock);
   for(j = 0; ; ) {
      if(bnum[0] == 0) add64(bnum, One, bnum);  /* skip NG blocks */
      sprintf(buff, "b%s.bc", bnum2hex(bnum));
      if(j == 60) {
         pdebug("syncup(): failed while downloading %s from %s",
                        buff, ntoa(&peerip, NULL));
         goto badsyncup;
      }
      lasttime = time(NULL);
      if(get_file(peerip, bnum, buff) != VEOK) {
         if(cmp64(bnum, txcblock) >= 0) break;  /* success */
         if(time(NULL) == lasttime) sleep(1);
         j++;  /* retry counter */
         continue;
      }
      if(update(buff, 0) != VEOK) {
         pdebug("syncup(): cannot update peer's block.");
         goto badsyncup;
      }
      add64(bnum, One, bnum);
   }
   system("cp split/b0000000000000000.bc bc");
   system("rm split/*");
   /* re-compute tfile weight */
   if(tf_val("tfile.dat", bnum, tfweight, 1)) {
      plog("syncup(): tf_val() error");
   } else plog("syncup() is good!");
   memcpy(Weight, tfweight, HASHLEN);
   Insyncup = 0;
   return VEOK;

badsyncup:
   /* Restore block chain from saved state after a bad re-sync attempt. */
   pdebug("syncup(): bad sync: restoring saved state...");
   le_close();
   system("mv split/tfile.dat .");
   system("mv split/ledger.dat .");
   system("rm *.bc bc/*");
   system("mv split/* bc");
   reset_chain();  /* reset Difficulty and others */
   memcpy(Weight, saveweight, HASHLEN);
   le_open("ledger.dat", "rb");
   Insyncup = 0;
   return VEOK;
}  /* end syncup() */

/* end include guard */
#endif
