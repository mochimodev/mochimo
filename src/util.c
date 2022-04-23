/**
 * util.c - Mochimo specific utilities support
 *
 * Copyright (c) 2018-2021 Adequate Systems, LLC. All Rights Reserved.
 * For more information, please refer to ../LICENSE
 *
 * Date: 2 January 2018
 * Revised: 28 October 2021
 *
*/

/* include guard */
#ifndef MOCHIMO_UTIL_C
#define MOCHIMO_UTIL_C


#include "util.h"

#ifndef _WIN32
   #include <sys/file.h>  /* for flock() */
   #include <unistd.h>  /* for open() & close() */

#endif

#include <string.h>  /* for memory handling */
#include <stdlib.h>  /* for system() */
#include <time.h>    /* for time_t */
#include <errno.h>   /* for errno */

#include "extint.h"
#include "extio.h"
#include "extlib.h"
#include "extmath.h"
#include "extprint.h"

#include "crc16.h"

#include "peach.h"
#include "trigg.h"

#include "data.c"

#ifndef _WIN32

/* Get exclusive lock on lockfile.
 * Returns: -1 if lock not made within 'seconds' or open() failed
 *          else a descriptor to be used with unlock()
 */
int lock(char *lockfile, int seconds)
{
   time_t timeout;
   int fd, status;

   timeout = time(NULL) + seconds;
   fd = open(lockfile, O_NONBLOCK | O_RDONLY);
   if(fd == -1) return -1;
   for(;;) {
      status = flock(fd, LOCK_EX | LOCK_NB);
      if(status == 0) return fd;
      if(time(NULL) >= timeout) {
         close(fd);
         return -1;
      }
   }
}

/* Unlock a decriptor returned from lock() */
int unlock(int fd)
{
   int status;

   status =  flock(fd, LOCK_UN);
   close(fd);
   return status;
}

#endif

void crctx(TX *tx)
{
   put16(tx->crc16, crc16((word8 *) tx, sizeof(TX) - (2+2)));
}

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
      perrno(errno, "readtrailer() failed on fopen() for %s", fname);
      return VERROR;
   } else {
      seekerr = fseek(fp, -(sizeof(BTRAILER)), SEEK_END);
      if (seekerr == 0) count = fread(trailer, 1, sizeof(BTRAILER), fp);
      fclose(fp);
   }
   if(seekerr) {
      perr("readtrailer() failed on fseek() for %s: ecode=%d", fname, seekerr);
      return VERROR;
   }
   if(count != sizeof(BTRAILER)) {
      perr("readtrailer() failed on fread() for %s: read %zu/%zu bytes",
         fname, count, sizeof(BTRAILER));
      return VERROR;
   }
   return VEOK;
}

char *val2hex64(void *val, char hex[])
{
   word8 *bp = (word8 *) val;
   sprintf(hex, "%02x%02x%02x%02x%02x%02x%02x%02x",
      bp[7], bp[6], bp[5], bp[4], bp[3], bp[2], bp[1], bp[0]);
   return hex;
}

/* bnum is little-endian on disk and core. */
char *bnum2hex(void *bnum)
{
   static char buff[18];
   word8 *bp;

   bp = (word8 *) bnum;
   sprintf(buff, "%02x%02x%02x%02x%02x%02x%02x%02x",
      bp[7], bp[6], bp[5], bp[4], bp[3], bp[2], bp[1], bp[0]);
   return buff;
}

/* bnum is little-endian on disk and core. */
#define weight2hex(_weight)   val2hex(_weight, 32, NULL, 0)
char *val2hex(void *val, int len, char *buf, int bufsize)
{
   static char str[20];
   unsigned char *bp;
   char *cp;
   int elip;

   /* fault protections */
   if (buf == NULL) {
      bufsize = sizeof(str);
      buf = str;
   } else if (bufsize < 8) {
      if (bufsize < 4) *buf = '\0';
      else strncpy(buf, "...", 4);
      return buf;
   }

   /* initialize - trim leading zeros and set elipsis position */
   bp = (unsigned char *) val;
   for(elip = 0; len > 0 && bp[len - 1] == 0; len--);
   if (--bufsize < (len * 2)) elip = len - ((bufsize - 3) / 4);
   /* print value, respecting elipsis */
   for (cp = buf; len > 0; len--, cp += 2) {
      if (len <= elip) {
         sprintf(cp, "...");
         len = bufsize / 4;
         elip = 0;
         cp += 3;
      }
      sprintf(cp, "%02x", bp[len - 1]);
   }  /* ensure null-termination */
   cp = '\0';

   return buf;
}

char *addr2str(void *addr)
{
   static char str[10];
   word8 *bp;

   bp = (word8 *) addr;
   sprintf(str, "%02x%02x%02x%02x", bp[0], bp[1], bp[2], bp[3]);
   return str;
}


/* Return static printable string of hash[HASHLEN] input */
char *hash2str(word8 *hash)
{
   static char s[(HASHLEN*2)+4];
   int n;
   char *cp;

   for(cp = s, n = 0; n < HASHLEN; n++, cp += 2)
      sprintf(cp, "%02x", *hash++);
   return s;
}


int moveublock(char *ublock, word8 *newnum)
{
   char buff[256];
   char cmd[288];
   char *bnum;

   bnum = bnum2hex(newnum);
   sprintf(buff, "b%s.bc", bnum);
   sprintf(cmd, "%s/b%s.bc", Bcdir, bnum);
   if(fexists(buff) || fexists(cmd)) {
      perr("moveublock() failed: %s already exists!", buff);
      return VERROR;
   }
   if(rename(ublock, buff) != 0) {
      perrno(errno, "moveublock() failed on rename() %s to %s", ublock, buff);
      return VERROR;
   }
   sprintf(cmd, "mv %s %s", buff, Bcdir);
   if (system(cmd)) return VERROR;
   sprintf(buff, "%s/b%s.bc", Bcdir, bnum);
   if(!fexists(buff)) {
      perr("moveublock() failed on system(%s): %s missing", cmd, buff);
      return VERROR;
   }
   return VEOK;
}

/* Read in common global data */
int read_global(void)
{
   FILE *fp;
   size_t count;

   fp = fopen("global.dat", "rb");
   if (fp == NULL) {
      perrno(errno, "read_global() failed on fopen() for global.dat");
      return VERROR;
   } else {
      count = 0;
      count += fread(Cblocknum,    1,  8, fp);
      count += fread(Cblockhash,   1, 32, fp);
      count += fread(Prevhash,     1, 32, fp);
      count += fread(&Peerip,      1,  4, fp);
      count += fread(&Mfee,        1,  8, fp);
      count += fread(&Difficulty,  1,  4, fp);
      count += fread(&Time0,       1,  4, fp);
      count += fread(&Bgflag,      1,  1, fp);
      fclose(fp);
   }
   if(count != (8+32+32+4+8+4+4+1)) {
      perr("read_global() failed on fread() for %s: read %zu/%zu bytes",
         "global.dat", count, (size_t) (8+32+32+4+8+4+4+1));
      return VERROR;
   }
   return VEOK;
}  /* end read_global() */


/* Write out common global data */
int write_global(void)
{
   FILE *fp;
   size_t count;

   fp = fopen("global.dat", "wb");
   if (fp == NULL) {
      perrno(errno, "write_global() failed on fopen() for global.dat");
      return VERROR;
   } else {
      count = 0;
      count += fwrite(Cblocknum,    1,  8, fp);
      count += fwrite(Cblockhash,   1, 32, fp);
      count += fwrite(Prevhash,     1, 32, fp);
      count += fwrite(&Peerip,      1,  4, fp);
      count += fwrite(&Mfee,        1,  8, fp);
      count += fwrite(&Difficulty,  1,  4, fp);
      count += fwrite(&Time0,       1,  4, fp);
      count += fwrite(&Bgflag,      1,  1, fp);
      fclose(fp);
   }
   if(count != (8+32+32+4+8+4+4+1)) {
      perr("write_global() failed on fwrite() for %s: wrote %zu/%zu bytes",
         "global.dat", count, (size_t) (8+32+32+4+8+4+4+1));
      return VERROR;
   }
   return VEOK;
}  /* write_global() */

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
      perrno(errno, "append_tfile() failed on fopen() for %s", tfile);
      return VERROR;
   } else {
      count = fwrite(&bt, 1, sizeof(BTRAILER), fp);
      fclose(fp);
   }
   if(count != sizeof(BTRAILER)) {
      perr("append_tfile() failed on fwrite(): wrote %zu/%zu bytes to %s",
         count, sizeof(BTRAILER), tfile);
      return VERROR;
   }
   return VEOK;
}

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

/* Called from server.c to update globals */
int bupdata(void)
{
   BTRAILER bt;
   char bnumstr[24], haiku[256], *haiku1, *haiku2, *haiku3;
   unsigned btxs, btime, bdiff;
   word32 time1;
   int ecode;

   ecode = VEOK;

   if(add64(Cblocknum, One, Cblocknum)) {  /* increment block number */
      perr("new blocknum overflow");
      ecode = VERROR;
   }

   /* Update block hashes */
   memcpy(Prevhash, Cblockhash, HASHLEN);
   if(readtrailer(&bt, "ublock.dat") != VEOK) {
      perr("bupdata(): cannot read new ublock.dat hash");
      ecode = VERROR;
   }
   memcpy(Cblockhash, bt.bhash, HASHLEN);
   Difficulty = get32(bt.difficulty);
   Time0 = get32(bt.time0);
   time1 = get32(bt.stime);
   add_weight(Weight, bt.difficulty[0], bt.bnum);
   /* Update block difficulty */
   Difficulty = set_difficulty(&bt);
   pdebug("new: Difficulty = %d  seconds = %d", Difficulty, time1 - Time0);
   pdebug("Cblockhash: %s for block: 0x%s", hash2str(Cblockhash),
           bnum2hex(Cblocknum));
   Time0 = time1;
   /* display block update stats */
   btxs = (unsigned) get32(bt.tcount);
   btime = (unsigned) get32(bt.stime) - get32(bt.time0);
   bdiff = (unsigned) get32(bt.difficulty);
   if(!Bgflag && get32(bt.tcount)) {
      trigg_expand(bt.nonce, haiku);
      /* separate lines */
      haiku1 = strtok(haiku, "\n");
      haiku2 = strtok(&haiku1[strlen(haiku1) + 1], "\n");
      haiku3 = strtok(&haiku2[strlen(haiku2) + 1], "\n");
      /* rebuild one-line haiku */
      val2hex(bt.bnum, 8, bnumstr, sizeof(bnumstr));
      print("\nBlock #%" P32u " (0x%s):\n", get32(bt.bnum), bnumstr);
      print(" │ %s\n │ %s\n │ %s\n", haiku1, haiku2, haiku3);
      print(" └─ Time: %us / Diff: %u / Txs: %u\n", btime, bdiff, btxs);
   }
   return ecode;
}  /* end bupdata() */


/* Build a neo-genesis block -- called from server.c */
int do_neogen(void)
{
   char cmd[1024];
   word32 newnum[2];
   char *cp;
   int ecode;
   BTRAILER bt;

   unlink("neofail.lck");
   cp = bnum2hex(Cblocknum);
   sprintf(cmd, "../neogen %s/b%s.bc ngblock.dat", Bcdir, cp);
   pdebug("Creating neo-genesis block:\n '%s'", cmd);
   ecode = system(cmd);
   pdebug("do_neogen(): system():  ecode = %d", ecode);
   if(fexists("neofail.lck")) {
      perr("do_neogen failed");
      return VERROR;
   }
   add64(Cblocknum, One, newnum);
   cp = bnum2hex((word8 *) newnum);
   sprintf(cmd, "%s/b%s.bc", Bcdir, cp);
   if(rename("ngblock.dat", cmd) != 0) {
      perr("do_neogen failed to rename ngblock.dat to %s", cmd);
      return VERROR;
   }

   add64(Cblocknum, One, Cblocknum);
   /* Update block hashes */
   memcpy(Prevhash, Cblockhash, HASHLEN);
   if(readtrailer(&bt, cmd) != VEOK) {
      perr("do_neogen(): cannot read NG block hash");
      return VERROR;
   }
   memcpy(Cblockhash, bt.bhash, HASHLEN);
   Eon++;
   return VEOK;
}

/* end include guard */
#endif
