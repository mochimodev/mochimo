/**
 * @private
 * @headerfile util.h <util.h>
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_UTIL_C
#define MOCHIMO_UTIL_C


#include "util.h"

/* internal support */
#include "trigg.h"
#include "peach.h"
#include "network.h"
#include "global.h"

/* external support */
#include <time.h>
#include <string.h>
#include <signal.h>
#include <stdlib.h>
#include "extprint.h"
#include "extmath.h"
#include "extlib.h"
#include "extio.h"
#include "extinet.h"
#include <errno.h>
#include "crc16.h"

#ifdef OS_UNIX
   #include <dirent.h>

#endif

#ifndef NSIG
   #define NSIG 23

#endif

char *show(char *state)
{
   if(state == NULL) state = "(null)";
   if(Statusarg) strncpy(Statusarg, state, 8);
   return state;
}

/**
 * Print local host info on stdout.
 * @returns 0 on succesful operation, or (-1) on error.
*/
void phostinfo(void)
{
   char hostname[64] = "";
   char addrname[64] = "";

   /* get local machine name and IP address */
   gethostname(hostname, sizeof(hostname));
   gethostip(addrname, sizeof(addrname));
   print("Local Machine Info\n");
   print("  Machine name: %s\n", *hostname ? hostname : "unknown");
   print("  IPv4 address: %s\n", *addrname ? addrname : "0.0.0.0");
   print("\n");
}  /* end phostinfo() */

/**
 * Check argument list for options. @a chk1 and/or @a chk2 can be NULL.
 * Compatible with appended values using " " or ":" or "=".<br/>
 * e.g. `--arg <value>1 or `--arg:<value>` or `--arg=<value>`
 * @param argv Pointer to argument list item to check
 * @param chk1 First option to check against @a argv
 * @param chk2 Second option to check against @a argv
 * @returns 1 if either options match argument, else 0 for no match.
*/
int argument(char *argv, char *chk1, char *chk2)
{
   int result = 0;
   char tmp, *vp;

   /* remove value identifier, temporarily */
   vp = strpbrk(argv, ":=");
   if (vp) {
      tmp = *vp;
      *vp = '\0';
   }
   /* check argv for match */
   if (argv != NULL && *argv) {
      if (chk1 != NULL && strcmp(argv, chk1) == 0) result = 1;
      else if (chk2 != NULL && strcmp(argv, chk2) == 0) result = 1;
   }
   /* replace value identifier */
   if (vp) *vp = tmp;

   return result;
}  /* end argument() */


/**
 * Obtain the value associated with the current argument index.
 * Compatible with appended values using " " or ":" or "=".<br/>
 * e.g. `--arg <value>1 or `--arg:<value>` or `--arg=<value>`
 * @param idx Pointer to current argument index (i.e. argv[*idx])
 * @param argc Number of total arguments
 * @param argv Pointer to argument list
 * @returns Char pointer to argument value, else NULL for no value.
*/
 char *argvalue(int *idx, int argc, char *argv[])
{
   char *vp = NULL;

   /* check index */
   if (*idx >= argc) return NULL;
   /* remove value identifier, temporarily */
   vp = strpbrk(argv[*idx], ":=");
   if (vp) vp++;
   else if (++(*idx) < argc && argv[*idx][0] != '-') {
      vp = argv[*idx];
   } else --(*idx);

   return vp;
}  /* end argvalue() */

char *metric_reduce(double *value)
{
   static char M[8][3] = { "", "K", "M", "G", "T", "P", "E", "Z" };
   static int MLEN = sizeof(M) / sizeof (*M);
   int m;

   m = (*value >= 1.0) ? (int) (log10(*value) / 3) : 0;
   if (m >= MLEN) m = MLEN - 1;
   *value /= pow(1000.0, (double) m);

   return M[m];
}

/* kill the block constructor */
int stop_bcon(void)
{
   int status = VETIMEOUT;

   if (Bcon_pid) {
      pdebug("   Waiting for b_con() to exit");
      kill(Bcon_pid, SIGTERM);
      waitpid(Bcon_pid, NULL, 0);
      Bcon_pid = 0;
   }

   return status;
}

/* kill send_found() */
int stop_found(void)
{
   int status = VETIMEOUT;

   if (Found_pid) {
      pdebug("   Waiting for send_found() to exit");
      kill(Found_pid, SIGTERM);
      waitpid(Found_pid, &status, 0);
      Found_pid = 0;
   }

   return status;
}

/* kill the miner child */
int stop_miner(void)
{
   int status = VETIMEOUT;

   if (Mpid) {
      pdebug("   Waiting for miner to exit");
      kill(Mpid, SIGTERM);
      waitpid(Mpid, &status, 0);
      /* remove miner files */
      remove("miner.tmp");
      remove("bctx.dat");
      Mpid = 0;
   }

   return status;
}

/* kill mirror() children and grandchildren */
void stop_mirror(void)
{
   if(Mqpid) {
      pdebug("   Reaping mirror() zombies...");
      kill(Mqpid, SIGTERM);
      waitpid(Mqpid, NULL, 0);
      Mqpid = 0;
   }
}  /* end stop_mirror() */

/**
 * Kill critical services for clean block updates.
*/
void stop4update(void)
{
   NODE *np;
   word16 opcode;

   /* kill and wait for critical services to exit */
   stop_bcon();
   stop_found();
   stop_miner();

   /* Reap cblock and mblock-push children... */
   if (!Ininit && Allowpush) {
      /* Don't fear the Reaper, baby...It won't hurt... */
      for(np = Nodes; np < Hi_node; np++) {
         if(np->pid == 0) continue;
         opcode = get16(np->tx.opcode);
         if(opcode == OP_GET_CBLOCK || opcode == OP_MBLOCK) {
            kill(np->pid, SIGTERM);
            waitpid(np->pid, NULL, 0);
            freeslot(np);
         }
      }  /* end for(np = Nodes... */
   }  /* end if (!Ininit && Allowpush... */
}  /* end stop4update() */

/* Display terminal error message
 * and exit with exitcode after reaping zombies.
 */
void fatal2(int exitcode, char *message)
{
   pfatal("%s", message);
   /* stop all services */
   stop_bcon();
   stop_found();
   stop_miner();
   stop_mirror();
   /* wait for all children */
   while(waitpid(-1, NULL, 0) != -1);
   exit(exitcode);
}

void resign(char *mess)
{
   if(mess) pdebug("resigning in %s (sigterm)", mess);
   fatal2(0, NULL);
}

void restart(char *mess)
{
   pdebug("restart: %s", mess);
   remove("epink.lst");
   fatal2(1, NULL);
}

double diffclocktime(clock_t to, clock_t from)
{
   return (double) (to - from) / CLOCKS_PER_SEC;
}

int check_directory(char *dirname)
{
   char fname[FILENAME_MAX];

   mkdir_p(dirname);
   snprintf(fname, FILENAME_MAX, "%s/chkfile", dirname);
   if (ftouch(fname) == VEOK) return remove(fname);
   return perrno(errno, "Permission failure, %s", dirname);
}

int clear_directory(char *dname)
{
   DIR *dp;
   struct dirent *ep;
   char fname[FILENAME_MAX];

   dp = opendir(dname);
   if (dp == NULL) return perrno(errno, "failed to open dir %s...", dname);
   while ((ep = readdir(dp))) {
      snprintf(fname, FILENAME_MAX, "%s/%s", dname, ep->d_name);
      remove(fname); /* ignores non-empty directories */
   }
   closedir(dp);

   /* success */
   return VEOK;
}

void crctx(TX *tx)
{
   put16(tx->crc16, crc16((word8 *) tx, sizeof(TX) - (2+2)));
}

/**
 * Get the fixed length header value (hdrlen) of a blockchain file.
 * @param fname File name of a blockchain file
 * @returns @a hdrlen value of the blockchain file, else 0
*/
word32 gethdrlen(char *fname)
{
   FILE *fp;
   word32 len;

   fp = fopen(fname, "rb");
   if(fp == NULL) return 0;
   if(fread(&len, 1, 4, fp) !=  4) { fclose(fp); return 0; }
   fclose(fp);
   return len;
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

/**
 * Convert block number and block hash to a BlockID string.
 * @param bnum Pointer to 8 byte block number
 * @param bhash Pointer to 32 byte block hash
 * @param buf Character buffer to place resulting BlockID
 * @param bufsz Byte length of @a buf
 * @returns @a buf containing BlockID string or, where @a buf is NULL,
 * an internal character buffer containing the BlockID str.
*/
char *block2str(void *bnum, void *bhash, char *buf, size_t bufsz)
{
   static const char fmt[] = "0x%" P32x "%08" P32x " #%02x%02x%02x%02x";
   static const char fmt_shrt[] = "0x%" P32x " #%02x%02x%02x%02x";
   static char str[32];
   word32 *dp;
   word8 *bp;

   bp = (word8 *) bhash;
   dp = (word32 *) bnum;
   if (buf == NULL) {
      bufsz = 32;
      buf = str;
   }
   if (dp[1]) {
      /* print bnum as 2x 32-bit words if bnum value > 32-bit */
      snprintf(buf, bufsz, fmt, dp[1], dp[0], bp[0], bp[1], bp[2], bp[3]);
   } else snprintf(buf, bufsz, fmt_shrt, dp[0], bp[0], bp[1], bp[2], bp[3]);

   return buf;
}  /* end blockid2str() */

/**
 * Get string from terminal input without newline char.
 * @param buff Pointer to char array to place input
 * @param len Maximum length of char array @a buff
 * @returns Pointer to @a buff
*/
char *tgets(char *buff, int len)
{
   char *cp;

   if (fgets(buff, len, stdin) == NULL) *buff = '\0';
   cp = strchr(buff, '\n');
   if (cp) *cp = '\0';

   return buff;
}


int accept_block(char *ublock, word8 *newnum)
{
   char buff[256];
   char cmd[288];
   char *bnum;

   bnum = bnum2hex(newnum);
   sprintf(buff, "b%s.bc", bnum);
   sprintf(cmd, "%s/b%s.bc", Bcdir, bnum);
   if(fexists(buff) || fexists(cmd)) {
      perr("accept_block(): failed: %s already exists!", buff);
      return VERROR;
   }
   if(rename(ublock, buff) != 0) {
      perrno(errno, "accept_block(): failed on rename() %s to %s", ublock, buff);
      return VERROR;
   }
   sprintf(cmd, "mv %s %s", buff, Bcdir);
   if (system(cmd)) return VERROR;
   sprintf(buff, "%s/b%s.bc", Bcdir, bnum);
   if(!fexists(buff)) {
      perr("accept_block(): failed on system(%s): %s missing", cmd, buff);
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
      count += fread(&Mfee,        1,  8, fp);
      count += fread(&Difficulty,  1,  4, fp);
      count += fread(&Time0,       1,  4, fp);
      count += fread(&Bgflag,      1,  1, fp);
      fclose(fp);
   }
   if(count != (8+32+32+4+8+4+1)) {
      perr("read_global() failed on fread() for %s: read %zu/%zu bytes",
         "global.dat", count, (size_t) (8+32+32+4+8+4+1));
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
      count += fwrite(&Mfee,        1,  8, fp);
      count += fwrite(&Difficulty,  1,  4, fp);
      count += fwrite(&Time0,       1,  4, fp);
      count += fwrite(&Bgflag,      1,  1, fp);
      fclose(fp);
   }
   if(count != (8+32+32+4+8+4+1)) {
      perr("write_global() failed on fwrite() for %s: wrote %zu/%zu bytes",
         "global.dat", count, (size_t) (8+32+32+4+8+4+1));
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

#ifdef OS_UNIX

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

   /**
    * Segmentation fault handler.
    * @note compile with "-g -rdynamic" for readable backtrace
   */
   void segfault(int sig) {
      void *array[10];
      size_t size;

      /* get void*'s for all entries on the stack */
      size = backtrace(array, 10);

      /* print out all the frames to stderr */
      fprintf(stderr, "Error: signal %d:\n", sig);
      backtrace_symbols_fd(array, size, STDERR_FILENO);
      exit(1);
   }

#endif

/*
 * Signal handlers
 *
 * Enter monitor on ctrl-C
 */
void ctrlc(int sig)
{
   print("\n");
   pdebug("Got signal %i\n", sig);
   signal(SIGINT, ctrlc);
   if (Ininit) Running = 0;
   else Monitor = 1;
}


/*
 * Clear run flag, Running on SIGTERM
 */
void sigterm(int sig)
{
   pdebug("Got signal %i\n", sig);
   signal(SIGTERM, sigterm);
   Running = 0;
}


void fix_signals(void)
{
   int j;

   /*
    * Ignore all signals.
    */
   for(j = 0; j <= NSIG; j++)
      signal(j, SIG_IGN);

   signal(SIGINT, ctrlc);     /* then install ctrl-C handler */
   signal(SIGTERM, sigterm);  /* ...and software termination */
#ifdef OS_UNIX
   signal(SIGSEGV, segfault);   /* segmentation fault handler*/
#endif
}


void close_extra(void)
{
   int j;

   for(j = 3; j < 50; j++) sock_close(j);
}

void print_bup(BTRAILER *bt, char *solvestr)
{
   word32 bnum, btxs, btime, bdiff;
   char haiku[256], *haiku1, *haiku2, *haiku3;

   /* prepare block stats */
   bnum = get32(bt->bnum);
   btxs = get32(bt->tcount);
   btime = get32(bt->stime) - get32(bt->time0);
   bdiff = get32(bt->difficulty);
   /* print haiku if non-pseudo block */
   if (!Insyncup && btxs) {
      /* expand and split haiku into lines for printing */
      trigg_expand(bt->nonce, haiku);
      haiku1 = strtok(haiku, "\n");
      haiku2 = strtok(&haiku1[strlen(haiku1) + 1], "\n");
      haiku3 = strtok(&haiku2[strlen(haiku2) + 1], "\n");
      print("\n/)  %s\n(=:  %s\n\\)    %s\n", haiku1, haiku2, haiku3);
      /* print block update and details */
      plog("Time: %" P32u "s, Diff: %" P32u ", Txs: %" P32u,
         btime, bdiff, btxs);
   }
   /* print block identification */
   plog("%s-block: 0x%" P32x " #%s...",
      solvestr, bnum, addr2str(bt->bhash));
   /* print miner data if enabled */
   if (!Ininit && !Insyncup && !Nominer) {
      read_data(&Hps, sizeof(Hps), "hps.dat");
      print("Solved: %" P32u "  Hps: %" P32u "\n", Nsolved, Hps);
   }
}

void print_splash(char *execname, char *version)
{
   plog("%s %s, " __DATE__ " " __TIME__, execname, version);
   plog("Copyright (c) 2022 Adequate Systems, LLC. All Rights Reserved.");
   plog("See the License Agreement at the links below:");
   plog("   https://mochimo.org/license.pdf (PDF version)");
   plog("   https://mochimo.org/license (TEXT version)");
   print("\n");
}

/* end include guard */
#endif
