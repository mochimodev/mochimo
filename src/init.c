/* init.c  High-level Initialisation functions (included from mochimo.c)
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 11 February 2018
 *
*/

int hex2bnum(byte *bnum, char *hex)
{
   byte *bp;
   static char hextab[] = "0123456789abcdef";
   char *cp;

   for(bp = &bnum[7]; bp >= bnum; bp--, hex += 2) {
      cp = strchr(hextab, hex[0]);
      if(!cp) return VERROR;
      *bp = (cp - hextab) * 16;
      cp = strchr(hextab, hex[1]);
      if(!cp) return VERROR;
      *bp += (cp - hextab);
   }
   return VEOK;
}  /* end hex2bnum() */


char *find_last_block(char *bcdir)
{
   FILE *fp;
   char *cp;
   static char buff[256];

   sprintf(buff, "ls -1r %s/*.bc >bcdir.tmp", bcdir);
   system(buff);

   fp = fopen("bcdir.tmp", "rb");
   if(!fp) return NULL;
   if(fgets(buff, 255, fp) == NULL) {
      fclose(fp);
      unlink("bcdir.tmp");
      return NULL;
   }
   fclose(fp);
   unlink("bcdir.tmp");
   cp = strchr(buff, '\n');
   if(*cp) *cp = '\0';
   return buff;
}  /* end find_last_block() */


/* Find last block and reset
 * Time0, and Difficulty.
 * Set Cblocknum, Cblockhash, Prevhash, and Eon.
 * Return VEOK on success, else error code.
 */
int reset_difficulty(char *lastfname, char *bcdir)
{
   char *ext;
   byte lastbnum[8];
   BTRAILER bt;
   word32 time1;

   if(lastfname == NULL) {
      lastfname = find_last_block(bcdir);
      if(lastfname == NULL) return VERROR;
   }
   ext = strstr(lastfname, ".bc");
   if(!ext) return VERROR;
   if(strlen(lastfname) < 20) return VERROR;
   if(ext[-17] != 'b') return VERROR;
   if(hex2bnum(lastbnum, &ext[-16]) != VEOK) return VERROR;
   put64(Cblocknum, lastbnum);

   Eon = get32(&lastbnum[1]);

   if(readtrailer(&bt, lastfname) != VEOK) return VERROR;
   Time0 = get32(bt.time0);
   time1 = get32(bt.stime);
   Difficulty = set_difficulty(get32(bt.difficulty), time1 - Time0,
                               time1, bt.bnum);
   memcpy(Cblockhash, bt.bhash, HASHLEN);
   memcpy(Prevhash, bt.phash, HASHLEN);
   Time0 = time1;
   return VEOK;
}  /* end reset_difficulty() */


/* Read-in the ip list text file
 * each line:
 * 1.2.3.4  or
 * host.domain.name
 */
int read_ipl(char *fname, word32 *plist, uint32_t plist_len)
{
   FILE *fp;
   char buff[128];
   int j;
   char *addrstr;
   word32 ip;

   if(Trace) plog("Entering read_ipl()");
   if(fname == NULL || *fname == '\0') return VERROR;
   fp = fopen(fname, "rb");
   if(fp == NULL) return VERROR;

   for(j = 0; j < plist_len; ) {
      if(fgets(buff, 128, fp) == NULL) break;
      if(*buff == '#') continue;
      addrstr = strtok(buff, " \r\n\t");
      if(Trace > 1) plog("   parse: %s", addrstr);  /* debug */
      if(addrstr == NULL) break;
      ip = str2ip(addrstr);
      if(!ip) continue;
      /* put ip in plist[j] */
      plist[j++] = ip;
      if(Trace) plog("Added 0x%08x to plist", ip);  /* debug */
   }
   fclose(fp);
   return j;
}  /* end read_coreipl() */

/* Read in the core ip list text file */
int read_coreipl(char *fname) {
	return read_ipl(fname, Coreplist, CORELISTLEN);
}

/* Read in the local ip list text file */
int read_localipl(char *fname) {
	return read_ipl(fname, Lplist, LPLISTLEN);
}


/* Get an ip list from ip and copy it into np,
 * also call addrecent() on the list.
 * Return VEOK if successful, else error code.
*/
int get_ipl(NODE *np, word32 ip)
{
   int len;
   word32 *ipp;

   if(Trace)
      plog("get_ipl() about to call get_tx2()");
   if(get_tx2(np, ip, OP_GETIPL) == VEOK) {  /* closes socket */
      len = get16(np->tx.len);
      if((unsigned) len > TRANLEN) return VEBAD;
      for(ipp = (word32 *) TRANBUFF(&np->tx); len > 0;
             ipp++, len -= 4) {
                if(*ipp) {
                   if(Trace) plog("adding 0x%x from TX to recent", *ipp);
                   addrecent(*ipp);
                }
      }
      return VEOK;
   }
   return VERROR;
}  /* end get_ipl() */


/* On server INIT rplist.lst is on disk
 * otherwise check for coreip.lst and read into Coreplist.
 * Peer's cblock number is returned in np->cblock and
 * hash in np->cblockhash.
 */
word32 init_coreipl(NODE *np, char *fname)
{
   word32 *ipp, ip, return_ip;
   int j;
   word32 rplistidx, rplist[RPLISTLEN];

   show("coreipl");
   if(Trace) plog("Entering init_coreipl()");
   if(exists("rplist.lst")) {
      readlist32(rplist, 4, RPLISTLEN, "rplist.lst", &rplistidx);
      shuffle32(rplist, RPLISTLEN);  /* maybe embedded zeros */
      for(ipp = rplist, j = 0; j < RPLISTLEN; j++) {
         if(*ipp == 0) continue;
         addcurrent(*ipp++);
         addrecent(*ipp++);
      }
   }  /* end if rplist.lst */
   /* If fname exists, overlay internal Coreplist[] with it. */
   read_coreipl(fname);
   for(j = 0; j < CORELISTLEN && Coreplist[j] != 0; ) j++;
   shuffle32(Coreplist, j);
   for(j = 0, ipp = Coreplist; j < CORELISTLEN; j++, ipp++) {
      if(*ipp == 0) continue;
      if(Trace) plog("adding core 0x%x to recent and current", *ipp);
      addrecent(*ipp);
      addcurrent(*ipp);
   }

   /*
    * Get a recent peer list.
    */
   return_ip = 0;
   for(j = 0 ; j < RPLISTLEN; j++) {
      ip = Rplist[j];  /* select random peer */
      if(ip == 0) continue;
      if(Trace)
         plog("init_coreipl() about to call get_ipl(%s)", ntoa((byte *) &ip));
      if(get_ipl(np, ip) != VEOK) continue;
      return_ip = ip;
      break;
   }  /* end for */
   unlink("rplist.lst");
   if(Trace) plog("return_ip: %s", ntoa((byte *) &return_ip));
   return return_ip;
}  /* end init_coreipl() */





/* Accumulate weight based on difficulty */
void add_weight2(byte *weight, byte difficulty)
{
   byte temp[32];

   memset(temp, 0, 32);
   /* temp = 2**difficulty */
   temp[difficulty / 8] = 1 << (difficulty % 8);
   multi_add(weight, temp, weight, 32);
}  /* end add_weight2() */


/* Accumulate weight based on difficulty */
void add_weight(byte *weight, int difficulty, byte *bnum)
{
   static byte temp[32];
   static word32 trigger[2] = { WTRIGGER31, 0 };

   /* add difficulty to weight */
   if(cmp64(bnum, trigger) < 0) {
      temp[0] = difficulty;
      multi_add(weight, temp, weight, 32);
   }
   else add_weight2(weight, difficulty);

   if(Trace)
      plog("add_weight(): + %d --> %u", difficulty, get32(weight));
}  /* end add_weight() */


/* Little-endian compare two weights.
 * Returns = 0 if w1 == w2, negative if w1 < w2, positive if w1 > w2.
 */
int cmp_weight(byte *w1, byte *w2)
{
   int j, d;

   for(j = 31; j >= 0; j--) {
      d = w1[j] - w2[j];
      if(d) return d;
   }
   return 0;
}


void resign(char *mess)
{
   if(Trace && mess)
      plog("resigning in %s (sigterm)", mess);
   fatal2(0, NULL);
}


/* Validate a tfile
 * Returns: a pointer to static weight.
 *          *result is set to 0 on success with the block number
 *          of last good tfile record is left in highblock,
 *          otherwise *result is set to non-zero error code.
 *
 * Error codes 1-10 are validation errors; codes >= 100 are I/O,
 * codes >= 200 are (errno + 200).
 */
byte *tfval(char *fname, byte *highblock, int weight_only, int *result)
{
   FILE *fp;
   BTRAILER bt;
   word32 difficulty = 0;
   word32 time1 = 0;
   word32 stime;
   byte prevhash[HASHLEN];
   static byte weight[HASHLEN];   /* return value */
   long filelen;
   int ecode, gblock;
   char genfile[100];
   word32 now;
   word32 tcount;
   static word32 tottrigger[2] = { V23TRIGGER, 0 };
   static word32 v24trigger[2] = { V24TRIGGER, 0 };

   /* Adding constants to skip validation on BoxingDay corrupt block
    * provided the blockhash matches.  See "Boxing Day Anomaly" write
    * up on the Wiki or on [ REDACTED ] for more details. */
   static word32 boxingday[2] = { 0x52d3c, 0 };
   static char boxdayhash[32] = {
      0x2f, 0xfa, 0xb9, 0xb9, 0x00, 0xe1, 0xbc, 0xa8,
      0x25, 0x19, 0x20, 0xc2, 0xdd, 0xf0, 0x46, 0xb8,
      0x07, 0x44, 0x2a, 0xbb, 0xfa, 0x5e, 0x94, 0x51,
      0xb0, 0x60, 0x03, 0xcc, 0x82, 0x2d, 0xb1, 0x12
   };

   *result = 100;                 /* I/O high error code */
   memset(highblock, 0, 8);       /* start from genesis block */
   memset(weight, 0, HASHLEN);

   if(Trace) plog("Entering tfval()");
   show("tfval");

   sprintf(genfile, "%s/b0000000000000000.bc", Bcdir);
   /* get trailer from our Genesis Block */
   if(readtrailer(&bt, genfile) != VEOK) return weight;  /* error 100 */
   memcpy(prevhash, bt.bhash, HASHLEN);

   fp = fopen(fname, "rb");
   if(!fp) {
      error("tfval(): Cannot open %s", fname);
      *result = 101;
      return weight;
   }

   fseek(fp, 0, SEEK_END);
   filelen = ftell(fp);
   if((filelen % sizeof(BTRAILER)) != 0) {
      fclose(fp);
      *result = 102;
      return weight;
   }
   fseek(fp, 0, SEEK_SET);

   now = time(NULL);
   /* Validate every block trailer in tfile and compute weight. */
   for(gblock = 1; ; gblock = 0) {
      if(Monitor && Bgflag == 0) resign("tfile user break");  /* DSL */
      ecode = 0;
      if(fread(&bt, 1, sizeof(BTRAILER), fp)
            != sizeof(BTRAILER)) break;  /* EOF */

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
         if(stime > now && (stime - now) > BCONFREQ) break;
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
      if(highblock[0] && tcount && get32(bt.bnum) > Trustblock && Trustblock >= 0) {
         if(cmp64(bt.bnum, v24trigger) > 0) { /* v2.4 */
            if(cmp64(bt.bnum, boxingday) == 0) { /* Boxing Day Anomaly -- Bugfix */
               if(memcmp(bt.bhash, boxdayhash, 32)) {
                  if (Trace) plog("Boxing Day Bugfix Bhash Failure!");
                  break;
               }
            } else
            if(peach(&bt, get32(bt.difficulty), NULL, 1)){
            break;
            }
         }
         if(cmp64(bt.bnum, v24trigger) <= 0) { /* v2.3 and prior */
            if(trigg_check(bt.mroot, bt.difficulty[0], bt.bnum) == NULL) {
            break;
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
      if(Trace) plog("block: 0x%s difficulty: %d  seconds: %d",
           bnum2hex(bt.bnum), difficulty, time1 - get32(bt.time0));
      /*
       * Let the neo-genesis (not the 0xff) block change the
       * difficulty for the next 0x01 block.
       */
      if(highblock[0] != 0xff) {
         add_weight(weight, difficulty, bt.bnum);
         difficulty = set_difficulty(difficulty, time1 - get32(bt.time0),
                                     time1, bt.bnum);
         if(Trace) plog("new difficulty: %d", difficulty);
      }
next:
      /* set previous hash for next iteration */
      memcpy(prevhash, bt.bhash, HASHLEN);
      add64(highblock, One, highblock);  /* bnum in next trailer */
   }  /* end for */
   sub64(highblock, One, highblock);     /* fix high block number */
   fclose(fp);
   if(Trace) plog("tfval(): ecode = %d  bnum = 0x%s  weight = 0x...%x",
                  ecode, bnum2hex(highblock), weight[0]);
   *result = ecode;
   return weight;
}  /* end tfval() */


/* Delete all blocks above bc/matchblock.
 * Returns number of blocks deleted.
 */
int delete_blocks(byte *matchblock)
{
   char fname[128];
   int j;
   byte bnum[8];

   put64(bnum, matchblock);
   if(iszero(bnum,8)) add64(bnum, One, bnum);
   for(j = 0; ; j++) {
      sprintf(fname, "%s/b%s.bc", Bcdir, bnum2hex(bnum));
      if(unlink(fname) != 0) break;
      add64(bnum, One, bnum);
   }
   return j;
}


int trim_tfile(byte *highbnum)
{
   FILE *fp, *fpout;
   BTRAILER bt;
   byte bnum[8], flag;

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
   error("tfile(): flag = %d  bnum = 0x%s", flag, bnum2hex(bnum));
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
byte *get_treward(void *sum, void *bnum)
{
   word32 reward[2], bnum2[2];

   put64(bnum2, bnum);
   if(!iszero(bnum, 8)) {
      for(memset(sum, 0, 8); ;) {
         if(((byte *) bnum2)[0]) {
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
 * (reset_difficulty() has already been called to set Cblockhash.)
 */
int check_ng(char *fname, byte *bnum)
{
   static word32 premine[2]
      = { 0xbd1a6400, 0x0010e686 };  /* 4757066000000000 */
   static word32 tlen[2] = { sizeof(BTRAILER), 0 };
   byte sum[8], sum2[8], temp[8];
   LENTRY le;
   BTRAILER bt;
   long toffset;
   byte chash[HASHLEN];
   byte bhash[HASHLEN];
   byte buff[NGBUFFLEN];
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
   /* and the reset_difficulty() return value. */
   if(memcmp(chash, Cblockhash, HASHLEN) != 0) NGERROR(9);

   /* Compute total reward + premine into sum. */
   get_treward(sum, bnum);
   add64(premine, sum, sum);
   if(Trace > 1) {
      plog("premine: %lu  0x%lx\n", *((long *) premine), *((long *) premine));
      plog("sum:  %lu  0x%lx\n", *((long *) sum), *((long *) sum));
   }
   /* Check sum of amounts in NG ledger. */
   fseek(fp, 4, SEEK_SET);
   for(memset(sum2, 0, 8); ; ) {
      if(fread(&le, 1, sizeof(LENTRY), fp) != sizeof(LENTRY)) break;
      /* add64(sum2, le.balance, sum2);
      if(cmp64(sum2, sum) > 0) NGERROR(10); */
   }
   if(Trace > 1) {
      plog("sum2: %lu  0x%lx\n", *((long *) sum2), *((long *) sum2));
   }
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


/* Get blocks that we need up to network Cblocknum */
int get_eon(NODE *np, word32 peerip)
{
   FILE *fp, *tofp;           /* to "lock" and copy files */
   pid_t gpid[MAXQUORUM];     /* Gang children */
   word32 gang[MAXQUORUM];
   byte bnum[8], ngnum[8], highbnum[8];
   byte dlbnum[8], clbnum[8];  /* download/clear block number */
   byte highhash[HASHLEN], *tfweight;
   byte highweight[HASHLEN];
   int i, j, k, n, result;
   size_t cpbytes;            /* neo-gen transfer */
   char cpbuff[NGBUFFLEN];    /* neo-gen transfer */
   char fname[128], tofname[128];
   time_t timeout;
   static byte val256[8] = { 0x0, 0x1 };

   plog("Entering get_eon()");

   timeout = time(NULL) + 300;

top:
   memset(gpid, 0, sizeof(pid_t)*MAXQUORUM);
   memset(gang, 0, sizeof(gang));
   k = 0;
   tfweight = NULL;
   gang[0] = peerip;
   put64(highbnum, np->tx.cblock);
   memcpy(highhash, np->tx.cblockhash, HASHLEN);
   memcpy(highweight, np->tx.weight, HASHLEN);

   /* ****************
    * Fill gang[] with a list of ip's that all have
    * the highest block in the network such that
    * all the block numbers and block hashes are the same!
    */
   show("quorum");
   plog("Acquiring quorum");
   for(j = 1; j < Quorum && Running; ) {
      if(Monitor && Bgflag == 0) resign("user break 1");  /* DSL */
      if(time(NULL) >= timeout) goto try_again;
      /* select random peer from Rplist */
      for(peerip = n = 0; peerip == 0 && Running && n < 1000; n++)
         peerip = Rplist[rand16fast() % RPLISTLEN];
      /* no duplicate gang[] members */
      if(search32(peerip, gang, Quorum) != NULL) continue;
      /* fetch her ip list and compare block height/weight/hash */
      if(get_ipl(np, peerip) != VEOK) continue;
      result = cmp_weight(np->tx.weight, highweight);
      if(result > 0) goto top;
      if(result != 0) continue;
      if(memcmp(highhash, np->tx.cblockhash, HASHLEN) != 0) continue;
      gang[j++] = peerip;
   }
   if(!Running) resign("quorum debate");  /* System/360 Emergency Pull! */

   /* ****************
    * Obtain a tfile from a peer in the gang[], validate it
    * and compare its current bnum/weight against our [high]bnum/weight.
    */
   show("tfile");
   plog("Downloading tfile");
   unlink("tfile.dat");
   if(Trace) plog("   fetching tfile.dat from %s", ntoa((byte *) &gang[0]));
   for(k = 0; k < Quorum && Running; k++) {
      if(Monitor && Bgflag == 0) resign("user break 2");  /* DSL */
      peerip = gang[k];
      if(get_block2(peerip, NULL, "tfile.dat", OP_GET_TFILE) != VEOK)
         continue;  /* try to get block again from next peer in gang[] */
      tfweight = tfval("tfile.dat", bnum, 0, &result);
      if(result) goto try_again;  /* I/O error */
      if(cmp64(highbnum, bnum) > 0) goto try_again;
      if(cmp_weight(tfweight, highweight) < 0) goto try_again;
      break;  /* success */
   }
   if(!Running) resign("quorum tfile");
   if(k >= Quorum) goto try_again;
   if(Trace) plog("get_eon(): tfile.dat is valid.");

   /* ****************
    * Determine ngnum (starting neo-genesis block) calculated as
    * last neo-genesis block minus 1 aeon (256 blocks).
    */
   show("ngnum");
   put64(ngnum, highbnum);
   ngnum[0] = 0;
   if(sub64(ngnum, val256, ngnum)) memset(ngnum, 0, 8);
   if(Trace) plog("neo-genesis number: 0x%s", bnum2hex(ngnum));
   /* clean bc/ directory of block >= ngnum */
   delete_blocks(ngnum);
   /* trim the tfile back to the neo-genesis block and close the ledger */
   if(trim_tfile(ngnum) != VEOK) restart("trim_tfile()");  /* panic */
   le_close();
   /* copy neo-genesis backup to working directory, if available */
   sprintf(fname, "%s/b%s.bc", Ngdir, bnum2hex(ngnum));
   sprintf(tofname, "%s/b%s.bc", Bcdir, bnum2hex(ngnum));
   if(exists(fname)) {
      fp = fopen(fname, "rb");
      tofp = fopen("ngblock.dat", "wb");
      if(!fp || !tofp) {
         error("init: neo-genesis backup copy failed to open files!");
      } else {
         while (0 < (cpbytes = fread(cpbuff, 1, sizeof(cpbuff), fp))) {
            if(fwrite(cpbuff, 1, cpbytes, tofp) != cpbytes) {
               error("init: neo-genesis backup copy write failure!");
               unlink("ngblock.dat");
               break;
            }
         }
         /* transfer neo-genesis copy to bcdir */
         if(exists("ngblock.dat") && rename("ngblock.dat", tofname) != 0) {
            error("init: cannot rename backup copy neo-genesis to %s", tofname);
         }
      }
      if(fp) fclose(fp);
      if(tofp) fclose(tofp);
   }
   /* cleanup stray files */
   if(exists(tofname)) unlink(fname); /* ensures full resync on failure */
   else unlink("ngblock.dat");

   /* ****************
    * Get peer's neo-genesis block for new ledger.dat, set
    * the difficulty, open the ledger and validate the block.
    */
   show("neo-gen");
   peerip = gang[0];
   k = 1;
   put64(bnum, ngnum);
   /* use neo-genesis backup if available */
   sprintf(fname, "%s/b%s.bc", Bcdir, bnum2hex(bnum));
   if(exists(fname)) plog("Using neo-genesis backup 0x%s", bnum2hex(bnum));
   /* download neo-genesis block if no backup */
   else if(!iszero(bnum, 8)) {
      plog("Downloading neo-genesis 0x%s", bnum2hex(bnum));
      sprintf(fname, "%s/b%s.bc", Bcdir, bnum2hex(bnum));
      for( ; Running; ) {
         if(Monitor && Bgflag == 0) resign("user break 3");  /* DSL */
         if(get_block2(peerip, bnum, "ngblock.dat", OP_GETBLOCK) == VEOK) break;
         if(k >= Quorum) goto try_again;
         peerip = gang[k++];
      }
      /* transfer neo-genesis download to bcdir */
      if(exists("ngblock.dat") && rename("ngblock.dat", fname) != 0) {
         error("init: cannot rename downloaded neo-genesis to %s", fname);
         unlink("ngblock.dat");
         exit(VERROR);
      }
   }  /* end if neo-genesis download */
   if(!iszero(bnum, 8)) {
      if(extract(fname, "ledger.dat") != VEOK) {
         unlink(fname);  /* incase the backup was the problem */
         restart("qu extract");
      }
   } else extract_gen("ledger.dat");  /* use genesis block ledger */
   /* setup the difficulty, based on [neo]genesis block */
   if(reset_difficulty(NULL, Bcdir) != VEOK) {
      unlink(fname);  /* incase the backup was the problem */
      restart("qu diff");
   }
   le_open("ledger.dat", "rb");
   /* Cblockhash was set from NG by reset_difficulty()
    * Check a non-Genesis NG hash against Cblockhash.
    */
   if(!iszero(bnum, 8)) {
      result = check_ng(fname, bnum);
      if(result != 0) {
         plog("get_eon(): Bad NG block! ecode: %d", result);
         unlink(fname);
         goto try_again;
      }
   }

   /* ****************
    * Download the blockchain asynchronously using
    * all accessible gang[] members.
    */
   show("dlblocks");
   printf("Downloading blockchain...\n");
   put64(clbnum, bnum);
   add64(bnum, One, bnum);
   for( ; Running; ) {
      put64(dlbnum, bnum);
      /* i -> count children finished downloading
       * j -> count peers discarded (get_block2() unsuccessful)
       * k -> gang index
       */
      for(i=0, j=0, k=0, result=0; Running && k<Quorum; result=0) {
         if(Monitor && Bgflag == 0) resign("user break 4");  /* DSL */
         /* poll children */
         if(gpid[k] > 0 && waitpid(gpid[k], &result, WNOHANG) > 0) {
            if(result) gang[k] = 0;
            gpid[k] = 0;
            i++;
         }
         /* check peer ok */
         peerip = gang[k];
         if(peerip == 0) { j++; k++; continue; }
         /* set block number to download */
         sprintf(fname, "dblock%02X%02X.dat", dlbnum[1], dlbnum[0]);
         sprintf(tofname, "rblock%02X%02X.dat", dlbnum[1], dlbnum[0]);
         /* cleanup downloaded blocks from past sync attempts */
         if(dlbnum[0] != 0 && cmp64(dlbnum, clbnum) > 0) {
            put64(clbnum, dlbnum);
            unlink(fname);
            unlink(tofname);
         }
         /* check block duplicate */
         if(dlbnum[0] == 0 || exists(tofname) || exists(fname)) {
            add64(dlbnum, One, dlbnum);
            continue;
         }
         /* spawn child to download block */
         if(!gpid[k]) {
            gpid[k] = fork();
            if(gpid[k] > 0) add64(dlbnum, One, dlbnum); /* parent */
            if(gpid[k] < 0) gang[k] = 0; /* parent */
            if(gpid[k] == 0) { /* child */
               fp = fopen(fname, "ab+"); /* get first dibs */
               if(!fp) {
                  error("init: cannot secure block file %s", fname);
                  exit(VERROR);
               } else fclose(fp);
               result = get_block2(peerip, dlbnum, fname, OP_GETBLOCK);
               if(result != VEOK) unlink(fname);
               else if(rename(fname, tofname) != 0) {
                  error("init: cannot rename %s",fname);
                  unlink(fname);
                  exit(VERROR);
               }
               exit(result);
            }
         }
         k++;
      }  /* end for thread handling */
      /* sleep cpu during no activity */
      if(!i && Dynasleep) usleep(Dynasleep);
      /* update downloaded blocks (also constructs NG on 0xff blocks) */
      sprintf(fname, "rblock%02X%02X.dat", bnum[1], bnum[0]);
      while (exists(fname)) {
         if(update(fname, 0) != VEOK) goto try_again;
         add64(bnum, One, bnum);
         if(bnum[0] == 0) add64(bnum, One, bnum);
         sprintf(fname, "rblock%02X%02X.dat", bnum[1], bnum[0]);
      }
      /* unable to download more blockchain */
      if(j>=Quorum) {
         if(cmp64(bnum, highbnum) >= 0) break;
         goto try_again;
      }
   }  /* end for block download-update */

   /* Post-sync hook for external SQL database export */
   /* Shell script in /bin directory */
   if(Exportflag && exists("../init-external.sh")) {
     plog("Calling ../init-external.sh\n");  /* first time call */
     system("../init-external.sh");
   }

   if(!Running) resign("quorum update");

   /* ****************
    * Re-compute Weight[].
    * Check tfval() set bnum to high block number on chain
    * Done!
    */
   tfweight = tfval("tfile.dat", bnum, 1, &result);
   if(result) goto try_again;  /* should not happen */
   if(cmp64(bnum, Cblocknum) != 0) {
      error("Block number mismatch, try_again");  /* should not happen */
      goto try_again;
   }
   memcpy(Weight, tfweight, HASHLEN);

   if(Trace) plog("re-computed Weight = 0x...%x", Weight[0]);
   plog("Veronica says, 'You're done!'");

   return VEOK;

try_again:
   plog(":) (k: %d  Will restart in %d seconds.)", k,
        (int) (timeout - time(NULL)));
   if(Trace) {
      if(tfweight)
         plog("tfweight = 0x%s...", addr2str(tfweight));
      plog("highweight = 0x%s...", addr2str(highweight));
      plog("bnum = 0x%s", bnum2hex(bnum));
      plog("Cblocknum = 0x%s", bnum2hex(Cblocknum));
      plog("highbnum = 0x%s", bnum2hex(highbnum));
   }
   if(time(NULL) >= timeout) restart(":) timeout");  /* v.28 */
   if(Monitor && Bgflag == 0) resign("user break 5");  /* DSL */

   for(peerip = n = 0; peerip == 0 && Running && n < 1000; n++)
      peerip = Rplist[rand16fast() % RPLISTLEN];
   if(get_ipl(np, peerip) != VEOK && Running) goto try_again;
   if(Running) goto top;  /* v.28 */
   resign("try again");
   return VERROR;  /* never gets here */
}  /* end get_eon() */


/* Bring the server/client up from a cold start
 * after gomochi script runs
 */
int init(void)
{
   NODE node;  /* holds peer tx.cblock and tx.cblockhash */
   int result;
   byte diff[8], highblock[8], *wp;
   word32 solved;

   Running = 1;
   Ininit = 1;

   plog("Entering init()");
   show("init");

   /* open ledger read-only */
   if(!exists("ledger.dat") || le_open("ledger.dat", "rb") != VEOK) {
      /* extract the ledger from our Genesis Block */
      if(extract_gen("ledger.dat") != VEOK)
         fatal("init(): no ledger.dat");
      le_open("ledger.dat", "rb");  /* try again */
   }

   /* read epoch pink list */
   readpink();
   if(read_data(&solved, 4, "solved.dat") == 4) Nsolved = solved;

   /* Find the last block in bc/ and reset Time0,
    * and Difficulty.
    */
   if(reset_difficulty(NULL, Bcdir) != VEOK)
      error("init(): reset_difficulty()");

   /* Read and validate our own tfile.dat to compute Weight */
   wp = tfval("tfile.dat", highblock, 0, &result);
   if(Trustblock != -1){
      if(result || cmp64(Cblocknum, highblock) != 0) {
         plog("init(): %d %d", Cblocknum[0], highblock[0]);
         fatal("init(): bad tfile.dat -- gomochi!");
      }
   }   
   memcpy(Weight, wp, HASHLEN);

   /* read local nodes into Lplist */
   read_localipl(Lpfname);


   /* read into Coreplist[], shuffle, and get IPL */
   if(*Corefname)
      plog("   %s is the bootstrap nodes file list.", Corefname);
   Peerip = init_coreipl(&node, Corefname);
   if(Peerip) {
      result = cmp64(Cblocknum, node.tx.cblock);
      if(result > 0) goto done;  /* We have a higher block */
      if(result == 0
         && memcmp(Cblockhash, node.tx.pblockhash, HASHLEN) == 0)
              goto done;  /* already in synch */
      sub64(node.tx.cblock, Cblocknum, diff);
      /* then begin to fetch blocks... */
      result = cmp64(diff, One);
      if(result == 0) {
         /* If only 1 block behind, just get this one... */
         if(get_block2(Peerip, node.tx.cblock, "rblock.dat",
                           OP_GETBLOCK) == VEOK)
            update("rblock.dat", 0);
      }
      if(result > 0)
         get_eon(&node, Peerip);  /* ...otherwise get the eon. */
   }  /* end if Peerip */
done:
   write_global();
   Ininit = 0;
   return VEOK;
}  /* end init() */
