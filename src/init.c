/* init.c  High-level Initialisation functions (included from mochimo.c)
 *
 * Copyright (c) 2018 by Adequate Systems, LLC.  All Rights Reserved.
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
   char fname[128];
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

   /* make file name from neobnum into fname */
   sprintf(fname, "%s/b%s.bc", Bcdir, bnum2hex(lastbnum));
   if(readtrailer(&bt, lastfname) != VEOK) return VERROR;
   Time0 = get32(bt.time0);
   time1 = get32(bt.stime);
   Difficulty = set_difficulty(get32(bt.difficulty), time1 - Time0,
                               time1, bt.bnum);
   memcpy(Cblockhash, bt.bhash, HASHLEN);
   memcpy(Prevhash, bt.phash, HASHLEN);
   return VEOK;
}  /* end reset_difficulty() */


/* Read-in the core ip list text file
 * each line:
 * 1.2.3.4  or
 * host.domain.name
 */
int read_coreipl(char *fname)
{
   FILE *fp;
   char buff[128];
   int j;
   char *addrstr;
   word32 ip, *ipp;

   if(Trace) plog("Entering read_coreipl()");
   if(fname == NULL || *fname == '\0') return VERROR;
   fp = fopen(fname, "rb");
   if(fp == NULL) return VERROR;
   for(j = 0; j < CORELISTLEN; ) {
      if(fgets(buff, 128, fp) == NULL) break;
      if(*buff == '#') continue;
      addrstr = strtok(buff, " \r\n\t");
      if(Trace > 1) plog("   parse: %s", addrstr);  /* debug */
      if(addrstr == NULL) break;
      ip = str2ip(addrstr);
      if(!ip) continue;
      /* put ip in Coreplist[j] */
      Coreplist[j++] = ip;
      if(Trace) plog("Added 0x%08x to Coreplist", ip);  /* debug */
   }
   fclose(fp);
   return j;
}  /* end read_coreipl() */


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
   int len;
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


/* Extract the ledger from a neo-genesis block and
 * put it in ledger file lfile (ledger.dat)
 * Return VEOK on success, else VERROR.
 */
int extract(char *fname, char *lfile)
{
   word32 hdrlen;    /* to read-in block header length */
   FILE *fp, *lfp;
   LENTRY le;        /* buffer to read ledger entry */
   byte prevaddr[TXADDRLEN];  /* to check block addr sort */
   byte first;

   if(Trace) plog("extract() ledger from %s to %s", fname, lfile);

   /* open the neo-genesis block and read in file header length */
   fp = fopen(fname, "rb");
   if(!fp) return VERROR;;
   if(fread(&hdrlen, 1, 4, fp) != 4) goto ioerror;

   lfp = fopen(lfile, "wb");
   if(!lfp) {
      error("extract(): Cannot open %s", lfile);
      goto ioerror;
   }

   /* Make sure that NG header contains at least
    * one ledger entry.
    */
   if(hdrlen < (sizeof(LENTRY) + 4)) {
      error("extract(): Not a neo-genesis block: %s", fname);
      goto error2;
   }

   /*
    * Read the ledger from fp and copy it to lfp,
    * creating a new ledger.dat file.
    * NOTE: block trailer must be less than sizeof(LENTRY)
    */
   if(fseek(fp, 4, SEEK_SET)) goto error2;
   for(hdrlen -= 4, first = 1; ; first = 0) {
      if(fread(&le, 1, sizeof(LENTRY), fp) != sizeof(LENTRY))
         break;
      hdrlen -= sizeof(LENTRY);
      /* check ledger sort in NG block */
      if(!first && memcmp(le.addr, prevaddr, TXADDRLEN) <= 0) {
         error("extract(): bad ledger sort in neo-genesis block");
         goto error2;
      }
      memcpy(prevaddr, le.addr, TXADDRLEN);
      if(fwrite(&le, 1, sizeof(LENTRY), lfp) != sizeof(LENTRY))
         goto error2;
   }
   if(hdrlen) {
      error("extract(): bad neo-genesis block length");
      goto error2;
   }
   fclose(fp);
   fclose(lfp);
   return VEOK;
ioerror:
      fclose(fp);
      unlink(lfile);  /* remove bad ledger */
      return error("extract() failed!");
error2:
   fclose(lfp);
   goto ioerror;
}  /* end extract() */


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
      plog("add_weight(): + %d --> %d", difficulty, get32(weight));
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


/* Validate a tfile
 * Returns: a pointer to static weight.
 *          *result is set to 0 on success with the block number
 *          of last good tfile record is left in highblock,
 *          otherwise *result is set to non-zero error code.
 *
 * Error codes 1-9 are validation errors; codes >= 100 are I/O, 
 * codes >= 200 are (errno + 200).
 */
byte *tfval(char *fname, byte *highblock, int weight_only, int *result)
{
   FILE *fp;
   BTRAILER bt;
   word32 difficulty = 0;
   word32 time1 = 0;
   word32 stemp;
   byte prevhash[HASHLEN];
   static byte weight[HASHLEN];   /* return value */
   long filelen;
   int ecode, gblock;
   char genfile[100];
   word32 now;

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
      ecode = 0;
      if(fread(&bt, 1, sizeof(BTRAILER), fp)
            != sizeof(BTRAILER)) break;  ecode++;

      /* The Genesis Block is very special. 1 */
      if(gblock) {
         if(!iszero(&bt, (sizeof(BTRAILER) - HASHLEN))) break;  ecode++;
         if(memcmp(prevhash, bt.bhash, HASHLEN) != 0) break;  /* 2 */
         difficulty = 1;  /* difficulty of block one. */
         goto next;
      }
      if(weight_only) goto skipval;

      ecode = 3;
      /* validate block trailer -- Mfee: 3 */
      if(highblock[0]) {
         if(memcmp(Mfee, bt.mfee, 8) != 0) break;
      } else if(!iszero(bt.mfee, 8)) break;  /* for NG block */

      ecode++;  /* difficulty ecode = 4 */
      if(get32(bt.difficulty) != difficulty) break;  ecode++;

      /* check for early or future block time 5 */
      stemp = get32(bt.stime);
      if(highblock[0]) {
         if(stemp <= time1 || stemp > now)  /* unsigned time here */
            break;
      }
      else if(stemp != time1) break;  /* for NG block */
      ecode++;
      /* bad block number 6 */
      if(cmp64(highblock, bt.bnum) != 0) break;     ecode++;
      /* bad previous hash 7 */
      if(memcmp(prevhash, bt.phash, HASHLEN) != 0) break;  ecode++;
      /* check enforced delay 8 */
      if(highblock[0]) {
         if(trigg_check(bt.mroot, bt.difficulty[0], bt.bnum) == NULL)
            break;
         ecode++;
         /* empty block 9 */
         if(get32(bt.tcount) == 0) break;
      }
      ecode = 10;

skipval:
      /* update for next loop 10 */
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
   for(j = 0; ; j++) {
      add64(bnum, One, bnum);
      sprintf(fname, "%s/b%s.bc", Bcdir, bnum2hex(bnum));
      if(unlink(fname) != 0) break;
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
   return extract(fname, lfile);
}


void resign(char *mess)
{
   if(Trace && mess)
      plog("resigning in %s (sigterm)", mess);
   fatal2(0, NULL);
}


/* Get blocks that we need up to network Cblocknum */
int get_eon(NODE *np, word32 peerip)
{
   FILE *fp;                 /* to "lock" files */
   pid_t Gpid[MAXQUORUM];    /* Gang children */
   word32 gang[MAXQUORUM];
   byte bnum[8], ngnum[8], highbnum[8], dlbnum[8];
   byte highhash[HASHLEN], *tfweight;
   byte highweight[HASHLEN];
   int i, j, k, result, result2;
   char fname[128];
   char flock[128];    /* tame the multi-threaded madness */
   time_t timeout;
   BTRAILER bt;
   static byte val256[8] = { 0x0, 0x1 };

   plog("Entering get_eon()");

   timeout = time(NULL) + 180;

top:
   show("quorum");
   memset(gang, 0, sizeof(gang));
   k = 0;
   tfweight = NULL;
   gang[0] = peerip;
   put64(highbnum, np->tx.cblock);
   memcpy(highhash, np->tx.cblockhash, HASHLEN);
   memcpy(highweight, np->tx.weight, HASHLEN);

   /* Fill gang[] with a list of ip's that all have 
    * the highest block in the network such that
    * all the block numbers and block hashes are the same!
    */
   for(j = 1; j < Quorum && Running; ) {
      if(time(NULL) >= timeout) goto try_again;
      /* find new peerip not in gang[]
       * and fetch her ip list into *np
       */
      for(peerip = 0; peerip == 0 && Running; )
         peerip = Rplist[rand16() % RPLISTLEN];
      if(get_ipl(np, peerip) != VEOK) continue;
      /* check if peer's block number is higher */
      result = cmp64(np->tx.cblock, highbnum);
      result2 = cmp_weight(np->tx.weight, highweight);
      if(result > 0 && result2 > 0) goto top;
      if(result != 0 || result2 != 0) continue;
      if(search32(peerip, gang, Quorum) != NULL) continue;
      if(memcmp(highhash, np->tx.cblockhash, HASHLEN) != 0) continue;
      gang[j++] = peerip;
   }
   if(!Running) resign("quorum debate");  /* System/360 Emergency Pull! */

   if(Trace) plog("   fetching tfile.dat from %s", ntoa((byte *) &gang[0]));
   show("tfile");

   unlink("tfile.dat");

   /* get the peer's tfile and validate it */
   for(k = 0; k < Quorum && Running; k++) {
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

   /* ngnum is peer's last neo-genesis block
    * so delete all bc/ blocks greater.
    */   
   put64(ngnum, highbnum);
   ngnum[0] = 0;
   result = sub64(ngnum, val256, ngnum);  /* go back another 256 */
   if(result) memset(ngnum, 0, 8);
   if(Trace) plog("neo-genesis number: 0x%s", bnum2hex(ngnum));

   /* If we have any blocks >= the peer's NG block, delete them */
   delete_blocks(ngnum);
   if(trim_tfile(ngnum) != VEOK) restart("trim_tfile()");  /* panic */

   le_close();  /* close ledger */

   k = 0;
   peerip = gang[k];
   put64(bnum, ngnum);
   /* Get peer's neo-genesis block for new ledger.dat */
   if(!iszero(bnum, 8)) {
      sprintf(fname, "%s/b%s.bc", Bcdir, bnum2hex(bnum));
      for( ; Running; ) {
         if(get_block2(peerip, bnum, fname, OP_GETBLOCK) != VEOK) {
            if(++k >= Quorum) goto try_again;
            peerip = gang[k];
            continue;
         }
         if(extract(fname, "ledger.dat") != VEOK) restart("qu extract");
         if(reset_difficulty(NULL, Bcdir) != VEOK) restart("qu diff");
         break;
      }  /* end for */
   }  /* end if neo-genesis download */
   else extract_gen("ledger.dat");  /* use our genesis block ledger */

   le_open("ledger.dat", "rb");    /* re-open extracted ledger */
   reset_difficulty(NULL, Bcdir);  /* based on [neo-]genesis block */

   /* clear any failed download attempts */
   memset(Gpid, 0, sizeof(pid_t)*MAXQUORUM);
   memcpy(dlbnum, bnum, 8);
   do {
      add64(dlbnum, One, dlbnum);
      sprintf(fname, "rblock%02X%02X.dat", dlbnum[1], dlbnum[0]);
      sprintf(flock, "rblock%02X%02X.lck", dlbnum[1], dlbnum[0]);
      unlink(fname);
      unlink(flock);
   } while(dlbnum[0] != 0);

   /* download the blockchain */
   add64(bnum, One, bnum);
   for( ; Running; ) {
      memcpy(dlbnum, bnum, 8);
      for(i=0, j=0, k=0, result=0; Running && k<Quorum; result=0) {
         /* poll children */
         if(Gpid[k] > 0 && waitpid(Gpid[k], &result, WNOHANG) > 0) {
            if(result) gang[k] = 0;
            Gpid[k] = 0;
            ++i;
         }
         /* check peer */
         if((peerip = gang[k]) == 0 && ++k && ++j) continue;
         /* check blockchain */
         sprintf(fname, "rblock%02X%02X.dat", dlbnum[1], dlbnum[0]);
         sprintf(flock, "rblock%02X%02X.lck", dlbnum[1], dlbnum[0]);
         if(dlbnum[0] == 0 || exists(fname)) {
            add64(dlbnum, One, dlbnum);
            continue;
         }
         /* make-uh da baby */
         if(!Gpid[k]) {
            Gpid[k] = fork();
            if(Gpid[k] > 0) add64(dlbnum, One, dlbnum); /* parent */
            if(Gpid[k] < 0) gang[k] = 0; /* parent */
            if(Gpid[k] == 0) { /* child */
               fclose(fp = fopen(flock, "ab+")); /* get first dibs */
               result = get_block2(peerip, dlbnum, fname, OP_GETBLOCK);
               if(result != VEOK) unlink(fname);
               unlink(flock);
               exit(result);
            }
         }
         ++k;
      }
      /* rest during no activity */
      if(!i) usleep(Dynasleep);
      /* update downloaded blocks (also constructs NG on 0xff blocks) */
      sprintf(fname, "rblock%02X%02X.dat", bnum[1], bnum[0]);
      sprintf(flock, "rblock%02X%02X.lck", bnum[1], bnum[0]);
      while (exists(fname) && !exists(flock)) {
         if(update(fname, 0) != VEOK) goto try_again;
         add64(bnum, One, bnum);
         if(bnum[0] == 0) add64(bnum, One, bnum);
         sprintf(fname, "rblock%02X%02X.dat", bnum[1], bnum[0]);
         sprintf(flock, "rblock%02X%02X.lck", bnum[1], bnum[0]);
      }
      /* unable to download more blockchain */
      if(j>=Quorum) {
         if(cmp64(bnum, highbnum) >= 0) break;
         goto try_again;
      }
   }  /* end for block download-update */
   if(!Running) resign("quorum update");

   /* re-compute Weight[] */
   tfweight = tfval("tfile.dat", bnum, 1, &result);
   if(result) goto try_again;  /* should not happen */
   /* tfval set bnum to high block num on chain. */
   if(cmp64(bnum, Cblocknum) != 0) {
      error("get_eon(): line %d", __LINE__);  /* should not happen */
      goto try_again;
   }
   memcpy(Weight, tfweight, HASHLEN);

   if(Trace) plog("re-computed Weight = 0x...%x", Weight[0]);
   plog("Veronica says, 'You're done!'");

   return VEOK;

try_again:
   plog(":) (k: %d  time left: %d)", k, (int) (timeout - time(NULL)));
   if(Trace) {
      if(tfweight)
         plog("tfweight = 0x%s...", addr2str(tfweight));
      plog("highweight = 0x%s...", addr2str(highweight));
      plog("bnum = 0x%s", bnum2hex(bnum));
      plog("Cblocknum = 0x%s", bnum2hex(Cblocknum));
      plog("highbnum = 0x%s", bnum2hex(highbnum));
   }
   if(time(NULL) >= timeout) restart(":) timeout");  /* v.28 */

   for(peerip = 0; peerip == 0 && Running; )
      peerip = Rplist[rand16() % RPLISTLEN];
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
   char fname[128];
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
   if(result || cmp64(Cblocknum, highblock) != 0)
      fatal("init(): bad tfile.dat -- gomochi!");
   memcpy(Weight, wp, HASHLEN);

   /* read into Coreplist[], shuffle, and get IPL */
   printf("\n%s is the bootstrap nodes file list.\n", Corefname);
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
