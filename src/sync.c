/**
 * @private
 * @file sync.c
 * @copyright © Adequate Systems LLC, 2018-2021. All Rights Reserved.
 * <br />For more information, please refer to ../LICENSE
*/

/* include guard */
#ifndef MOCHIMO_SYNC_C
#define MOCHIMO_SYNC_C


#include "sync.h"

/* internal support */
#include "util.h"
#include "types.h"
#include "trigg.h"
#include "tfile.h"
#include "peach.h"
#include "network.h"
#include "ledger.h"
#include "global.h"
#include "bval.h"
#include "bup.h"

/* external support */
#include <sys/types.h>
#include <string.h>
#include <dirent.h>
#include "extthread.h"
#include "extmath.h"
#include "extlib.h"
#include "extint.h"
#include "extio.h"

#define THREADS_MAX  64

typedef struct {
   volatile int tr;  /* thread function result -- set by thread */
   word8 bnum[8];    /* blockchain file to download */
   word32 ip;        /* source ip */
} BIP_THREAD_ARGS;

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


/* Extract Genesis Block to ledger.dat */
int extract_gen(char *lfile)
{
   char fname[128];

   sprintf(fname, "%s/b0000000000000000.bc", Bcdir);
   /* extract the ledger from our Genesis Block */
   return le_extract(fname, lfile);
}

/**
 * Catch up by getting blocks from peers in plist[count].
 * Returns VEOK if updates made, else b_update() error code. */
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
            res = b_update(fname2, 0);
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
      result = ng_val(fname, bnum);
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

   /* Stop constructing and sending update blocks, since we're behind */
   stop_bcon();
   stop_found();

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
      if(b_update("spblock.tmp", 1) != VEOK) {
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
      if(b_update(buff, 0) != VEOK) {
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

/* Handle contention
 * Returns:  0 = nothing else to do
 *           1 = do fetch block with child
 */
int contention(NODE *np)
{
   word32 splitblock;
   TX *tx;
   word32 j;
   int result;
   BTRAILER *bt;

   pdebug("contention(): IP: %s", ntoa(&np->ip, NULL));

   tx = &np->tx;
   /* ignore low weight */
   if(cmp256(tx->weight, Weight) <= 0) {
      pdebug("contention(): Ignoring low weight");
      return 0;
   }
   /* ignore NG blocks */
   if(tx->cblock[0] == 0) {
      epinklist(np->ip);
      return 0;
   }

   if(memcmp(Cblockhash, tx->pblockhash, HASHLEN) == 0) {
      pdebug("contention(): get the expected block");
      return 1;  /* get block */
   }

   /* Try to do a simple catchup() of more than 1 block on our own chain. */
   j = get32(tx->cblock) - get32(Cblocknum);
   if(j > 1 && j <= NTFTX) {
        bt = (BTRAILER *) TRANBUFF(tx);  /* top of tx proof array */
        /* Check for matching previous hash in the array. */
        if(memcmp(Cblockhash, bt[NTFTX - j].phash, HASHLEN) == 0) {
           result = catchup(&np->ip, 1);
           if(result == VEOK) goto done;  /* we updated */
           if(result == VEBAD) return 0;  /* EVIL: ignore bad bval2() */
        }
   }
   /* Catchup failed so check the tx proof and chain weight. */
   if(checkproof(tx, &splitblock) != VEOK) return 0;  /* ignore bad proof */
   /* Proof is good so try to re-sync to peer */
   if(syncup(splitblock, tx->cblock, np->ip) != VEOK) return 0;
done:
   /* send_found on good catchup or syncup */
   send_found();  /* start send_found() child */
   addrecent(np->ip);
   return 0;  /* nothing else to do */
}  /* end contention() */

/* end include guard */
#endif
