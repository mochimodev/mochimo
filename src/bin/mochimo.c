/* mochimo.c
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * The Mochimo Project System Software
 * This file builds a server.
 *
 * Revised: 20 August 2018
*/

/* include guard */
#ifndef MOCHIMO_C
#define MOCHIMO_C


#define GPUMAX 64

/* ensure GIT_VERSION exists */
#ifndef GIT_VERSION
   #define GIT_VERSION "no-git-version"

#endif

/* system support */
#include <unistd.h>
#include <time.h>
#include <sys/wait.h>  /* for waitpid() */
#include <sys/file.h>  /* for flock() */
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <signal.h>
#include <fcntl.h>
#include <errno.h>
#include <ctype.h>

/* external support */
#include "sha256.h"
#include "extinet.h"    /* socket support */
#include "extlib.h"     /* general support */
#include "extmath.h"    /* 64-bit math support */
#include "extprint.h"   /* print/logging support */
#include "crc32.h"      /* for mirroring */
#include "crc16.h"

/* Include everything that we need */
#include "wots.h"
#include "util.h"       /* server support */
#include "tx.h"
#include "trigg.h"
#include "tfile.h"
#include "sync.h"
#include "peer.h"
#include "peach.h"
#include "network.h"
#include "ledger.h"
#include "global.h"     /* System wide globals  */
#include "bup.h"
#include "bcon.h"

int veronica(void)
{
   char haiku[256];
   trigg_generate(haiku);
   printf("\n%s\n\n", haiku);
   return VEOK;
}


/* Display system statistics */
int print_stats(void)
{
   print("Status:\n\n");
   print("   Aeon:            %u\n", Eon);
   print("   Generation:      %u\n", Ngen);
   print("   Online:          %u\n", Nonline);
   print("   Raw TX in:       %u\n", Nlogins);
   print("   Bad peers:       %u\n", Nbadlogs);
   print("   No space:        %u\n", Nspace);
   print("   Client timeouts: %u\n", Ntimeouts);
   print("   Server errors:   %u\n", get_num_errs());
   print("   TX recvd:        %u\n", Nrec);
   print("   TX dups:         %u\n", Ndups);
   print("   txq1 count:      %u\n", Txcount);
   print("   Balances sent:   %u\n", Nbalance);
   print("   Sends blocked:   %u\n", Nsenderrs);
   print("   Blocks solved:   %u\n", Nsolved);
   print("   Blocks updated:  %u\n\n", Nupdated);

   print("Current block: 0x%s\n", bnum2hex(Cblocknum));
   print("Weight:        0x...%s\n", bnum2hex(Weight));
   print("Difficulty:    %d  %s\n", Difficulty,
      Mpid ? "solving..." : "waiting for tx...");
   return 0;
} /* end print_stats() */

/* short stat display */
void betabait(void)
{
   word32 hps; /* haiku per second from miner.c hps.dat */

   if(read_data(&hps, sizeof(hps), "hps.dat") == sizeof(hps))
      Hps = hps;

   printf(     "Status:\n\n"
               "   Aeon:          %u\n"
               "   Generation:    %u\n"
               "   Online:        %u\n"
               "   TX recvd:        %u\n"
               "   Balances sent:   %u\n"
               "   Blocks solved:   %u\n"
               "   Blocks updated:  %u\n"
               "   Haiku/second:    %u %s\n"
               "\n",

                Eon, Ngen,
                Nonline,  Nrec, Nbalance, Nsolved, Nupdated,
                (word32) Hps, Hps ? "" : "(calculated after 2 TXs/updates)"
   );
   printf("Current block: 0x%s\n"
          "Difficulty:    %d  %s\n\n", bnum2hex(Cblocknum),
          Difficulty, Mpid ? "solving..." : "waiting for TX...");
} /* end betabait() */


void monitor(void)
{
   static word8 runmode = 0;   /* 1 for single stepping */
   static char buff[81] = { 0 };
   static char logfile[81] = LOGFNAME;     /* log file name */

   /*
    * Print banner if not single stepping.
    */
   if (runmode == 0) {
      print("\n\nMochimo System Monitor " GIT_VERSION "\n? for help\n\n");
   }

   show("monitor");
   /*
    * Command loop.
    */
   for(;;) {
      print("mochimo> ");
      tgets(buff, 80);

      /* process input */
      if (strcmp(buff, "st") == 0) print_stats();  /* stats command */
      else if (strcmp(buff, "si") == 0) {    /* single step command */
         runmode = 1 - runmode;
         print("Single step is %s\n", runmode ? "on." : "off.");
      } else if (strcmp(buff, "ll") == 0) {  /* print level command */
         print("Note: affects the level of logs printed to screen.\n");
         print("Log level (%d-%d): ", PLEVEL_NONE, PLEVEL_DEBUG);
         /* additional input required */
         if (*tgets(buff, 80)) set_print_level(atoi(buff));
      } else if (strcmp(buff, "ol") == 0) {  /* log level command */
         print("Note: affects the level of logs printed to files.\n");
         print("Output level (%d-%d): ", PLEVEL_NONE, PLEVEL_DEBUG);
         /* additional input required */
         if (*tgets(buff, 80)) set_output_level(atoi(buff));
      } else if (strcmp(buff, "r") == 0) {    /* restart command */
         print("Confirm restart (Y/n)? ");
         /* additional input required */
         tgets(buff, 80);
         if (strcmp(buff, "Y") == 0) restart("monitor");
      } else if (strcmp(buff, "q") == 0) {   /* signal server to exit */
         Monitor = runmode;
         Running = 0;
         return;
      } else if (strcmp(buff, "p") == 0) {   /* print peers command */
         print("Trusted peers:\n");
         print_ipl(Tplist, TPLISTLEN);
         print("Recent peers:\n");
         print_ipl(Rplist, RPLISTLEN);
         print("Pinklisted:\n");
         print_ipl(Epinklist, RPLISTLEN);
         continue;
      } else if (strcmp(buff, "o") == 0) {   /* toggle log file */
         set_output_file(NULL, NULL);
         print("Log file is closed.\n");
         print("Log file [%s]: ", logfile);
         /* additional input required */
         if(*tgets(buff, 80)) strncpy(logfile, buff, 80);
         if (set_output_file(logfile, "a")) {
            print("Cannot open %s\n", logfile);
         } else print("Log file %s is open.\n", logfile);
      } else if (strcmp(buff, "m") == 0) {   /* mining mode */
         print("Enable mining (Y/n) [%s]? ", Nominer ? "n" : "Y");
         /* additional input required */
         if (*tgets(buff, 80)) {
            if (strcmp(buff, "Y") == 0) Nominer = 0;
            else Nominer = 1;
         }
      } else if (*buff == '\0') {   /* ENTER to continue server */
         Monitor = runmode;
         print("In server() loop...\n");
         return;
      }

      /*
       * Print help message.
       */
      printf("\nCommands:\n\n"
             "<enter> resume server\n"
             "q       quit server\n"
             "r       restart server\n"
             "ll      set log level (screen)\n"
             "ol      set log level (output)\n"
             "o       toggle log file\n"
             "p       display peer lists\n"
             "m       set mining mode\n"
             "si      toggle single step mode\n"
             "st      display system status\n"
             "?       this message\n\n" );

   }  /* end command loop */
}  /* end monitor() */

/* miner blockin blockout -- child process */
int miner(char *blockin, char *blockout)
{
#ifdef CUDA
   DEVICE_CTX GPU[GPUMAX] = { 0 };
   DEVICE_CTX *CudaGPUs = NULL;
   int num_cuda_gpus = 0;
   int m;

#endif
   BTRAILER bt, btout;
   SHA256_CTX bctx;
   FILE *fp;
   time_t start, prev;
   word32 n;
   char phaiku[256];
   int ecode;

   show("solving");

   /* pre-checks */
   if (!fexists(blockin)) {
      mError(FAIL, "miner(): missing blockin %s", blockin);
   }

   /* init */
   memset(&btout, 0, sizeof(btout));
   time(&start);
   time(&prev);
   n = 0;

#ifdef CUDA
      CudaGPUs = GPU;
      num_cuda_gpus = peach_init_cuda(CudaGPUs, GPUMAX);
      pdebug("Cuda Devices = %d", num_cuda_gpus);

#endif

   /* begin miner loop */
   while (Running) {
      /* read candidate block file and bctx */
      if (read_data(&bctx, sizeof(bctx), "bctx.dat") != sizeof(bctx)) {
         mError(FAIL, "miner(): failed to read_data(bctx)");
      } else if (readtrailer(&bt, blockin) != VEOK) {
         mError(FAIL, "miner(): failed to read %s trailer", blockin);
      }
      /* remove miner files */
      remove("bctx.dat");
      remove("miner.tmp");
      /* copy blockin to miner.tmp to perform current work */
      if (fcopy(blockin, "miner.tmp") != 0) {
         mError(FAIL, "miner(): failed to copy %s to miner.tmp", blockin);
      }


         pdebug("miner(): solving block: 0x%s", bnum2hex(bt.bnum));
#ifdef CUDA
         for (m = 0, n = 0; Running; m++, usleep(Dynasleep)) {
            if (m >= num_cuda_gpus) m = 0;

#else
         for (peach_init(&bt); Running; n++) {

#endif
            /* check time every 256 iterations -- or always if -DCUDA */
            if ((n & 0xff) == 0) {
               /* check for new bctx.dat file every second */
               if (difftime(time(NULL), prev)) {
                  time(&prev);
                  /* on new bctx, break inner loop for efficient restart */
                  if (fexists("bctx.dat")) break;
               }
            }

#ifdef CUDA
            /* perform solve work -- cpu */
            if (peach_solve_cuda(&CudaGPUs[m], &bt, 0, &btout) == VEOK) {
               memcpy(bt.nonce, btout.nonce, HASHLEN);

#else
            /* perform solve work -- cpu */
            if (peach_solve(&bt, bt.difficulty[0], btout.nonce) == VEOK) {

#endif
               /* double check solve is valid */
               if (peach_check(&bt) != VEOK) {
                  mError(FAIL, "miner(): invalid solve!\n");
                  memset(&btout, 0, sizeof(btout));
               } else {
                  /* solve is valid */
                  show("solved");
                  trigg_expand(bt.nonce, phaiku);
                  if (!Bgflag) print("\nSOLVED!!!\n\n%s\n", phaiku);
                  /* ensure solve time is valid */
                  while (Running) {
                     if (difftime(time(NULL), get32(bt.time0)) > 0) break;
                     pwarn("miner(): early solve! Check system clock...");
                     sleep(1);
                  }
                  /* put solve time in trailer */
                  put32(bt.stime, time(NULL));
                  /* add nonce and stime to bctx; put hash in trailer */
                  sha256_update(&bctx, bt.nonce, HASHLEN + 4);
                  sha256_final(&bctx, bt.bhash);
                  /* write trailer to miner.tmp */
                  fp = fopen("miner.tmp", "r+b");
                  if (fp == NULL) {
                     mError(FAIL, "miner(): cannot re-open miner.tmp");
                  } else if (fseek(fp, -(sizeof(bt)), SEEK_END) != 0) {
                     mError(FAIL_FP, "miner(): fseek(bt) on miner.tmp");
                  } else if (fwrite(&bt, sizeof(bt), 1, fp) != 1) {
                     mError(FAIL_FP, "miner(): fwrite(bt) on miner.tmp");
                  } else fclose(fp);
                  /* move mblock.tmp to blockout */
                  remove(blockout);
                  if (rename("miner.tmp", blockout) != 0) {
                     mError(FAIL, "miner(): cannot rename miner.tmp");
                  }
                  /* final log */
                  pdebug("miner: solved 0x%s", bnum2hex(bt.bnum));
                  Running = 0;
               }  /* end if (peach_check(&bt)... else... */
            }  /* end if (peach_solve(&bt, diff, bt.nonce)... */
         }  /* end for (peach_init(&bt)... */


      /* record hps */
#ifdef CUDA
      for (Hps = 0, m = 0; m < num_cuda_gpus; m++) {
         Hps += (word32)
            (CudaGPUs[m].total_work / difftime(time(NULL), start));
      }
#else
      Hps = n / (word32) difftime(time(NULL), start);
#endif
      write_data(&Hps, sizeof(Hps), "hps.dat");
   }  /* end while(Running) */

   /* finish -- gracefully */
   return VEOK;

   /* failure -- cleanup/error handling */
FAIL_FP:
   fclose(fp);
FAIL:

   return ecode;
}  /* end miner() */

/* Start the miner as a child process */
int start_miner(void)
{
   pid_t pid;

   if(Mpid) return VEOK;
   pid = fork();
   if(pid < 0) return VERROR;
   if(pid) { Mpid = pid; return VEOK; }  /* parent */
   /* child */
   miner("cblock.dat", "mblock.dat");
   exit(0);
}  /* end start_miner() */

/**
 * Initialize the server/client from any state
 * after executing the gomochi script. */
int init(void)
{
   /* static word8 FortyEight[8] = { 48, }; */
   char fname[FILENAME_MAX], weightstr[65];
   word32 peer, qlen, quorum[MAXQUORUM];
   word8 nethash[HASHLEN], peerhash[HASHLEN];
   word8 netweight[32], netbnum[8]; //, bnum[8];
   /* BTRAILER bt; */
   NODE node;  /* holds peer tx.cblock and tx.cblockhash */
   int result, status, attempts, count;
   word8 highblock[8];

   /* init */
   show("init");
   plog("Initializing...");
   status = VEOK;
   attempts = 0;

   /* ensure appropriate directories and permissions exist */
   if (check_directory(Bcdir) || check_directory(Spdir)) return VERROR;

   /* update coreip list where available */
   snprintf(fname, FILENAME_MAX, "../%s", Coreipfname);
   if (fcopy(fname, Coreipfname) != VEOK) {
      if (!fexists(Coreipfname)) {
         pwarn("missing Core ip list..., %s", Coreipfname);
      }
   }
   /* update trustedip list where available */
   snprintf(fname, FILENAME_MAX, "../%s", Trustedipfname);
   fcopy(fname, Trustedipfname);
   /* update maddr.dat - use maddr.MAT as last resort only */
   if (fcopy("../maddr.dat", "maddr.dat") != VEOK) {
      if (!fexists("maddr.dat")) {
         if (fcopy("../maddr.mat", "maddr.dat") != VEOK) {
            return perr("Failed to copy mining address");
         } else pwarn("using maddr.MAT (the founder's mining address)");
      }
   }
   /* restore core chain files if any do not exist */
   snprintf(fname, FILENAME_MAX, "%s/b0000000000000000.bc", Bcdir);
   if (!fexists("tfile.dat") || !fexists(fname)) {
      pdebug("Core chain files compromised, attempting restoration...");
      if (fcopy("../genblock.bc", fname) != VEOK) {
         return perr("Failed to restore %s from ../genblock.bc", fname);
      } else if (fcopy("../tfile.dat", "tfile.dat") != VEOK) {
         return perr("Failed to restore tfile.dat from ../tfile.dat");
      }
   }
   /* open ledger or extract from genesis block */
   if (!fexists("ledger.dat") || le_open("ledger.dat", "rb") != VEOK) {
      pdebug("Extracting ledger from ../genblock.bc ...");
      if (le_extract("../genblock.bc", "ledger.dat") != VEOK) {
         return perr("Failed to extract ledger from ../genblock.bc");
      } else if (le_open("ledger.dat", "rb") != VEOK) { /* try again */
         return perr("Failed to open ledger.dat");
      }
   }

   plog("Init chain...");
   /* Find the last block in bc/ and reset Time0, and Difficulty */
   if (reset_chain() != VEOK) return perr("reset_chain() failed");
   /* validate our own tfile.dat to compute Weight */
   if (tf_val("tfile.dat", highblock, Weight, 1)) {
      perr("init(): bad tfile.dat -- resync");
      memset(Cblocknum, 0, 8);  /* flag resync */
   } else if (cmp64(Cblocknum, highblock)) {
      pdebug("init(): %d %d", get32(Cblocknum), get32(highblock));
      pdebug("init(): %s", weight2hex(Weight));
      perr("init(): tfile mismatch -- resync");
      memset(Cblocknum, 0, 8);  /* flag resync */
   } else if (!iszero(Cblocknum, 8)) {
      plog(" - 0x%" P32x " 0x%s", get32(Cblocknum),
         val2hex(Weight, 32, weightstr, 65));
   }

   /* scan entire network of peers */
   while (Running) {
      /* reset peers, download start peers, initialize peer lists */
      plog("Init peers...");
      remove("start.lst");
      http_get("https://mochimo.org/peers/start", "start.lst", STD_TIMEOUT);
      count = read_ipl(Epinkipfname, Epinklist, EPINKLEN, &Epinkidx);
      if (count > 0) plog(" - added %" P32u " pinklisted peers", count);
      count = read_ipl(Trustedipfname, Tplist, TPLISTLEN, &Tplistidx);
      if (count > 0) plog(" - added %" P32u " trusted peers", count);
      count = read_ipl(Trustedipfname, Rplist, RPLISTLEN, &Rplistidx);
      count += read_ipl(Coreipfname, Rplist, RPLISTLEN, &Rplistidx);
      count += read_ipl("start.lst", Rplist, RPLISTLEN, &Rplistidx);
      if (count > 0) plog(" - added %" P32u " recent peers", count);
      /* ensure recent peers list is shuffled */
      shuffle32(Rplist, RPLISTLEN);
      /* scan network for quorum and highest hash/weight/bnum */
      plog("Init network...");
      qlen = scan_network(quorum, MAXQUORUM, nethash, netweight, netbnum);
      plog(" - %d/%d 0x%" P32x " 0x%s", qlen, MAXQUORUM, get32(netbnum),
         val2hex(netweight, 32, weightstr, 65));
      if (qlen == 0) break; /* all alone... */
      else if (qlen < Quorum) {  /* insufficient quorum */
         plog("Insufficient quorum, try again...");
         /* without considering the expansion of acceptable
          * quorum size, infinite loop is possible... */
         continue;
      } else shuffle32(quorum, qlen);
      if (!iszero(Cblocknum, 8)) {  /* we've got a chain */
         status = VEOK;  /* don't panic... EVERYTHING IS FINE! */
         result = cmp256(Weight, netweight);  /* compare network weight */
         if (result < 0) {
            pdebug("network weight compares higher");
            print("\n");
            print(" ┌────────────────────────────────────┐\n");
            print(" │ an overwhelming sense of confusion │\n");
            print(" └────────────────────────────────────┘\n\n");
         } else if (result > 0) {
            pdebug("network weight compares lower");
            print("\n");
            print(" ┌────────────────────────────────────┐\n");
            print(" │   an overwhelming sense of power   │\n");
            print(" └────────────────────────────────────┘\n\n");
            break;  /* we're heavier, finish */
         } else if (memcmp(Cblockhash, nethash, HASHLEN) == 0) {
            pdebug("network weight and hash compares equal");
            print("\n");
            print(" ┌────────────────────────────────────┐\n");
            print(" │ an overwhelming sense of belonging │\n");
            print(" └────────────────────────────────────┘\n\n");
            break;  /* we're in sync, finish */
         }
         /* have we fallen behind or split from the chain? */
         while (Running && (peer = *quorum)) {  /* use quorum to check... */
            if (status == VEOK) {  /* chain status not yet known... */
               plog("Checking blockchain alignment...");
               if (get_hash(&node, peer, Cblocknum, peerhash) == VEOK) {
                  if (memcmp(Cblockhash, peerhash, HASHLEN)) {
                     status = VEBAD; /* 2319! Foreign entity (block) */
                  } else status = VERROR;  /* we're just behind */
                  continue;  /* restart loop with new status */
               }
            } else if (status == VERROR) {  /* chain is fallen... */
               plog("Blockchain is aligned, catchup...");
               catchup(quorum, qlen);  /* try to catchup with blockchain */
               break;
            } else if (status == VEBAD) {  /* chain is split... */
               plog("2319!!! CHAIN FORK DETECTED...");
               /* attempt chain recovery... DISABLED FOR NOW
               put64(bnum, cmp64(Cblocknum, netbnum) > 0 ? netbnum : Cblocknum);
               if (sub64(bnum, FortyEight, bnum)) break;
               count = readtf(&bt, get32(bnum), get32(FortyEight));
               if (count % sizeof(BTRAILER)){
                  perr("init(): error reading tfile, count= %s", count);
                  break;
               } else {
                  // acquire same segment of Tfile as above and compare
                  if (get_hash(&node, peer, bnum, peerhash) == VEOK) {}
                  if (memcmp(Cblockhash, peerhash, HASHLEN) == 0) {
                     if (syncup(bnum, netbnum, peer) == VEOK) break;
                  } else continue;
               } */
               break;
            }
            remove32(peer, quorum, MAXQUORUM, &qlen);
         }  /* ... did we catch up? */
         if (cmp256(Weight, netweight) >= 0) {
            if (cmp64(Cblocknum, netbnum) >= 0) break;
         }  /* ... whatever we did, it didn't work... */
      }  /* ... at this point, we might as well resync */
      if (!qlen) plog("Quorum members exhausted...");
      else {
         if (attempts++) plog("Blockchain recovery failed: resync...");
         if (resync(quorum, &qlen, netweight, netbnum) == VEOK) break;
      }
      /* we're either out of quorum members, or our resync failed */
      plog("Resync failure: try again...\n\n");
   }

   write_global();
   le_txclean();
   Ininit = 0;

   return Running ? VEOK : VERROR;
}  /* end init() */

/**
 * The Mochimo Server/Client!
 *
 * Uses globals from data.c
 */
int server(void)
{
   /* real time of current server loop - set by server() */
   static time_t Ltime;
   static time_t Stime;    /* status display update time */
   static time_t nsd_time;  /* event timers */
   static time_t bctime, mwtime, mqtime, sftime, vtime;
   static time_t ipltime;
   static SOCKET lsd, nsd;
   static NODE *np, node;
   static struct sockaddr_in addr;
   static int status;   /* child return status */
   static pid_t pid;    /* child pid */
   static int lfd;      /* for lock() */
   static word32 hps;  /* same as Hps in monitor.c */
   static word16 opcode;
   char fname[FILENAME_MAX];

   /* Initialise event timers */
   Ltime = time(NULL);      /* real time GMT in seconds */
   Stime = Ltime + 10;      /* status display time */
   bctime = Ltime + 30;     /* block constructor time */
   mwtime = Ltime + 6;
   mqtime = Ltime + 5;      /* mirror() time */
   Utime = Ltime;           /* for watchdog timer */
   Watchdog = WATCHTIME + (rand16() % 600);
   ipltime = Ltime + (rand16() % 300) + 10;  /* ip list fetch time */
   sftime = Ltime + (rand16() % 300) + 300;  /* send_found() time */
   vtime = Ltime + 4;  /* Verisimility restart check time */

   lsd = socket(AF_INET, SOCK_STREAM, 0);
   if (lsd == INVALID_SOCKET) restart("Cannot open listening socket.");
   memset(&addr, 0, sizeof(addr));    /* clear address structure   */
   addr.sin_port = htons(Port);
   addr.sin_addr.s_addr = INADDR_ANY;
   addr.sin_family = AF_INET;

   show("bind");
   for(;;) {
      if(!Running) { sock_close(lsd); return 0; }
      if(bind(lsd, (struct sockaddr *) &addr, sizeof(addr)) == 0) break;
      plog("Trying to bind port %d...", Port);
      sleep(5);
      if(Monitor && !Bgflag) monitor();
   }

   /* set listening port non-blocking for accept() */
   if (sock_set_nonblock(lsd) == SOCKET_ERROR) {
      restart("sock_set_nonblock() failed on lsd.");
   }
   listen(lsd, LQLEN);  /* LQSIZ */
   nsd = INVALID_SOCKET;

   if (Safemode && !iszero(Cblocknum, 8)) {
      plog("\nSafemode...\n");
      send_found();
   } else plog("\nListening...\n");

   unlink("vstart.lck");  /* signal Verisimility that we are up. */

   /*
    * Main server loop.
    */

   while(Running) {
      /*
       * Get current time for this generation.
       */
      Ltime = time(NULL);

      show("listen");  /* display status for ps */

      /* Reap zombies and collect status.
       * No child left behind...
       */
      for(np = Nodes; np < Hi_node; np++) {
         if(np->pid == 0) continue;
         pid = waitpid(np->pid, &status, WNOHANG);
         if(pid <= 0) continue;  /* child still running or signal */
         freeslot(np);
         opcode = get16(np->tx.opcode);
         pdebug("np->pid: %d  pid: %d  status: 0x%x  op: %" P16u "  (%d)",
                        np->pid, pid, status, opcode, errno);  /* debug */
         /* Adds to lists if needed and returns exit status 0-3 */
         status = child_status(np, pid, status);
         if(opcode == OP_FOUND) {
            if(Blockfound == 0) perr("server(): line %d", __LINE__);
            else {
               stop4update();
               if(b_update("rblock.dat", 0) == VEOK) {
                  send_found();  /* start send_found() child */
                  addrecent(np->ip);   /* v.28 */
                  Stime = Ltime + 20;  /* hold status display */
               }
               Blockfound = 0;
            }
         }  /* end if OP_FOUND child */
         else if(opcode == OP_GET_BLOCK || opcode == OP_GET_TFILE) {
            if(get16(np->tx.len) == 0 && status == 0) {
               addrecent(np->ip);
            }
         }
      }  /* end for check Node[] zombies */

      /* Reap a send_found() child.  If she is done, pid != 0. */
      if(Found_pid > 0) {
         pid = waitpid(Found_pid, &status, WNOHANG);
         if(pid > 0) Found_pid = 0;
      }

      /* Check for new connection with accept() and set nsd. */
      if(nsd == INVALID_SOCKET) {
         if((nsd = accept(lsd, NULL, NULL)) != INVALID_SOCKET) {
            sock_set_nonblock(nsd);
            nsd_time = Ltime;
         }
      }

      /*
       * If accept() good,
       * try to read data from socket, nsd, using gettx().
       */
      if(nsd != INVALID_SOCKET) {
         /* gettx() completes the initial handshake and fills node
          * and some parent tables.  It returns -1 if no data yet.
          * If gettx() completes the transaction, it returns 0, 1, 2, or 3;
          * otherwise it returns sizeof(TX) and needs help from child
          * so getslot() allocates a new np and copies node into it.
          */
         status = gettx(&node, nsd);  /* fills in node */
         if(status != -1) {
            if(status == VEOK && (np = getslot(&node)) != NULL) {
               pid = fork();  /* create child to handle TX */
               if(pid == 0) {
                  /* in child -- execute() */
                  opcode = get16(np->tx.opcode);
                  pdebug("execute(): opcode = %d", opcode);
                  switch (opcode) {
                     case OP_FOUND:
                        /* get the advertised found block -- synchronous
                        * Blockfound was set by gettx()
                        */
                        sock_close(np->sd);  /* close initial connection */
                        status =
                           get_file(np->ip, np->tx.cblock, "rblock.dat");
                        break;
                     case OP_GET_BLOCK:
                        /* send np->tx.blocknum to peer */
                        status = send_file(np, NULL);
                        break;
                     case OP_GET_TFILE:
                        /* send out tfile.dat to peer */
                        status = send_file(np, "tfile.dat");
                        break;
                     case OP_GET_CBLOCK:
                        /* send out cblock.dat to peer via file copy */
                        status = fexists("cblock.dat") ? VEOK : VERROR;
                        if (status == VEOK) {
                           sprintf(fname, "cb%u.tmp", (unsigned) getpid());
                           status = fcopy("cblock.dat", fname);
                           if (status == VEOK) {
                              status = send_file(np, fname);
                              remove(fname);
                           }
                        }
                        break;
                     case OP_MBLOCK:
                        /* receive mined block as mblock.dat from peer */
                        status = recv_file(np, "mblock.tmp");
                        if (status != VEOK || fexists("mblock.dat")) {
                           remove("mblock.tmp");
                        } else {
                           rename("mblock.tmp", "mblock.dat");
                           ftouch("cblock.lck");
                        }
                        break;
                     case OP_TF:
                        /* send tfile.dat section to peer */
                        status = send_tf(np);
                        break;
                     default:
                        Nbadlogs++;  /* bad OP's */
                        pdebug("execute(): bad opcode: %d", opcode);
                        status = VEBAD;
                  }  /* end switch op */
                  sock_close(np->sd);
                  /* IMPORTANT: the exit status MUST NOT be less than 0.
                   * When the parent calls WEXITSTATUS(), only 8-bits of
                   * the status are returned. VETIMEOUT results in an
                   * underflow and (currently) causes pinklisted peers! */
                  if (status < 0) status = VERROR;
                  exit(status);  /* parent calls waitpid() for status */
               }
               /* parent puts valid child pid in parent table */
               if(pid != -1) np->pid = pid;
               else {
                  /* fork() failed so freeslot() removes child data from
                   * parent Node[] table.
                   */
                  freeslot(np);
                  perr("fork() failed!");
                  restart("cannot fork()");
               }
            }  /* end if need child and slot found */
            /* parent closes its socket if gettx() did not */
            if(node.sd != INVALID_SOCKET)
               sock_close(nsd);
            nsd = INVALID_SOCKET;
         } else {   /* status == -1 no data yet -- so check timeout */
            if(Ltime - nsd_time > INIT_TIMEOUT) {
               Ntimeouts++;  /* log statistics */
               sock_close(nsd);
               nsd = INVALID_SOCKET;
            }
         }  /* end if timeout */
      }  /* endif nsd valid */

      Ngen++;  /* loop counter */

      /*
       * Take care of business...
       */

      /* Check miner */
      if(Blockfound == 0 && fexists("mblock.dat")) {
         Blockfound = 1;
         if(cmp64(Cblocknum, Bcbnum) == 0) {
            /* We solved a block! */
            stop4update();
            if (b_update("mblock.dat", 1) == VEOK) {
               send_found();  /* start send_found() child */
               Stime = Ltime + 20;  /* hold status display */
            }
         }
         unlink("mblock.dat");
         Blockfound = 0;
      }

      /* generate pseudo-block in "times of trouble", else check bcon */
      if(Ltime >= (Time0 + BRIDGE) && TIMES_OF_TROUBLE(Cblocknum)) {
         if (pseudo("pblock.dat") != VEOK) {
            restart("Failed to make pseudo-block");
         } else {
            stop4update();
            if (b_update("pblock.dat", 2) != VEOK) {
               restart("Failed to update pseudo-block");
            } else Stime = Ltime + 20;  /* hold status display */
         }
      } else {
         if (Txcount >= TXQUEBIG) bctime = Ltime;
         if (Bcon_pid == 0 && Blockfound == 0 && Ltime >= bctime &&
            (Txcount > 0 || (Mpid == 0 && fexistsnz("txclean.dat")))) {
            pdebug("spawning bcon with %d more transactions", Txcount);
            /* append txq1.dat to txclean.dat */
            system("cat txq1.dat >>txclean.dat 2>/dev/null");
            remove("txq1.dat");
            Txcount = 0;  /* txq1.dat is empty now */
            put64(Bcbnum, Cblocknum);  /* save current block number */
            Bcon_pid = fork();
            if (Bcon_pid == -1) {
               perr("Cannot fork() for b_con()");
               Bcon_pid = 0;
            } else if (Bcon_pid == 0) {
               /* in child */
               exit(b_con("cblock.dat"));
            }
            bctime = Ltime + BCONFREQ;
         }
      }

      /* Collect bcon status when she is 'done'.  pid == 0 means she
       * is still busy.
       */
      if(Bcon_pid > 0) {
         pid = waitpid(Bcon_pid, &status, WNOHANG);
         if(pid > 0) {
            Bcon_pid = 0;  /* pid not zero means she is done. */
            if(!Nominer) {
               start_miner();  /* start or re-start miner */
            }
         }
      }
      /* bcon sequence will wait on miner if Txcount > 0,
       * else...
       */
      if(Mpid && Ltime >= mwtime) {
         pid = waitpid(Mpid, &status, WNOHANG);
         if(pid > 0) Mpid = 0;  /* Miner exited. */
         mwtime = Ltime + 120;
      }

      /* Start mirror()? */
      if(Ltime >= mqtime && Mqcount > 0 && Mqpid == 0) {
         /* get exclusive access to txq1.dat */
         lfd = lock("mq.lck", 10);
         if(lfd != -1) {
            unlink("mirror.dat");
            rename("mq.dat", "mirror.dat");
            Mqcount = 0;
            unlock(lfd);
            Mqpid = mirror();  /* start child */
         }
      }
      if(Mqpid) {
         pid = waitpid(Mqpid, NULL, WNOHANG);
         if(pid > 0) {
            Mqpid = 0;
            mqtime = Ltime + 2;
         }
      }

      /*
       * Display system statistics
       */
      if(Ltime >= Stime) {
         if(read_data(&hps, sizeof(hps), "hps.dat") == sizeof(hps))
            Hps = hps;
         if(Betabait && Bgflag == 0) betabait();
         Stime = Ltime + STATUSFREQ;
      }
      /*
       * Monitor interrupt on Ctrl-C if not in background
       */
      if(Monitor && !Bgflag) monitor();

      if(Watchdog && (Ltime - Utime) >= Watchdog) {
         restart("watchdog");
      }

      /* Check for restart signal from Verisimility every 4 seconds */
      if(Ltime >= vtime) {
         if(fexists("vstart.lck")) restart("Verisimility");
         vtime += 4;
      }

      if(Ltime >= ipltime) {
         refresh_ipl();  /* refresh ip list */
         ipltime = Ltime + (rand16() % 300) + 10;
      }

      /* Check random send_found() timer */
      if(Ltime >= sftime) {
         if(Found_pid == 0) send_found();
         sftime = Ltime + (rand16() % 300) + 300;
      }

      /* dynamic sleep function */
      if(Dynasleep != 0 && Nonline < 1) usleep(Dynasleep);
   } /* end while(Running) */

   /* cleanup */
   plog("Server exiting, please wait...");
   sock_close(lsd);  /* close listening socket */

   return 0;
} /* end server() */

int usage(void)
{
   printf("\n"
      "usage: mochimo [-option...]\n"
      "options:\n"
      "   -lN        set Lo Level to N (0-5)\n"
      "   -o         open mochi.log file\n"
      "   -oFNAME    open log file FNAME\n"
      "   -e         enable error.log file\n"
      "   -qN        set Quorum to N (default 4)\n"
      "   -vN        set virtual mode: N = 1 or 2\n"
      "   -cFNAME    read core ip list from FNAME\n"
      "   -tFNAME    read trusted ip list from FNAME\n"
      "   -c         disable read core ip list\n"
      "   -d         disable pink lists\n"
      "   -pN        set port to N\n"
      "   -D         Daemon ignore ctrl-c and no term output\n"
      "   -sN        sleep N usec. on each loop if not busy\n"
      "   -xxxxxxx   replace xxxxxxx with state\n"
      "   -S         Safe mode\n"
      "   -F         Filter private IP's\n"  /* v.28 */
      "   -P         Allow pushed mblocks\n"
      "   -n         Do not solve blocks\n"
      "   -Mn        set transaction fee to n\n"
      "   -Sanctuary=N,Lastday\n"
      "   -Tn        set Trustblock to n for tfval() speedup\n"
   );
#ifdef BX_MYSQL
   printf("   -X         Export to MySQL database on block update\n");
#endif
   printf("\n");

   return VEOK;
}  /* end usage() */

/* 
 * Initialise data and call the server.
 */
int main(int argc, char **argv)
{
   static int k, j;
   static char proc_name[64], *cp;
   static word8 endian[] = { 0x34, 0x12 };

   /* sanity checks -- for undesired structure padding */
   if (sizeof(word32) != 4) resign("word32 should be 4 bytes");
   else if (sizeof(TX) != TXBUFFLEN) {
      resign("struct size error TX != TXBUFFLEN");
   } else if (sizeof(LTRAN) != (TXADDRLEN + 1 + TXAMOUNT)) {
      resign("struct size error: LTRAN != (TXADDRLEN + 1 + TXAMOUNT)");
   } else if (sizeof(BTRAILER) != BTSIZE) {
      resign("struct size error: BTRAILER != BTSIZE");
   } else if (sizeof(MTX) != sizeof(TXQENTRY)) {
      resign("struct size error: MTX != TXQENTRY");
   } else if (get16(endian) != 0x1234) {
      resign("little-endian machine required for this build.");
   }

   /* pre-init */
   Ininit = 1;
   Running = 1;
   fix_signals();             /* redirect signals */
   signal(SIGCHLD, SIG_DFL);  /* so waitpid() works */
   set_output_level(PLEVEL_DEBUG);
   set_print_level(PLEVEL_LOG);

   /* sanity check -- for duplicate processes */
   cp = strrchr(argv[0], '/');
   if (cp == NULL) cp = strrchr(argv[0], '\\');
   if (cp++ && *cp) strncpy(proc_name, cp, sizeof(proc_name));
   else strncpy(proc_name, argv[0], sizeof(proc_name));
   if (proc_dups(proc_name)) resign("Mochimo is already running!");

   /* improve random generators */
   srand16fast(time(NULL) ^ getpid());
   srand16(time(NULL), 0, 123456789 ^ getpid());
   /* enable socket support */
   sock_startup();

   /* Parse command line arguments. */
   for (j = 1; Running && j < argc; j++) {
      if(argv[j][0] != '-') return usage();
      switch(argv[j][1]) {
         case '-':  /* advanced commands */
            cp = &argv[j][2];
            if (*cp == '\0') goto EOA;  /* -- end of args */
            else if (strcmp("help", cp) == 0) exit(usage());
            else if (strcmp("testnet", cp) == 0) exit(testnet());
            else if (strcmp("veronica", cp) == 0) exit(veronica());
            else perr("Unknown argument, %s", argv[j]);
            break;
         case 'c':  /* set core ip list */
            if (!argv[j][2]) {
               perr("missing coreip list file");
               exit(usage());
            }
            Coreipfname = &argv[j][2];  /* master network */
            break;
         case 'd':  /* disable pink lists */
            Nopinklist = 1;
            break;
         case 'D':  /* enable daemon mode */
            Bgflag = 1;
            setpgrp();
            break;
         case 'F':  /* disable private IPs */
            Noprivate = 1;  /* v.28 */
            break;
         case 'l':  /* set log level  */
            if (!argv[j][2]) {
               perr("missing log level value\n");
               exit(usage());
            }
            set_print_level((k = atoi(&argv[j][2])));
            set_output_level(k);
            break;
         case 'n':  /* disable miner */
            Nominer = 1;
            break;
         case 'P':  /* enabled cblock push */
            Allowpush = 1;
            Cbits |= C_PUSH;
            break;
         case 'M':  /* set own Mining fee */
            Myfee[0] = atoi(&argv[j][2]);
            if (Myfee[0] < Mfee[0]) Myfee[0] = Mfee[0];
            else Cbits |= C_MFEE;
            break;
         case 'o':  /* open log file used by plog() */
            if (argv[j][2]) set_output_file(&argv[j][2], "a");
            else set_output_file(LOGFNAME, "a");
            Cbits |= C_LOGGING;
            break;
         case 'p':  /* set home/dst communication port */
            Port = Dstport = atoi(&argv[j][2]);
            break;
         case 'q':  /* set quorum */
            Quorum = atoi(&argv[j][2]);
            if (Quorum > MAXQUORUM) {
               perr("quorum exceeds MAXQUORUM=%u", MAXQUORUM);
               exit(usage());
            }
            break;
         case 's':  /* set Dynasleep */
            Dynasleep = atoi(&argv[j][2]);  /* usleep time */
            break;
         case 'S':  /* Safemode or Sanctuary */
            if (strncmp(argv[j], "-Sanctuary=", 11) == 0) {
               cp = strchr(argv[j], ',');
               if (cp == NULL) {
                  perr("Invalid Sanctuary Protocol");
                  exit(usage());
               }
               Sanctuary = strtoul(&argv[j][11], NULL, 0);
               Lastday = (strtoul(cp + 1, NULL, 0) + 255) & 0xffffff00;
               Cbits |= C_SANCTUARY;
               printf("\nSanctuary=%u  Lastday 0x%0x...", Sanctuary,
                  Lastday); fflush(stdout); sleep(2);
               printf("\b\b\b accounted for.\n"); sleep(1);
               break;
            }
            if (argv[j][2]) exit(usage());
            Safemode = 1;
            break;
         case 't':  /* set trusted ip list */
            if (!argv[j][2]) {
               perr("missing trusted ip list file");
               exit(usage());
            }
            Trustedipfname = &argv[j][2];  /* master network */
            break;
         case 'T':  /* enabled Trustblock */
            Trustblock = atoi(&argv[j][2]);
            break;
         case 'v':
            if (argv[j][2] == '2') { Dstport = PORT1; Port = PORT2; }
            else { Dstport = PORT2; Port = PORT1; }
            break;
         case 'x':
            if (strlen(argv[j]) != 8) exit(usage());
            Statusarg = argv[j];
            break;
   #ifdef BX_MYSQL
         case 'X':
            Exportflag = 1;
            break;
   #endif
         case 'h':   /* fallthrough */
         default: exit(usage());
      }  /* end switch */
   }  /* end for j */
EOA:  /* end of arguments */

   /* print header & disclaimer */
   plog("Mochimo Server " GIT_VERSION " on " __DATE__ " " __TIME__);
   plog("Mochimo Mainnet Live Since June 25, 2018 15:43:45 GMT");
   plog("Mochimo Codebase v2 Released October 27, 2018");
   plog("Copyright (c) 2022 Adequate Systems, LLC. All Rights Reserved.");
   plog("See the PDF/TEXT versions of the license agreement:");
   plog("   https://mochimo.org/license.pdf");
   plog("   https://mochimo.org/license");
   plog("");

   /* perform init and start server */
   if (Running) {
      sleep(3);  /* for effect */
      phostinfo();  /* for info */
      if (Running && init() == VEOK) {
         server();
         /* shutdown sockets */
         sock_cleanup();
         /* stop services */
         stop_miner();
         stop_mirror();
         stop_found();
         stop_bcon();
         /* save dynamic peer lists */
         save_ipl(Recentipfname, Rplist, RPLISTLEN);
         save_ipl(Epinkipfname, Epinklist, EPINKLEN);
      }
   }

   /* clear any sticky */
   psticky("");

   return 0;
}  /* end main() */

/* end include guard */
#endif
