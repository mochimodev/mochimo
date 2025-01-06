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


/* define EXEC_NAME and GIT_VERSION (if not defined) */
#ifndef GIT_VERSION
   #define GIT_VERSION
#endif
#ifndef EXEC_NAME
   #define EXEC_NAME "Mochimo Server " /* "Daemon " */ GIT_VERSION
#endif

/* system support */
#ifndef _WIN32
   #include <execinfo.h>

#endif

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
#include <dirent.h>  /* UNIX directory utils */
#include <errno.h>
#include <ctype.h>

/* external support */
#include "sha256.h"
#include "extinet.h"    /* socket support */
#include "extlib.h"     /* general support */
#include "extmath.h"    /* 64-bit math support */
#include "crc32.h"      /* for mirroring */
#include "crc16.h"
#include "exttime.h"    /* for time functions */

/* Include everything that we need */
#include "wots.h"
#include "tx.h"
#include "trigg.h"
#include "tfile.h"
#include "sync.h"
#include "peer.h"
#include "peach.h"
#include "network.h"
#include "ledger.h"
#include "global.h"     /* System wide globals  */
#include "error.h"
#include "bup.h"
#include "bcon.h"

char *Opt_cplistfile = "coreip.lst";
char *Opt_rplistfile = "recent.lst";
char *Opt_eplistfile = "epink.lst";

#ifdef _WIN32
#include <windows.h>
#include <wincrypt.h>
#else
#include <unistd.h>
#include <fcntl.h>
#endif

/**
 * @private
 * Fills a buffer with random data from urandom or CryptGenRandom (WIN32).
 * Obtain a random unsigned value from urandom or CryptGenRandom (WIN32).
 * @returns Random unsigned value
 */
unsigned urandom(void *buf, size_t bufsz) {
   unsigned seed = 123456789;

   /* use local seed if no buffer */
   if (!buf || !bufsz) {
      bufsz = sizeof(seed);
      buf = &seed;
   }

#ifdef _WIN32
   HCRYPTPROV prov = 0;
   if (CryptAcquireContext(&prov, NULL, NULL, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT | CRYPT_SILENT)) {
      CryptGenRandom(prov, (DWORD) bufsz, (BYTE *) buf);
      CryptReleaseContext(prov, 0);
   }
#else
   int fd = open("/dev/urandom", O_RDONLY);
   if (fd != -1) {
      read(fd, buf, bufsz);
      close(fd);
   }
#endif

   /* return unsigned value from buffer where available */
   if (buf && bufsz >= sizeof(unsigned)) {
      return *((unsigned *) buf);
   }

   return seed;
}

/* END RANDOM SEED FUNCTION */

int check_directory(char *dirname)
{
   char fname[FILENAME_MAX];

   mkdir_p(dirname);
   snprintf(fname, FILENAME_MAX, "%s/chkfile", dirname);
   if (ftouch(fname) == VEOK) return remove(fname);
   perrno("Permission failure, %s", dirname);
   return VERROR;
}

int clear_directory(char *dname)
{
   DIR *dp;
   struct dirent *ep;
   char fname[FILENAME_MAX];

   dp = opendir(dname);
   if (dp == NULL) {
      perrno("failed to open dir %s...", dname);
      return VERROR;
   }
   while ((ep = readdir(dp))) {
      snprintf(fname, FILENAME_MAX, "%s/%s", dname, ep->d_name);
      remove(fname); /* ignores non-empty directories */
   }
   closedir(dp);

   /* success */
   return VEOK;
}

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
   printf("Status:\n\n");
   printf("   Aeon:            %u\n", Eon);
   printf("   Generation:      %u\n", Ngen);
   printf("   Online:          %u\n", Nonline);
   printf("   Raw TX in:       %u\n", Nlogins);
   printf("   Bad peers:       %u\n", Nbadlogs);
   printf("   No space:        %u\n", Nspace);
   printf("   Client timeouts: %u\n", Ntimeouts);
   printf("   Server errors:   %u\n", perrcount());
   printf("   TX recvd:        %u\n", Nrec);
   printf("   TX dups:         %u\n", Ndups);
   printf("   txq1 count:      %u\n", Txcount);
   printf("   Balances sent:   %u\n", Nbalance);
   printf("   Sends blocked:   %u\n", Nsenderrs);
   printf("   Blocks updated:  %u\n\n", Nupdated);

   printf("Current block: 0x%s\n", bnum2hex(Cblocknum, NULL));
   printf("Weight:        0x...%s\n", bnum2hex(Weight, NULL));
   printf("Difficulty:    %d\n", Difficulty);

   return 0;
} /* end print_stats() */

/* short stat display */
void betabait(void)
{
   printf(     "Status:\n\n"
               "   Aeon:          %u\n"
               "   Generation:    %u\n"
               "   Online:        %u\n"
               "   TX recvd:        %u\n"
               "   Balances sent:   %u\n"
               "   Blocks updated:  %u\n"
               "\n",

                Eon, Ngen,
                Nonline,  Nrec, Nbalance, Nupdated
   );
   printf("Current block: 0x%s\n"
          "Difficulty:    %d\n\n", bnum2hex(Cblocknum, NULL), Difficulty);
} /* end betabait() */


void monitor(void)
{
   static word8 runmode = 0;   /* 1 for single stepping */
   static char buff[81] = { 0 };

   /*
    * Print banner if not single stepping.
    */
   if (runmode == 0) {
      printf("\n\nMochimo System Monitor " GIT_VERSION "\n? for help\n\n");
   }

   show("monitor");
   /*
    * Command loop.
    */
   for(;;) {
      printf("mochimo> ");
      tgets(buff, 80);

      /* process input */
      if (strcmp(buff, "st") == 0) print_stats();  /* stats command */
      else if (strcmp(buff, "si") == 0) {    /* single step command */
         runmode = 1 - runmode;
         printf("Single step is %s\n", runmode ? "on." : "off.");
      } else if (strcmp(buff, "ll") == 0) {  /* print level command */
         printf("Note: affects the level of logs printed to screen.\n");
         printf("Log level (0-%d): ", PLOG_DEBUG);
         /* additional input required */
         if (*tgets(buff, 80)) setploglevel(atoi(buff));
      } else if (strcmp(buff, "r") == 0) {    /* restart command */
         printf("Confirm restart (Y/n)? ");
         /* additional input required */
         tgets(buff, 80);
         if (strcmp(buff, "Y") == 0) restart("monitor");
      } else if (strcmp(buff, "q") == 0) {   /* signal server to exit */
         Monitor = runmode;
         Running = 0;
         return;
      } else if (strcmp(buff, "p") == 0) {   /* print peers command */
         printf("Recent peers:\n");
         print_ipl(Rplist, RPLISTLEN);
         printf("Pinklisted:\n");
         print_ipl(Epinklist, EPINKLEN);
         continue;
      } else if (*buff == '\0') {   /* ENTER to continue server */
         Monitor = runmode;
         printf("In server() loop...\n");
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
             "si      toggle single step mode\n"
             "st      display system status\n"
             "?       this message\n\n" );

   }  /* end command loop */
}  /* end monitor() */

int update_file(const char *dstfile, const char *srcfile, int force)
{
   if (!fexists((char *) dstfile) || force) {
      pdebug("updating %s from %s", dstfile, srcfile);
      return fcopy((char *) srcfile, (char *) dstfile);
   }
   return VEOK;
}

/**
 * Initialize the server/client from any state
 * after executing the gomochi script. */
int init(void)
{
   /* static word8 FortyEight[8] = { 48, }; */
   char dir[FILENAME_MAX], fname[FILENAME_MAX];
   char bcfname[FILENAME_MAX], copyfile[FILENAME_MAX];
   char weighthex[65], bnumhex[17];
   word32 qlen, quorum[MAXQUORUM];
   word8 nethash[HASHLEN], peerhash[HASHLEN];
   word8 netweight[32], netbnum[8]; //, bnum[8];
   FILE *fp;
   BTRAILER bt;
   word32 hdrlen;
   NODE node;  /* holds peer tx.cblock and tx.cblockhash */
   int result, status, attempts, count;

   /* init */
   show("init");
   status = VEOK;
   attempts = 0;
   Ininit = 1;

   plog("Init core...");
   /* prepare mochimo filesystem structure */
   if (check_directory(Bcdir) || check_directory(Spdir)) return VERROR;
   if (ftouch("mq.lck")) return VERROR;

   /* (ALWAYS) update core peer list and mining address file */
   path_join(copyfile, "..", Opt_cplistfile);
   if (update_file(Opt_cplistfile, copyfile, 1) != VEOK) {
      perrno("failed to restore %s", Opt_cplistfile);
      return VERROR;
   }
   path_join(copyfile, "..", "maddr.dat");
   if (update_file("maddr.dat", copyfile, 1) != VEOK) {
      pwarn("using \"maddr.mat\" (NOT YOUR MINING ADDRESS)");
      path_join(copyfile, "..", "maddr.mat");
      if (update_file("maddr.dat", copyfile, 0) != VEOK) {
         perrno("failed to restore maddr.dat");
         return VERROR;
      }
   }
   /* (IF NOT EXISTS) update blockchain files */
   path_join(fname, Bcdir, "b0000000000000000.bc");
   path_join(copyfile, "..", "genblock.bc");
   if (update_file(fname, copyfile, 0) != VEOK) {
      perrno("failed to restore %s", fname);
      return VERROR;
   }
   path_join(copyfile, "..", "tfile.dat");
   if (update_file("tfile.dat", copyfile, 0) != VEOK) {
      pwarn("deriving Tfile from genesis block");
      if (read_trailer(&bt, fname) != VEOK || \
            append_tfile(&bt, 1, "tfile.dat") != VEOK) {
         perrno("append_tfile() FAILURE; %s", fname);
         return VERROR;
      }
   }

   plog("Init chain...");
   /* reset internal chain data based on Tfile */
   if (reset_chain() != VEOK) {
      perrno("reset_chain() FAILURE");
      memset(Cblocknum, 0, 8);  /* flag resync */
   } else if (!iszero(Cblocknum, 8)) {
      pdebug("intialized chain... 0x%s 0x%s",
         bnum2hex(Cblocknum, bnumhex), weight2hex(Weight, weighthex));
   }

   plog("Init peers...");
   /* initialize peer lists */
   count = read_ipl(Opt_eplistfile, Epinklist, EPINKLEN, &Epinkidx);
   if (count > 0) plog(" - added %" P32u " pinklisted peers", count);
   count = read_ipl(Opt_cplistfile, Rplist, RPLISTLEN, &Rplistidx);
   count += read_ipl(Opt_rplistfile, Rplist, RPLISTLEN, &Rplistidx);
   if (count > 0) plog(" - added %" P32u " recent peers", count);

   /* scan entire network of peers */
   while (Running) {
      /* ensure recent peers list is shuffled */
      shuffle32(Rplist, RPLISTLEN);
      /* scan network for quorum and highest hash/weight/bnum */
      plog("Init network...");
      qlen = scan_network(quorum, MAXQUORUM, nethash, netweight, netbnum);
      plog(" - %d/%d 0x%s 0x...%s", qlen, MAXQUORUM,
         bnum2hex(netbnum, bnumhex), weight2hex(netweight, weighthex));
      if (qlen == 0) break; /* all alone... */
      else if (qlen < Quorum) {  /* insufficient quorum */
         /* pinklist peers to avoid infinite loop of network spam */
         while (*quorum) {
            /* use epinklist for purge after init */
            epinklist(*quorum);
            remove32(*quorum, Rplist, RPLISTLEN, &Rplistidx);
            remove32(*quorum, quorum, MAXQUORUM, &qlen);
         }
         /* report, wait, and re-scan... */
         plog("Insufficient quorum, try again in 30 seconds...");
         millisleep(5000);
         if (!Running) return VERROR;
         continue;
      } else shuffle32(quorum, qlen);
      /* check network is v3.0 compatible */
      if (cmp64(netbnum, CL64_32(V30TRIGGER)) < 0) {
         perr("discovered network is not v3.0 compatible");
         plog("check network port...");
         break;
      }
      if (!iszero(Cblocknum, 8)) {  /* we've got a chain */
         status = VEOK;  /* don't panic... EVERYTHING IS FINE! */
         result = cmp256(Weight, netweight);  /* compare network weight */
         if (result < 0) {
            pdebug("network weight compares higher");
            printf("\n");
            printf(" ┌────────────────────────────────────┐\n");
            printf(" │ an overwhelming sense of confusion │\n");
            printf(" └────────────────────────────────────┘\n\n");
         } else if (result > 0) {
            pdebug("network weight compares lower");
            printf("\n");
            printf(" ┌────────────────────────────────────┐\n");
            printf(" │   an overwhelming sense of power   │\n");
            printf(" └────────────────────────────────────┘\n\n");
            break;  /* we're heavier, finish */
         } else if (memcmp(Cblockhash, nethash, HASHLEN) == 0) {
            pdebug("network weight and hash compares equal");
            printf("\n");
            printf(" ┌────────────────────────────────────┐\n");
            printf(" │ an overwhelming sense of belonging │\n");
            printf(" └────────────────────────────────────┘\n\n");
            break;  /* we're in sync, finish */
         }
         /* have we fallen behind or split from the chain? */
         while (Running && *quorum) {  /* use quorum to check... */
            if (status == VEOK) {  /* chain status not yet known... */
               plog("Checking blockchain alignment...");
               if (get_hash(&node, *quorum, Cblocknum, peerhash) == VEOK) {
                  if (memcmp(Cblockhash, peerhash, HASHLEN)) {
                     status = VEBAD; /* 2319! Foreign entity (block) */
                  } else status = VERROR;  /* we're just behind */
                  continue;  /* restart loop with new status */
               }
            } else if (status == VERROR) {  /* chain is fallen... */
               plog("Blockchain is aligned, catchup...");
               /* check "catchup" doesn't cross v3 trigger */
               if (cmp64(Cblocknum, CL64_32(V30TRIGGER)) < 0) {
                  perr("catchup() cannot cross v3.0 trigger...");
                  break;
               }
               catchup(quorum, qlen);  /* try to catchup with blockchain */
               break;
            } else if (status == VEBAD) {  /* chain is split... */
               plog("2319!!! CHAIN FORK DETECTED...");
               /* attempt chain recovery... DISABLED FOR NOW
               put64(bnum, cmp64(Cblocknum, netbnum) > 0 ? netbnum : Cblocknum);
               if (sub64(bnum, FortyEight, bnum)) break;
               count = readtf(&bt, get32(bnum), get32(FortyEight));
               if (count % sizeof(BTRAILER)){
                  perr("error reading tfile, count= %s", count);
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
            remove32(*quorum, quorum, MAXQUORUM, &qlen);
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

   /* check state of v3.0 reboot */
   if (cmp64(Cblocknum, CL64_32(V30TRIGGER)) < 0) {
      /* legacy neogenesis block is reconstructed by extracting the
      * ledger data with le_extract() (handles both block types)
      * and rebuilding the block with neogen().
      */

      /* verify reboot files... */
      if (cwd(dir, FILENAME_MAX) == NULL) {
         perrno("cwd() FAILURE");
         return VERROR;
      };

      /* ... v3.0 Tfile */
      path_join(copyfile, dir, "v3", "tfile.dat");
      plog("Waiting for %s...", copyfile);
      while (update_file("tfile.dat", copyfile, 1)) {
         if (errno != ENOENT) {
            perrno("fexists() FAILURE");
            return VERROR;
         }
         while (Running && !fexists(copyfile)) sleep(1);
         if (!Running) return VERROR;
      }
      /* ... v3.0 blockchain file (neogen) */
      bnum2fname(CL64_32(V30TRIGGER), bcfname);
      path_join(copyfile, dir, "v3", bcfname);
      path_join(fname, Bcdir, bcfname);
      plog("Waiting for %s...", copyfile);
      while (update_file(fname, copyfile, 1)) {
         if (errno != ENOENT) {
            perrno("fexists() FAILURE");
            return VERROR;
         }
         while (Running && !fexists(copyfile)) sleep(1);
         if (!Running) return VERROR;
      }

      /* read file to check conversion */
      fp = fopen(fname, "rb");
      if (fp == NULL) {
         perrno("fopen() FAILURE");
         return VERROR;
      }
      if (fread(&hdrlen, 4, 1, fp) != 1) {
         if (!ferror(fp)) set_errno(EMCM_EOF);
         perrno("fread() FAILURE");
         fclose(fp);
         return VERROR;
      }
      fclose(fp);

      /* perform ledger extraction (compatible with legacy blocks) */
      if (le_extract(fname, "ledger.dat") != VEOK) {
         perrno("le_extract(v3) FAILURE");
         return VERROR;
      }

      /* check if ledger conversion is required */
      if (hdrlen != sizeof(NGHEADER)) {
         /* read and modify trailer data for neogen() as "previous" data...
         * NOTE: Why? "previous" data MAY NOT be available for fresh start.
         */
         if (read_trailer(&bt, fname) != VEOK) {
            perrno("read_trailer(v3) FAILURE");
            return VERROR;
         }
         /* ... block hash needs to be replaced with previous block hash */
         memcpy(bt.bhash, bt.phash, HASHLEN);
         /* ... block number needs to be replaced with previous block number */
         sub64(bt.bnum, ONE64, bt.bnum);
         /* ... reamining data is either duplicate or irrelevant */

         /* reconstruct v3 neogenesis block */
         if (neogen(&bt, "ledger.dat", "ngblock.dat") != VEOK) {
            perrno("neogen(v3) FAILURE");
            return VERROR;
         }
         /* replace old neogenesis block */
         remove(fname);
         if (rename("ngblock.dat", fname) != 0) {
            perrno("rename(v3) FAILURE");
            return VERROR;
         }
      }  /* end v3.0 ledger conversion */

      /* prep the Tfile for v3 transition */
      if (trim_tfile("tfile.dat", CL64_32(V30TRIGGER - 1)) != VEOK) {
         perrno("trim_tfile(v3) FAILURE");
         return VERROR;
      }
      /* the new neogenesis block needs to be re-written to the Tfile */
      if (read_trailer(&bt, fname) != VEOK) {
         perrno("read_trailer(v3) FAILURE");
         return VERROR;
      }
      if (append_tfile(&bt, 1, "tfile.dat") != VEOK) {
         perrno("append_tfile(v3) FAILURE");
         return VERROR;
      }

      /* reset internal chain data */
      if (reset_chain() != VEOK) {
         perrno("V3REBOOT reset_chain() FAILURE");
         return VERROR;
      }

      plog("Neogenesis reboot successful: %s", fname);
   }  /* end v3.0 reboot */

   txclean("txclean.dat", NULL);
   purge_epoch();
   Ininit = 0;

   return Running ? VEOK : VERROR;
}  /* end init() */

int start_bcon(void)
{
   put64(Bcbnum, Cblocknum);  /* save current block number */
   Bcon_pid = fork();
   if (Bcon_pid == -1) {
      perr("Cannot fork() for b_con()");
      Bcon_pid = 0;
   } else if (Bcon_pid == 0) {
      /* in child */
      if (b_con("cblock.dat") != VEOK) {
         perrno("b_con() FAILURE");
         exit(1);  /* child exits */
      }
      exit(0);  /* child exits */
   }
   return Bcon_pid;
}

/**
 * The Mochimo Server/Client!
 *
 * Uses globals from data.c
 */
int server(int reuse_addr)
{
   /* real time of current server loop - set by server() */
   static word8 Lblock[8];
   static time_t Ltime;
   static time_t Stime;    /* status display update time */
   static time_t nsd_time;  /* event timers */
   static time_t bctime, mtime, mqtime, sftime, vtime;
   static time_t ipltime;
   static SOCKET lsd, nsd;
   static NODE *np, node;
   static struct sockaddr_in addr;
   static int status;   /* child return status */
   static pid_t pid;    /* child pid */
   static int lfd;      /* for lock() */
   static word16 opcode;
   char fname[FILENAME_MAX];

   /* passive mining stuff */
   BTRAILER bt;
   FILE *fp;

   /* Initialise event timers */
   Ltime = time(NULL);      /* real time GMT in seconds */
   Stime = Ltime + 10;      /* status display time */
   bctime = Ltime + 30;     /* block constructor time */
   mqtime = Ltime + 5;      /* mirror() time */
   mtime = Ltime + 5;       /* miner time */
   Utime = Ltime;           /* for watchdog timer */
   Watchdog = get_bridge(NULL) + (rand16() % 600);
   ipltime = Ltime + (rand16() % 300) + 10;  /* ip list fetch time */
   sftime = Ltime + (rand16() % 300) + 300;  /* send_found() time */
   vtime = Ltime + 4;  /* Verisimility restart check time */

   lsd = socket(AF_INET, SOCK_STREAM, 0);
   if (lsd == INVALID_SOCKET) restart("Cannot open listening socket.");
   memset(&addr, 0, sizeof(addr));    /* clear address structure   */
   addr.sin_port = htons(Port);
   addr.sin_addr.s_addr = INADDR_ANY;
   addr.sin_family = AF_INET;

   /* reuse_addr is passed in from main() as runtime option */
   setsockopt(lsd, SOL_SOCKET, SO_REUSEADDR, &reuse_addr, sizeof(int));

   show("bind");
   for(;;) {
      if(!Running) { sock_close(lsd); return 0; }
      if(bind(lsd, (struct sockaddr *) &addr, sizeof(addr)) == 0) break;
      /* timeout after ~5 minutes */
      if (difftime(time(NULL), Ltime) > 300) {
         perr("Timeout binding port %d", Port);
         sock_close(lsd);
         return 0;
      }
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

   remove("vstart.lck");  /* signal Verisimility that we are up. */

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
            if(Blockfound == 0) perr("line %d", __LINE__);
            else {
               /* exit services */
               stop_bcon();
               stop_found();
               /* update recv'd block */
               if(b_update("rblock.dat", 0) == VEOK) {
                  send_found();  /* start send_found() child */
                  addrecent(np->ip);   /* v.28 */
                  Stime = Ltime + 20;  /* hold status display */
               }
               /* check txclean.dat contains transactions */
               if (fexistsnz("txclean.dat")) start_bcon();
               Blockfound = 0;
            }
         }  /* end if OP_FOUND child */
         else if(opcode == OP_GET_BLOCK || opcode == OP_GET_TFILE) {
            /* only add those that "optin" with a successful op */
            if (status == 0 && np->tx.version[1] & C_OPTIN) {
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
                  pdebug("opcode = %d", opcode);
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
                        pdebug("bad opcode: %d", opcode);
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

      /* Check mined (push) blocks */
      if(Blockfound == 0 && fexists("mblock.dat")) {
         Blockfound = 1;
         if(cmp64(Cblocknum, Bcbnum) == 0) {
            /* exit services */
            stop_bcon();
            stop_found();
            /* We found a pushed block! Update... */
            if (b_update("mblock.dat", 1) == VEOK) {
               send_found();  /* start send_found() child */
               Stime = Ltime + 20;  /* hold status display */
            }
            /* check txclean.dat contains transactions */
            if (fexistsnz("txclean.dat")) start_bcon();
         }
         remove("mblock.dat");
         Blockfound = 0;
      }

      /* generate pseudo-block in "times of trouble", else check bcon...
       * NOTE: after V30TRIGGER TIMES_OF_TROUBLE() is no longer relevant
       */
      if(Ltime >= (Time0 + get_bridge(NULL))) {
         if (pseudo("pblock.dat") != VEOK) {
            perrno("pseudo() FAILURE");
            restart("Failed to make pseudo-block");
         } else {
            /* exit services */
            stop_bcon();
            stop_found();
            /* update pseudoblock */
            if (b_update("pblock.dat", 2) != VEOK) {
               restart("Failed to update pseudo-block");
            } else Stime = Ltime + 20;  /* hold status display */
            /* check txclean.dat contains transactions */
            if (fexistsnz("txclean.dat")) start_bcon();
         }
      } else {
         /* block constructor time trigger checks */
         if (bctime < (Time0 + BCONFREQ)) bctime = Time0 + BCONFREQ;
         if (Txcount >= TXQUEBIG) bctime = Ltime;

         if (Blockfound == 0 && Ltime >= bctime) {
            /* check conditions for Transaction Bot processor */
            if (tx_bot_is_active() && cmp64(Lblock, Cblocknum) != 0) {
               if (tx_bot_process() == VEOK) {
                  /* update block number and Txbot data */
                  put64(Lblock, Cblocknum);
               }
            }
            /* check conditions for Transaction Queue processor */
            if (Bcon_pid == 0 && Txcount > 0) {
               pdebug("spawning bcon with %d more transactions", Txcount);
               /* append txq1.dat to txclean.dat */
               system("cat txq1.dat >>txclean.dat 2>/dev/null");
               remove("txq1.dat");
               Txcount = 0;  /* txq1.dat is empty now */
               start_bcon();  /* start child */
               bctime = Ltime + BCONFREQ;
            }
         }
      }

      /* Collect bcon status when she is 'done'.  pid == 0 means she
       * is still busy.
       */
      if(Bcon_pid > 0) {
         pid = waitpid(Bcon_pid, &status, WNOHANG);
         if(pid > 0) {
            Bcon_pid = 0;  /* pid not zero means she is done. */
            /* check cblock and prepare passive mining */
            if (fexistsnz("cblock.dat")) {
               /* make isolated copy, read trailer and init */
               if (read_trailer(&bt, "cblock.dat") != VEOK) {
                  perrno("read_trailer() FAILURE");
               } else peach_init(&bt);
            }
         }
      } else if (mtime != Ltime && fexistsnz("cblock.dat")) {
         mtime = Ltime;
         /* perform passive mining once every second */
         if (peach_solve(&bt, bt.difficulty[0], bt.nonce) == VEOK) {
            /* record solve time and hash block trailer */
            put32(bt.stime, (word32) time(NULL));
            sha256(&bt, sizeof(BTRAILER) - HASHLEN, bt.bhash);
            /* rewrite block trailer to disk */
            fp = fopen("cblock.dat", "r+b");
            if (fp == NULL) {
               perrno("fopen() FAILURE");
            } else if (fseek64(fp, -(sizeof(BTRAILER)), SEEK_END) != 0) {
               perrno("fseek64() FAILURE");
               fclose(fp);
            } else if (fwrite(&bt, sizeof(BTRAILER), 1, fp) != 1) {
               perrno("fwrite() FAILURE");
               fclose(fp);
            } else {
               fclose(fp);
               /* check for and trigger mined block update */
               if (!fexists("mblock.dat")) {
                  rename("cblock.dat", "mblock.dat");
                  ftouch("cblock.lck");
               }
            }
         }
      }

      /* Start mirror()? */
      if(Ltime >= mqtime && Mqcount > 0 && Mqpid == 0) {
         /* get exclusive access to txq1.dat */
         lfd = lock("mq.lck", 10);
         if(lfd != -1) {
            remove("mirror.dat");
            rename("mq.dat", "mirror.dat");
            Mqcount = 0;
            unlock(lfd);
            Mqpid = mirror();  /* start child */
            if (Mqpid == 0) perrno("mirror() FORK FAILURE");
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

/* print (and more importantly, log) server host information */
void phostinfo(void)
{
   char hostname[64];
   char addrname[18];

   /* get local machine name and IP address */
   gethostname(hostname, sizeof(hostname));
   gethostip(addrname, sizeof(addrname));
   /* print (and log) host information */
   plog("Network Host Information...");
   plog("  Machine name: %s", *hostname ? hostname : "unknown");
   plog("  IPv4 address: %s", *addrname ? addrname : "0.0.0.0");
   printf("\n");
}

/**
 * Segmentation fault handler.
 * @note compile with "-g -rdynamic" for readable backtrace
 * @note Not compatible with Windows at this stage
*/
void segfault(int sig)
{
#ifndef _WIN32
   void *array[10];
   size_t size;

   /* get void*'s for all entries on the stack */
   size = backtrace(array, 10);
#else
   fprintf(stderr, "*no backtrace on this system*\n");
#endif

   /* print out all the frames to stderr */
   fprintf(stderr, "Error: signal %d:\n", sig);

#ifndef _WIN32
   backtrace_symbols_fd(array, size, STDERR_FILENO);
#endif

   exit(1);
}

/*
 * Signal handlers
 *
 * Enter monitor on ctrl-C
 */
void ctrlc(int sig)
{
   printf("\n");
   pdebug("Got signal %i", sig);
   signal(SIGINT, ctrlc);
   if (Ininit) Running = 0;
   else Monitor = 1;
}

/*
 * Clear run flag, Running on SIGTERM
 */
void sigterm(int sig)
{
   printf("\n");
   pdebug("Got signal %i", sig);
   signal(SIGTERM, sigterm);
   sock_cleanup();
   Running = 0;
}

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
      "   -c         disable read core ip list\n"
      "   -d         disable pink lists\n"
      "   -pN        set port to N\n"
      "   -D         Daemon ignore ctrl-c and no term output\n"
      "   -sN        sleep N usec. on each loop if not busy\n"
      "   -xxxxxxx   replace xxxxxxx with state\n"
      "   -S         Safe mode\n"
      "   -F         Filter private IP's\n"  /* v.28 */
      "   -P         Allow pushed mblocks\n"
      "   -Mn        set transaction fee to n\n"
      "   -Sanctuary=N,Lastday\n"
      "   -Tn        set Trustblock to n for tfval() speedup\n"
      "\n"
      "\n\nOPTIONS (advanced):"
      "\n   --reuse-addr"
      "\n       enable listening server socket option SO_REUSEADDR"
#ifdef BX_MYSQL
      "\n   -X         Export to MySQL database on block update"
#endif
      "\n\n"
   );

   return VEOK;
}  /* end usage() */

/*
 * Initialise data and call the server.
 */
int main(int argc, char **argv)
{
   char *argp;          /* argument pointer (for argv) */
/* unsigned long argu;     argument unsigned value */

   unsigned seeds[8];   /* random seed values */
   int reuse_addr;
   char *cp;
   int j;

   {
      /* ensure little endian check -- executed in isolation */
      STATIC_ASSERT(sizeof(word32) == 4, word32_size);
      if (get16((word8[2]) { 0x34, 0x12 }) != 0x1234u) {
         perr("incompatible endian type");
         return EXIT_FAILURE;
      }
   }

   /* logging setup */
   setploglevel(PLOG_DEBUG);
   /* Ignore all signals. */
   for (j = 0; j <= 23; j++) signal(j, SIG_IGN);
   /*signal(SIGINT, ctrlc);*/ /* then install ctrl-C handler */
   signal(SIGINT, sigterm);   /* ...and ctrl-C termination */
   signal(SIGTERM, sigterm);  /* ...and software termination */
   signal(SIGSEGV, segfault); /* segmentation fault handler */
#ifndef _WIN32
   signal(SIGCHLD, SIG_DFL);  /* so waitpid() works */
#endif
   /* seed random generators with urandom (or equivalent) */
   srand16fast(urandom(seeds, sizeof(seeds)));
   srand16(seeds[1], seeds[2], seeds[3]);
   srand32(*((unsigned long long *) &seeds[4]));
   /* enable socket support */
   sock_startup();

   /* local init */
   reuse_addr = 0;
   Cbits |= C_OPTIN;  /* default to opt-in for Node */

   /* Parse command line arguments. */
   pdebug("... skipping 0th argument (program name): %s", argv[0]);
   for (j = 1; Running && j < argc; j++) {
      pdebug("... parsing argument: %s", argv[j]);
      /* ADVANCED OPTIONS */
      if (argv[j][0] == '-') {
         if (argument(argv[j], NULL, "--reuse-addr")) {
            /* set reuse_addr option and continue */
            reuse_addr = 1;
            continue;
         }
         if (argument(argv[j], NULL, "--tx-bot")) {
            argp = argvalue(&j, argc, argv);
            if (argp != NULL) pdebug("    argument value: %s", argp);
            /* set tx-bot option and continue */
            if (tx_bot_activate(argp) != VEOK) {
               perrno("tx_bot_activate() FAILURE");
               return EXIT_FAILURE;
            }
            continue;
         }
      } else return usage();
      /* legacy argument parsing */
      switch(argv[j][1]) {
         case '-':  /* advanced commands */
            cp = &argv[j][2];
            if (*cp == '\0') goto EOA;  /* -- end of args */
            else if (strcmp("help", cp) == 0) exit(usage());
            else if (strcmp("veronica", cp) == 0) exit(veronica());
            else perr("Unknown argument, %s", argv[j]);
            break;
         case 'c':  /* set core ip list */
            if (!argv[j][2]) {
               perr("missing coreip list file");
               exit(usage());
            }
            Opt_cplistfile = &argv[j][2];  /* master network */
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
            setploglevel(atoi(&argv[j][2]));
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

   /* print (and log) copyright and version information */
   plog(EXEC_NAME ", built " __DATE__ " " __TIME__);
   plog("Copyright (c) 2024 Adequate Systems, LLC.  All Rights Reserved.");
   plog("See the License Agreement at the links below:");
   plog("   https://mochimo.org/license.pdf (PDF version)");
   plog("   https://mochimo.org/license (TEXT version)");
   printf("\n");
   phostinfo();
   sleep(3);

   /* perform init and start server */
   if (Running) {
      if (Running && init() == VEOK) {
         server(reuse_addr);
         /* shutdown sockets */
         sock_cleanup();
         /* stop services */
         stop_mirror();
         stop_found();
         stop_bcon();
         /* save dynamic peer lists */
         save_ipl(Opt_rplistfile, Rplist, RPLISTLEN);
         save_ipl(Opt_eplistfile, Epinklist, EPINKLEN);
      }
   }

   return EXIT_SUCCESS;
}  /* end main() */

/* end include guard */
#endif
