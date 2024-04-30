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
         printf("Trusted peers:\n");
         print_ipl(Tplist, TPLISTLEN);
         printf("Recent peers:\n");
         print_ipl(Rplist, RPLISTLEN);
         printf("Pinklisted:\n");
         print_ipl(Epinklist, RPLISTLEN);
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

/**
 * Initialize the server/client from any state
 * after executing the gomochi script. */
int init(void)
{
   /* static word8 FortyEight[8] = { 48, }; */
   char fname[FILENAME_MAX], weighthex[65], bnumhex[17];
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

   /* prepare mochimo filesystem structure */
   if (check_directory(Bcdir) || check_directory(Spdir)) return VERROR;
   if (ftouch("mq.lck")) return VERROR;

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
            perr("Failed to copy mining address");
            return VERROR;
         } else pwarn("using maddr.MAT (the founder's mining address)");
      }
   }
   /* restore core chain files if any do not exist */
   snprintf(fname, FILENAME_MAX, "%s/b0000000000000000.bc", Bcdir);
   if (!fexists("tfile.dat") || !fexists(fname)) {
      pdebug("Core chain files compromised, attempting restoration...");
      if (fcopy("../genblock.bc", fname) != VEOK) {
         perr("Failed to restore %s from ../genblock.bc", fname);
         return VERROR;
      } else if (fcopy("../tfile.dat", "tfile.dat") != VEOK) {
         perr("Failed to restore tfile.dat from ../tfile.dat");
         return VERROR;
      }
   }
   /* open ledger or extract from genesis block */
   if (!fexists("ledger.dat") || le_open("ledger.dat", "rb") != VEOK) {
      pdebug("Extracting ledger from ../genblock.bc ...");
      if (le_extract("../genblock.bc", "ledger.dat") != VEOK) {
         perr("Failed to extract ledger from ../genblock.bc");
         return VERROR;
      } else if (le_open("ledger.dat", "rb") != VEOK) { /* try again */
         perr("Failed to open ledger.dat");
         return VERROR;
      }
   }

   plog("Init chain...");
   /* Find the last block in bc/ and reset Time0, and Difficulty */
   if (reset_chain() != VEOK) {
      perr("reset_chain() failed");
      return VERROR;
   }
   /* validate our own tfile.dat to compute Weight */
   if (tf_val("tfile.dat", highblock, Weight, 1)) {
      perr("bad tfile.dat -- resync");
      memset(Cblocknum, 0, 8);  /* flag resync */
   } else if (cmp64(Cblocknum, highblock)) {
      pdebug("%d %d", get32(Cblocknum), get32(highblock));
      pdebug("0x...%s", weight2hex(Weight, weighthex));
      perr("tfile mismatch -- resync");
      memset(Cblocknum, 0, 8);  /* flag resync */
   } else if (!iszero(Cblocknum, 8)) {
      plog(" - 0x%s 0x%s", bnum2hex(Cblocknum, bnumhex),
         weight2hex(Weight, weighthex));
   }

   /* scan entire network of peers */
   while (Running) {
      /* reset peers, download start peers, initialize peer lists */
      plog("Init peers...");
      count = read_ipl(Epinkipfname, Epinklist, EPINKLEN, &Epinkidx);
      if (count > 0) plog(" - added %" P32u " pinklisted peers", count);
      count = read_ipl(Trustedipfname, Tplist, TPLISTLEN, &Tplistidx);
      if (count > 0) plog(" - added %" P32u " trusted peers", count);
      count = read_ipl(Trustedipfname, Rplist, RPLISTLEN, &Rplistidx);
      count += read_ipl(Coreipfname, Rplist, RPLISTLEN, &Rplistidx);
      if (count > 0) plog(" - added %" P32u " recent peers", count);
      /* ensure recent peers list is shuffled */
      shuffle32(Rplist, RPLISTLEN);
      /* scan network for quorum and highest hash/weight/bnum */
      plog("Init network...");
      qlen = scan_network(quorum, MAXQUORUM, nethash, netweight, netbnum);
      plog(" - %d/%d 0x%s 0x...%s", qlen, MAXQUORUM,
         bnum2hex(netbnum, bnumhex), weight2hex(netweight, weighthex));
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

   txclean("txclean.dat", NULL);
   Ininit = 0;

   return Running ? VEOK : VERROR;
}  /* end init() */

/**
 * The Mochimo Server/Client!
 *
 * Uses globals from data.c
 */
int server(int reuse_addr)
{
   /* real time of current server loop - set by server() */
   static time_t Ltime;
   static time_t Stime;    /* status display update time */
   static time_t nsd_time;  /* event timers */
   static time_t bctime, mqtime, sftime, vtime;
   static time_t ipltime;
   static SOCKET lsd, nsd;
   static NODE *np, node;
   static struct sockaddr_in addr;
   static int status;   /* child return status */
   static pid_t pid;    /* child pid */
   static int lfd;      /* for lock() */
   static word16 opcode;
   char fname[FILENAME_MAX];

   /* Initialise event timers */
   Ltime = time(NULL);      /* real time GMT in seconds */
   Stime = Ltime + 10;      /* status display time */
   bctime = Ltime + 30;     /* block constructor time */
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
               Blockfound = 0;
            }
         }  /* end if OP_FOUND child */
         else if(opcode == OP_GET_BLOCK || opcode == OP_GET_TFILE) {
            if (status == 0) {
               /* NODEs should use appropriate capability bit */
               if ((np->c_vpdu && ~(np->tx.version[1] & C_WALLET)) ||
                     (!np->c_vpdu && get16(np->tx.len) == 0)) {
                  /* identified as NOT a wallet, add to peers */
                  addrecent(np->ip);
               }
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
         }
         remove("mblock.dat");
         Blockfound = 0;
      }

      /* generate pseudo-block in "times of trouble", else check bcon */
      if(Ltime >= (Time0 + BRIDGE) && TIMES_OF_TROUBLE(Cblocknum)) {
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
         }
      } else {
         if (Txcount >= TXQUEBIG) bctime = Ltime;
         if (Bcon_pid == 0 && Blockfound == 0 && Ltime >= bctime &&
            (Txcount > 0 || fexistsnz("txclean.dat"))) {
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
      "   -Mn        set transaction fee to n\n"
      "   -Sanctuary=N,Lastday\n"
      "   -Tn        set Trustblock to n for tfval() speedup\n"
      "   --reuse-addr\n"
      "        enable listening server socket option SO_REUSEADDR\n"
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
   static char *cp;
   static int k, j;
   static word8 endian[] = { 0x34, 0x12 };
   int reuse_addr = 0;
   char hostname[64];
   char addrname[18];

   set_print_level(PLEVEL_LOG);

   /* sanity checks -- for undesired structure padding */
   if (sizeof(word32) != 4) {
      resign("word32 should be 4 bytes");
   /*else if (sizeof(TX) != TXBUFFLEN) {
      resign("struct size error TX != TXBUFFLEN");
   }*/ else if (sizeof(LTRAN) != (TXWOTSLEN + 1 + TXAMOUNT)) {
      resign("struct size error: LTRAN != (TXWOTSLEN + 1 + TXAMOUNT)");
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
   /* Ignore all signals. */
   for(j = 0; j <= 23; j++) {
      signal(j, SIG_IGN);
   }
   signal(SIGINT, ctrlc);     /* then install ctrl-C handler */
   signal(SIGTERM, sigterm);  /* ...and software termination */
   signal(SIGSEGV, segfault); /* segmentation fault handler */
#ifndef _WIN32
   signal(SIGCHLD, SIG_DFL);  /* so waitpid() works */
#endif

   /* improve random generators */
   srand16fast(time(NULL) ^ getpid());
   srand16(time(NULL), 0, 123456789 ^ getpid());
   /* enable socket support */
   sock_startup();
   /* lof functions */
   setplogfunctions(1);

   /* Parse command line arguments. */
   for (j = 1; Running && j < argc; j++) {
      if(argv[j][0] != '-') return usage();
      switch(argv[j][1]) {
         case '-':  /* advanced commands */
            cp = &argv[j][2];
            if (*cp == '\0') goto EOA;  /* -- end of args */
            else if (strcmp("help", cp) == 0) exit(usage());
            else if (strcmp("reuse-addr", cp) == 0) reuse_addr = 1;
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
            setploglevel((k = atoi(&argv[j][2])));
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

   /* print (and log) copyright and version information */
   plog(EXEC_NAME ", built " __DATE__ " " __TIME__);
   plog("Copyright (c) 2018-2023 Adequate Systems, LLC.  All Rights Reserved.");
   plog("See the License Agreement at the links below:");
   plog("   https://mochimo.org/license.pdf (PDF version)");
   plog("   https://mochimo.org/license (TEXT version)");
   printf("\n");
   /* get local machine name and IP address */
   gethostname(hostname, sizeof(hostname));
   gethostip(addrname, sizeof(addrname));
   /* print (and log) host information */
   plog("Network Host Information...");
   plog("  Machine name: %s", *hostname ? hostname : "unknown");
   plog("  IPv4 address: %s", *addrname ? addrname : "0.0.0.0");
   printf("\n");
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
         save_ipl(Recentipfname, Rplist, RPLISTLEN);
         save_ipl(Epinkipfname, Epinklist, EPINKLEN);
      }
   }

   return 0;
}  /* end main() */

/* end include guard */
#endif
