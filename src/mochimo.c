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

/* build sequence */
#define PATCHLEVEL 37
#define VERSIONSTR  "37"   /*   as printable string */

/* Include everything that we need */
#include "config.h"
#include "sock.h"     /* BSD sockets */
#include "mochimo.h"
#include "proto.h"

/* Include global data . . . */
#include "data.c"       /* System wide globals  */

/* Support functions  */
#include "error.c"      /* error logging etc.   */
#include "add64.c"      /* 64-bit assist        */
#include "crypto/crc16.c"
#include "crypto/crc32.c"      /* for mirroring          */
#include "rand.c"       /* fast random numbers    */

/* Server control */
#include "util.c"       /* server support */
#include "sock.c"       /* inet utilities */
#include "pink.c"       /* manage pinklist                 */
#include "connect.c"    /* make outgoing connection        */
#include "call.c"       /* callserver() and friends        */
#include "ledger.c"
#include "tag.c"        /* address tag support             */
#include "gettx.c"      /* poll and read NODE socket       */
#include "txval.c"      /* validate transactions           */
#include "mirror.c"
#include "execute.c"
#include "phost.c"      /* utility to print host info      */
#include "monitor.c"    /* system monitor/debugger prompt  */
#include "daemon.c"
#include "bupdata.c"    /* for block updates               */
#include "str2ip.c"
#include "miner.c"
#include "pval.c"       /* pseudo-blocks                   */
#include "optf.c"       /* for OP_HASH and OP_TF           */
#include "proof.c"
#include "renew.c"
#include "update.c"
#include "init.c"       /* read Coreplist[] and get_eon()  */
#include "syncup.c"     /* Resync Node on Inferior Chain   */
#include "server.c"     /* tcp server                      */


void usage(void)
{
   printf("usage: mochimo [-option...]\n"
          "         -l         open mochi.log file\n"
          "         -lFNAME    open log file FNAME\n"
          "         -e         enable error.log file\n"
          "         -tN        set Trace to N (0, 1, 2, 3)\n"
          "         -qN        set Quorum to N (default 4)\n"
          "         -vN        set virtual mode: N = 1 or 2\n"
          "         -cFNAME    read core ip list from FNAME\n"
          "         -LFNAME    read local peer ip list from FNAME\n"
          "         -c         disable read core ip list\n"
          "         -d         disable pink lists\n"
          "         -pN        set port to N\n"
          "         -D         Daemon ignore ctrl-c and no term output\n"
          "         -sN        sleep N usec. on each loop if not busy\n"
          "         -xxxxxxx   replace xxxxxxx with state\n"
          "         -f         frisky mode (promiscuous mirroring)\n"
          "         -S         Safe mode\n"
          "         -F         Filter private IP's\n"  /* v.28 */
          "         -P         Allow pushed mblocks\n"
          "         -n         Do not solve blocks\n"
          "         -Mn        set transaction fee to n\n"
          "         -Sanctuary=N,Lastday\n"
          "         -Tn        set Trustblock to n for tfval() speedup\n"
   );
#ifdef BX_MYSQL
   printf("         -X         Export to MySQL database on block update\n");
#endif
   exit(0);
}


void veronica(void)
{
   byte h[64];
   char *cp;

   cp = trigg_generate(h, 0);
   if(cp) printf("\n%s\n\n", cp);
   exit(0);
}


/* 
 * Initialise data and call the server.
 */

int main(int argc, char **argv)
{
   static int j;
   static byte endian[] = { 0x34, 0x12 };
   static char *cp;

   /* sanity checks */
   if(sizeof(word32) != 4) fatal("word32 should be 4 bytes");
   if(sizeof(TX) != TXBUFFLEN || sizeof(LTRAN) != (TXADDRLEN + 1 + TXAMOUNT)
      || sizeof(BTRAILER) != BTSIZE)
      fatal("struct size error.\nSet compiler options for byte alignment.");    
   if(get16(endian) != 0x1234)
      fatal("little-endian machine required for this build.");

   /* improve random generators w/additional entropy from maddr.dat */
   read_data(Maddr, TXADDRLEN, "maddr.dat");
   srand16(time(&Ltime) ^ get32(Maddr) ^ getpid());
   srand2(Ltime ^ get32(Maddr+4), 0, 123456789 ^ get32(Maddr+8) ^ getpid());

   Port = Dstport = PORT1;    /* default receive port */
   /*
    * Parse command line arguments.
    */
   for(j = 1; j < argc; j++) {
      if(argv[j][0] != '-') usage();
      switch(argv[j][1]) {
         case 't':  Trace = atoi(&argv[j][2]); /* set trace level  */
                    break;
         case 'q':  Quorum = atoi(&argv[j][2]); /* peers in gang[Quorum] */
                    if((unsigned) Quorum > MAXQUORUM) usage();
                    break;
         case 'p':  Port = Dstport = atoi(&argv[j][2]); /* home/dst */
                    break;
         case 'l':  if(argv[j][2]) /* open log file used by plog()   */
                       Logfp = fopen(&argv[j][2], "a");
                    else
                       Logfp = fopen(LOGFNAME, "a");
                    Cbits |= C_LOGGING;
                    break;
         case 'e':  Errorlog = 1;  /* enable "error.log" file */
                    break;
         case 'c':  Corefname = &argv[j][2];  /* master network */
                    break;
         case 'L':  Lpfname = &argv[j][2];  /* local peer network */
                    break;
         case 'd':  Disable_pink = 1;  /* disable pink lists */
                    break;
         case 'f':  Frisky = 1;
                    break;
         case 'S':  if(strncmp(argv[j], "-Sanctuary=", 11) == 0) {
                       cp = strchr(argv[j], ',');
                       if(cp == NULL) usage();
                       Sanctuary = strtoul(&argv[j][11], NULL, 0);
                       Lastday = (strtoul(cp + 1, NULL, 0) + 255) & 0xffffff00;
                       Cbits |= C_SANCTUARY;
                       printf("\nSanctuary=%u  Lastday 0x%0x...",
                              Sanctuary, Lastday);  fflush(stdout); sleep(2);
                       printf("\b\b\b accounted for.\n");  sleep(1);
                       break;
                    }
                    if(argv[j][2]) usage();
                    Safemode = 1;
                    break;
         case 'n':  Nominer = 1;
                    break;
         case 'F':  Noprivate = 1;  /* v.28 */
                    break;
         case 'P':  Allowpush = 1;  Cbits |= C_PUSH;
                    break;
         case 'x':  if(strlen(argv[j]) != 8) break;
                    Statusarg = argv[j];
                    break;
         case 'D':  Bgflag = 1;
                    break;
         case 's':  Dynasleep = atoi(&argv[j][2]);  /* usleep time */
                    break;
         case 'M':  Myfee[0] = atoi(&argv[j][2]);
                    if(Myfee[0] < Mfee[0]) Myfee[0] = Mfee[0];
                    else Cbits |= C_MFEE;
                    break;
         case 'V':  if(strcmp(&argv[j][1], "Veronica") == 0)
                       veronica();
                    usage();
         case 'v':  Dstport = PORT2;  Port = PORT1;
                    if(argv[j][2] == '2') {
                       Dstport = PORT1;  Port = PORT2;
                    }
                    break;
         case 'T':  if(argv[j][2] == '\0') {
                       Trustblock = -1;
                       break;
                    }
                    else {
                       Trustblock = atoi(&argv[j][2]);
                       break;
                    }
#ifdef BX_MYSQL
         case 'X':  Exportflag = 1;
                    break;
#endif
         default:   usage();
      }  /* end switch */
   }  /* end for j */

   if(Trace == 3) { Trace = 0; Betabait = 1; }
   if(Bgflag) setpgrp();  /* detach */

   /*
    * Redirect signals.
    */
   fix_signals();
   signal(SIGCHLD, SIG_DFL);  /* so waitpid() works */

   init();  /* fetch initial block chain */
   if(!Bgflag) printf("\n");

   plog("\nMochimo Server (Build %d)  PVERSION: %d  Built on %s %s\n"
        "Copyright (c) 2019 Adequate Systems, LLC.  All rights reserved.\n"
        "\nBooting",
        PATCHLEVEL, PVERSION, __DATE__, __TIME__);

   /* 
    * Show local host info
    */
   if(!Bgflag) phostinfo();

   server();                  /* start server */

   plog("Server exiting . . .");
   save_rplist();
   savepink();
   pause_server();
   return 0;              /* never gets here */
} /* end main() */
