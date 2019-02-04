/* mochimo.c
 *
 * Copyright (c) 2018 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * The Mochimo Project System Software
 * This file builds a server.
 *
 * Revised: 20 August 2018
*/

/* build sequence */
#define PATCHLEVEL 31
#define VERSIONSTR  "31"   /*   as printable string */

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
#include "update.c"
#include "init.c"       /* read Coreplist[] and get_eon()  */
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
          "         -R         activate Relay mode\n"
   );
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

   /* sanity checks */
   if(sizeof(word32) != 4) fatal("word32 should be 4 bytes");
   if(sizeof(TX) != TXBUFFLEN || sizeof(LTRAN) != (TXADDRLEN + 1 + TXAMOUNT)
      || sizeof(BTRAILER) != BTSIZE)
      fatal("struct size error.\nSet compiler options for byte alignment.");    
   if(get16(endian) != 0x1234)
      fatal("little-endian machine required for this build.");

   srand16(time(&Ltime));       /* seed ID token generator */
   srand2(Ltime, 0, 0);

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
                    break;
         case 'e':  Errorlog = 1;  /* enable "error.log" file */
                    break;
         case 'c':  Corefname = &argv[j][2];  /* master network */
                    break;
         case 'd':  Disable_pink = 1;  /* disable pink lists */
                    break;
         case 'f':  Frisky = 1;
                    break;
         case 'S':  Safemode = 1;
                    break;
         case 'R':  Relaymode = 1;
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
         case 'V':  if(strcmp(&argv[j][1], "Veronica") == 0)
                       veronica();
                    usage();
         case 'v':  Dstport = PORT2;  Port = PORT1;
                    if(argv[j][2] == '2') {
                       Dstport = PORT1;  Port = PORT2;
                    }
                    break;
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

   init();  /* Initialise -- does not fork() */
   printf("\n");

   plog("\nMochimo Server (Build %d)  Built on %s %s\n"
        "Copyright (c) 2018 Adequate Systems, LLC.  All rights reserved.\n"
        "\nBooting",
        PATCHLEVEL, __DATE__, __TIME__);

   /* 
    * Show local host info
    */
   phostinfo();

   server();                  /* start server */

done:
    plog("Server exiting . . .");
    save_rplist();
    savepink();
    pause_server();
    return 0;              /* never gets here */
} /* end main() */
