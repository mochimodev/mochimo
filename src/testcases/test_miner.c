/* test_miner.c   Test case for validating GPU Solves and reporting
 * on the statistical efficiency of the miner.
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * The Mochimo Project System Software
 *
 * Date: 22 January 2019
 *
 * Expect this file to be if-def'd up for various miners.
 * Currently forced to compile a CUDA miner only.
 *
 */

#include <inttypes.h>
/* build sequence */
#define PATCHLEVEL 34
#define VERSIONSTR  "34"   /*   as printable string */

/* Include everything that we need */
#include "../config.h"
#include "../sock.h"     /* BSD sockets */
#include "../mochimo.h"
#include "../proto.h"

/* Include global data . . . */
#include "../data.c"       /* System wide globals  */

/* Support functions  */
#include "../error.c"      /* error logging etc.   */
#include "../add64.c"      /* 64-bit assist        */
#include "../crypto/crc16.c"
#include "../crypto/crc32.c"      /* for mirroring          */
#include "../rand.c"       /* fast random numbers    */

/* Server control */
#include "../util.c"       /* server support */
#include "../sock.c"       /* inet utilities */
#include "../pink.c"       /* manage pinklist                 */
#include "../connect.c"    /* make outgoing connection        */
#include "../call.c"       /* callserver() and friends        */
#include "../ledger.c"
#include "../tag.c"        /* address tag support             */
#include "../gettx.c"      /* poll and read NODE socket       */
#include "../txval.c"      /* validate transactions           */
#include "../mirror.c"
#include "../execute.c"
#include "../phost.c"      /* utility to print host info      */
#include "../monitor.c"    /* system monitor/debugger prompt  */
#include "../daemon.c"
#include "../bupdata.c"    /* for block updates               */
#include "../str2ip.c"
#include "../miner.c"
#include "../pval.c"       /* pseudo-blocks                   */
#include "../optf.c"       /* for OP_HASH and OP_TF           */
#include "../proof.c"
#include "../renew.c"
#include "../update.c"
#include "../init.c"       /* read Coreplist[] and get_eon()  */
#include "../server.c"     /* tcp server                      */

/* peach algo prototypes */
int init_cuda_peach(byte difficulty, byte *prevhash, byte *blocknumber);
void cuda_peach(byte *bt, uint32_t *hps, byte *runflag);
void free_cuda_peach();


int test_miner(byte nsolve)
{
   BTRAILER bt;
   char phaiku[256];
   time_t inittime[256];
   time_t timer[256];
   time_t avginit;
   time_t avgtime;
   uint32_t hps[256] = {0};
   uint32_t avghps;
   uint32_t actualhps;
   float efficiency;
   int i, j;
   
   /* Output intro and notes */
   if(nsolve < 16 || Difficulty < 22) {
      printf(
      "*Please note... For a more accurate test, it is recommended to\n"
      "use at least 16 iterations in combination with a difficulty\n"
      "that solves at least every 30 seconds on your machine.\n");
      printf("Run  ./test_miner -h  for usage information.\n\n");
   }
   printf("Entering test_miner loop - %d iterations @ Difficulty %d\n",
          (int) nsolve, Difficulty);
   
   /* Begin miner loop */
   for(i = 0; i < nsolve && Running; i++)
   {
      if(Trace == 1)
         printf("Generating random block trailer...\n");
      /* Generate block trailer with random data */
      byte* bt_byte = (byte*)(&bt);
      for (j = 0; j < sizeof(BTRAILER); j++)
      {
         bt_byte[j] = rand16();
      }
      if(Trace == 1)
         printf("Setting block trailer difficulty...\n");
      /* ... setup block difficulty */
      bt.difficulty[0] = Difficulty;
      bt.difficulty[1] = 0;
      bt.difficulty[2] = 0;
      bt.difficulty[3] = 0;
      if(Trace == 1)
         printf("Clearing block trailer nonce...\n");
      /* ... clear block nonce */
      for (j = 0; j < 32; j++)
      {
         bt.nonce[j] = 0;
      }
      /* ... output initial block trailer data */
      if(Trace == 1) {
         printf("Initial Block Trailer Data:\n");
         for (j = 0; j < 160; j++)
         {
            printf("%03i ", bt_byte[j]);
            if((j + 1) % 16 == 0)
               printf("\n");
         }
         printf("\n");
      }
      
      /* Initialize miner and record initialization time */
      if(Trace == 1)
         printf("Initializing miner...\n");
      inittime[i] = time(NULL);
      if (init_cuda_peach(Difficulty, bt.phash, (byte *) &bt) < 1)
      {
         error("Miner failed to initilize CUDA devices\n");
         return 0;
      }
      inittime[i] = time(NULL) - inittime[i];
      
      /* Begin mining and record mining time */
      if(Trace == 1)
         printf("Begin mining...\n");
      timer[i] = time(NULL);
      cuda_peach((byte *) &bt, &hps[i], &Running);
      timer[i] = time(NULL) - timer[i];
      
      /* Free GPU memory */
      free_cuda_peach();
      
      /* Validate GPU solve on the CPU */
      if(!Running) {
         printf("\ntest_miner ended unexpectedly...\n");
      } else if(peach(&bt, Difficulty, NULL, 1)) {
         printf("Solve FAILED on the CPU!!!\n");
         printf("Last Block Trailer Data:\n");
         byte* bt_bytes = (byte*) &bt;
         for(j = 0; j < 160; j++){
            printf("%03i ", bt_bytes[j]);
            if((j + 1) % 16 == 0)
               printf("\n");
         }
         printf("\n");
         Running = 0;
      } else {
         printf("Solve #%i Diff %d in %li seconds | %u Haiku/s\n",
                i, Difficulty, timer[i], hps[i]);
         if(Trace == 1) {
            trigg_expand2(bt.nonce, phaiku);
            printf("\n%s\n\n", phaiku);
         }
      }
   }
   
   /* Calculate average miner initialization time */
   /* Calculate average solve time */
   /* Calculate average hps */
   avginit = avgtime = avghps = 0;
   for(i = j = 0; i < nsolve; i++) {
      if(!inittime[i] || !timer[i] || !hps[i])
         continue;
      avginit += inittime[i];
      avgtime += timer[i];
      avghps += hps[i];
      j++;
   }
   avginit /= j;
   avgtime /= j;
   avghps /= j;
   /* Calculate actual hps */
   actualhps = (1 << Difficulty) / avgtime;
   /* Calculate hps efficiency*/
   efficiency = ( ((float) actualhps) / ((float) avghps) ) * 100;
   
   printf("\ntest_miner statistics...\n");
   printf("%d Valid mining iterations performed.\n", j);
   printf("Total test runtime: %li seconds\n", time(NULL) - Ltime);
   printf("Average init time:  %li seconds\n", avginit);
   printf("Reported avg HPS:   %u Haiku/s\n", avghps);
   printf("Calculated avg HPS: %u Haiku/s\n", actualhps);
   printf("Miner Efficiency:   %.02f%%\n", efficiency);
   
   printf("\nMiner exiting...\n");
   return 0;
}  /* end miner() */


void usage(void)
{
   printf("usage: ./test_miner [-option...]\n"
          "         -tN        set Trace to N (0, 1)\n"
          "         -iN        set solve iterations\n"
          "         -dN        set difficulty of blocks\n"
          "         -h         this message\n"
   );
   exit(0);
}


/* Start the miner as a child process */
int main(int argc, char **argv)
{
   int j;
   int iterations;
   
   /* Initialize globals */
   Running = 1;
   Difficulty = 20;
   iterations = 1;
   
   /* Initialize pseudo-random number generators */
   srand16(time(&Ltime) ^ getpid());
   srand2(Ltime, 0, 123456789 ^ getpid());
   
   /*
    * Parse command line arguments.
    */
   for(j = 1; j < argc; j++) {
      if(argv[j][0] != '-') usage();
      switch(argv[j][1]) {
         case 't':  Trace = atoi(&argv[j][2]); /* set trace level  */
                    break;
         case 'i':  iterations = atoi(&argv[j][2]); /* set iterations */
                    break;
         case 'd':  Difficulty = atoi(&argv[j][2]); /* set difficulty */
                    break;
         case 'h':  usage();
                    return 0;
         default:   usage();
                    return 1;
      }  /* end switch */
   }  /* end for j */
   
   /* Set Running=0 on ctrl+c and terminate signals */
   for(j = 0; j <= NSIG; j++)
      signal(j, SIG_IGN);
   signal(SIGINT, sigterm);   /* ctrl+c    */
   signal(SIGTERM, sigterm);  /* terminate */
   
   test_miner(iterations);
   return 0;
}  /* end start_miner() */
