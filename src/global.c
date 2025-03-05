/**
 * @private
 * @headerfile global.h <global.h>
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_GLOBAL_C
#define MOCHIMO_GLOBAL_C


#include "global.h"

/* internal support */
#include "error.h"

/* external support */
#include "extinet.h"
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
   #include <sys/wait.h>

#endif

int Nonline;         /* number of pid's in Nodes[]                */
word32 Quorum = 3;   /* Number of peers in get_eon() gang[MAXQUORUM] */
word32 Trustblock;   /* trust block validity up to this block     */
word32 Dynasleep;    /* sleep usec. per loop if Nonline < 1       */
word32 Trace;        /* non-zero plog()  trace log                */
word32 Nbalance;     /* total balances sent                       */
word32 Nbadlogs;     /* total bad login attempts                  */
word32 Nspace;       /* Node[] table full count                   */
word32 Nlogins;      /* total logins since boot                   */
word32 Ntimeouts;    /* total client timeouts                     */
word32 Nrec;         /* total TX received                         */
word32 Ngen;         /* total number of main loop iterations      */
word32 Ndups;        /* number of dup TX's received               */
word32 Nupdated;     /* number of blocks updated                  */
word32 Eon;          /* Eons since boot                           */
word32 Txcount;      /* transactions in txq1.dat                  */
word16 Port = PORT1; /* Our listening port                        */
word16 Dstport = PORT1; /* Our send destination port              */
word8 Blockfound;    /* set on receiving OP_FOUND from peer       */
word8 Exportflag;    /* enable database export: #ifdef BX_MYSQL   */
word8 Errorlog;      /* non-zero to log errors to "error.log"     */
word8 Monitor;       /* set non-zero by ctrlc() to enter monitor  */
word8 Bgflag;        /* ignore ctrl-c Monitor and no term output  */
word8 Running = 1;   /* non-zero when server is online            */

char *Statusarg;     /* Statusarg->"message_string" shows on ps */
char *Bcdir = BCDIR; /* block chain directory */
char *Spdir = SPDIR; /* split chain directory */

time_t Utime;        /* update time for watchdog */
word8 Allowpush;     /* set by -P flag in mochimo.c */
word8 Cbits = CBITS; /* 8 capability bits */
word8 Safemode;      /* Safe mode enable */
word8 Ininit;        /* non-zero when init() runs */
word8 Insyncup;      /* non-zero when syncup() runs */
word8 Betabait;      /* betabait() display */
word32 Watchdog;     /* enable watchdog timeout -wN */

/* state globals */
word32 Time0;
word32 Difficulty;
word32 Myfee[2] = { MFEE, 0 };
word32 Mfee[2] = { MFEE, 0 };
word8 Cblocknum[8];
word8 Cblockhash[HASHLEN];
word8 Prevhash[HASHLEN];
word8 Weight[HASHLEN];

/* lock files    writes   reads     deletes
 * mq.lck        gomochi            gomochi
 * neofail.lck   neogen   bupdata   bupdata
*/

/* Global semaphores */
pid_t Bcon_pid;         /* bcon process id */
word8 Bcbnum[8];        /* Cblocknum at time of execl bcon */
pid_t Found_pid;
pid_t Mqpid;            /* mirror() */
int Mqcount;            /* count of mq.dat records */

word8 One[8] = { 1 };   /* for 64-bit maths */

#ifndef _WIN32

/**
 * Terminate services and exit with @a ecode.
 * @param ecode value to supply to exit()
*/
void kill_services_exit(int ecode)
{
   if (Found_pid) kill(Found_pid, SIGTERM);
   if (Bcon_pid) kill(Bcon_pid, SIGTERM);
   if (Mqpid) kill(Mqpid, SIGTERM);
   sock_cleanup();
   Running = 0;
   while (waitpid(-1, NULL, 0) != -1);
   exit(ecode);
}

char *show(char *state)
{
   if(state == NULL) state = "(null)";
   if(Statusarg) strncpy(Statusarg, state, 8);
   return state;
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

#else

void kill_services_exit(int ecode) { exit(ecode); }
int stop_bcon(void) { exit(1); }
int stop_found(void) { exit(1); }

#endif

/* end include guard */
#endif
