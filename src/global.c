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

int Nonline;         /* number of pid's in Nodes[]                */
word32 Quorum = 4;   /* Number of peers in get_eon() gang[MAXQUORUM] */
word32 Trustblock;   /* trust block validity up to this block     */
word32 Hps;          /* haiku per second from miner.c hps.dat     */
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
word32 Nsolved;      /* number of blocks solved by miner          */
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
word8 Running;       /* non-zero when server is online            */

char *Statusarg;     /* Statusarg->"message_string" shows on ps */
char *Bcdir = BCDIR; /* block chain directory */
char *Spdir = SPDIR; /* split chain directory */

time_t Utime;        /* update time for watchdog */
word8 Allowpush;     /* set by -P flag in mochimo.c */
word8 Cbits = CBITS; /* 8 capability bits */
word8 Safemode;      /* Safe mode enable */
word8 Ininit;        /* non-zero when init() runs */
word8 Insyncup;      /* non-zero when syncup() runs */
word8 Nominer;       /* Do not start miner if true -n */
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
pid_t Mpid;             /* miner */
pid_t Mqpid;            /* mirror() */
int Mqcount;            /* count of mq.dat records */

word8 One[8] = { 1 };   /* for 64-bit maths */

/* end include guard */
#endif
