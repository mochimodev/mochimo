/**
 * @file global.h
 * @brief Mochimo global declarations.
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_GLOBAL_H
#define MOCHIMO_GLOBAL_H


/* external support */
#include "types.h"

/* emergency stops */
#define restart(msg) { palert(msg); kill_services_exit(1); }
#define resign(msg) { palert(msg); kill_services_exit(0); }

extern int Nonline;         /* number of pid's in Nodes[]                */
extern word32 Quorum;       /* Number of peers in get_eon() gang[MAXQUORUM] */
extern word32 Trustblock;   /* trust block validity up to this block     */
extern word32 Dynasleep;    /* sleep usec. per loop if Nonline < 1       */
extern word32 Trace;        /* non-zero plog()  trace log                */
extern word32 Nbalance;     /* total balances sent                       */
extern word32 Nbadlogs;     /* total bad login attempts                  */
extern word32 Nspace;       /* Node[] table full count                   */
extern word32 Nlogins;      /* total logins since boot                   */
extern word32 Ntimeouts;    /* total client timeouts                     */
extern word32 Nrec;         /* total TX received                         */
extern word32 Ngen;         /* total number of main loop iterations      */
extern word32 Ndups;        /* number of dup TX's received               */
extern word32 Nupdated;     /* number of blocks updated                  */
extern word32 Eon;          /* Eons since boot                           */
extern word32 Txcount;      /* transactions in txq1.dat                  */
extern word16 Port;         /* Our listening port                        */
extern word16 Dstport;      /* Our send destination port                 */
extern word8 Blockfound;    /* set on receiving OP_FOUND from peer       */
extern word8 Exportflag;    /* enable database export: #ifdef BX_MYSQL   */
extern word8 Errorlog;      /* non-zero to log errors to "error.log"     */
extern word8 Monitor;       /* set non-zero by ctrlc() to enter monitor  */
extern word8 Bgflag;        /* ignore ctrl-c Monitor and no term output  */
extern word8 Running;       /* non-zero when server is online            */

extern char *Statusarg;     /* Statusarg->"message_string" shows on ps */
extern char *Bcdir;         /* block chain directory */
extern char *Spdir;         /* block chain directory */

extern time_t Utime;        /* update time for watchdog */
extern word8 Allowpush;     /* set by -P flag in mochimo.c */
extern word8 Cbits;         /* 8 capability bits */
extern word8 Safemode;      /* Safe mode enable */
extern word8 Ininit;        /* non-zero when init() runs */
extern word8 Insyncup;      /* non-zero when syncup() runs */
extern word8 Betabait;      /* betabait() display */
extern word32 Watchdog;     /* enable watchdog timeout -wN */

/* state globals */
extern word32 Time0;
extern word32 Difficulty;
extern word32 Myfee[2];
extern word32 Mfee[2];
extern word8 Cblocknum[8];
extern word8 Cblockhash[HASHLEN];
extern word8 Prevhash[HASHLEN];
extern word8 Weight[HASHLEN];

/* lock files    writes   reads     deletes
 * mq.lck        gomochi            gomochi
 * neofail.lck   neogen   bupdata   bupdata
*/

/* Global semaphores */
extern pid_t Bcon_pid;              /* bcon process id */
extern word8 Bcbnum[8];           /* Cblocknum at time of execl bcon */
extern pid_t Found_pid;
extern pid_t Mqpid;              /* mirror() */
extern int Mqcount;              /* count of mq.dat records */

extern word8 One[8];             /* for 64-bit maths */

/* C/C++ compatible function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

void kill_services_exit(int ecode);
char *show(char *state);
int stop_bcon(void);
int stop_found(void);
void stop_mirror(void);

#ifdef __cplusplus
}  /* end extern "C" */
#endif

/* end include guard */
#endif
