/* data.c  Global data structures.
 *
 * Copyright (c) 2018 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 1 January 2018
*/

/* build sequence */
#define PATCHLEVEL  31

/*
 * Globals
 */

byte Running;        /* non-zero when server is online            */
byte Errorlog;       /* non-zero to log errors to "error.log"     */
byte Monitor;        /* set non-zero by ctrlc() to enter monitor  */
byte Bgflag;         /* ignore ctrl-c Monitor and no term output  */
word32 Dynasleep;    /* sleep usec. per loop if Nonline < 1       */
word32 Trace;        /* non-zero plog()  trace log                */
int Nonline;         /* number of pid's in Nodes[]                */
word32 Nbadlogs;     /* total bad login attempts                  */
word32 Nspace;       /* Node[] table full count                   */
word32 Nlogins;      /* total logins since boot                   */
word32 Ntimeouts;    /* total client timeouts                     */
word32 Nrec;         /* total TX received                         */
word32 Nsent;        /* total TX sent                             */
word32 Ngen;         /* total number of main loop iterations      */
word32 Nsenderr;     /* number of send errors                     */
word32 Ndups;        /* number of dup TX's received               */
word32 Nsolved;      /* number of blocks solved by miner          */
word32 Nupdated;     /* number of blocks updated                  */
word32 Eon;          /* Eons since boot                           */
word32 Txcount;      /* transactions in txq1.dat                  */
word32 Time0;        /* for set_difficulty()                      */

/*
 * real time of current server loop - set by server()
 */
time_t Ltime;
time_t Stime;            /* status display update time */

word16 Port;             /* Our listening port */
word16 Dstport;          /* Our send destination port */
char *Bcdir = BCDIR;     /* block chain directory */
char *Ngdir = NGDIR;     /* block chain directory */

#ifndef EXCLUDE_NODES
NODE Nodes[MAXNODES];  /* data structure for connected NODE's     */
NODE *Hi_node = Nodes; /* points one beyond last logged in NODE   */

word32 Rplist[RPLISTLEN];  /* recent peer list */
word32 Rplistidx;
word32 Cplist[CPLISTLEN];  /* current peer list */
word32 Cplistidx;
word32 Crclist[CRCLISTLEN];  /* crc's of recent TX's */
word32 Crclistidx;

#define CORELISTLEN 16
#if CORELISTLEN > RPLISTLEN
#error Fix CORELISTLEN
#endif
word32 Coreplist[CORELISTLEN] = {  /* ip's of the Core Network */
   0x0100007f,    /* local host  debug */
};

int Quorum = 4;         /* Number of peers in get_eon() gang[MAXQUORUM] */
byte Ininit;            /* non-zero when init() runs */
byte Safemode;          /* Safe mode enable */
byte Nominer;           /* set true to stop mining in server data.c @ */
byte Watchdog;          /* restart if stuck on block 0x0 for long */
byte Betabait;          /* betabait() display */
byte Cbits = CBITS;     /* 8 capability bits */
time_t Pushtime;        /* time of last OP_MBLOCK */
byte Allowpush;         /* set by -P flag in mochimo.c */

#endif  /* !EXCLUDE_NODES Nodes[] and ip data */

word32 Mfee[2] = { 500, 0 };  /* mining fee */
byte Maddr[TXADDRLEN];        /* mining address read by bcon and bval */
word32 Difficulty;
byte One[8] = { 1 };          /* for 64-bit maths */

byte Cblocknum[8];
byte Cblockhash[HASHLEN];  /* [32] */
byte Prevhash[HASHLEN];
byte Weight[HASHLEN];

/* lock files    writes   reads     deletes
 * mq.lck        gomochi            gomochi
 * neofail.lck   neogen   bupdata   bupdata
 * ufail.lck     bup                gomochi
 * vbad.lck      bval               bval
 * ubad.lck      bup                bup
*/

/* Global semaphores */
byte Blockfound;          /* set on receiving OP_FOUND from peer */
word32 Peerip;            /* gift to bval and others */
byte Disable_pink;
byte Needcleanup;         /* set true when Winsock is started */
char *Corefname = "coreip.lst";  /* Master ip list by main() */
pid_t Bcpid;              /* bcon process id */
byte Bcbnum[8];           /* Cblocknum at time of execl bcon */
pid_t Sendfound_pid;
pid_t Mpid;               /* miner */
pid_t Mqpid;              /* mirror() */
int Mqcount;              /* count of mq.dat records */
