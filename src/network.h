/**
 * @file network.h
 * @brief Mochimo network communication support
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
 * @note This unit uses INET support from extended-c extinet header and
 * requires the use of sock_startup() to activate socket support.
*/

/* include guard */

#ifndef MOCHIMO_NETWORK_H
#define MOCHIMO_NETWORK_H

/* system support */
#include <sys/types.h>  /* for pid_t */

/* extended-c support */
#include "extinet.h"
#include "extthread.h"

/* mochimo support */
#include "types.h"



#ifndef HTTPPUSHPEERS
#define HTTPPUSHPEERS "https://new-api.mochimap.com/network/peers/push"
#endif
#ifndef MAXNODES
#define MAXNODES  37    /**< maximum number of connected nodes  */
#endif
#ifndef RPLISTLEN
#define RPLISTLEN 128   /**< recent peer list */
#endif
#ifndef TPLISTLEN
#define TPLISTLEN 32    /**< trusted peer list */
#endif
#ifndef CBITS
#define CBITS  0
#endif

#ifndef TXBUFFLEN
#define TXBUFFLEN  ( (2*5) + (8*2) + (32*3) + 2 + \
                        (TXADDRLEN*3) + (TXAMOUNT*3) + TXSIGLEN + (2+2) )
#endif
#ifndef TXCRC_COUNT
#define TXCRC_COUNT  ( (2*5) + (8*2) + (32*3) + 2 + \
                        (TXADDRLEN*3) + (TXAMOUNT*3) + TXSIGLEN )
#endif

#ifndef STD_TIMEOUT
#define STD_TIMEOUT 10
#endif

#define CPINKLEN     100       /* maximum entries in pinklists       */
#define LPINKLEN     100
#define EPINKLEN     100

/* The Node struct */
typedef struct {
   TX tx;            /* packet buffer */
   word32 ip;        /* source ip *//*
   word16 port;      // unused... */
   word16 id1, id2;  /* from tx handshake */
   char id[22];      /* "0.0.0.0 AB~EF" - for logging identification */
   pid_t pid;        /* process id of child -- zero if empty slot */
   volatile int ts;  /* thread status -- set by thread */
   SOCKET sd;
} NODE;

typedef struct {
   TX tx;            /* packet buffer */
   word32 ip;        /* source ip */
   volatile int tr;  /* thread function result -- set by thread */
   volatile int ts;  /* thread status -- set by thread */
} THREAD_SCAN_ARGS;

#define addrecent(ip)   addpeer(ip, Rplist, RPLISTLEN, &Rplistidx)
#define recentip(ip)    search32(ip, Rplist, RPLISTLEN)
#define save_rplist()   save_ipl("recentip.lst", Rplist, RPLISTLEN)

/* C/C++ compatible function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

char *Bcdir;
int Nonline;
word32 Mfee[2];
word8 Cblocknum[8];
word8 Cblockhash[HASHLEN];
word8 Prevhash[HASHLEN];
word8 Weight[HASHLEN];
word8 Running;

NODE Nodes[MAXNODES];  /* data structure for connected NODE's     */
NODE *Hi_node;         /* points one beyond last logged in NODE   */


time_t Pushtime;           /* time of last OP_MBLOCK */
word8 Allowpush;           /* set by -P flag in mochimo.c */
word8 Cbits;               /* Node capability bits */
word8 Noprivate;           /* filter out private IP's when set v.28 */
word16 Dstport;            /* Destination port (default 2095) */
word32 Ntimeouts;          /* Number of send/recv timeouts */
word32 Nsends, Nsenderrs;  /* Number of sent TX's and send errors */
word32 Nrecvs, Nrecverrs;  /* Number of recv'd TX's and recv errors */
word32 Rplist[RPLISTLEN];  /* Recent peer list */
word32 Rplistidx;
word32 Tplist[TPLISTLEN];  /* Trusted peer list - preserved */
word32 Tplistidx;

/* pink lists of EVIL IP addresses read in from disk */
word32 Cpinklist[CPINKLEN];
word32 Lpinklist[LPINKLEN];
word32 Epinklist[EPINKLEN];
word32 Cpinkidx, Lpinkidx, Epinkidx;
word8 Disable_pink;


word32 *search32(word32 val, word32 *list, unsigned len);
word32 remove32(word32 bad, word32 *list, unsigned maxlen, word32 *idx);
word32 include32(word32 val, word32 *list, unsigned len, word32 *idx);
void shuffle32(word32 *list, word32 len);
int isprivate(word32 ip);
word32 addpeer(word32 ip, word32 *list, word32 len, word32 *idx);
int save_ipl(char *fname, word32 *list, word32 len);
word32 read_ipl(char *fname, word32 *plist, word32 plistlen, word32 *plistidx);
int readpeers(void);
int readpink(void);
int savepink(void);
int pinklisted(word32 ip);
int cpinklist(word32 ip);
int pinklist(word32 ip);
int lpinklist(word32 ip);
int epinklist(word32 ip);
void mergepinklists(void);
void purge_epoch(void);
int recv_tx(NODE *np, double timeout);
int recv_file(NODE *np, char *fname);
int send_tx(NODE *np, double timeout);
int send_op(NODE *np, int opcode);
int send_file(NODE *np, char *fname);
int send_tf(NODE *np);
int send_hash(NODE *np);
int get_op(NODE *np, word16 opcode);
int callserver(NODE *np, word32 ip);
int get_tx(NODE *np, word32 ip, word16 opcode);
int get_file(word32 ip, word8 *bnum, char *fname);
int get_ipl(NODE *np, word32 ip);
int get_hash(NODE *np, word32 ip, void *bnum, void *blockhash);
ThreadProc thread_get_ipl(void *arg);
int scan_network(word32 quorum[], word32 qlen, void *hash, void *weight,
   void *bnum);

/* end extern "C" {} for C++ */
#ifdef __cplusplus
}
#endif

/* end include guard */
#endif
