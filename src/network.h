/**
 * @file network.h
 * @brief Mochimo network communication support.
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
 * @note This unit uses INET support from the extended-c extinet header and
 * requires the use of sock_startup() to activate socket support.
*/

/* include guard */

#ifndef MOCHIMO_NETWORK_H
#define MOCHIMO_NETWORK_H

/* system support */
#include <sys/types.h>  /* for pid_t */

/* extended-c support */
#include "extinet.h"

/* mochimo support */
#include "types.h"
#include "peer.h"

/* The Node struct */
typedef struct {
   TX tx;            /* packet buffer */
   word32 ip;        /* source ip *//*
   word16 port;      // unused... */
   word16 id1, id2;  /* from tx handshake */
   char id[24];      /* "0.0.0.0 AB~EF" - for logging identification */
   pid_t pid;        /* process id of child -- zero if empty slot */
   volatile int ts;  /* thread status -- set by thread */
   SOCKET sd;
} NODE;

/* C/C++ compatible function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

NODE Nodes[MAXNODES];   /* data structure for connected NODE's */
NODE *Hi_node;          /* points one beyond last logged in NODE */

word32 Nrecvs;          /* number of receive errors */
word32 Nsends;          /* number of send errors */
word32 Nrecverrs;       /* number of receive errors */
word32 Nsenderrs;       /* number of send errors */

NODE *getslot(NODE *np);
int freeslot(NODE *np);
int child_status(NODE *np, pid_t pid, int status);
int recv_tx(NODE *np, double timeout);
int recv_file(NODE *np, char *fname);
int send_tx(NODE *np, double timeout);
int send_op(NODE *np, int opcode);
int send_file(NODE *np, char *fname);
int send_balance(NODE *np);
int send_ipl(NODE *np);
int send_hash(NODE *np);
int send_tf(NODE *np);
int send_identify(NODE *np);
int send_resolve(NODE *np);
int send_found(void);
int callserver(NODE *np, word32 ip);
int get_tx(NODE *np, word32 ip, word16 opcode);
int get_file(word32 ip, word8 *bnum, char *fname);
int get_ipl(NODE *np, word32 ip);
int get_hash(NODE *np, word32 ip, void *bnum, void *blockhash);
int handle_tx(NODE *np, SOCKET sd);
int scan_network
(word32 quorum[], word32 qlen, void *hash, void *weight, void *bnum);
int refresh_ipl(void);


/* end extern "C" {} for C++ */
#ifdef __cplusplus
}
#endif

/* end include guard */
#endif
