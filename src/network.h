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
#ifdef _WIN32  /* Windows no likey */
   #define pid_t  int
#endif

/* extended-c support */
#include "extinet.h"

/* mochimo support */
#include "types.h"
#include "peer.h"

/* The Node struct */
typedef struct {
   TX tx;               /* packet buffer */
   word32 ip;           /* source ip *//*
   word16 port;         // unused... */
   word16 id1, id2;     /* from tx handshake */
   char id[32];         /* "0.0.0.0 AB~EF" - for logging identification */
   pid_t pid;           /* process id of child -- zero if empty slot */
   SOCKET sd;
} NODE;

/* global variables */
extern NODE Nodes[MAXNODES];
extern NODE *Hi_node;
extern word32 Nrecvs;
extern word32 Nsends;
extern word32 Nrecverrs;
extern word32 Nsenderrs;

/* C/C++ compatible function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

NODE *getslot(NODE *np);
int freeslot(NODE *np);
int child_status(NODE *np, pid_t pid, int status);
int recv_tx(NODE *np, double timeout);
int recv_file(NODE *np, char *fname);
int send_tx(NODE *np, double timeout);
int send_op(NODE *np, int opcode);
int send_nack(NODE *np, int errnum);
int send_file(NODE *np, char *fname);
int send_balance(NODE *np);
int send_ipl(NODE *np);
int send_hash(NODE *np);
int send_tf(NODE *np);
int send_identify(NODE *np);
int send_found(void);
int callserver(NODE *np, word32 ip);
int get_file(word32 ip, word8 *bnum, char *fname);
int get_ipl(NODE *np, word32 ip);
int get_hash(NODE *np, word32 ip, void *bnum, void *blockhash);
int gettx(NODE *np, SOCKET sd);
int scan_network
(word32 quorum[], word32 qlen, void *hash, void *weight, void *bnum);
int refresh_ipl(void);

#ifdef __cplusplus
}  /* end extern "C" */
#endif

/* end include guard */
#endif
