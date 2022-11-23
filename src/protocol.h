/**
 * @file protocol.h
 * @brief Mochimo network communication protocol support.
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
 * @note This unit uses INET support from the extended-c extinet header
 * and requires the use of sock_startup() to activate socket support.
*/

/* include guard */
#ifndef MOCHIMO_PROTOCOL_H
#define MOCHIMO_PROTOCOL_H


/* internal support */
#include "types.h"

extern unsigned Nbalance;
extern unsigned Nhashes;
extern unsigned Niplist;
extern unsigned Nnacks;
extern unsigned Nrecvs;
extern unsigned Nrecverrs;
extern unsigned Nrecvsbad;
extern unsigned Nsends;
extern unsigned Nsenderrs;

extern word8 Cbits;
extern word8 Cblocknum[8];
extern word8 Cblockhash[32];
extern word8 Prevhash[32];
extern word8 Weight[32];

/* C/C++ compatible function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

void init_pkt(NODE *np, word16 opcode);
int recv_pkt(NODE *np);
int recv_file(NODE *np);
int send_pkt(NODE *np);
int send_balance(NODE *np);
int send_file(NODE *np);
int send_fp(NODE *np);
int send_hash(NODE *np);
int send_ipl(NODE *np);
void node_cleanup(NODE *np);
void node_init(NODE *np, word32 ip, word16 port, word16 opreq, void *bnum);
int node_receive_handshake(NODE *np);
int node_receive_operation(NODE *np);
int node_request_connect(NODE *np, int nonblock);
int node_request_handshake(NODE *np);
int node_request_operation(NODE *np);

#ifdef __cplusplus
}  /* end extern "C" */
#endif

/* end include guard */
#endif
