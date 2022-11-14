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
extern unsigned Niplist;
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

void cleanup_node(SNODE *snp);
void init_receive(SNODE *snp, SOCKET sd, word32 ip);
void init_request
   (SNODE *snp, word32 ip, word16 port, word16 opreq, void *bnum);
void init_pkt(SNODE *snp, word16 opcode);
int recv_pkt(SNODE *snp);
int recv_file(SNODE *snp);
int send_pkt(SNODE *snp);
int send_ipl(SNODE *snp);
int send_balance(SNODE *snp);
int receive_node(SNODE *snp);
int request_node(SNODE *snp);

#ifdef __cplusplus
}  /* end extern "C" */
#endif

/* end include guard */
#endif
