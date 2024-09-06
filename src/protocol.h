/**
 * @file protocol.h
 * @brief Mochimo network protocol and communication support.
 * @copyright Adequate Systems LLC, 2018-2024. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
 * @note This unit uses INET support from the extended-c extinet header and
 * requires the use of wsa_startup() to activate socket support on Windows.
*/

/* include guard */
#ifndef MOCHIMO_PROTOCOL_H
#define MOCHIMO_PROTOCOL_H


/* internal support */
#include "memdata.h"
#include "server.h"
#include "types.h"

/* external support */
#include "extinet.h"

#ifdef _WIN32
   /* Windows errors need separate processing from Standard C errors */
   #define set_sockerrno(e)   set_alterrno(e)

#else
   #define set_sockerrno(e)   set_errno(e)

#endif

#ifdef STD_TIMEOUT
   #define PROTOCOL_TIMEOUT   STD_TIMEOUT
#else
   #define PROTOCOL_TIMEOUT   3
#endif

#define NODEPEER(np) ( (np)->addr.sin_addr.s_addr )

/** Protocol Data Unit struct */
typedef struct {
   word8 version[2];          /* { PVERSION, Cbits } */
   word8 network[2];          /* { 0x39, 0x05 } TXNETWORK -- Mochimo */
   word8 id1[2];              /* handshake identification */
   word8 id2[2];              /* handshake identification */
   word8 opcode[2];           /* operation code of this packet */
   word8 cblock[8];           /* current block num 64-bit */
   word8 blocknum[8];         /* block num for I/O in progress */
   word8 cblockhash[HASHLEN]; /* sha-256 hash of current block */
   word8 pblockhash[HASHLEN]; /* sha-256 hash of previous block */
   word8 weight[32];          /* sum of block diffs (or TX ip map) */
   word8 len[2];              /* length of data in buffer for I/O op's */
   word8 buffer[WORD16_MAX];  /* packet buffer (actual data may vary) */
   word8 __align_crc16[1];    /* (re)align crc16 to 4 byte boundary */
   word8 crc16[2];            /* CRC16 hash of PKT[124 + PKT.len] */
   word8 trailer[2];          /* { 0xcd, 0xab } TXEOT -- always */
} PDU;

/** The Mochimo network connection handler struct */
typedef struct {
   PDU pdu;       /* Protocol Data Unit for active socket operations */
   MEM *mp;       /* mem pointer for MEM stream operations */
   FILE *fp;      /* file pointer for DISK stream operations */
   word16 id1;    /* handshake verification ID (OP_HELLO) */
   word16 id2;    /* handshake verification ID (OP_HELLO_ACK) */
   word16 oplast; /* last operation code (fully) recv'd or sent */
   word16 opreq;  /* operation request (OUTGOING CONNECTIONS) */
   word8 bnum[8]; /* reference blocknum (OUTGOING CONNECTIONS) */
   int errnum;    /* latest error number */
   int status;    /* latest status result */
   int bytes;     /* PDU bytes recv'd or sent */
   struct sockaddr_in addr; /* remote IPv4 address */
} NODE;

extern word32 Ntimeouts;
extern word32 Nerrors;
extern word32 Ndrops;
extern word32 Nrecvs;
extern word32 Nsends;

extern word8 Weight[32];
extern word8 Prevhash[32];
extern word8 Cblockhash[32];
extern word8 Cblocknum[8];
extern word8 Cbits;

/* C/C++ compatible function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

struct sockaddr_in ipv4_addr(word32 ip, word16 port);
int node_accept(CONNECTION *cp, struct sockaddr *addrp, socklen_t len);
NODE *node_create(struct sockaddr *addrp, socklen_t len);
void node_destroy(NODE *np);
void node_prepare(NODE *np, word16 opcode);
int node_recv(CONNECTION *cp);
int node_send(CONNECTION *cp);
int node_tranceive__incoming(CONNECTION *cp);
int node_tranceive__outgoing(CONNECTION *cp);
int node_tranceive(CONNECTION *cp);

#ifdef __cplusplus
}  /* end extern "C" */
#endif

/* end include guard */
#endif
