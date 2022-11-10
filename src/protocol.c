/**
 * @private
 * @headerfile protocol.h <protocol.h>
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_PROTOCOL_C
#define MOCHIMO_PROTOCOL_C


#include "protocol.h"

/* internal support */
#include "peer.h"
#include "error.h"

/* external support */
#include <string.h>
#include "extlib.h"
#include "crc16.h"

#ifdef _WIN32
   #define set_sockerrno(e)   set_alterrno(e)

#else
   #define set_sockerrno(e)   set_errno(e)

#endif

/** Lifetime packets received */
unsigned Nrecvs = 0;
/** Lifetime packet receive errors */
unsigned Nrecverrs = 0;
/** Lifetime packet violations */
unsigned Nrecvsbad = 0;
/** Lifetime packets sent */
unsigned Nsends = 0;
/** Lifetime packet send errors */
unsigned Nsenderrs = 0;

/** Capability bits sent with packet during communication protocol */
word8 Cbits = 0;
/** Current block number sent with packet during communication protocol */
word8 Cblocknum[8] = { 0 };
/** Current block hash sent with packet during communication protocol */
word8 Cblockhash[32] = { 0 };
/** Previous block hash sent with packet during communication protocol */
word8 Prevhash[32] = { 0 };
/** Current weight sent with packet during communication protocol */
word8 Weight[32] = { 0 };

/**
 * Deallocate (cleanup) resources allocated within a SNODE.
 * NOTE: DOES NOT DEALLOCATE THE SNODE POINTER.
 * @param snp Pointer to SNODE
*/
void node_cleanup(SNODE *snp)
{
   /* ensure socket is closed */
   if (snp->sd != INVALID_SOCKET) {
      sock_close(snp->sd);
      snp->sd = INVALID_SOCKET;
   }
   /* ensure DATA pointer is deallocated */
   if (snp->fp != NULL) {
      fclose(snp->fp);
      snp->fp = NULL;
   }
}  /* end node_cleanup() */

/**
 * Prepare a SNODE pointer for receiving.
 * @param snp Pointer to SNODE to prepare
 * @param sd Connection socket of SNODE
 * @param ip Connection ip of SNODE
*/
void prep_receive(SNODE *snp, SOCKET sd, word32 ip)
{
   /* prepare SNODE data receive */
   snp->sd = sd;
   snp->ip = ip;
   snp->port = 0;
   snp->opreq = OP_NULL;
   snp->opcode = OP_NULL;
   snp->iowait = IO_RECV;
   snp->status = VEWAITING;
   ntoa(&ip, snp->id);
}  /* end prep_receive() */

/**
 * Prepare a SNODE pointer for requesting.
 * @param snp Pointer to SNODE to prepare
 * @param ip Connection ip of SNODE
 * @param opreq Request operation code
 * @param bnum Request IO value (blocknum), or NULL
*/
void prep_request
   (SNODE *snp, word32 ip, word16 port, word16 opreq, void *bnum)
{
   /* prepare SNODE data request */
   snp->sd = INVALID_SOCKET;
   snp->ip = ip;
   snp->port = port;
   snp->opreq = opreq;
   snp->opcode = OP_NULL;
   snp->iowait = IO_CONN;
   snp->status = VEWAITING;
   ntoa(&ip, snp->id);
   if (bnum) memcpy(snp->io, bnum, 8);
}  /* end prep_request() */

/**
 * Prepare a packet of SNODE with protocol data.
 * @param snp Pointer to SNODE containing packet to prepare
 * @param opcode Operation code of packet
*/
void prep_pkt(SNODE *snp, word16 opcode)
{
   word16 len;

   /* fill packet with relevant information... */
   snp->pkt.version[0] = PVERSION;
   snp->pkt.version[1] = Cbits | C_VPDU;
   put16(snp->pkt.network, TXNETWORK);
   put16(snp->pkt.id1, snp->id1);
   put16(snp->pkt.id2, snp->id2);
   put16(snp->pkt.opcode, opcode);
   put64(snp->pkt.cblock, Cblocknum);
   memcpy(snp->pkt.cblockhash, Cblockhash, HASHLEN);
   memcpy(snp->pkt.pblockhash, Prevhash, HASHLEN);
   /* ... but, do not overwrite TX ip map */
   if (opcode != OP_TX) memcpy(snp->pkt.weight, Weight, HASHLEN);

   /* store (actual) packet buffer length for CRC hash */
   len = get16(snp->pkt.len);

   /************************************/
   /* PROTOCOL VERSION 4 COMPATIBILITY */

   /* check for VPDU capability */
   /** @todo adjust after v3.0 */
   if (!snp->c_vpdu) {
      /* protocol version 4 packets use a fixed length PDU */
      if (len < PKTBUFFLEN_OLD) len = PKTBUFFLEN_OLD;
      /* opcode specific checks */
      switch (get16(snp->pkt.opcode)) {
         /* for initial compatibility, set len param to fixed length PDU */
         case OP_HELLO: put16(snp->pkt.len, PKTBUFFLEN_OLD); break;
         /* for peerlist compatibility, some opcodes MUST have ZERO len */
         case OP_TX: put16(snp->pkt.len, 0); break;
         case OP_FOUND: put16(snp->pkt.len, 0); break;
         case OP_GET_IPL: put16(snp->pkt.len, 0); break;
         case OP_GET_TFILE: put16(snp->pkt.len, 0); break;
      }
   }

   /* END PROTOCOL VERSION 4 COMPATIBILITY */
   /****************************************/

   /* compute packet crc16 checksum -- add trailer */
   put16(snp->pkt.crc16, crc16(&(snp->pkt), PKTCRC_INLEN(len)));
   put16(snp->pkt.trailer, TXEOT);
}  /* end prep_pkt() */

/**
 * @private
 * Prepare variables for next recv(). For compatibility between
 * pversion 4, C_VPDU capable pversion 4, and pversion 5 onwards, we
 * require the capabilities information in the header of a packet to
 * know how much buffer data to receive.
 * @param snp Pointer to SNODE
 * @todo Remove protocol version 4 compatibility after v3.0
*/
static int recv_len(SNODE *snp, char **buf, int *len, int *n)
{
   /* determine position of and length of next data to recv */
   if (snp->bytes < PKTHDRLEN) {
      /* receive packet header */
      *n = snp->bytes;
      *len = PKTHDRLEN;
      *buf = (char *) snp->pkt.version;
   } else if (PKT_HAS_C_VPDU(&(snp->pkt))) {
      *len = (int) get16(snp->pkt.len);
      if (snp->bytes < (PKTHDRLEN + *len)) {
         /* receive packet buffer */
         *n = snp->bytes - PKTHDRLEN;
         *buf = (char *) snp->pkt.buffer;
      } else {
         /* receive packet trailer */
         *n = snp->bytes - (PKTHDRLEN + *len);
         *len = PKTTLRLEN;
         *buf = (char *) snp->pkt.crc16;
      }
   } else if (snp->bytes < (PKTHDRLEN + PKTBUFFLEN_OLD)) {
      /* receive packet buffer */
      *n = snp->bytes - PKTHDRLEN;
      *len = PKTBUFFLEN_OLD;
      *buf = (char *) snp->pkt.buffer;
   } else {
      /* receive packet trailer */
      *n = snp->bytes - (PKTHDRLEN + PKTBUFFLEN_OLD);
      *len = PKTTLRLEN;
      *buf = (char *) snp->pkt.crc16;
   }

   return *len - *n;
}  /* end recv_len() */

/**
 * Receive a packet of data from a node.
 * NOTE: return value is also placed in snp->status
 * @param snp Pointer to a node
 * @return (int) value representing receive result
 * @retval VEWAITING on waiting for data
 * @retval VETIMEOUT on connection timeout
 * @retval VEOK on success; packet received
 * @retval VERROR on socket error
 * @retval VEBAD on protocol violation
*/
int recv_pkt(SNODE *snp)
{
   PKT *pkt;
   char *buf;
   int ecode, count, len, n;

   /* init */
   pkt = &(snp->pkt);

   /* receive variable PDU into pkt */
   while (recv_len(snp, &buf, &len, &n)) {
      if (len > PKTBUFFLEN) goto FAIL_OVERFLOW;
      count = recv(snp->sd, buf + n, len - n, 0);
      switch (count) {
         case (-1):
            ecode = sock_errno;
            /* check timeout if waiting for data */
            if (sock_waiting(ecode)) {
               if (difftime(time(NULL), snp->to) > 0) {
                  return (snp->status = VETIMEOUT);
               } else return (snp->status = VEWAITING);
            } else goto FAIL_ECODE;    /* socket error ocurred */
         case 0: goto FAIL_SHUTDOWN;   /* socket was shutdown */
         default:
            /* reset timeout, add recv'd bytes, update length */
            snp->to = time(NULL) + TIMEOUT;
            snp->bytes += count;
      }  /* end switch (count... */
   }  /* end for (n = snp... */

   /* full packet received: set c_vpdu, opcode and reset bytes */
   snp->c_vpdu = PKT_HAS_C_VPDU(&(snp->pkt));
   snp->opcode = get16(snp->pkt.opcode);
   snp->bytes = 0;

   /* check crc16 checksum, network version, and trailer */
   len = snp->c_vpdu ? get16(snp->pkt.len) : PKTBUFFLEN_OLD;
   if (get16(pkt->crc16) != crc16(pkt, PKTCRC_INLEN(len))) goto BAD_CRC;
   if (get16(pkt->network) != TXNETWORK) goto BAD_NET;
   if (get16(pkt->trailer) != TXEOT) goto BAD_TLR;
   /* check handshake IDs on all operations (except during handshake) */
   if (snp->opcode >= FIRST_OP) {
      if (snp->id1 != get16(pkt->id1)) goto BAD_IDS;
      if (snp->id2 != get16(pkt->id2)) goto BAD_IDS;
   }

   /* success -- increment recv's */
   Nrecvs++;
   return (snp->status = VEOK);

/* error handling */
FAIL_OVERFLOW: set_errno(EOVERFLOW); goto FAIL;
FAIL_SHUTDOWN: set_errno(ECONNABORTED); goto FAIL;
FAIL_ECODE: set_sockerrno(ecode);
FAIL:
   Nrecverrs++;
   return (snp->status = VERROR);

/* protocol violation handling */
BAD_CRC: set_errno(EMCM_PKTCRC); goto BAD;
BAD_NET: set_errno(EMCM_PKTNET); goto BAD;
BAD_TLR: set_errno(EMCM_PKTTLR); goto BAD;
BAD_IDS: set_errno(EMCM_PKTIDS);
BAD:
   Nrecvsbad++;
   return (snp->status = VEBAD);
}  /* end recv_pkt() */

/**
 * Receive multiple packets of data into a file for a SNODE.
 * NOTE: File data is received into a DATA pointer.
 * @param snp Pointer to a SNODE
 * @return (int) value representing operation result
 * @retval VEWAITING on waiting for data
 * @retval VETIMEOUT on connection timeout
 * @retval VEOK on success; pkt is complete
 * @retval VERROR on internal error
 * @retval VEBAD on protocol violation
*/
int recv_file(SNODE *snp)
{
   int len;

   /* receive packet of data */
   while (recv_pkt(snp) == VEOK) {
      /* check recv'd opcode is OP_SEND_FILE */
      if (snp->opcode == OP_NACK) goto FAIL_NACK;
      if (snp->opcode != OP_SEND_FILE) goto BAD_OPCODE;
      /* check received length */
      len = get16(snp->pkt.len);
      if (len) {
         /* allocate temporary file if no file allocated */
         if (snp->fp == NULL) {
            snp->fp = tmpfile();
            if (snp->fp == NULL) goto FAIL;
         }
         /* write packet buffer to DATA pointer */
         if (fwrite(PKTBUFF(&(snp->pkt)), len, 1, snp->fp) != 1) goto FAIL;
      }
      /* check EOF condition -- check VPDU capability bit */
      /** @todo adjust after v3.0 */
      if (!snp->c_vpdu) {
         if (len < PKTBUFFLEN_OLD) break;
      } else if (len < PKTBUFFLEN) break;
   } /* end while(recv_pkt()) */

   /* snp->status is set during recv_pkt() */
   return snp->status;

/* error handling */
FAIL_NACK: set_errno(EMCM_PKTNACK);
FAIL: return (snp->status = VERROR);

/* protocol violation handling */
BAD_OPCODE: set_errno(EMCM_PKTOPCODE); return (snp->status = VEBAD);
}  /* end recv_file() */

/**
 * @private
 * Obtain the next send() length, in bytes. For compatibility between
 * pversion 4, C_VPDU capable pversion 4, and pversion 5 onwards, we
 * require the information in the header of a packet. This is NOT
 * immediately possible if we are initiating a connection, so OP_HELLO
 * packets will always be padded, until we can identify the capability
 * of the connection we make, by checking the next received packet.
 * NOTE: Maintaining a list of IPs identifying C_VPDU capabilities is
 * flawed because it's possible to receive connections from multiple
 * sources behind the same IP using different capabilities; be it
 * nodes, wallets, APIs, tx bots or other scripts.
 * @param snp Pointer to SNODE
 * @returns (int) length, in bytes, of next send()
*/
static int send_len(SNODE *snp, char **buf, int *len, int *n)
{
   /* determine position of and length of next data to send */
   if (snp->bytes < PKTHDRLEN) {
      /* send packet header */
      *n = snp->bytes;
      *len = PKTHDRLEN;
      *buf = (char *) snp->pkt.version;
   } else if (snp->c_vpdu) {
      *len = (int) get16(snp->pkt.len);
      if (snp->bytes < (PKTHDRLEN + *len)) {
         /* send packet buffer */
         *n = snp->bytes - PKTHDRLEN;
         *buf = (char *) snp->pkt.buffer;
      } else {
         /* send packet trailer */
         *n = snp->bytes - (PKTHDRLEN + *len);
         *len = PKTTLRLEN;
         *buf = (char *) snp->pkt.crc16;
      }
   } else if (snp->bytes < (PKTHDRLEN + PKTBUFFLEN_OLD)) {
      /* send packet buffer */
      *n = snp->bytes - PKTHDRLEN;
      *len = PKTBUFFLEN_OLD;
      *buf = (char *) snp->pkt.buffer;
   } else {
      /* send packet trailer */
      *n = snp->bytes - (PKTHDRLEN + PKTBUFFLEN_OLD);
      *len = PKTTLRLEN;
      *buf = (char *) snp->pkt.crc16;
   }

   return *len - *n;
}  /* end send_len() */

/**
 * Send a packet of data to an SNODE.
 * NOTE: return value is also placed in snp->status
 * @param snp Pointer to a SNODE
 * @return (int) value representing send result
 * @retval VEWAITING on waiting for data
 * @retval VETIMEOUT on connection timeout
 * @retval VEOK on success; pkt is complete
 * @retval VERROR on internal error
 * @retval VEBAD on protocol violation
*/
int send_pkt(SNODE *snp)
{
   char *buf;
   int ecode, count, len, n;

   /* send PDUs of varying size and capabilities */
   while (send_len(snp, &buf, &len, &n)) {
      count = send(snp->sd, buf + n, len - n, 0);
      switch (count) {
         case (-1):
            ecode = sock_errno;
            /* check timeout if waiting for data */
            if (sock_waiting(ecode)) {
               if (difftime(time(NULL), snp->to) > 0) {
                  return (snp->status = VETIMEOUT);
               } else return (snp->status = VEWAITING);
            } else goto FAIL_ECODE;    /* socket error ocurred */
         case 0: goto FAIL_SHUTDOWN;   /* socket was shutdown */
         default:
            /* reset timeout, add recv'd bytes, update length */
            snp->to = time(NULL) + TIMEOUT;
            snp->bytes += count;
      }  /* end switch (count... */
   }  /* end for (n = snp... */

   /* full packet received: set opcode and reset bytes */
   snp->opcode = get16(snp->pkt.opcode);
   snp->bytes = 0;

   /* success -- increment send's */
   Nsends++;  /* requires atomic operation */
   return (snp->status = VEOK);

FAIL_SHUTDOWN: set_errno(ECONNABORTED); goto FAIL;
FAIL_ECODE: set_sockerrno(ecode);
FAIL:
   Nsenderrs++;
   return (snp->status = VERROR);
}  /* end send_pkt() */

/**
 * Network communication protocol for receiving.
 * NOTE: return value is also placed in snp->status
 * @param snp Pointer to a SNODE
 * @return (int) value representing operation result
 * @retval VEWAITING on waiting for data
 * @retval VETIMEOUT on connection timeout
 * @retval VEOK on success; pkt is complete
 * @retval VERROR on internal error
 * @retval VEBAD on protocol violation; peer is now pinklisted
*/
int node_receive(SNODE *snp)
{
OP_RESTART:
   /* check stage of communication */
   switch (snp->opcode) {
      case OP_NULL: {
         /* receive/check OP_HELLO packet */
         if (recv_pkt(snp)) break;
         if (snp->opcode != OP_HELLO) {
            set_errno(EMCM_NOHELLO);
            snp->status = VEBAD;
            break;
         }
         /* prepare "acknowledgement" packet with handshake IDs */
         snp->id1 = get16(snp->pkt.id1);
         snp->id2 = rand16();
         prep_pkt(snp, OP_HELLO_ACK);
         /* update iowait type */
         snp->iowait = IO_SEND;
      } /* fallthrough -- end OP_NULL*/
      case OP_HELLO: {
         /* send OP_HELLO_ACK packet */
         if (send_pkt(snp)) break;
         /* update iowait type and break */
         snp->iowait = IO_RECV;
         snp->status = VEWAITING;
         break;  /* VEWAITING for recv */
      }  /* end case OP_HELLO (or OP_NULL) */
      case OP_HELLO_ACK: {
         /* receive request packet */
         if (recv_pkt(snp)) break;
         /* recv'd opcode MUST be a "valid" operation code */
         /* NOTE: recv'd opcode MUST be checked here */
         if (snp->opcode < FIRST_OP || snp->opcode > LAST_OP) {
            set_errno(EMCM_OPINVAL);
            snp->status = VERROR;
            break;
         }
         /* restart switch block on success */
         goto OP_RESTART;
      }  /* end case OP_HELLO_ACK */
      default: {
         set_errno(EMCM_OPCODE);
         snp->status = VERROR;
      }  /* end default */
   }  /* end switch (snp->opcode) */

   /* check protocol status */
   if (snp->status != VEWAITING) {
      /* close socket, set iowait done, remove timeout */
      sock_close(snp->sd);
      snp->sd = INVALID_SOCKET;
      snp->iowait = IO_DONE;
      snp->to = 0;
      /* check naughty peers -- pinklist */
      if (snp->status == VEBAD2 || snp->status == VEBAD) {
         if (snp->status == VEBAD2) epinklist(snp->ip);
         pinklist(snp->ip);
      }
   }

   /* return resulting status */
   return snp->status;
}  /* end node_receive() */

/**
 * Initiate a request to connect with a server.
 * NOTE: return value is also placed in snp->status
 * @param snp Pointer to a SNODE
 * @return (int) value representing operation result
 * @retval VEWAITING on waiting for data
 * @retval VETIMEOUT on connection timeout
 * @retval VEOK on success; pkt is complete
 * @retval VERROR on internal error
*/
static int request_connect(SNODE *snp)
{
   static const socklen_t len = (socklen_t) sizeof(struct sockaddr_in);
   struct sockaddr_in addr;
   int ecode;

   /* create non-blocking socket, AF_INET = IPv4 */
   if (snp->sd == INVALID_SOCKET) {
      snp->sd = socket(AF_INET, SOCK_STREAM, 0);
      if (snp->sd == INVALID_SOCKET || sock_set_nonblock(snp->sd)) {
         set_sockerrno(sock_errno);
         return (snp->status = VERROR);
      }
      /* reset timeout for initial connect */
      snp->to = time(NULL) + TIMEOUT;
   }  /* end if (snp->sd... */
   /* prepare socket address struct */
   memset((char *) &addr, 0, sizeof(addr));
   addr.sin_addr.s_addr = snp->ip;
   addr.sin_family = AF_INET;
   addr.sin_port = htons(snp->port);
   /* check connection to addr */
   if (connect(snp->sd, (struct sockaddr *) &addr, len)) {
      ecode = sock_errno;
      /* check timeout if waiting for connect */
      if (sock_connected(ecode)) ecode = 0;
      else if (sock_connecting(ecode)) {
         if (difftime(time(NULL), snp->to) > 0) {
            return (snp->status = VETIMEOUT);
         } else return (snp->status = VEWAITING);
      }
      set_sockerrno(sock_errno);
      return (snp->status = VERROR);
   }  /* end if (connect... */

   return (snp->status = VEOK);
}  /* end request_connect() */

static word16 opreq2opcode(word16 opreq)
{
   switch (opreq) {
      case REQ_LAST_NEOGEN: return OP_GET_BLOCK;
      case REQ_VAL_BLOCK: return OP_GET_BLOCK;
      case REQ_VAL_TFILE: return OP_GET_TFILE;
      default: return opreq;
   }
}

/**
 * Network communication protocol for requesting.
 * @param snp Pointer to a SNODE
 * @returns Status of request, as integer.
 * @retval VEWAITING on waiting for data
 * @retval VETIMEOUT on socket timeout
 * @retval VEOK on protocol completion
 * @retval VERROR on socket shutdown and/or error
 * @retval VEBAD on pkt protocol violation
*/
int node_request(SNODE *snp)
{
OP_RESTART:
   /* check stage of communication */
   switch (snp->opcode) {
      case OP_NULL: {
         /* check connection wait for initial connect */
         if (snp->iowait == IO_CONN) {
            if (request_connect(snp)) break;
            /* prepare HELLO packet with initial handshake IDs */
            snp->id1 = rand16();
            snp->id2 = 0;
            prep_pkt(snp, OP_HELLO);
            /* update socket operation type */
            snp->iowait = IO_SEND;
         }
         /* send OP_HELLO packet */
         if (send_pkt(snp)) break;
         /* update socket operation type -- wait for recv */
         snp->iowait = IO_RECV;
         snp->status = VEWAITING;
         break;
      } /* end case OP_NULL */
      case OP_HELLO: {
         /* recv OP_HELLO packet */
         if (recv_pkt(snp)) break;
         /* check initial handshake protocol */
         if (snp->opcode != OP_HELLO_ACK) {
            set_errno(EMCM_NOHELLOACK);
            snp->status = VEBAD;
            break;
         }
         /* save second handshake ID */
         snp->id2 = get16(snp->pkt.id2);
         /* check request type for additional packet io requirements */
         switch (snp->opreq) {
            case REQ_LAST_NEOGEN: {
               /* derive last neo-genesis from advertsied cblock */
               put64(snp->io, snp->pkt.cblock);
               snp->io[0] = 0;
            }  /* fallthrough */
            case REQ_VAL_BLOCK: /* fallthrough */
            case OP_GET_BLOCK: /* fallthrough */
            case OP_TF: memcpy(snp->pkt.blocknum, snp->io, 8); break;
         }
         /* prepare request operation */
         prep_pkt(snp, opreq2opcode(snp->opreq));
         /* update socket operation type */
         snp->iowait = IO_SEND;
      }  /* fallthrough -- end case OP_HELLO */
      case OP_HELLO_ACK: {
         /* send request packet */
         if (send_pkt(snp)) break;
         /* check special operations that only send */
         switch (snp->opcode) {
            case OP_MBLOCK: goto OP_RESTART;
            default: break;
         }
         /* update socket operation type -- wait for recv */
         snp->iowait = IO_RECV;
         snp->status = VEWAITING;
         break;
      }  /* end case OP_HELLO_ACK */
      case OP_GET_IPL: {
         /* receive request packet */
         if (recv_pkt(snp)) break;
         /* check response opcode */
         if (snp->opcode != OP_SEND_IPL) {
            set_errno(EMCM_OPINVAL);
            snp->status = VEBAD;
            break;
         }
         break;
      }  /* end case OP_GET_IPL */
      case OP_GET_BLOCK: /* fallthrough */
      case OP_GET_TFILE: /* fallthrough */
      case OP_GET_CBLOCK: /* fallthrough */
      case OP_TF: /* fallthrough */
      case OP_SEND_FILE: {
         /* receive file -- recv_file() checks opcode */
         snp->status = recv_file(snp);
         /* if (snp->status) break; */
         break;
      }  /* end case OP_TF */
      default: {
         set_errno(EMCM_OPCODE);
         snp->status = VERROR;
      }  /* end default */
   }  /* end switch (snp->opcode) */

   /* check protocol status */
   if (snp->status != VEWAITING) {
      /* close socket, set iowait done, remove timeout */
      sock_close(snp->sd);
      snp->sd = INVALID_SOCKET;
      snp->iowait = IO_DONE;
      snp->to = 0;
      /* check naughty peers -- pinklist */
      if (snp->status == VEBAD2 || snp->status == VEBAD) {
         if (snp->status == VEBAD2) epinklist(snp->ip);
         pinklist(snp->ip);
      }
   }

   /* return resulting status */
   return snp->status;
}  /* end node_request() */

/* end include guard */
#endif
