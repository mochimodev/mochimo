/**
 * @private
 * @headerfile packet.h <packet.h>
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_PACKET_C
#define MOCHIMO_PACKET_C


#include "protocol.h"

/* internal support */
#include "peer.h"
#include "error.h"

/* external support */
#include <string.h>
#include "extlib.h"
#include "crc16.h"

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
 * Prepare a packet of SNODE with protocol data.
 * @param snp Pointer to SNODE containing packet to prepare
 * @param opcode Operation code of packet
*/
void prep_pkt(SNODE *snp, word16 opcode)
{
   word16 len;

   /* fill packet with relevant information... */
   snp->pkt.version[0] = PVERSION;
   snp->pkt.version[1] = Cbits;
   put16(snp->pkt.network, TXNETWORK);
   put16(snp->pkt.trailer, TXEOT);
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

   /* compute packet crc16 checksum */
   put16(snp->pkt.crc16, crc16(&(snp->pkt), PKTCRC_INLEN(len)));
}  /* end prep_pkt() */

/**
 * @private
 * Obtain the next recv() length, in bytes. For compatibility between
 * pversion 4, C_VPDU capable pversion 4, and pversion 5 onwards, we
 * require the information in the header of a packet. Therefore, to
 * provide appropriate breakpoints for determining the amount of data
 * to receive, the result of this function may be PKTHDRLEN (initially),
 * then either PKTBUFFLEN_OLD for pversion 4, or the 16-bit value of
 * pkt->len for C_VPDU capable pversion 4, and pversion 5 onwards.
 * @param snp Pointer to SNODE
 * @returns (int) length, in bytes, of next recv()
 * @todo Remove protocol version 4 compatibility after v3.0
*/
static int recv_len(SNODE *snp)
{
   /* determine length of data to recv */
   if (snp->bytes >= PKTHDRLEN) {
      /* check protocol version for implied VPDU capability */
      /** @todo adjust after v3.0 */
      if (PKT_HAS_C_VPDU(&(snp->pkt))) {
         return get16(snp->pkt.len);
      } else return PKTBUFFLEN_OLD;
   } else return PKTHDRLEN;
}

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
   int ecode, count, len, n;

   /* init */
   pkt = &(snp->pkt);

   /* receive variable PDU sizes into &pkt[n] */
   for (n = snp->bytes, len = recv_len(snp); n < len; ) {
      if (len > PKTBUFFLEN) goto FAIL_OVERFLOW;
      count = recv(snp->sd, ((char *) pkt) + n, len - n, 0);
      switch (count) {
         case (-1):
            ecode = sock_errno;
            /* check timeout if waiting for data */
            if (sock_err_is_waiting(ecode)) {
               if (difftime(time(NULL), snp->to) > 0) {
                  return (snp->status = VETIMEOUT);
               } else return (snp->status = VEWAITING);
            } else goto FAIL_ECODE;    /* socket error ocurred */
         case 0: goto FAIL_SHUTDOWN;   /* socket was shutdown */
         default:
            /* reset timeout, add recv'd bytes, update length */
            snp->to = time(NULL) + TIMEOUT;
            snp->bytes = n = n + count;
            len = recv_len(snp);
      }  /* end switch (count... */
   }  /* end for (n = snp... */

   /* full packet received: set opcode and reset bytes */
   snp->opcode = get16(snp->pkt.opcode);
   snp->bytes = 0;

   /* check crc16 checksum, network version, and trailer */
   if (get16(pkt->crc16) != crc16(pkt, PKTCRC_INLEN(len))) goto BAD_CRC;
   if (get16(pkt->network) != TXNETWORK) goto BAD_NET;
   if (get16(pkt->trailer) != TXEOT) goto BAD_TLR;
   /* check handshake IDs on all operations (except during handshake) */
   if (snp->opcode >= FIRST_OP) {
      if (snp->id1 != get16(pkt->id1)) goto BAD_IDS;
      if (snp->id2 != get16(pkt->id2)) goto BAD_IDS;
   }

   /* on success, flag VPDU capable peer */
   /** @todo adjust after v3.0 */
   if (PKT_HAS_C_VPDU(&(snp->pkt))) snp->c_vpdu = 1;

   /* success -- increment recv's */
   Nrecvs++;
   return (snp->status = VEOK);

/* error handling */
FAIL_OVERFLOW: errno = EOVERFLOW; goto FAIL;
FAIL_SHUTDOWN: errno = ECONNABORTED; goto FAIL;
FAIL_ECODE: errno = resolve_wsa_conflicts(ecode);
FAIL:
   Nrecverrs++;
   return (snp->status = VERROR);

/* protocol violation handling */
BAD_CRC: errno = EMCM_PKTCRC; goto BAD;
BAD_NET: errno = EMCM_PKTNET; goto BAD;
BAD_TLR: errno = EMCM_PKTTLR; goto BAD;
BAD_IDS: errno = EMCM_PKTIDS;
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
      if (!PKT_HAS_C_VPDU(&(snp->pkt))) {
         if (len < PKTBUFFLEN_OLD) break;
      } else if (len < PKTBUFFLEN) break;
   } /* end while(recv_pkt()) */

   /* snp->status is set during recv_pkt() */
   return snp->status;

/* error handling */
FAIL_NACK: errno = EMCM_PKTNACK;
FAIL: return (snp->status = VERROR);

/* protocol violation handling */
BAD_OPCODE: errno = EMCM_PKTOPCODE; return (snp->status = VEBAD);
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
static int send_len(SNODE *snp)
{
   /* determine length of data to send() */
   /** @todo adjust logic after v3.0 */
   if (get16(snp->pkt.opcode) == OP_HELLO || !snp->c_vpdu) {
      return PKTBUFFLEN_OLD;
   } else return PKTHDRLEN;
}

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
   PKT *pkt;
   int ecode, count, len, n;

   pkt = &(snp->pkt);

   /* send PDUs of varying size and capabilities */
   for (n = snp->bytes, len = send_len(snp); n < len; ) {
      count = send(snp->sd, ((char *) pkt) + n, len - n, 0);
      switch (count) {
         case (-1):
            ecode = sock_errno;
            /* check timeout if waiting for data */
            if (sock_err_is_waiting(ecode)) {
               if (difftime(time(NULL), snp->to) > 0) {
                  return (snp->status = VETIMEOUT);
               } else return (snp->status = VEWAITING);
            } else goto FAIL_ECODE;    /* socket error ocurred */
         case 0: goto FAIL_SHUTDOWN;   /* socket was shutdown */
         default:
            /* reset timeout, add recv'd bytes, update length */
            snp->to = time(NULL) + TIMEOUT;
            snp->bytes = n = n + count;
            len = send_len(snp);
      }  /* end switch (count... */
   }  /* end for (n = snp... */

   /* full packet received: set opcode and reset bytes */
   snp->opcode = get16(snp->pkt.opcode);
   snp->bytes = 0;

   /* success -- increment send's */
   Nsends++;  /* requires atomic operation */
   return (snp->status = VEOK);

FAIL_SHUTDOWN: errno = ECONNABORTED; goto FAIL;
FAIL_ECODE: errno = resolve_wsa_conflicts(ecode);
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
int receive_protocol(SNODE *snp)
{
OP_RESTART:
   /* check stage of communication */
   switch (snp->opcode) {
      case OP_NULL: {
         /* receive/check OP_HELLO packet */
         if (recv_pkt(snp)) break;
         if (snp->opcode != OP_HELLO) {
            errno = EMCM_NOHELLO;
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
            errno = EMCM_OPINVAL;
            snp->status = VERROR;
            break;
         }
         /* restart switch block on success */
         goto OP_RESTART;
      }  /* end case OP_HELLO_ACK */
      default: {
         errno = EMCM_OPCODE;
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

   /* clear errno on non-error status */
   if (snp->status <= VEOK) errno = 0;
   return snp->status;
}  /* end receive_protocol() */

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
         errno = resolve_wsa_conflicts(sock_errno);
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
      if (sock_err_is_success(ecode)) ecode = 0;
      else if (sock_err_is_waiting(ecode)) {
         if (difftime(time(NULL), snp->to) > 0) {
            return (snp->status = VETIMEOUT);
         } else return (snp->status = VEWAITING);
      }
      errno = resolve_wsa_conflicts(sock_errno);
      return (snp->status = VERROR);
   }  /* end if (connect... */

   errno = 0;
   return (snp->status = VEOK);
}  /* end request_connect() */

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
int request_protocol(SNODE *snp)
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
            errno = EMCM_NOHELLOACK;
            snp->status = VEBAD;
            break;
         }
         /* check request type for additional packet io requirements */
         switch (snp->opreq) {
            case REQ_VAL_BLOCK: /* fallthrough */
            case OP_GET_BLOCK: /* fallthrough */
            case OP_TF: memcpy(snp->pkt.blocknum, snp->io, 8); break;
         }
         /* save second handshake ID */
         snp->id2 = get16(snp->pkt.id2);
         /* prepare request operation */
         prep_pkt(snp, snp->opreq);
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
            errno = EMCM_OPINVAL;
            snp->status = VEBAD;
            break;
         }
         break;
      }  /* end case OP_GET_IPL */
      case REQ_VAL_BLOCK: /* fallthrough */
      case REQ_VAL_TFILE: /* fallthrough */
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
         errno = EMCM_OPCODE;
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

   /* clear errno on non-error status */
   if (snp->status <= VEOK) errno = 0;
   return snp->status;
}  /* end request_protocol() */

/* end include guard */
#endif
