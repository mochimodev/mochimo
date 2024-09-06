/**
 * @private
 * @headerfile protocol.h <protocol.h>
 * @copyright Adequate Systems LLC, 2018-2024. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_PROTOCOL_C
#define MOCHIMO_PROTOCOL_C


#include "protocol.h"

/* internal support */
#include "peer.h"

/* external support */
#include "crc16.h"

word32 Ntimeouts;       /* number of comm timeouts */
word32 Nerrors;         /* number of comm errors */
word32 Ndrops;          /* number of receive violations */
word32 Nrecvs;          /* number of receive errors */
word32 Nsends;          /* number of send errors */

/* Current weight for PDU send protocol */
word8 Weight[32] = { 0 };
/* Previous block hash for PDU send protocol */
word8 Prevhash[32] = { 0 };
/* Current block hash for PDU send protocol */
word8 Cblockhash[32] = { 0 };
/* Current block number for PDU send protocol */
word8 Cblocknum[8] = { 0 };
/* Capability bits for PDU send protocol */
word8 Cbits = 0;

/* inline helper function for checking recv()/send() */
static inline int node__check(int retval)
{
   int ecode;

   /* check socket operation error */
   if (retval == (-1)) {
      ecode = socket_errno;
      /* check idle socket error */
      if (socket_is_waiting(ecode)) return VEWAITING;
      /* pass "socket errno" to "errno" */
      set_sockerrno(ecode);
      return VERROR;
   }

   /* check socket shutdown */
   if (retval == 0) {
   #ifdef _WIN32
      set_alterrno(WSAESHUTDOWN);
   #else
      set_errno(ESHUTDOWN);
   #endif
      return VERROR;
   }

   return VEOK;
} /* end node__check() */

/* inline helper function for closing socket connections */
static inline void node__close_connection(CONNECTION *cp)
{
   if (cp->pollfd.fd != INVALID_SOCKET) {
      closesocket(cp->pollfd.fd);
      cp->pollfd.fd = INVALID_SOCKET;
   }
}

/**
 * Create auxiliary CONNECTION data for an incoming NODE connection.
 * @param cp Pointer to incoming CONNECTION
 * @param addrp Pointer to a sockaddr struct making the connection
 * @param len Length of the sockaddr struct
 * @return (int) value representing operation result
 * @retval VEOK on successful connection
 * @retval VERROR on internal error; check errno for details
 * @retval VEBAD on pinklisted peer
*/
int node_accept(CONNECTION *cp, struct sockaddr *addrp, socklen_t len)
{
   int ecode = VERROR;
   word32 peer;

   /* only IPv4 support for now... */
   if (addrp->sa_family != AF_INET) {
      set_errno(EAFNOSUPPORT);
      goto ERROR_CLEANUP;
   }

   /* check pinklist */
   peer = ((struct sockaddr_in *) addrp)->sin_addr.s_addr;
   if (pinklisted(peer)) goto DROP_CLEANUP;

   /* create NODE for auxiliary CONNECTION data */
   cp->data = node_create(addrp, len);
   if (cp->data == NULL) goto ERROR_CLEANUP;

   return VEOK;

   /* error handling / cleanup */
DROP_CLEANUP:
   ecode = VEBAD;
ERROR_CLEANUP:
   node__close_connection(cp);

   return ecode;
}  /* end node_accept() */

/**
 * Create a NODE for connection to the Mochimo Network.
 * To prevent a memory leak, use node_destroy() before discarding.
 * @param ptr Pointer for data (or FILE) to be held by the NODE
 * @param ptrsz Size of the pointer held by the NODE, or 0 for (FILE *)
 * @param addrp Pointer to a sockaddr struct making the connection
 * @param len Length of the sockaddr struct
 * @return (int) value representing operation result
 * @retval VEWAITING on waiting to connect
 * @retval VETIMEOUT on connection timeout
 * @retval VEOK on successful connection
 * @retval VERROR on internal error; Check errno for details
*/
NODE *node_create(struct sockaddr *addrp, socklen_t len)
{
   NODE *np;

   /* only IPv4 support for now... */
   if (addrp->sa_family != AF_INET) {
      set_errno(EAFNOSUPPORT);
      return NULL;
   }

   /* allocate space for NODE */
   np = malloc(sizeof(NODE));
   if (np == NULL) return NULL;
   /* clear NODE data and set provided address */
   memset(np, 0, sizeof(NODE));
   memcpy(&np->addr, addrp, len);

   return np;
}  /* end node_create() */

/**
 * Destroy a NODE and free any held data.
 * @param np Pointer to a NODE
 */
void node_destroy(NODE *np)
{
   /* check NODE parameter */
   if (np == NULL) return;
   /* check NODE data fields */
   if (np->fp) fclose(np->fp);
   if (np->mp) memfree(np->mp);

   /* deallocate NODE */
   free(np);
}  /* end node_destroy() */

/**
 * Prepare a PDU for a NODE with a given operation code.
 * Includes buffering of any data held by the NODE.
 * @param np Pointer to a NODE
 * @param opcode Operation code for the PDU
*/
void node_prepare(NODE *np, word16 opcode)
{
   PDU *pdu = &(np->pdu);
   size_t len = 0;
   int errnum;

   /* determine buffer protocol */
   switch (opcode) {
      case OP_NACK: {
         errnum = errno;
         /* OP_NACK includes detailed error information.
          * The OP_NACK buffer format:
          *    [8 byte value][32 byte error][variable length description]
          * 8 byte reference value; block number, MDST number, etc.
          * 32 byte error name; constant null-terminated string for client
          * Variable length error decription; null-terminated string
          *    for developer debugging or additional client information.
          */

         /* set necessary zero fill */
         memset(pdu->buffer, 0, 8 + 32);
         /* error number is an arguably useless system dependant value,
          * but is used as example filler for now...
          */
         put32(pdu->buffer, (word32) errnum);
         /* MCM errnum name (e.g. "EMCM_MADDR") */
         mcm_strerrorname(errnum, (char *) pdu->buffer + 8, 32);
         /* MCM errnum description (e.g. "Bad miner address") */
         mcm_strerror(errnum, (char *) pdu->buffer + 8 + 32, 256);
         /* get total length of NACK */
         len = 8 + 32 + strlen((char *) pdu->buffer + 40) + 1;
         put16(pdu->len, (word16) len);
         break;
      }  /* end case OP_NACK */
      default: {
         /* no handshake buffer data */
         if (opcode < FIRST_OP) break;
         /* check buffer source data */
         if (np->fp) {
            /* read FILE data into buffer -- check result */
            len = fread(pdu->buffer, 1, sizeof(pdu->buffer), np->fp);
            /* place len result in PDU and check EOF (or error) */
            put16(pdu->len, (word16) len);
            if (len < sizeof(pdu->buffer)) {
               if (ferror(np->fp)) {
                  node_prepare(np, OP_NACK);
                  return;
               }
               /* close FILE pointer on EOF */
               fclose(np->fp);
               np->fp = NULL;
            }
         } else if (np->mp) {
            /* read MEM data into buffer*/
            len = memread(pdu->buffer, 1, sizeof(pdu->buffer), np->mp);
            /* place len result in PDU and check EOF */
            put16(pdu->len, (word16) len);
            if (len < sizeof(pdu->buffer)) {
               /* close MEM pointer on EOF */
               memfree(np->mp);
               np->mp = NULL;
            }
         }
      }  /* end default */
   }  /* end switch (opcode) */

   /* fill PDU with relevant information... */
   pdu->version[0] = PVERSION;
   pdu->version[1] = Cbits;
   put16(pdu->network, TXNETWORK);
   put16(pdu->id1, np->id1);
   put16(pdu->id2, np->id2);
   put16(pdu->opcode, opcode);
   put64(pdu->cblock, Cblocknum);
   put64(pdu->blocknum, np->bnum);
   memcpy(pdu->cblockhash, Cblockhash, HASHLEN);
   memcpy(pdu->pblockhash, Prevhash, HASHLEN);
   /* ... but, do not overwrite TX ip map */
   if (opcode != OP_TX) memcpy(pdu->weight, Weight, HASHLEN);

   /* compute packet crc16 checksum -- add trailer */
   len = (word16) get16(pdu->len) + (pdu->buffer - (word8 *) pdu);
   put16(pdu->crc16, crc16(pdu, len));
   put16(pdu->trailer, TXEOT);
}  /* end node_prepare() */

/**
 * Receive PDU data from a NODE using the Mochimo Network Protocol.
 * @param cp Pointer to CONNECTION
 * @return (int) value representing operation result
 * @retval VEBAD on protocol violation; check errno for details
 * @retval VERROR on internal error; check errno for details
 * @retval VEOK on successful receive
 * @retval VEWAITING on waiting for receive buffer
 * @exception errno=EMCM_PKTCRC Packet CRC16 checksum error
 * @exception errno=EMCM_PKTNET Packet network version error
 * @exception errno=EMCM_PKTTLR Packet trailer error
 * @exception errno=EMCM_PKTIDS Packet handshake ID error
 */
int node_recv(CONNECTION *cp)
{
   NODE *np = (NODE *) cp->data;
   PDU *pdu = &(np->pdu);
   int count, len, n;

   /* derive byte length of PDU header */
   len = (int) (pdu->buffer - (word8 *) pdu);
   /* loop until PDU header is recv'd */
   for (n = np->bytes; n < len; n += count) {
      count = recv(cp->pollfd.fd, (word8 *) pdu + n, len - n, 0);
      np->status = node__check(count);
      if (np->status != VEOK) return np->status;
      /* update timeout and count bytes */
      cp->to = time(NULL) + PROTOCOL_TIMEOUT;
      np->bytes += count;
   }  /* end for (n... */

   /* adjust length to include buffer (if any) */
   if (get16(pdu->len) > 0) {
      len += (int) get16(pdu->len);
      /* loop until PDU buffer is recv'd (as above with larger len) */
      for (n = np->bytes; n < len; n += count) {
         count = recv(cp->pollfd.fd, (word8 *) pdu + n, len - n, 0);
         np->status = node__check(count);
         if (np->status != VEOK) return np->status;
         /* update timeout and count bytes */
         cp->to = time(NULL) + PROTOCOL_TIMEOUT;
         np->bytes += count;
      }  /* end for (n... */
   }

   /* finally, loop until PDU trailer is recv'd -- fixed length */
   for (n = np->bytes - len; n < 4; n += count) {
      count = recv(cp->pollfd.fd, (word8 *) pdu->crc16 + n, 4 - n, 0);
      np->status = node__check(count);
      if (np->status != VEOK) return np->status;
      /* update timeout and count bytes */
      cp->to = time(NULL) + PROTOCOL_TIMEOUT;
      np->bytes += count;
   }  /* end for (n... */

   /* whole PDU recv'd -- set oplast and reset bytes */
   np->oplast = get16(np->pdu.opcode);
   np->bytes = 0;

   /* check crc16 checksum, network version, and trailer */
   if (get16(pdu->crc16) != crc16(pdu, len)) {
      set_errno(EMCM_PKTCRC);
      np->status = VEBAD;
      return VEBAD;
   }
   if (get16(pdu->network) != TXNETWORK) {
      set_errno(EMCM_PKTNET);
      np->status = VEBAD;
      return VEBAD;
   }
   if (get16(pdu->trailer) != TXEOT) {
      set_errno(EMCM_PKTTLR);
      np->status = VEBAD;
      return VEBAD;
   }
   /* check handshake IDs on all operations (except during handshake) */
   if (np->oplast >= FIRST_OP) {
      if (np->id1 != get16(pdu->id1) || np->id2 != get16(pdu->id2)) {
         set_errno(EMCM_PKTIDS);
         np->status = VEBAD;
         return VEBAD;
      }
   }

   /* PDU recv'd */
   Nrecvs++;
   return VEOK;
}  /* end node_recv() */

/**
 * Send PDU data to a NODE using the Mochimo Network Protocol.
 * @param cp Pointer to CONNECTION
 * @return (int) value representing operation result
 * @retval VEBAD on protocol violation; check errno for details
 * @retval VERROR on internal error; check errno for details
 * @retval VEOK on successful send
 * @retval VEWAITING on waiting for send buffer
 */
int node_send(CONNECTION *cp)
{
   NODE *np = (NODE *) cp->data;
   PDU *pdu = &(np->pdu);
   int count, len, n;

   /* derive byte length of PDU header + buffer (pdu.len) */
   len = (int) get16(pdu->len) + (pdu->buffer - (word8 *) pdu);
   /* loop until PDU header and buffer is sent */
   for (n = np->bytes; n < len; n += count) {
      count = recv(cp->pollfd.fd, (word8 *) pdu + n, len - n, 0);
      np->status = node__check(count);
      if (np->status != VEOK) return np->status;
      /* update timeout and count bytes */
      cp->to = time(NULL) + PROTOCOL_TIMEOUT;
      np->bytes += count;
   }  /* end for (n... */

   /* loop until PDU trailer is sent */
   for (n = np->bytes - len; n < 4; n += count) {
      count = recv(cp->pollfd.fd, (word8 *) pdu->crc16 + n, 4 - n, 0);
      np->status = node__check(count);
      if (np->status != VEOK) return np->status;
      /* update timeout and count bytes */
      cp->to = time(NULL) + PROTOCOL_TIMEOUT;
      np->bytes += count;
   }  /* end for (n... */

   /* whole PDU sent -- set oplast and reset bytes */
   np->oplast = get16(np->pdu.opcode);
   np->bytes = 0;

   /* PDU sent */
   Nsends++;
   return VEOK;
}  /* end node_send() */

/**
 * Protocol for processing NODE operations on the Mochimo Network Protocol.
 * NOTE: return value is also placed in np->status
 * @param cp Pointer to a CONNECTION (containing NODE data)
 * @returns (int) value representing the operation result
 * @retval VEWAITING on waiting for data; check cp->pollfd.events for type
 * @retval VETIMEOUT on communication timeout
 * @retval VEOK on successful communication
 * @retval VERROR on internal error; check errno for details
 * @retval VEBAD on protocol violation; check errno for details
 * @exception errno=EMCMOPCODE Unexpected operation code
 * @exception errno=EMCMOPRECV Received unexpected operation code
*/
int node_tranceive__incoming(CONNECTION *cp)
{
   NODE *np = (NODE *) cp->data;

LOOPBACK:
   /* check last received OPCODE for next processing step */
   switch (np->oplast) {
   /* HANDSHAKE PROTOCOL OPERATIONS */
      case OP_NULL: {
         /* receive/check OP_HELLO */
         if (node_recv(cp) != VEOK) break;
         if (np->oplast != OP_HELLO) {
            set_errno(EMCM_OPHELLO);
            np->status = VEBAD;
            break;
         }
         /* prepare "acknowledgement" with secondary handshake ID */
         np->id1 = get16(np->pdu.id1);
         np->id2 = rand16();
         node_prepare(np, OP_HELLO_ACK);
         /* update event type for send */
         cp->pollfd.events = POLLOUT;
      }  /* fallthrough -- end OP_NULL */
      case OP_HELLO: {
         /* send OP_HELLO_ACK */
         if (node_send(cp) != VEOK) break;
         /* update event type for receive */
         cp->pollfd.events = POLLIN;
         np->status = VEWAITING;
         break;
      }  /* end case OP_HELLO (or OP_NULL) */

   /* INITIAL REQUEST OPERATION */
      case OP_HELLO_ACK: {
         /* receive request packet */
         if (node_recv(cp) != VEOK) break;
         /* recv'd opcode MUST be a "valid" operation code */
         /* NOTE: recv'd opcode MUST be checked here */
         if (np->oplast < FIRST_OP || np->oplast > LAST_OP) {
            set_errno(EMCM_OPNVAL);
            np->status = VEBAD;
         }
         /* re-check switch */
         goto LOOPBACK;
      }  /* end case OP_HELLO_ACK */

   /* BROADCAST OPERATIONS */
      case OP_TX: /* fallthrough */
      case OP_FOUND: {
         /* no further network operations */
         break;
      }

   /* SINGLE/MULTI RESPONSE OPERATION CHECKS */
      case OP_GET_BLOCK: /* fallthrough */
      case OP_GET_IPL: /* fallthrough */
      case OP_GET_TFILE: /* fallthrough */
      case OP_BALANCE: /* fallthrough */
      case OP_HASH: /* fallthrough */
      case OP_TF: {
         /* response data required to proceed... */
         if (np->fp == NULL && np->mp == NULL) {
            np->status = VEOK;
            /* ... VEOK indicates no further network operations which, at
             * least in this case, allows the server to pass the CONNECTION
             * to the designated cleanup function for processing before
             * being returned to the server to finish network operations
             */
            break;
         }
      }  /* fallthrough */

   /* CONTINUE RESPONSE OPERATION */
      case OP_SEND_FILE: {
         do {
            /* only prepare after previous successful operation */
            if (np->status == VEOK) node_prepare(np, OP_SEND_FILE);
            /* send PDU and check for EOF condition */
         } while (node_send(cp) == VEOK && (np->fp || np->mp));
         break;
      }  /* end case OP_SEND_FILE */

   /* INVALID REQUEST OPERATION */
      default: {
         set_errno(EMCM_OPCODE);
         np->status = VERROR;
      }  /* end default */
   }  /* end switch (np->oplast) */

   /* return resulting status */
   return np->status;
}  /* end node_tranceive__incoming() */

/**
 * Protocol for requesting NODE operations on the Mochimo Network Protocol.
 * Assumes a successfully connected CONNECTION.
 * @param np Pointer to a NODE
 * @returns (int) value representing the operation result
 * @retval VEWAITING on waiting for data; check np->iowait data type
 * @retval VETIMEOUT on communication timeout
 * @retval VEOK on successful communication
 * @retval VERROR on internal error; check errno for details
 * @retval VEBAD on protocol violation; check errno for details
 * @exception errno=EMCMOPCODE Unexpected operation code
 * @exception errno=EMCMOPRECV Received unexpected operation code
*/
int node_tranceive__outgoing(CONNECTION *cp)
{
   NODE *np = (NODE *) cp->data;
   PDU *pdu = &(np->pdu);
   size_t len;

   /* check last received OPCODE for next processing step */
   switch (np->oplast) {
      /* HANDSHAKE OPERATIONS */
      case OP_NULL: {
         /* prepare "hello" with initial handshake ID */
         np->id1 = np->id2 = rand16();
         node_prepare(np, OP_HELLO);
         /* send OP_HELLO */
         if (node_send(cp) != VEOK) break;
         /* update event type for receive */
         cp->pollfd.events = POLLIN;
         np->status = VEWAITING;
         break;
      } /* end case OP_NULL */
      case OP_HELLO: {
         /* receive/check OP_HELLO_ACK */
         if (node_recv(cp) != VEOK) break;
         /* check initial handshake protocol */
         if (np->oplast != OP_HELLO_ACK) {
            set_errno(EMCM_OPHELLOACK);
            np->status = VEBAD;
            break;
         }
         /* save secondary handshake ID */
         np->id2 = get16(pdu->id2);
         /* prepare request operation */
         node_prepare(np, np->opreq);
         /* update event type for send */
         cp->pollfd.events = POLLOUT;
      }  /* fallthrough -- end case OP_HELLO */

   /* INITIAL REQUEST OPERATION */
      case OP_HELLO_ACK: {
         /* send request packet */
         if (node_send(cp) != VEOK) break;
         /* end connection for broadcast operations */
         if (np->opreq == OP_TX || np->opreq == OP_FOUND) {
            np->status = VEOK;
            break;
         }
         /* update event type for receive */
         cp->pollfd.events = POLLIN;
         np->status = VEWAITING;
         break;
      }  /* end case OP_HELLO_ACK */

   /* SINGLE REQUEST OPERATIONS */
      case OP_TX: break;
      case OP_FOUND: break;

   /* SINGLE REQUEST/RESPONSE OPERATIONS */
      case OP_GET_IPL: {
         /* receive request packet */
         if (node_recv(cp) != VEOK) break;
         /* check response opcode */
         if (np->oplast != OP_SEND_IPL) {
            set_errno(EMCM_OPRECV);
            np->status = VERROR;
            break;
         }
         break;
      }  /* end case OP_GET_IPL */

   /* SIMPLE FILE DOWNLOAD OPERATIONS */
      case OP_TF: /* fallthrough */
      case OP_GET_TFILE: /* fallthrough */
      case OP_GET_BLOCK: /* fallthrough */
      case OP_GET_CBLOCK: /* fallthrough */
      case OP_SEND_FILE: {
         /* receive data */
         while (node_recv(cp) == VEOK) {
            /* check recv'd opcode is OP_SEND_FILE */
            if (np->oplast != OP_SEND_FILE) {
               if (np->oplast == OP_NACK) {
                  set_errno(EMCM_PKTNACK);
                  np->status = VERROR;
               } else {
                  set_errno(EMCM_PKTOPCODE);
                  np->status = VEBAD;
               }
               /* failure condition */
               break;
            }
            /* check received length and EOF condition */
            len = get16(pdu->len);
            if (len) {
               /* check FILE buffer */
               if (np->fp) {
                  /* write buffer length to FILE pointer */
                  if (fwrite(pdu->buffer, len, 1, np->fp) != 1) {
                     np->status = VERROR;
                     break;
                  }
               } else {
                  /* ... no FILE buffer, use MEM buffer */
                  if (np->mp == NULL) memdynamic(NULL, len);
                  /* write buffer length to DATA pointer */
                  if (memwrite(pdu->buffer, len, 1, np->mp) != 1) {
                     /* ... memwrite() also checks invalid parameters */
                     np->status = VERROR;
                     break;
                  }
               }
            }
            /* check EOF condition */
            if (len < sizeof(pdu->buffer)) break;
         }  /* end while (node_recv... */
         break;
      }

   /* INVALID REQUEST OPERATION */
      default: {
         set_errno(EMCM_OPCODE);
         np->status = VERROR;
      }  /* end default */
   }  /* end switch (np->opcode) */

   /* return resulting status */
   return np->status;
}  /* end node_tranceive__outgoing() */

/**
 * Process NODE operations on the Mochimo Network Protocol.
 * @param cp Pointer to a CONNECTION (containing NODE data)
 * @return (int) value representing the operation result
 * @retval VEOK on successful communication
 * @retval VERROR on internal error; check errno for details
 * @retval VEBAD on protocol violation; check errno for details
 * @exception errno=EMCM_OPCODE Unexpected operation code
 * @exception errno=EMCM_OPRECV Unexpected response operation code
 */
int node_tranceive(CONNECTION *cp)
{
   NODE *np;

   /* check parameters */
   if (cp == NULL || cp->data == NULL) {
      if (cp) node__close_connection(cp);
      set_errno(EINVAL);
      return VERROR;
   }

   /* dereference NODE data */
   np = (NODE *) cp->data;

   /* "opreq" field indicates the request is outgoing... */
   if (np->opreq) node_tranceive__outgoing(cp);
   /* ... otherwise, the request is incoming */
   else node_tranceive__incoming(cp);

   /* set poll idle, if not waiting for more data... */
   if (np->status != VEWAITING) cp->pollfd.events = 0;

   /* increment stat counters */
   if (np->status == VETIMEOUT) Ntimeouts++;
   if (np->status == VERROR) Nerrors++;
   if (np->status == VEBAD) Ndrops++;

   return np->status;
}  /* end node_tranceive() */

/* end include guard */
#endif
