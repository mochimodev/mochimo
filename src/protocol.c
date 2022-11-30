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
#include "block.h"
#include "chain.h"
#include "error.h"
#include "ledger.h"
#include "peer.h"

/* external support */
#include "crc16.h"
#include "extlib.h"
#include "extmath.h"
#include <string.h>

#ifdef _WIN32
   #define set_sockerrno(e)   set_alterrno(e)

#else
   #define set_sockerrno(e)   set_errno(e)

#endif

#define PKT_IS_PV5(p)   ( (p)->version[0] == 5 )

/** Lifetime balance requests processed */
unsigned Nbalance = 0;
/** Lifetime hash requests processed */
unsigned Nhashes = 0;
/** Lifetime IP list requests processed */
unsigned Niplist = 0;
/** Lifetime Negative Acknowledgements */
unsigned Nnacks = 0;
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
 * Initialize a packet of NODE with protocol data.
 * @param np Pointer to NODE
 * @param opcode Operation code of packet
*/
void init_pkt(NODE *np, word16 opcode)
{
   word16 len;

   /* fill packet with relevant information... */
   np->pkt.version[0] = PVERSION;
   np->pkt.version[1] = Cbits | C_VPDU;
   put16(np->pkt.network, TXNETWORK);
   put16(np->pkt.id1, np->id1);
   put16(np->pkt.id2, np->id2);
   put16(np->pkt.opcode, opcode);
   put64(np->pkt.cblock, Cblocknum);
   memcpy(np->pkt.cblockhash, Cblockhash, HASHLEN);
   memcpy(np->pkt.pblockhash, Prevhash, HASHLEN);
   /* ... but, do not overwrite TX ip map */
   if (opcode != OP_TX) memcpy(np->pkt.weight, Weight, HASHLEN);

   /* store (actual) packet buffer length for CRC hash */
   len = get16(np->pkt.len);

   /************************************/
   /* PROTOCOL VERSION 4 COMPATIBILITY */

   /* check for VPDU capability */
   /** @todo adjust after v3.0 */
   if (!np->c_vpdu) {
      /* protocol version 4 packets use a fixed length PDU */
      if (len < PKTBUFFLEN_OLD) len = PKTBUFFLEN_OLD;
      /* opcode specific checks */
      switch (get16(np->pkt.opcode)) {
         /* for initial compatibility, set len param to fixed length PDU */
         case OP_HELLO: put16(np->pkt.len, PKTBUFFLEN_OLD); break;
         /* for peerlist compatibility, some opcodes MUST have ZERO len */
         case OP_TX: put16(np->pkt.len, 0); break;
         case OP_FOUND: put16(np->pkt.len, 0); break;
         case OP_GET_IPL: put16(np->pkt.len, 0); break;
         case OP_GET_TFILE: put16(np->pkt.len, 0); break;
      }
   }

   /* END PROTOCOL VERSION 4 COMPATIBILITY */
   /****************************************/

   /* compute packet crc16 checksum -- add trailer */
   put16(np->pkt.crc16, crc16(&(np->pkt), PKTCRC_INLEN(len)));
   put16(np->pkt.trailer, TXEOT);
}  /* end init_pkt() */

/**
 * Initialize a packet of NODE with protocol data and OP_NACK.
 * OP_NACK refers to a Negative Acknowledgment operation (usually
 * sent in response to a request operation) identifying that the
 * request was received successfully, but it contained an error.
 * @param np Pointer to NODE
*/
void init_nack(NODE *np)
{
   /* initialize NACK protocol */
   put16(np->pkt.len, 0);
   init_pkt(np, OP_NACK);
   /* increment NACK counter */
   Nnacks++;
}  /* end init_nack() */

/**
 * @private
 * Obtain the next data length, in bytes. For compatibility between
 * pversion 4, C_VPDU capable pversion 4, and pversion 5 onwards, we
 * require the information in the header of a packet. This is NOT
 * immediately possible if we are initiating a connection, so OP_HELLO
 * packets will always be padded, until we can identify the capability
 * of the connection we make, by checking the next received packet.
 * NOTE: Maintaining a list of IPs identifying C_VPDU capabilities is
 * flawed because it's possible to receive connections from multiple
 * sources behind the same IP using different capabilities; be it
 * nodes, wallets, APIs, tx bots or other scripts.
 * @param np Pointer to NODE
 * @returns (int) length, in bytes, of next packet
*/
static int len_pkt(NODE *np, char **buf, int *len, int *n)
{
   /* determine position and length of next packet data */
   if (np->bytes < PKTHDRLEN) {
      /* packet header */
      *n = np->bytes;
      *len = PKTHDRLEN;
      *buf = (char *) np->pkt.version;
   } else if (np->c_vpdu) {
      *len = (int) get16(np->pkt.len);
      if (np->bytes < (PKTHDRLEN + *len)) {
         /* VPDU packet buffer */
         *n = np->bytes - PKTHDRLEN;
         *buf = (char *) np->pkt.buffer;
      } else {
         /* VPDU packet trailer */
         *n = np->bytes - (PKTHDRLEN + *len);
         *len = PKTTLRLEN;
         *buf = (char *) np->pkt.crc16;
      }
   } else if (np->bytes < (PKTHDRLEN + PKTBUFFLEN_OLD)) {
      /* !VPDU packet buffer */
      *n = np->bytes - PKTHDRLEN;
      *len = PKTBUFFLEN_OLD;
      *buf = (char *) np->pkt.buffer;
   } else {
      /* !VPDU packet trailer */
      *n = np->bytes - (PKTHDRLEN + PKTBUFFLEN_OLD);
      *len = PKTTLRLEN;
      *buf = (char *) np->pkt.crc16;
   }

   return *len - *n;
}  /* end len_pkt() */

/**
 * Receive a packet of data from a node.
 * NOTE: return value is also placed in np->status
 * @param np Pointer to a node
 * @return (int) value representing receive result
 * @retval VEWAITING on waiting for data
 * @retval VETIMEOUT on connection timeout
 * @retval VEOK on success; packet received
 * @retval VERROR on socket error
 * @retval VEBAD on protocol violation
*/
int recv_pkt(NODE *np)
{
   PKT *pkt;
   char *buf;
   int ecode, count, len, n;

   /* init */
   pkt = &(np->pkt);

   /* receive variable PDU into pkt */
   while (len_pkt(np, &buf, &len, &n)) {
      if (len > PKTBUFFLEN) goto FAIL_OVERFLOW;
      count = recv(np->sd, buf + n, len - n, 0);
      switch (count) {
         case (-1):
            ecode = sock_errno;
            /* check timeout if waiting for data */
            if (sock_waiting(ecode)) {
               if (difftime(time(NULL), np->to) > 0) {
                  return (np->status = VETIMEOUT);
               } else return (np->status = VEWAITING);
            } else goto FAIL_ECODE;    /* socket error ocurred */
         case 0: goto FAIL_SHUTDOWN;   /* socket was shutdown */
         default:
            /* reset timeout, add recv'd bytes, update length */
            np->to = time(NULL) + TIMEOUT;
            np->bytes += count;
      }  /* end switch (count... */
   }  /* end for (n = np... */

   /* full packet received: set c_vpdu, opcode and reset bytes */
   np->c_vpdu = PKT_HAS_C_VPDU(&(np->pkt));
   np->opcode = get16(np->pkt.opcode);
   np->bytes = 0;

   /* check crc16 checksum, network version, and trailer */
   len = np->c_vpdu ? get16(np->pkt.len) : PKTBUFFLEN_OLD;
   if (get16(pkt->crc16) != crc16(pkt, PKTCRC_INLEN(len))) goto BAD_CRC;
   if (get16(pkt->network) != TXNETWORK) goto BAD_NET;
   if (get16(pkt->trailer) != TXEOT) goto BAD_TLR;
   /* check handshake IDs on all operations (except during handshake) */
   if (np->opcode >= FIRST_OP) {
      if (np->id1 != get16(pkt->id1)) goto BAD_IDS;
      if (np->id2 != get16(pkt->id2)) goto BAD_IDS;
   }

   /* success -- increment recv's */
   Nrecvs++;
   return (np->status = VEOK);

/* error handling */
FAIL_OVERFLOW: set_errno(EOVERFLOW); goto FAIL;
FAIL_SHUTDOWN: set_errno(ECONNABORTED); goto FAIL;
FAIL_ECODE: set_sockerrno(ecode);
FAIL:
   Nrecverrs++;
   return (np->status = VERROR);

/* protocol violation handling */
BAD_CRC: set_errno(EMCMPKTCRC); goto BAD;
BAD_NET: set_errno(EMCMPKTNET); goto BAD;
BAD_TLR: set_errno(EMCMPKTTLR); goto BAD;
BAD_IDS: set_errno(EMCMPKTIDS);
BAD:
   Nrecvsbad++;
   return (np->status = VEBAD);
}  /* end recv_pkt() */

/**
 * Receive multiple packets of data into a file for a NODE.
 * NOTE: File data is received into a DATA pointer.
 * @param np Pointer to a NODE
 * @return (int) value representing operation result
 * @retval VEWAITING on waiting for data
 * @retval VETIMEOUT on connection timeout
 * @retval VEOK on success; pkt is complete
 * @retval VERROR on internal error
 * @retval VEBAD on protocol violation
*/
int recv_file(NODE *np)
{
   int len;

   /* receive packet of data */
   while (recv_pkt(np) == VEOK) {
      /* check recv'd opcode is OP_SEND_FILE */
      if (np->opcode == OP_NACK) goto FAIL_NACK;
      if (np->opcode != OP_SEND_FILE) goto BAD_OPCODE;
      /* check received length */
      len = get16(np->pkt.len);
      if (len) {
         /* allocate temporary file if no file allocated */
         if (np->fp == NULL) {
            np->fp = tmpfile();
            if (np->fp == NULL) goto FAIL;
         }
         /* write packet buffer to DATA pointer */
         if (fwrite(PKTBUFF(&(np->pkt)), len, 1, np->fp) != 1) goto FAIL;
      }
      /* check EOF condition -- check VPDU capability bit */
      /** @todo adjust after v3.0 */
      if (!np->c_vpdu) {
         if (len < PKTBUFFLEN_OLD) break;
      } else if (len < PKTBUFFLEN) break;
   } /* end while(recv_pkt()) */

   /* np->status is set during recv_pkt() */
   return np->status;

/* error handling */
FAIL_NACK: set_errno(EMCMPKTNACK);
FAIL: return (np->status = VERROR);

/* protocol violation handling */
BAD_OPCODE: set_errno(EMCMPKTOPCODE); return (np->status = VEBAD);
}  /* end recv_file() */

/**
 * Send a ledger balance to a requesting server node connection.
 * Non-Hashed Address search requests are converted before search.
 * NOTE: Protocol version 5 accepts Hashed Addresses and returns
 * a Ledger Transaction (LTRAN) containing Ledger data where the
 * value of @a trancode[0] is set if the address was "found".
 * @param np Pointer to server node to send balance
 * @return (int) value representing operation result
 * @retval VEWAITING on waiting for data
 * @retval VETIMEOUT on connection timeout
 * @retval VEOK on success; packet sent
 * @retval VERROR on internal error
*/
int send_balance(NODE *np)
{
   TXW *txwp;
   LTRAN *ltp;
   LENTRY *lep;

   /* check if balance was retrieved */
   if (np->iowait == IO_RECV) {
      /* check protocol version... */
      if (PKT_IS_PV5(&(np->pkt))) {
         /* ... PVERSION 5 onwards uses Hashed Address */
         ltp = (LTRAN *) np->pkt.buffer;
         if (get16(np->pkt.len) != TXADDRLEN) {
            /* unsupported address type, send NACK... */
            init_nack(np);
            goto SEND;
         }
         /* update the packet buffer data length */
         put16(np->pkt.len, sizeof(*ltp));
         /* check if ledger data exists... */
         lep = le_find(ltp->addr);
         if (lep) {
            /* return "found" and ledger balance */
            ltp->trancode[0] = 1;
            put64(ltp->amount, lep->balance);
         } else {
            /* return "not found" and zero balance */
            ltp->trancode[0] = 0;
            memset(ltp->amount, 0, sizeof(ltp->amount));
         }
      } else {
         /* ... PVERSION 4 and below uses WOTS+ (TXW) */
         txwp = (TXW *) np->pkt.buffer;
         /* initialize return values */
         memset(txwp->send_total, 0, sizeof(txwp->send_total));
         memset(txwp->change_total, 0, sizeof(txwp->change_total));
         /* look up source address in ledger */
         lep = le_findw(txwp->src_addr);
         if (lep) {
            /* insert balance and indicate address was found */
            put64(txwp->send_total, lep->balance);
            txwp->change_total[0] = 1;
         }
      }

      /* increment balance request counter */
      Nbalance++;

      /* initialize packet for sending */
      init_pkt(np, OP_SEND_BAL);
SEND: np->iowait = IO_SEND;
   }

   /* send packet with ledger balance */
   return send_pkt(np);
}  /* end send_balance() */

/**
 * Send various files to a requesting server node.
 * @param np Pointer to server node to send IP list
 * @return (int) value representing operation result
 * @retval VEWAITING on waiting for send buffer
 * @retval VETIMEOUT on connection timeout
 * @retval VEOK on success; finished
 * @retval VERROR on internal error
*/
int send_file(NODE *np)
{
   BTRAILER tft;
   long long first, count;
   FILE *fp;
   word8 *bp;
   word32 bnum[2];
   word16 len;
   char fname[FILENAME_MAX];
   char fpath[FILENAME_MAX];

   /* check if file is already allocated */
   if (np->iowait == IO_RECV) {
      bp = np->pkt.blocknum;
      /* determine file type requested */
      switch (get16(np->pkt.opcode)) {
         case OP_GET_BLOCK: {
            /* read Tfile for hash of requested block */
            if (read_tfile(&tft, bp, 1, "tfile.dat") != 1) return VERROR;
            /* get archived file name */
            bc_fqan(fname, np->pkt.blocknum, tft.bhash);
            path_join(fpath, Bcdir_opt, fname);
            /* try opening file -- VERROR if unable to send */
            np->fp = fopen(fpath, "rb");
            if (np->fp == NULL) return VERROR;
            /* ... file is open and ready for send */
            break;
         }  /* end case OP_GET_BLOCK */
         case OP_GET_TFILE: {
            /* FULL Tfile download */
            first = 0;
            count = LLONG_MAX;
            goto PREP_TFILE;
         }  /* end case OP_GET_TFILE */
         case OP_TF: {
            /* obtain parameters from packet IO blocknum */
            bnum[1] = 0;
            bnum[0] = get32(np->pkt.blocknum);
            count = get16(np->pkt.blocknum + 4);
            /* protocol version 5 introduces additional feedback... */
            if (PKT_IS_PV5(&(np->pkt))) {
               /* ... IO MUST NOT be greater than current block number,
               * and count must be non-zero up to 1000 */
               if (cmp64(bnum, Cblocknum) > 0
                  || count < 1 || count > 1000) {
                  /* unsupported IO values, send NACK... */
                  init_nack(np);
                  goto SEND;
               }
            } else if (count < 1 || count > 1000) return VERROR;
            /* seek to appropriate location */
            first = 0;
            put64(&first, bnum);
            first *= (long long) sizeof(tft);
PREP_TFILE: /* create temporary file ("wb+") -- remove buffer */
            np->fp = tmpfile();
            if (np->fp == NULL) return VERROR;
            setvbuf(np->fp, NULL, _IONBF, 0);
            /* open Tfile for copy */
            fp = fopen("tfile.dat", "rb");
            if (fp == NULL) return VERROR;
            /* seek to appropriate location */
            if (fseek64(fp, first, SEEK_SET) != 0) goto FAIL;
            /* load requested Tfile data into temporary file */
            while (count-- && fread(&tft, sizeof(tft), 1, fp)) {
               if (fwrite(&tft, sizeof(tft), 1, np->fp) != 1) goto FAIL;
            }
            break;
         }  /* end case OP_TF (or OP_GET_TFILE) */
         default: return VERROR;
      }  /* end switch (get16(np->pkt.opcode)) */

      /* rewind temp file */
      rewind(np->fp);
      /* read first chunk of data into packet buffer */
      len = np->c_vpdu ? PKTBUFFLEN : PKTBUFFLEN_OLD;
      count = fread(np->pkt.buffer, 1, len, np->fp);
      /* initialize packet for first send */
      put16(np->pkt.len, (word16) count);
      init_pkt(np, OP_SEND_FILE);
SEND: np->iowait = IO_SEND;
   }  /* end if (np->iowait == IO_RECV) */

   /* divert to send_pkt() for NACK */
   if (get16(np->pkt.opcode) == OP_NACK) return send_pkt(np);

   return send_fp(np);

/* error handling */
FAIL:
   fclose(fp);
   return VERROR;
}  /* end send_file() */

/**
 * Send multiple packets of data froma FILE pointer to a NODE.
 * @param np Pointer to a NODE containing open FILE pointer
 * @return (int) value representing operation result
 * @retval VEWAITING on waiting for send buffer
 * @retval VETIMEOUT on connection timeout
 * @retval VEOK on success; FILE pointer EOF
 * @retval VERROR on internal error
*/
int send_fp(NODE *np)
{
   size_t count, len;

   /* send packet of data */
   while (send_pkt(np) == VEOK) {
      /* check file pointer for end */
      if (np->fp == NULL) break;
      /* read next chunk of data into packet buffer */
      len = np->c_vpdu ? PKTBUFFLEN : PKTBUFFLEN_OLD;
      count = fread(np->pkt.buffer, 1, len, np->fp);
      if (count < len) {
         /* check for specific error -- close file pointer */
         if (ferror(np->fp)) np->status = VERROR;
         fclose(np->fp);
         np->fp = NULL;
      }
      /* prepare packet of data */
      put16(np->pkt.len, (word16) count);
      init_pkt(np, OP_SEND_FILE);
   } /* end while(send_pkt()) */

   /* np->status is set during send_pkt() */
   return np->status;
}  /* end send_fp() */

/**
 * Send the block hash associated with a block number of the current chain.
 * Uses the Trailer file (Tfile) for determining the current chain.
 * @param np Pointer to server node to send hash
 * @return (int) value representing operation result
 * @retval VEWAITING on waiting for data
 * @retval VETIMEOUT on connection timeout
 * @retval VEOK on success; packet sent
 * @retval VERROR on internal error
*/
int send_hash(NODE *np)
{
   BTRAILER tft;

   /* check if iIP list was retrieved */
   if (np->iowait == IO_RECV) {
      /* IO block number must contain a block number <= the current */
      if (PKT_IS_PV5(&(np->pkt))) {
         if (cmp64(np->pkt.blocknum, Cblocknum) > 0) {
            /* unsupported block number, send NACK... */
            init_nack(np);
            goto SEND;
         }
      }
      /* read hash of Trailer file */
      if (read_tfile(&tft, np->pkt.blocknum, 1, "tfile.dat") != 1) {
         return VERROR;
      }
      /* place hash in packet buffer and adjust data length */
      memcpy(np->pkt.buffer, tft.bhash, sizeof(tft.bhash));
      put16(np->pkt.len, sizeof(tft.bhash));

      /* increment hash request counter */
      Nhashes++;

      /* initialize packet for sending */
      init_pkt(np, OP_HASH);
SEND: np->iowait = IO_SEND;
   }

   /* send packet with block hash of current chain */
   return send_pkt(np);
}  /* end send_hash() */

/**
 * Send an IP list to a requesting server node.
 * The "recent" peers list is expected.
 * @param np Pointer to server node to send IP list
 * @return (int) value representing operation result
 * @retval VEWAITING on waiting for data
 * @retval VETIMEOUT on connection timeout
 * @retval VEOK on success; packet sent
 * @retval VERROR on internal error
*/
int send_ipl(NODE *np)
{
   size_t sz = sizeof(*Rplist);
   word16 len;

   /* check if iIP list was retrieved */
   if (np->iowait == IO_RECV) {
      /* fill IP list into packet buffer */
      len = Rplistidx * sz;
      if (PKT_IS_PV5(&(np->pkt))) {
         if (len > PKTBUFFLEN) len = (PKTBUFFLEN / sz) * sz;
      } else if (len > PKTBUFFLEN_OLD) len = (PKTBUFFLEN_OLD / sz) * sz;
      /* fill IP list into packet buffer */
      memcpy(np->pkt.buffer, Rplist, len);
      put16(np->pkt.len, len);

      /* increment IP list request counter */
      Niplist++;

      /* initialize packet for sending */
      init_pkt(np, OP_SEND_IPL);
      np->iowait = IO_SEND;
   }

   /* send packet with ledger balance */
   return send_pkt(np);
}  /* end send_ipl() */

/**
 * Send a packet of data to an NODE.
 * NOTE: return value is also placed in np->status
 * @param np Pointer to a NODE
 * @return (int) value representing send result
 * @retval VEWAITING on waiting for data
 * @retval VETIMEOUT on connection timeout
 * @retval VEOK on success; pkt is complete
 * @retval VERROR on internal error
 * @retval VEBAD on protocol violation
*/
int send_pkt(NODE *np)
{
   char *buf;
   int ecode, count, len, n;

   /* send PDUs of varying size and capabilities */
   while (len_pkt(np, &buf, &len, &n)) {
      count = send(np->sd, buf + n, len - n, 0);
      switch (count) {
         case (-1):
            ecode = sock_errno;
            /* check timeout if waiting for data */
            if (sock_waiting(ecode)) {
               if (difftime(time(NULL), np->to) > 0) {
                  return (np->status = VETIMEOUT);
               } else return (np->status = VEWAITING);
            } else goto FAIL_ECODE;    /* socket error ocurred */
         case 0: goto FAIL_SHUTDOWN;   /* socket was shutdown */
         default:
            /* reset timeout, add recv'd bytes, update length */
            np->to = time(NULL) + TIMEOUT;
            np->bytes += count;
      }  /* end switch (count... */
   }  /* end for (n = np... */

   /* full packet received: set opcode and reset bytes */
   np->opcode = get16(np->pkt.opcode);
   np->bytes = 0;

   /* success -- increment send's */
   Nsends++;  /* requires atomic operation */
   return (np->status = VEOK);

FAIL_SHUTDOWN: set_errno(ECONNABORTED); goto FAIL;
FAIL_ECODE: set_sockerrno(ecode);
FAIL:
   Nsenderrs++;
   return (np->status = VERROR);
}  /* end send_pkt() */

/**
 * Close an open file pointer of NODE.
 * @param np Pointer to NODE
 * @private for internal use only
*/
static void node__close_file(NODE *np)
{
   /* ensure FILE pointer is deallocated */
   if (np->fp != NULL) {
      fclose(np->fp);
      np->fp = NULL;
   }
}  /* end node__close_file() */

/**
 * Close an open socket descriptor of NODE, and clear iowait and timeout.
 * @param np Pointer to NODE
 * @private for internal use only
*/
static void node__close_socket(NODE *np)
{
   /* ensure socket is closed */
   if (np->sd != INVALID_SOCKET) {
      sock_close(np->sd);
      np->sd = INVALID_SOCKET;
   }
   np->iowait = IO_DONE;
   np->to = 0;
}  /* end node__close_socket() */

/**
 * Cleanup NODE resources.
 * @param np Pointer to NODE
*/
void node_cleanup(NODE *np)
{
   node__close_file(np);
   node__close_socket(np);
}  /* end node_cleanup() */

/**
 * Initialize a NODE for receive or request operations.
 * @param np Pointer to NODE
 * @param sd Receiving SOCKET descriptor, or INVALID_SOCKET for request
 * @param ip Request ip, or receive ip
 * @param port Request connection port, or 0 for receive
 * @param opreq Request operation code, or OP_NULL for receive
 * @param bnum Request IO value (blocknum), or NULL for receive
*/
void node_init
   (NODE *np, SOCKET sd, word32 ip, word16 port, word16 opreq, void *bnum)
{
   memset(np, 0, sizeof(*np));
   /* prepare NODE data */
   np->sd = sd;
   np->ip = ip;
   np->port = port;
   np->opreq = opreq;
   np->iowait = opreq ? IO_CONN : IO_RECV;
   np->status = VEWAITING;
   np->to = time(NULL) + TIMEOUT;
   if (bnum) memcpy(np->io, bnum, 8);
   ntoa(&ip, np->id);
}  /* end node_init() */

/**
 * Network communication protocol for receive operations.
 * NOTE: return value is also placed in np->status
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
int node_receive_operation(NODE *np)
{
CONTINUE:
   /* check stage of communication */
   switch (np->opcode) {
      case OP_NULL: {
         /* receive/check OP_HELLO packet */
         if (recv_pkt(np)) break;
         if (np->opcode != OP_HELLO) {
            set_errno(EMCMOPHELLO);
            np->status = VEBAD;
            break;
         }
         /* prepare "acknowledgement" packet with handshake IDs */
         np->id1 = get16(np->pkt.id1);
         np->id2 = rand16();
         init_pkt(np, OP_HELLO_ACK);
         /* update iowait type */
         np->iowait = IO_SEND;
      } /* fallthrough -- end OP_NULL*/
      case OP_HELLO: {
         /* send OP_HELLO_ACK packet */
         if (send_pkt(np)) break;
         /* update iowait type and break */
         np->iowait = IO_RECV;
         np->status = VEWAITING;
         break;  /* VEWAITING for recv */
      }  /* end case OP_HELLO (or OP_NULL) */
      case OP_HELLO_ACK: {
         /* receive request packet */
         if (recv_pkt(np)) break;
         /* recv'd opcode MUST be a "valid" operation code */
         /* NOTE: recv'd opcode MUST be checked here */
         if (np->opcode < FIRST_OP || np->opcode > LAST_OP) {
            set_errno(EMCMOPNVAL);
            np->status = VEBAD;
         }
         goto CONTINUE;  /* cheap immitation */
      }  /* end case OP_HELLO_ACK */
      case OP_TX: /* fallthrough */
      case OP_FOUND: np->status = VEOK; break;
      case OP_GET_BLOCK: send_file(np); break;
      case OP_GET_IPL: send_ipl(np); break;
      case OP_SEND_FILE: send_fp(np); break;
      case OP_GET_TFILE: send_file(np); break;
      case OP_BALANCE: send_balance(np); break;
      case OP_HASH: send_hash(np); break;
      case OP_TF: send_file(np); break;
      default: {
         set_errno(EMCMOPCODE);
         np->status = VERROR;
      }  /* end default */
   }  /* end switch (np->opcode) */

   /* check status of operation -- close on fail */
   if (np->status != VEWAITING) node__close_socket(np);

   /* return resulting status */
   return np->status;
}  /* end node_receive_operation() */

/**
 * Initiate a connection to a NODE.
 * NOTE: return value is also placed in np->status
 * @param np Pointer to a NODE
 * @param nonblock When set, configures socket for nonblocking operations
 * @return (int) value representing operation result
 * @retval VEWAITING on waiting for connection
 * @retval VETIMEOUT on connection timeout
 * @retval VEOK on successful connection
 * @retval VERROR on internal error; check errno for details
 * @private for internal use only
*/
int node_request_connect(NODE *np, int nonblock)
{
   static const socklen_t len = (socklen_t) sizeof(struct sockaddr_in);
   struct sockaddr_in addr;
   int ecode;

   /* create socket, AF_INET = IPv4 */
   if (np->sd == INVALID_SOCKET) {
      np->sd = socket(AF_INET, SOCK_STREAM, 0);
      if (np->sd == INVALID_SOCKET ||
            (nonblock && sock_set_nonblock(np->sd))) {
         set_sockerrno(sock_errno);
         return (np->status = VERROR);
      }
      /* reset timeout for initial connect */
      np->to = time(NULL) + TIMEOUT;
   }  /* end if (np->sd... */
   /* prepare socket address struct */
   memset((char *) &addr, 0, sizeof(addr));
   addr.sin_addr.s_addr = np->ip;
   addr.sin_family = AF_INET;
   addr.sin_port = htons(np->port);
   /* check connection to addr */
   if (connect(np->sd, (struct sockaddr *) &addr, len)) {
      ecode = sock_errno;
      /* check timeout if waiting for connect */
      if (sock_connected(ecode)) ecode = 0;
      else if (sock_connecting(ecode)) {
         if (difftime(time(NULL), np->to) > 0) {
            node__close_socket(np);
            return (np->status = VETIMEOUT);
         } else return (np->status = VEWAITING);
      }
      set_sockerrno(sock_errno);
      node__close_socket(np);
      return (np->status = VERROR);
   }  /* end if (connect... */

   return (np->status = VEOK);
}  /* end node_request_connect() */

/**
 * Network communication protocol for request operation.
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
int node_request_operation(NODE *np)
{
   void *buffp, *bnump;
   int count;

CONTINUE:
   switch (np->opcode) {
      case OP_NULL: {
         /* check connection wait for initial connect */
         if (np->iowait == IO_CONN) {
            if (node_request_connect(np, 1)) break;
            /* prepare HELLO packet with initial handshake IDs */
            np->id1 = rand16();
            np->id2 = 0;
            init_pkt(np, OP_HELLO);
            /* update socket operation type */
            np->iowait = IO_SEND;
         }
         /* send OP_HELLO packet */
         if (send_pkt(np)) break;
         /* update socket operation type */
         np->iowait = IO_RECV;
      } /* fallthrough -- end case OP_NULL */
      case OP_HELLO: {
         /* recv OP_HELLO packet */
         if (recv_pkt(np)) break;
         /* check initial handshake protocol */
         if (np->opcode != OP_HELLO_ACK) {
            set_errno(EMCMOPHELLOACK);
            np->status = VEBAD;
            break;
         }
         /* save second handshake ID */
         np->id2 = get16(np->pkt.id2);
         /* check request type for additional packet io requirements */
         switch (np->opreq) {
            case OP_GET_BLOCK: /* fallthrough */
            case OP_TF: memcpy(np->pkt.blocknum, np->io, 8); break;
         }
         /* prepare request operation */
         init_pkt(np, np->opreq);
         /* update socket operation type */
         np->iowait = IO_SEND;
      }  /* fallthrough -- end case OP_HELLO */
      case OP_HELLO_ACK: {
         /* send request packet */
         if (send_pkt(np)) break;
         /* update socket operation type -- wait for recv */
         np->iowait = IO_RECV;
         goto CONTINUE;  /* cheap imitation */
      }  /* end case OP_HELLO_ACK */
      default: {
         set_errno(EMCMOPCODE);
         np->status = VERROR;
      }  /* end default */
   }  /* end switch (np->opcode) */

   /* check status of handshake -- close on fail */
   if (np->status != VEWAITING && np->status != VEOK) {
      node__close_socket(np);
   }

   /* return resulting status */
   return np->status;
}  /* end node_request_handshake() */

/**
 * Network communication protocol for request operation.
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
int node_request_operation(NODE *np)
{
   switch (np->opcode) {
      case OP_GET_IPL: {
         /* receive request packet */
         if (recv_pkt(np)) break;
         /* check response opcode */
         if (np->opcode != OP_SEND_IPL) {
            set_errno(EMCMOPRECV);
            np->status = VERROR;
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
         np->status = recv_file(np);
         /* if (np->status) break; */
         break;
      }  /* end case OP_TF */
      default: {
         set_errno(EMCMOPCODE);
         np->status = VERROR;
      }  /* end default */
   }  /* end switch (np->opcode) */

   /* check status of operation -- close on fail */
   if (np->status != VEWAITING) node__close_socket(np);

   /* return resulting status */
   return np->status;
}  /* end node_request_operation() */

/* end include guard */
#endif
