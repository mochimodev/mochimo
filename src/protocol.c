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
 * Initialize a SNODE pointer for receive operation.
 * @param snp Pointer to SNODE
 * @param sd Connection socket of SNODE
 * @param ip Connection ip of SNODE
*/
void init_receive(SNODE *snp, SOCKET sd, word32 ip)
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
}  /* end init_receive() */

/**
 * Initialize a SNODE pointer for request operation.
 * @param snp Pointer to SNODE to prepare
 * @param ip Connection ip of SNODE
 * @param opreq Request operation code
 * @param bnum Request IO value (blocknum), or NULL
*/
void init_request
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
}  /* end init_request() */

/**
 * Initialize a packet of SNODE with protocol data.
 * @param snp Pointer to SNODE
 * @param opcode Operation code of packet
*/
void init_pkt(SNODE *snp, word16 opcode)
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
}  /* end init_pkt() */

/**
 * Initialize a packet of SNODE with protocol data and OP_NACK.
 * OP_NACK refers to a Negative Acknowledgment operation (usually
 * sent in response to a request operation) identifying that the
 * request was received successfully, but it contained an error.
 * @param snp Pointer to SNODE
*/
void init_nack(SNODE *snp)
{
   /* initialize NACK protocol */
   put16(snp->pkt.len, 0);
   init_pkt(snp, OP_NACK);
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
 * @param snp Pointer to SNODE
 * @returns (int) length, in bytes, of next packet
*/
static int len_pkt(SNODE *snp, char **buf, int *len, int *n)
{
   /* determine position and length of next packet data */
   if (snp->bytes < PKTHDRLEN) {
      /* packet header */
      *n = snp->bytes;
      *len = PKTHDRLEN;
      *buf = (char *) snp->pkt.version;
   } else if (snp->c_vpdu) {
      *len = (int) get16(snp->pkt.len);
      if (snp->bytes < (PKTHDRLEN + *len)) {
         /* VPDU packet buffer */
         *n = snp->bytes - PKTHDRLEN;
         *buf = (char *) snp->pkt.buffer;
      } else {
         /* VPDU packet trailer */
         *n = snp->bytes - (PKTHDRLEN + *len);
         *len = PKTTLRLEN;
         *buf = (char *) snp->pkt.crc16;
      }
   } else if (snp->bytes < (PKTHDRLEN + PKTBUFFLEN_OLD)) {
      /* !VPDU packet buffer */
      *n = snp->bytes - PKTHDRLEN;
      *len = PKTBUFFLEN_OLD;
      *buf = (char *) snp->pkt.buffer;
   } else {
      /* !VPDU packet trailer */
      *n = snp->bytes - (PKTHDRLEN + PKTBUFFLEN_OLD);
      *len = PKTTLRLEN;
      *buf = (char *) snp->pkt.crc16;
   }

   return *len - *n;
}  /* end len_pkt() */

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
   while (len_pkt(snp, &buf, &len, &n)) {
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
BAD_CRC: set_errno(EMCMPKTCRC); goto BAD;
BAD_NET: set_errno(EMCMPKTNET); goto BAD;
BAD_TLR: set_errno(EMCMPKTTLR); goto BAD;
BAD_IDS: set_errno(EMCMPKTIDS);
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
FAIL_NACK: set_errno(EMCMPKTNACK);
FAIL: return (snp->status = VERROR);

/* protocol violation handling */
BAD_OPCODE: set_errno(EMCMPKTOPCODE); return (snp->status = VEBAD);
}  /* end recv_file() */

/**
 * Send a ledger balance to a requesting server node connection.
 * Non-Hashed Address search requests are converted before search.
 * NOTE: Protocol version 5 accepts Hashed Addresses and returns
 * a Ledger Transaction (LTRAN) containing Ledger data where the
 * value of @a trancode[0] is set if the address was "found".
 * @param snp Pointer to server node to send balance
 * @return (int) value representing operation result
 * @retval VEWAITING on waiting for data
 * @retval VETIMEOUT on connection timeout
 * @retval VEOK on success; packet sent
 * @retval VERROR on internal error
*/
int send_balance(SNODE *snp)
{
   TXW *txwp;
   LTRAN *ltp;
   LENTRY *lep;

   /* check if balance was retrieved */
   if (snp->iowait == IO_RECV) {
      /* check protocol version... */
      if (PKT_IS_PV5(&(snp->pkt))) {
         /* ... PVERSION 5 onwards uses Hashed Address */
         ltp = (LTRAN *) snp->pkt.buffer;
         if (get16(snp->pkt.len) != TXADDRLEN) {
            /* unsupported address type, send NACK... */
            init_nack(snp);
            goto SEND;
         }
         /* update the packet buffer data length */
         put16(snp->pkt.len, sizeof(*ltp));
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
         txwp = (TXW *) snp->pkt.buffer;
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
      init_pkt(snp, OP_SEND_BAL);
SEND: snp->iowait = IO_SEND;
   }

   /* send packet with ledger balance */
   return send_pkt(snp);
}  /* end send_balance() */

/**
 * Send various files to a requesting server node.
 * @param snp Pointer to server node to send IP list
 * @return (int) value representing operation result
 * @retval VEWAITING on waiting for send buffer
 * @retval VETIMEOUT on connection timeout
 * @retval VEOK on success; finished
 * @retval VERROR on internal error
*/
int send_file(SNODE *snp)
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
   if (snp->iowait == IO_RECV) {
      bp = snp->pkt.blocknum;
      /* determine file type requested */
      switch (get16(snp->pkt.opcode)) {
         case OP_GET_BLOCK: {
            /* read Tfile for hash of requested block */
            if (read_tfile(&tft, bp, 1, "tfile.dat") != 1) return VERROR;
            /* get archived file name */
            bc_fqan(fname, snp->pkt.blocknum, tft.bhash);
            path_join(fpath, Bcdir_opt, fname);
            /* try opening file -- VERROR if unable to send */
            snp->fp = fopen(fpath, "rb");
            if (snp->fp == NULL) return VERROR;
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
            bnum[0] = get32(snp->pkt.blocknum);
            count = get16(snp->pkt.blocknum + 4);
            /* protocol version 5 introduces additional feedback... */
            if (PKT_IS_PV5(&(snp->pkt))) {
               /* ... IO MUST NOT be greater than current block number,
               * and count must be non-zero up to 1000 */
               if (cmp64(bnum, Cblocknum) > 0
                  || count < 1 || count > 1000) {
                  /* unsupported IO values, send NACK... */
                  init_nack(snp);
                  goto SEND;
               }
            } else if (count < 1 || count > 1000) return VERROR;
            /* seek to appropriate location */
            first = 0;
            put64(&first, bnum);
            first *= (long long) sizeof(tft);
PREP_TFILE: /* create temporary file ("wb+") -- remove buffer */
            snp->fp = tmpfile();
            if (snp->fp == NULL) return VERROR;
            setvbuf(snp->fp, NULL, _IONBF, 0);
            /* open Tfile for copy */
            fp = fopen("tfile.dat", "rb");
            if (fp == NULL) return VERROR;
            /* seek to appropriate location */
            if (fseek64(fp, first, SEEK_SET) != 0) goto FAIL;
            /* load requested Tfile data into temporary file */
            while (count-- && fread(&tft, sizeof(tft), 1, fp)) {
               if (fwrite(&tft, sizeof(tft), 1, snp->fp) != 1) goto FAIL;
            }
            break;
         }  /* end case OP_TF (or OP_GET_TFILE) */
         default: return VERROR;
      }  /* end switch (get16(snp->pkt.opcode)) */

      /* rewind temp file */
      rewind(snp->fp);
      /* read first chunk of data into packet buffer */
      len = snp->c_vpdu ? PKTBUFFLEN : PKTBUFFLEN_OLD;
      count = fread(snp->pkt.buffer, 1, len, snp->fp);
      /* initialize packet for first send */
      put16(snp->pkt.len, (word16) count);
      init_pkt(snp, OP_SEND_FILE);
SEND: snp->iowait = IO_SEND;
   }  /* end if (snp->iowait == IO_RECV) */

   /* divert to send_pkt() for NACK */
   if (get16(snp->pkt.opcode) == OP_NACK) return send_pkt(snp);

   return send_fp(snp);

/* error handling */
FAIL:
   fclose(fp);
   return VERROR;
}  /* end send_file() */

/**
 * Send multiple packets of data froma FILE pointer to a SNODE.
 * @param snp Pointer to a SNODE containing open FILE pointer
 * @return (int) value representing operation result
 * @retval VEWAITING on waiting for send buffer
 * @retval VETIMEOUT on connection timeout
 * @retval VEOK on success; FILE pointer EOF
 * @retval VERROR on internal error
*/
int send_fp(SNODE *snp)
{
   size_t count, len;

   /* send packet of data */
   while (send_pkt(snp) == VEOK) {
      /* check file pointer for end */
      if (snp->fp == NULL) break;
      /* read next chunk of data into packet buffer */
      len = snp->c_vpdu ? PKTBUFFLEN : PKTBUFFLEN_OLD;
      count = fread(snp->pkt.buffer, 1, len, snp->fp);
      if (count < len) {
         /* check for specific error -- close file pointer */
         if (ferror(snp->fp)) snp->status = VERROR;
         fclose(snp->fp);
         snp->fp = NULL;
      }
      /* prepare packet of data */
      put16(snp->pkt.len, (word16) count);
      init_pkt(snp, OP_SEND_FILE);
   } /* end while(send_pkt()) */

   /* snp->status is set during send_pkt() */
   return snp->status;
}  /* end send_fp() */

/**
 * Send the block hash associated with a block number of the current chain.
 * Uses the Trailer file (Tfile) for determining the current chain.
 * @param snp Pointer to server node to send hash
 * @return (int) value representing operation result
 * @retval VEWAITING on waiting for data
 * @retval VETIMEOUT on connection timeout
 * @retval VEOK on success; packet sent
 * @retval VERROR on internal error
*/
int send_hash(SNODE *snp)
{
   BTRAILER tft;

   /* check if iIP list was retrieved */
   if (snp->iowait == IO_RECV) {
      /* IO block number must contain a block number <= the current */
      if (PKT_IS_PV5(&(snp->pkt))) {
         if (cmp64(snp->pkt.blocknum, Cblocknum) > 0) {
            /* unsupported block number, send NACK... */
            init_nack(snp);
            goto SEND;
         }
      }
      /* read hash of Trailer file */
      if (read_tfile(&tft, snp->pkt.blocknum, 1, "tfile.dat") != 1) {
         return VERROR;
      }
      /* place hash in packet buffer and adjust data length */
      memcpy(snp->pkt.buffer, tft.bhash, sizeof(tft.bhash));
      put16(snp->pkt.len, sizeof(tft.bhash));

      /* increment hash request counter */
      Nhashes++;

      /* initialize packet for sending */
      init_pkt(snp, OP_HASH);
SEND: snp->iowait = IO_SEND;
   }

   /* send packet with block hash of current chain */
   return send_pkt(snp);
}  /* end send_hash() */

/**
 * Send an IP list to a requesting server node.
 * The "recent" peers list is expected.
 * @param snp Pointer to server node to send IP list
 * @return (int) value representing operation result
 * @retval VEWAITING on waiting for data
 * @retval VETIMEOUT on connection timeout
 * @retval VEOK on success; packet sent
 * @retval VERROR on internal error
*/
int send_ipl(SNODE *snp)
{
   size_t sz = sizeof(*Rplist);
   word16 len;

   /* check if iIP list was retrieved */
   if (snp->iowait == IO_RECV) {
      /* fill IP list into packet buffer */
      len = Rplistidx * sz;
      if (PKT_IS_PV5(&(snp->pkt))) {
         if (len > PKTBUFFLEN) len = (PKTBUFFLEN / sz) * sz;
      } else if (len > PKTBUFFLEN_OLD) len = (PKTBUFFLEN_OLD / sz) * sz;
      /* fill IP list into packet buffer */
      memcpy(snp->pkt.buffer, Rplist, len);
      put16(snp->pkt.len, len);

      /* increment IP list request counter */
      Niplist++;

      /* initialize packet for sending */
      init_pkt(snp, OP_SEND_IPL);
      snp->iowait = IO_SEND;
   }

   /* send packet with ledger balance */
   return send_pkt(snp);
}  /* end send_ipl() */

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
   while (len_pkt(snp, &buf, &len, &n)) {
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
 * Deallocate (cleanup) allocated resources within a SNODE.
 * @param snp Pointer to SNODE
*/
void cleanup_node(SNODE *snp)
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
}  /* end cleanup_node() */

/**
 * Network communication protocol for receive operations.
 * NOTE: return value is also placed in snp->status
 * @param snp Pointer to a SNODE
 * @return (int) value representing operation result
 * @retval VEWAITING on waiting for data
 * @retval VETIMEOUT on connection timeout
 * @retval VEOK on success; pkt is complete
 * @retval VERROR on internal error
 * @retval VEBAD on protocol violation; peer is now pinklisted
*/
int receive_node(SNODE *snp)
{
OP_RESTART:
   /* check stage of communication */
   switch (snp->opcode) {
      case OP_NULL: {
         /* receive/check OP_HELLO packet */
         if (recv_pkt(snp)) break;
         if (snp->opcode != OP_HELLO) {
            set_errno(EMCMOPHELLO);
            snp->status = VEBAD;
            break;
         }
         /* prepare "acknowledgement" packet with handshake IDs */
         snp->id1 = get16(snp->pkt.id1);
         snp->id2 = rand16();
         init_pkt(snp, OP_HELLO_ACK);
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
            set_errno(EMCMOPCODE);
            snp->status = VERROR;
            break;
         }
         /* restart switch block on success */
         goto OP_RESTART;
      }  /* end case OP_HELLO_ACK */
      case OP_GET_BLOCK: send_file(snp); break;
      case OP_GET_IPL: send_ipl(snp); break;
      case OP_SEND_FILE: send_fp(snp); break;
      case OP_GET_TFILE: send_file(snp); break;
      case OP_BALANCE: send_balance(snp); break;
      case OP_HASH: send_hash(snp); break;
      case OP_TF: send_file(snp); break;
      default: {
         set_errno(EMCMOPRECV);
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
}  /* end receive_node() */

/**
 * @private
 * @brief Initiate a connection to a server node.
 * Return value is also placed in snp->status
 * @param snp Pointer to a SNODE
 * @return (int) value representing operation result
 * @retval VEWAITING on waiting for data
 * @retval VETIMEOUT on connection timeout
 * @retval VEOK on successful connection
 * @retval VERROR on internal error
*/
static int connect_node(SNODE *snp)
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
}  /* end connect_node() */

/**
 * Network communication protocol for request operations.
 * @param snp Pointer to a SNODE
 * @returns Status of request, as integer.
 * @retval VEWAITING on waiting for data
 * @retval VETIMEOUT on socket timeout
 * @retval VEOK on protocol completion
 * @retval VERROR on socket shutdown and/or error
 * @retval VEBAD on pkt protocol violation
*/
int request_node(SNODE *snp)
{
OP_RESTART:
   /* check stage of communication */
   switch (snp->opcode) {
      case OP_NULL: {
         /* check connection wait for initial connect */
         if (snp->iowait == IO_CONN) {
            if (connect_node(snp)) break;
            /* prepare HELLO packet with initial handshake IDs */
            snp->id1 = rand16();
            snp->id2 = 0;
            init_pkt(snp, OP_HELLO);
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
            set_errno(EMCMOPHELLOACK);
            snp->status = VEBAD;
            break;
         }
         /* save second handshake ID */
         snp->id2 = get16(snp->pkt.id2);
         /* check request type for additional packet io requirements */
         switch (snp->opreq) {
            case OP_GET_BLOCK: /* fallthrough */
            case OP_TF: memcpy(snp->pkt.blocknum, snp->io, 8); break;
         }
         /* prepare request operation */
         init_pkt(snp, snp->opreq);
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
            set_errno(EMCMOPCODE);
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
         set_errno(EMCMOPSEND);
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
}  /* end request_node() */

/* end include guard */
#endif
