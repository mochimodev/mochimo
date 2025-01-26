/**
 * @private
 * @headerfile network.h <network.h>
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_NETWORK_C
#define MOCHIMO_NETWORK_C


#include "network.h"

/* internal support */
#include "tx.h"
#include "tfile.h"
#include "sync.h"
#include "parallel.h"
#include "ledger.h"
#include "global.h"
#include "error.h"

/* external support */
#include <string.h>
#include "exttime.h"
#include "extthrd.h"
#include "extmath.h"
#include "extlib.h"
#include "extinet.h"
#include "crc16.h"

#define valid_op(op)  ((op) >= FIRST_OP && (op) <= LAST_OP)
#define crowded(op)   (Nonline > (MAXNODES - 5) && (op) != OP_FOUND)
#define can_fork_tx() (Nonline <= (MAXNODES - 5))

#define TXHDRLEN 124
#define TXTLRLEN 4

NODE Nodes[MAXNODES];   /* data structure for connected NODE's */
NODE *Hi_node = Nodes;  /* points one beyond last logged in NODE */
word32 Nrecvs;          /* number of receive errors */
word32 Nsends;          /* number of send errors */
word32 Nrecverrs;       /* number of receive errors */
word32 Nsenderrs;       /* number of send errors */

/**
 * Is called after initial accept() or connect()
 * Adds the connection to Node[] array.
 * and returns a new NODE * with *np's data.
*/
NODE *getslot(NODE *np)
{
   NODE *newnp;

   /*
    * Find empty slot
    */
   for(newnp = Nodes; newnp < &Nodes[MAXNODES]; newnp++)
      if(newnp->pid == 0) break;

   if(newnp >= &Nodes[MAXNODES]) {
      perr("Nodes[] full!");
      Nspace++;
      return NULL;
   }

   Nonline++;    /* number of currently connected sockets */
   pdebug("added NODE %d", (int) (newnp - Nodes));
   if (newnp >= Hi_node) Hi_node = newnp + 1;
   memcpy(newnp, np, sizeof(NODE));
   return newnp;
}  /* end getslot() */

/* Mark NODE np in Nodes[] empty by setting np->pid to zero.
 * Adjust Nonline and Hi_node.
 * Caller must close np->sd if needed.
 */
int freeslot(NODE *np)
{
   if(np->pid == 0) {
      perr("*** NODE %ld already free ***", (long) (np - Nodes));
      return VERROR;
   }
   pdebug("idx=%d  ip = %s pid = %d", (int) (np - Nodes),
           ntoa(&np->ip, NULL), (int) np->pid);
   Nonline--;
   np->pid = 0;
   /* Update pointer to just beyond highest used slot in Nodes[] */
   while(Hi_node > Nodes && (Hi_node - 1)->pid == 0)
      Hi_node--;
   if (Nonline < 0) {
      Nonline = 0;
      perr("Nonline < 0");
      return VERROR;
   }
   return VEOK;
}  /* end freeslot() */

/* Return child status of pid.
 * Add peer ip to lists if needed.
 * Increment counts.
 * Returns 1 if child did not call exit(1),
 * else the exit() code (which can also be 1).
 */
int child_status(NODE *np, pid_t pid, int status)
{
   pdebug("pid = %d  status = 0x%x", pid, status);
   if(pid > 0) {  /* child existed and called exit() */
      if(WIFEXITED(status)) {
         status = WEXITSTATUS(status);
         if(status != 0) {
            if(status >= 2) epinklist(np->ip);
            if(status >= 3) pinklist(np->ip);
            return status;
         }
      } else return 1;  /* error if not exited */
      return status;  /* 0 */
   }  /* end if child exit()'ed */
   return 1;  /* error if child caught signal */
}  /* end child_status() */

/**
 * Receive next packet from NODE *np.
 * SOCKET np->sd is already set non-blocking.
 * Returns: VEOK (0) = good, else error code. */
int recv_tx(NODE *np, double timeout)
{
   int count, len, n;
   time_t start;
   TX *tx;

   /* init recv_tx() */
   tx = &(np->tx);
   put16(tx->len, 0);
   time(&start);

   /* loop until PDU is recv'd
    * NOTE: tx.len[2] may extend the requirement recv()
    */

#define recv_len(lenp) ( TXHDRLEN + get16((lenp)) + TXTLRLEN )
   len = recv_len(tx->len);
   for (n = 0; n < len; n += count, len = recv_len(tx->len)) {
      count = recv(np->sd, (word8 *) tx + n, len - n, 0);
      switch (count) {
         case (-1): {
            if (sock_waiting(sock_errno)) {
               if (difftime(time(NULL), start) >= timeout) {
                  set_errno(ETIMEDOUT);
                  return VETIMEOUT;
               }
               /* wait patiently */
               millisleep(10);
               count = 0;
               continue;
            }
            perrno("%s recv() failed", np->id);
         }  /* fallthrough */
         case 0: {
            pdebug("%s abort", np->id);
            return VERROR;
         }
      }  /* end switch */
   }  /* end for (n... */

   /* shift crc16 and trailer to correct position in TX struct */
   memmove(tx->crc16, tx->buffer + get16(tx->len), 4);

   /* compute crc16 checksum and verify packet integrity */
   if (get16(tx->crc16) != crc16(tx, len - TXTLRLEN)) {
      pdebug("%s *** CRC16 mismatch, 0x%" P16X " != 0x%" P16X,
         np->id, get16(tx->crc16), crc16(tx, TXHDRLEN + len));
      Nrecverrs++;
      return VEBAD;
   }
   /* check packet network protocol version */
   if (get16(tx->network) != TXNETWORK) {
      pdebug("%s *** invalid network, %" P16u " != %" P16u,
         np->id, get16(tx->network), TXNETWORK);
      Nrecverrs++;
      return VEBAD;
   }
   /* check packet trailer */
   if (get16(tx->trailer) != TXEOT) {
      pdebug("%s *** invalid trailer, 0x%" P16X " != 0x%" P16X,
         np->id, get16(tx->trailer), TXEOT);
      Nrecverrs++;
      return VEBAD;
   }
   /* check handshake IDs on all operations (except during handshake) */
   if (get16(tx->opcode) >= FIRST_OP) {
      if (np->id1 != get16(tx->id1) || np->id2 != get16(tx->id2)) {
         pdebug("%s *** unexpected ID 0x%" P32x, np->id,
            (word32) (get16(tx->id1) | ((word32)get16(tx->id2) << 16)));
         Nrecverrs++;
         return VEBAD;
      }
   }

   /* packet recv'd */
   Nrecvs++;
   return VEOK;
}  /* end recv_tx() */

/**
 * Receive packets from NODE *np, and write to file, fname.
 * SOCKET np->sd is set non-blocking, ready to recv data.
 * Returns: VEOK (0) = good, else error code. */
int recv_file(NODE *np, char *fname)
{
   TX *tx;
   FILE *fp;
   time_t prevtime;
   word16 len;

   /* init recv_file() */
   time(&prevtime);
   tx = &(np->tx);

   /* open file for writing recv'd data */
   fp = fopen(fname, "wb");
   if (fp == NULL) {
      perrno("(%s, %s) fopen() failed", np->id, fname);
      return VERROR;
   }

   /* receive packets and write */
   pdebug("(%s, %s) receiving...", np->id, fname);
   while (recv_tx(np, STD_TIMEOUT) == VEOK) {
      /* check recv'd packet */
      if (get16(tx->opcode) != OP_SEND_FILE) {
         pdebug("(%s, %s) *** invalid opcode", np->id, fname);
         break;
      }
      len = get16(tx->len);
      if (len && fwrite(tx->buffer, len, 1, fp) != 1) {
         pdebug("(%s, %s) *** I/O error", np->id, fname);
         break;
      }
      /* check EOF */
      if (len < sizeof(tx->buffer)) {
         fclose(fp);
         pdebug("(%s, %s) EOF", np->id, fname);
         return VEOK;
      } /* end if EOF */
   }  /* end for */
   fclose(fp);
   /* delete partial downloads */
   remove(fname);

   return VERROR;
}  /* end recv_file() */

/**
 * Send next packet to NODE *np.
 * Set advertised fields and compute CRC16.
 * Returns VEOK on success, else VERROR. */
int send_tx(NODE *np, double timeout)
{
   int count, len, n;
   time_t start;
   TX *tx;

   /* init send_tx() */
   tx = &(np->tx);
   time(&start);

   /* fill tx packet with relevant information... */
   tx->version[0] = PVERSION;
   tx->version[1] = Cbits;
   put16(tx->network, TXNETWORK);
   put16(tx->trailer, TXEOT);
   put16(tx->id1, np->id1);
   put16(tx->id2, np->id2);
   put64(tx->cblock, Cblocknum);
   memcpy(tx->cblockhash, Cblockhash, HASHLEN);
   memcpy(tx->pblockhash, Prevhash, HASHLEN);
   /* ... but, do not overwrite TX ip map */
   if (get16(tx->opcode) != OP_TX) {
      memcpy(tx->weight, Weight, HASHLEN);
   }

   /* compute packet crc16 checksum -- use length from previous step */
   put16(tx->crc16, crc16(tx, TXHDRLEN + get16(tx->len)));
   /* shift crc16 and trailer to correct position in buffer */
   memmove(tx->buffer + get16(tx->len), tx->crc16, 4);

   /* loop until PDU is recv'd
    * NOTE: tx.len[2] requirement DOES NOT change here
    */

   len = TXHDRLEN + get16(tx->len) + TXTLRLEN;
   for (n = 0; n < len; n += count) {
      count = send(np->sd, (word8 *) tx + n, len - n, 0);
      switch (count) {
         case (-1): {
            if (sock_waiting(sock_errno)) {
               if (difftime(time(NULL), start) >= timeout) {
                  set_errno(ETIMEDOUT);
                  return VETIMEOUT;
               }
               /* wait patiently */
               millisleep(10);
               count = 0;
               continue;
            }
            perrno("%s send() failed", np->id);
         }  /* fallthrough */
         case 0: {
            pdebug("%s abort", np->id);
            return VERROR;
         }
      }  /* end switch */
   }  /* end for (n... */

   /* packet sent */
   Nsends++;
   return VEOK;
}  /* end send_tx() */


int send_op(NODE *np, int opcode)
{
   put16(np->tx.opcode, opcode);
   return send_tx(np, STD_TIMEOUT);
}

/**
 * Send a Negative Acknowledgment (NACK) to a NODE, involving an error.
 * The NACK is prepared with detailed information about an error number.
 * The NACK error buffer is formatted as follows:
 *    [8 byte value][32 byte name][variable length description]
 * The first 8 bytes of space is currently reserved as a reference value;
 * perhaps a block number, multi-destination transaction number, etc.
 * The next 32 bytes contains the error name (null terminated); unchanging
 * string names, used by clients to determine the next appropriate action.
 * Any remaining bytes contains the error description (null terminated); for
 * developer debugging or additional helpful client information.
 * @param np Pointer to NODE
 * @param errnum Value of error number
 * @return (int) value representing operation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 */
int send_nack(NODE *np, int errnum)
{
   char *error = (char *) np->tx.buffer;

   /* set necessary zero fill */
   memset(error, 0, 8 + 32);
   /* providing the value of an error number is somewhat useless as it can
    * change between systems and updates, but is placed here as example */
   put32(error, (word32) errnum);
   /* derive error name from errnum (e.g. "EMCM_MADDR") */
   mcm_strerrorname(errnum, error + 8, 32);
   /* derive error description from errnum (e.g. "Bad miner address") */
   mcm_strerror(errnum, error + 8 + 32, 256);

   /* check length of description and send NACK */
   put16(np->tx.len, 8 + 32 + strlen(error + 40) + 1);
   return send_op(np, OP_NACK);
}  /* end send_nack() */

/**
 * Send packets to NODE *np, and write to file, fname.
 * SOCKET np->sd is set non-blocking, ready to recv data.
 * Set fname NULL send np->tx.blocknum request.
 * Returns: VEOK (0) = good, else error code. */
int send_file(NODE *np, char *fname)
{
   char dummy[FILENAME_MAX];
   char bcfname[22];
   size_t count;
   int ecode;
   FILE *fp;
   TX *tx;

   /* init send_file() */
   tx = &(np->tx);
   if (fname == NULL) {
      bnum2fname(tx->blocknum, bcfname);
      fname = path_join(dummy, Bcdir, bcfname);
   }
   pdebug("(%s, %s) sending...", np->id, fname);

   /* open file for writing recv'd data */
   fp = fopen(fname, "rb");
   if (fp == NULL) {
      pdebug("(%s, %s) cannot send file", np->id, fname);
      return VERROR;
   }
   /* read and send packets */
   do {
      /* read file data and break on error */
      count = fread(tx->buffer, 1, sizeof(tx->buffer), fp);
      if (count != sizeof(tx->buffer) && ferror(fp)) {
         perr("(%s, %s) *** I/O error", np->id, fname);
         ecode = VERROR;
         break;
      }
      /* send file data and break on EOF */
      put16(tx->len, (word16) count);
      ecode = send_op(np, OP_SEND_FILE);
      if (count != sizeof(tx->buffer)) {
         pdebug("(%s, %s) EOF", np->id, fname);
         break;
      }
      /* Make upload bandwidth dynamic. */
      if (Nonline > 1) millisleep(Nonline - 1);
   } while (ecode == VEOK);
   /* cleanup */
   fclose(fp);
   return ecode;
}  /* end send_file() */

/**
 * Send a ledger.dat balance query to np.
 * Called from gettx() OP_BALANCE
 * layout:
 * on entry:
 *     np->tx.src_addr    address to query
 * on return:
 *     np->tx.send_total = balance of src_addr (0 if not found)
 *
 * Returns 1 on I/O errors, else 0.
*/
int send_balance(NODE *np)
{
   LENTRY le;
   word16 len;

   len = get16(np->tx.len);
   if (len > ADDR_LEN) len = ADDR_LEN;

   /* look up source address in ledger */
   if (le_find(np->tx.buffer, &le, len)) {
      memcpy(np->tx.buffer, &le, sizeof(LENTRY));
      put16(np->tx.len, sizeof(LENTRY));
      send_op(np, OP_SEND_BAL);
   }

   Nbalance++;
   return 0;  /* success */
} /* end send_balance() */

/* Send our recent peer list to NODE np in response to OP_GETIPL.
 * Called from execute().
 */
int send_ipl(NODE *np)
{
   int count = RPLISTLEN;

   if (count > 32) count = 32;
   /* copy recent peer list to TX */
   memcpy(np->tx.buffer, Rplist, sizeof(word32) * count);
   put16(np->tx.len, sizeof(word32) * count);
   return send_op(np, OP_SEND_IPL);  /* send ip list */
}

/* Process OP_HASH.  Return VEOK on success, else VERROR.
 * Called by gettx().
 */
int send_hash(NODE *np)
{
   BTRAILER bt;
   char fname[FILENAME_MAX];
   char bcfname[21];

   bnum2fname(np->tx.blocknum, bcfname);
   path_join(fname, Bcdir, bcfname);
   if (read_trailer(&bt, fname) != VEOK) {
      return VERROR;
   }
   /* copy hash of tx.blocknum to TX */
   memcpy(np->tx.buffer, bt.bhash, HASHLEN);
   put16(np->tx.len, HASHLEN);
   return send_op(np, OP_HASH);  /* send back to peer */
}  /* end send_hash() */

/* Process OP_TF.  Return VEOK on success, else VERROR.
 * Called by child -- execute().
 */
int send_tf(NODE *np)
{
   int status;
   word32 first, count;
   char cmd[128], fname[32];

   sprintf(fname, "tf%u.tmp", (int) getpid());

   first = get32(np->tx.blocknum);      /* first trailer to send */
   count = get32(&np->tx.blocknum[4]);  /* count of trailers to send */

   /* limit tfile extract to 1000 trailers */
   if(count > 1000) return VERROR;
   sprintf(cmd, "dd if=tfile.dat of=%s bs=%u skip=%u count=%u 2>/dev/null",
                fname, (int) sizeof(BTRAILER), first, count);
   system(cmd);
   status = send_file(np, fname);  /* returns VEOK or VERROR */
   remove(fname);
   return status;
}  /* end send_tf() */


int send_identify(NODE *np)
{
   /* copy recent peer list to TX */
   sprintf((char *) np->tx.buffer, "Sanctuary=%u,Lastday=%u,Mfee=%u",
           Sanctuary, Lastday, Myfee[0]);
   put16(np->tx.len, (word16) strlen((char *) np->tx.buffer));
   return send_op(np, OP_IDENTIFY);
}

/* Creates child to send OP_FOUND to all recent peers */
int send_found(void)
{
   word32 plist[RPLISTLEN];
   NODE node;
   BTRAILER bt;
   char fname[FILENAME_MAX];
   char bcfname[21];
   char bnumhex[17];
   int ecode, count, len, i;
   TX tx;
   word8 bnum[8];

   if (Found_pid) {
      pdebug("send_found() is already running -- rerun it.");
      stop_found();
   }

   Found_pid = fork();
   if(Found_pid == -1) {
      Found_pid = 0;
      return VERROR;  /* fork() failed */
   }
   if(Found_pid) return VEOK;          /* parent returns */

   /* in child */
   show("found");

   /* Check if "found" NG block v.23 */
   if(Cblocknum[0] == 0) {
      ecode = 1;
      /* Back up our Cblocknum in child only to 0x...ff block. */
      if(sub64(Cblocknum, One, Cblocknum)) goto bad;
      bnum2fname(Cblocknum, bcfname);
      path_join(fname, Bcdir, bcfname);
      ecode = 2;
      if (read_trailer(&bt, fname) != VEOK
         || cmp64(Cblocknum, bt.bnum) != 0) {
bad:
         perr("ecode: %d", ecode);
         exit(VERROR);
      }
      ecode = 3;
      if(memcmp(Prevhash, bt.bhash, HASHLEN)) goto bad;
      memcpy(Cblockhash, bt.bhash, HASHLEN);
      memcpy(Prevhash, bt.phash, HASHLEN);
   }  /* end if NG block v.23 */

   pdebug("send_found(0x%s)", bnum2hex(Cblocknum, bnumhex));
   pdebug("...weight(0x%s)", weight2hex(Weight, NULL));

   /* get proof from tfile.dat (!!! (NTFTX - 1) ) */
   if (sub64(Cblocknum, CL64_32(NTFTX - 1), bnum)) memset(bnum, 0, 8);
   count = read_tfile(tx.buffer, bnum, NTFTX, "tfile.dat");

   /* build peerlist with Rplist (shuffled) */
   memset(plist, 0, sizeof(plist));
   shufflenz(Rplist, sizeof(*Rplist), RPLISTLEN);
   len = loadpeers(plist, RPLISTLEN, Rplist, RPLISTLEN);

   /* Send found message to peerlist */
   for(i = 0; i < len && Running; i++) {
      if(plist[i] == 0) break;
      if(callserver(&node, plist[i]) != VEOK) continue;
      memcpy(&node.tx, &tx, sizeof(TX));  /* copy in tfile proof */
      put16(node.tx.len, (word16) count * sizeof(BTRAILER));
      send_op(&node, OP_FOUND);
      sock_close(node.sd);
   }

   exit(0);
}  /* end send_found() */

/**
 * Call peer and complete Three-Way handshake */
int callserver(NODE *np, word32 ip)
{
   char ipaddr[16];  /* for threadsafe ntoa() usage */
   word8 id1, id2;

   /* init callserver() */
   id1 = id2 = 0;
   ntoa(&ip, ipaddr);
   memset(np, 0, sizeof(NODE));   /* clear structure */
   snprintf(np->id, sizeof(np->id), "%.15s %.02x~%.02x", ipaddr, id1, id2);
   /* begin connection */
   np->ip = ip;
   np->sd = sock_connect_ip(ip, Dstport, INIT_TIMEOUT);
   if(np->sd == INVALID_SOCKET) {
      pdebug("%s failed to connect", np->id);
      goto FAIL_ERRSOCK;
   }
   /* initiate Three-Way Handshake */
   np->id1 = rand16();
   id1 = (word8) (np->id1 >> 8);
   put16(np->tx.opcode, OP_HELLO);
   snprintf(np->id, sizeof(np->id), "%.15s %.02x~%.02x", ipaddr, id1, id2);
   if (send_tx(np, 1) != VEOK) {
      pdebug("%s failed to send handshake", np->id);
      goto FAIL_ERR3WAY;
   } else if (recv_tx(np, INIT_TIMEOUT) != VEOK) {
      pdebug("%s *** handshake not recv'd", np->id);
      goto FAIL_ERR3WAY;
   }
   /* validate Three-Way Handshake */
   np->id2 = get16(np->tx.id2);
   id2 = (word8) np->id2;
   snprintf(np->id, sizeof(np->id), "%.15s %.02x~%.02x", ipaddr, id1, id2);
   if (get16(np->tx.opcode) != OP_HELLO_ACK) {
      pdebug("%s *** missing hello acknowledgement", np->id);
      goto FAIL_BAD3WAY;
   } else if (get16(np->tx.id1) != np->id1) {
      pdebug("%s *** handshake ID mismatch", np->id);
      goto FAIL_BAD3WAY;
   }

   /* success -- made a new friend */
   return VEOK;

   /* failure -- cleanup/error handling */
FAIL_BAD3WAY:
   sock_close(np->sd);
   np->sd = INVALID_SOCKET;
   return VEBAD;
FAIL_ERR3WAY:
   sock_close(np->sd);
   np->sd = INVALID_SOCKET;
   return VERROR;
FAIL_ERRSOCK:
   return VERROR;
}  /* end callserver() */

/**
 * Used for simple one packet responses like OP_GET_IPL.
 * Closes socket and sets np->sd to INVALID_SOCKET on return.
 * Returns VEOK on success, else VERROR. */
int get_tx(NODE *np, word32 ip, word16 opcode)
{
   int ecode = VEOK;

   /* initiate connection with ip */
   ecode = callserver(np, ip);
   if(ecode != VEOK) return ecode;
   else {
      /* send and receive single packet */
      ecode = send_op(np, opcode);
      if(ecode == VEOK) ecode = recv_tx(np, STD_TIMEOUT);
      /* cleanup */
      sock_close(np->sd);
      np->sd = INVALID_SOCKET;
   }
   return ecode;
}  /* end get_tx() */

/**
 * Get a file from peer, ip, and store in fname.
 * Tfile is downloaded if bnum is set NULL, else block file.
 * The current "candidate" block file is requested, if bnum
 * is equal to the maximum block value, WORD64_MAX.
 * Returns VEOK (0) on good download, else error code. */
int get_file(word32 ip, word8 *bnum, char *fname)
{
   static word32 maxbnum[2] = { WORD32_MAX, WORD32_MAX };
   int ecode;
   NODE node;

   /* initiate connection for file download */
   ecode = callserver(&node, ip);
   if (ecode) return ecode;
   /* set opcode and block number (as necessary) */
   if (bnum) {
      if (cmp64(bnum, maxbnum)) {
         put64(node.tx.blocknum, bnum);
         put16(node.tx.opcode, OP_GET_BLOCK);
      } else put16(node.tx.opcode, OP_GET_CBLOCK);
   } else {
      put64(node.tx.blocknum, node.tx.cblock);  /* for recv_file() */
      put16(node.tx.opcode, OP_GET_TFILE);
   }
   /* send request for block number, and recv into fname */
   ecode = send_tx(&node, STD_TIMEOUT);
   if (ecode == VEOK) ecode = recv_file(&node, fname);

   /* cleanup */
   sock_close(node.sd);
   node.sd = INVALID_SOCKET;
   return ecode;
}  /* end get_file() */

/**
 * Get an ip list from ip, and call addrecent() on the list.
 * Return VEOK if successful, else error code.
 * NOTE: DOES NOT call addrecent() when Rplist is full. */
int get_ipl(NODE *np, word32 ip)
{
   char ipaddr[16];  /* for threadsafe ntoa() usage */
   int ecode;

   pdebug("%s sending OP_GET_IPL...", ntoa(&ip, ipaddr));

   /* initiate connection with ip */
   ecode = callserver(np, ip);
   if (ecode == VEOK) {
      /* send OP_GET_IPL and receive single packet response */
      ecode = send_op(np, OP_GET_IPL);
      if (ecode == VEOK) ecode = recv_tx(np, STD_TIMEOUT);
      /* cleanup */
      sock_close(np->sd);
      np->sd = INVALID_SOCKET;
   }

   return ecode;
}  /* end get_ipl() */

/**
 * Get a blockhash of a particular block number from ip.
 * Uses node.tx.cblock from node when bnum is NULL.
 * Place returned hash in *blockhash.
 * Return VEOK if successful, else error code. */
int get_hash(NODE *np, word32 ip, void *bnum, void *blockhash)
{
   TX *tx;
   int ecode;
   char ipaddr[16];  /* for threadsafe ntoa() usage */

   pdebug("%s calling...", ntoa(&ip, ipaddr));
   if (callserver(np, ip) != VEOK) return VERROR;

   /* insert blocknum request */
   tx = &(np->tx);
   if (bnum == NULL) {
      pdebug("%s passing node's cblock to blocknum...", np->id);
      put64(tx->blocknum, tx->cblock);
   } else put64(tx->blocknum, bnum);

   /* perform OP_HASH request and receive -- close socket */
   pdebug("%s sending OP_HASH...", np->id);
   ecode = send_op(np, OP_HASH);
   if (ecode != VEOK) return ecode;
   ecode = recv_tx(np, STD_TIMEOUT);
   if (ecode != VEOK) return ecode;

   /* cleanup -- check response */
   sock_close(np->sd);
   np->sd = INVALID_SOCKET;
   if (get16(tx->opcode) != OP_HASH) {
      pdebug("%s unexpected opcode...", np->id);
      return VERROR;
   } else if (get16(tx->len) != HASHLEN) {
      pdebug("%s unexpected len...", np->id);
      return VERROR;
   }
   /* pass blockhash on success, if not NULL */
   if (blockhash) memcpy(blockhash, tx->buffer, HASHLEN);

   /* success */
   return VEOK;
}  /* end get_hash() */

/**
 * Handle an incoming packets from the Mochimo network. Reads a TX structure
 * from SOCKET sd.  Handles 3-way handshake and validates crc and id's.
 * Also cares for requests that do not need a child process.
 *
 * Returns:
 *          -1 no data yet
 *          0 to create child NODE to process read np->tx
 *          1 to close connection ("You're done, no child")
 *          2 ip was pinklisted (She was very naughty.)
 *
 * On entry: sd is non-blocking.
 *
 * Op sequence: OP_HELLO,OP_HELLO_ACK,OP_(?x)
*/
int gettx(NODE *np, SOCKET sd)
{
   char ipaddr[16];  /* for threadsafe ntoa() usage */
   int status;
   word16 opcode, id1, id2;
   TX *tx;

   /* init */
   tx = &np->tx;
   opcode = id1 = id2 = 0;
   memset(np, 0, sizeof(NODE));   /* clear structure */

   np->sd = sd;
   np->ip = get_sock_ip(sd);  /* uses getpeername() */
   ntoa(&np->ip, ipaddr);
   snprintf(np->id, sizeof(np->id), "%.15s %.02x~%.02x", ipaddr, id1, id2);
   pdebug("%s connected...", np->id);

   /* There are many ways to be bad...
    * Check pink lists... */
   if (pinklisted(np->ip)) {
      pdebug("%s dropped (pink)", np->id);
      Nbadlogs++;
      return VEBAD;
   }

   /* hello? */
   if (recv_tx(np, 1)) return VERROR;
   if (get16(tx->opcode) != OP_HELLO) goto bad1;

   /* hi! */
   np->id2 = id2 = rand16();
   np->id1 = id1 = get16(tx->id1);
   snprintf(np->id, sizeof(np->id), "%.15s %.02x~%.02x", ipaddr, id1, id2);
   put16(tx->opcode, OP_HELLO_ACK);
   if (send_tx(np, 1) != VEOK) return VERROR;

   /* how can I help you? */
   status = recv_tx(np, INIT_TIMEOUT);
   opcode = get16(tx->opcode);  /* execute() will check opcode */
   pdebug("%s got opcode = %d  status = %d", np->id, opcode, status);
   if (status == VEBAD) goto bad2;
   if (status != VEOK) return VERROR;  /* bad packet -- timeout? */
   opcode = get16(tx->opcode);  /* execute() will check opcode */
   pdebug("%s got opcode = %d  status = %d", np->id, opcode, status);
   if (!valid_op(opcode)) goto bad1;  /* she was a bad girl */

   /* check simple responses */
   switch (opcode) {
      case OP_GET_IPL: {
         /* only add those that "optin" */
         if (tx->version[1] & C_OPTIN) {
            addrecent(np->ip);
         }
         send_ipl(np);
         return 1;
      }
      case OP_TX: {
         Nlogins++;  /* raw TX in */
         status = process_tx(np);
         if (status != VEOK) {
            if (status == VEBAD2) goto bad1;
            if (status == VEBAD) goto bad2;
         } else if (tx->version[1] & C_OPTIN) {
            /* only add those that "optin" with a successful op */
            addrecent(np->ip);
         }

         return 1;
      }
      case OP_FOUND: {
         /* getblock child, catchup, re-sync, or ignore */
         if(Blockfound) return 1;  /* Already found one so ignore.  */
         status = contention(np);  /* Do we want this block? */
         if(status != 1) return 1; /* nothing to do: contention() fixed it */

         /* Get block */
         /* Check if bcon is running and if so stop her. */
         stop_bcon();
         stop_found();
         /* Now we can fetch the found block, validate it, and update. */
         Blockfound = 1;
         break;
      }
      case OP_BALANCE:     send_balance(np); return 1;
      case OP_RESOLVE:     /* send_resolve(np); */ return 1;
      case OP_GET_CBLOCK:  /* fallthrough */
      case OP_MBLOCK:      if (!Allowpush) return 1; break;
      case OP_HASH:        send_hash(np); return 1;
      case OP_IDENTIFY:    send_identify(np); return 1;
      case OP_BUSY:        /* fallthrough */
      case OP_NACK:        /* fallthrough */
      case OP_HELLO_ACK:   return 1;
      default: pdebug("%s requires child...", np->id);
   }

   /* If too many children in too small a space... */
   if (crowded(opcode)) return 1;  /* suppress child unless OP_FOUND */
   return VEOK;  /* success -- fork() child in server() */

bad1: epinklist(np->ip);
bad2: pinklist(np->ip);
      Nbadlogs++;
      pdebug("%s pinklisted, opcode = %d", np->id, opcode);

   return VEBAD;
}  /* end gettx() */

/**
 * Perform a network scan, refreshing Rplist[] with available nodes.
 * The highest advertised network hash, weight and bnum is placed in
 * @a *hash, @a *weight, and @a bnum, respectively.
 * Qualifying Quorum members are placed in quorum[qlen].
 * Returns number of qualifying quorum members, or number of
 * consensus nodes on the highest chain, if quorum is NULL. */
int scan_network
(word32 quorum[], word32 qlen, void *hash, void *weight, void *bnum)
{
   TX *tx;
   NODE node;
   word32 done = 0;
   word32 next = 0;
   word32 qcount = 0;
   word32 peer, *ipp;
   word16 len;
   word8 highhash[HASHLEN] = { 0 };
   word8 highweight[32] = { 0 };
   word8 highbnum[8] = { 0 };
   char weighthex[65], bnumhex[17];
   int result;

   plog("begin network scan... ");

   OMP_PARALLEL_(private(tx, node, peer, ipp, len, result))
   {
      while (Running && next < RPLISTLEN && Rplist[next]) {
         OMP_CRITICAL_()
         {
            peer = Rplist[next];
            if (peer) next++;
         }
         /* idle condition */
         if (peer == 0) {
            if (done < next) {
               millisleep(100);
               continue;
            }
            break;
         }
         /* get ip list from peer */
         if (get_ipl(&node, peer) == VEOK) {
            /* get ip list from TX */
            tx = &(node.tx);
            len = get16(tx->len);
            ipp = (word32 *) tx->buffer;
            OMP_CRITICAL_()
            {
               /* iterate peerlist adding to recent peers */
               for( ; len > 0; ipp++, len -= 4) {
                  if (Rplistidx >= RPLISTLEN) break;
                  if (*ipp == 0 || pinklisted(*ipp)) continue;
                  addrecent(*ipp);
               }
               /* check peer's chain weight against highweight */
               result = cmp256(tx->weight, highweight);
               if (result >= 0) {  /* higher or same chain detection */
                  if (result > 0) {  /* higher chain detection */
                     pdebug("new highweight");
                     memcpy(highhash, tx->cblockhash, HASHLEN);
                     memcpy(highweight, tx->weight, 32);
                     put64(highbnum, tx->cblock);
                     qcount = 0;
                     if (quorum) {
                        memset(quorum, 0, qlen);
                        pdebug("higher chain found, quourum reset...");
                     }
                  }  /* check block hash and add to quorum */
                  if (memcmp(tx->cblockhash, highhash, HASHLEN) >= 0) {
                     /* add ip to quorum, or q consensus */
                     if (quorum && qcount < qlen) {
                        quorum[qcount++] = peer;
                        pdebug("%s qualified", ntoa(&peer, NULL));
                     } else if (quorum == NULL) qcount++;
                  }
               }  /* end if higher or same chain */
            }  /* end OMP_CRITICAL_() */
         }  /* end if get_ipl() == VEOK */
         OMP_ATOMIC_()
            done++;
      }  /* end while() */
   }  /* end OMP_PARALLEL_() */
   pdebug("qualifying weight 0x...%s", weight2hex(highweight, weighthex));
   pdebug("qualifying block 0x%s", bnum2hex(highbnum, bnumhex));
   pdebug("qualifying nodes %d...", qcount);
   print_ipl(quorum, qcount);

   /* set highest hash, weight and block number */
   if (hash) memcpy(hash, highhash, HASHLEN);
   if (weight) memcpy(weight, highweight, 32);
   if (bnum) put64(bnum, highbnum);

   return qcount;
}  /* end scan_network() */

/* Refresh the ip list and send_found() to low-weight peer if needed.
 * Called from server().
 * Returns result code.
 */
int refresh_ipl(void)
{
   NODE node;
   int j, count;
   word32 ip, *ipp;
   word16 len;
   TX tx;
   word8 bnum[8];

   for(j = ip = 0; j < 1000 && ip == 0; j++)
      ip = Rplist[rand16() % RPLISTLEN];
   if(ip == 0) goto FAIL;
   if (get_ipl(&node, ip) == VEOK) {
      /* add iplist to recent peers */
      len = get16(node.tx.len);
      ipp = (word32 *) node.tx.buffer;
      for( ; len > 0; ipp++, len -= 4) {
         if (*ipp == 0) continue;
         if (Rplist[RPLISTLEN - 1]) break;
         addrecent(*ipp);
      }
   } else goto FAIL;
   /* Check peer's chain weight against ours. */
   if(cmp256(node.tx.weight, Weight) < 0) {
      /* get proof from tfile.dat */
      memset(tx.buffer, 0, sizeof(tx.buffer));
      if (sub64(Cblocknum, CL64_32(NTFTX), bnum)) memset(bnum, 0, 8);
      count = read_tfile(tx.buffer, bnum, NTFTX, "tfile.dat");
      /* Send found message to low weight peer */
      memset(tx.buffer, 0, sizeof(tx.buffer));
      if (read_tfile(tx.buffer, Cblocknum, 54, "tfile.dat") == 0) goto FAIL;
      if(callserver(&node, ip) != VEOK) goto FAIL;
      memcpy(&node.tx, &tx, sizeof(TX));  /* copy in tfile proof */
      put16(node.tx.len, (word16) count * sizeof(BTRAILER));
      send_op(&node, OP_FOUND);
      sock_close(node.sd);
   }

   /* success */
   return VEOK;

FAIL:
   return VERROR;
}  /* end refresh_ipl() */

/* end include guard */
#endif
