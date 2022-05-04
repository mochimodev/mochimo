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
#include "util.h"
#include "tx.h"
#include "tfile.h"
#include "tag.h"
#include "sync.h"
#include "ledger.h"
#include "global.h"

/* external support */
#include <string.h>
#include "exttime.h"
#include "extthread.h"
#include "extprint.h"
#include "extmath.h"
#include "extlib.h"
#include "extinet.h"
#include "crc16.h"

#define valid_op(op)  ((op) >= FIRST_OP && (op) <= LAST_OP)
#define crowded(op)   (Nonline > (MAXNODES - 5) && (op) != OP_FOUND)
#define can_fork_tx() (Nonline <= (MAXNODES - 5))

NODE Nodes[MAXNODES];  /* data structure for connected NODE's     */
NODE *Hi_node = Nodes; /* points one beyond last logged in NODE   */

static const char Metric[8][3] = { "", "K", "M", "G", "T", "P", "E", "Z" };

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
      perr("getslot(): Nodes[] full!");
      Nspace++;
      return NULL;
   }

   Nonline++;    /* number of currently connected sockets */
   pdebug("getslot() added NODE %d", (int) (newnp - Nodes));
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
   if(np->pid == 0)
      return perr("*** NODE %ld already called freeslot() ***",
                   (long) (np - Nodes));
   pdebug("freeslot(): idx=%d  ip = %s pid = %d", (int) (np - Nodes),
           ntoa(&np->ip, NULL), (int) np->pid);
   Nonline--;
   np->pid = 0;
   /* Update pointer to just beyond highest used slot in Nodes[] */
   while(Hi_node > Nodes && (Hi_node - 1)->pid == 0)
      Hi_node--;
   if(Nonline < 0) { Nonline = 0; return perr("Nonline < 0"); }
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
   pdebug("child_status(): pid = %d  status = 0x%x", pid, status);
   if(pid > 0) {  /* child existed and called exit() */
      if(WIFEXITED(status)) {
         status = WEXITSTATUS(status);
         if(status != 0) {
            if(status >= 2) pinklist(np->ip);
            if(status >= 3) epinklist(np->ip);
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
   int ecode;
   word16 hash;
   TX *tx;

   /* init recv_tx() */
   tx = &(np->tx);

   /* recv tx packet */
   ecode = sock_recv(np->sd, tx, TXBUFFLEN, 0, timeout);
   if (ecode != VEOK) {
      if (ecode == VETIMEOUT) {
         pdebug("recv_tx(%s): connection timed out", np->id);
         Ntimeouts++;
      } else {
         pdebug("recv_tx(%s): aborting", np->id);
         Nrecverrs++;
      }
      return ecode;
   }

   /* compute crc16 checksum and verify packet integrity */
   hash = crc16(tx, TXCRC_INLEN);
   if (get16(tx->crc16) != hash) {
      pdebug("recv_tx(%s): *** CRC16 mismatch, 0x%" P16X " != 0x%" P16X,
         np->id, get16(tx->crc16), hash);
      Nrecverrs++;
      return VEBAD;
   }
   /* check packet network protocol version */
   if (get16(tx->network) != TXNETWORK) {
      pdebug("recv_tx(%s): *** invalid network, %" P16u " != %" P16u,
         np->id, get16(tx->network), TXNETWORK);
      Nrecverrs++;
      return VEBAD;
   }
   /* check packet trailer */
   if (get16(tx->trailer) != TXEOT) {
      pdebug("recv_tx(%s): *** invalid trailer, 0x%" P16X " != 0x%" P16X,
         np->id, get16(tx->trailer), TXEOT);
      Nrecverrs++;
      return VEBAD;
   }
   /* check handshake IDs on all operations (except during handshake) */
   if (get16(tx->opcode) >= FIRST_OP) {
      if (np->id1 != get16(tx->id1) || np->id2 != get16(tx->id2)) {
         pdebug("recv_tx(%s): *** unexpected ID 0x%" P32x, np->id,
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
   static long mlen = (long) sizeof(Metric);
   long expect, current, lastsec, persec, eta, mps, m;
   time_t start, now;
   float percent;
   word16 len;
   FILE *fp;
   TX *tx;

   /* init recv_file() */
   tx = &(np->tx);
   start = time(&now);
   percent = expect = current = lastsec = persec = eta = mps = m = 0;
   if (get16(np->tx.opcode) == OP_GET_TFILE) {
      expect = (long) get32(np->tx.blocknum) * sizeof(BTRAILER);
   }

   /* open file for writing recv'd data */
   fp = fopen(fname, "wb");
   if(fp == NULL) {
      perrno(errno, "recv_file(%s, %s): fopen() failed", np->id, fname);
      return VERROR;
   }

   /* receive packets and write */
   pdebug("recv_file(%s, %s): receiving...", np->id, fname);
   while(recv_tx(np, STD_TIMEOUT) == VEOK) {
      /* update progress */
      current = ftell(fp);
      if (difftime(time(NULL), now)) {
         now = time(NULL);
         mps = persec = current - lastsec;
         for(m = 0; mps > 999 && (m + 1) < mlen; mps /= 1000, m++);
      }
      /* print sticky progress */
      if (expect) {
         percent = 100.0 * current / expect;
         eta = persec ? expect / persec : 0;
         psticky("Downloading %s... %.02f%% (%ld%sB/s) | ETA: %lds",
            fname, percent, mps, Metric[m], eta);
      } else {
         psticky("Downloading %s... %ld (%ld%sB/s) | Elapsed: %gs",
            fname, current, mps, Metric[m], difftime(start, now));
      }
      /* check recv'd packet */
      if(get16(tx->opcode) != OP_SEND_FILE) {
         pdebug("recv_file(%s, %s): *** invalid opcode", np->id, fname);
         break;
      }
      len = get16(tx->len);
      if(len > TRANLEN) {
         pdebug("recv_file(%s, %s): *** oversized TX length", np->id, fname);
         break;
      }
      if(len && fwrite(TRANBUFF(tx), len, 1, fp) != 1) {
         pdebug("recv_file(%s, %s): *** I/O error", np->id, fname);
         break;
      }
      /* check EOF */
      if(len < TRANLEN) {
         fclose(fp);
         psticky("");
         pdebug("recv_file(%s, %s): EOF", np->id, fname);
         return VEOK;
      } /* end if EOF */
   }  /* end for */
   fclose(fp);
   remove(fname);  /* delete partial downloads */
   psticky("");
   return VERROR;
}  /* end recv_file() */

/**
 * Send next packet to NODE *np.
 * Set advertised fields and compute CRC16.
 * Returns VEOK on success, else VERROR. */
int send_tx(NODE *np, double timeout)
{
   int ecode;
   TX *tx;

   /* init send_tx() */
   tx = &(np->tx);

   /* fill tx packet with relevant information... */
   put16(tx->version, PVERSION | (Cbits << 8));
   put16(tx->network, TXNETWORK);
   put16(tx->trailer, TXEOT);
   put16(tx->id1, np->id1);
   put16(tx->id2, np->id2);
   put64(tx->cblock, Cblocknum);
   memcpy(tx->cblockhash, Cblockhash, HASHLEN);
   memcpy(tx->pblockhash, Prevhash, HASHLEN);
   /* ... but, do not overwrite TX ip map */
   if(get16(tx->opcode) != OP_TX) {
      memcpy(tx->weight, Weight, HASHLEN);
   }

   /* compute packet crc16 checksum */
   put16(tx->crc16, crc16(tx, TXCRC_INLEN));

   /* send tx packet */
   ecode = sock_send(np->sd, tx, TXBUFFLEN, 0, timeout);
   if (ecode != VEOK) {
      if (ecode == VETIMEOUT) {
         pdebug("send_tx(%s): connection timed out", np->id);
         Ntimeouts++;
      } else {
         pdebug("send_tx(%s): aborting", np->id);
         Nsenderrs++;
      }
      return ecode;
   }

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
 * Send packets to NODE *np, and write to file, fname.
 * SOCKET np->sd is set non-blocking, ready to recv data.
 * Set fname NULL send np->tx.blocknum request.
 * Returns: VEOK (0) = good, else error code. */
int send_file(NODE *np, char *fname)
{
   char name[32];
   int ecode;
   word16 len;
   FILE *fp;
   TX *tx;

   /* init send_file() */
   len = TRANLEN;
   tx = &(np->tx);
   if (fname == NULL) {
      sprintf(name, "%.6s/b%.16s.bc", Bcdir, bnum2hex(tx->blocknum));
      fname = name;
   }
   pdebug("send_file(%s, %s): sending...", np->id, fname);

   /* open file for writing recv'd data */
   fp = fopen(fname, "rb");
   if(fp == NULL) {
      perrno(errno, "send_file(%s, %s): fopen() failed", np->id, fname);
      /* send unable to deliver request acknowledgement */
      put16(tx->opcode, OP_NACK);
      send_tx(np, STD_TIMEOUT);
      return VERROR;
   }
   /* read and send packets */
   do {
      len = fread(TRANBUFF(tx), 1, TRANLEN, fp);
      put16(tx->len, len);
      ecode = send_op(np, OP_SEND_FILE);
      /* Make upload bandwidth dynamic. */
      if(Nonline > 1) millisleep((Nonline - 1) * UBANDWIDTH);
   } while(ecode == VEOK && len == TRANLEN);
   /* check for errors */
   if (len < TRANLEN) {
      if (feof(fp)) pdebug("send_file(%s, %s): EOF", np->id, fname);
      else {  /* likely file error */
         perr("send_file(%s, %s): *** I/O error", np->id, fname);
         ecode = VERROR;
      }
   }
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
   memset(np->tx.send_total, 0, 8);
   memset(np->tx.change_total, 0, 8);
   /* check for old OP_BALANCE Request with ZEROED Tag */
   if(len == 0 && ((word8 *) (np->tx.src_addr))[2196] == 0x00) {
     len = TXADDRLEN - 12;
   }
   /* look up source address in ledger */
   if(le_find(np->tx.src_addr, &le, NULL, len) == TRUE) {
     put64(np->tx.send_total, le.balance);
     put64(np->tx.change_total, One); /* indicate address was found */
     memcpy(np->tx.src_addr, le.addr, TXADDRLEN); /* return found address */
   }
   send_op(np, OP_SEND_BAL);

   Nbalance++;
   return 0;  /* success */
} /* end send_balance() */

/* Send our recent peer list to NODE np in response to OP_GETIPL.
 * Called from execute().
 */
int send_ipl(NODE *np)
{
   memset(TRANBUFF(&np->tx), 0, TRANLEN);
   /* copy recent peer list to TX */
   memcpy(TRANBUFF(&np->tx), Rplist, RPLISTLEN);
   put16(np->tx.len, RPLISTLEN);
   return send_op(np, OP_SEND_IPL);  /* send ip list */
}

/* Process OP_HASH.  Return VEOK on success, else VERROR.
 * Called by handle_tx().
 */
int send_hash(NODE *np)
{
   BTRAILER bt;
   char fname[128];

   sprintf(fname, "%s/b%s.bc", Bcdir, bnum2hex(np->tx.blocknum));
   if(readtrailer(&bt, fname) != VEOK) return VERROR;
   memset(TRANBUFF(&np->tx), 0, TRANLEN);
   /* copy hash of tx.blocknum to TX */
   memcpy(TRANBUFF(&np->tx), bt.bhash, HASHLEN);
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
   unlink(fname);
   return status;
}  /* end send_tf() */


int send_identify(NODE *np)
{
   memset(TRANBUFF(&np->tx), 0, TRANLEN);
   /* copy recent peer list to TX */
   sprintf((char *) TRANBUFF(&np->tx), "Sanctuary=%u,Lastday=%u,Mfee=%u",
           Sanctuary, Lastday, Myfee[0]);
   return send_op(np, OP_IDENTIFY);
}

/**
 * Look-up and return an address tag to np.
 * Called from handle_tx() opcode == OP_RESOLVE
 *
 * on entry:
 *     tag string at ADDR_TAG_PTR(np->tx.dst_addr)    tag to query
 * on return:
 *     np->tx.send_total = 1 if found, or 0 if not found.
 *     if found: np->tx.dst_addr has full found address with tag.
 *               np->tx.change_total has balance.
 *
 * Returns VEOK if found, else VERROR.
*/
int send_resolve(NODE *np)
{
   word8 foundaddr[TXADDRLEN];
   static word8 zeros[8];
   word8 balance[TXAMOUNT];
   int status, ecode = VERROR;

   put64(np->tx.send_total, zeros);
   put64(np->tx.change_total, zeros);
   /* find tag in leger.dat */
   status = tag_find(np->tx.dst_addr, foundaddr, balance, get16(np->tx.len));
   if(status == VEOK) {
      memcpy(np->tx.dst_addr, foundaddr, TXADDRLEN);
      memcpy(np->tx.change_total, balance, TXAMOUNT);
      put64(np->tx.send_total, One);
      ecode = VEOK;
   }
   send_op(np, OP_RESOLVE);
   return ecode;
}  /* end send_resolve() */

/* Creates child to send OP_FOUND to all recent peers */
int send_found(void)
{
   word32 *ipp;
   NODE node;
   BTRAILER bt;
   char fname[128];
   int ecode;
   TX tx;

   if (Found_pid) {
      plog("send_found() is already running -- rerun it.");
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
      sprintf(fname, "%s/b%s.bc", Bcdir, bnum2hex(Cblocknum));
      ecode = 2;
      if(readtrailer(&bt, fname) != VEOK
         || cmp64(Cblocknum, bt.bnum) != 0) {
bad:
         exit(perr("send_found(): ecode: %d", ecode));
      }
      ecode = 3;
      if(memcmp(Prevhash, bt.bhash, HASHLEN)) goto bad;
      memcpy(Cblockhash, bt.bhash, HASHLEN);
      memcpy(Prevhash, bt.phash, HASHLEN);
   }  /* end if NG block v.23 */

   if(Trace)
      plog("send_found(0x%s)", bnum2hex(Cblocknum));

   loadproof(&tx);  /* get proof from tfile.dat */
   /* Send found message to recent peers */
   shuffle32(Rplist, RPLISTLEN);
   for(ipp = Rplist; ipp < &Rplist[RPLISTLEN] && Running; ipp++) {
      if(*ipp == 0) continue;
      if(callserver(&node, *ipp) != VEOK) continue;
      memcpy(&node.tx, &tx, sizeof(TX));  /* copy in tfile proof */
      send_op(&node, OP_FOUND);
      sock_close(node.sd);
   }
   /* Send found message to trusted peers */
   for(ipp = Tplist; ipp < &Tplist[TPLISTLEN] && Running; ipp++) {
      if(*ipp == 0) continue;
      if(callserver(&node, *ipp) != VEOK) continue;
      memcpy(&node.tx, &tx, sizeof(TX));  /* copy in tfile proof */
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
   np->sd = INVALID_SOCKET;
   np->ip = np->id1 = np->id2 = 0;
   memset(&(np->tx), 0, sizeof(TX));   /* clear structure */
   sprintf(np->id, "%.15s %02x~%02x", ipaddr, id1, id2);
   /* begin connection */
   np->ip = ip;
   np->sd = sock_connect_ip(ip, Dstport, INIT_TIMEOUT);
   if(np->sd == INVALID_SOCKET) {
      pdebug("callserver(%s): failed to connect", np->id);
      return VERROR;
   }
   /* initiate Three-Way Handshake */
   np->id1 = rand16();
   id1 = (word8) (np->id1 >> 8);
   put16(np->tx.opcode, OP_HELLO);
   sprintf(np->id, "%.15s %02x~%02x", ipaddr, id1, id2);
   if(send_tx(np, ACK_TIMEOUT) || recv_tx(np, ACK_TIMEOUT)) {
      pdebug("callserver(%s): *** incomplete handshake", np->id);
      sock_close(np->sd);
      np->sd = INVALID_SOCKET;
      return VERROR;
   }
   /* validate Three-Way Handshake */
   np->id2 = get16(np->tx.id2);
   id2 = (word8) np->id2;
   sprintf(np->id, "%.15s %02x~%02x", ipaddr, id1, id2);
   if(get16(np->tx.opcode) != OP_HELLO_ACK || get16(np->tx.id1) != np->id1) {
      pdebug("callserver(%s): *** invalid handshake", np->id);
      sock_close(np->sd);
      np->sd = INVALID_SOCKET;
      return VEBAD;
   }

   return VEOK;  /* made a new friend */
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
   char ipaddr[16];  /* for threadsafe ntoa() usage */
   int ecode;
   NODE node;

   /* init get_file() */
   pdebug("get_file(%s, %s%s, %s): entered...", ntoa(&ip, ipaddr),
      bnum ? "0x" : "", bnum ? bnum2hex(bnum) : "Tfile", fname);

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
   word16 len;
   word32 *ipp;

   pdebug("get_ipl(%s): sending get_tx(OP_GET_IPL)", ntoa(&ip, ipaddr));
   if(get_tx(np, ip, OP_GET_IPL) == VEOK) {  /* closes socket */
      len = get16(np->tx.len);  /* ^^ get_tx() checks len <= TRANLEN */
      for(ipp = (word32 *) TRANBUFF(&np->tx); len > 0; ipp++, len -= 4) {
         if(*ipp) {
            if (Rplist[RPLISTLEN - 1]) {
               pdebug("get_ipl(%s): Rplist[] is full...", np->id);
               break;
            } else if (addrecent(*ipp)) {
               pdebug("get_ipl(%s): TX added %s", np->id, ntoa(ipp, ipaddr));
            }
         }
      }
      return VEOK;
   }
   return VERROR;
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

   pdebug("get_hash(%s): calling...", ntoa(&ip, ipaddr));
   if (callserver(np, ip) != VEOK) return VERROR;

   /* insert blocknum request */
   tx = &(np->tx);
   if (bnum == NULL) {
      pdebug("get_hash(%s): passing node's cblock to blocknum...", np->id);
      put64(tx->blocknum, tx->cblock);
   } else put64(tx->blocknum, bnum);

   /* perform OP_HASH request and receive -- close socket */
   pdebug("get_hash(%s): sending OP_HASH...", np->id);
   ecode = send_op(np, OP_HASH) || recv_tx(np, STD_TIMEOUT);
   sock_close(np->sd);
   np->sd = INVALID_SOCKET;

   if (ecode == VEOK) {
      /* check response validity */
      if (get16(tx->opcode) != OP_HASH) {
         pdebug("get_hash(%s): unexpected opcode...", np->id);
         return VERROR;
      } else if (get16(tx->len) != HASHLEN) {
         pdebug("get_hash(%s): unexpected len...", np->id);
         return VERROR;
      }
      /* pass blockhash on success, if not NULL */
      if (blockhash) memcpy(blockhash, TRANBUFF(tx), HASHLEN);
   }

   return ecode;
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
int handle_tx(NODE *np, SOCKET sd)
{
   char ipaddr[16];  /* for threadsafe ntoa() usage */
   int status;
   word32 ip;
   word16 opcode, id1, id2;
   TX *tx;

   /* init */
   tx = &np->tx;
   ip = opcode = id1 = id2 = 0;
   memset(np, 0, sizeof(NODE));   /* clear structure */

   np->sd = sd;
   np->ip = ip = get_sock_ip(sd);  /* uses getpeername() */
   sprintf(np->id, "%.15s %02x~%02x", ntoa(&ip, ipaddr), id1, id2);
   pdebug("handle_tx(%s): connected...", np->id);

   /* There are many ways to be bad...
    * Check pink lists... */
   if (pinklisted(np->ip)) {
      pdebug("handle_tx(%s): dropped (pink)", np->id);
      Nbadlogs++;
      return VEBAD;
   }

   /* hello? */
   if (recv_tx(np, INIT_TIMEOUT)) return VERROR;
   if (get16(tx->opcode) != OP_HELLO) goto bad1;

   /* hi! */
   np->id2 = id2 = rand16();
   np->id1 = id1 = get16(tx->id1);
   if (send_op(np, OP_HELLO_ACK) != VEOK) return VERROR;

   /* how can I help you? */
   status = recv_tx(np, INIT_TIMEOUT);
   opcode = get16(tx->opcode);  /* execute() will check opcode */
   sprintf(np->id, "%.15s %02x~%02x", ipaddr, id1, id2);
   pdebug("handle_tx(%s): got opcode = %d  status = %d", np->id, opcode, status);
   if (status == VEBAD) goto bad2;
   if (status != VEOK) return VERROR;  /* bad packet -- timeout? */
   if (!valid_op(opcode)) goto bad1;  /* she was a bad girl */

   /* check simple responses */
   switch (opcode) {
      case OP_GET_IPL:
         send_ipl(np);
         if (get16(np->tx.len) == 0) {  /* do not add wallets */
            addrecent(np->ip);
         }
         return 1;
      case OP_TX:
         if (txcheck(tx->src_addr) != VEOK) {
            pdebug("got dup src_addr");
            Ndups++;
            return 1;
         }
         Nlogins++;  /* raw TX in */
         status = process_tx(np);
         if (status > 2) goto bad1;
         if (status > 1) goto bad2;
         if (get16(np->tx.len) == 0) {  /* do not add wallets */
            addrecent(np->ip);
         }
         return 1;
      case OP_FOUND:
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
      case OP_BALANCE:
         send_balance(np);
         return 1;
      case OP_RESOLVE:
         send_resolve(np);
         send_op(np, OP_RESOLVE);
         return 1;
      case OP_GET_CBLOCK:
         if (!Allowpush || !fexists("miner.tmp")) return 1;
         break;
      case OP_MBLOCK:
         if (!Allowpush || (time(NULL) - Pushtime) < 150) return 1;
         Pushtime = time(NULL);
         break;
      case OP_HASH:
         send_hash(np);
         return 1;
      case OP_IDENTIFY:
         send_identify(np);
         return 1;
      case OP_BUSY:
      case OP_NACK:
      case OP_HELLO_ACK:
         return 1;
      default:
         pdebug("handle_tx(%s): requires child...", np->id);
   }

   /* If too many children in too small a space... */
   if (crowded(opcode)) return 1;  /* suppress child unless OP_FOUND */
   return VEOK;  /* success -- fork() child in server() */

bad1: epinklist(np->ip);
bad2: pinklist(np->ip);
      Nbadlogs++;
      pdebug("handle_tx(%s): pinklisted, opcode = %d", np->id, opcode);

   return VEBAD;
}  /* end handle_tx() */

ThreadProc thread_get_ipl(void *arg)
{
   NODE *np = (NODE *) arg;

   /* embed result and done indicator in np->ts */
   np->ts = (get_tx(np, np->ip, OP_GET_IPL) << 8) | 1;
   Unthread;
}

/**
 * Perform a network scan, refreshing Rplist[] with available nodes.
 * The highest advertised network weight is placed in *highweight.
 * Qualifying Quorum members are placed in quorum[qlen].
 * Returns number of qualifying quorum members, or number of
 * consensus nodes on the highest chain, if quorum is NULL. */
int scan_network
(word32 quorum[], word32 qlen, void *hash, void *weight, void *bnum)
{
   NODE node[MAXNODES] = { 0 };
   ThreadId tid[MAXNODES] = { 0 };
   char ipaddr[16];  /* for threadsafe ntoa() usage */
   char progress[BUFSIZ];
   int j, result;
   word32 done, next, count;
   word8 highhash[HASHLEN] = { 0 };
   word8 highweight[32] = { 0 };
   word8 highbnum[8] = { 0 };
   word32 *ipp;
   word16 len;
   time_t start;
   float percent;

   pdebug("scan_network(): begin scan... ");
   done = next = count = 0;
   time(&start);
   do {
      /* update sticky progress */
      percent = 100.0 * done / Rplistidx;
      sprintf(progress, "Network Scan %.2f%% (%d/%d) | Elapsed %.0fs",
         percent, done, Rplistidx, difftime(time(NULL), start));
      psticky(progress);
      /* check threads */
      for (j = 0; j < MAXNODES; j++) {
         /* check thread status */
         if (tid[j] > 0 && node[j].ts) {
            /* thread is finished */
            result = thread_join(tid[j]);
            if (result != VEOK) perrno(result, "thread_join()");
            if ((node[j].ts >> 8) == VEOK) {
               /* get ip list from TX */
               len = get16(node[j].tx.len);
               ipp = (word32 *) TRANBUFF(&(node[j].tx));
               for( ; len > 0; ipp++, len -= 4) {
                  if (*ipp == 0) continue;
                  if (Rplist[RPLISTLEN - 1]) break;
                  addrecent(*ipp);
               }  /* check peer's chain weight against highweight */
               result = cmp256(node[j].tx.weight, highweight);
               if (result >= 0) {  /* higher or same chain detection */
                  if (result > 0) {  /* higher chain detection */
                     pdebug("scan_network(): new highweight");
                     memcpy(highhash, node[j].tx.cblockhash, HASHLEN);
                     memcpy(highweight, node[j].tx.weight, 32);
                     put64(highbnum, node[j].tx.cblock);
                     count = 0;
                     if (quorum) {
                        memset(quorum, 0, qlen);
                        pdebug("scan_network(): higher chain found, quourum reset...");
                     }
                  }  /* check block hash and add to quorum */
                  if (cmp256(node[j].tx.cblockhash, highhash) >= 0) {
                     /* add ip to quorum, or count consensus */
                     if (quorum && count < qlen) {
                        quorum[count++] = node[j].ip;
                        ntoa(&(node[j].ip), ipaddr);
                        pdebug("scan_network(): %s qualified", ipaddr);
                     } else if (quorum == NULL) count++;
                  }
               }
            }  /* clear thread data */
            node[j].ts = 0;
            tid[j] = 0;
            done++;
         } else if (tid[j] == 0) {
            if (next < Rplistidx && Rplist[next]) {
               node[j].ip = Rplist[next];
               node[j].ts = 0;
               result = thread_create(&(tid[j]), &thread_get_ipl, &node[j]);
               if (result != VEOK) {
                  perrno(result, "thread_create()");
                  node[j].ts = 0;
                  tid[j] = 0;
               } else next++;
            }
         }
      }
      millisleep(1);
      if (!Running) {
         psticky("");
         plog("Waiting for scanning threads to finish...");
         thread_multijoin(tid, MAXNODES);
         break;
      }
   } while(done < next || (next < Rplistidx && Rplist[next]));
   pdebug("scan_network(): found %d qualifying nodes...", count);
   pdebug("scan_network(): qualifying weight 0x%s", weight2hex(highweight));
   pdebug("scan_network(): qualifying block 0x%s", bnum2hex(highbnum));
   psticky("");

   /* set highest weight and block number */
   if (hash) memcpy(hash, highhash, HASHLEN);
   if (weight) memcpy(weight, highweight, 32);
   if (bnum) put64(bnum, highbnum);

   return count;
}  /* end scan_network() */

/* Refresh the ip list and send_found() to low-weight peer if needed.
 * Called from server().
 * Returns result code.
 */
int refresh_ipl(void)
{
   NODE node;
   int j, message = 0;
   word32 ip;
   TX tx;

   for(j = ip = 0; j < 1000 && ip == 0; j++)
      ip = Rplist[rand16() % RPLISTLEN];
   if(ip == 0) BAIL(1);
   if(get_ipl(&node, ip) != VEOK) BAIL(2);
   /* Check peer's chain weight against ours. */
   if(cmp256(node.tx.weight, Weight) < 0) {
      /* Send found message to low weight peer */
      loadproof(&tx);  /* get proof from tfile.dat */
      if(callserver(&node, ip) != VEOK) BAIL(3);
      memcpy(&node.tx, &tx, sizeof(TX));  /* copy in tfile proof */
      send_op(&node, OP_FOUND);
      sock_close(node.sd);
   }
bail:
   pdebug("refresh_ipl(): %d", message);
   return message;
}  /* end refresh_ipl() */

/* end include guard */
#endif
