/**
 * node.c - Mochimo Node communication support
 *
 * Copyright (c) 2018-2021 by Adequate Systems, LLC.  All Rights Reserved.
 * For more information, please refer to ../LICENSE   *** NO WARRANTY ***
 *
 * Date: 18 February 2018
 * Revised: 10 November 2021
 *
*/

#ifndef MOCHIMO_NODE_C
#define MOCHIMO_NODE_C  /* include guard */


#include "network.h"

/* extended-c support */
#include "extlib.h"
#include "extinet.h"
#include "extmath.h"
#include "extprint.h"
#include "exttime.h"

/* crypto support */
#include "crc16.h"

/* mochimo support */
#include "config.h"
#include "util.h"

#include <string.h>

NODE Nodes[MAXNODES];  /* data structure for connected NODE's     */
NODE *Hi_node = Nodes; /* points one beyond last logged in NODE   */

word8 Cbits = CBITS;       /* Node capability bits */
word16 Dstport = PORT1;    /* Destination port (default 2095) */

static const char Metric[9][3] =
   { "", "K", "M", "G", "T", "P", "E", "Z", "Y" };


/* Search a list[] of 32-bit unsigned integers for a non-zero value.
 * A zero value marks the end of list (zero cannot be in the list).
 * Returns NULL if not found, else a pointer to value. */
word32 *search32(word32 val, word32 *list, unsigned len)
{
   for( ; len; len--, list++) {
      if(*list == 0) break;
      if(*list == val) return list;
   }
   return NULL;
}

/* Remove bad from list[maxlen]
 * Returns 0 if bad is not in list, else bad.
 * NOTE: *idx queue index is adjusted if idx is non-NULL. */
word32 remove32(word32 bad, word32 *list, unsigned maxlen, word32 *idx)
{
   word32 *bp, *end;

   bp = search32(bad, list, maxlen);
   if(bp == NULL) return 0;
   if(idx && &list[*idx] > bp) idx[0]--;
   for(end = &list[maxlen - 1]; bp < end; bp++) bp[0] = bp[1];
   *bp = 0;
   return bad;
}

/* Append a non-zero 32-bit unsigned integer to a list[].
 * Returns 0 if val was not added, else val.
 * NOTE: *idx queue index is always adjusted, as idx is required. */
word32 include32(word32 val, word32 *list, unsigned len, word32 *idx)
{
   if(idx == NULL || val == 0) return 0;
   if(search32(val, list, len) != NULL) return 0;
   if(idx[0] >= len) idx[0] = 0;
   list[idx[0]++] = val;
   return val;
}

/* Shuffle a list of < 64k 32-bit unsigned integers using Durstenfeld's
 * implementation of the Fisher-Yates shuffling algorithm.
 * NOTE: the shuffling length limitation is due to rand16fast(). */
void shuffle32(word32 *list, word32 len)
{
   word32 *ptr, *p2, temp;

   if (len < 2) return; /* list length is not long enough to shuffle, bail */
   while (list[--len] == 0 && len > 0);  /* determine non-zero list length */
   for(ptr = &list[len]; len > 1; len--, ptr--) {
      p2 = &list[rand16fast() % len];
      temp = *ptr;
      *ptr = *p2;
      *p2 = temp;
   }
}

/* Returns non-zero if ip is private, else 0. */
int isprivate(word32 ip)
{
   word8 *bp;

   bp = (word8 *) &ip;
   if(bp[0] == 10) return 1;  /* class A */
   if(bp[0] == 172 && bp[1] >= 16 && bp[1] <= 31) return 2;  /* class B */
   if(bp[0] == 192 && bp[1] == 168) return 3;  /* class C */
   if(bp[0] == 169 && bp[1] == 254) return 4;  /* auto */
   return 0;  /* public IP */
}

word32 addpeer(word32 ip, word32 *list, word32 len, word32 *idx)
{
   if(ip == 0) return 0;
   if(Noprivate && isprivate(ip)) return 0;  /* v.28 */
   if(search32(ip, list, len) != NULL) return 0;
   if(*idx >= len) *idx = 0;
   list[idx[0]++] = ip;
   return ip;
}

/**
 * Save the Rplist[] list to disk.
 * Returns VEOK on success, else VERROR */
int save_ipl(char *fname, word32 *list, word32 len)
{
   static char preface[] = "# Peer list (built by node)\n";
   char ipaddr[16];  /* for threadsafe ntoa() usage */
   word32 j;
   FILE *fp;

   pdebug("save_ipl(%s): saving...", fname);

   /* open file for writing */
   fp = fopen(fname, "w");
   if (fp == NULL) {
      perrno(errno, "save_ipl(%s): fopen failed", fname);
      return VERROR;
   };

   /* save non-zero entries */
   for(j = 0; j < len && list[j] != 0; j++) {
      ntoa(&list[j], ipaddr);
      if ((j == 0 && fwrite(preface, strlen(preface), 1, fp) != 1) ||
         (fwrite(ipaddr, strlen(ipaddr), 1, fp) != 1) ||
         (fwrite("\n", 1, 1, fp) != 1)) {
         fclose(fp);
         remove(fname);
         perr("save_ipl(%s): *** I/O error writing address line", fname);
         return VERROR;
      }
   }

   fclose(fp);
   plog("save_ipl(%s): recent peers saved", fname);
   return VEOK;
}  /* end save_ipl() */

/**
 * Read an IP list file, fname, into plist.
 * Valid lines in IP list include:
 *    host.domain.name
 *    1.2.3.4
 * Returns number of  on success, else VERROR. */
word32 read_ipl(char *fname, word32 *plist, word32 plistlen, word32 *plistidx)
{
   char buff[128];
   word32 count;
   FILE *fp;

   pdebug("read_ipl(%s): reading...", fname);
   count = 0;

   /* check valid fname and open for reading */
   if (fname == NULL || *fname == '\0') return VERROR;
   fp = fopen(fname, "r");
   if (fp == NULL) return VERROR;

   /* read file line-by-line */
   while(fgets(buff, 128, fp)) {
      if (strtok(buff, " #\r\n\t") == NULL) break;
      if (*buff == '\0') continue;
      if (addpeer(aton(buff), plist, plistlen, plistidx)) {
         pdebug("read_ipl(%s): added %s", fname, buff);
         count++;
      }
   }
   /* check for read errors */
   if (ferror(fp)) perr("read_ipl(%s): *** I/O error", fname);

   fclose(fp);
   return count;
}  /* end read_ipl() */

/* */
int readpeers(void)
{
   if(fexists("coreip.lst")) {
      pdebug("init_peers(): adding coreip.lst to Rplist[]...");
      read_ipl("coreip.lst", Rplist, RPLISTLEN, &Rplistidx);
   }
   if(fexists("start.lst")) {
      pdebug("init_peers(): adding start.lst to Rplist[]...");
      read_ipl("start.lst", Rplist, RPLISTLEN, &Rplistidx);
      remove("start.lst");
   }
   if(fexists("recentip.lst")) {
      pdebug("init_peers(): adding " "recentip.lst" " to Rplist[]...");
      read_ipl("recentip.lst", Rplist, RPLISTLEN, &Rplistidx);
   }
   if(fexists("trustedip.lst")) {
      pdebug("init_peers(): adding " "trustedip.lst" " to Rplist[] and Tplist[]...");
      read_ipl("trustedip.lst", Rplist, RPLISTLEN, &Rplistidx);
      read_ipl("trustedip.lst", Tplist, TPLISTLEN, &Tplistidx);
   }

   /* report node count */
   pdebug("init_peers(): Rplistidx= %" P32u, Rplistidx);
   pdebug("init_peers(): Tplistidx= %" P32u, Tplistidx);

   return VEOK;
}

/* Re-read epoch pink list from init(). */
int readpink(void)
{
   pdebug("reading epoch pink list...");
   return read_ipl("epink.lst", Epinklist, EPINKLEN, &Epinkidx);
}

/*
 * Save pink lists to disk.
 */
int savepink(void)
{
   int j;

   pdebug("saving epoch pink list...");

   /* save non-zero entries */
   for(j = 0; j < EPINKLEN; j++)
      if(Epinklist[j] == 0) break;

   save_ipl("epink.lst", Epinklist, j);
   return VEOK;
}  /* end savepink() */


int pinklisted(word32 ip)
{
   if(Disable_pink) return 0;   /* for debug */

   if(search32(ip, Cpinklist, CPINKLEN) != NULL
      || search32(ip, Lpinklist, LPINKLEN) != NULL
      || search32(ip, Epinklist, EPINKLEN) != NULL)
         return 1;
   return 0;
}


/* Add ip address to current pinklist.
 * Call pinklisted() first to check if already on list.
 */
int cpinklist(word32 ip)
{
   if(Cpinkidx >= CPINKLEN)
      Cpinkidx = 0;
   Cpinklist[Cpinkidx++] = ip;
   return VEOK;
}

/* Add ip address to current pinklist and remove it from
 * current and recent peer lists.
 * Checks the list first...
 */
int pinklist(word32 ip)
{
   pdebug("%s pink-listed", ntoa(&ip, NULL));

   if(!pinklisted(ip)) {
      if(Cpinkidx >= CPINKLEN)
         Cpinkidx = 0;
      Cpinklist[Cpinkidx++] = ip;
   }
   if(!Disable_pink) {
      remove32(ip, Rplist, RPLISTLEN, &Rplistidx);
   }
   return VEOK;
}  /* end pinklist() */


/* Add ip address to last pinklist.
 * Caller checks if already on list.
 */
int lpinklist(word32 ip)
{
   if(Lpinkidx >= LPINKLEN)
      Lpinkidx = 0;
   Lpinklist[Lpinkidx++] = ip;
   return VEOK;
}


int epinklist(word32 ip)
{
   if(Epinkidx >= EPINKLEN) {
      pdebug("Epoch pink list overflow");
      Epinkidx = 0;
   }
   Epinklist[Epinkidx++] = ip;
   return VEOK;
}


/* Call after each epoch.
 * Merges current pink list into last pink list
 * and purges current pink list.
 */
void mergepinklists(void)
{
   int j;
   word32 ip, *ptr;

   for(j = 0; j < CPINKLEN; j++) {
      ip = Cpinklist[j];
      if(ip == 0) continue;  /* empty */
      ptr = search32(ip, Lpinklist, LPINKLEN);
      if(ptr == NULL) lpinklist(ip);  /* add to last bad list */
      Cpinklist[j] = 0;
   }
   Cpinkidx = 0;
}


/* Erase Epoch Pink List */
void purge_epoch(void)
{
   pdebug("   purging epoch pink list");
   unlink("epink.lst");
   memset(Epinklist, 0, sizeof(Epinklist));
   Epinkidx = 0;
}

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
   hash = crc16(tx, TXCRC_COUNT);
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
   put16(tx->crc16, crc16(tx, TXCRC_COUNT));

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


/* Process OP_HASH.  Return VEOK on success, else VERROR.
 * Called by gettx().
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

/**
 * Used for simple one packet responses like OP_GET_IPL.
 * Assumes socket np->sd is connected and non-blocking.
 * Closes socket and sets np->sd to INVALID_SOCKET on return.
 * Returns VEOK on success, else VERROR. */
int get_op(NODE *np, word16 opcode)
{
   int ecode = VEOK;

   /* send and receive single packet */
   ecode = send_op(np, opcode);
   if(ecode == VEOK) ecode = recv_tx(np, STD_TIMEOUT);
   /* cleanup */
   sock_close(np->sd);
   np->sd = INVALID_SOCKET;

   return ecode;
}  /* end get_op() */

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
   pdebug("callserver(%s): connecting...", np->id);
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
   pdebug("callserver(%s): initiating handshake...", np->id);
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
   pdebug("callserver(%s): validating handshake...", np->id);
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
   char ipaddr[16];  /* for threadsafe ntoa() usage */
   TX *tx;

   pdebug("get_hash(%s): calling...", ntoa(&ip, ipaddr));
   if (callserver(np, ip) != VEOK) return VERROR;

   /* insert blocknum request */
   tx = &(np->tx);
   if (bnum == NULL) {
      pdebug("get_hash(%s): passing node's cblock to blocknum...", np->id);
      put64(tx->blocknum, tx->cblock);
   } else put64(tx->blocknum, bnum);

   pdebug("get_hash(%s): sending OP_HASH...", np->id);
   if(get_op(np, OP_HASH) == VEOK) {  /* closes socket */
      if (get16(tx->opcode) != OP_HASH) {
         pdebug("get_hash(%s): unexpected opcode...", np->id);
         return VERROR;
      }
      if (get16(tx->len) != HASHLEN) {
         pdebug("get_hash(%s): unexpected len...", np->id);
         return VERROR;
      }
      if (blockhash) memcpy(blockhash, TRANBUFF(tx), HASHLEN);
      return VEOK;
   }

   return VERROR;
}  /* end get_hash() */

ThreadProc thread_get_ipl(void *arg)
{
   THREAD_SCAN_ARGS *args = (THREAD_SCAN_ARGS *) arg;
   NODE node;
   int res;

   res = callserver(&node, args->ip);
   res = res ? res : get_op(&node, OP_GET_IPL);
   if (res == VEOK) memcpy(&(args->tx), &(node.tx), sizeof(TX));

   args->tr = res;
   args->ts = 1;
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
   THREAD_SCAN_ARGS scan[MAXNODES];
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
      for(j = 0; j < MAXNODES; j++) {
         if (tid[j] > 0 && scan[j].ts) {
            /* thread is finished */
            result = thread_join(tid[j]);
            if (result != VEOK) perrno(result, "thread_join()");
            if ((scan[j].tr) == VEOK) {  /* get ip list from TX */
               len = get16(scan[j].tx.len);
               ipp = (word32 *) TRANBUFF(&(scan[j].tx));
               for( ; len > 0; ipp++, len -= 4) {
                  if (*ipp == 0) continue;
                  if (Rplist[RPLISTLEN - 1]) break;
                  addrecent(*ipp);
               }  /* check peer's chain weight against highweight */
               result = cmp256(scan[j].tx.weight, highweight);
               if (result >= 0) {  /* higher or same chain detection */
                  if (result > 0) {  /* higher chain detection */
                     pdebug("scan_network(): new highweight");
                     memcpy(highhash, scan[j].tx.cblockhash, HASHLEN);
                     memcpy(highweight, scan[j].tx.weight, 32);
                     put64(highbnum, scan[j].tx.cblock);
                     count = 0;
                     if (quorum) {
                        memset(quorum, 0, qlen);
                        pdebug("scan_network(): higher chain found, quourum reset...");
                     }
                  }  /* check block hash and add to quorum */
                  if (cmp256(scan[j].tx.cblockhash, highhash) >= 0) {
                     /* add ip to quorum, or count consensus */
                     if (quorum && count < qlen) {
                        quorum[count++] = scan[j].ip;
                        ntoa(&(scan[j].ip), ipaddr);
                        pdebug("scan_network(): %s qualified", ipaddr);
                     } else if (quorum == NULL) count++;
                  }
               }
            }  /* clear thread data */
            scan[j].ts = 0;
            tid[j] = 0;
            done++;
         } else if (tid[j] == 0) {
            if (next < Rplistidx && Rplist[next]) {
               scan[j].ip = Rplist[next];
               scan[j].ts = 0;
               result = thread_create(&(tid[j]), &thread_get_ipl, &scan[j]);
               if (result != VEOK) {
                  perrno(result, "thread_create()");
                  scan[j].ts = 0;
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

/* end include guard */
#endif
