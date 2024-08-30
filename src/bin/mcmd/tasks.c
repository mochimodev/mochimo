
#include "error.h"
#include "peer.h"
#include "protocol.h"
#include "server.h"

static NODE *Syncwait;

/* Quorum peer list parameters -- + bnum, bhash, weight */
static word8 Qbnum[8], Qbhash[HASHLEN], Qweight[32];
static word32 *Qplist, Qplistidx, Qplistlen;
/* Scanned peer list parameters -- for quorum scan */
static word32 *Splist, Splistidx, Splistlen;
static int Scanning;

/* external references defined in upstream binary source */
extern int Ininit;
extern int Opt_quorum;
extern word16 Opt_dstport;

int mcmd__request
   (SERVER *sp, word32 ip, word16 opreq, word8 bnum[8], int syncwait)
{
   struct sockaddr_in addr;
   NODE *np;
   int result;
   char ipstr[16];
   char bnumstr[17];

   /* log request */
   ntoa(&ip, ipstr);
   bnum2hex(bnum, bnumstr);
   pdebug("request %s(0x%s) from %s...", op2str(opreq), bnumstr, ipstr);

   /* create NODE for request */
   addr = ipv4_addr(ip, Opt_dstport);
   np = node_create((struct sockaddr *) &addr, (socklen_t) sizeof(addr));
   if (np == NULL) {
      perrno("... %s node_create() FAILURE", ipstr);
      return VERROR;
   }

   /* initialize NODE request and send to Server handler */
   np->opreq = opreq;
   put64(np->bnum, bnum);
   result = server_queue(sp, np, (struct sockaddr *) &addr);
   if (result != VEOK) {
      perrno("... %s server_queue() FAILURE", ipstr);
      free(np);
      return VERROR;
   }
   /* set Syncwait if requested */
   if (syncwait) {
      /* Syncwait is set to the pointer of a NODE that we need to "wait"
       * for before performing other synchronous operations
       */
      if (Syncwait) perr("Syncwait was reset!");
      Syncwait = np;
   }

   return VEOK;
}  /* end mcmd__request() */

int mcmd_scan_request(SERVER *sp, word32 *plist, word32 plistlen)
{
   void *ptr;
   word32 i;

   if (plist == NULL) return VERROR;

   /* request OP_GET_IPL on peerlist members */
   for (i = 0; i < plistlen; i++) {
      /* ignore zero and pinklisted peers */
      if (plist[i] == 0 || pinklisted(plist[i])) continue;
      /* ensure available space in Splist */
      if (Splistidx >= Splistlen) {
         ptr = realloc(Splist, sizeof(word32) * (Splistlen + 32));
         if (ptr == NULL) {
            perrno("Scanned peer list FAILURE");
            return VERROR;
         }
         /* update Scanned peer list */
         Splist = ptr;
         memset(&Splist[Splistlen], 0, sizeof(word32) * 32);
         Splistlen = Splistlen + 32;
      }
      /* add peer to scan list if not already */
      if (search32(plist[i], Splist, Splistlen)) continue;
      Splist[Splistidx++] = plist[i];
      /* request peerlist for network scan and peer compatibility */
      if (mcmd_request(sp, plist[i], OP_GET_IPL, NULL, 0) != VEOK) continue;
      /* increment the number of peers being queried */
      Scanning++;
   }  /* end for () */

   return VEOK;
}  /* end mcmd_scan_request() */

int mcmd_scan_response(SERVER *sp, NODE *np)
{
   void *ptr;
   word32 *plist, plistlen, i;
   int result;
   char hexstr[65];
   char bnumstr[17];
   char ipstr[16];

   /* log resulting node data */
   bnum2hex(np->pdu.cblock, bnumstr);
   weight2hex(np->pdu.weight, hexstr);
   ntoa(np->addr.sin_addr.s_addr, ipstr);
   pdebug("%s returned %d 0x%s 0x%s", ipstr, np->status, bnumstr, hexstr);

   /* decrement scanning */
   Scanning--;
   /* check success of request */
   if (np->status == VEOK) {
      /* compare NODE weight against Quorum */
      result = cmp256(Qweight, np->pdu.weight);
      if (result < 0) {
         /* set quorum to higher advertised chain */
         if (Qplist) memset(Qplist, 0, sizeof(word32) * Qplistlen);
         memcpy(Qbhash, np->pdu.cblockhash, 32);
         memcpy(Qweight, np->pdu.weight, 32);
         put64(Qbnum, np->pdu.cblock);
         Qplistidx = 0;
      }
      /* compare block hash on same or higher advertised chain */
      if (result <= 0 && memcmp(Qbhash, np->pdu.cblockhash, HASHLEN) == 0) {
         /* ensure available space in Qplist */
         if (Qplistidx >= Qplistlen) {
            ptr = realloc(Qplist, sizeof(word32) * (Qplistlen + 32));
            if (ptr == NULL) {
               perrno("quorum list increase FAILURE");
               return VERROR;
            }
            /* update Quorum peer list */
            Qplist = ptr;
            memset(&Qplist[Qplistlen], 0, sizeof(word32) * 32);
            Qplistlen = Qplistlen + 32;
         }
         /* add peer to Quorum list if not already */
         Qplist[Qplistidx++] = np->addr.sin_addr.s_addr;
      }  /* end for () */
      /* process provided peer list */
      plist = (word32 *) np->pdu.buffer;
      plistlen = (word32) get16(np->pdu.len) / sizeof(word32);
      /* perform scan on additional peers */
      mcmd_scan_request(sp, plist, plistlen);
   }  /* end if (np->status == VEOK) */

   /* wait for all "scanning" requests */
   pdebug("waiting for %d requests...", Scanning);
   if (Scanning > 0) return VEOK;

   /* check quorum requirements */
   if (Qplistidx == 0) {
      plog("No higher chain available");
      return VEOK;
   } else if (Qplistidx < (word32) Opt_quorum) {
      bnum2hex(Qbnum, bnumstr);
      weight2hex(Qweight, hexstr);
      pdebug("Quorum chain data 0x%s / 0x%s", bnumstr, hexstr);
      plog("Insufficient Quorum: %u / %d", Qplistidx, Opt_quorum);
      /* enable (soft block) pinklist on insufficient quorum peers */
      for (i = 0; i < Qplistidx; i++) {
         remove32(Qplist[i], Splist, Splistlen, &Splistidx);
         epinklist(Qplist[i]);
      }
      /* clear quorum peers list */
      memset(Qplist, 0, sizeof(word32) * Qplistlen);
      Qplistidx = 0;
      /* perform rescan on remaining peers */
      plog("(re)Scanning network...");
      return mcmd_scan_request(sp, Splist, Splistlen);
   }

   /* done, begin sync with quorum */
   return mcmd_syncup_request(NULL);
} /* end mcmd_scan_response() */
