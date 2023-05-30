
/* internal support */
#include "block.h"
#include "chain.h"
#include "error.h"
#include "ledger.h"
#include "peer.h"
#include "protocol.h"
#include "server.h"
#include "transaction.h"
#include "trigg.h"

/* external support */
#include "extio.h"
#include "extlib.h"
#include "extmath.h"
#include "exttime.h"
#include <signal.h>
#include <string.h>

/* define EXEC_NAME and GIT_VERSION (if not defined) */

#ifndef EXEC_NAME
   #define EXEC_NAME "Mochimo Server Daemon"

#endif

#ifndef GIT_VERSION
   #define GIT_VERSION "v-unknown"

#endif

#define perrno_exit(...)   { perrno(__VA_ARGS__); exit(0); }
#define perr_exit(...)     { perr(__VA_ARGS__); exit(0); }

#define perrno_fatal(...)  { fatal(); perrno(__VA_ARGS__); }
#define perr_fatal(...)    { fatal(); perr(__VA_ARGS__); }

#define NODE_IS_WALLET(np)   ( \
   ((np)->pkt.version[0] == 4 && get16((np)->pkt.len) == 1) || \
   ((np)->pkt.version[0] == 5 && ((np)->pkt.version[1] & C_WALLET)) )

#define WORKLIST_INITIALIZER \
   { .lock = MUTEX_INITIALIZER, .alarm = CONDITION_INITIALIZER, { 0 } }

typedef struct {
   Mutex lock;
   Condition alarm;
   LinkedList list;
} WorkList;

/* work processing list (incl. lock and alarm) */
Server MCMDServer, *MCMDIO = &MCMDServer;
WorkList CompleteIO = WORKLIST_INITIALIZER;

static RWLock UpdateLock = RWLOCK_INITIALIZER;
/* NOTE: Update lock states...
 * - read locked (shared) during tx validation
 * - write locked (exclusive) during block update */

/** Recent transactions list (for checking duplicates) */
/* static RWLock RecentTxsLock = RWLOCK_INITIALIZER; */
/* static HashSet RecentTxs; */

/* Mochimo globals */
static time_t Ptime;
static NODE *Syncwait;     /* Block sync until this NODE is returned */
static int Ininit;
static int Running;
static int Quorum_opt;
static int Nopush_opt;
static word64 Trustblock_opt;
static word32 Mfee[2];
static word32 Myfee[2];
static word16 Dstport_opt;

/* cli options for peerlist filenames */
static char *Epinkip_opt = "epink.lst";
static char *Localip_opt = "local.lst";
static char *Recentip_opt = "recent.lst";

/* Scanned peer list parameters -- for quorum scan */
static int Scanning;
static word32 *Splist, Splistidx, Splistlen;
/* Quorum peer list parameters -- + bnum, bhash, weight */
static word32 *Qplist, Qplistidx, Qplistlen;
static word8 Qbnum[8], Qbhash[HASHLEN], Qweight[32];

static void fatal(void) {
   palert("CRITICAL RUNTIME ERROR!");
   /* elevate trace details */
   ptrace_functions(1);
   /* flag shutdown */
   Running = 0;
}

static void mcmd__cleanup(LinkedNode *lnp)
{
   NODE *np;

   /* check valid NODE data */
   np = (NODE *) lnp->data;
   if (np != NULL) {
      /* acquire (exclusive) recent peers list lock */
      if (rwlock_wrlock(&Rplistlock) != 0) {
         perrno_fatal("Rplistlock LOCK FAILURE");
         return;
      }
      /* update peerlists under certain conditions */
      switch (np->status) {
         case VEBAD2: {
            /* naughty peers go to (epoch) pinklist */
            epinklist(np->ip);
            /* remove peer from recent peers list */
            remove32(np->ip, Rplist, RPLISTLEN, &Rplistidx);
         }  /* fallthrough */
         /* add peer to (standard) pinklist if BAD */
         case VEBAD: pinklist(np->ip); break;
         case VEOK: {
            /* add recent peer under verified network activity */
            if (np->opcode == OP_TX || np->opcode == OP_FOUND) {
               addpeer(np->ip, Rplist, RPLISTLEN, &Rplistidx);
            }
            /* clear (soft) pinklist on block update, outside init */
            if (!Ininit && np->opreq == OP_GET_BLOCK) {
               purge_pinklist();
            }
         }
      }  /* end switch (np->status) */
      /* release (exclusive) recent peers list lock */
      if (rwlock_wrunlock(&Rplistlock) != 0) {
         perrno_fatal("Rplistlock UNLOCK FAILURE");
         return;
      }
      /* cleanup/deallocate resources */
      node_cleanup(np);
      free(np);
   }  /* end if (np != NULL) */
   free(lnp);
}  /* end mcmd__cleanup() */

static int mcmd__mirrortx(NODE *mp)
{
   NODE *np;
   word32 plist[RPLISTLEN];
   int result, i;

   /* acquire (exclusive) recent peers list lock */
   if (rwlock_rdlock(&Rplistlock) != 0) {
      perrno_fatal("Rplistlock LOCK FAILURE");
      return VERROR;
   }
   /* copy recent peers in it's current state */
   memcpy(plist, Rplist, sizeof(Rplist));
   /* release (exclusive) recent peers list lock */
   if (rwlock_rdunlock(&Rplistlock) != 0) {
      perrno_fatal("Rplistlock UNLOCK FAILURE");
      return VERROR;
   }

   /* mirror transaction to recent peers */
   for (i = 0; i < RPLISTLEN && plist[i]; i++) {
      /* malloc space for a new node */
      np = malloc(sizeof(*np));
      if (np == NULL) {
         perrno("malloc FAILURE");
         return VERROR;
      }
      /* initialize node request and load transaction data */
      node_init(np, INVALID_SOCKET, plist[i], Dstport_opt, OP_TX, NULL);
      np->fp = tmpfile();
      if (np->fp == NULL) {
         perrno("tmpfile() FAILURE");
         node_cleanup(np);
         return VERROR;
      }
      result = fwrite(mp->pkt.buffer, sizeof(TXW) - HASHLEN, 1, np->fp);
      if (result != 1) {
         perrno("fwrite() FAILURE");
         node_cleanup(np);
         return VERROR;
      }
      rewind(np->fp);
      /* send NODE to server handler */
      result = server_work_create(MCMDIO, np);
      if (result != VEOK) {
         perrno("server work FAILURE");
         node_cleanup(np);
         return VERROR;
      }
   }

   /* done */
   return VEOK;
}  /* end mcmd__mirrortx() */

/**
 * Proof of Work validation. Typically Multi-Threaded.
 * @param arg Pointer to NODE
 * @private for internal use only
 */
static ThreadProc mcmd__pow_val(void *arg)
{
   NODE *np = arg;

   Mutex lock = MUTEX_INITIALIZER;
   BTRAILER bt;
   int result;
   char bnumstr[17];

   /* set name of thread - visible in htop */
   thread_setname(thread_self(), "PoW-validation");

   /* acquire (exclusive) validation lock */
   result = mutex_lock(&lock);
   if (result != 0) {
      perrno_fatal("PoW-validation LOCK FAILURE");
      Unthread;
   }
   /* read and validate all available data */
   while (Running && np->status == VEOK) {
      /* read-in data and check for EOF */
      result = fread(&bt, sizeof(bt), 1, np->fp);
      if (result != 1) {
         if (feof(np->fp)) break;
         /* file error ocurred... */
         np->status = VERROR;
         break;
      }
      /* skip to trusted block numbers */
      if (cmp64(bt.bnum, &Trustblock_opt) <= 0) continue;
      /* (re)release (exclusive) validation lock */
      result = mutex_unlock(&lock);
      if (result != 0) {
         perrno_fatal("PoW-validation (re)UNLOCK FAILURE");
         Unthread;
      }
      /* validate PoW */
      result = validate_pow(&bt);
      if (result != VEOK) {
         np->status = result;
         bnum2hex(bt.bnum, bnumstr);
         perrno("PoW-validation 0x%s FAILURE", bnumstr);
         Unthread;
      }
      /* (re)acquire (exclusive) validation lock */
      result = mutex_unlock(&lock);
      if (result != 0) {
         perrno_fatal("PoW-validation (re)UNLOCK FAILURE");
         Unthread;
      }
   }  /* end while (Running && np->status == VEOK) */
   /* release (exclusive) validation lock */
   result = mutex_unlock(&lock);
   if (result != 0) {
      perrno_fatal("PoW-validation UNLOCK FAILURE");
   }

   /* done */
   Unthread;
}  /* end mcmd__pow_val() */

int mcmd__request(word32 ip, word16 opreq, void *bnum, int syncwait)
{
   NODE *np;
   int result;
   char ipstr[16];
   char bnumstr[17];

   /* log request */
   ntoa(&ip, ipstr);
   bnum2hex(bnum, bnumstr);
   pdebug("request %s(%s) from %s...", op2str(opreq), bnumstr, ipstr);

   /* malloc space for a new node */
   np = malloc(sizeof(*np));
   if (np == NULL) {
      perrno("%s malloc FAILURE", ipstr);
      return VERROR;
   }
   /* initialize node request and send to Server handler */
   node_init(np, INVALID_SOCKET, ip, Dstport_opt, opreq, bnum);
   result = server_work_create(MCMDIO, np);
   if (result != VEOK) {
      perrno("%s server work FAILURE", ipstr);
      free(np);
      return VERROR;
   }
   /* set Syncwait if requested */
   if (syncwait) {
      if (Syncwait) perr("Syncwait was reset!");
      Syncwait = np;
   }

   return VEOK;
}  /* end mcmd__request() */

int mcmd__scan(word32 *plist, word32 plistlen)
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
            perrno_fatal("scan list increase FAILURE");
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
      if (mcmd__request(plist[i], OP_GET_IPL, NULL, 0) != VEOK) continue;
      /* increment the number of peers being queried */
      Scanning++;
   }  /* end for () */

   return VEOK;
}  /* end mcmd__scan() */

int mcmd__scaninit(void)
{
   word32 plist[RPLISTLEN];
   word32 plistidx;
   int result;

   /* init */
   Ininit = 1;
   plistidx = 0;
   memset(plist, 0, sizeof(plist));
   /* read recent peers from disk into "start" peers list */
   result = read_ipl(Recentip_opt, plist, RPLISTLEN, &plistidx);
   if (result == 0) pwarn("No scan peers. Network may not be mapped.");
   if (result < 0) {
      perrno("ip list read FAILURE");
      return VERROR;
   }
   /* trigger initial network scan on peer list */
   return mcmd__scan(plist, RPLISTLEN);
}  /* end mcmd__scaninit() */

/**
 * @brief 
 * @param np 
 * @return 
 */
int mcmd__sendfound(NODE *np)
{
   static word32 plist[RPLISTLEN];

   BTRAILER bt[54];
   word32 bnum[2];
   int inprogress;
   int result, count;
   char ipstr[16];

   /* init */
   inprogress = *plist ? 1 : 0;

   /* refresh plist on new send_found trigger, else remove NODE ip */
   if (np == NULL) {
      /* acquire (exclusive) recent peers list lock */
      if (rwlock_rdlock(&Rplistlock) != 0) {
         perrno_fatal("Rplistlock LOCK FAILURE");
         return VERROR;
      }
      /* copy recent peers in it's current state */
      memcpy(plist, Rplist, sizeof(Rplist));
      /* release (exclusive) recent peers list lock */
      if (rwlock_rdunlock(&Rplistlock) != 0) {
         perrno_fatal("Rplistlock UNLOCK FAILURE");
         return VERROR;
      }
      pdebug("OP_FOUND broadcast started...");
      /* bail if already in progress */
      if (inprogress) return VEOK;
   } else remove32(np->ip, plist, RPLISTLEN, NULL);

   /* broadcast to next peer (if any) */
   if (*plist) {
      ntoa(plist, ipstr);
      pdebug("broadcasting OP_FOUND to %s...", ipstr);
      /* malloc space for a new node */
      np = malloc(sizeof(*np));
      if (np == NULL) {
         perrno("malloc FAILURE");
         return VERROR;
      }
      /* initialize node request and load tfile data */
      node_init(np, INVALID_SOCKET, *plist, Dstport_opt, OP_FOUND, NULL);
      if (sub64(Cblocknum, (word32[2]) { (54 - 1), 0 }, bnum)) {
         memset(bnum, 0, 8);
         count = (int) get32(Cblocknum) + 1;
      } else count = 54;
      result = read_tfile(bt, bnum, count, "tfile.dat");
      if (result != count) {
         perrno("read tfile FAILURE");
         node_cleanup(np);
         return VERROR;
      }
      np->fp = tmpfile();
      if (np->fp == NULL) {
         perrno("tmpfile() FAILURE");
         node_cleanup(np);
         return VERROR;
      }
      result = fwrite(bt, sizeof(*bt), 54, np->fp);
      if (result != 54) {
         perrno("fwrite() FAILURE");
         node_cleanup(np);
         return VERROR;
      }
      rewind(np->fp);
      /* send NODE to server handler */
      result = server_work_create(MCMDIO, np);
      if (result != VEOK) {
         perrno("server work FAILURE");
         node_cleanup(np);
         return VERROR;
      }
      return VEOK;
   }

   pdebug("OP_FOUND broadcast finished");
   return VEOK;
}  /* end mcmd__sendfound() */

int mcmd__syncnext(NODE *np)
{
   static word32 one[2] = { 1, 0 };

   word32 bnum[2];
   char ipstr[16];

   /* check provided NODE pointer */
   if (np) {
      /* check NODE result */
      if (np->status == VEOK) {
         /* check sync for more chain */
         if (cmp64(Cblocknum, np->pkt.cblock) < 0) {
            /* request next block */
            add64(Cblocknum, one, bnum);
            return mcmd__request(np->ip, OP_GET_BLOCK, bnum, 1);
         }
         /* synchronized, broadcast and purge outside of init */
         pdebug("synchronized with %s", np->id);
         if (!Ininit) {
            mcmd__sendfound(NULL);
            purge_pinklist();
         }
      }  /* end if (np->status == VEOK) */
      if (!Ininit) return np->status;
      /* check sync against quorum during init */
      if (cmp64(Cblocknum, Qbnum) >= 0) {
         /* log initial sync completion -- clear Ininit */
         plog("Veronica says, \"You're done!\"");
         /* trigger block found broadcast */
         mcmd__sendfound(NULL);
         /* clear Quorum and Scanned lists */
         if (Splist) {
            Splistidx = Splistlen = 0;
            free(Splist);
         }
         if (Qplist) {
            Qplistidx = Qplistlen = 0;
            free(Qplist);
         }
         /* clear (soft) pinklist on syncup */
         purge_pinklist();
         /* done */
         Ininit = 0;
         return VEOK;
      }
      pdebug("quorum sync incomplete, drop %s...", np->id);
      /* drop node ip from Scanned and Quorum lists */
      if (Splist) remove32(np->ip, Splist, Splistlen, &Splistidx);
      if (Qplist) remove32(np->ip, Qplist, Qplistlen, &Qplistidx);
      /* (soft) pinklist during init */
      pinklist(np->ip);
   }  /* end if (np) */

   /* pull next node from quorum */
   if (Qplistidx) {
      /* shuffle Quorum peer list */
      /* shuffle32(Qplist, Qplistlen); */
      /* build OP_TF bnum parameter for synchronization */
      bnum[0] = get32(Qbnum);
      bnum[1] = (bnum[0] < 1000) ? bnum[0] + 1 : 1000;
      bnum[0] -= (bnum[0] < 1000) ? bnum[0] : (1000 - 1);
      /* request OP_TF from next quorum member (consumed) */
      plog("Synchronizing with %s...", ntoa(Qplist, ipstr));
      return mcmd__request(Qplist[0], OP_TF, bnum, 1);
   }
   /* if no node in quorum, trigger network (re)scan */
   if (Splistidx) return mcmd__scan(Splist, Splistlen);

   /* restart network scan initialization */
   return mcmd__scaninit();
}  /* end mcmd__syncnext() */

int mcmd_check_block(NODE *np)
{
   BTRAILER bt;
   int result;
   const char *filename;

   /* check NODE parmater... */
   if (np) {
      /* ensure NODE operation was successful */
      if (np->status != VEOK) {
         pdebug("block download FAILURE");
         return mcmd__syncnext(np);
      }
      /* ensure received block number is as requested */
      result = read_bnum_fp(bt.bnum, np->fp);
      if (result != 0) {
         perrno_fatal("read block number FAILURE");
         return result;
      }
      if (cmp64(np->io, bt.bnum)) {
         np->status = VEBAD;
         pdebug("block number mismatch");
         return mcmd__syncnext(np);
      }
      /* validate block file data before saving */
      np->status = validate_block_fp(np->fp, "tfile.dat");
      if (np->status != VEOK) {
         perrno("block validation FAILURE");
         return mcmd__syncnext(np);
      }
      /* save block data to file */
      filename = "block.dat";
      result = fsave(np->fp, filename);
      if (result != 0) {
         perrno_fatal("file save FAILURE");
         return result;
      }
      /* close temporary file */
      fclose(np->fp);
      np->fp = NULL;
   } else {
      /* generate and validate a pseudoblock for update */
      filename = "pseudo.dat";
      result = generate_pseudo(filename, "tfile.dat");
      if (result != VEOK) {
         perrno_fatal("pseudo generation FAILURE");
         return result;
      }
      result = validate_pseudo(filename, "tfile.dat");
      if (result != VEOK) {
         perrno_fatal("pseudo validation FAILURE");
         return result;
      }
   }  /* end if (np)... else... */

   /* acquire (exclusive) update lock for update operation */
   if (rwlock_wrlock(&UpdateLock) != 0) {
      perrno_fatal("UpdateLock LOCK FAILURE");
      return VERROR;
   }
   /* perform the block update */
   result = block_update(filename);
   if (result != VEOK) {
      perrno_fatal("blockchain update FAILURE");
      if (rwlock_wrunlock(&UpdateLock) != 0) {
         perrno_fatal("inner UpdateLock UNLOCK FAILURE");
      }
      return result;
   }
   /* release (exclusive) update lock */
   if (rwlock_wrunlock(&UpdateLock) != 0) {
      perrno_fatal("UpdateLock UNLOCK FAILURE");
      return VERROR;
   }

   /* print block trailer information */
   if (read_trailer(&bt, filename) == VEOK) {
      ptrailer(&bt);
   }

   /* archive blockchain file and return final result */
   result = archive_block(filename, Bcdir_opt);
   if (result != VEOK) {
      perrno_fatal("blockchain archive FAILURE");
      return result;
   }

   /* VEOK, sync next */
   return mcmd__syncnext(np);
}  /* end mcmd_check_block() */

/* simple proxy function for OP_FOUND broadcast checking */
int mcmd_check_broadcast(NODE *np)
{
   return mcmd__sendfound(np);
}

int mcmd_check_peers(NODE *np)
{
   void *ptr;
   word32 *plist, plistlen, i;
   int result;
   char hexstr[65];
   char bnumstr[17];

   /* log resulting node data */
   pdebug("%s returned %s 0x%s 0x%s", np->id, ve2str(np->status),
      bnum2hex(np->pkt.cblock, bnumstr), weight2hex(np->pkt.weight, hexstr));

   /* decrement scanning */
   Scanning--;
   /* check success of request */
   if (np->status == VEOK) {
      /* compare NODE weight against Quorum */
      result = cmp256(Qweight, np->pkt.weight);
      if (result < 0) {
         /* set quorum to higher advertised chain */
         if (Qplist) memset(Qplist, 0, sizeof(word32) * Qplistlen);
         memcpy(Qbhash, np->pkt.cblockhash, 32);
         memcpy(Qweight, np->pkt.weight, 32);
         put64(Qbnum, np->pkt.cblock);
         Qplistidx = 0;
      }
      /* compare block hash on same or higher advertised chain */
      if (result <= 0 && memcmp(Qbhash, np->pkt.cblockhash, HASHLEN) == 0) {
         /* ensure available space in Qplist */
         if (Qplistidx >= Qplistlen) {
            ptr = realloc(Qplist, sizeof(word32) * (Qplistlen + 32));
            if (ptr == NULL) {
               perrno_fatal("quorum list increase FAILURE");
               return VERROR;
            }
            /* update Quorum peer list */
            Qplist = ptr;
            memset(&Qplist[Qplistlen], 0, sizeof(word32) * 32);
            Qplistlen = Qplistlen + 32;
         }
         /* add peer to Quorum list if not already */
         Qplist[Qplistidx++] = np->ip;
      }  /* end for () */
      /* process provided peer list */
      plist = (word32 *) PKTBUFF(&np->pkt);
      plistlen = (word32) get16(np->pkt.len) / sizeof(word32);
      /* limit list length to size of packet (just in case) */
      if (plistlen > PKTBUFFLEN) {
         plistlen = (word32) PKTBUFFLEN / sizeof(word32);
      }
      /* perform scan on additional peers */
      mcmd__scan(plist, plistlen);
   }  /* end if (np->status == VEOK) */

   /* wait for all "scanning" requests */
   pdebug("waiting for %d requests...", Scanning);
   if (Scanning) return VEOK;

   /* check quorum requirements */
   if (Qplistidx == 0) {
      plog("No higher chain available");
      return VEOK;
   } else if (Qplistidx < (word32) Quorum_opt) {
      bnum2hex(Qbnum, bnumstr);
      weight2hex(Qweight, hexstr);
      pdebug("Quorum chain data 0x%s / 0x%s", bnumstr, hexstr);
      plog("Insufficient Quorum: %u / %d", Qplistidx, Quorum_opt);
      /* remove quorum peers from scanned peers list */
      for (i = 0; i < Qplistidx; i++) {
         remove32(Qplist[i], Splist, Splistlen, &Splistidx);
      }
      /* clear quorum peers list */
      memset(Qplist, 0, sizeof(word32) * Qplistlen);
      Qplistidx = 0;
      /* perform rescan on scan list peers */
      plog("(re)Scanning network...");
      return mcmd__scan(Splist, Splistlen);
   }

   /* done, begin sync with quorum */
   return mcmd__syncnext(NULL);
}  /* end mcmd_check_peers() */

int mcmd_check_proof(NODE *np)
{
   Thread *thrdp;
   BTRAILER bt, cmpbt;
   word8 weight[32], cmpweight[32];
   word8 bnum[8], cmpbnum[8];
   int partial, core_count, i;
   int result, code;
   char hexstr[65], bnumstr[17];

   /* init */
   partial = 0;
   memset(weight, 0, 32);
   if (np->opcode == OP_SEND_FILE) {
      code = np->opreq;
   } else code = np->opcode;

   /* determine appropriate action... */
   switch (code) {
      case OP_FOUND: {
         pdebug("%s prepare proof...", op2str(code));
         /* perform prepare proof...
          * - is advertised weight sufficient?
          * - prepare proof in temporary file pointer */

         /* check advertised weight */
         if (cmp256(np->pkt.weight, Weight) <= 0) {
            np->status = VERROR;
            weight2hex(np->pkt.weight, hexstr);
            pdebug("insufficient weight, 0x%s", hexstr);
            return mcmd__syncnext(np);
         }
         /* move tfile data to tmpfile() */
         np->fp = tmpfile();
         if (np->fp == NULL) {
            np->status = VERROR;
            perrno("temporary file creation FAILURE");
            return mcmd__syncnext(np);
         }
         result = fwrite(np->pkt.buffer, sizeof(BTRAILER), 54, np->fp);
         if (result != 54) {
            np->status = VERROR;
            perrno_fatal("temporary file write FAILURE");
            return result;
         }
      }  /* fallthrough -- end case OP_FOUND */
      case OP_TF: {
         pdebug("%s proof analysis...", op2str(code));
         /* perform proof analysis...
          * - is there a gap between our chain and the proof?
          * - is there a chain split preceeding the proof?
          * - is resync required to synchronize to the proof? */

         /* read initial trailer in proof */
         rewind(np->fp);
         result = fread(&bt, sizeof(bt), 1, np->fp);
         if (result != 1) {
            if (feof(np->fp)) {
               set_errno(EMCM_EOF);
            }  /* file error */
            np->status = VERROR;
            perrno("read trailer proof FAILURE");
            return mcmd__syncnext(np);
         }
         /* read our trailer file at the same block number */
         result = read_tfile(&cmpbt, bt.bnum, 1, "tfile.dat");
         if (result != 1) {
            np->status = VERROR;
            pdebug("chain gap preceeding proof...");
            /* fail on out-of-range "proof" during init */
            if (!Ininit) {
               pdebug("cannot accept chain gap out-of-range");
               return mcmd__syncnext(np);
            }
            /* request a bigger picture (OP_GET_TFILE)... */
            return mcmd__request(np->ip, OP_GET_TFILE, NULL, 1);
         }
         /* compare our trailer file with initial proof trailer */
         if (memcmp(&cmpbt, &bt, sizeof(bt)) != 0) {
            np->status = VERROR;
            pdebug("chain split preceeding proof...");
            /* fail on out-of-range "proof split" during init */
            if (!Ininit) {
               pdebug("cannot accept chain split out-of-range");
               return mcmd__syncnext(np);
            }
            /* request the bigger picture (OP_GET_TFILE)... */
            return mcmd__request(np->ip, OP_GET_TFILE, NULL, 1);
         }
         /* pre-calculate weight up-to proof */
         result = weigh_tfile("tfile.dat", bt.bnum, weight);
         if (result != VEOK) {
            perrno_fatal("proof weight calculation FAILURE");
            return result;
         }
         /* flag tfile proof type as "partial" */
         partial = 1;
      }  /* fallthrough -- end case OP_TF */
      case OP_GET_TFILE: {
         pdebug("%s proof validation...", op2str(code));
         /* perform proof validation...
          * - validate proof as a tfile (partial or otherwise)
          * - is there a gap between our chain and the proof?
          * - is there a chain split preceeding the proof?
          * - is resync required to synchronize to the proof? */

         np->status = validate_tfile_fp(np->fp, bnum, weight, partial);
         if (np->status != VEOK) {
            perrno("proof validation FAILURE");
            return mcmd__syncnext(np);
         }
         /* check proof bnum against Quorum (or self) */
         put64(cmpbnum, Qplistidx ? Qbnum : Cblocknum);
         if (cmp64(bnum, cmpbnum) < 0) {
            np->status = VEBAD;
            pdebug("proof bnum less than advertised");
            pdebug("- proof 0x%s", bnum2hex(bnum, bnumstr));
            pdebug("- quorum 0x%s", bnum2hex(cmpbnum, bnumstr));
            return mcmd__syncnext(np);
         }
         /* check proof weight against Quorum (or self) */
         memcpy(cmpweight, Qplistidx ? Qweight : Weight, 32);
         if (cmp256(weight, cmpweight) < 0) {
            np->status = VEBAD;
            pdebug("proof weight less than advertised");
            pdebug("- proof 0x%s", weight2hex(weight, hexstr));
            pdebug("- quorum 0x%s", weight2hex(cmpweight, hexstr));
            return mcmd__syncnext(np);
         }
         /* estimate time of validation for OP_GET_TFILE */
         core_count = cpu_cores();
         if (code == OP_GET_TFILE) {
            long long eta = 0;
            put64(&eta, bnum);
            eta = ((eta >> 8) / core_count) / 60;
            if (eta <= 1) plog("Validating PoW (may take a minute)...");
            else plog("Validating PoW (may take %lld minutes)...", eta);
         }
         /* rewind file pointer and create threads for PoW validation */
         thrdp = malloc(core_count * sizeof(*thrdp));
         if (thrdp == NULL) perrno("malloc(threads) FAILURE");
         for (rewind(np->fp), i = 0; i < core_count; i++) {
            if (thread_create(&thrdp[i], mcmd__pow_val, np) != 0) {
               perrno("thread_create() FAILURE");
               pwarn("PoW validation may be slower than usual");
               core_count = i;
               break;
            }
         }
         /* if no threads were created, provide fallback method */
         if (core_count == 0) {
            pwarn("using (SLOW) fallback PoW validation");
            mcmd__pow_val(np);
         }
         /* wait for threads to finish */
         for (i = 0; i < core_count; i++) {
            if (thread_join(thrdp[i]) != 0) {
               perrno("thread_join(%d) FAILURE", i);
            }
         }
         /* cleanup thread malloc */
         if (thrdp) free(thrdp);
         if (!Running) return VERROR;
         /* check result of PoW validation */
         if (np->status != VEOK) {
            perr("PoW validation FAILURE");
            return mcmd__syncnext(np);
         }

         /* PROOF IS VALID */

         /* ensure chain is synchronized to NODE's chain data */
         pdebug("synchronizing available blockchain...");
         np->status = block_syncup_fp(np->fp, bnum);
         if (np->status != VEOK) {
            perrno("blockchain syncup FAILURE");
            return mcmd__syncnext(np);
         }

         /* request next block (download) */
         return mcmd__request(np->ip, OP_GET_BLOCK, bnum, 1);
      }  /* end case OP_GET_TFILE */
   }  /* end switch (code) */

   /* unknown operation */
   np->status = VERROR;
   return VERROR;
}  /* end mcmd_check_proof() */

/**
 * @brief Validate and process incoming transactions
 * @details Reference Chart: https://app.code2flow.com/HaB1Ut
 * @param np Pointer to NODE containing incoming transaction
 * @return (int) value representing the operation result
 */
int mcmd_check_tx(NODE *np)
{
   static word32 one[2] = { 1 };

   TXW txw;
   word32 bnum[2];

   /* check status of receive */
   if (np->status == VEOK) {
      /* generate transaction ID *//*
      sha256(np->pkt.buffer, sizeof(txw) - HASHLEN, txw.tx_id);*/
      /* check/add recent transactions by ID *//*
      if (hashset_contains(&RecentTxs, &txw.tx_id)) {
         return (np->status = VERROR);
      } else if (hashset_add(&RecentTxs, &(txw.tx_id), HASHLEN) != 0) {
         perrno("hashset_add(recent) FAILURE");
         return (np->status = VERROR);
      }*/
      /* copy transaction to local buffer */
      memcpy(&txw, np->pkt.buffer, sizeof(txw) - HASHLEN);
      /* validate transaction */
      np->status = txw_val(&txw, Myfee);
      if (np->status == VEOK) {
         pdebug("transaction is valid, store...");
         /* mirror transaction to network */
         return mcmd__mirrortx(np);
      } else if (np->status == VERROR) {
         pdebug("transaction validation error, check chain...");
         /* transaction is invalid, discard unless behind one block */
         if (cmp64(np->pkt.cblock, Cblocknum) > 0) {
            /* hold transaction for update */
            pdebug("chain is behind, request proof...");
            /* build OP_TF bnum parameter (immitate OP_FOUND) */
            add64(Cblocknum, one, bnum);
            bnum[1] = (bnum[0] < 54) ? bnum[0] + 1 : 54;
            bnum[0] -= (bnum[0] < 54) ? bnum[0] : (54 - 1);
            /* request OP_TF (immitate OP_FOUND) */
            return mcmd__request(np->ip, OP_TF, bnum, 0);
         } else pdebug("no chain updates, discard...");
      } else perrno("transaction is BAD, pinklist...");
   }  /* end if (np->status == VEOK) */

   return np->status;
}  /* end mcmd_check_tx() */

int mcmd_worker_sync(LinkedNode *lnp)
{
   static Mutex syncLock = MUTEX_INITIALIZER;
   static Mutex syncIOLock = MUTEX_INITIALIZER;
   static LinkedList syncIO = { 0 };

   LinkedNode *next_lnp;
   NODE *np;
   time_t now;
   int result;

   /* acquire (exclusive) syncIO lock */
   result = mutex_lock(&syncIOLock);
   if (result != 0) {
      perrno_fatal("syncIO LOCK FAILURE");
      return VERROR;
   }

   /* append any work to syncIO */
   if (lnp && lnp->data) {
      result = link_node_append(lnp, &syncIO);
      if (result != 0) {
         perrno_fatal("syncIO append FAILURE");
         if (mutex_unlock(&syncIOLock) != 0) {
            perrno_fatal("inner syncIO UNLOCK FAILURE");
         }
         return VERROR;
      }
   }

   /* TRY acquire (exclusive) sync lock */
   result = mutex_trylock(&syncLock);
   if (result != 0) {
      if (errno != EBUSY) {
         perrno_fatal("sync TRYLOCK FAILURE");
      }
      /* release (exclusive) syncIO lock */
      if (mutex_unlock(&syncIOLock) != 0) {
         perrno_fatal("syncIO UNLOCK FAILURE");
      }
      return VERROR;
   }

   /* THREAD IS NOW PROCESSING SYNCHRONOUS TASKS */
   thread_setname(thread_self(), "mcmd-sync");

   /* walk syncIO list for processing */
   for (lnp = syncIO.next; lnp && Running; lnp = next_lnp) {
      /* store next node in list and check Syncwait */
      next_lnp = lnp->next;
      if (Syncwait && Syncwait != lnp->data) continue;
      /* remove syncIO node from list */
      result = link_node_remove(lnp, &syncIO);
      if (result != 0) {
         perrno_fatal("syncIO LIST FAILURE");
         break;
      }
      /* (re)release (exclusive) syncIO lock */
      result = mutex_unlock(&syncIOLock);
      if (result != 0) {
         perrno_fatal("syncIO (re)UNLOCK FAILURE");
         break;
      }
      /* nullify Syncwait and process NODE */
      Syncwait = NULL;
      np = (NODE *) lnp->data;
      /* determine method of synchronous processing */
      switch (np->opreq) {
         case OP_GET_BLOCK: mcmd_check_block(np); break;
         case OP_GET_IPL: mcmd_check_peers(np); break;
         case OP_GET_TFILE: mcmd_check_proof(np); break;
         case OP_TF: mcmd_check_proof(np); break;
         /* case OP_NULL: */
         default: {
            /* NOTE: some incoming requests are ignored during init */
            if (np->opcode == OP_FOUND) {
               if (!Ininit) mcmd_check_proof(np);
               else pdebug("ignore OP_FOUND broadcast during init");
            } else if (np->opreq == OP_NULL) {
               pdebug("unhandled opcode %s...", op2str(np->opcode));
            } else perr("unhandled opreq %s...", op2str(np->opreq));
         }
      }  /* end switch (np->opreq) */
      /* cleanup node resources */
      mcmd__cleanup(lnp);
      /* (re)acquire (exclusive) SyncIO Lock */
      result = mutex_lock(&syncIOLock);
      if (result != 0) {
         perrno_fatal("syncIO (re)LOCK FAILURE");
         break;
      }
   }  /* end for (lnp... */

   /* periodic synchronous tasks, after sync... */
   if (!Ininit && !Syncwait) {
      time(&now);
      /* check pseudoblock trigger time */
      if (Running && Ptime && difftime(Ptime, now) <= 0) {
         /* trigger pseudoblock update */
         result = mcmd_check_block(NULL);
         if (result == VEOK && Cblocknum[0] == 0xff) {
            /* if a pseudoblock occurs on 0x..ff, we cannot generate
            * an appropriate WOTS+ neogenesis block; RESYNC(init)... */
            save_ipl(Recentip_opt, Rplist, RPLISTLEN);
            mcmd__scaninit();  /* restarts initial sync */
         } else perrno_fatal("pseudoblock trigger FAILURE");
      }
      /* for additional checks do: */
      /* ... if (Running &&... */
   }

   /* release (exclusive) sync lock */
   result = mutex_unlock(&syncLock);
   if (result != 0) {
      perrno_fatal("sync UNLOCK FAILURE");
      return VERROR;
   }

   /* release (exclusive) syncIO lock */
   result = mutex_unlock(&syncIOLock);
   if (result != 0) {
      perrno_fatal("syncIO UNLOCK FAILURE");
      return VERROR;
   }

   /* THREAD IS NO LONGER PROCESSING SYNCHRONOUS TASKS */
   thread_setname(thread_self(), "mcmd-async");

   /* done */
   return VEOK;
}  /* end mcmd_worker_sync() */

ThreadProc mcmd_worker(void *arg)
{
   static time_t Stime;
   WorkList *wlp = (WorkList *) arg;
   Condition *alarmp = &(wlp->alarm);
   LinkedList *listp = &(wlp->list);
   Mutex *lockp = &(wlp->lock);

   LinkedNode *lnp;
   NODE *np;
   time_t now;
   int result;

   thread_setname(thread_self(), "mcmd-async");
   pdebug("daemon worker(thrd.%x) startup...", thread_self());

   /* acquire (exclusive) lock */
   result = mutex_lock(lockp);
   if (result != 0) {
      perrno_fatal("LOCK FAILURE");
      Unthread;
   }

   /* daemon worker thread loop */
   while (Running) {
      /* check/pull next work */
      while (Running && (lnp = listp->next)) {
         /* remove NODE from list */
         result = link_node_remove(lnp, listp);
         if (result != 0) {
            perrno_fatal("LIST FAILURE");
            if (mutex_unlock(lockp) != 0) {
               perrno_fatal("(re)UNLOCK LIST FAILURE");
            }
            Unthread;
         }
         /* (re)release (exclusive) lock */
         result = mutex_unlock(lockp);
         if (result != 0) {
            perrno_fatal("(re)UNLOCK FAILURE");
            Unthread;
         }
         /* determine (a)synchronous workflow */
         np = (NODE *) lnp->data;
         switch (np->opreq) {
            case OP_TX: break;  /* no further processing for TX mirror */
            case OP_FOUND: mcmd_check_broadcast(np); break;
            default: {
               /* is request an incoming request... */
               if (np->opreq == OP_NULL) {
                  if (np->opcode == OP_TX) {
                     if (!Ininit) mcmd_check_tx(np);
                     else pdebug("ignore OP_TX during init");
                     break;
                  }
               }  /* ... incoming OP_FOUND will fall through for sync */
               time(&Stime);
               /* remaining request operations require sync */
               mcmd_worker_sync(lnp);
               lnp = NULL;
            }
         }  /* end switch (np->opreq) */
         /* cleanup remaining data */
         if (lnp) mcmd__cleanup(lnp);
         /* (re)acquire (exclusive) lock */
         result = mutex_lock(lockp);
         if (result != 0) {
            perrno_fatal("(re)LOCK FAILURE");
            Unthread;
         }
      }  /* end while ((lnp = listp->next)) */
      /* periodic checks after initial sync... */
      if (Running) {
         time(&now);
         /* check pseudoblock trigger time */
         if (difftime(Stime, now) < 0) {
            time(&Stime);
            /* periodicly check sync processing */
            mcmd_worker_sync(NULL);
         }
      }
      /* wait for work, sleepy time (1 second timeout)... */
      result = condition_timedwait(alarmp, lockp, 1000);
      if (result != 0 && errno != CONDITION_TIMEOUT) {
         perrno_fatal("CONDITION FAILURE");
         if (mutex_unlock(lockp) != 0) {
            perrno_fatal("UNLOCK CONDITION FAILURE");
         }
         Unthread;
      }
   }  /* end while (Running) */
   pdebug("daemon worker(thrd.%x) shutdown...", thread_self());

   /* release (exclusive) lock */
   result = mutex_unlock(lockp);
   if (result != 0) {
      perrno_fatal("UNLOCK FAILURE");
   }

   Unthread;
}  /* end mcmd_worker() */

/* initialize AsyncWork with NODE data for the MCM Server Daemon */
static int mcmdio_accept(AsyncWork *wp)
{
   word32 ip;
   char ipstr[16];

   /* ensure valid work/socket or log error */
   if (wp == NULL || wp->sd == INVALID_SOCKET) {
      perr("mcmdio_accept() was called with invalid work!");
      return VERROR;
   }

   /* get socket ip and perform initialization checks */
   ip = get_sock_ip(wp->sd);
   if (pinklisted(ip)) {
      pdebug("drop connection from %s: pinklisted", ntoa(&ip, ipstr));
      return VERROR;
   }
   /* create space for NODE in work pointer */
   wp->data = malloc(sizeof(NODE));
   if (wp->data == NULL) {
      perrno("node creation FAILURE");
      return VERROR;
   }
   /* set socket non-blocking */
   if (sock_set_nonblock(wp->sd)) {
      set_sockerrno(sock_errno);
      perrno("set non-blocking FAILURE");
      return VERROR;
   }
   /* initialize NODE for receiving -- update work timeout */
   node_init((NODE *) wp->data, wp->sd, ip, 0, OP_NULL, NULL);
   wp->to = ((NODE *) wp->data)->to;

   /* success */
   return VEOK;
}  /* end mcmdio_accept() */

/* transfer completed AsyncWork to CompleteIO for processing */
static int mcmdio_finish(AsyncWork *wp)
{
   LinkedNode *lnp;
   NODE *np;
   int result;

   /* ensure valid work/data or log error */
   if (wp == NULL || wp->data == NULL) {
      perr("mcmdio_finish() was called with invalid work!");
      return VERROR;
   }

   /* log communication result */
   np = (NODE *) wp->data;
   if (np->opreq) {
      pdebug("%s request(%s) finished on %s...",
         np->id, op2str(np->opreq), op2str(np->opcode));
   } else pdebug("%s receive(%s) finished...", np->id, op2str(np->opcode));

   /* create an empty LinkedNode -- acquire CompleteIO lock */
   lnp = link_node_create(0);
   if (lnp == NULL) {
      perrno("link node creation FAILURE");
      return VERROR;
   }
   /* acquire (exclusive) CompleteIO lock */
   result = mutex_lock(&(CompleteIO.lock));
   if (result != 0) {
      perrno_fatal("CompleteIO LOCK FAILURE");
      return VERROR;
   }
   /* add LinkedNode to CompleteIO list */
   result = link_node_append(lnp, &(CompleteIO.list));
   if (result != 0) {
      perrno("link node append FAILURE");
      if (mutex_unlock(&(CompleteIO.lock))) {
         perrno_fatal("inner CompleteIO UNLOCK FAILURE");
      }
      free(lnp);
      return VERROR;
   }
   /* move work data to LinkedNode */
   lnp->data = wp->data;
   wp->data = NULL;
   /* flag additional work signal */
   condition_signal(&(CompleteIO.alarm));
   /* release (exclusive) CompleteIO lock */
   result = mutex_unlock(&(CompleteIO.lock));
   if (result != 0) {
      perrno_fatal("CompleteIO UNLOCK FAILURE");
      return VERROR;
   }

   /* success */
   return VEOK;
}  /* end mcmdio_finish() */

/* process AsyncWork for the MCM Server Daemon */
static int mcmdio_io(AsyncWork *wp)
{
   NODE *np;
   int ecode;
   char error[64];

   /* ensure valid work/data or log error */
   if (wp == NULL || wp->data == NULL) {
      perr("mcmdio_io() was called with invalid work!");
      return VERROR;
   }

   /* init and check communication status */
   np = (NODE *) wp->data;
   if (np->iowait != IO_DONE) {
      /* execute communication protocols */
      if (np->opreq) node_request_operation(np);
      else node_receive_operation(np);
      /* debug log resulting error descriptions */
      if (np->status != VEOK && np->status != VEWAITING) {
         ecode = errno;
         strerror_mcm(ecode, error, sizeof(error));
         pdebug("%s: (%d) %s", np->id, ecode, error);
      }
   }
   /* update work */
   wp->to = np->to;
   wp->sd = np->sd;
   wp->sio = np->iowait;
   /* defer disk IO when incomplete */
   wp->defer = np->fp ? 1 : 0;
   return np->status;
}  /* end mcmdio_io() */

void signal_handler(int sig)
{
   printf("\n");
   switch (sig) {
		case SIGABRT: plog("Caught SIGABRT"); break;
		case SIGFPE:  plog("Caught SIGFPE"); break;
		case SIGILL:  plog("Caught SIGILL"); break;
		case SIGINT:  plog("Caught SIGINT"); break;
		case SIGSEGV: plog("Caught SIGSEGV"); break;
		case SIGTERM: plog("Caught SIGTERM"); break;
		default: plog("Caught SIG(%d)", sig);
	}
   /* signal server shutdown -- or exit */
   if (Running) {
      plog("Server exiting, please wait...");
      Running = 0;
   } else {
      plog("Server (violently) terminated");
      exit(1);
   }
}  /* end signal_handler() */

void global_data_dump(void)
{
   char hex[65];

   /*
   pdebug("GLOBAL COUNTERS DATA DUMP...");
   pdebug("Nbalance = %u", Nbalance);
   pdebug("Nhashes = %u", Nhashes);
   pdebug("Niplist = %u", Niplist);
   pdebug("Nnacks = %u", Nnacks);
   pdebug("Nrecvs = %u", Nrecvs);
   pdebug("Nrecverrs = %u", Nrecverrs);
   pdebug("Nrecvsbad = %u", Nrecvsbad);
   pdebug("Nsends = %u", Nsends);
   pdebug("Nsenderrs = %u", Nsenderrs);
   */

   pdebug("");
   pdebug("CHAIN STATE DATA DUMP...");
   pdebug("Cbits = 0x%x", Cbits);
   pdebug("Cblocknum = 0x%s", bnum2hex(Cblocknum, hex));
   pdebug("Cblockhash = 0x%s", hash2hex(Cblockhash, 32, hex));
   pdebug("Prevhash = 0x%s", hash2hex(Prevhash, 32, hex));
   pdebug("Weight = 0x%s", weight2hex(Weight, hex));
   pdebug("");

}  /* end global_data_dump() */

int check_permissions(const char *dir) {
   char permchk[FILENAME_MAX];

   /* build permission check file path */
   snprintf(permchk, FILENAME_MAX, "%s%sperm.chk",
      dir ? dir : "", dir ? PATH_SEPARATOR : "");
   /* touch the file and remove it, checking for failures */
   if (ftouch(permchk) || remove(permchk)) {
      perrno("%s FAILURE", permchk);
      return VERROR;
   }

   return VEOK;
}

int veronica(void)
{
   char haiku[256], buff[16];
   printf("\n%s\n\n", trigg_expand(trigg_generate(buff), haiku));
   return VEOK;
}

int usage(void)
{
   printf(
      "\nUSAGE: mcmd [OPTIONS]... [--] [DIRECTORY]"

      "\n\nDIRECTORY:"
      "\n   Defaults to \"d/\""

      "\n\nOPTIONS:"
      "\n   --                         Forces the end of OPTIONS arguments"
      "\n   -a, --server-addr=<ipv4>   Set server IPv4 address to <ipv4>"
      "\n   -d, --daemon-threads=<num> Set daemon thread count to <num>"
      "\n   --dir-bc=<dir>             Set block archive directory to <dir>"
      "\n   -h, --help                 Print this usage information"
      "\n   --no-pinklist              Disable the pinklist of evil peers"
      "\n   --no-pushblock             Disable block push capability"
      "\n   -p, --port=<num>           Set server port number to <num>"
      "\n   --private-peers            Allow private peers in peer lists"
      "\n   -q, --quorum=<num>         Set network quorum size to <num>"
      "\n   -s, --server-threads=<num> Set server thread count to <num>"
      "\n   --version                  Print the current software version"

      "\n\nOPTIONS (trace logging):"
      "\n   -tf, --trace-functions     Trace functions in trace logs"
      "\n   -tl, --trace-level=<ll>    Set trace log level to <ll>"
      "\n      Trace levels (0-5) represent the level of detail in logs."
      "\n      Each trace level includes the logs of all lower levels."
      "\n      0: Alert, 1: Errno, 2: Error, 3: Warning, 4: Info, 5: Debug"
      "\n   -tt, --trace-timestamp     Trace timestamp in trace logs"

      "\n\nOPTIONS (peerlist):"
      "\n   -ep, --epink-plist=<file>  Set epoch pinklist to <file>"
      "\n   -lp, --local-plist=<file>  Set local peer list to <file>"
      "\n   -rp, --recent-plist=<file> Set recent peer list to <file>"

      "\n\nOPTIONS (advanced):"
      "\n   -A, --API"
      "\n      API mode enables mysql export of blockchain data and"
      "\n      a simple http server for query and analysis of data."
      "\n   -M, --Mfee=<num>"
      "\n      Set the minimum mining fee threshold to <num> nMCM."
      "\n      All transactions received that do not meet this"
      "\n      threshold will be ignored. Cannot be < 500 nMCM."
      "\n   -S, --Sanctuary=<num>,<last>"
      "\n      Set Sanctuary to <num> and Lastday to the neogenesis"
      "\n      block proceeding <last>. All ledger addresses not in"
      "\n      compliance with network consensus will be dropped."
      "\n   -T, --Testnet=<port>"
      "\n      Testnet initialization requires a donor blockchain."
      "\n      Donor blockchains can be obtained via network sync."
      "\n   -V, --Virtual=<num>"
      "\n      Virtual mode uses alternate ports for receiving and"
      "\n      and sending network communications; exact ports are"
      "\n      defined as PORT1 and PORT2 at compile time. Virtual"
      "\n      modes 1 and 2 (as <num>) may be specified."
      "\n\n"
   );

   return 0;
}  /* end usage() */

int main (int argc, char *argv[])
{
   static const size_t enable_sz = sizeof(int);
   static const int enable = 1;

   static Thread *thrdp;
   static int j, eoa;
   static int int_opt;                    /* integer for cli options */
   static unsigned uint_opt;              /* unsigned for cli options */
   static char *char_opt, *char_opt2;     /* char for cli options */
   static char *proc_name, *working_dir;
   static char hostname[64], addrname[64];
   static char fpath[FILENAME_MAX];

   int result;
   int mcmd_threads;
   int server_threads;
   unsigned long server_addr;
   word16 server_port;

#ifndef _WIN32
   /* initialize signals -- POSIX */
   struct sigaction handle, ignore;
   handle.sa_handler = signal_handler;
   ignore.sa_handler = SIG_IGN;
   sigemptyset(&(handle.sa_mask));
   sigemptyset(&(ignore.sa_mask));
   handle.sa_flags = 0;
   ignore.sa_flags = 0;

   sigaction(SIGABRT, &handle, NULL);
   sigaction(SIGFPE, &handle, NULL);
   sigaction(SIGILL, &handle, NULL);
   sigaction(SIGINT, &handle, NULL);
   sigaction(SIGSEGV, &handle, NULL);
   sigaction(SIGTERM, &handle, NULL);

   sigaction(SIGPIPE, &ignore, NULL);

/* end POSIX signal handling */
#endif

   /* use multiple sources of entropy for improved prng */
   srand((unsigned int) time(NULL));
   srand16((word32) time(NULL), (word32) rand(), (word32) getpid());
   srand16fast((word32) time(NULL) ^ rand() ^ getpid());
   /* init process utils */
   ptrace_level(PTRACE_INFO);
   atexit(global_data_dump);
   sock_startup();
   /* init defaults */
   mcmd_threads = 2;
   server_threads = 1;
   server_addr = INADDR_ANY;
   server_port = PORT1;
   Dstport_opt = PORT1;
   Noprivate_opt = 1;
   Quorum_opt = 4;
   Cbits |= C_PUSH;
   Myfee[0] = MFEE;
   Mfee[0] = MFEE;
   Running = 1;

   /* derive process name, check for duplicates */
   proc_name = strrchr(argv[0], PATH_SEPARATOR[0]);
   proc_name = proc_name ? proc_name + 1 : argv[0];
   if (proc_dups(proc_name)) {
      perr("Daemon is already running!");
      return VERROR;
   }

   /* parse command line arguments */
   for (j = 1, eoa = 0; Running && j < argc; j++) {
      if (argv[j][0] == '-') {
         /***********/
         /* OPTIONS */
         if (eoa || argument(argv[j], NULL, "--")) {
            /* flag to skip remaining options with leading '-' */
            if (eoa++ == 0) {
               plog("... end of arguments");
            }
         }
         else if (argument(argv[j], "-a", "--server-addr")) {
            char_opt = argvalue(&j, argc, argv);
            if (char_opt == NULL) {
               perr("Missing server IPv4 address");
               return VERROR;
            }
            server_addr = aton(char_opt);
            plog("... server address = %s (%lu)", char_opt, server_addr);
         }
         else if (argument(argv[j], "-d", "--daemon-threads")) {
            /* obtain daemon thread count */
            char_opt = argvalue(&j, argc, argv);
            if (char_opt == NULL) {
               perr("Missing daemon thread count");
               return VERROR;
            }
            int_opt = atoi(char_opt);
            if (int_opt < 1) {
               perr("Invalid daemon thread count");
               return VERROR;
            }
            if (int_opt > cpu_cores()) {
               pwarn("daemon threads exceeds logical CPU cores");
            }
            plog("... daemon thread count = %s", char_opt);
            /* set daemon thread count */
            mcmd_threads = int_opt;
         }
         else if (argument(argv[j], NULL, "--dir-bc")) {
            /* obtain blockchain archive directory */
            char_opt = argvalue(&j, argc, argv);
            if (char_opt == NULL) {
               perr("Missing bc directory");
               return VERROR;
            }
            plog("... blockchain archive directory = %s", char_opt);
            /* set blockchain archive directory */
            Bcdir_opt = char_opt;
         }
         else if (argument(argv[j], "-h", "--help")) {
            /* exit with usage information */
            return usage();
         }
         else if (argument(argv[j], NULL, "--no-pinklist")) {
            plog("... pinklist disabled");
            /* set "nopinklist" flag */
            Nopinklist_opt = 1;
         }
         else if (argument(argv[j], NULL, "--no-pushblock")) {
            plog("... push blocks disabled");
            /* unset PUSH capability bit and set "nopush" flag */
            Cbits &= ~(C_PUSH);
            Nopush_opt = 1;
         }
         else if (argument(argv[j], "-p", "--port")) {
            /* obtain/check port number as integer */
            char_opt = argvalue(&j, argc, argv);
            if (char_opt == NULL) {
               perr("Missing port number");
               return VERROR;
            }
            int_opt = atoi(char_opt);
            if (int_opt < 1 || int_opt > 65535) {
               perr("Invalid port number");
               return VERROR;
            }
            plog("... port = %s (%d)", char_opt, int_opt);
            /* set port number for receive and destination ports */
            server_port = Dstport_opt = (word16) int_opt;
         }
         else if (argument(argv[j], NULL, "--private-peers")) {
            plog("... private peers enabled");
            /* set "noprivate" flag */
            Noprivate_opt = 0;
         }
         else if (argument(argv[j], "-q", "--quorum")) {
            /* obtain/check quorum number as integer */
            char_opt = argvalue(&j, argc, argv);
            if (char_opt == NULL) {
               perr("Missing quorum size");
               return VERROR;
            }
            uint_opt = strtoul(char_opt, NULL, 0);
            if (uint_opt < 1) {
               perr("Invalid quorum size");
               return VERROR;
            }
            plog("... quorum = %s (%u)", char_opt, uint_opt);
            /* set quorum number */
            Quorum_opt = uint_opt;
         }
         else if (argument(argv[j], "-s", "--server-threads")) {
            /* obtain server thread count */
            char_opt = argvalue(&j, argc, argv);
            if (char_opt == NULL) {
               perr("Missing thread count");
               return VERROR;
            }
            int_opt = atoi(char_opt);
            if (int_opt < 1) {
               perr("Invalid thread count");
               return VERROR;
            }
            if (int_opt > cpu_cores()) {
               pwarn("server threads exceeds logical CPU cores");
            }
            plog("... server thread count = %s", char_opt);
            /* set server thread count */
            server_threads = int_opt;
         }
         else if (argument(argv[j], NULL, "--Veronica")) {
            return veronica();
         }
         else if (argument(argv[j], NULL, "--version")) {
            /* print (only) GIT_VERSION information */
            printf(GIT_VERSION "\n");
            return VEOK;
         }
         /*********************/
         /* TRACE LOG OPTIONS */
         else if (argument(argv[j], "-tf", "--trace-functions")) {
            plog("... trace functions");
            ptrace_functions(1);
         }
         else if (argument(argv[j], "-tl", "--trace-level")) {
            char_opt = argvalue(&j, argc, argv);
            if (char_opt == NULL) {
               perr("Missing trace level");
               return VERROR;
            }
            int_opt = atoi(char_opt);
            if (int_opt < PTRACE_ALERT || int_opt > PTRACE_DEBUG) {
               perr("Invalid trace level");
               return VERROR;
            }
            plog("... trace level = %s (%d)", char_opt, int_opt);
            ptrace_level(int_opt);
         }
         else if (argument(argv[j], "-tf", "--trace-timestamp")) {
            plog("... trace timestamps");
            ptrace_timestamp(1);
         }
         /********************/
         /* PEERLIST OPTIONS */
         else if (argument(argv[j], "-ep", "--epink-plist")) {
            /* obtain peerlist filename */
            Epinkip_opt = argvalue(&j, argc, argv);
            if (Epinkip_opt == NULL) perr_exit("Missing peerlist");
            plog("... epoch pinklist = %s", Epinkip_opt);
         }
         else if (argument(argv[j], "-lp", "--local-plist")) {
            /* obtain peerlist filename */
            char_opt = argvalue(&j, argc, argv);
            if (char_opt == NULL) {
               perr("Missing local peerlist filename");
               return VERROR;
            }
            /* read-in peers from filename and log results */
            int_opt = read_ipl(char_opt, Lplist, LPLISTLEN, &Lplistidx);
            plog("... local peerlist = %s (%d peers)", char_opt, int_opt);
         }
         else if (argument(argv[j], "-rp", "--recent-plist")) {
            /* obtain peerlist filename */
            Recentip_opt = argvalue(&j, argc, argv);
            if (Recentip_opt == NULL) perr_exit("Missing peerlist");
            plog("... recent peerlist = %s", Recentip_opt);
         }
         /********************/
         /* ADVANCED OPTIONS */
         else if (argument(argv[j], "-mf", "--mining-fee")) {
            /* obtain/check mining fee as integer */
            char_opt = argvalue(&j, argc, argv);
            if (char_opt == NULL) perr_exit("Missing fee value");
            int_opt = atoi(char_opt);
            if (int_opt < MFEE) perr_exit("Invalid fee value");
            plog("... mining fee (Myfee) = %s (%d)", char_opt, int_opt);
            /* set Myfee, and MFEE capability bit if non-standard */
            Myfee[0] = int_opt;
            if (cmp64(Myfee, Mfee)) Cbits |= C_MFEE;
         }
         else if (argument(argv[j], NULL, "--Sanctuary")) {
            /* obtain/check Sanctuary/Lastday as unsigned long */
            char_opt = argvalue(&j, argc, argv);
            if (char_opt == NULL) perr_exit("Missing protocol data");
            char_opt2 = strchr(char_opt, ',');
            if (char_opt2) *(char_opt2++) = '\0';  /* create separation */
            else perr_exit("Malformed protocol data");
            Sanctuary_opt = strtoul(char_opt, NULL, 0);
            Lastday_opt = (strtoul(char_opt2, NULL, 0) + 255) & 0xffffff00;
            plog("... Sanctuary %s (%lu), %s (%lu)",
               char_opt, (unsigned long) Sanctuary_opt,
               char_opt2, (unsigned long) Lastday_opt);
         }
         else if (argument(argv[j], "-V", "--Virtual")) {
            /* obtain/check virtual mode as integer */
            char_opt = argvalue(&j, argc, argv);
            if (char_opt == NULL) int_opt = 1;
            else int_opt = atoi(char_opt);
            if (int_opt < 1 || int_opt > 2) {
               perr_exit("Invalid Virtual mode");
            }
            Dstport_opt = int_opt == 2 ? PORT1 : PORT2;
            server_port = int_opt == 2 ? PORT2 : PORT1;
            plog("... virtual mode = %d", int_opt);
         }
         /*********************/
         /* DEVELOPER OPTIONS */
         else if (argument(argv[j], "-tb", "--trust-block")) {
            /* Set a trusted block number (for development use only).
             * Skips PoW validation up to specified block (inclusive). */
            /* obtain trust block as unsigned long */
            char_opt = argvalue(&j, argc, argv);
            if (char_opt == NULL) perr_exit("Missing trust block number");
            Trustblock_opt = strtoul(char_opt, NULL, 0);
            plog("... trust block = %s (%lu)", char_opt,
               (unsigned long) Trustblock_opt);
         }
         /******************/
         /* UNKNOWN OPTION */
         else perr_exit("Unknown argument, %s", argv[j]);
      } else if (argv[j][0]) {
         /* additional non-option arguments */
         if (working_dir == NULL) {
            working_dir = argv[j];
            plog("-- working directory = %s", working_dir);
         }
      }  /* end if arguments... */
   }  /* end for j */
   printf("\n");

   /* change working directory -- check location */
   if (working_dir == NULL) working_dir = "d";
   if (cd(working_dir) != 0) {
      perrno("Cannot change DIRECTORY to \"%s\"", working_dir);
      plog("Working directory unavailable. Check installation.");
      exit(0);
   } else if (fexists(proc_name)) {
      perr("Found executing binary '%s' in working directory", proc_name);
      plog("Cowardly refusing to work in specified directory");
      exit(0);
   }

   /* check permissions of directory structure */
   if (check_permissions(NULL) != VEOK) exit(0);
   if (check_permissions(Bcdir_opt) != VEOK) exit(0);

   /* print copyright and version information */
   plog("%s %s, built " __DATE__ " " __TIME__, EXEC_NAME, GIT_VERSION);
   plog("Copyright (c) 2018-2023 Adequate Systems, LLC.  All Rights Reserved.");
   plog("See the License Agreement at the links below:");
   plog("   https://mochimo.org/license.pdf (PDF version)");
   plog("   https://mochimo.org/license (TEXT version)");
   printf("\n");
   /* get local machine name and IP address */
   gethostname(hostname, sizeof(hostname));
   gethostip(addrname, sizeof(addrname));
   /* print host info -- 1 second */
   plog("Local Machine Info");
   plog("  Machine name: %s", *hostname ? hostname : "unknown");
   plog("  IPv4 address: %s", *addrname ? addrname : "0.0.0.0");
   printf("\n");

   /* initialize genesis block filename */
   path_join(fpath, Bcdir_opt, "b0000000000000000.00170c67.bc");
   /* (try) restore core chain files if any do not exist */
   if (!fexists("tfile.dat") || !fexists(fpath)) {
      pdebug("Core chain files missing, attempting restoration...");
      if (!fexists("genblock.bc")) {
         perr_exit("Genesis block missing, cannot restore files!");
      } else if (fcopy("genblock.bc", fpath) != VEOK) {
         perr_exit("Failed to restore Genesis block");
      }
      remove("tfile.dat");
      if (append_tfile(fpath, "tfile.dat") != VEOK) {
         perr_exit("Failed to restore Tfile from Genesis block");
      } else if (le_extract("genblock.bc") != VEOK) {
         perr_exit("Failed to restore Ledger from Genesis block");
      } else pdebug("Restoration of core chain files successful!");
   }

   /* init Mochimo network communication server */
   result = server_init(MCMDIO, "mcmdio", AF_INET, SOCK_STREAM, 0);
   if (result != VEOK) perrno_exit("server initialization FAILURE");
   /* set additional server work event handlers */
   MCMDIO->on_accept = mcmdio_accept;
   MCMDIO->on_finish = mcmdio_finish;
   MCMDIO->on_io = mcmdio_io;
   /* set server socket address for REUSE */
   /** @todo Disable SO_REUSEADDR outside of development builds. */
   setsockopt(MCMDIO->lsd, SOL_SOCKET, SO_REUSEADDR, &enable, enable_sz);
   /* start network node server */
   result = server_start(MCMDIO, server_addr, server_port, server_threads);
   if (result != VEOK) perrno_exit("server start FAILURE");

   /* trigger daemon initialization/synchronization */
   if (mcmd__scaninit() != VEOK) perr_exit("daemon initialization FAILURE");

   /* start Mochimo daemon threads for processing */
   thrdp = calloc(mcmd_threads, sizeof(Thread));
   if (thrdp == NULL) perrno_exit("daemon threads FAILURE");
   /* start daemon worker threads for processing */
   for (j = 0; j < mcmd_threads; j++) {
      result = thread_create(&thrdp[j], mcmd_worker, &CompleteIO);
      if (result != 0) {
         perrno_fatal("daemon worker thread#%d FAILURE", j);
         mcmd_threads = j + 1;
         break;
      }
   }

   /* BLOCK and wait for daemon worker threads to exit */
   while (mcmd_threads > 0) thread_join(thrdp[--mcmd_threads]);
   /* cleanup daemon threads pointer */
   free(thrdp);

   /* clear quorum pools */
   if (Splistidx) free(Splist);
   if (Qplistidx) free(Qplist);
   /* shutdown io server */
   server_shutdown(MCMDIO);
   /* save recnt peers */
   if (Rplist[0]) save_ipl(Recentip_opt, Rplist, RPLISTLEN);
   /* cleanup sockets */
   sock_cleanup();

   /* ignore unused compiler warnings for... */
   (void)Localip_opt;

   /* done */
   return VEOK;
}  /* end main() */
