
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
WorkList CompleteIO = WORKLIST_INITIALIZER;

Server NodeServer;

/* Recent peers lock */
RWLock RplistLock = RWLOCK_INITIALIZER;

/* Mochimo globals */
int Ininit = 1;
int Inscan = 1;
int Running = 1;
int Nopush_opt = 0;
NODE *Syncwait = NULL;
word32 Mfee[2] = { MFEE, 0 };
word32 Myfee[2] = { MFEE, 0 };
word32 Quorum_opt = 4;
word64 Trustblock_opt = 0;
word16 Dstport_opt = PORT1;
word16 Port_opt = PORT1;

/* cli options for directories */
char *Spdir_opt = "sp";

/* cli options for peerlist filenames */
char *Coreip_opt = "coreip.lst";
char *Epinkip_opt = "epink.lst";
char *Localip_opt = "local.lst";
char *Recentip_opt = "recent.lst";
char *Startip_opt = "start.lst";

/* cli options for peerlist web address */
char *Starthttp_opt = "https://mochimo.org/peers/start";

int archive_block(char *blockfile)
{
   BTRAILER bt;
   char fname[FILENAME_MAX];
   char fpath[FILENAME_MAX];

#undef FnMSG
#define FnMSG(x) "archive_block(%s): " x, blockfile

   /* get blockfile trailer data */
   if (read_trailer(&bt, blockfile) != VEOK) {
      return perrno(errno, FnMSG("read_trailer(%s) FAILURE"), blockfile);
   }
   /* build archive path and name -- clear path */
   bc_fqan(fname, bt.bnum, bt.bhash);
   path_join(fpath, Bcdir_opt, fname);
   remove(fpath);
   /* archive file to specified path */
   if (rename(blockfile, fpath) != 0) {
      return perrno(errno, FnMSG("rename(%s, %s) FAILURE"), fname, fpath);
   }

   return VEOK;
}  /* end archive_block() */

/**
 * Distributed PoW validation. Typically Multi-Threaded.
 * @param arg Pointer to NODE
 * @private for internal use only
 */
ThreadProc dist__pow_val(void *arg)
{
   NODE *np = arg;

   Mutex lock = MUTEX_INITIALIZER;
   BTRAILER bt;
   int ecode;
   char bnumstr[17];

#undef FnMSG
#define FnMSG(x) "dist__pow_val(): " x

   /* set name of thread - visible in htop */
   thread_setname(thread_self(), "PoW-validation");

   /* acquire lock before loop condition */
   if (mutex_lock(&lock) != 0) goto FAIL;

   while (Running && np->status == VEOK) {
      if (fread(&bt, sizeof(bt), 1, np->fp) != 1) {
         if (feof(np->fp)) break;
         np->status = VERROR;
         break;
      }
      /* release lock and validate pow */
      mutex_unlock(&lock);
      if (cmp64(bt.bnum, &Trustblock_opt) > 0) {
         ecode = validate_pow(&bt);
      } else ecode = VEOK;
      /* (re)acquire lock before loop condition */
      if (mutex_lock(&lock) != 0) goto FAIL;
      if (ecode) {
         bnum2hex(bt.bnum, bnumstr);
         perrno(errno, FnMSG("0x%s INVALID"), bnumstr);
         np->status = ecode;
      }
   }

   /* release lock -- exit */
   mutex_unlock(&lock);
   Unthread;

FAIL:
   np->status = VERROR;
   Unthread;
}  /* end dist__pow_val() */

static void mcmd__cleanup_node(LinkedNode *lnp)
{
   /* perform cleanup of NODE resources -- deallocate memory */
   if (lnp->data) {
      node_cleanup(lnp->data);
      free(lnp->data);
   }
   free(lnp);
}  /* end mcmd__cleanup_node() */

int mcmd__create_request(word32 ip, word16 opreq, void *bnum)
{
   NODE *np;
   int ecode;
   char ipstr[16];
   char bnumstr[17];

#undef FnMSG
#define FnMSG(x) "mcmd__create_request(%s, %s, 0x%s): " x, \
      ntoa(&ip, ipstr), op2str(opreq), bnum ? bnum2hex(bnum, bnumstr) : "0"
   pdebug(FnMSG("requesting..."));

   np = malloc(sizeof(*np));
   if (np == NULL) goto_perrno(FAIL, FnMSG("malloc() FAILURE"));
   /* initialize node requests and send to Server handler */
   node_init(np, INVALID_SOCKET, ip, Dstport_opt, opreq, bnum);
   on_ecode_goto_perrno( server_work_create(&NodeServer, np),
      FAIL, FnMSG("server_work_create() FAILURE"));

   /* store Syncwait task on certain tasks */
   switch (opreq) {
      case OP_GET_BLOCK:  /* fallthrough */
      case OP_GET_TFILE:  /* fallthrough */
      case OP_TF: Syncwait = np; break;
   }

   return VEOK;

/* error handling */
FAIL: return VERROR;
}  /* end mcmd__create_request() */

/**
 * @brief 
 * @param np 
 * @return 
int mcmd__restore(void)
{
   restore saved blockchain data: Tfile, and (compressed) Ledger
   - possibly from a blockchain state file? *.bs ...
   reset protocol state based on latest Tfile data
}
 */

/**
 * @brief https://app.code2flow.com/zMh76o
 * @param np 
 * @return 
 */
int mcmd__resync(NODE *np)
{
   static word8 eon[8] = { 0, 1 };
   static word8 one[8] = { 1 };

   BTRAILER bt, ibt, tbt;
   long long seek;
   FILE *tfp;
   word8 lpngblock[8];
   word8 syncblock[8];
   char bnumstr[17];
   char fname[FILENAME_MAX];
   char fpath[FILENAME_MAX];

#undef FnMSG
#define FnMSG(x) "mcmd__resync(%s): " x, np->id

   /* ensure backups of chain before continuing */
   if (!fexists("tfile.bak") && fcopy("tfile.dat", "tfile.bak") != 0) {
      goto_perrno(FATAL, FnMSG("Tfile backup FAILURE"));
   }

   /* obtain final block trailer from update */
   if (fseek64(np->fp, -(sizeof(bt)), SEEK_END) != 0) goto FAIL;
   if (fread(&bt, sizeof(bt), 1, np->fp) != 1) goto FAIL_FP;
   /* derive resync block number (last-previous neogenesis) */
   if (sub64(bt.bnum, eon, lpngblock)) {
      memset(lpngblock, 0, sizeof(lpngblock));
   } else lpngblock[0] = 0;
   /* obtain initial block trailer from update */
   if (fseek64(np->fp, 0LL, SEEK_SET) != 0) goto FAIL;
   if (fread(&bt, sizeof(bt), 1, np->fp) != 1) goto FAIL_FP;
   memcpy(&ibt, &bt, sizeof(bt));  /* save for later */
   /* open Tfile for binary read */
   tfp = fopen("tfile.dat", "rb");
   if (tfp == NULL) return VERROR;
   /* derive and seek to update block number */
   seek = 0;
   put64(&seek, bt.bnum);
   seek *= sizeof(bt);
   if (fseek64(tfp, seek, SEEK_SET) != 0) goto FAIL_TFP;
   /* obtain block trailer from Tfile */
   if (fread(&tbt, sizeof(tbt), 1, tfp) != 1) goto FAIL_TFP;
   /* find split block, if any */
   while (memcmp(&bt, &tbt, sizeof(tbt)) == 0) {
      /* update splitblock */
      put64(syncblock, bt.bnum);
      /* on Tfile EOF, we have our splitblock */
      if (fread(&tbt, sizeof(tbt), 1, tfp) != 1) {
         if (feof(tfp)) break;
         goto FAIL_TFP;
      }
      /* fail on update fp error */
      if (fread(&bt, sizeof(bt), 1, np->fp) != 1) goto FAIL_TFP;
   }
   fclose(tfp);

   /* reduce erroneous synchronization -- update Tfile */
   if (cmp64(syncblock, lpngblock) < 0) {
      put64(syncblock, lpngblock);
      /* trim Tfile to the sync block number */
      if (trim_tfile("tfile.dat", syncblock, NULL) != VEOK) goto FAIL;
      /* update remaining chain data (append) */
      if (append_tfile_fp(np->fp, "tfile.dat") != VEOK) goto FAIL;
   }

   /* finalize (re)synchronization starting block */
   if (cmp64(syncblock, Cblocknum) < 0) syncblock[0] = 0;
   else add64(Cblocknum, one, syncblock);
   if (cmp64(syncblock, lpngblock) < 0) put64(syncblock, lpngblock);

   rewind(np->fp);
   /* running check -- try syncing to archived blocks */
   while (Running) {
      /* OP_FOUND might not contain sync block, use Tfile */
      if (cmp64(ibt.bnum, syncblock) > 0) {
         if (read_tfile(&bt, syncblock, 1, "tfile.dat") != 1) break;
      } else {
         /* skip to sync block (when reading from np->fp) */
         if (fread(&bt, sizeof(bt), 1, np->fp) != 1) break;
         if (cmp64(bt.bnum, syncblock) < 0) continue;
      }
      /* get name of archive block */
      bc_fqan(fname, bt.bnum, bt.bhash);
      path_join(fpath, Bcdir_opt, fname);
      if (!fexists(fpath)) break;
      pfine(FnMSG("recovering 0x%s..."), bnum2hex(bt.bnum, bnumstr));
      /* update chain with valid block file */
      if (update_block(fpath) != VEOK) {
         perrno(errno, FnMSG("update_block(%s)"), fpath);
         break;
      }
      /* maintain next sync block */
      add64(syncblock, one, syncblock);
   }

   /* get next block number */
   return mcmd__create_request(np->ip, OP_GET_BLOCK, syncblock);

/* error handling */
FAIL_TFP:
   if (feof(tfp)) set_errno(EMCM_EOF);
   fclose(tfp);
FAIL_FP:
   if (feof(np->fp)) set_errno(EMCM_EOF);
   goto FAIL;
FATAL: Running = 0;
FAIL: return (np->status = VERROR);
}

/**
 * @brief 
 * @param np 
 * @return 
 */
static int mcmd__send_found(NODE *np)
{
   static word32 plist[RPLISTLEN];
   static word32 plistidx;

   int inprogress = plist[0];
   int ecode;

#undef FnMSG
#define FnMSG(x) "mcmd__send_found(): " x

   /* refresh plist on new send_found trigger, else remove last */
   if (np == NULL) {
      on_ecode_goto_perrno( rwlock_rdlock(&RplistLock),
         FATAL, FnMSG("RplistLock LOCK FAILURE"));
      memcpy(plist, Rplist, sizeof(Rplist));
      plistidx = Rplistidx;
      on_ecode_goto_perrno( rwlock_rdunlock(&RplistLock),
         FATAL, FnMSG("RplistLock UNLOCK FAILURE"));
      pfine(FnMSG("broadcasting OP_FOUND to recent peers, %u"), plistidx);
      /* bail if already in progress */
      if (inprogress) return VEOK;
   } else remove32(np->ip, plist, RPLISTLEN, &plistidx);

   /* broadcast to next peer (if any) */
   if (plist[0]) return mcmd__create_request(plist[0], OP_FOUND, NULL);
   return pfine(FnMSG("OP_FOUND broadcast complete"));

FATAL:
   Running = 0;
   return VERROR;
}  /* end mcmd__send_found() */

/**
 * Same chain NODE synchronization.
 * @param np 
 * @return 
*/
int mcmd__syncup(NODE *np)
{
   static QUORUM quorum, scan;
   static word32 one[2] = { 1 };
   static int first_scan = 1;
   static int network_found;
   static int requests;

   BTRAILER tbt, bt;
   Thread *thrdp;
   word32 *plist, plistlen, ip, u;
   word32 bnum[2];
   word8 weight[32];
   int count, i;
   int partial;
   int ecode;
   char hexstr[65];
   char bnumstr[17];

#undef FnMSG
#define FnMSG(x) "mcmd__syncup(%s, %"P16u"): " x, np->id, np->opcode

   /* init */
   ecode = VERROR;
   plistlen = 0;
   plist = NULL;

   /* ORDER OF OPERATIONS:
    * - is np NULL? syncup initialization trigger
    * - is OP_GET_IPL? process OP_GET_IPL request (any status)
    * - is status VEOK? check blockchain synchronization stage
    * - is (any) opreq? DROP peer during init
    */

   if (np == NULL) {
      /* initialize network scan */
      plog("Scanning network...");
      plistlen = RPLISTLEN;
      plist = Rplist;
      goto SCAN_PEERS;
   } else if (np->opreq == OP_GET_IPL) {
      /*******************************/
      /* PEERLIST REQUEST PROCESSING */
      requests--;
      /* check success of connection */
      if (np->status == VEOK) {
         if (quorum_update(&quorum, np)) network_found = 1;
         /* process IP List on first scan */
         if (first_scan) {
            plist = (word32 *) PKTBUFF(&np->pkt);
            plistlen = (word32) get16(np->pkt.len) / sizeof(word32);
            /* limit list length to size of packet (just in case) */
            if (plistlen > PKTBUFFLEN) {
               plistlen = (word32) PKTBUFFLEN / sizeof(word32);
            }
         }
      }  /* end if (np->status == VEOK) */
      /* log found Node */
      bnum2hex(np->pkt.cblock, bnumstr);
      weight2hex(np->pkt.weight, hexstr);
      pfine(FnMSG("%s 0x%s 0x%s -- %d remaining"),
         np->id, bnumstr, hexstr, requests);
SCAN_PEERS:
      /* "recent" peers list requires read lock */
      if (plist == Rplist) {
         on_ecode_goto_perrno( rwlock_rdlock(&RplistLock),
            FATAL, FnMSG("RplistLock LOCK FAILURE"));
      }
      /* request OP_GET_IPL on peerlist members */
      for (u = 0; u < plistlen && plist[u]; u++) {
         ip = plist[u];
         if (quorum_addpeer(&scan, ip)) {
            if (mcmd__create_request(ip, OP_GET_IPL, NULL) == VEOK) {
               requests++;
            }
         }  /* end if (quorum_addpeer... */
      }  /* end for (u = 0; u < plistlen... */
      /* "recent" peers list requires read unlock */
      if (plist == Rplist) {
         on_ecode_goto_perrno( rwlock_rdunlock(&RplistLock),
            FATAL, FnMSG("RplistLock UNLOCK FAILURE"));
      }   /* wait for OP_GET_IPL requests to finish */
      /* wait for all async requests */
      if (requests) return VEOK;
      /* report highchain */
      bnum2hex(quorum.bnum, bnumstr);
      weight2hex(quorum.weight, hexstr);
      plog("Quorum members: %d / %d", quorum.idx, Quorum_opt);
      plog("Quorum chain: 0x%s / 0x%s", bnumstr, hexstr);
      /* check quorum requirements */
      if (quorum.idx == 0) {
         plog("No higher chain available");
         if (first_scan && !network_found) {
            pwarn("NETWORK NOT FOUND!");
            plog("Please check network configuration...");
         }
      } else if (quorum.idx < Quorum_opt) {
         /* remove quorum peers from scan list and clear quorum list */
         for (u = 0; u < quorum.idx; u++) {
            quorum_drop(&scan, quorum.list[u]);
         }
         quorum_cleanup(&quorum);
         plog("(re)Scanning network...");
         /* perform rescan on scan list peers */
         for (first_scan = 0, u = 0; u < scan.idx; u++) {
            ip = scan.list[u];
            if (mcmd__create_request(ip, OP_GET_IPL, NULL) == VEOK) {
               requests++;
            }
         }
      } else {
         /* build OP_TF bnum parameter */
         bnum[0] = get32(quorum.bnum);
         bnum[1] = (bnum[0] < 1000) ? bnum[0] + 1 : 1000;
         bnum[0] -= (bnum[0] < 1000) ? bnum[0] : (1000 - 1);
         /* request OP_TF from quorum members */
         plog("Synchronizing with %s...", ntoa(quorum.list, bnumstr));
         return mcmd__create_request(quorum.list[0], OP_TF, bnum);
      }
   } else if (np->status == VEOK) {
      /*************************************/
      /* TFILE/PROOF VALIDATION PROCESSING */
      memset(weight, 0, 32);
      if (np->opcode == OP_FOUND) {
         pdebug(FnMSG("checking OP_FOUND broadcast..."));
         /* check advertised weight */
         if (cmp256(np->pkt.weight, Weight) <= 0) {
            weight2hex(np->pkt.weight, hexstr);
            pfine(FnMSG("insufficient weight, 0x%s"), hexstr);
            goto FAIL;
         }
         /* move packet data to tmpfile() */
         np->fp = tmpfile();
         if (np->fp == NULL) goto_perrno(FAIL, FnMSG("tmpfile() FAILURE"));
         if (fwrite(np->pkt.buffer, sizeof(BTRAILER), 54, np->fp) != 54) {
            goto_perrno(FAIL, FnMSG("fwrite(tmpfile) FAILURE"));
         }
         /* jump to next section */
         goto TF;
      }  /* end if (np->opcode == OP_FOUND) */
      if (np->opreq == OP_TF) {
         pdebug(FnMSG("checking OP_TF response..."));
TF:      /* perform partial chain analysis checks
          * - is there a gap between our chain and the proof?
          * - is there a chain split preceeding the proof?
          * - is resync required to synchronize to the proof? */
         rewind(np->fp);
         if (fread(&bt, sizeof(bt), 1, np->fp) != 1) {
            if (feof(np->fp)) set_errno(EMCM_EOF);
            goto_perrno(FATAL, FnMSG("fread(bt) FAILURE"));
         } else if (read_tfile(&tbt, bt.bnum, 1, "tfile.dat") != 1) {
            /* cannot accept out of range "proof" outside of Ininit */
            if (!Ininit) {
               pfine(FnMSG("Ignoring proof out of range"));
               goto FAIL;
            }
            /* request the bigger picture (OP_GET_TFILE)... */
            pfine(FnMSG("Chain gap preceeding proof, request Tfile..."));
            return mcmd__create_request(np->ip, OP_GET_TFILE, NULL);
         } else if (memcmp(&tbt, &bt, sizeof(bt)) != 0) {
            /* cannot accept out of range "proof split" outside of Ininit */
            if (!Ininit) {
               pfine(FnMSG("Ignoring proof split out of range"));
               goto FAIL;
            }
            /* request the bigger picture (OP_GET_TFILE)... */
            pfine(FnMSG("Chain split preceeding proof, request Tfile..."));
            return mcmd__create_request(np->ip, OP_GET_TFILE, NULL);
         } else if (weigh_tfile("tfile.dat", bt.bnum, weight) != VEOK) {
            goto_perrno(FATAL, FnMSG("weigh_tfile(bt.bnum) FAILURE"));
         }
         partial = 1;
         /* jump to next section */
         goto TFILE;
      }  /* end if (np->opreq == OP_TF) */
      if (np->opreq == OP_GET_TFILE) {
         partial = 0;
TFILE:   /* validate partial Tfile in file pointer */
         pdebug(FnMSG("validating Tfile data..."));
         on_ecode_goto_perrno(
            validate_tfile_fp(np->fp, bnum, weight, partial),
            DROP, FnMSG("(partial) Tfile validation FAILURE"));
         /* check Tfile bnum/weight against Quorum (or self) */
         if (cmp64(bnum, quorum.bnum) < 0) {
            pdebug(FnMSG("Tfile 0x%s"), bnum2hex(bnum, bnumstr));
            pdebug(FnMSG("Quorum 0x%s"), bnum2hex(quorum.bnum, bnumstr));
            goto_perr(DROP, FnMSG("tfile bnum less than advertised"));
         } else if (cmp256(weight, quorum.weight) < 0) {
            pdebug(FnMSG("Tfile 0x%s"), weight2hex(weight, hexstr));
            pdebug(FnMSG("Quorum 0x%s"), weight2hex(quorum.weight, hexstr));
            goto_perr(DROP, FnMSG("tfile weight less than advertised"));
         }
         /* rewind file pointer and create threads for PoW validation */
         thrdp = malloc(cpu_cores() * sizeof(*thrdp));
         if (thrdp == NULL) perrno(FnMSG("malloc(threads) FAILURE"));
         for (rewind(np->fp), count = cpu_cores(), i = 0; i < count; i++) {
            if (thread_create(&thrdp[i], dist__pow_val, np) != 0) {
               perrno(FnMSG("thread_create() FAILURE"));
               pwarn("PoW validation may be slower than usual");
               count = i;
               break;
            }
         }
         /* if no threads were created, provide fallback method */
         if (count == 0) {
            pwarn("Single thread Trailer PoW validation (SLOW)");
            dist__pow_val(np);
         }
         /* wait for threads to finish */
         for (i = 0; i < count; i++) {
            if (thread_join(thrdp[i]) != 0) {
               perrno(FnMSG("thread_join(%d) FAILURE"), i);
            }
         }
         /* cleanup thread malloc */
         if (thrdp) free(thrdp);
         /* check PoW is VEOK and RESYNC */
         if (np->status == VEOK) {
            if (mcmd__resync(np) != VEOK) goto DROP;
         } else goto_perr(DROP, FnMSG("dist__pow_val() FAILURE"));
      }
      /****************************/
      /* BLOCK REQUEST PROCESSING */
      if (np->opreq == OP_GET_BLOCK) {
         bnum2hex(np->io, hexstr);
         /* ensure received block number is as requested */
         pfine(FnMSG("validating 0x%s..."), hexstr);
         if (fseek64(np->fp, -(sizeof(bt)), SEEK_END) != 0) {
            goto_perrno(DROP, FnMSG("fseek() FAILURE"));
         } else if (fread(&bt, sizeof(bt), 1, np->fp) != 1) {
            if (feof(np->fp)) set_errno(EMCM_EOF);
            goto_perrno(DROP, FnMSG("fread() FAILURE"));
         } else if (cmp64(np->io, bt.bnum)) {
            goto_perr(DROP, FnMSG("Block number mismatch"));
         }
         /* perform validation of block data */
         on_ecode_goto_perrno( validate_block_fp(np->fp, "tfile.dat"),
            DROP, FnMSG("validate_block() FAILURE"));
         /* save block data to file */
         on_ecode_goto_perrno( fsave(np->fp, "block.dat"),
            FATAL, FnMSG("fsave(tf) FAILURE"));
         /* close temporary file */
         fclose(np->fp);
         np->fp = NULL;
         /* update validated blockchain file */
         pfine(FnMSG("processing 0x%s..."), hexstr);
         on_ecode_goto_perrno( update_block("block.dat"),
            DROP, FnMSG("update_block(block.dat) FAILURE"));
         /* archive blockchain file */
         on_ecode_goto_perrno( archive_block("block.dat"),
            FATAL, FnMSG("archive_block() FAILURE"));
         /* trigger send_found update */
         if (!Ininit) mcmd__send_found(NULL);
         /* print block information */
         ptrailer(&bt);
         /* get next block number */
         add64(Cblocknum, one, bnum);
         return mcmd__create_request(np->ip, OP_GET_BLOCK, bnum);
      }  /* end if (np->opreq == OP_GET_BLOCK) */
   } else if (np->opreq) {
DROP: np->status = ecode;
      if (Ininit) {
         if (cmp64(Cblocknum, quorum.bnum) >= 0) {
            plog("\nVeronica says, \"You're done!\"\n");
            /* trigger OP_FOUND broadcast -- cleanup */
            mcmd__send_found(NULL);
            quorum_cleanup(&quorum);
            quorum_cleanup(&scan);
            Ininit = 0;
         } else {
            plog("Dropping %s...", np->id);
            remove32(np->ip, quorum.list, quorum.len, &(quorum.idx));
            remove32(np->ip, scan.list, scan.len, &(scan.idx));
         }
      }
   }

   /* hmmm... */
   return VERROR;

FATAL:
   Running = 0;
FAIL:
   return VERROR;
}  /* end mcmd__syncup() */

/**
 * @brief Validate and process incoming transactions
 * @details Reference Chart: https://app.code2flow.com/HaB1Ut
 * @param np Pointer to NODE containing incoming transaction
 * @return (int) value representing the operation result
 */
int mcmd__transaction(NODE *np)
{
   static word32 one[2] = { 1 };

   TXW txw;
   FILE *fp;
   char *txfile;
   word32 bnum[2];

#undef FnMSG
#define FnMSG(x) "mcmd__transaction(%s): " x, np->id

   /* check status of receive */
   if (np && np->status == VEOK) {
      txfile = NULL;
      /* copy transaction to local buffer */
      memcpy(&txw, np->pkt.buffer, sizeof(txw));
      /* validate transaction */
      np->status = txw_val(&txw, Myfee);
      /* VEBAD or VEBAD2; discard (always) invalid transactions */
      if (np->status == VEOK) {
         pfine(FnMSG("is valid -> send to txclean.dat"));
         txfile = "txclean.dat";
      } else if (np->status == VERROR) {
         pfine(FnMSG("is invalid..."));
         /* transaction is invalid, discard unless behind one block */
         add64(Cblocknum, one, bnum);
         if (cmp64(np->pkt.cblock, bnum) == 0) {
            /* build OP_TF bnum parameter */
            bnum[1] = (bnum[0] < 54) ? bnum[0] + 1 : 54;
            bnum[0] -= (bnum[0] < 54) ? bnum[0] : (54 - 1);
            /* request OP_FOUND-like request with OP_TF */
            pfine(FnMSG("triggered OP_TF proof request..."));
            mcmd__create_request(np->ip, OP_TF, bnum);
            /* hold transaction for update */
            pfine(FnMSG("holding transaction..."));
            txfile = "txhold.dat";
         }
      } else pfine(FnMSG("is invalid!"));
      /* store transaction in appropriate file, or discard */
      if (txfile) {
         fp = fopen(txfile, "ab");
         if (fp == NULL) return perrno(FnMSG("fopen(%s) FAILURE"), txfile);
         sha256(&txw, sizeof(txw) - HASHLEN, txw.tx_id);
         if (fwrite(&txw, sizeof(txw), 1, fp) != 1) {
            perrno(FnMSG("fwrite(%s) FAILURE"), txfile);
         }
         fclose(fp);
      }
   }  /* end if (np && np->status == VEOK) */

   return np->status;
}  /* end mcmd__transaction() */

static void mcmd__worker_process(LinkedNode *lnp)
{
   NODE *np;

   /* dereference and process NODE data */
   np = (NODE *) lnp->data;
   switch (np->opreq) {
      /* process "request" operations */
      case OP_FOUND: mcmd__send_found(np); break;
      case OP_GET_BLOCK: mcmd__syncup(np); break;
      case OP_GET_IPL: mcmd__syncup(np); break;
      case OP_GET_TFILE: mcmd__syncup(np); break;
      case OP_TF: mcmd__syncup(np); break;
      default: switch (np->opcode) {
         /* process "receive" (incoming) operations */
         default: pdebug(FnMSG("Unhandled opcode= %u"), np->opcode); break;
         case OP_TX: mcmd__transaction(np); break;
         case OP_FOUND: mcmd__syncup(np); break;
      }
   }
   /* check naughty peers -- pinklist */
   if (np->status == VEBAD2 || np->status == VEBAD) {
      if (np->status == VEBAD2) epinklist(np->ip);
      pinklist(np->ip);
   }
   /* cleanup resources */
   mcmd__cleanup_node(lnp);
}  /* end mcmd__worker_process() */

static void mcmd__worker_sync(LinkedNode *lnp)
{
   static Mutex syncupLock = MUTEX_INITIALIZER;
   static WorkList syncupIO = { 0 };

   LinkedNode *next_lnp;
   int ecode;

#undef FnMSG
#define FnMSG(x)  "mcmd__worker_sync(%x): " x, thread_selfid()

   /* place syncup work in syncupIO list */
   lock_on_ecode_goto_perrno( syncupIO.lock, FATAL, {
      on_ecode_goto_perrno( link_node_append(lnp, &(syncupIO.list)),
         SYNCUPIO_LOCKED, FnMSG("link_node_append() FAILURE"));
      /* try acquire syncupLock */
      trylock_on_ecode_goto_perrno( syncupLock, SYNCUPIO_LOCKED, {
         thread_setname(thread_self(), "mcmd-sync");
         /* check next syncupIO */
         for (lnp = syncupIO.list.next; Running && lnp; lnp = next_lnp) {
            next_lnp = lnp->next;
            /* skip sync tasks while waiting for Syncwait */
            if (Syncwait && Syncwait != lnp->data) continue;
            /* reset Syncwait */
            Syncwait = NULL;
            /* remove syncupIO node -- release lock */
            on_ecode_goto_perrno(
               link_node_remove(lnp, &(syncupIO.list)),
               FATAL, FnMSG("LIST FAILURE"));
            on_ecode_goto_perrno(
               mutex_unlock(&(syncupIO.lock)),
               FATAL, FnMSG("UNLOCK FAILURE"))
            /* process data -- cleanup occurs in function */
            mcmd__worker_process(lnp);
            /* (re)acquire SyncupIO Lock */
            on_ecode_goto_perrno(
               mutex_lock(&(syncupIO.lock)),
               FATAL, FnMSG("LOCK FAILURE"));
            /* restart list processing */
            next_lnp = syncupIO.list.next;
         }  /* while (Running && ... */
         thread_setname(thread_self(), "mcmd-async");
      });  /* end trylock_on_ecode... */
   });  /* end lock_on_ecode...*/

SYNCUPIO_LOCKED:
   /* try release locked mutex */
   on_ecode_goto_perrno(
      mutex_unlock(&(syncupIO.lock)), FATAL, FnMSG("UNLOCK FAILURE"));

   return;

FATAL:
   Running = 0;
}  /* end mcmd__worker_sync() */

/**
 * @private
 */
static ThreadProc mcmd__worker(void *arg)
{
   WorkList *wlp = (WorkList *) arg;
   Condition *alarmp = &(wlp->alarm);
   LinkedList *listp = &(wlp->list);
   Mutex *lockp = &(wlp->lock);

   LinkedNode *lnp;
   NODE *np;
   int ecode;

#undef FnMSG
#define FnMSG(x)  "mcmd__worker(%x): " x, thread_selfid()
   thread_setname(thread_self(), "mcmd-async");
   pdebug(FnMSG("created..."));

   /* acquire syncIO Lock */
   on_ecode_goto_perrno(
      mutex_lock(lockp), FATAL, FnMSG("(init)LOCK FAILURE"));

   /* main thread loop */
   while (Running) {
      /* check/pull next work */
      while ((lnp = listp->next)) {
         /* remove work node -- release worklock */
         on_ecode_goto_perrno(
            link_node_remove(lnp, listp), FATAL, FnMSG("LIST FAILURE"));
         on_ecode_goto_perrno(
            mutex_unlock(lockp), FATAL, FnMSG("UNLOCK FAILURE"));
         /* determine if work requires synchronous processing */
         np = (NODE *) lnp->data;
         switch (np->opreq) {
            /* transfer work to sync processing for following opreq */
            case OP_FOUND:  /* fallthrough -- requires sync */
            case OP_GET_BLOCK:  /* fallthrough -- requires sync */
            case OP_GET_IPL:  /* fallthrough -- requires sync */
            case OP_GET_TFILE:  /* fallthrough -- requires sync */
            case OP_TF: mcmd__worker_sync(lnp); break;
            default: {
               /* OP_FOUND broadcasts also require sync processing */
               /* ... otherwise, process "incoming" asynchronously */
               if (np->opcode == OP_FOUND) {
                  mcmd__worker_sync(lnp);
               } else mcmd__worker_process(lnp);
            }
         }  /* end switch (op->req) */
         /* (re)acquire worklock */
         on_ecode_goto_perrno(
            mutex_lock(lockp), FATAL, FnMSG("(re)LOCK FAILURE"));
         /* check SHUTDOWN for jump */
         if (!Running) goto SHUTDOWN;
      }  /* end while ((lnp = listp->next)) */

      /* perform additional checks */

      /* wait for work, sleepy time capped at 1 second intervals... */
      if (condition_timedwait(alarmp, lockp, 1000)) {
         /* ... wakeup (spurious?), check errno ... */
         if (errno != CONDITION_TIMEOUT) {
            perrno(FnMSG("CONDITION FAILURE"));
            goto FATAL;
         }
      }
   }  /* end while (Running) */

SHUTDOWN:
   pdebug(FnMSG("recv'd shutdown signal"));

   /* release worklock */
   on_ecode_goto_perrno(
      mutex_unlock(lockp), FATAL, FnMSG("(exit) UNLOCK FAILURE"));

   Unthread;

FATAL:
   pdebug(FnMSG("FATAL ERROR, TERMINATING..."));
   /* kill on fatal error */
   Running = 0;
   Unthread;
}  /* end mcmd__worker() */

/* transfer a NODE out of AsyncWork into a WorkList */
static int mcmd_asyncwork_transfer(AsyncWork *wp)
{
   LinkedNode *lnp;
   NODE *np;
   int ecode;

#undef FnMSG
#define FnMSG(x) "mcmd_asyncwork_transfer(): " x

   if (wp->data == NULL) return VERROR;
   np = (NODE *) wp->data;
   /* add NODE pointer to an empty LinkedNode */
   lnp = link_node_create(0);
   if (lnp) lnp->data = np;
   else return perrno(FnMSG("link_node_create() FAILURE"));
   /* add reference to LinkedNode and append to SyncIO (guarded) */
   lock_on_ecode_goto_perrno( CompleteIO.lock, FAIL, {
      on_ecode_goto_perrno( link_node_append(lnp, &(CompleteIO.list)),
         FAIL_LOCKED, FnMSG("link_node_append() FAILURE"));
      condition_signal(&(CompleteIO.alarm));
   });

   /* remove NODE reference and return VEOK */
   wp->data = NULL;
   return VEOK;

FAIL_LOCKED: mutex_unlock(&(CompleteIO.lock));
FAIL: free(lnp);
   return VERROR;
}  /* end mcmd_asyncwork_transfer() */

/* update AsyncWork with available NODE data */
static void mcmd_asyncwork_update(AsyncWork *wp)
{
   NODE *np;

   if (wp->data == NULL) return;
   np = (NODE *) wp->data;
   /* defer work involving Disk IO */
   wp->defer = np->fp ? 1 : 0;
   /* update AsyncWork state */
   wp->sio = np->iowait;
   wp->sd = np->sd;
   wp->to = np->to;
}  /* end mcmd_asyncwork_update() */

/* deallocate and dereference NODE data from completed AsyncWork */
static int mcmd_iodone(AsyncWork *wp)
{
   /* perform NODE specific cleanup and update AsyncWork */
   if (wp->data) node_cleanup(wp->data);
   mcmd_asyncwork_update(wp);
   return VEOK;
}  /* end mcmd_iodone() */

/* initialize AsyncWork with NODE data for the MCM Server Daemon */
static int mcmd_ioinit(AsyncWork *wp)
{
   NODE *np;
   word32 ip;
   char ipstr[16];

#undef FnMSG
#define FnMSG(x) "mcmd_ioinit(%d, %s)" x, wp->sd, ntoa(&ip, ipstr)

   /* get socket ip and perform initialization checks */
   ip = get_sock_ip(wp->sd);
   if (pinklisted(ip)) return perr(FnMSG("pinklisted"));

   /* create NODE for io handling and set socket non-blocking */
   wp->data = malloc(sizeof(NODE));
   if (wp->data == NULL) return perrno(FnMSG("malloc(NODE) FAILURE"));
   if (sock_set_nonblock(wp->sd)) return perr(FnMSG("non-block FAILURE"));
   /* initialize NODE for receiving */
   np = (NODE *) wp->data;
   node_init(np, wp->sd, ip, 0, OP_NULL, NULL);
   /* update timeout */
   wp->to = np->to;

   return VEOK;
}  /* end mcmd_ioinit() */

/* process AsyncWork for the MCM Server Daemon */
static int mcmd_ioproc(AsyncWork *wp)
{
   NODE *np = (NODE *) wp->data;

#undef FnMSG
#define FnMSG(x) "mcmd_ioproc(%d, %s): " x, wp->sd, np ? np->id : "(null)"

   /* check for NODE reference is valid */
   if (np == NULL) return perr(FnMSG("invalid NODE reference"));

   /* execute asynchronous communication protocols if NOT DONE */
   if (np->iowait != IO_DONE) {
      if (np->opreq) node_request_operation(np);
      else node_receive_operation(np);
   }

   /* update AsyncWork state */
   wp->sio = np->iowait;
   wp->sd = np->sd;
   wp->to = np->to;
   /* defer disk IO work */
   wp->defer = np->fp ? 1 : 0;

   /* transfer completed work, or return NODE status */
   if (np->iowait == IO_DONE) {
      return mcmd_asyncwork_transfer(wp);
   } else return np->status;
}  /* end mcmd_ioproc() */

void signal_handler(int sig)
{
   print("\n");
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
}

void redirect_signals(void)
{
   signal(SIGABRT, signal_handler);
	signal(SIGFPE,  signal_handler);
	signal(SIGILL,  signal_handler);
	signal(SIGINT,  signal_handler);
	signal(SIGSEGV, signal_handler);
	signal(SIGTERM, signal_handler);
}

int check_directory(const char *dir) {
   char permchk[FILENAME_MAX];

   /* build permission check file path */
   snprintf(permchk, FILENAME_MAX, "%s%sperm.chk",
      dir ? dir : "", dir ? PATH_SEPARATOR : "");
   /* touch the file and remove it, checking for failures */
   if (ftouch(permchk) || remove(permchk)) {
      return perrno(errno, "%s%spermission FAILURE",
         dir ? dir : "", dir ? PATH_SEPARATOR " " : "");
   }

   return VEOK;
}

int veronica(void)
{
   char haiku[256], buff[16];
   print("\n%s\n\n", trigg_expand(trigg_generate(buff), haiku));
   return VEOK;
}

int init(void)
{
   /* word8 highblock[8]; */
   word8 genbnum[8] = { 0 };
   word8 genbhash[4] = { 0x00, 0x17, 0x0c, 0x67 };
   char fname[FILENAME_MAX], fpath[FILENAME_MAX];
   int nochaindata;

#undef FnMSG
#define FnMSG(x) "init()" x

   /* read stored peers from disk */
   read_ipl(Epinkip_opt, Epinklist, EPINKLEN, &Epinkidx);
   read_ipl(Localip_opt, Lplist, LPLISTLEN, &Lplistidx);
   if (read_ipl(Recentip_opt, Rplist, RPLISTLEN, &Rplistidx) < 1) {
      /* if no recent peers, try (downloading) start peers */
      remove(Startip_opt);
      // http_get(Starthttp_opt, Startip_opt, 3);
      if (read_ipl(Startip_opt, Rplist, RPLISTLEN, &Rplistidx) < 1) {
         cwd(fpath, sizeof(fpath));
         path_join(fname, fpath, Startip_opt);
         pwarn("No start peers. Consider refreshing %s", fname);
         /* if no start peers, try core peers */
         if (read_ipl(Coreip_opt, Rplist, RPLISTLEN, &Rplistidx) < 1) {
            pwarn("Failed to load core peers. Check installation.");
         }
      }
   }

   /* initialize genesis block filename */
   bc_fqan(fname, genbnum, genbhash);
   path_join(fpath, Bcdir_opt, fname);

   /* derive Ledger filename */
   snprintf(fname, FILENAME_MAX, "%s.0", Lefname_opt);

   /* determine appropriate chain data exists */
   nochaindata = !fexists("tfile.dat");
   nochaindata |= !fexists(fpath);
   nochaindata |= !fexists(fname);

   /* (try) restore core chain files if any do not exist */
   if (nochaindata) {
      pdebug("Core chain files missing, attempting restoration...");
      if (!fexists("genblock.bc")) {
         return perr("Genesis block missing, cannot restore files!");
      } else if (fcopy("genblock.bc", fpath) != VEOK) {
         return perr("Failed to restore %s from Genesis block", fpath);
      }
      remove("tfile.dat");
      if (append_tfile(fpath, "tfile.dat") != VEOK) {
         return perr("Failed to restore Tfile from Genesis block");
      } else if (le_extract("genblock.bc") != VEOK) {
         return perr("Failed to restore Ledger from Genesis block");
      } else pdebug("Restoration of core chain files successful!");
   }

   return VEOK;
}  /* end init() */

int usage(void)
{
   print(
      "\nUSAGE: mcmd [OPTIONS]... [DIRECTORY]"

      "\n\nDIRECTORY:"
      "\n   Defaults to \"d/\""

      "\n\nOPTIONS:"
      "\n   --                         Forces the end of OPTIONS arguments"
      "\n   --dir-bc=<dir>             Set block archive directory to <dir>"
      "\n   --dir-sp=<dir>             Set chain split directory to <dir>"
      "\n   -h, --help                 Print this usage information"
      "\n   --no-pinklist              Disable the pinklist of evil peers"
      "\n   --no-pushblock             Disable block push capability"
      "\n   -p, --port=<num>"
      "\n      Set server port number to <num>. Valid range is (1-65535)."
      "\n      The operating system may impose additional restrictions..."
      "\n   --private-peers            Allow private peers in peer lists"
      "\n   -q, --quorum=<num>         Set network quorum size to <num>"
      "\n   --version                  Print the current software version"

      "\n\nOPTIONS (logging):"
      "\n   Logging levels (0-5) represent the logging level of detail."
      "\n   A log level includes the logs of all levels below it."
      "\n      5 = debug logs"
      "\n      4 = fine logs"
      "\n      3 = general logs"
      "\n      2 = warnings"
      "\n      1 = errors"
      "\n      0 = none"
      "\n   -ll, --log-level=<num>     Set screen log level to <num>"
      "\n   -o, --output-file=<file>   Set output log file to <file>"
      "\n   -ol, --output-level=<num>  Set output log level to <num>"

      "\n\nOPTIONS (peerlist):"
      "\n   -cp, --core-plist=<file>   Set core fallback list to <file>"
      "\n   -ep, --epink-plist=<file>  Set epoch pinklist to <file>"
      "\n   -lp, --local-plist=<file>  Set local peer list to <file>"
      "\n   -rp, --recent-plist=<file> Set recent peer list to <file>"
      "\n   -sp, --start-plist=<file>  Set start peer list to <file>"
      "\n   -sw, --start-web=<http>    Download start peers from <http>"

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
   static int j, eoa, ecode;
   static int one = 1;
   static int int_opt;                    /* integer for cli options */
   static unsigned uint_opt;              /* unsigned for cli options */
   static char *char_opt, *char_opt2;     /* char for cli options */
   static char *proc_name, *working_dir;

   Thread syncthread;
   Server *nsp = &NodeServer;             /* Node Server pointer */
   //int node_io_threads;

   redirect_signals();
   /* use multiple sources of entropy for improved prng */
   srand((unsigned int) time(NULL));
   srand16((word32) time(NULL), (word32) rand(), (word32) getpid());
   srand16fast((word32) time(NULL) ^ rand() ^ getpid());
   /* init process defaults */
   set_print_level(PLEVEL_LOG);
   Noprivate_opt = 1;
   Cbits |= C_PUSH;
   /* init server defaults */
   //node_io_threads = cpu_cores();
   sock_startup();

   /* derive process name, check for duplicates */
   proc_name = strrchr(argv[0], '/');
   if (proc_name == NULL) proc_name = strrchr(argv[0], '\\');
   if (proc_name) proc_name++; else proc_name = argv[0];
   if (proc_dups(proc_name)) return perr("Process is already running!");

   /* parse command line arguments */
   for (j = 1, eoa = 0, ecode = VEOK; Running && j < argc; j++) {
      if (argv[j][0] == '-') {
         /***********/
         /* OPTIONS */
         if (eoa || argument(argv[j], NULL, "--")) {
            /* flag to skip remaining options with leading '-' */
            if (eoa++ == 0) pfine("... end of arguments");
         }
         else if (argument(argv[j], NULL, "--dir-bc")) {
            /* obtain blockchain archive directory */
            char_opt = argvalue(&j, argc, argv);
            if (char_opt == NULL) goto_perr(USAGE, "Missing bc directory");
            pfine("... blockchain archive directory = %s", char_opt);
            /* set blockchain archive directory */
            Bcdir_opt = char_opt;
         }
         else if (argument(argv[j], NULL, "--dir-sp")) {
            /* obtain blockchain split directory */
            char_opt = argvalue(&j, argc, argv);
            if (char_opt == NULL) goto_perr(USAGE, "Missing sp directory");
            pfine("... blockchain split directory = %s", char_opt);
            /* set blockchain split directory */
            Spdir_opt = char_opt;
         }
         else if (argument(argv[j], "-h", "--help")) {
            /* print usage information and exit */
USAGE:      usage();
            goto EXIT;
         }
         else if (argument(argv[j], NULL, "--no-pinklist")) {
            pfine("... pinklist disabled");
            /* set "nopinklist" flag */
            Nopinklist_opt = 1;
         }
         else if (argument(argv[j], NULL, "--no-pushblock")) {
            pfine("... push blocks disabled");
            /* unset PUSH capability bit and set "nopush" flag */
            Cbits &= ~(C_PUSH);
            Nopush_opt = 1;
         }
         else if (argument(argv[j], "-p", "--port")) {
            /* obtain/check port number as integer */
            char_opt = argvalue(&j, argc, argv);
            if (char_opt == NULL) goto_perr(USAGE, "Missing port number");
            int_opt = atoi(char_opt);
            if (int_opt < 1 || int_opt > 65535)
               goto_perr(USAGE, "Invalid port number");
            pfine("... port = %s (%d)", char_opt, int_opt);
            /* set port number for receive and destination ports */
            Port_opt = Dstport_opt = (word16) int_opt;
         }
         else if (argument(argv[j], NULL, "--private-peers")) {
            pfine("... private peers enabled");
            /* set "noprivate" flag */
            Noprivate_opt = 0;
         }
         else if (argument(argv[j], "-q", "--quorum")) {
            /* obtain/check quorum number as integer */
            char_opt = argvalue(&j, argc, argv);
            if (char_opt == NULL) goto_perr(USAGE, "Missing quorum size");
            uint_opt = strtoul(char_opt, NULL, 0);
            if (uint_opt < 1) goto_perr(USAGE, "Invalid quorum size");
            pfine("... quorum = %s (%u)", char_opt, uint_opt);
            /* set quorum number */
            Quorum_opt = uint_opt;
         }
         else if (argument(argv[j], NULL, "--Veronica")) {
            veronica();
            goto EXIT;
         }
         else if (argument(argv[j], NULL, "--version")) {
            /* print GIT_VERSION information */
            print(GIT_VERSION);
            goto EXIT;
         }
         /*********************/
         /* LOG LEVEL OPTIONS */
         else if (argument(argv[j], "-ll", "--log-level")) {
            /* obtain/check log level as integer */
            char_opt = argvalue(&j, argc, argv);
            if (char_opt == NULL) goto_perr(USAGE, "Missing log level");
            int_opt = atoi(char_opt);
            if (int_opt < 0) goto_perr(USAGE, "Invalid log level");
            pfine("... log level = %s (%d)", char_opt, int_opt);
            /* set (printed) log level */
            set_print_level(int_opt);
         }
         else if (argument(argv[j], "-o", "--output-file")) {
            /* obtain output log filename, use LOGNAME if not specified */
            char_opt = argvalue(&j, argc, argv);
            if (char_opt == NULL) char_opt = LOGNAME;
            pfine("... output log file = %s", char_opt);
            /* set LOGGING capability bit and open output log file */
            Cbits |= C_LOGGING;
            set_output_file(char_opt, "a");
         }
         else if (argument(argv[j], "-ol", "--output-level")) {
            /* obtain/check (output) log level as integer */
            char_opt = argvalue(&j, argc, argv);
            if (char_opt == NULL) goto_perr(USAGE, "Missing output level");
            int_opt = atoi(char_opt);
            if (int_opt < 0) goto_perr(USAGE, "Invalid output level");
            pfine("... output level = %s (%d)", char_opt, int_opt);
            /* set (output) log level */
            set_output_level(int_opt);
         }
         /********************/
         /* PEERLIST OPTIONS */
         else if (argument(argv[j], "-cp", "--core-plist")) {
            /* obtain peerlist filename */
            Coreip_opt = argvalue(&j, argc, argv);
            if (Coreip_opt == NULL) goto_perr(USAGE, "Missing peerlist");
            pfine("... core peerlist = %s", Coreip_opt);
         }
         else if (argument(argv[j], "-ep", "--epink-plist")) {
            /* obtain peerlist filename */
            Epinkip_opt = argvalue(&j, argc, argv);
            if (Epinkip_opt == NULL) goto_perr(USAGE, "Missing peerlist");
            pfine("... epoch pinklist = %s", Epinkip_opt);
         }
         else if (argument(argv[j], "-lp", "--local-plist")) {
            /* obtain peerlist filename */
            Localip_opt = argvalue(&j, argc, argv);
            if (Localip_opt == NULL) goto_perr(USAGE, "Missing peerlist");
            pfine("... local peerlist = %s", Localip_opt);
         }
         else if (argument(argv[j], "-rp", "--recent-plist")) {
            /* obtain peerlist filename */
            Recentip_opt = argvalue(&j, argc, argv);
            if (Recentip_opt == NULL) goto_perr(USAGE, "Missing peerlist");
            pfine("... recent peerlist = %s", Recentip_opt);
         }
         else if (argument(argv[j], "-sp", "--start-plist")) {
            /* obtain peerlist filename */
            Startip_opt = argvalue(&j, argc, argv);
            if (Startip_opt == NULL) goto_perr(USAGE, "Missing peerlist");
            pfine("... start peerlist = %s", Startip_opt);
         }
         else if (argument(argv[j], "-sw", "--start-weblist")) {
            /* obtain peerlist address */
            Starthttp_opt = argvalue(&j, argc, argv);
            if (Starthttp_opt == NULL) goto_perr(USAGE, "Missing address");
            pfine("... start weblist = %s", Starthttp_opt);
         }
         /********************/
         /* ADVANCED OPTIONS */
         else if (argument(argv[j], "-mf", "--mining-fee")) {
            /* obtain/check mining fee as integer */
            char_opt = argvalue(&j, argc, argv);
            if (char_opt == NULL) goto_perr(USAGE, "Missing fee value");
            int_opt = atoi(char_opt);
            if (int_opt < MFEE) goto_perr(USAGE, "Invalid fee value");
            pfine("... mining fee (Myfee) = %s (%d)", char_opt, int_opt);
            /* set Myfee, and MFEE capability bit if non-standard */
            Myfee[0] = int_opt;
            if (cmp64(Myfee, Mfee)) Cbits |= C_MFEE;
         }
         else if (argument(argv[j], NULL, "--Sanctuary")) {
            /* obtain/check Sanctuary/Lastday as unsigned long */
            char_opt = argvalue(&j, argc, argv);
            if (char_opt == NULL) goto_perr(USAGE, "Missing protocol data");
            char_opt2 = strchr(char_opt, ',');
            if (char_opt2) *(char_opt2++) = '\0';  /* create separation */
            else goto_perr(USAGE, "Malformed protocol data");
            Sanctuary_opt = strtoul(char_opt, NULL, 0);
            Lastday_opt = (strtoul(char_opt2, NULL, 0) + 255) & 0xffffff00;
            pfine("... Sanctuary %s (%lu), %s (%lu)",
               char_opt, (unsigned long) Sanctuary_opt,
               char_opt2, (unsigned long) Lastday_opt);
         }
         else if (argument(argv[j], "-V", "--Virtual")) {
            /* obtain/check virtual mode as integer */
            char_opt = argvalue(&j, argc, argv);
            if (*char_opt == '2') { Dstport_opt = PORT1; Port_opt = PORT2; }
            else { char_opt = "1"; Dstport_opt = PORT2; Port_opt = PORT1; }
            pfine("... virtual mode = %c", *char_opt);
         }
         /*********************/
         /* DEVELOPER OPTIONS */
         else if (argument(argv[j], "-tb", "--trust-block")) {
            /* Set a trusted block number (for development use only).
             * Skips PoW validation up to specified block (inclusive). */
            /* obtain trust block as unsigned long */
            char_opt = argvalue(&j, argc, argv);
            if (char_opt == NULL) goto_perr(USAGE, "Missing block number");
            Trustblock_opt = strtoul(char_opt, NULL, 0);
            pfine("... trust block = %s (%lu)", char_opt,
               (unsigned long) Trustblock_opt);
         }
         /******************/
         /* UNKNOWN OPTION */
         else goto_perr(USAGE, "Unknown argument, %s", argv[j]);
      } else if (argv[j][0]) {
         /* additional non-option arguments */
         if (working_dir == NULL) {
            working_dir = argv[j];
            pfine("-- working directory = %s", working_dir);
         }
      }  /* end if arguments... */
   }  /* end for j */

   /* print splashscreen -- 2 seconds */
   psplash(EXEC_NAME, GIT_VERSION, 1);
   millisleep(2000);
   /* Running check */
   if (!Running) goto EXIT;

   /* print host info -- 1 second */
   phostinfo();
   millisleep(1000);
   /* Running check */
   if (!Running) goto EXIT;

   /* change working directory -- check location */
   if (working_dir == NULL) working_dir = "d";
   if (cd(working_dir) != 0) {
      perrno(errno, "Cannot change DIRECTORY to \"%s\"", working_dir);
      plog("Working directory unavailable. Check installation.");
      goto EXIT;
   } else if (fexists(proc_name)) {
      perr("Found executing binary '%s' in working directory", proc_name);
      plog("Cowardly refusing to work in specified directory");
      goto EXIT;
   }

   /* check directory structure and permissions */
   if (check_directory("") != VEOK) goto EXIT;
   if (check_directory(Bcdir_opt) != VEOK) goto EXIT;

   /* intialize peers, chain files -- start server */
   on_ecode_goto_perrno( init(), EXIT, "Process initialization FAILURE");

   /* start Mochimo Node server */
   on_ecode_goto_perrno(
      server_init(nsp, AF_INET, SOCK_STREAM, 0),
      EXIT, "server_init(node, AF_INET) FAILURE");
   on_ecode_goto_perrno(
      server_setsockopt(nsp, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)),
      EXIT, "server_setsockopt(node, SOL_SOCKET, SO_REUSEADDR) FAILURE");
   on_ecode_goto_perrno(
      server_setioprocess(nsp, mcmd_iodone, mcmd_ioinit, mcmd_ioproc),
      EXIT, "server_setioprocess(node) FAILURE");
   on_ecode_goto_perrno(
      server_start(nsp, INADDR_ANY, Port_opt, 3 /*node_io_threads*/),
      EXIT, "server_start(node) FAILURE");

   plog("Node Server started on 0.0.0.0:%u...", Port_opt);

   /* start the Mochimo Server Daemon sync thread */
   on_ecode_goto_perrno(
      thread_create(&syncthread, mcmd__worker, NULL),
      SHUTDOWN, "Failed to start mcmd__worker() thread");

   /* BLOCK and wait for main processing thread to exit */
   thread_join(syncthread);

SHUTDOWN:
   /* shutdown server/s */
   server_shutdown(nsp);
   /* save dynamic peer lists */
   save_ipl(Recentip_opt, Rplist, RPLISTLEN);
   save_ipl(Epinkip_opt, Epinklist, EPINKLEN);

EXIT:
   /* shutdown active sockets */
   sock_cleanup();
   plog("");

   return ecode;
}  /* end main() */
