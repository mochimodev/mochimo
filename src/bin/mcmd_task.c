
#include "mcmd.h"

#ifndef POW_THREADS
   #define POW_THREADS  ( cpu_cores() )   /* system dependant */

#endif

ThreadProc thrd_validate_pow(void *arg)
{
   SNODE *snp = arg;

   Mutex lock = MUTEX_INITIALIZER;
   BTRAILER bt;
   int ecode;
   char bnumstr[17];

   /* set name of thread - visible in htop */
   thread_setname(thread_self(), "pow_validation");

#undef FnMSG
#define FnMSG(x) "thrd_validate_pow(): " x

   /* acquire lock before loop condition */
   if (mutex_lock(&lock) != 0) goto FAIL;

   while (Running && snp->status == VEOK) {
      if (fread(&bt, sizeof(bt), 1, snp->fp) != 1) {
         if (feof(snp->fp)) break;
         snp->status = VERROR;
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
         perrno(errno, FnMSG("Block Trailer 0x%s POW INVALID"), bnumstr);
         snp->status = ecode;
      }
   }

   /* release lock -- exit */
   mutex_unlock(&lock);
   Unthread;

FAIL:
   snp->status = VERROR;
   Unthread;
}

/**
 * @private
 * Validate Trailer data and perform multi-threaded PoW validation.
*/
static int server_async_tval(SNODE *snp, void *highbnum, void *highweight)
{
   Thread *thrdp;
   size_t count, i, j;

#undef FnMSG
#define FnMSG(x) "server_async_tval(): " x

   /* check standard Tfile data validation */
   snp->status = validate_tfile_fp(snp->fp, highbnum, highweight);
   if (snp->status) goto_perrno(FAIL, FnMSG("validate_tfile_fp() FAILURE"));

   /* rewind file pointer and create threads for PoW validation */
   thrdp = malloc(POW_THREADS * sizeof(*thrdp));
   for (rewind(snp->fp), count = POW_THREADS, i = 0; i < count; i++) {
      if (thread_create(&thrdp[i], thrd_validate_pow, snp) != 0) {
         perrno(errno, FnMSG("thread_create() FAILURE"));
         snp->status = VERROR;
         break;
      }
   }

   /* wait for threads to finish */
   for (j = 0; j < i; j++) {
      if (thread_join(thrdp[j]) != 0) {
         perrno(errno, FnMSG("thread_join() FAILURE"));
      }
   }

   /* cleanup */
   free(thrdp);

FAIL:
   return snp->status;
}

/**
 * @private
 * Validate block data stored by a SNODE.
 * Requires a valid Tfile ("tfile.dat") for all block types.
 * Requires a valid and current Ledger ("ledger.dat" and "tag.dat")
 * for non-NG block types.
 *
*/
static int server_async_bval(SNODE *snp)
{
   word8 bnum[8];
   char bnumstr[17];

#undef FnMSG
#define FnMSG(x) "server_async_bval(%s, 0x%s): " \
   x, snp->id, bnum2hex(snp->io, bnumstr)

   /* ensure received block number is as requested */
   if (fseek64(snp->fp, -(sizeof(BTRAILER) - HASHLEN), SEEK_END) != 0) {
      return snp->status = perrno(errno, FnMSG("fseek() FAILURE"));
   } else if (fread(bnum, 8, 1, snp->fp) != 1) {
      if (feof(snp->fp)) set_errno(EMCM_EOF);
      return snp->status = perrno(errno, FnMSG("fread() FAILURE"));
   } else if (cmp64(snp->io, bnum)) {
      return snp->status = perr(FnMSG("Block number mismatch."));
   }

   /* perform validation of block data */
   snp->status = validate_block_fp(snp->fp, "tfile.dat");
   if (snp->status) perrno(errno, FnMSG("FAILURE"));

   return snp->status;
}  /* end server_async_bval() */

int server_async(SNODE *snp)
{
#undef FnMSG
#define FnMSG(x) "server_async(%s):" x, snp->id

   /* execute async protocol communication */
   if (snp->iowait <= IO_RECV) {
      if (snp->opreq) node_request(snp);
      else node_receive(snp);
   }

   /* execute additional async routines (on protocol completion) */
   if (snp->iowait == IO_DONE && snp->status == VEOK) {
      switch (snp->opreq) {
         case REQ_VAL_BLOCK: server_async_bval(snp); break;
         case REQ_VAL_TFILE: server_async_tval(snp, NULL, NULL); break;
      }
   }

   return snp->status;
}  /* end server_async() */

int server_init(SNODE *snp)
{
   static word32 *qlist, qlistidx, qlistlen; /* dynamic quorum list */
   static word32 *slist, slistidx, slistlen; /* dynamic scan list */
   static word32 eon[2] = { 0x100, 0 };
   static word32 one[2] = { 1, 0 };
   static word8 qbnum[8], qhash[32], qweight[32];
   static word8 tfbnum[8], tfweight[32];
   static int first_scan = 1;
   static int network_found;
   static int requests;

   BTRAILER bt;
   LinkedNode *lnp;
   long long fpos;
   word32 *plist, plistlen, i;
   char bnumstr[17], weightstr[65];
   char fname[FILENAME_MAX], fpath[FILENAME_MAX];
   word8 bnum[8];
   int ecode;

#undef FnMSG
#define FnMSG(x) "server_init(): " x

   /************************************************/
   /* BLOCKCHAIN PROCESSING, VALIDATION AND UPDATE */

   /* check request for block download */
   if (snp && snp->opreq == REQ_VAL_BLOCK) {
      requests = 0;
      if (snp->status == VEOK) {
         pfine(FnMSG("updating 0x%s..."), bnum2hex(snp->io, bnumstr));
         /* save block data to file */
         on_ecode_goto_perrno( fsave(snp->fp, "block.dat"),
            FATAL, FnMSG("fsave(tf) FAILURE"));
         fclose(snp->fp);
         snp->fp = NULL;
         /* update with validated block file */
         on_ecode_goto_perrno( snp->status = update_block("block.dat"),
            INVALID, FnMSG("update_block(block.dat) FAILURE"));
         /* archive blockchain file */
         on_ecode_goto_perrno( read_trailer(&bt, "block.dat"),
            FATAL, FnMSG("read_trailer(block.dat) FAILURE"));
         bc_fqan(fname, bt.bnum, bt.bhash);
         path_join(fpath, Bcdir_opt, fname);
         remove(fpath);
         on_ecode_goto_perrno( rename("block.dat", fpath),
            FATAL, FnMSG("rename(%s, %s) FAILURE"), fname, fpath);
         /* increment bnum -- skip NG (NOT YET) */
         put64(bnum, snp->io);
         add64(bnum, one, bnum);
         /* if (bnum[0] == 0) add64(bnum, one, bnum); */
         /* repurpose SNODE to download next block */
         node_cleanup(snp);
         prep_request(snp, snp->ip, snp->port, REQ_VAL_BLOCK, bnum);
         pfine(FnMSG("requesting 0x%s..."), bnum2hex(bnum, bnumstr));
         return VEWAITING;  /* NOTE: SNODE is repurposed */
      } else if (cmp64(snp->io, tfbnum) > 0) {
         plog("Blockchain download completed!");
         return VEOK;
      }
   }

   /****************************************/
   /* TFILE DATA PROCESSING AND VALIDATION */

   /* check request for Tfile download */
   if (snp && snp->opreq == OP_GET_TFILE) {
      requests--;
      /* check result of download and save Tfile */
      if (snp->status == VEOK) {
         pfine(FnMSG("validating Tfile..."));
         /* validate Tfile data */
         on_ecode_goto_perrno( server_async_tval(snp, tfbnum, tfweight),
            INVALID, FnMSG("tf_val() FAILURE"));
         /* check advertised bnum/weight match Tfile */
         on_ecode_goto_perrno( snp->status = (cmp64(tfbnum, qbnum) < 0),
            INVALID, FnMSG("tfile bnum less than advertised"));
         on_ecode_goto_perrno( snp->status = (cmp256(tfweight, qweight) < 0),
            INVALID, FnMSG("tfile weight less than advertised"));
         /* save Tfile data to file */
         on_ecode_goto_perrno( fsave(snp->fp, "tfile.dat"),
            FATAL, FnMSG("fsave(tf) FAILURE"));
         /* derive previous neogenesis block number from Tfile bnum */
         if (sub64(tfbnum, eon, &fpos)) fpos = 0;
         else fpos = fpos & WORD64_C(0xffffffffffffff00);
         /* seek to neogen tfile trailer */
         fpos *= sizeof(bt);
         on_ecode_goto_perrno( fseek64(snp->fp, fpos, SEEK_SET),
            FATAL, FnMSG("fseek() FAILURE"));
         while (fread(&bt, sizeof(bt), 1, snp->fp)) {
            /* get name of archive block */
            put64(bnum, bt.bnum);
            bc_fqan(fname, bt.bnum, bt.bhash);
            path_join(fpath, Bcdir_opt, fname);
            if (!fexists(fpath)) break;
            pfine(FnMSG("recovering 0x%s..."), bnum2hex(bt.bnum, bnumstr));
            /* update chain with valid block file */
            on_ecode_goto_perrno( snp->status = update_block(fpath),
               INVALID, FnMSG("update_block(%s)"), fpath);
            add64(bt.bnum, one, bnum);
         }
         /* repurpose SNODE to initiate neo-genesis download */
         node_cleanup(snp);
         prep_request(snp, snp->ip, snp->port, REQ_VAL_BLOCK, bnum);
         pfine(FnMSG("requesting 0x%s..."), bnum2hex(bnum, bnumstr));
         return VEWAITING;  /* NOTE: SNODE is repurposed */
      }
   }

INVALID:

   /****************************************/
   /* INITIALIZATION OR IP LIST PROCESSING */

   /* clear plist */
   plist = NULL;
   plistlen = 0;

   /* determine initial or returning request, respectively */
   if (snp == NULL) {
      ServerInit = 1;
      /* prepare recent peers for scan if no list provided */
      if (slistidx == 0) {
         plog("Initializing network...");
         plistlen = RPLISTLEN;
         plist = Rplist;
      }
   } else if (snp->opreq == OP_GET_IPL) {
      requests--;
      /* check success of connection */
      if (snp->status == VEOK) {
         network_found = 1;
         /* check existing quorum to compare */
         if (qlistidx) {
            ecode = cmp256(qweight, snp->pkt.weight);
            if (ecode < 0) {
               /* reset quorum (for higher advertised chain) */
               while (qlistidx > 0) {
                  qlist[qlistidx - 1] = 0;
                  qlistidx--;
               }
            } else if (ecode == 0) {
               /* add peer to quorum on matching hash */
               if (memcmp(qhash, snp->pkt.cblockhash, 32) == 0) {
                  addpeer_d(snp->ip, &qlist, &qlistlen, &qlistidx);
               }
            }
         }  /* end if (qlistidx) */
         /* if no quorum (or quorum reset), compare our Weight */
         if (qlistidx == 0 && cmp256(snp->pkt.weight, Weight) > 0) {
            /* add peer to quorum, and set quorum data on success */
            if (addpeer_d(snp->ip, &qlist, &qlistlen, &qlistidx)) {
               memcpy(qhash, snp->pkt.cblockhash, 32);
               memcpy(qweight, snp->pkt.weight, 32);
               put64(qbnum, snp->pkt.cblock);
            }
         }
         /* process IP List on first scan */
         if (first_scan) {
            plist = (word32 *) PKTBUFF(&snp->pkt);
            plistlen = (word32) get16(snp->pkt.len) / sizeof(word32);
            /* limit list length to size of packet (just in case) */
            if (plistlen > PKTBUFFLEN) {
               plistlen = (word32) PKTBUFFLEN / sizeof(word32);
            }
         }
      }  /* end if (snp->status == VEOK) */
      /* log found Node */
      bnum2hex(snp->pkt.cblock, bnumstr);
      weight2hex(snp->pkt.weight, weightstr);
      pdebug(FnMSG("found %s 0x%s 0x%s"), snp->id, bnumstr, weightstr);
   }  /* end if (snp == NULL)... else if (snp->opreq == OP_GET_IPL)... */

   /* "recent" peers list requires read lock */
   if (plist == Rplist) {
      on_ecode_goto_perrno( rwlock_rdlock(&RplistLock),
         FATAL, FnMSG("RplistLock LOCK FAILURE"));
   }

   /* perform network scan of peerlist */
   for (i = 0; i < plistlen && plist[i]; i++) {
      if (addpeer_d(plist[i], &slist, &slistlen, &slistidx)) {
         /* create SNODE request in LinkedNode */
         lnp = server_task_request(plist[i], OP_GET_IPL, NULL);
         if (lnp && server_task_append(lnp, &ActiveIO) != VEOK) {
            perr(FnMSG("OP_GET_IPL REQUEST FAILURE"));
            server_task_cleanup(lnp);
            break;
         } else requests++;
      }  /* end if (addpeer_d... */
   }  /* end for (i = 0; i < plistlen... */

   /* "recent" peers list requires read unlock */
   if (plist == Rplist) {
      on_ecode_goto_perrno( rwlock_rdunlock(&RplistLock),
         FATAL, FnMSG("RplistLock UNLOCK FAILURE"));
   }

   /* wait for OP_GET_IPL requests to finish */
   if (requests) return VEOK;

   /* drop non-OP_GET_IPL peer failures from peer lists */
   if (snp && snp->opreq != OP_GET_IPL && snp->status != VEOK) {
      plog("Dropping %s...", snp->id);
      remove32(snp->ip, qlist, qlistlen, &qlistidx);
      remove32(snp->ip, slist, slistlen, &slistidx);
   }

   /* report highchain */
   plog("Quorum members: %d / %d", qlistidx, Quorum_opt);
   plog("Quorum chain: 0x%s / 0x%s",
      bnum2hex(qbnum, bnumstr), weight2hex(qweight, weightstr));

   /* check quorum requirements */
   if (qlistidx == 0) {
      plog("No higher chain available");
      if (first_scan && !network_found) {
         pwarn("NETWORK NOT FOUND!");
         plog("Please check network configuration...");
      }
   } else if (qlistidx < Quorum_opt) {
      /* remove quorum peers from scan list, reset quorum peers */
      while (qlistidx > 0) {
         remove32(qlist[qlistidx - 1], slist, slistlen, &slistidx);
         qlist[qlistidx - 1] = 0;
         qlistidx--;
      }
      plog("(re)Initializing network...");
      /* perform rescan on scanned peers list */
      for (first_scan = i = 0; i < slistlen && slist[i]; i++, requests++) {
         /* create SNODE request in LinkedNode */
         lnp = server_task_request(slist[i], OP_GET_IPL, NULL);
         if (lnp && server_task_append(lnp, &ActiveIO) != VEOK) {
            perr(FnMSG("OP_GET_IPL REQUEST FAILURE"));
            server_task_cleanup(lnp);
            break;
         }
      }  /* end for (i = 0; i < slistlen... */
   } else {
      /* shuffle quorum members??? or use first quorum member... */
      // shuffle32(qlist, qlistidx);
      requests++;
      /* repurpose SNODE to initiate Tfile download */
      node_cleanup(snp);
      prep_request(snp, *qlist, Dstport_opt, OP_GET_TFILE, bnum);
      plog("Synchronizing with %s...", snp->id);
      pfine(FnMSG("requesting Tfile..."));
      return VEWAITING;  /* NOTE: SNODE is repurposed */
   }

   return VEOK;

FATAL:
   ServerOk = 0;
   return VERROR;
}  /* end server_init() */

int server_sync(SNODE *snp)
{
   int ecode = VERROR;

#undef FnMSG
#define FnMSG(x) "server_sync(): " x

   /* sanity checks */
   if (snp == NULL) return VERROR;

   /* check server synchronization */
   if (ServerSyncup && snp) {
      switch (snp->opreq) {
         default: break;
      }
      /* check for completion */
      if (ServerSyncup == 0) {
         plog("\n");
         plog("Veronica says, \"You're done!\"");
         plog("\n");
      }
   }

   /* check server initialization */
   if (ServerInit) ecode = server_init(snp);

   /* process standard synchronous activities
   switch (snp->opreq) {
      case OP_GET_BLOCK: {
         snp->status = server_block_update(snp);
         break;
      }
   } */

   return ecode;
}  /* end server_sync() */
