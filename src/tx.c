/**
 * @private
 * @headerfile tx.h <tx.h>
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_TX_C
#define MOCHIMO_TX_C


#include "tx.h"

/* internal support */
#include "wots.h"
#include "trigg.h"
#include "tag.h"
#include "peach.h"
#include "ledger.h"
#include "global.h"
#include "error.h"

/* external support */
#include <sys/wait.h>
#include <string.h>
#include "exttime.h"
#include "extmath.h"
#include "extlib.h"
#include "crc16.h"

/**
 * @private Transaction ID reference structure.
 * Contains an ID and file position type pair.
*/
typedef struct {
   word8 id[HASHLEN];
   fpos_t pos;
} TXREF;

/**
 * @private
 * Comparison function to sort TXREF objects by id.
*/
static int compare_txref(const void *va, const void *vb)
{
   TXREF *a = (TXREF *) va;
   TXREF *b = (TXREF *) vb;

   return memcmp(a->id, b->id, sizeof(a->id));
}

#ifndef _WIN32

   /* Get exclusive lock on lockfile.
   * Returns: -1 if lock not made within 'seconds' or open() failed
   *          else a descriptor to be used with unlock()
   */
   int lock(char *lockfile, int seconds)
   {
      time_t timeout;
      int fd, status;

      timeout = time(NULL) + seconds;
      fd = open(lockfile, O_NONBLOCK | O_RDONLY);
      if(fd == -1) return -1;
      for(;;) {
         status = flock(fd, LOCK_EX | LOCK_NB);
         if(status == 0) return fd;
         if(time(NULL) >= timeout) {
            close(fd);
            return -1;
         }
      }
   }

   /* Unlock a decriptor returned from lock() */
   int unlock(int fd)
   {
      int status;

      status =  flock(fd, LOCK_UN);
      close(fd);
      return status;
   }

#endif

/**
 * Read a single Transaction and eXtended Data entry in to the provided
 * buffers, @a txe and @a xdata, from the given input @a stream. The
 * file position indicator is advanced by the size of the whole entry.
 * @param txe Pointer to Transaction Entry buffer
 * @param xdata Pointer to eXtended Data buffer, or NULL
 * @param stream The stream to read from
 * @return (int) value representing the read result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 */
int tx_fread(TXQENTRY *txe, XDATA *xdata, FILE *stream)
{
   size_t res, xlen;

   /* read up to transaction change address to check for MDST */
   res = fread(txe, XTX_HDRLEN, 1, stream);
   if (res != 1) return VERROR;

   /* determine if eXtended Data is present */
   if (IS_XTX(txe)) {
      switch (XTX_TYPE(txe)) {
         /* ... add eXtended Transaction data types here */
         case XTX_MDST:
            /* infer +1 MDST count due to byte limitations */
            xlen = ((size_t) XTX_COUNT(txe) + 1) * sizeof(MDST);
            break;
         default:
            /* no eXtended Data available */
            xlen = 0;
      }
      /* read eXtended Data or skip if xdata is NULL */
      if (xlen > 0) {
         if (xdata == NULL) {
            res = (size_t) fseek64(stream, (long long) xlen, SEEK_CUR);
            if (res != 0) return VERROR;
         } else {
            res = fread(xdata, xlen, 1, stream);
            if (res != 1) return VERROR;
         }
      }
   }

   /* read remaining transaction data (from chg_addr) */
   res = fread(txe->chg_addr, sizeof(TXQENTRY) - XTX_HDRLEN, 1, stream);
   if (res != 1) return VERROR;

   return VEOK;
}  /* end tx_fread() */

/**
 * Write a single Transaction and eXtended Data entry from the provided
 * buffers, @a txe and @a xdata, to the given input @a stream. The file
 * position indicator is advanced by the size of the whole entry. If the
 * provided Transaction entry does not indicate an eXtended Transaction
 * entry that contains eXtended Data, it MAY be left NULL.
 * @param txe Pointer to Transaction Entry buffer
 * @param xdata Pointer to eXtended Data buffer, or NULL
 * @param stream The stream to read from
 * @return (int) value representing the write result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 * @exception errno=ENODATA Write requires xdata, which was not provided
 */
int tx_fwrite(TXQENTRY *txe, XDATA *xdata, FILE *stream)
{
   size_t res, xlen;

   /* write transaction data up to possible eXtended Data */
   res = fwrite(txe, XTX_HDRLEN, 1, stream);
   if (res != 1) return VERROR;

   /* determine if eXtended Data is present */
   if (IS_XTX(txe)) {
      switch (XTX_TYPE(txe)) {
         /* ... add eXtended Transaction data types here */
         case XTX_MDST:
            /* infer +1 MDST count due to byte limitations */
            xlen = ((size_t) XTX_COUNT(txe) + 1) * sizeof(MDST);
            break;
         default:
            /* no eXtended Data available */
            xlen = 0;
      }
      /* read eXtended Data or skip if xdata is NULL */
      if (xlen > 0) {
         if (xdata == NULL) {
            set_errno(ENODATA);
            return VERROR;
         } else {
            res = fwrite(xdata, xlen, 1, stream);
            if (res != 1) return VERROR;
         }
      }
   }

   /* write remaining transaction data */
   res = fwrite(txe->chg_addr, XTX_TLRLEN, 1, stream);
   if (res != 1) return VERROR;

   return VEOK;
}  /* end tx_fwrite() */

/* Validates a multi-dst transaction MTX.
 * (Does all tag checking as well.)
 * tx->src_addr is already checked in ledger.dat and totals tally.
 * tx_val() sets fee parameter to Myfee and bval.c sets fee to Mfee.
 * Returns 0 on valid, else error code.
 */
int mtx_val(MTX *mtx, word32 *fee)
{
   int j;
   word8 total[8], mfees[8], *bp, *limit;
   static word8 addr[TXWOTSLEN];

   limit = &mtx->zeros[0];

   /* Check that src and chg have tags.
    * Check that src and chg have same tag.
    * tx_val() or bval.c has already checked src != chg, src exists,
    *   sig is good, and totals are good.
    */
   if(!ADDR_HAS_TAG(mtx->src_addr)) goto bail;
   if(memcmp(ADDR_TAG_PTR(mtx->src_addr),
             ADDR_TAG_PTR(mtx->chg_addr), TXTAGLEN) != 0) goto bail;
   if(cmp64(mtx->change_total, Mfee) <= 0) goto bail;

   memset(total, 0, 8);
   memset(mfees, 0, 8);
   /* Tally each dst[] amount and mfees... */
   for(j = 0; j < MDST_NUM_DST; j++) {
      /* zero dst[] tag marks end of list.  */
      if(iszero(mtx->dst[j].tag, TXTAGLEN)) {
         for(bp = mtx->dst[j].amount; bp < limit; bp++) {
            if(*bp) goto bail;  /* Check that rest of dst[] list is zeros. */
         }
         break;
      }
      if(iszero(mtx->dst[j].amount, 8)) goto bail;  /* bad send amount */
      /* no dst to src */
      if(memcmp(mtx->dst[j].tag,
                ADDR_TAG_PTR(mtx->src_addr), TXTAGLEN) == 0) goto bail;
      /* tally fees and send_total */
      if(add64(total, mtx->dst[j].amount, total)) goto bail;
      if(add64(mfees, fee, mfees)) goto bail;  /* Mfee or Myfee */
      if(get32(Cblocknum) >= MTXTRIGGER) {
         memcpy(ADDR_TAG_PTR(addr), mtx->dst[j].tag, TXTAGLEN);
         mtx->zeros[j] = 0;
         /* If dst[j] tag not found, put error code in zeros[] array. */
         if(tag_find(addr, NULL, NULL, TXTAGLEN) != VEOK) mtx->zeros[j] = 1;
      }
   }  /* end for j */
   /* Check tallies... */
   if(cmp64(total, mtx->send_total) != 0) goto bail;
   if(cmp64(mtx->tx_fee, mfees) < 0) goto bail;
   return VEOK;  /* valid */
bail:
   return VERROR;  /* bad */
}  /* end mtx_val() */

/* Validate a transaction against ledger
 *
 * Returns: 0 if vaild (accept)
 *          1 if server error (drop)
 *          2 or 3 if evil    (drop)
 */
int tx_val(TX *tx)
{
   int cond;
   static LENTRY src_le;            /* source ledger entry */
   word32 total[2];                 /* for 64-bit maths */
   static word8 message[HASHLEN];    /* transaction hash for WOTS */
   static word8 pk2[WOTSSIGBYTES];   /* more WOTS */
   static word8 rnd2[32];            /* for WOTS addr[] */
   MTX *mtx;
   static TX txs;

   if(memcmp(tx->src_addr, tx->chg_addr, TXWOTSLEN) == 0) {
      pdebug("src == chg");  /* also mtx */
      return 2;
   }

   if(!TX_IS_MTX(tx) && memcmp(tx->src_addr, tx->dst_addr, TXWOTSLEN) == 0) {
      pdebug("src == dst");
      return 2;
   }

   /* validate transaction fixed fee */
   if(cmp64(tx->tx_fee, Mfee) < 0) {
      pdebug("bad mining fee");
      return 2;
   }
   /* validate my fee */
   if(cmp64(tx->tx_fee, Myfee) < 0) {
      pdebug("fee < %u", Myfee[0]);
      return 1;
   }

   /* check WTOS signature */
   if(TX_IS_MTX(tx) && get32(Cblocknum) >= MTXTRIGGER) {
      memcpy(&txs, tx, sizeof(txs));
      mtx = (MTX *) TRANBUFF(&txs);  /* poor man's union */
      memset(mtx->zeros, 0, MDST_NUM_DZEROS);
      sha256(txs.src_addr, TXSIG_INLEN, message);
   } else {
      sha256(tx->src_addr, TXSIG_INLEN, message);
   }

   memcpy(rnd2, &tx->src_addr[TXSIGLEN+32], 32);  /* copy WOTS addr[] */
   wots_pk_from_sig(pk2, tx->tx_sig, message, &tx->src_addr[TXSIGLEN],
                    (word32 *) rnd2);
   if(memcmp(pk2, tx->src_addr, TXSIGLEN) != 0) {
      plog("WOTS signature failed!");
      return 3;
   }

   /* look up source address in ledger */
   if(le_find(tx->src_addr, &src_le, NULL, TXWOTSLEN) == FALSE) {
      pdebug("src_addr not in ledger");
      return 1;
   }
   total[0] = total[1] = 0;
   /* use add64() to check for overflow */
   cond =  add64(tx->send_total, tx->change_total, total);
   cond += add64(tx->tx_fee, total, total);
   if(cond) {
      plog("TX amount overflow");
      return 2;
   }
   if(cmp64(src_le.balance, total) != 0) {
      pdebug("bad transaction total != src_le.balance");
      return 1;
   }
   if(TX_IS_MTX(tx)) {
      mtx = (MTX *) TRANBUFF(tx);  /* poor man's union */
      if(mtx_val(mtx, Myfee)) return 1;  /* bad mtx */
   } else {
      if(tag_valid(tx->src_addr, tx->chg_addr, tx->dst_addr,
                   NULL) != VEOK) return 1;  /* bad tag */
   }
   return 0;  /* tx valid */
}  /* end tx_val() */

/* Search txq1.dat and txclean.dat for src_addr.
 * Return VEOK if the src_addr is not found, otherwise VERROR.
 */
int txcheck(word8 *src_addr)
{
   FILE *fp;
   TXQENTRY tx;

   fp = fopen("txq1.dat", "rb");
   if(fp != NULL) {
      for(;;) {
         if(fread(&tx, 1, sizeof(TXQENTRY), fp) != sizeof(TXQENTRY)) break;
         if(memcmp(tx.src_addr, src_addr, TXWOTSLEN) == 0) {
            fclose(fp);
            return VERROR;  /* found */
         }
      }  /* end for */
      fclose(fp);
   }  /* end if fp */

   fp = fopen("txclean.dat", "rb");
   if(fp != NULL) {
      for(;;) {
         if(fread(&tx, 1, sizeof(TXQENTRY), fp) != sizeof(TXQENTRY)) break;
         if(memcmp(tx.src_addr, src_addr, TXWOTSLEN) == 0) {
            fclose(fp);
            return VERROR;  /* found */
         }
      }  /* end for */
      fclose(fp);
   }  /* end if fp */
   return VEOK;  /* src_addr not found */
}  /* end txcheck() */

/**
 * Clean a Transaction Queue file, @a txfname, of entries that may have
 * been invalidated by an updated Ledger or may have been solved into a
 * recent Blockchain file, @a bcfname. If a Blockchain file is not 
 * provided, a Ledger-only clean will be performed.
 * @param txfname Filename of the clean transaction queue file
 * @param bcfname Filename of the block to clean against, or NULL
 * @return (int) value representing the clean result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 */
int txclean(const char *txfname, const char *bcfname)
{
   TXQENTRY txe, txc;      /* block entry and txclean transactions */
   XDATA xdata;            /* eXtended transaction data (for transfer) */
   FILE *fp, *bfp, *tfp;   /* input, blockchain and temporary files */
   TXREF *tx;              /* malloc'd transaction ID references */
   size_t count, actual;   /* malloc'd and actual element counts */
   size_t j, nout;
   fpos_t pos;             /* file position offset indicator */
   long long offset;       /* file position offset value */
   word32 hdrlen;          /* for block header length */
   int cond;
   char tmpfname[FILENAME_MAX];

   /* ensure ledger is open (required) */
   if (le_open("ledger.dat", "rb") != VEOK) {
      perr("failed to le_open(ledger.dat)");
      return VERROR;
   }

   /* GENERATE SORTED (ASCENDING) TXID REFERENCES FOR COMPARE */

   /* open provided txclean file */
   fp = fopen(txfname, "rb");
   if (fp == NULL) return VERROR;

   /* obtain EOF offset */
   if (fseek64(fp, 0LL, SEEK_END) != 0) goto FAIL_FP;
   offset = ftell64(fp);
   if (offset == EOF) goto FAIL_FP;
   if (offset == 0LL) {
      /* nothing to clean */
      fclose(fp);
      return VEOK;
   }
   if ((size_t) offset < sizeof(TXQENTRY)) {
      /* file contains unknown data */
      set_errno(EMCM_FILEDATA);
      goto FAIL_FP;
   }

   /* malloc required space (approximate) */
   count = (size_t) offset / sizeof(TXQENTRY);
   tx = malloc(count * sizeof(TXREF));
   if (tx == NULL) goto FAIL_FP;

   /* store txid and associated fpos_t value in arrays */
   for(rewind(fp), actual = 0; actual < count; actual++) {
      /* store position for later use (if tx is read) */
      if (fgetpos(fp, &pos) != 0) goto FAIL_FP_MEM;
      if (tx_fread(&txc, NULL, fp) != VEOK) {
         if (ferror(fp)) goto FAIL_FP_MEM;
         break;  /* EOF */
      }
      /* set txid reference data */
      memcpy(&(tx[actual].id), txc.tx_id, HASHLEN);
      tx[actual].pos = pos;
   }
   /* check for leftover data */
   if (ftell64(fp) < offset) {
      set_errno(EMCM_FILEDATA);
      goto FAIL_FP_MEM;
   }
   /* sort the txid reference array */
   qsort(tx, actual, sizeof(TXREF), compare_txref);

   /* PREPARE BLOCKCHAIN FILE FOR TRANSACTION COMPARISON (IF PROVIDED) */

   bfp == NULL;
   /* NOTE: bfp MUST BE initialized NULL for error handling */
   if (bcfname != NULL) {
      /* open validated block file */
      bfp = fopen(bcfname, "rb");
      if (bfp == NULL) goto FAIL_FP_MEM;
      /* read and check fixed length header */
      if (fread(&hdrlen, 4, 1, bfp) != 1) {
         if (!ferror(fp)) set_errno(EMCM_EOF);
         goto FAIL_FP_MEM_BFP;
      }
      if (hdrlen != sizeof(BHEADER)) {
         set_errno(EMCM_HDRLEN);
         goto FAIL_FP_MEM_BFP;
      }
      /* seek to start of Merkle Array */
      if (fseek(bfp, (long) hdrlen, SEEK_SET) != 0) {
         goto FAIL_FP_MEM_BFP;
      }
   }

   /* FILTER OLD QUEUE BASED ON BLOCK AND LEDGER */

   /* generate temporary filename */
   snprintf(tmpfname, sizeof(tmpfname), "txc-%04x.tmp", rand16());
   tfp = fopen(tmpfname, "wb");
   if (tfp == NULL) goto FAIL_FP_MEM_BFP;

   /* Remove TX_ID's from clean TX queue that are in the new block.
    * Remove remaining invalidated (per ledger) transactions.
    * Merkel Array in new block is sorted on TX_ID (checked in bval).
    * Clean queue, txclean.dat, is sorted by reference array above.
    *
    * NOTE: end of file check on Merkel Block (bfp) depends on block
    *       trailer being shorter than a TXQENTRY !
    *       STATIC_ASSERT() prevents this.
    */
   STATIC_ASSERT(sizeof(BTRAILER) >= sizeof(TXQENTRY), "LOOP EOF CHECK");
   for (cond = j = nout = 0; j < actual; j++) {
      if (bfp != NULL) {
         do {
            /* if block transaction ID compared AFTER reference, hold... */
            if (cond <= 0) {
               /* read next transaction from block */
               if (tx_fread(&txe, NULL, bfp) != VEOK) {
                  if (ferror(bfp)) goto FAIL_ALL;
                  /* EOF -- break inner loop */
                  fclose(bfp);
                  bfp = NULL;
                  break;
               }
            }
            /* compare block transaction ID with reference ID */
            cond = memcmp(txe.tx_id, tx[j].id, HASHLEN);
            /* if ID from block compares BEFORE reference, redo... */
         } while (cond < 0);
         /* if ID from block compares EQUAL TO reference, skip... */
         if (bfp != NULL && cond == 0) continue;
      }
      /* .. else; read reference transaction from previously set fpos */
      if (fsetpos(fp, &(tx[j].pos)) != 0) goto FAIL_ALL;
      if (tx_fread(&txc, &xdata, fp) != VEOK) goto FAIL_ALL;
      /* if (re)validation fails, skip... */
      /** @todo: replace tx_val with less wasteful tx_reval process */
      if (tx_val(&txc, &xdata) != VEOK) continue;
      /* ... else; write clean (valid) transaction to output */
      if (tx_fwrite(&txc, &xdata, tfp) != VEOK) goto FAIL_ALL;
      nout++;
   }  /* end for() */

   /* cleanup */
   fclose(tfp);
   fclose(bfp);
   fclose(fp);
   free(tx);

   /* out with the old, in with the new */
   remove(txfname);
   if (nout > 0) {
      if (rename(tmpfname, txfname) != 0) {
         remove(tmpfname);
         return VERROR;
      }
   }

   /* success */
   return VEOK;

   /* cleanup / error handling */
FAIL_ALL:
   fclose(tfp);
   remove(tmpfname);
FAIL_FP_MEM_BFP:
   if (bfp != NULL) fclose(bfp);
FAIL_FP_MEM:
   free(tx);
FAIL_FP:
   fclose(fp);

   return VERROR;
}  /* end txclean() */

/* Add src_ip to tx address map (weight[])
 * Called from process_tx()
 * Returns VERROR if no space in map, else VEOK.
 */
int txmap(TX *tx, word32 src_ip)
{
   int j;
   word32 *ipp;

   /* Apply Matt's Algorithm v1.0 to control mirroring... */
   if(get16(tx->len) != 0) {
      /* from wallet */
      memset(tx->weight, 0, 32);  /* clear address map */
   } else {
      /* try to put src_ip in map */
      for(ipp = (word32 *) tx->weight, j = 0; j < 8; ipp++, j++) {
         if(*ipp == 0) {
            *ipp = src_ip;
            break;
         }
      }
      if(j >= 8) return VERROR;  /* no space in map to mirror() */
   }  /* end if not from wallet */
   return VEOK;
}  /* end txmap() */


/* Create a grandchild to send TX's in mirror.dat to ip... */
pid_t mgc(word32 ip)
{
   pid_t pid;
   FILE *fp;
   long offset;
   int lockfd, count;
   TX mtx;
   NODE node;

   /* create grandchild */
   pid = fork();
   if(pid < 0) {
      perr("Cannot fork()");
      return 0;  /* to parent */
   }
   if(pid) return pid;  /* to parent */

   /* in (grand) child */
   pdebug("mgc()...");
   show("mgc()");

   fp = fopen("mirror.dat", "rb");
   if(fp == NULL) {
      perr("Cannot open mirror.dat");
      exit(1);
   }
   offset = 0;

   while(Running) {
      lockfd = lock("mq.lck", 20);
      if(lockfd == -1) {
         perr("Cannot lock mq.lck"); fclose(fp); exit(1);
      }
      if(fseek(fp, offset, SEEK_SET)) {
         unlock(lockfd); fclose(fp); exit(1);
      }
      /* read the TX from mirror.dat */
      count = fread(&mtx, 1, sizeof(TX), fp);
      /* preserve seek pos because other mgc()'s may be running */
      offset = ftell(fp);
      unlock(lockfd);
      if(count != sizeof(TX)) break;
      /* if not in -v modes... */
      if(Port == Dstport) {
         /* Skip this TX if ip address is already in map. */
         if(search32(ip, (word32 *) mtx.weight, 8)) continue;
      }
      if(callserver(&node, ip) != VEOK) break;
      put16(node.tx.len, TRANLEN);
      memcpy(TRANBUFF(&node.tx), TRANBUFF(&mtx), TRANLEN);
      /* copy ip address map to outgoing TX */
      memcpy(node.tx.weight, mtx.weight, 32);
      send_op(&node, OP_TX);
      sock_close(node.sd);
   }  /* end while Running */
   fclose(fp);
   exit(0);
}  /* end mgc() */

/* Send tx to all current or recent peers on iplist.
 * Called from server()       --  becomes child
 */
pid_t mirror1(word32 *iplist, int len)
{
   pid_t pid, peer[RPLISTLEN];
   int j;
   word8 busy;

   /* create child */
   pid = fork();
   if(pid < 0) {
      perr("Cannot fork()");
      return 0;
   }
   if(pid) return pid;  /* to parent */

   /* in child */
   pdebug("mirror()...");
   show("mirror");

   shuffle32(iplist, len);  /* NOTE: can create embedded zeros. */
   /* Create up to len mgc() grandchildren */
   for(j = 0; j < len; j++) {
      if(iplist[j] == 0) { peer[j] = 0; continue; }
      peer[j] = mgc(iplist[j]);  /* grandchild */
   }

   /* while Running, wait for grandchildren to finish. */
   while(Running) {
      busy = 0;
      for(j = 0; j < len; j++) {
         if(peer[j] == 0) continue;
         pid = waitpid(peer[j], NULL, WNOHANG);
         if(pid <= 0) busy = 1; else peer[j] = 0;
      }
      if(!busy) exit(0);
      else millisleep(1);
   }  /* end while Running */
   /* got SIGTERM */
   for(j = 0; j < len; j++) {
      if(peer[j]) {
         kill(peer[j], SIGTERM);     /* Kill grandchild */
         waitpid(peer[j], NULL, 0);  /* and burry her. */
      }
   }
   exit(0);
}  /* end mirror1() */

/* Send tx to either current or recent peers
 * Called from server()       --  becomes child
 */
pid_t mirror(void)
{
   word32 Splist[TPLISTLEN + RPLISTLEN] = { 0 };
   int i, num;

   num = 0;
   for (i = 0; i < TPLISTLEN; i++) {
      if (Tplist[i] == 0) break; /* no more trusted peers */
      Splist[num++] = Tplist[i];
   }
   for (i = 0; i < RPLISTLEN; i++) {
      if (Rplist[i] == 0) break; /* no more recent peers */
      Splist[num++] = Rplist[i];
   }

   return mirror1(Splist, num);
}


/* Called by gettx()  -- in parent
 *
 * Validate a TX, write clean TX to txq1.dat, and raw TX to
 * mirror queue, mq.dat.
 * Locks mq.lck while appending mq.dat.
 */
int process_tx(NODE *np)
{
   TX *tx;
   int evilness;
   int count, lockfd;
   int ecode;
   word8 tx_id[HASHLEN];
   FILE *fp;

   pdebug("process_tx()");
   show("tx");

   tx = &np->tx;

   /* Validate addresses, fee, signature, source balance, and total. */
   evilness = tx_val(tx);
   if(evilness) return evilness;

   /* Compute tx_id[] (hash of tx->src_addr) to append to txq1.dat. */
   sha256(tx->src_addr, TXWOTSLEN, tx_id);

   fp = fopen("txq1.dat", "ab");
   if(!fp) {
      perr("Cannot open txq1.dat");
      return 1;
   }

   /* Now write transaction to txq1.dat followed by tx_id */
   ecode = 0;
   /* 3 addresses (TXWOTSLEN*3) + 3 amounts (8*3) + signature (TXSIGLEN) */
   count = fwrite(TRANBUFF(tx), 1, TRANLEN, fp);
   if(count != TRANLEN) ecode = 1;
   /* then append source tx_id */
   count = fwrite(tx_id, 1, HASHLEN, fp);
   if(count != HASHLEN) ecode = 1;
   pdebug("writing TX to txq1.dat");
   fclose(fp);  /* close txq1.dat */
   if(ecode) {
      perr("bad write on txq1.dat");
      return 1;
   }
   else {
      Txcount++;
      pdebug("incrementing Txcount to %d", Txcount);
   }
   Nrec++;  /* total good TX received */

   /* lock mirror file */
   lockfd = lock("mq.lck", 20);
   if(lockfd == -1) {
      perr("Cannot lock mq.lck");  /* should not happen */
      return 1;
   }
   fp = fopen("mq.dat", "ab");
   if(!fp) {
      perr("Cannot open mq.dat");
      unlock(lockfd);
      return 1;
   }
   ecode = 0;
   /* If empty slot in mirror address map, fill it
    * in and then write tx to mirror queue, mq.dat.
    */
   pdebug("before txmap()");
   if(txmap(tx, np->ip) == VEOK) {
      count = fwrite(tx, 1, sizeof(TX), fp);
      if(count != sizeof(TX)) {
         perr("bad write on mq.dat");
         ecode = 1;
      } else Mqcount++;
   }
   pdebug("after txmap()");
   fclose(fp);      /* close mirror queue, mq.dat */
   unlock(lockfd);  /* unlock mirror queue lock, mq.lck */
   pdebug("done %d", ecode);
   return ecode;
}  /* end process_tx() */

/* end include guard */
#endif
