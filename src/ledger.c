/**
 * @private
 * @headerfile ledger.h <ledger.h>
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_LEDGER_C
#define MOCHIMO_LEDGER_C


#include "ledger.h"

/* internal support */
#include "tag.h"
#include "sort.h"
#include "global.h"
#include "error.h"

/* external support */
#include <string.h>
#include "extmath.h"
#include "extlib.h"
#include <errno.h>

static FILE *Lefp;
static unsigned long Nledger;
word32 Sanctuary;
word32 Lastday;

/* Open ledger "ledger.dat" */
int le_open(char *ledger, char *fopenmode)
{
   unsigned long offset;

   /* Already open? */
   if(Lefp) return VEOK;
   Nledger = 0;
   Lefp = fopen(ledger, fopenmode);
   if(Lefp == NULL) {
      perrno("Cannot open ledger");
      return VERROR;
   }
   if(fseek(Lefp, 0, SEEK_END)) goto bad;
   offset = ftell(Lefp);
   if(offset < sizeof(LENTRY) || (offset % sizeof(LENTRY)) != 0) goto bad;
   Nledger = offset / sizeof(LENTRY);  /* number of ledger entries */
   return VEOK;
bad:
   fclose(Lefp);
   Lefp = NULL;
   perr("Bad ledger I/O format");
   return VERROR;
}  /* end le_open() */


void le_close(void)
{
   if(Lefp == NULL) return;
   fclose(Lefp);
   Lefp = NULL;
   Nledger = 0;
}

/* Binary search ledger.dat (Lefp) for addr.
 * input: addr
 * outputs: *le, *position, and return code.
 * Returns 1 if found, 0 if not found.
 * If found, le is filled in with ledger entry.
 * If position is non-NULL put the index of found LENTRY struct there,
 * else the index of where to insert addr in ledger.dat.
 */
int le_find(word8 *addr, LENTRY *le, long *position, word16 len)
{
   long cond, mid, hi, low;
   size_t addrlen;

   if(Lefp == NULL) {
      perr("use le_open() first!");
      return 0;
   }

   low = 0;
   hi = Nledger - 1;
   addrlen = len < 2 ? TXADDRLEN : len;

   while(low <= hi) {
      mid = (hi + low) / 2;
      if(fseek(Lefp, mid * sizeof(LENTRY), SEEK_SET) != 0) break;
      if(fread(le, 1, sizeof(LENTRY), Lefp) != sizeof(LENTRY)) break;
      cond = memcmp(addr, le->addr, addrlen);
      if(cond == 0) {
         if(position) *position = mid;
         return 1;  /* found target addr */
      }
      if(cond < 0) hi = mid - 1; else low = mid + 1;
   }  /* end while */
   /* Not found.
    * To add target addr, move ledger[position] up and insert target
    * at ledger[position].
    */
   if(position) *position = low;
   return 0;  /* not found */
}  /* end le_find() */

/* Extract the ledger from a neo-genesis block and
 * put it in ledger file lfile (ledger.dat)
 * Return VEOK on success, else VERROR.
 */
int le_extract(char *fname, char *lfile)
{
   word32 hdrlen;    /* to read-in block header length */
   FILE *fp, *lfp;
   LENTRY le;        /* buffer to read ledger entry */
   word8 prevaddr[TXADDRLEN];  /* to check block addr sort */
   word8 first;

   pdebug("le_extract() ledger from %s to %s", fname, lfile);

   /* open the neo-genesis block and read in file header length */
   fp = fopen(fname, "rb");
   if(!fp) return VERROR;;
   if(fread(&hdrlen, 1, 4, fp) != 4) goto ioerror;

   lfp = fopen(lfile, "wb");
   if(!lfp) {
      perr("Cannot open %s", lfile);
      goto ioerror;
   }

   /* Make sure that NG header contains at least
    * one ledger entry.
    */
   if(hdrlen < (sizeof(LENTRY) + 4)) {
      perr("Not a neo-genesis block: %s", fname);
      goto error2;
   }

   /*
    * Read the ledger from fp and copy it to lfp,
    * creating a new ledger.dat file.
    * NOTE: block trailer must be less than sizeof(LENTRY)
    */
   if(fseek(fp, 4, SEEK_SET)) goto error2;
   for(hdrlen -= 4, first = 1; ; first = 0) {
      if(fread(&le, 1, sizeof(LENTRY), fp) != sizeof(LENTRY))
         break;
      hdrlen -= sizeof(LENTRY);
      /* check ledger sort in NG block */
      if(!first && memcmp(le.addr, prevaddr, TXADDRLEN) <= 0) {
         perr("bad ledger sort in neo-genesis block");
         goto error2;
      }
      memcpy(prevaddr, le.addr, TXADDRLEN);
      if(fwrite(&le, 1, sizeof(LENTRY), lfp) != sizeof(LENTRY))
         goto error2;
   }
   if(hdrlen) {
      perr("bad neo-genesis block length");
      goto error2;
   }
   fclose(fp);
   fclose(lfp);
   return VEOK;
ioerror:
      fclose(fp);
      remove(lfile);  /* remove bad ledger */
      perr("le_extract() failed!");
      return VERROR;
error2:
   fclose(lfp);
   goto ioerror;
}  /* end le_extract() */

/* Returns 0 on success, else error code. */
int le_renew(void)
{
   FILE *fp, *fpout;
   LENTRY le;
   int message = 0;
   word32 n, m;
   static word32 sanctuary[2];

   if(Sanctuary == 0) return VEOK;  /* success */
   le_close();  /* make sure ledger.dat is closed */
   plog("Lastday 0x%0x.  Carousel begins...", Lastday);
   n = m = 0;
   sanctuary[0] = Sanctuary;

   fp = fopen("ledger.dat", "rb");
   if(fp == NULL) return VERROR;
   fpout = fopen("ledger.tmp", "wb");
   if(fpout == NULL) goto CLEANUP_DAT;
   for(;;) {
      if(fread(&le, sizeof(le), 1, fp) != 1) break;
      n++;
      if(sub64(le.balance, sanctuary, le.balance)) continue;
      if(cmp64(le.balance, Mfee) <= 0) continue;
      if(fwrite(&le, sizeof(le), 1, fpout) != 1) goto CLEANUP_TMP;
      m++;
   }
   fclose(fp);
   fclose(fpout);
   fp = fpout = NULL;
   remove("ledger.dat");
   if(rename("ledger.tmp", "ledger.dat")) goto CLEANUP_TMP;
   plog("%u citizens renewed out of %u", n - m, n);

   /* success */
   return VEOK;

   /* cleanup / error handling */
CLEANUP_TMP:
   fclose(fpout);
   remove("ledger.tmp");
CLEANUP_DAT:
   fclose(fp);
   perr("Carousel renewal code: %d (%u,%u)", message, n - m, n);
   return VERROR;
}  /* end le_renew() */

/**
 * Remove bad TX's from a txclean file based on the ledger file.
 * Uses "ledger.dat" as (input) ledger file, "txq.tmp" as temporary (output)
 * txclean file and renames to "txclean.dat" on success.
 * @returns VEOK on success, else VERROR
 * @note Nothing else should be using the ledger.
 * @note Leaves ledger.dat open on return.
*/
int le_txclean(void)
{
   TXQENTRY tx;            /* Holds one transaction in the array */
   LENTRY src_le;          /* for le_find() */
   FILE *fp, *fpout;       /* input/output txclean file pointers */
   MTX *mtx;               /* for MTX checks */
   word32 j;               /* unsigned iteration and comparison */
   word32 total[2];
   word32 nout, tnum; /* transaction record counters */
   word8 addr[TXADDRLEN];  /* for tag_find() (MTX checks) */
   clock_t ticks;
   char addrhex[10];

   /* init */
   ticks = clock();
   nout = 0;
   tnum = 0;

   /* check txclean exists AND has transactions to clean */
   if (!fexists("txclean.dat")) {
      pdebug("nothing to clean, done...");
      return VEOK;
   }

   /* ensure ledger is open */
   if (le_open("ledger.dat", "rb") != VEOK) {
      perr("failed to le_open(ledger.dat)");
      return VERROR;
   }

   /* open clean TX queue and new (temp) clean TX queue */
   fp = fopen("txclean.dat", "rb");
   if (fp == NULL) {
      perrno("cannot open txclean");
      return VERROR;
   }
   fpout = fopen("txq.tmp", "wb");
   if (fpout == NULL) {
      perrno("cannot open txq");
      goto CLEANUP_TXC;
   }

   /* read TX from txclean.dat and process */
   for(; fread(&tx, sizeof(TXQENTRY), 1, fp); tnum++) {
      /* check src in ledger, balances and amounts are good */
      if (le_find(tx.src_addr, &src_le, NULL, TXADDRLEN) == FALSE) {
         hash2hex(tx.tx_id, 4, addrhex);
         pdebug("le_find, drop %s...", addrhex);
         continue;
      } else if (cmp64(tx.tx_fee, Myfee) < 0) {
         hash2hex(tx.tx_id, 4, addrhex);
         pdebug("tx_fee, drop %s...", addrhex);
         continue;
      } else if (add64(tx.send_total, tx.change_total, total)) {
         hash2hex(tx.tx_id, 4, addrhex);
         pdebug("amounts, drop %s...", addrhex);
         continue;
      } else if (add64(tx.tx_fee, total, total)) {
         hash2hex(tx.tx_id, 4, addrhex);
         pdebug("total, drop %s...", addrhex);
         continue;
      } else if (cmp64(src_le.balance, total) != 0) {
         hash2hex(tx.tx_id, 4, addrhex);
         pdebug("balance, drop %s...", addrhex);
         continue;
      } else if (TX_IS_MTX(&tx) && get32(Cblocknum) >= MTXTRIGGER) {
         pdebug("MTX detected...");
         mtx = (MTX *) &tx;
         for(j = 0; j < MDST_NUM_DST; j++) {
            if (iszero(mtx->dst[j].tag, TXTAGLEN)) break;
            memcpy(ADDR_TAG_PTR(addr), mtx->dst[j].tag, TXTAGLEN);
            mtx->zeros[j] = 0;
            /* If dst[j] tag not found, put error code in zeros[] array. */
            if (tag_find(addr, NULL, NULL, TXTAGLEN) != VEOK) {
               mtx->zeros[j] = 1;
            }
         }
      } else if (tag_valid(tx.src_addr, tx.chg_addr, tx.dst_addr,
            NULL) != VEOK) {
         hash2hex(tx.tx_id, 4, addrhex);
         pdebug("tags, drop %s...", addrhex);
         continue;
      }
      /* write TX to new queue */
      if (fwrite(&tx, sizeof(TXQENTRY), 1, fpout) != 1) {
         perr("failed to fwrite(tx): TX#%u", tnum);
         goto CLEANUP_TXQ;
      }
      nout++;
   }  /* end for (nout = tnum = 0... */

   /* cleanup */
   fclose(fpout);
   fclose(fp);

   /* finalize */
   remove("txclean.dat");
   if (nout == 0) pdebug("txclean.dat is empty");
   else if (rename("txq.tmp", "txclean.dat") != VEOK) {
      perr("failed to move txq to txclean");
      remove("txq.tmp");
      return VERROR;
   }

   /* clean TX queue is updated */
   pdebug("wrote %u/%u entries to txclean", nout, tnum);
   pdebug("ledger txclean done in %gs", diffclocktime(ticks));

   /* success */
   return VEOK;

   /* cleanup / error handling */
CLEANUP_TXQ:
   fclose(fpout);
   remove("txq.tmp");
CLEANUP_TXC:
   fclose(fp);
   return VERROR;
}  /* end le_txclean() */

/**
 * Update leadger by applying ledger transaction deltas to a ledger. Uses
 * "ltran.dat" as (input) ledger transaction deltas file, "ledger.tmp" as
 * temporary (output) ledger file and renames to "ledger.dat" on success.
 * Ledger file is kept sorted on addr. Ledger transaction file is sorted by
 * sortlt() on addr+trancode, where '-' comes before 'A'.
 * @returns VEOK on success, else VERROR
*/
int le_update(void)
{
   LENTRY oldle, newle;    /* input/output ledger entries */
   LTRAN lt;               /* ledger transaction  */
   FILE *ltfp, *fp;        /* input ltran and output ledger pointers */
   FILE *lefp;             /* ledger file pointers */
   clock_t ticks;
   word32 nout;            /* temp file output counter */
   word8 hold;             /* hold ledger entry for next loop */
   word8 taddr[TXADDRLEN];    /* for transaction address hold */
   word8 le_prev[TXADDRLEN];  /* for ledger sequence check */
   word8 lt_prev[TXADDRLEN];  /* for tran delta sequence check */
   int cond;
   char addrhex[10];

   /* init */
   ticks = clock();
   nout = 0;         /* output record counter */
   hold = 0;         /* hold ledger flag */
   memset(le_prev, 0, TXADDRLEN);
   memset(lt_prev, 0, TXADDRLEN);

   /* ensure ledger reference is closed for update */
   le_close();

   /* sort the ledger transaction file */
   if (sortlt("ltran.dat") != VEOK) {
      perr("le_update: bad sortlt(ltran.dat)");
      return VERROR;
   }

   /* open ledger (local ref), ledger transactions, and new ledger */
   lefp = fopen("ledger.dat", "rb");
   if (lefp == NULL) {
      perrno("failed to fopen(ledger.dat)");
      return VERROR;
   }
   ltfp = fopen("ltran.dat", "rb");
   if (ltfp == NULL) {
      perrno("failed to fopen(ltran.dat)");
      goto CLEANUP_LE;
   }
   fp = fopen("ledger.tmp", "wb");
   if (fp == NULL) {
      perrno("failed to fopen(ledger.tmp)");
      goto CLEANUP_LT;
   }

   /* prepare initial ledger transaction */
   fread(&lt, sizeof(LTRAN), 1, ltfp);
   if (ferror(ltfp)) {
      perrno("failed to fread(lt)");
      goto CLEANUP_TMP;
   }

   /* while one of the files is still open */
   while (feof(lefp) == 0 || feof(ltfp) == 0) {
      /* if ledger entry on hold, skip read, else do read and sort checks */
      if (hold) hold = 0;
      else if (feof(lefp) == 0) {
         /* read ledger entry, check sort, and store entry in le_prev */
         if (fread(&oldle, sizeof(LENTRY), 1, lefp) != 1) {
            /* check file errors, else "continue" loop for eof check */
            if (ferror(lefp)) {
               perrno("fread(oldle)");
               goto CLEANUP_TMP;
            } else continue;
         } else if (memcmp(oldle.addr, le_prev, TXADDRLEN) < 0) {
            perr("bad ledger.dat sort");
            goto CLEANUP_TMP;
         } else memcpy(le_prev, oldle.addr, TXADDRLEN);
      }
      /* compare ledger address to latest transaction address */
      cond = memcmp(oldle.addr, lt.addr, TXADDRLEN);
      if (cond == 0 && feof(ltfp) == 0 && feof(lefp) == 0) {
         /* If ledger and transaction addr match,
          * and both files not at end...
          * copy the old ledger entry to a new struct for editing */
         hash2hex(lt.addr, 4, addrhex);
         pdebug("editing address %s...", addrhex);
         memcpy(&newle, &oldle, sizeof(LENTRY));
      } else if ((cond < 0 || feof(ltfp)) && feof(lefp) == 0) {
         /* If ledger compares "before" transaction or transaction eof,
          * and ledger file is NOT at end...
          * write the old ledger entry to temp file */
         if (fwrite(&oldle, sizeof(LENTRY), 1, fp) != 1) {
            perr("bad write on temp ledger");
            goto CLEANUP_TMP;
         }
         nout++;  /* count records in temp file */
         continue;  /* nothing else to do */
      } else if((cond > 0 || feof(lefp)) && feof(ltfp) == 0) {
         /* If the next ledger entry comes "after" the current transaction
          * or ledger file is EOF, AND transaction file is NOT EOF... */
         if(lt.trancode[0] != 'A') {
            /* ... the ONLY acceptable trancode is an append ("A"), and is
             * considered malicious intent if missed by previous checks */
            perr("create tran not 'A'");
            goto CLEANUP_DROP;
         }
         hash2hex(lt.addr, 4, addrhex);
         pdebug("creating address %s...", addrhex);
         /* CREATE NEW ADDR
          * Copy address from transaction to new ledger entry.
          */
         memcpy(&newle.addr, lt.addr, TXADDRLEN);
         memset(newle.balance, 0, 8);  /* but zero balance for apply_tran */
         /* Hold old ledger entry to insert before this addition. */
         hold = 1;
      }

      /* save ledger transaction address */
      memcpy(taddr, lt.addr, TXADDRLEN);

      do {
         hash2hex(lt.addr, 4, addrhex);
         pdebug("Applying '%c' to %s...", (char) lt.trancode[0], addrhex);
         /* '-' transaction sorts before 'A' */
         if (lt.trancode[0] == 'A') {
            if (add64(newle.balance, lt.amount, newle.balance)) {
               pdebug("balance OVERFLOW! Zero-ing balance...");
               memset(newle.balance, 0, 8);
            }
         } else if(lt.trancode[0] == '-') {
            if (cmp64(newle.balance, lt.amount) != 0) {
               perr("'-' balance != trans amount");
               goto CLEANUP_DROP;
            }
            memset(newle.balance, 0, 8);
         } else {
            perr("bad trancode");
            goto CLEANUP_TMP;
         }
         /* --- ^ shouldn't happen */
         /* read next transaction */
         pdebug("apply -- reading transaction");
         if (fread(&lt, sizeof(LTRAN), 1, ltfp) != 1) {
            if (ferror(ltfp)) {
               perrno("fread(lt)");
               goto CLEANUP_TMP;
            }
            pdebug("eof on tran");
            break;
         }
         /* Sequence check on lt.addr */
         if (memcmp(lt.addr, lt_prev, TXADDRLEN) < 0) {
            perr("bad ltran.dat sort");
            goto CLEANUP_TMP;
         }
         memcpy(lt_prev, lt.addr, TXADDRLEN);

         /* Check for multiple transactions on a single address:
         * '-' must come before 'A'
         * (Transaction file did not run out and its addr matches
         *  the previous transaction...)
         */
      } while (memcmp(lt.addr, taddr, TXADDRLEN) == 0);

      /* Only balances > Mfee are written to updated ledger. */
      if (cmp64(newle.balance, Mfee) > 0) {
         pdebug("writing new balance");
         /* write new balance to temp file */
         if (fwrite(&newle, sizeof(LENTRY), 1, fp) != 1) {
            perr("bad write on temp ledger");
            goto CLEANUP_TMP;
         }
         nout++;  /* count output records */
      } else pdebug("new balance <= Mfee is not written");
   }  /* end while not both on EOF  -- updating ledger */

   /* cleanup */
   fclose(fp);
   fclose(ltfp);
   fclose(lefp);

   /* finalize */
   if (nout) {
      /* if there are entries in ledger.tmp */
      remove("ledger.dat");
      rename("ledger.tmp", "ledger.dat");
      remove("ltran.dat.last");
      rename("ltran.dat", "ltran.dat.last");
   } else {
      remove("ledger.tmp");  /* remove empty temp file */
      perr("the ledger is empty!");
      return VERROR;
   }

   pdebug("wrote %u entries to new ledger", nout);
   pdebug("ledger update completed in %gs", diffclocktime(ticks));

   /* success */
   return VEOK;

   /* failure / error handling */
CLEANUP_DROP:
   fclose(fp);
   remove("ledger.tmp");
   fclose(ltfp);
   fclose(lefp);
   return VEBAD2;
CLEANUP_TMP:
   fclose(fp);
   remove("ledger.tmp");
CLEANUP_LT:
   fclose(ltfp);
CLEANUP_LE:
   fclose(lefp);
   return VERROR;
}  /* end le_update() */

/* end include guard */
#endif
