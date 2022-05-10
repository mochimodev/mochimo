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
#include "util.h"
#include "tag.h"
#include "sort.h"
#include "global.h"

/* external support */
#include <string.h>
#include "extprint.h"
#include "extmath.h"
#include "extlib.h"
#include <errno.h>

static FILE *Lefp;
static unsigned long Nledger;

/* Open ledger "ledger.dat" */
int le_open(char *ledger, char *fopenmode)
{
   unsigned long offset;

   /* Already open? */
   if(Lefp) return VEOK;
   Nledger = 0;
   Lefp = fopen(ledger, fopenmode);
   if(Lefp == NULL)
      return perrno(errno, "le_open(): Cannot open ledger");
   if(fseek(Lefp, 0, SEEK_END)) goto bad;
   offset = ftell(Lefp);
   if(offset < sizeof(LENTRY) || (offset % sizeof(LENTRY)) != 0) goto bad;
   Nledger = offset / sizeof(LENTRY);  /* number of ledger entries */
   return VEOK;
bad:
   fclose(Lefp);
   Lefp = NULL;
   return perr("le_open(): Bad ledger I/O format");
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
      perr("le_find(): use le_open() first!");
      return 0;
   }

   low = 0;
   hi = Nledger - 1;
   addrlen = len < 2 ? TXADDRLEN : len;

   while(low <= hi) {
      mid = (hi + low) / 2;
      if(fseek(Lefp, mid * sizeof(LENTRY), SEEK_SET) != 0)
         { perr("le_find(): fseek");  break; }
      if(fread(le, 1, sizeof(LENTRY), Lefp) != sizeof(LENTRY))
         { perrno(errno, "le_find(): fread");  break; }
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
      perr("le_extract(): Cannot open %s", lfile);
      goto ioerror;
   }

   /* Make sure that NG header contains at least
    * one ledger entry.
    */
   if(hdrlen < (sizeof(LENTRY) + 4)) {
      perr("le_extract(): Not a neo-genesis block: %s", fname);
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
         perr("le_extract(): bad ledger sort in neo-genesis block");
         goto error2;
      }
      memcpy(prevaddr, le.addr, TXADDRLEN);
      if(fwrite(&le, 1, sizeof(LENTRY), lfp) != sizeof(LENTRY))
         goto error2;
   }
   if(hdrlen) {
      perr("le_extract(): bad neo-genesis block length");
      goto error2;
   }
   fclose(fp);
   fclose(lfp);
   return VEOK;
ioerror:
      fclose(fp);
      remove(lfile);  /* remove bad ledger */
      return perr("le_extract() failed!");
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

   if(Sanctuary == 0) return 0;  /* success */
   le_close();  /* make sure ledger.dat is closed */
   plog("Lastday 0x%0x.  Carousel begins...", Lastday);
   n = m = 0;
   fp = fpout = NULL;
   sanctuary[0] = Sanctuary;

   fp = fopen("ledger.dat", "rb");
   if(fp == NULL) BAIL(1);
   fpout = fopen("ledger.tmp", "wb");
   if(fpout == NULL) BAIL(2);
   for(;;) {
      if(fread(&le, sizeof(le), 1, fp) != 1) break;
      n++;
      if(sub64(le.balance, sanctuary, le.balance)) continue;
      if(cmp64(le.balance, Mfee) <= 0) continue;
      if(fwrite(&le, sizeof(le), 1, fpout) != 1) BAIL(3);
      m++;
   }
   fclose(fp);
   fclose(fpout);
   fp = fpout = NULL;
   remove("ledger.dat");
   if(rename("ledger.tmp", "ledger.dat")) BAIL(4);
   plog("%u citizens renewed out of %u", n - m, n);
   return 0;  /* success */
bail:
   if(fp != NULL) fclose(fp);
   if(fpout != NULL) fclose(fpout);
   perr("Carousel renewal code: %d (%u,%u)", message, n - m, n);
   return message;
}  /* end le_renew() */

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
   FILE *fp, *fpout;       /* input txclean and output txq file pointers */
   FILE *lfp;              /* ledger file pointers */
   clock_t ticks;
   word32 nout;            /* temp file output counter */
   word8 hold;             /* hold ledger entry for next loop */
   word8 taddr[TXADDRLEN];    /* for transaction address hold */
   word8 le_prev[TXADDRLEN];  /* for ledger sequence check */
   word8 lt_prev[TXADDRLEN];  /* for tran delta sequence check */
   int cond, ecode;

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
      mError(FAIL, "le_update: bad sortlt(ltran.dat)");
   }

   /* open input, ledger, temp output ledger files */
   lfp = fopen("ledger.dat", "rb");
   if (lfp == NULL) {
      mErrno(FAIL, "le_update(): failed to fopen(ledger.dat)");
   }
   fp = fopen("ltran.dat", "rb");
   if (fp == NULL) {
      mErrno(FAIL_IN, "le_update(): failed to fopen(ltran.dat)");
   }
   fpout = fopen("ledger.tmp", "wb");
   if (fpout == NULL) {
      mErrno(FAIL_OUT, "le_update(): failed to fopen(ledger.tmp)");
   }

   /* prepare initial ledger transaction */
   fread(&lt, sizeof(LTRAN), 1, fp);
   if (ferror(fp)) {
      mErrno(FAIL_IO, "le_update(): failed to fread(lt)");
   }

   /* while one of the files is still open */
   while (feof(lfp) == 0 || feof(fp) == 0) {
      /* if ledger entry on hold, skip read, else do read and sort checks */
      if (hold) hold = 0;
      else if (feof(lfp) == 0) {
         /* read ledger entry, check sort, and store entry in le_prev */
         if (fread(&oldle, sizeof(LENTRY), 1, lfp) != 1) {
            /* check file errors, else "continue" loop for eof check */
            if (ferror(lfp)) {
               mErrno(FAIL_IO, "le_update(): fread(oldle)");
            } else continue;
         } else if (memcmp(oldle.addr, le_prev, TXADDRLEN) < 0) {
            mError(FAIL_IO, "le_update(): bad ledger.dat sort");
         } else memcpy(le_prev, oldle.addr, TXADDRLEN);
      }
      /* compare ledger address to latest transaction address */
      cond = memcmp(oldle.addr, lt.addr, TXADDRLEN);
      if (cond == 0 && feof(fp) == 0 && feof(lfp) == 0) {
         /* If ledger and transaction addr match,
          * and both files not at end...
          * copy the old ledger entry to a new struct for editing */
         pdebug("le_update(): editing address %s...", addr2str(lt.addr));
         memcpy(&newle, &oldle, sizeof(LENTRY));
      } else if ((cond < 0 || feof(fp)) && feof(lfp) == 0) {
         /* If ledger compares "before" transaction or transaction eof,
          * and ledger file is NOT at end...
          * write the old ledger entry to temp file */
         if (fwrite(&oldle, sizeof(LENTRY), 1, fpout) != 1) {
            mError(FAIL_IO, "le_update(): bad write on temp ledger");
         }
         nout++;  /* count records in temp file */
         continue;  /* nothing else to do */
      } else if((cond > 0 || feof(lfp)) && feof(fp) == 0) {
         /* If ledger compares "after" transaction or ledger eof,
          * and transaction file is NOT at end...
          */
         if(lt.trancode[0] != 'A') {
            mEdrop(FAIL_IO, "le_update(): create tran not 'A'");
         }
         pdebug("le_update(): creating address %s...", addr2str(lt.addr));
         /* CREATE NEW ADDR
          * Copy address from transaction to new ledger entry.
          */
         memcpy(&newle, lt.addr, TXADDRLEN);
         memset(newle.balance, 0, 8);  /* but zero balance for apply_tran */
         /* Hold old ledger entry to insert before this addition. */
         hold = 1;
      }

      /* save ledger transaction address */
      memcpy(taddr, lt.addr, TXADDRLEN);

      do {
         pdebug("le_update(): Applying '%c' to %s...",
            (char) lt.trancode[0], addr2str(lt.addr));
         /* '-' transaction sorts before 'A' */
         if (lt.trancode[0] == 'A') {
            if (add64(newle.balance, lt.amount, newle.balance)) {
               pdebug("le_update(): balance OVERFLOW! Zero-ing balance...");
               memset(newle.balance, 0, 8);
            }
         } else if(lt.trancode[0] == '-') {
            if (cmp64(newle.balance, lt.amount) != 0) {
               mEdrop(FAIL_IO, "le_update(): '-' balance != trans amount");
            }
            memset(newle.balance, 0, 8);
         } else mError(FAIL_IO, "le_update(): bad trancode");
         /* --- ^ shouldn't happen */
         /* read next transaction */
         pdebug("le_update(): apply -- reading transaction");
         if (fread(&lt, sizeof(LTRAN), 1, fp) != 1) {
            if (ferror(fp)) mErrno(FAIL_IO, "le_update(): fread(lt)");
            pdebug("le_update(): eof on tran");
            break;
         }
         /* Sequence check on lt.addr */
         if (memcmp(lt.addr, lt_prev, TXADDRLEN) < 0) {
            mError(FAIL_IO, "le_update(): bad ltran.dat sort");
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
         pdebug("le_update(): writing new balance");
         /* write new balance to temp file */
         if (fwrite(&newle, sizeof(LENTRY), 1, fpout) != 1) {
            mError(FAIL_IO, "le_update(): bad write on temp ledger");
         }
         nout++;  /* count output records */
      } else pdebug("le_update(): new balance <= Mfee is not written");
   }  /* end while not both on EOF  -- updating ledger */

   fclose(fpout);
   fclose(fp);
   fclose(lfp);
   if (nout) {
      /* if there are entries in ledger.tmp */
      remove("ledger.dat");
      rename("ledger.tmp", "ledger.dat");
      remove("ltran.dat.last");
      rename("ltran.dat", "ltran.dat.last");
   } else {
      remove("ledger.tmp");  /* remove empty temp file */
      mError(FAIL, "le_update(): the ledger is empty!");
   }

   pdebug("le_update(): wrote %u entries to new ledger", nout);
   pdebug("le_update(): completed in %gs", diffclocktime(clock(), ticks));

   /* success */
   return VEOK;

   /* failure / error handling */
FAIL_IO:
   fclose(fpout);
   remove("ledger.tmp");
FAIL_OUT:
   fclose(lfp);
FAIL_IN:
   fclose(lfp);
FAIL:

   return ecode;
}  /* end le_update() */

/* end include guard */
#endif
