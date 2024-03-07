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
#include "sha256.h"
#include "extmath.h"
#include "extlib.h"
#include <errno.h>

static FILE *Lefp;
static word64 Nledger;
word32 Sanctuary;
word32 Lastday;

/**
 * Hashed-based address comparison function. Includes tag in comparison.
 * @param a Pointer to data to compare
 * @param b Pointer to data to compare against
 * @return (int) value representing comparison result
 * @retval 0 @a a is equal to @a b
 * @retval <0 @a a is less than @a b
 * @retval >0 @a a is greater than @a b
*/
static int addr_compare(const void *a, const void *b)
{
   return memcmp(a, b, TXADDRLEN);
}

/**
 * WOTS+ address comparison function. Includes tag in comparison.
 * @param a Pointer to data to compare
 * @param b Pointer to data to compare against
 * @return (int) value representing comparison result
 * @retval 0 @a a is equal to @a b
 * @retval <0 @a a is less than @a b
 * @retval >0 @a a is greater than @a b
*/
static int addr_compare_wots(const void *a, const void *b)
{
   return memcmp(a, b, TXWOTSLEN);
}

/**
 * Convert a WOTS+ address to a Hashed-based address. Copies tag data.
 * @param hash Pointer to destination hash-based address
 * @param wots Pointer to source WOTS+ address
*/
void hash_wots_addr(void *hash, const void *wots)
{
   sha256(wots, TXSIGLEN, hash);
   memcpy(
      (unsigned char *) hash + (TXADDRLEN - TXTAGLEN),
      (unsigned char *) wots + (TXWOTSLEN - TXTAGLEN),
      TXTAGLEN
   );
}

/* Open ledger "ledger.dat" */
int le_open(char *ledger, char *fopenmode)
{
   word64 offset;

   /* Already open? */
   if(Lefp) return VEOK;

   /* open ledger and seek to EOF */
   Lefp = fopen(ledger, fopenmode);
   if (Lefp == NULL) return VERROR;
   if (fseek64(Lefp, 0LL, SEEK_END) != 0) {
      le_close();
      return VERROR;
   }

   /* determine file size (via position) and check validity */
   offset = ftell64(Lefp);
   if(offset < sizeof(LENTRY) || (offset % sizeof(LENTRY)) != 0) {
      le_close();
      set_errno(EMCM_FILELEN);
      return VERROR;
   }

   /* set number of ledger entries */
   Nledger = offset / sizeof(LENTRY);

   return VEOK;
}  /* end le_open() */


void le_close(void)
{
   if(Lefp == NULL) return;
   fclose(Lefp);
   Lefp = NULL;
   Nledger = 0;
}

/**
 * Binary search for ledger address. If found, le is filled with the found
 * ledger entry data. Hash-based addresses are derived from supplied WOTS+
 * addresses where an appropriate length parameter is provided.
 * @param addr Address data to search for
 * @param le Pointer to place found ledger entry
 * @param len Length of address data to search
 * @return (int) value representing found result
 * @retval 0 on not found; check errno for details
 * @retval 1 on found; check le pointer for ledger data
*/
int le_find(word8 *addr, LENTRY *le, word16 len)
{
   word64 mid, hi, low;
   int cond;

   if(Lefp == NULL) {
      perr("use le_open() first!");
      return 0;
   }

   /* search length cannot be zero */
   if (len == 0) return 0;
   /* clamp search length to TXADDRLEN */
   if (len > TXADDRLEN) len = TXADDRLEN;

   low = 0;
   hi = Nledger - 1;

   while(low <= hi) {
      mid = (hi + low) / 2;
      if(fseek(Lefp, mid * sizeof(LENTRY), SEEK_SET) != 0) break;
      if(fread(le, 1, sizeof(LENTRY), Lefp) != sizeof(LENTRY)) break;
      cond = memcmp(addr, le->addr, len);
      if(cond == 0) return 1;  /* found target addr */
      if(cond < 0) hi = mid - 1; else low = mid + 1;
   }  /* end while */

   return 0;  /* not found */
}  /* end le_find() */

/**
 * Extract a ledger from a neo-genesis block. Checks sort.
 * @param neogen_file Filename of the neo-genesis block
 * @param ledger_file Filename of the ledger
 * @return (int) value representing extraction result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int le_extract(const char *neogen_file, const char *ledger_file)
{
   LENTRY le;              /* buffer for Hashed ledger entries */
   NGHEADER ngh;           /* buffer for neo-genesis header */
   FILE *fp, *lfp;         /* FILE pointers */
   word8 paddr[TXADDRLEN]; /* ledger address sort check */
   word64 lbytes;

   /* open files */
   fp = fopen(neogen_file, "rb");
   if (fp == NULL) return VERROR;
   lfp = fopen(ledger_file, "wb");
   if (lfp == NULL) goto FAIL_NEO;

   /* read/check neo-genesis hdrlen value */
   if (fread(&ngh.hdrlen, sizeof(ngh.hdrlen), 1, fp) != 1) goto FAIL_IO;
   if (get32(ngh.hdrlen) != sizeof(ngh)) {
      set_errno(EMCM_HDRLEN);
      goto FAIL_IO;
   }

   /* read/check ledger size and alignment */
   if (fread(&ngh.lbytes, sizeof(ngh.lbytes), 1, fp) != 1) goto FAIL_IO;
   put64(&lbytes, ngh.lbytes);
   if (lbytes < sizeof(le) || lbytes % sizeof(le) != 0) {
      set_errno(EMCM_LEEXTRACT);
      goto FAIL_IO;
   }

   /* read/write first ledger entry */
   if (fread(&le, sizeof(le), 1, fp) != 1) goto FAIL_IO;
   if (fwrite(&le, sizeof(le), 1, lfp) != 1) goto FAIL_IO;
   /* store entry for comparison */
   memcpy(paddr, le.addr, sizeof(le.addr));

   /* process remaining ledger from fp, copy it to lfp, check ledger sort */
   for (lbytes -= sizeof(le); lbytes > 0; lbytes -= sizeof(le)) {
      if (fread(&le, sizeof(le), 1, fp) != 1) goto FAIL_IO;
      /* check ledger sort*/
      if (addr_compare(le.addr, paddr) <= 0) {
         set_errno(EMCM_LESORT);
         goto FAIL_IO;
      }
      /* store entry for comparison */
      memcpy(paddr, le.addr, sizeof(le.addr));
      /* write hashed ledger entries to ledger file */
      if (fwrite(&le, sizeof(le), 1, lfp) != 1) goto FAIL_IO;
   }  /* end for() */

   /* close files */
   fclose(lfp);
   fclose(fp);

   /* ledger extracted */
   return VEOK;

   /* cleanup / error handling */
FAIL_IO:
   fclose(lfp);
FAIL_NEO:
   fclose(fp);

   return VERROR;
}  /* end le_extract() */

/**
 * Apply the Sanctuary Protocol to renew the ledger.
 * @return (int) value representing renew result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int le_renew(void)
{
   FILE *fp, *fpout;
   LENTRY le;
   word32 sanctuary[2];

   if(Sanctuary == 0) return VEOK;  /* success */
   le_close();  /* make sure ledger.dat is closed */
   sanctuary[0] = Sanctuary;
   sanctuary[1] = 0;

   /* open ledger and replacement files */
   fp = fopen("ledger.dat", "rb");
   if (fp == NULL) return VERROR;
   fpout = fopen("ledger.tmp", "wb");
   if (fpout == NULL) goto FAIL_DAT;

   /* renew the ledger per Carousal requirements */
   for(;;) {
      if (fread(&le, sizeof(le), 1, fp) != 1) {
         if (ferror(fp)) goto FAIL_IO;
         break;  /* EOF */
      }
      if(sub64(le.balance, sanctuary, le.balance)) continue;
      if(cmp64(le.balance, Mfee) <= 0) continue;
      if(fwrite(&le, sizeof(le), 1, fpout) != 1) goto FAIL_IO;
   }

   /* cleanup files -- swap ledger */
   fclose(fp);
   fclose(fpout);
   remove("ledger.dat");
   if (rename("ledger.tmp", "ledger.dat") != 0) {
      return VERROR;
   }

   /* success */
   return VEOK;

   /* cleanup / error handling */
FAIL_IO:
   fclose(fpout);
   remove("ledger.tmp");
FAIL_DAT:
   fclose(fp);

   return VERROR;
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
         hash2hex32(lt.addr, addrhex);
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
         hash2hex32(lt.addr, addrhex);
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
         hash2hex32(lt.addr, addrhex);
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
