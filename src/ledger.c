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
 * @private
 * Comparison function to sort LTRAN objects by address + transaction code.
 * DOES NOT CONSIDER Ledger transaction amount in sorting process.
 */
static int lt_compare(const void *va, const void *vb)
{
   return memcmp(va, vb, TXADDRLEN + 1);
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
 * Update a leadger file, @a lefname, by applying deltas from a ledger
 * transaction file, @a ltfname. Ledger file is kept sorted on addr.
 * Ledger transaction file is sorted by addr+code, '-' comes before 'A'.
 * @param lefname Filename of the Ledger file to update
 * @param ltfname Filename of the Ledger transaction file containing deltas
 * @return (int) value representing the update result
 * @retval VEBAD2 on malicious; check errno for details
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 */
int le_update(const char *lefname, const char *ltfname)
{
   LENTRY le, le_hold;        /* for ledger entry and ledger hold data */
   LTRAN lt;                  /* for ledger transaction data */
   FILE *fp, *lefp, *ltfp;    /* output, ledger, and ltran file pointers */
   word8 le_prev[TXADDRLEN];     /* for ledger sequence check */
   word8 lt_prev[TXADDRLEN + 1]; /* for ltran sequence check */
   word8 hold, empty;
   int compare;
   char tmpfname[FILENAME_MAX];

   /* ensure ledger (global ref) is closed for update */
   le_close();

   /* sort the ledger transaction file */
   if (filesort(ltfname, sizeof(LTRAN), LEBUFSZ, lt_compare) != 0) {
      return VERROR;
   }

   /* open ledger (local ref), ledger transactions */
   lefp = fopen(lefname, "rb");
   if (lefp == NULL) return VERROR;
   ltfp = fopen(ltfname, "rb");
   if (ltfp == NULL) goto FAIL_LE;

   /* generate temporary filename and open as new ledger */
   snprintf(tmpfname, sizeof(tmpfname), "le-%04x.tmp", rand16());
   fp = fopen(tmpfname, "wb");
   if (fp == NULL) goto FAIL_LE_LT;

   /* read initial ledger entry and ledger transaction records */
   if (fread(&le, sizeof(LENTRY), 1, lefp) != 1) {
      if (ferror(lefp)) goto FAIL_ALL;
      /* EOF -- shouldn't happen, shouldn't matter */
      fclose(lefp);
      lefp = NULL;
   }
   if (fread(&lt, sizeof(LTRAN), 1, ltfp) != 1) {
      if (ferror(lefp)) goto FAIL_ALL;
      /* EOF -- shouldn't happen, shouldn't matter */
      fclose(ltfp);
      ltfp = NULL;
   }

   /* iterate through files while either files are NOT EOF */
   for (hold = 0, empty = 1; lefp != NULL || ltfp != NULL; ) {

      /* check ledger transaction file is open for processing */
      if (ltfp != NULL) {
         /* only perform initial comparison on ledger entry file */
         if (lefp != NULL) compare = addr_compare(le.addr, lt.addr);

         /* if ledger entry compares AFTER ledger transaction, OR
          * if ledger entry file is EOF... */
         if (compare > 0 || lefp == NULL) {
            /* the ONLY acceptable ledger transaction code here is an
             * CREDIT ("A") code, else assume malicious intent */
            if (lt.trancode[0] != 'A') {
               set_errno(EMCM_LTCREDIT);
               goto FAIL_DROP;
            }
            /* hold ledger entry while associated file is open */
            if (lefp != NULL) {
               memcpy(&le_hold, &le, sizeof(LENTRY));
               hold = 1;
            }
            /* clear ledger entry data, set ledger transaction address */
            memset(&le, 0, sizeof(LENTRY));
            memcpy(le.addr, lt.addr, TXADDRLEN);
            /* set compare for ledger transaction processing */
            compare = 0;
         }  /* end if (compare > 0 || lefp == NULL) */

         /* while ledger entry compares EQUAL TO ledger transaction... */
         while (compare == 0) {
            /* apply ledger transaction */
            switch ((char) lt.trancode[1]) {
               case 'A':
                  /* transaction CREDIT operation */
                  if (add64(le.balance, lt.amount, le.balance)) {
                     /** @todo: reconsider math overflow as error? */
                     /* set_errno(EMCM_MATH64_OVERFLOW); */
                     /* goto FAIL_DROP; */
                     memset(le.balance, 0, sizeof(le.balance));
                  }
                  break;
               case '-':
                  /* transaction DEBIT operation */
                  if (cmp64(le.balance, lt.amount) != 0) {
                     set_errno(EMCM_LTDEBIT);
                     goto FAIL_DROP;
                  }
                  memset(le.balance, 0, sizeof(le.balance));
                  break;
               default:
                  /* invalid transaction operation */
                  set_errno(EMCM_LTCODE);
                  goto FAIL_ALL;
            }
            /* read next ledger transaction */
            memcpy(lt_prev, lt.addr, TXADDRLEN + 1);
            if (fread(&lt, sizeof(LTRAN), 1, ltfp) != 1) {
               if (ferror(lefp)) goto FAIL_ALL;
               /* EOF -- cleanup, break inner loop */
               fclose(ltfp);
               ltfp = NULL;
               break;
            }
            /* check sort -- MUST BE ascending, ALLOW duplicates */
            /* NOTE: sort check SHOULD include transaction code */
            if (memcmp(lt_prev, lt.addr, TXADDRLEN + 1) > 0) {
               set_errno(EMCM_LTSORT);
               goto FAIL_ALL;
            }
            /* recompare latest ledger transaction */
            compare = addr_compare(le.addr, lt.addr);
         }  /* end while (compare == 0) */
      }  /* end if (ltfp != NULL) */

      /* if lendger entry compares BEFORE ledger transaction, OR
       * ledger transaction file is EOF... */
      if (compare < 0 || ltfp == NULL) {
         /* if ledger entry balance > MFEE, write to output */
         if (cmp64(le.balance, Mfee) > 0) {
            if (fwrite(&le, sizeof(LENTRY), 1, fp) != 1) {
               goto FAIL_ALL;
            } else empty = 0;
         }
         /* if ledger entry file open... */
         if (lefp != NULL) {
            /* copy ledger hold to ledger entry, OR... */
            if (hold) {
               memcpy(&le, &le_hold, sizeof(LENTRY));
               hold = 0;
               continue;
            }
            /* read next ledger transaction, AND... */
            memcpy(le_prev, le.addr, TXADDRLEN);
            if (fread(&le, sizeof(LENTRY), 1, lefp) != 1) {
               if (ferror(lefp)) goto FAIL_ALL;
               /* EOF -- cleanup */
               fclose(lefp);
               lefp = NULL;
               continue;
            }
            /* check sort -- MUST BE ascending, NO duplicates */
            if (memcmp(le_prev, le.addr, TXADDRLEN) >= 0) {
               set_errno(EMCM_LESORT);
               goto FAIL_ALL;
            }
         }  /* end  if (lefp != NULL) */
      }  /* end if (compare < 0... */
   }  /* end while () */
   /* empty ledger check */
   if (empty) {
      set_errno(EMCM_LEEMPTY);
      goto FAIL_ALL;
   }

   /* cleanup -- lefp/ltfp already closed */
   fclose(fp);

   /* finalize */
   remove(lefname);
   if (rename(tmpfname, lefname) != 0) return VERROR;
   /* ... make copy of ledger transaction file */
   snprintf(tmpfname, sizeof(tmpfname), "%s.last", ltfname);
   remove(tmpfname);
   if (rename(ltfname, tmpfname) != 0) return VERROR;

   /* success */
   return VEOK;

   /* cleanup / error handling */
FAIL_DROP:
   fclose(fp);
   remove(tmpfname);
   if (ltfp) fclose(ltfp);
   if (lefp) fclose(lefp);

   return VEBAD2;
FAIL_ALL:
   fclose(fp);
   remove(tmpfname);
FAIL_LE_LT:
   if (ltfp) fclose(ltfp);
FAIL_LE:
   if (lefp) fclose(lefp);

   return VERROR;
}  /* end le_update() */

/* end include guard */
#endif
