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

/* LEGACY WOTS+ ledger entry struct */
typedef struct {
   word8 addr[TXWOTSLEN];
   word8 balance[8];
} LENTRY_W;

static FILE *Lefp;
static long long Nledger;
static char Lefile[FILENAME_MAX];
word32 Sanctuary;
word32 Lastday;

/**
 * @private
 * Efficient 20-byte equality check, extending the legacy 12-byte one.
 * @param a Pointer to data to compare
 * @param b Pointer to data to compare against
 * @returns 1 if tags match, else 0
 */
static inline int equality_check_20bytes(const void *a, const void *b)
{
   return (
      ((word32 *) a)[0] == ((word32 *) b)[0] &&
      ((word32 *) a)[1] == ((word32 *) b)[1] &&
      ((word32 *) a)[2] == ((word32 *) b)[2] &&
      ((word32 *) a)[3] == ((word32 *) b)[3] &&
      ((word32 *) a)[4] == ((word32 *) b)[4]
   );
}

/**
 * Hashed-based address comparison function. Includes tag in comparison.
 * @param a Pointer to address to compare
 * @param b Pointer to address to compare against
 * @return (int) value representing result
 * @retval 0 @a a is equal to @a b
 * @retval <0 @a a is less than @a b
 * @retval >0 @a a is greater than @a b
 */
int addr_compare(const void *a, const void *b)
{
   return memcmp(a, b, ADDR_LEN);
}

/**
 * Equality check for address hash. ONLY compares hash.
 * Implements an efficient 20-byte check, mimicing the tag check one.
 * @param a Pointer to address with hash to compare
 * @param b Pointer to address with hash to compare against
 * @returns 1 if address hashes match, else 0
 */
int addr_hash_equal(const void *a, const void *b)
{
   return equality_check_20bytes(ADDR_HASH_PTR(a), ADDR_HASH_PTR(b));
}

/**
 * Convert an implicit Mochimo Address tag to a full Hash-based Address.
 * @param tag Pointer to tag to convert
 * @param addr Pointer to address to store result
 */
void addr_from_implicit(const void *tag, void *addr)
{
   memcpy(ADDR_TAG_PTR(addr), tag, ADDR_TAG_LEN);
   memcpy(ADDR_HASH_PTR(addr), tag, ADDR_TAG_LEN);
}

/**
 * Generate a Mochimo Address hash using SHA3-512 and RIPEMD-160.
 * @param in Pointer to data to hash
 * @param inlen Length of data to hash
 * @param out Pointer to store hash result
 */
void addr_hash_generate(const void *in, size_t inlen, void *out)
{
   word8 hash[SHA3LEN512]; /* intermediate sha3-512 compound hash */

   /* perform compound hash -- ripemd160(sha3(in)) */
   sha3(in, inlen, hash, SHA3LEN512);
   ripemd160(hash, SHA3LEN512, out);
}

/**
 * Convert Legacy WOTS+ address to hash-based Mochimo Address.
 * @param wots Pointer to WOTS+ address to convert
 * @param addr Pointer to hash-based address to store result
 */
void addr_from_wots(const void *wots, void *addr)
{
   const word32 default_tag[] = { 0x42, 0x0e, 0x01 };

   addr_hash_generate(wots, WOTS_PK_LEN, ADDR_HASH_PTR(addr));

   /* legacy "default tags" require explicit tagging */
   if (memcmp(WOTS_TAG_PTR(wots), default_tag, WOTS_TAG_LEN) == 0) {
      memcpy(ADDR_TAG_PTR(addr), ADDR_HASH_PTR(addr), ADDR_HASH_LEN);
      return;
   }

   /* ... otherwise, copy legacy tags (append zeros to fill) */
   memcpy(ADDR_TAG_PTR(addr), WOTS_TAG_PTR(wots), WOTS_TAG_LEN);
   memset(ADDR_TAG_PTR(addr) + WOTS_TAG_LEN, 0, ADDR_HASH_LEN - WOTS_TAG_LEN);
}  /* end addr_from_wots() */

/**
 * Hashed-based address tag comparison function. ONLY compares tag.
 * @param a Pointer to data to compare
 * @param b Pointer to data to compare against
 * @return (int) value representing result
 * @retval 0 @a a is equal to @a b
 * @retval <0 @a a is less than @a b
 * @retval >0 @a a is greater than @a b
 */
int addr_tag_compare(const void *a, const void *b)
{
   return tag_compare(ADDR_TAG_PTR(a), ADDR_TAG_PTR(b));
}

/**
 * Equality check for address tags. ONLY compares tag.
 * Implements an efficient 20-byte check, extending the legacy 12-byte one.
 * @param a Pointer to address with tag to compare
 * @param b Pointer to address with tag to compare against
 * @returns 1 if address tags match, else 0
 */
int addr_tag_equal(const void *a, const void *b)
{
   return equality_check_20bytes(ADDR_TAG_PTR(a), ADDR_TAG_PTR(b));
}

/**
 * Read an address tag from a file. Supports various address types,
 * including legacy WOTS+, Hash-based, and Tag-only file formats.
 * PRimarily to obtain Tags from mining address files (e.g., "maddr.dat").
 * @param tag Pointer to address tag to read into
 * @param filename Filename of file to read from
 * @return (int) value representing read result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 * @exception errno=EMCM_FILEDATA if file format is unsupported
 */
int addr_tag_readfile(void *tag, const char *filename)
{
   FILE *fp;
   long long llen;
   size_t count;

   /* open file for reading */
   fp = fopen(filename, "rb");
   if (fp == NULL) return VERROR;

   /* determine provided file format per length */
   if (fseek64(fp, 0LL, SEEK_END) != 0) goto ERROR_CLEANUP;
   llen = ftell64(fp);
   if (llen == (-1)) goto ERROR_CLEANUP;

   switch (llen) {
      case ADDR_TAG_LEN:
         /* read directly into output */
         count = fread(tag, ADDR_TAG_LEN, 1, fp);
         if (count != 1) goto ERROR_CLEANUP;
         break;
      case ADDR_LEN:
         /* read directly into output, from tag offset */
         if (fseek64(fp, ADDR_TAG_OFF, SEEK_SET) != 0) goto ERROR_CLEANUP;
         count = fread(tag, ADDR_TAG_LEN, 1, fp);
         if (count != 1) goto ERROR_CLEANUP;
         break;
      case WOTS_ADDR_LEN: {
         /* local block scope ({}) to contain local vars */
         word8 addr[ADDR_LEN];
         word8 wots[WOTS_ADDR_LEN];
         /* read into local temp */
         if (fseek64(fp, 0LL, SEEK_SET) != 0) goto ERROR_CLEANUP;
         count = fread(wots, WOTS_ADDR_LEN, 1, fp);
         if (count != 1) goto ERROR_CLEANUP;
         /* convert wots, and copy into output */
         addr_from_wots(wots, addr);
         memcpy(tag, ADDR_TAG_PTR(addr), ADDR_TAG_LEN);
         break;
      }  /* end (scoped) case WOTS_ADDR_LEN */
      default:
         /* unsupported file format */
         set_errno(EMCM_FILEDATA);
         goto ERROR_CLEANUP;
   }  /* end switch() */

   /* success */
   fclose(fp);
   return VEOK;

ERROR_CLEANUP:
   fclose(fp);
   return VERROR;
}  /* end addr_tag_readfile() */

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
 * Open ledger file for internal operations. Ledger file is read-only.
 * @param lefile Filename of the ledger file to open
 * @return (int) value representing open result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 */
int le_open(const char *lefile)
{
   FILE *fp;
   long long offset;

   /* Already open? */
   if (Lefp) {
      if (strcmp(lefile, Lefile) == 0) return VEOK;
      /* ... no, opening different ledger */
   }

   /* open ledger and seek to EOF */
   fp = fopen(lefile, "rb");
   if (fp == NULL) return VERROR;
   if (fseek64(fp, 0LL, SEEK_END) != 0) {
      goto ERROR_CLEANUP;
   }

   /* determine file size (via position) and check validity */
   offset = ftell64(fp);
   if (offset == (-1)) goto ERROR_CLEANUP;
   if ((size_t) offset < sizeof(LENTRY) || offset % sizeof(LENTRY) != 0) {
      set_errno(EMCM_FILEDATA);
      goto ERROR_CLEANUP;
   }

   /* replace existing ledger */
   le_close();
   Lefp = fp;
   /* update static ledger unit values */
   Nledger = offset / sizeof(LENTRY);
   strncpy(Lefile, lefile, sizeof(Lefile) - 1);

   return VEOK;

   /* cleanup / error handling */
ERROR_CLEANUP:
   fclose(fp);

   return VERROR;
}  /* end le_open() */

/**
 * Close the internal ledger file. No operation if ledger was not opened
 * with le_open().
 */
void le_close(void)
{
   if(Lefp == NULL) return;
   fclose(Lefp);
   Lefp = NULL;
   Nledger = 0;
}

/**
 * Binary search for ledger address. If found, le is filled with the found
 * ledger entry data. Ledger must have been opened with le_open().
 * @param addr Address data to search for
 * @param le Pointer to place found ledger entry
 * @param len Length of address data to search
 * @return (int) value representing found result
 * @retval 0 on not found; check errno for details
 * @retval 1 on found; check le pointer for ledger data
 * @exception errno=EMCM_LECLOSED if ledger is not open
 * @exception errno=EINVAL if address or le is NULL, or len is zero
 * @exception errno=0 if address is not found
*/
int le_find(const word8 *addr, LENTRY *le, word16 len)
{
   long long mid, hi, low;
   int cond;

   /* ledger must be open */
   if (Lefp == NULL) {
      set_errno(EMCM_LECLOSED);
      return 0;
   }

   /* check address pointer and non-zero search length */
   if (addr == NULL || le == NULL || len == 0) {
      set_errno(EINVAL);
      return 0;
   }

   /* clamp search length to TXADDRLEN */
   if (len > TXADDRLEN) len = TXADDRLEN;

   low = 0;
   hi = Nledger - 1;

   while(low <= hi) {
      mid = (hi + low) / 2;
      if (fseek64(Lefp, mid * sizeof(LENTRY), SEEK_SET) != 0) return 0;
      if (fread(le, sizeof(LENTRY), 1, Lefp) != 1) {
         if (!ferror(Lefp)) set_errno(EMCM_EOF);
         return 0;
      }
      cond = memcmp(addr, le->addr, len);
      if(cond == 0) return 1;  /* found target addr */
      if(cond < 0) hi = mid - 1; else low = mid + 1;
   }  /* end while */

   /* indicate successful operation in the absence of a result */
   set_errno(0);

   return 0;  /* not found */
}  /* end le_find() */

/**
 * Extract a ledger from a neo-genesis block. Checks sort.
 * @param ngfile Filename of the neo-genesis block
 * @param lefile Filename of the ledger
 * @return (int) value representing extraction result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int le_extract(const char *ngfile, const char *lefile)
{
   LENTRY_W lew;           /* buffer for WOTS+ ledger entries */
   LENTRY le;              /* buffer for Hashed ledger entries */
   NGHEADER ngh;           /* buffer for neo-genesis header */
   FILE *fp, *lfp;         /* FILE pointers */
   word8 paddr[TXADDRLEN]; /* ledger address sort check */
   word64 lbytes;
   size_t j, lcount;
   word32 hdrlen;

   /* open files */
   fp = fopen(ngfile, "rb");
   if (fp == NULL) return VERROR;
   lfp = fopen(lefile, "wb");
   if (lfp == NULL) {
      fclose(fp);
      return VERROR;
   }
   /* read hdrlen and determine NGHEADER or LEGACY processing */
   if (fread(&hdrlen, 4, 1, fp) != 1) goto RDERR_CLEANUP;
   if (hdrlen == sizeof(NGHEADER)) {
      pdebug("Processing NGHEADER neo-genesis block...\n");
      /* read/check neo-genesis block header */
      if (fseek64(fp, 0LL, SEEK_SET) != 0) goto ERROR_CLEANUP;
      if (fread(&ngh, sizeof(NGHEADER), 1, fp) != 1) goto RDERR_CLEANUP;
      if (get32(ngh.hdrlen) != sizeof(NGHEADER)) {
         set_errno(EMCM_HDRLEN);
         goto ERROR_CLEANUP;
      }
      put64(&lbytes, ngh.lbytes);
      if (lbytes < sizeof(LENTRY) || lbytes % sizeof(LENTRY) != 0) {
         set_errno(EMCM_FILEDATA);
         goto ERROR_CLEANUP;
      }
      /* process ledger data from fp, check sort, write to lfp */
      for (lcount = lbytes / sizeof(LENTRY), j = 0; j < lcount; j++) {
         if (fread(&le, sizeof(LENTRY), 1, fp) != 1) goto RDERR_CLEANUP;
         /* check ledger sort */
         if (j > 0 && addr_compare(le.addr, paddr) <= 0) {
            set_errno(EMCM_LESORT);
            goto ERROR_CLEANUP;
         }
         /* store entry for comparison */
         memcpy(paddr, le.addr, sizeof(le.addr));
         /* write hashed ledger entries to ledger file */
         if (fwrite(&le, sizeof(LENTRY), 1, lfp) != 1) goto ERROR_CLEANUP;
      }  /* end for() */
      /* close files */
      fclose(lfp);
      fclose(fp);
   } else {
      hdrlen -= 4;
      if (hdrlen % sizeof(LENTRY_W) == 0) {
         pdebug("Processing LEGACY neo-genesis block...\n");
         /* LEGACY (NEO)GENESIS BLOCK PROCESSING... */
         word8 waddr[TXWOTSLEN]; /* ledger address sort check */
         /* process ledger data from fp, check sort, write to lfp */
         lcount = hdrlen / sizeof(LENTRY_W);
         for (j = 0; j < lcount; j++) {
            if (fread(&lew, sizeof(LENTRY_W), 1, fp) != 1) {
               goto RDERR_CLEANUP;
            }
            /* check ledger sort */
            if (j > 0 && memcmp(lew.addr, waddr, TXWOTSLEN) <= 0) {
               set_errno(EMCM_LESORT);
               goto ERROR_CLEANUP;
            }
            /* store entry for comparison */
            memcpy(waddr, lew.addr, TXWOTSLEN);
            /* convert WOTS+ to hash -- copy tag and balance */
            addr_convert(lew.addr, le.addr);
            put64(le.balance, lew.balance);
            /* write hashed ledger entries to ledger file */
            if (fwrite(&le, sizeof(LENTRY), 1, lfp) != 1) goto ERROR_CLEANUP;
         }  /* end for() */
         /* close files */
         fclose(lfp);
         fclose(fp);
         /* sort the resulting ledger file */
         if (filesort(lefile, sizeof(LENTRY), LEBUFSZ, addr_compare) != 0) {
            return VERROR;
         }
         /* re-open for duplicate check */
         lfp = fopen(lefile, "rb");
         if (lfp == NULL) return VERROR;
         /* (re)process ledger entries to check sort */
         for (j = 0; j < lcount; j++) {
            if (fread(&le, sizeof(LENTRY), 1, lfp) != 1) {
               goto RDERR_CLEANUP;
            }
            /* check ledger sort, tags are ascending and unique */
            if (j > 0 && memcmp(le.addr, paddr, ADDR_TAG_LEN) <= 0) {
               set_errno(EMCM_LESORT);
               goto ERROR_CLEANUP;
            }
            /* store entry for comparison */
            memcpy(paddr, le.addr, ADDR_LEN);
         }  /* end for() */
         fclose(lfp);
      } else {
         set_errno(EMCM_HDRLEN);
         goto ERROR_CLEANUP;
      }
   }

   /* ledger extracted */
   return VEOK;

   /* cleanup / error handling */
RDERR_CLEANUP:
   if (!ferror(fp)) {
      set_errno(EMCM_EOF);
   }
ERROR_CLEANUP:
   fclose(lfp);
   fclose(fp);

   return VERROR;
}  /* end le_extract() */

/**
 * Update the ledger by applying deltas from a ledger transaction file.
 * Ledger transaction file is sorted by addr+code, '-' comes before 'A'.
 * Ledger file is kept sorted on addr. Ledger file must have been opened
 * with le_open().
 * @param ltfname Filename of the Ledger transaction (deltas) file
 * @return (int) value representing the update result
 * @retval VEBAD2 on malicious; check errno for details
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 */
int le_update(const char *ltfname)
{
   LENTRY le_hold;         /* for ledger entry hold data */
   LENTRY le, le_prev;     /* for ledger entry and sequence check data */
   LTRAN lt, lt_prev;      /* for ledger tran and sequence check data */
   FILE *fp, *lefp, *ltfp; /* output, ledger, and ltran file pointers */
   word8 hold, empty;
   int compare, ecode;

   /* ledger must be open */
   if (Lefp == NULL) {
      set_errno(EMCM_LECLOSED);
      return VERROR;
   }

   /* sort the ledger transaction file */
   ecode = filesort(ltfname, sizeof(LTRAN), LEBUFSZ, lt_compare);
   if (ecode != 0) return VERROR;

   /* init for error handling */
   fp = ltfp = NULL;
   lefp = Lefp;

   /* fseek and read initial ledger entry */
   if (fseek64(lefp, 0LL, SEEK_SET) != 0) return VERROR;
   if (fread(&le, sizeof(LENTRY), 1, lefp) != 1) {
      if (!ferror(lefp)) set_errno(EMCM_EOF);
      return VERROR;
   }
   /* open and read initial ledger transaction */
   ltfp = fopen(ltfname, "rb");
   if (ltfp == NULL) return VERROR;
   if (fread(&lt, sizeof(LTRAN), 1, ltfp) != 1) {
      if (!ferror(ltfp)) set_errno(EMCM_EOF);
      goto ERROR_CLEANUP;
   }

   /* generate temporary filename and open as new ledger */
   fp = fopen("ledger.update", "wb");
   if (fp == NULL) goto ERROR_CLEANUP;

   /* iterate through files while either files are NOT EOF */
   for (hold = 0, empty = 1; lefp != NULL || ltfp != NULL; ) {

      /* check ledger transaction file is open for processing */
      if (ltfp != NULL) {
         /* only perform initial comparison on ledger entry file */
         if (lefp != NULL) compare = addr_tag_compare(le.addr, lt.addr);

         /* if ledger entry compares AFTER ledger transaction, OR
          * if ledger entry file is EOF... */
         if (compare > 0 || lefp == NULL) {
            /* ... this is a "brand new" destination/address */
            /* assume malicious intent where non-CREDIT ('A') code here */
            if (lt.trancode[0] != 'A') {
               set_errno(EMCM_LTCREDIT);
               goto DROP_CLEANUP;
            }
            /* hold ledger entry while associated file is open */
            if (lefp != NULL) {
               memcpy(&le_hold, &le, sizeof(LENTRY));
               hold = 1;
            }
            /* clear ledger entry data */
            memset(&le, 0, sizeof(LENTRY));
            /* convert address from implicit tag */
            addr_from_implicit(ADDR_TAG_PTR(lt.addr), le.addr);
            /* set compare for ledger transaction processing */
            compare = 0;
         }  /* end if (compare > 0 || lefp == NULL) */

         /* while ledger entry compares EQUAL TO ledger transaction... */
         while (compare == 0) {
            /* apply ledger transaction */
            switch (lt.trancode[0]) {
               case 'H':
                  /* transaction REHASH operation */
                  memcpy(ADDR_HASH_PTR(le.addr), ADDR_HASH_PTR(lt.addr), ADDR_HASH_LEN);
                  /* fallthrough */
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
                  /* ... assume malicious intent where balance != amount */
                  if (cmp64(le.balance, lt.amount) != 0) {
                     set_errno(EMCM_LTDEBIT);
                     goto DROP_CLEANUP;
                  }
                  memset(le.balance, 0, sizeof(le.balance));
                  break;
               default:
                  /* invalid transaction operation */
                  set_errno(EMCM_LTCODE);
                  goto ERROR_CLEANUP;
            }
            /* read next ledger transaction */
            memcpy(&lt_prev, &lt, sizeof(LTRAN));
            if (fread(&lt, sizeof(LTRAN), 1, ltfp) != 1) {
               if (ferror(ltfp)) goto ERROR_CLEANUP;
               /* EOF -- cleanup, break inner loop */
               fclose(ltfp);
               ltfp = NULL;
               break;
            }
            /* check sort -- MUST BE ascending, ALLOW duplicates */
            if (lt_compare(&lt_prev, &lt) > 0) {
               set_errno(EMCM_LTSORT);
               goto ERROR_CLEANUP;
            }
            /* recompare latest ledger transaction */
            compare = addr_tag_compare(le.addr, lt.addr);
         }  /* end while (compare == 0) */
      }  /* end if (ltfp != NULL) */

      /* if lendger entry compares BEFORE ledger transaction, OR
       * ledger transaction file is EOF... */
      if (compare < 0 || ltfp == NULL) {
         /* write ledger entry to output */
         if (fwrite(&le, sizeof(LENTRY), 1, fp) != 1) goto ERROR_CLEANUP;
         /* flag output not empty */
         empty = 0;
         /* if ledger entry file open... */
         if (lefp != NULL) {
            /* copy ledger hold to ledger entry, OR... */
            if (hold) {
               memcpy(&le, &le_hold, sizeof(LENTRY));
               hold = 0;
               continue;
            }
            /* read next ledger transaction, AND... */
            memcpy(&le_prev, &le, sizeof(LENTRY));
            if (fread(&le, sizeof(LENTRY), 1, lefp) != 1) {
               if (ferror(lefp)) goto ERROR_CLEANUP;
               /* EOF -- DO NOT CLOSE, just decouple from Lefp */
               /* fclose(lefp); */
               lefp = NULL;
               continue;
            }
            /* check sort -- MUST BE ascending, NO duplicates */
            if (addr_compare(le_prev.addr, le.addr) >= 0) {
               set_errno(EMCM_LESORT);
               goto ERROR_CLEANUP;
            }
         }  /* end  if (lefp != NULL) */
      }  /* end if (compare < 0... */
   }  /* end while () */
   /* empty ledger check */
   if (empty) {
      set_errno(EMCM_LEEMPTY);
      goto ERROR_CLEANUP;
   }

   /* cleanup -- ltfp already closed */
   fclose(fp);

   /* close / replace ledger */
   le_close();
   remove(Lefile);
   if (rename("ledger.update", Lefile) != 0) return VERROR;

   /* return result of reopen ledger */
   return le_open(Lefile);

   /* cleanup / error handling */
ERROR_CLEANUP:
   ecode = VERROR;
   goto CLEANUP;
DROP_CLEANUP:
   ecode = VEBAD2;
CLEANUP:
   if (ltfp) fclose(ltfp);
   if (fp) {
      fclose(fp);
      remove("ledger.update");
   }

   return ecode;
}  /* end le_update() */

/**
 * Tag comparison function.
 * @param a Pointer to tag to compare
 * @param b Pointer to tag to compare against
 * @returns (int) value representing result
 * @retval >0 if tag A is greater than tag B
 * @retval <0 if tag A is less than tag B
 * @retval 0 if tags are equal
 */
int tag_compare(const void *a, const void *b)
{
   return memcmp(a, b, ADDR_TAG_LEN);
}  /* end tag_cmp() */

/**
 * Equality check for tags. Implements an efficient 20-byte check,
 * inspired by the legacy 12-byte check.
 * @param a Pointer to tag to check
 * @param b Pointer to tag to check against
 * @returns 1 if tags match, else 0
 */
int tag_equal(const void *a, const void *b)
{
   return equality_check_20bytes(a, b);
}  /* end tag_equal() */

/* end include guard */
#endif
