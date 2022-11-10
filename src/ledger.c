/**
 * @headerfile ledger.h <ledger.h>
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_LEDGER_C
#define MOCHIMO_LEDGER_C


#include "ledger.h"

/* internal support */
#include "sha256.h"
#include "error.h"

/* external support */
#include "extstring.h"
#include "extmath.h"
#include "extlib.h"

/* Log-structured Merge Tree Ledger array */
static struct {
   LENTRY *lmap;  /* pointer to memory mapped ledger file */
   TAGIDX *tmap;  /* pointer to memory mapped tag index file */
   size_t count;  /* number of ledger entries (lmap) */
   size_t tags;   /* number of tags (tmap) */
} Ledger[LEDEPTHMAX];
/* Shared read exclusive write access to Ledger lock */
static RWLock Lelock = RWLOCK_INITIALIZER;
/* Next depth of log-structured merge tree ledger array */
static int Leidx;

/** Last day (bnum) of sanctuary (configurable option) */
word32 Lastday_opt = 0;
/** Sanctuary fee (nanoMCM) (configurable option) */
word32 Sanctuary_opt = 0;
/** Ledger filename (configurable option) */
char *Lefname_opt = "ledger.dat";
/** Tag index filename (configurable option) */
char *Tifname_opt = "tagidx.dat";

static int fread_clean(void *buf, size_t size, size_t count, FILE **fp)
{
   if (fread(buf, size, count, *fp) != count) {
      if (ferror(*fp)) return VERROR;
      fclose(*fp);
      *fp = NULL;
      return EOF;
   }

   return VEOK;
}  /* end fread_clean() */

/**
 * Obtain the recommended compression depth of the current Ledger tree.
 * @returns (int) value representing the recommended compression depth
 * @retval >=0 on success
 * @retval EOF on error; check errno for details
 * @exception errno=EMCMLENOTAVAIL No ledger to recommend depth
*/
int auto_compression_depth(void)
{
   word64 bias, n;
   int depth, max;

   /* check ledger */
   if (Leidx == 0) goto FAIL_LENOTAVAIL;
   if (Leidx == LEDEPTHMAX) return 0;
   /* determine optimal depth of compression */
   for (depth = 0, max = Leidx - 1; depth < max; depth++) {
      n = WORD64_C(1) << (WORD64_C(1) << (depth + 1));
      bias = (word64) Ledger[depth + 1].count * n;
      if (Ledger[depth].count < bias) break;
   }

   return depth;

FAIL_LENOTAVAIL: set_errno(EMCMLENOTAVAIL); return EOF;
}  /* end auto_compression_depth() */

/**
 * Append a Ledger and Tagidx file to the next Ledger depth.
 * Supplied files are consumed (renamed) by this process.
 * @param lfname Filename of ledger
 * @param tfname Filename of tag index
 * @return (int) value representing the operation result
 * @retval VEOK on success
 * @retval VERROR on error; check errno for details
 * @exception errno=EINVAL A function parameter is invalid
 * @exception errno=EMCMLEDEPTH Ledger depth limit reached, compress failed
*/
int le_append(const char *lfname, const char *tfname)
{
   static const int prot = PROT_READ; /* | PROT_WRITE; */
   static const int flags = MAP_SHARED;

   long long len;
   FILE *fp;
   int depth, count;
   char fname[FILENAME_MAX];

   /* lfname must be a valid parameter */
   if (lfname == NULL) goto FAIL_INVAL;

   /* check current ledger depth */
   if (Leidx >= LEDEPTHMAX) {
      /* obtain recommended compression depth/count */
      depth = auto_compression_depth(0);
      count = Leidx - depth;
      /* perform compression and splice result */
      snprintf(fname, FILENAME_MAX, "%s.co", lfname);
      if (le_compress(fname, depth, count) != VEOK) return VERROR;
      if (le_splice(fname, depth, count) != VEOK) return VERROR;
      /* ensure compression was effective */
      if (Leidx >= LEDEPTHMAX) goto FAIL_DEPTH;
   }

   /* build tag index for ledger append data, if none was supplied */
   if (tfname == NULL) {
      if (tag_extract(lfname, Tifname_opt) != VEOK) return VERROR;
      tfname = Tifname_opt;
   }

   /* move ledger file to appropriate depth */
   snprintf(fname, FILENAME_MAX, "%s.%d", Lefname_opt, Leidx);
   if (rename(lfname, fname) != 0) return VERROR;
   /* open ledger read-only, seek to END, get length of file */
   if ((fp = fopen(fname, "rb")) == NULL) return VERROR;
   if (fseek64(fp, 0LL, SEEK_END) != 0) goto FAIL_IO;
   if ((len = ftell64(fp)) < 0LL) goto FAIL_IO;
   /* create memory mapped ledger file at Leidx */
   Ledger[Leidx].count = (size_t) (len / sizeof(LENTRY));
   Ledger[Leidx].lmap =
      mmap(NULL, (size_t) len, prot, flags, fileno(fp), 0);
   if (Ledger[Leidx].lmap == MAP_FAILED) goto FAIL_IO;
   fclose(fp);

   /* move tag index file to appropriate Leidx */
   snprintf(fname, FILENAME_MAX, "%s.%d", Tifname_opt, Leidx);
   if (rename(tfname, fname) != 0) return VERROR;
   /* open ledger read-only, seek to END, get length of file */
   if ((fp = fopen(fname, "rb")) == NULL) return VERROR;
   if (fseek64(fp, 0LL, SEEK_END) != 0) goto FAIL_IO;
   if ((len = ftell64(fp)) < 0LL) goto FAIL_IO;
   /* create memory mapped ledger file at Leidx */
   Ledger[Leidx].tags = (size_t) (len / sizeof(TAGIDX));
   if (Ledger[Leidx].tags) {
      Ledger[Leidx].tmap =
         mmap(NULL, (size_t) len, prot, flags, fileno(fp), 0);
      if (Ledger[Leidx].tmap == MAP_FAILED) goto FAIL_IO;
   } else Ledger[Leidx].tmap = NULL;
   fclose(fp);

   /* increment ledger index */
   Leidx++;

   /* success */
   return VEOK;

/* error handling -- cleanup */
FAIL_INVAL: set_errno(EINVAL); return VERROR;
FAIL_DEPTH: set_errno(EMCMLEDEPTH); return VERROR;
FAIL_IO: fclose(fp); return VERROR;
}  /* end le_append() */

/**
 * Close LSMT Ledger and Tagidx to a specified depth (inclusive).
 * @code le_close(0); @endcode ... closes the entire tree.
 * @param depth Depth at which to close the Ledger (inclusive)
*/
void le_close(int depth)
{
   /* close ledger tree to the specified depth */
   while (depth < Leidx) {
      Leidx--;
      /* close/clear Ledger entries */
      munmap(Ledger[Leidx].lmap, Ledger[Leidx].count * sizeof(LENTRY));
      Ledger[Leidx].lmap = NULL;
      Ledger[Leidx].count = 0;
      /* close/clear Tag index */
      munmap(Ledger[Leidx].tmap, Ledger[Leidx].tags * sizeof(TAGIDX));
      Ledger[Leidx].tmap = NULL;
      Ledger[Leidx].tags = 0;
   }
}  /* end le_close() */

/**
 * PK+ Ledger entry comparison function.
 * Compares ONLY the WOTS+ address of the ledger entry.
 * @param a Pointer to data to compare
 * @param b Pointer to data to compare against
 * @return (int) value representing comparison result
 * @retval 0 @a a is equal to @a b
 * @retval <0 @a a is less than @a b
 * @retval >0 @a a is greater than @a b
*/
int le_cmpw(const void *a, const void *b)
{
   return memcmp(a, b, TXWOTSLEN);
}  /* end le_cmpw() */

/**
 * Public Ledger entry comparison function.
 * Compares ONLY the Public address of the ledger entry.
 * @param a Pointer to data to compare
 * @param b Pointer to data to compare against
 * @return (int) value representing comparison result
 * @retval 0 @a a is equal to @a b
 * @retval <0 @a a is less than @a b
 * @retval >0 @a a is greater than @a b
*/
int le_cmp(const void *a, const void *b)
{
   return memcmp(a, b, TXADDRLEN);
}  /* end le_cmp() */

/**
 * Perform Ledger tree compression of the specified file depths.
 * Zero balance ledger entries are ommitted when target depth is zero.
 * NOTE: Compression count is limited to the value defined by LEDEPTHMAX.
 * @param filename Filename of final compressed output
 * @param depth The depth at which to start compression
 * @param count The number of depths to compress
 * @return (int) value representing the compression result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success, or nothing to compress
 * @exception errno=EINVAL Invalid function parameter
 * @exception errno=EMCMLEDEPTH @a count parameter exceeds max ledger depth
*/
int le_compress(const char *filename, int depth, int count)
{
   LENTRY lep[LEDEPTHMAX];
   FILE *fp, *fpp[LEDEPTHMAX];
   int cond, i, min;
   char fname[FILENAME_MAX];

   /* check basename validity */
   if (filename == NULL) goto FAIL_INVAL;
   /* check compression limit and minimum */
   if (count > LEDEPTHMAX) goto FAIL_INVAL;
   if (count < 2) goto SUCCESS;

   /* clear initial fp array */
   for (i = 0; i < count; fpp[i++] = NULL);

   /* open destination file */
   fp = fopen(filename, "wb");
   if (fp == NULL) goto FAIL_IO;
   /* open (all) files for merge */
   for (min = (-1), i = 0; i < count; i++) {
      snprintf(fname, FILENAME_MAX, "%s.%d", Lefname_opt, depth + i);
      fpp[i] = fopen(fname, "rb");
      if (fpp[i] == NULL) goto FAIL_IO;
      /* read initial data into buffers */
      if (fread_clean(&lep[i], sizeof(*lep), 1, &fpp[i]) > VEOK) {
         goto FAIL_IO;
      } else min = i;
   }

   /* perform selective merge sort on files */
   for ( ; min >= 0; ) {
      min = (-1);
      for (i = count - 1; i >= 0; i--) {
         /* !IMPORTANT! Loop iterates in reverse to acquire the
          * latest ledger value FIRST; ignoring old duplicates
         */
         if (fpp[i] == NULL) continue;
         if (min < 0) min = i;
         else {
            cond = le_cmp(&lep[i], &lep[min]);
            if (cond < 0) min = i;
            else if (cond == 0) {
               /* skip duplicate (OLD) address, read for next loop */
               if (fread_clean(&lep[i], sizeof(*lep), 1, &fpp[i]) > VEOK) {
                  goto FAIL_IO;
               }
            }
         }
      }  /* end for (i = count - 1; ... */
      /* ensure minimum value was found -- write value and read another */
      if (min >= 0) {
         /* when target depth (from) is 0, skip writing zero balances */
         if (depth || !iszero(lep[min].balance, 8)) {
            if (fwrite(&lep[min], sizeof(*lep), 1, fp) != 1) goto FAIL_IO;
         }
         /* read next value in */
         if (fread_clean(&lep[min], sizeof(*lep), 1, &fpp[min]) > VEOK) {
            goto FAIL_IO;
         }
      }
   }  /* end for ( ; min >= 0; ... */
   /* cleanup -- sources are closed in loop */
   fclose(fp);

SUCCESS:
   /* success */
   return 0;

/* error handling cleanup */
FAIL_INVAL: set_errno(EINVAL); return VERROR;
FAIL_IO:
   if (fp) fclose(fp);
   for (i = 0; i < count; i++) {
      if (fpp[i]) fclose(fpp[i]);
   }
   return VERROR;
}  /* end le_compress() */

/**
 * Convert a WOTS+ address to a Hashed address. Also copies tags.
 * @param hash Pointer to 64 byte Hashed address
 * @param wots Pointer to 2208 byte WOTS+ address
 */
void le_convert(void *hash, void *wots)
{
   sha256(wots, TXWOTSLEN, hash);
   memcpy(
      (char *) hash + (TXADDRLEN - HASHLEN),
      (char *) wots + (TXWOTSLEN - HASHLEN),
      HASHLEN
   );
}  /* end le_convert() */

/**
 * Delete Ledger to a specified depth (inclusive).
 * @code le_delete(0); @endcode ... deletes the entire tree.
 * @param depth Depth at which to close the Ledger (inclusive)
*/
void le_delete(int depth)
{
   char fname[FILENAME_MAX];

   /* delete ledger tree to the specified depth */
   while (depth < Leidx) {
      Leidx--;
      /* close/clear/delete Ledger entries */
      snprintf(fname, FILENAME_MAX, "%s.%d", Lefname_opt, Leidx);
      munmap(Ledger[Leidx].lmap, Ledger[Leidx].count * sizeof(LENTRY));
      Ledger[Leidx].lmap = NULL;
      Ledger[Leidx].count = 0;
      remove(fname);
      /* close/clear/delete Tag index */
      snprintf(fname, FILENAME_MAX, "%s.%d", Tifname_opt, Leidx);
      munmap(Ledger[Leidx].tmap, Ledger[Leidx].tags * sizeof(TAGIDX));
      Ledger[Leidx].tmap = NULL;
      Ledger[Leidx].tags = 0;
      remove(fname);
   }
}  /* end le_delete() */

/**
 * Extract a ledger from a neo-genesis block.
 * Compatible with both WOTS+ and Hashed ledgers.
 * Checks address sort of neo-genesis block while processing.
 * Extracted ledger is splice over existing ledger tree (overwrite).
 * @param ngfname Filename of the neo-genesis block
 * @return (int) value representing extraction result
 * @retval VEBAD on invalid; check errno for details
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int le_extract(const char *ngfname)
{
   static const char *lfname = "ledger.ex";

   LENTRY le;              /* buffer for Hashed ledger entries */
   LENTRY_W lew;           /* buffer for WOTS+ ledger entries */
   NGHEADER ngh;           /* buffer for neo-genesis header */
   FILE *fp, *lfp;         /* FILE pointers */
   long long remain;
   word32 hdrlen;          /* buffer for block header length */
   word8 paddr[TXWOTSLEN]; /* ledger address sort check */
   int first;

   /* open files */
   fp = fopen(ngfname, "rb");
   lfp = fopen(lfname, "wb");
   if (fp == NULL || lfp == NULL) goto FAIL_IO;
   /* read neo-genesis header and hdrlen value -- check ledger type */
   if (fread(&ngh, sizeof(ngh), 1, fp) != 1) goto FAIL_IO;
   hdrlen = get32(ngh.hdrlen);
   /* check header length for differing ledger types */
   if (hdrlen == sizeof(ngh)) {
      /* seek to start of Hashed ledger */
      if (fseek(fp, (long) hdrlen, SEEK_SET)) goto FAIL_IO;
      /* check ledger size and alignment */
      put64(&remain, ngh.lbytes);
      if (remain == 0 || remain % sizeof(le)) goto FAIL_HDRLEN;
      /* read the ledger from fp, copy it to lfp */
      for (first = 1; remain > 0LL; remain -= sizeof(le), first = 0) {
         if (fread(&le, sizeof(le), 1, fp) != 1) goto FAIL_IO;
         /* check ledger sort */
         if (!first && le_cmp(le.addr, paddr) <= 0) goto FAIL_SORT;
         memcpy(paddr, le.addr, sizeof(le.addr));
         /* write hashed ledger entries to ledger file */
         if (fwrite(&le, sizeof(le), 1, lfp) != 1) goto FAIL_IO;
      }  /* end for() */
      /* close files */
      fclose(lfp);
      fclose(fp);
   } else {
      /* seek to start of WOTS+ ledger */
      if (fseek(fp, (long) sizeof(hdrlen), SEEK_SET)) goto FAIL_IO;
      /* check ledger size and alignment */
      remain = (long long) hdrlen - sizeof(hdrlen);
      if (remain == 0 || remain % sizeof(lew)) goto FAIL_HDRLEN;
      /* convert the WOTS+ ledger from fp to hashed, and copy it to lfp */
      for (first = 1; remain > 0LL; remain -= sizeof(lew), first = 0) {
         if (fread(&lew, sizeof(lew), 1, fp) != 1) goto FAIL_IO;
         /* check ledger sort */
         if (!first && le_cmpw(lew.addr, paddr) <= 0) goto FAIL_SORT;
         memcpy(paddr, lew.addr, sizeof(lew.addr));
         /* convert WOTS+ to Hashed address */
         le_convert(le.addr, lew.addr);
         put64(le.balance, lew.balance);
         /* write hashed ledger entries to ledger file */
         if (fwrite(&le, sizeof(le), 1, lfp) != 1) goto FAIL_IO;
      }  /* end for() */
      /* close files */
      fclose(lfp);
      fclose(fp);
      /* ledger output requires a sort pass, due to WOTS+ conversion */
      if (filesort(lfname, sizeof(le), LEBUFSZ, le_cmp) != 0) {
         return VERROR;
      }
   }

   /* extraction success -- splice the ledger for safe overwrite */
   return le_splice(lfname, 0, Leidx);

/* error handling */
FAIL_SORT: set_errno(EMCM_LE_SORT); goto FAIL_IO;
FAIL_HDRLEN: set_errno(EMCM_HDRLEN);
FAIL_IO:
   if (lfp) fclose(lfp);
   if (fp) fclose(fp);
   return VERROR;
}  /* end le_extract() */

/**
 * Find a ledger entry by Hashed address.
 * @param addr Pointer to address to search for
 * @return (void *) pointer to ledger entry, or NULL.
 * Check @a errno for more details.
 * @exception errno=0 No errors, address was not found in Ledger
 * @exception errno=EINVAL The @a addr parameter is NULL
 * @exception errno=EMCMLENOTAVAIL The internal Ledger is NULL
*/
LENTRY *le_find(void *addr)
{
   LENTRY *found;
   int i;

   set_errno(0);

   /* sanity checks */
   if (addr == NULL) goto FAIL_INVAL;
   if (Leidx == 0) goto FAIL_LENOTAVAIL;

   /* acquire shared read access */
   if (rwlock_rdlock(&Lelock)) return NULL;

   /* walk ledger nodes searching for data -- search latest first */
   for (found = NULL, i = Leidx - 1; i >= 0; i--) {
      found = bsearch(
         addr, Ledger[i].lmap, Ledger[i].count, sizeof(*found), le_cmp);
      if (found) {
         /* nullify zero balance results */
         if (iszero(found->balance, 8)) {
            found = NULL;
         }
         break;
      }
   }

   /* release shared read access */
   if (rwlock_rdunlock(&Lelock)) return NULL;

   /* done */
   return found;

FAIL_INVAL: set_errno(EINVAL); return NULL;
FAIL_LENOTAVAIL: set_errno(EMCMLENOTAVAIL); return NULL;
}  /* end le_find() */

/**
 * Find a ledger entry by WOTS+ address. The WOTS+ address is first
 * converted into it's Hashed (pk) variant for search with le_find().
 * @param wots Pointer to WOTS+ address
 * @return (void *) pointer to ledger entry, or NULL.
 * Check @a errno for more details.
 * @exception errno=0 No errors, address was not found in Ledger
 * @exception errno=EINVAL The @a addr parameter is NULL
 * @exception errno=EMCMLENOTAVAIL The internal Ledger is NULL
*/
LENTRY *le_findw(void *wots)
{
   word8 addr[TXADDRLEN];

   set_errno(0);

   /* sanity checks */
   if (wots == NULL) { set_errno(EINVAL); return NULL; }
   /* convert wots address to public key address */
   le_convert(addr, wots);

   return le_find(addr);
}  /* end le_findw() */

/**
 * Activate the Sanctuary Protocol to renew the ledger.
 * @param fee Pointer to mining fee threshold
 * @return (int) value representing operation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int le_renew(void *fee)
{
   LENTRY le;
   FILE *fp, *fpout;
   word32 in, out;
   word32 sanctuary[2];
   char lfname[FILENAME_MAX];

   if (Sanctuary_opt == 0 || Leidx == 0) return 0;  /* success */

   sanctuary[0] = Sanctuary_opt;
   sanctuary[1] = 0;

   plog("Lastday 0x%0x.  Carousel begins...", Lastday_opt);

   /* compress ledger tree of more than a single depth */
   if (Leidx > 1) {
      snprintf(lfname, FILENAME_MAX, "%s.co", Lefname_opt);
      if (le_compress(lfname, 0, Leidx) != VEOK) return VERROR;
   } else snprintf(lfname, FILENAME_MAX, "%s.0", Lefname_opt);

   /* open I/O ledger file pointers */
   fp = fopen(lfname, "rb");
   fpout = fopen(Lefname_opt, "wb");
   if (fp == NULL || fpout == NULL) goto FAIL_IO;

   /* read ledger entries, writing ONLY if threshold is met */
   for (in = out = 0; ; ) {
      if (fread(&le, sizeof(le), 1, fp) != 1) {
         if (ferror(fp)) goto FAIL_IO;
         break;  /* EOF */
      }
      in++;
      /* subtract sanctuary fee and check fee threshold */
      if (sub64(le.balance, sanctuary, le.balance)) continue;
      if (cmp64(le.balance, fee) <= 0) continue;
      if (fwrite(&le, sizeof(le), 1, fpout) != 1) goto FAIL_IO;
      out++;
   }
   /* close FILE pointers */
   fclose(fpout);
   fclose(fp);

   /* renewal success -- splice the ledger for safe overwrite */
   plog("%u citizens renewed out of %u", in - out, in);
   return le_splice(Lefname_opt, 0, Leidx);

FAIL_IO:
   if (fpout) fclose(fpout);
   if (fp) fclose(fp);
   return VERROR;
}  /* end le_renew() */

/**
 * Splice a ledger file depth into the ledger tree.
 * Replaces @a count number of depths from the ledger tree.
 * @param filename Filename of the ledger depth
 * @param depth Depth at which to splice
 * @param count Number of depths to replace
 * @returns (int) value representing the splice result
 * @retval VEOK on success
 * @retval VERROR on error; check errno for details
*/
int le_splice(const char *filename, int depth, int count)
{
   char lfname[FILENAME_MAX];
   char tfname[FILENAME_MAX];
   int idx;

   /* create tag index for splice depth */
   snprintf(tfname, FILENAME_MAX, "%s.tags", filename);
   if (tag_extract(filename, tfname) != VEOK) return VERROR;
   /* acquire exclusive ledger lock */
   if (rwlock_wrlock(&Lelock)) return VERROR;
   /* store current depth index */
   idx = Leidx;
   /* close depths we keep, delete depths we splice */
   le_close(depth + count);
   le_delete(depth);
   /* append splice depth */
   if (le_append(filename, tfname) != VEOK) goto FAIL_APPEND;
   /* append the remaining closed depths */
   for (depth += count; depth < idx; depth++) {
      snprintf(lfname, FILENAME_MAX, "%s.%d", Lefname_opt, depth);
      snprintf(tfname, FILENAME_MAX, "%s.%d", Tifname_opt, depth);
      if (le_append(lfname, tfname) != VEOK) goto FAIL_APPEND;
   }
   /* release exclusive ledger lock */
   if (rwlock_wrunlock(&Lelock)) return VERROR;

   /* success */
   return VEOK;

FAIL_APPEND:
   rwlock_wrunlock(&Lelock);
   return VERROR;
}  /* end le_splice() */

/**
 * Transpose multiple Ledger tree depths into a single depth Ledger.
 * Zero balance ledger entries are ommitted.
 * @param filename Filename of final compressed output
 * @return (int) value representing the compression result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int le_transpose(void)
{
   static char *fname = "ledger.trs";

   if (le_compress(fname, 0, Leidx) != VEOK) return VERROR;
   if (fexists(fname) && le_splice(fname, 0, Leidx) != VEOK) return VERROR;
   return VEOK;
}  /* end le_transpose() */

/**
 * Comparison function for tags.
 * @param a Pointer to tag A
 * @param b Pointer to tag B
 * @returns (int) value representing comparison result
 * @retval >0 if tag A is greater than tag B
 * @retval <0 if tag A is less than tag B
 * @retval 0 if tags are equal
*/
int tag_cmp(const void *a, const void *b)
{
   return memcmp(a, b, TXTAGLEN);
}  /* end tag_cmp() */

/**
 * Efficient 12-byte Address Tag equality check.
 * @param a Pointer to tag A
 * @param b Pointer to tag B
 * @returns 1 if tags match, else 0
*/
int tag_equal(const void *a, const void *b)
{
   return (
      ((word32 *) a)[0] == ((word32 *) b)[0] &&
      ((word32 *) a)[1] == ((word32 *) b)[1] &&
      ((word32 *) a)[2] == ((word32 *) b)[2]
   );
}  /* end tag_equal() */

/**
 * Extract a Tag index from a WOTS+ Ledger.
 * @param lfname Filename of the ledger file
 * @param tfname Filename of the tag index file
 * @return (int) value representing extraction result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int tag_extract(const char *lfname, const char *tfname)
{
   LENTRY le;        /* comparison buffer for ledger entries */
   TAGIDX ti, ti1;   /* tag index buffer pointer */
   FILE *fp, *tfp0, *tfp1; /* FILE pointers */
   long long idx;
   int result, cond;
   char tfname0[FILENAME_MAX];
   char tfname1[FILENAME_MAX];

   /* Any non-zero ledger depth may have more than one duplicate tag:
    * - one representing the tags latest address/balance, and
    * - one or more representing (old) zero balance references
    *   (accumulating with every non-zero depth compression)
    * ... therefore, simple extraction of tags will certainly lead to
    * duplicate tags and an unreliable lookup table. To overcome this
    * issue tags must first be pre-sorted by reference to the balance
    * of the underlying address pointed to by the tag index, into our
    * zero (tfp0) and non-zero (tfp1) files. These files must then be
    * sorted individually before being merged together with the later
    * (tfp1) of the files taking precedence over the former (tfp0) to
    * ensure our tag lookups are pointing to the latest relevant data
   */

   /* build pre-sort tag names */
   snprintf(tfname0, FILENAME_MAX, "%s.tmp0", tfname);
   snprintf(tfname1, FILENAME_MAX, "%s.tmp1", tfname);
   /* open file pointers */
   fp = fopen(lfname, "rb");
   tfp0 = fopen(tfname0, "wb");
   tfp1 = fopen(tfname1, "wb");
   if (fp == NULL || tfp0 == NULL || tfp1 == NULL) goto FAIL_IO;

   /* Read ledger entries, extract tags and write index pairs */
   for (idx = 0; ; idx++) {
      result = fread_clean(&le, sizeof(le), 1, &fp);
      if (result == VERROR) goto FAIL_IO;
      if (result == EOF) break;
      /* if address includes a valid tag... */
      if (ADDR_HAS_TAG(le.addr)) {
         /* ... associate index and write to file */
         put64(ti.idx, &idx);
         memcpy(ti.tag, ADDR_TAGp(le.addr), TXTAGLEN);
         if (iszero(le.balance, 8)) {
            if (fwrite(&ti, sizeof(ti), 1, tfp0) != 1) goto FAIL_IO;
         } else if (fwrite(&ti, sizeof(ti), 1, tfp1) != 1) goto FAIL_IO;
      }
   }  /* end for() */
   /* close files */
   fclose(tfp0);
   fclose(tfp1);

   /* sort both tag index files */
   if (filesort(tfname0, sizeof(ti), LEBUFSZ, tag_cmp) != 0) return VERROR;
   if (filesort(tfname1, sizeof(ti), LEBUFSZ, tag_cmp) != 0) return VERROR;

   /* (re)open file pointers */
   fp = fopen(tfname, "wb");
   tfp0 = fopen(tfname0, "rb");
   tfp1 = fopen(tfname1, "rb");
   if (fp == NULL || tfp0 == NULL || tfp1 == NULL) goto FAIL_IO;
   /* pre-fill buffers with data */
   if (fread_clean(&ti, sizeof(ti), 1, &tfp0) > VEOK) goto FAIL_IO;
   if (fread_clean(&ti1, sizeof(ti1), 1, &tfp1) > VEOK) goto FAIL_IO;

   /* perform selective merge sort on files */
   while (tfp0 || tfp1) {
      if (tfp0 == NULL || tfp1 == NULL) {
         cond = (tfp0) ? -1 : 1;
      } else do {
         /* compare tags */
         cond = tag_cmp(&ti, &ti1);
         if (cond == 0 && tfp0) {
            /* skip duplicate zero balance tags, read for next loop */
            result = fread_clean(&ti, sizeof(ti), 1, &tfp0);
            if (result == VERROR) goto FAIL_IO;
            if (result == EOF) break;
         }
      } while (cond == 0);
      if (cond >= 0) {
         if (fwrite(&ti1, sizeof(ti1), 1, fp) != 1) goto FAIL_IO;
         if (fread_clean(&ti1, sizeof(ti1), 1, &tfp1) > VEOK) goto FAIL_IO;
      } else {  /* cond < 0 */
         if (fwrite(&ti, sizeof(ti), 1, fp) != 1) goto FAIL_IO;
         if (fread_clean(&ti, sizeof(ti), 1, &tfp0) > VEOK) goto FAIL_IO;
      }
   }
   /* cleanup -- sources are closed in loop */
   remove(tfname0);
   remove(tfname1);
   fclose(fp);

   /* success */
   return VEOK;

FAIL_IO:
   if (tfp1) fclose(tfp1);
   if (tfp0) fclose(tfp0);
   if (fp) fclose(fp);
   return VERROR;
}  /* end tag_extract() */

/**
 * Find a tag and return the associated ledger entry.
 * @param tag Pointer to tag to search for
 * @return (LENTRY *) pointer to ledger entry, or NULL.
 * @exception errno=0 No errors; tag was not found or zero balance
 * @exception errno=EINVAL A function parameter is invalid (NULL)
 * @exception errno=EMCMLENOTAVAIL The internal Ledger is not available
*/
LENTRY *tag_find(void *tag)
{
   LENTRY *lep;
   TAGIDX *found;
   long long idx;
   int i;

   set_errno(0);

   /* sanity checks */
   if (tag == NULL) goto FAIL_INVAL;
   if (Leidx == 0) goto FAIL_LENOTAVAIL;

   /* acquire shared read access */
   if (rwlock_rdlock(&Lelock)) return NULL;

   /* walk ledger nodes searching for data */
   for (found = NULL, lep = NULL, i = Leidx - 1; i >= 0; i--) {
      found = bsearch(
         tag, Ledger[i].tmap, Ledger[i].tags, sizeof(*found), tag_cmp);
      if (found) {
         put64(&idx, found->idx);
         lep = &(Ledger[i].lmap[idx]);
         /* zero balance entries are considered "not found" */
         if (iszero(lep->balance, 8)) lep = NULL;
         break;
      }
   }

   /* release shared read access */
   if (rwlock_rdunlock(&Lelock)) return NULL;

   /* done */
   return lep;

/* error handling */
FAIL_INVAL: set_errno(EINVAL); return NULL;
FAIL_LENOTAVAIL: set_errno(EMCMLENOTAVAIL); return NULL;
}  /* end tag_find() */

/* end include guard */
#endif
