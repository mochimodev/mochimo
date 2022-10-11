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
#include "types.h"
#include "sort.h"
#include "error.h"

/* external support */
#include <string.h>
#include "extmath.h"
#include "extlib.h"

typedef struct {
   void *next;
   void *lmap;
   TAGIDX *tmap;
   size_t lsize;
   size_t lcount;
   size_t tcount;
   int depth;
} LSMTNode;

word32 Sanctuary = 0;
word32 Lastday = 0;

/* shared read exclusive write access to LSMTree data */
static RWLock Lelock = RWLOCK_INITIALIZER;
/* Ledger LSMT (should always point to latest node) */
static LSMTNode *Ledger;

/** Ledger filename (configurable option) */
char *Ledger_opt = "ledger.dat";
/** Tag index filename (configurable option) */
char *Tagidx_opt = "tagidx.dat";

/**
 * Append a WOTS+ Ledger and Tagidx file to the next Ledger depth.
 * Supplied files are consumed by this process.
 * @param lfname Filename of WOTS+ ledger
 * @param tfname Filename of tagidx
 * @return (int) value representing the operation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 */
int le_appendw(char *lfname, char *tfname)
{
   static const int prot = PROT_READ | PROT_WRITE;
   static const int flags = MAP_SHARED;

   long long llen, tlen;
   FILE *lfp, *tfp;
   LSMTNode *node;
   int depth;
   char lfname2[FILENAME_MAX];
   char tfname2[FILENAME_MAX];

   /* obtain next logical depth */
   depth = Ledger ? Ledger->depth + 1 : 0;
   /* move files to file-parts indicating depth */
   snprintf(lfname2, FILENAME_MAX, "%s.%d", Ledger_opt, depth);
   snprintf(tfname2, FILENAME_MAX, "%s.%d", Tagidx_opt, depth);
   if (rename(lfname, lfname2) != 0) return VERROR;
   if (rename(tfname, tfname2) != 0) return VERROR;
   /* open files with read/write permissions, seek to END */
   lfp = fopen(lfname2, "rb+");
   tfp = fopen(tfname2, "rb+");
   if (lfp == NULL || fseek64(lfp, 0LL, SEEK_END) != 0) goto FAIL_IO;
   if (tfp == NULL || fseek64(tfp, 0LL, SEEK_END) != 0) goto FAIL_IO;
   /* obtain file lengths -- check ledger length */
   llen = ftell64(lfp);
   tlen = ftell64(tfp);
   if (llen == (-1LL) || tlen == (-1LL)) goto FAIL_IO;
   /* create tree node to append */
   node = malloc(sizeof(LSMTNode));
   if (node == NULL) goto FAIL_LSMT;
   /* obtain map pointers for ledger and tag index */
   node->lmap = mmap(NULL, (size_t) llen, prot, flags, fileno(lfp), 0);
   node->tmap = mmap(NULL, (size_t) tlen, prot, flags, fileno(tfp), 0);
   if (node->lmap == MAP_FAILED || node->tmap == MAP_FAILED) goto FAIL_MAP;
   /* derive entry size, and entry/index counts */
   node->lsize = sizeof(LENTRY_W);
   node->lcount = (size_t) (llen / node->lsize);
   node->tcount = (size_t) (tlen / sizeof(*(node->tmap)));
   /* set next depth and ledger pointer */
   node->depth = depth;
   node->next = Ledger;
   /* close files */
   fclose(lfp);
   fclose(tfp);
   /* promote LSMT node */
   Ledger = node;

   /* success */
   errno = 0;
   return VEOK;

/* error handling cleanup */
FAIL_MAP:
   if (node && node->lmap) munmap(node->lmap, 0);
   if (node && node->tmap) munmap(node->tmap, 0);
FAIL_LSMT:
   if (node) free(node);
FAIL_IO:
   if (lfp) fclose(lfp);
   if (tfp) fclose(tfp);
   return VERROR;
}  /* end le_appendw() */

/**
 * Close Ledger to a specified depth (inclusive).
 * @code le_close(0); @endcode ... closes the entire tree.
 * @param depth Depth at which to close the Ledger (inclusive)
*/
void le_close(int depth)
{
   LSMTNode *node;

   /* close the ledger tree up to the specified depth */
   while (Ledger && Ledger->depth >= depth) {
      node = Ledger;
      Ledger = node->next;
      munmap(node->lmap, 0);
      munmap(node->tmap, 0);
      free(node);
   }
}  /* end le_close() */

int le_cmpp(const void *a, const void *b)
{
   return memcmp(a, b, TXPADDRLEN);
}

int le_cmpw(const void *a, const void *b)
{
   return memcmp(a, b, TXWADDRLEN);
}

/**
 * Compress WOTS+ Ledger file-parts of the specified depths. A file-part
 * is identified by a numbered file extension whose number is one "part"
 * of a continuous set of numbers (e.g. fname.0, fname.1, ..., fname.n).
 * @param fname Destination filename and basename to the file-parts
 * @param from Start (inclusive) of compression depth range
 * @param to End (inclusive) of compression depth range
 * @return (int) value representing the compression result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int le_compressw(char *fname, int from, int to)
{
   LENTRY_W lep[LEDEPTHMAX];
   FILE *fp, *fpp[LEDEPTHMAX];
   int cond, i, files, fmerge, min;
   char fppname[LEDEPTHMAX][FILENAME_MAX];

   /* check parameter validity */
   if (Ledger == NULL) goto FAIL_INVAL;

   /* calc number of merge files */
   files = 1 + to - from;
   if (files < 2) goto FAIL_INVAL;

   /* clear initial arrays */
   memset(fppname, 0, sizeof(fppname));
   memset(fpp, 0, sizeof(fpp));
   fp = NULL;

   /* open destination file -- increase buffer */
   fp = fopen(fname, "wb");
   if (fp == NULL) goto FAIL_IO;
   if (setvbuf(fp, NULL, _IOFBF, LERWBUFSZ) != 0) goto FAIL_IO;
   /* open (all) files for merge -- increase buffers */
   for (min = (-1), fmerge = from, i = 0; i < files; i++) {
      snprintf(fppname[i], FILENAME_MAX, "%s.%d", fname, fmerge++);
      fpp[i] = fopen(fppname[i], "rb");
      if (fpp[i] == NULL) goto FAIL_IO;
      if (setvbuf(fpp[i], NULL, _IOFBF, LERWBUFSZ) != 0) goto FAIL_IO;
      /* read initial data into buffers */
      if (fread(&lep[i], sizeof(*lep), 1, fpp[i]) != 1) {
         if (ferror(fpp[i])) goto FAIL_IO;
         fclose(fpp[i]);
         fpp[i] = NULL;
      } else min = i;
   }

   /* perform merge sort on file data -- fill lep buffer as necessary */
   for ( ; min >= 0; ) {
      min = (-1);
      for (i = files - 1; i >= 0; i--) {
         /* !IMPORTANT! Loop iterates in reverse to acquire the
          * latest ledger value FIRST; ignoring old duplicates
         */
         if (fpp[i] == NULL) continue;
         if (min < 0) min = i;
         else {
            cond = le_cmpw(&lep[i], &lep[min]);
            if (cond < 0) min = i;
            else if (cond == 0) {
               /* skip duplicate (OLD) address, read for next loop */
               if (fread(&lep[i], sizeof(*lep), 1, fpp[i]) != 1) {
                  if (ferror(fpp[i])) goto FAIL_IO;
                  fclose(fpp[i]);
                  fpp[i] = NULL;
               }
            }
         }
      }  /* end for (i = files - 1; ... */
      /* ensure minimum value was found -- write value and read another */
      if (min >= 0) {
         if (fwrite(&lep[min], sizeof(*lep), 1, fp) != 1) goto FAIL_IO;
         if (fread(&lep[min], sizeof(*lep), 1, fpp[min]) != 1) {
            if (ferror(fpp[min])) goto FAIL_IO;
            fclose(fpp[min]);
            fpp[min] = NULL;
         }
      }
   }  /* end for ( ; min >= 0; ... */
   /* cleanup -- sources are closed in loop */
   fclose(fp);

   /* merge successful */
   errno = 0;
   return 0;

/* error handling cleanup */
FAIL_IO:
   for (i = 0; i < files; i++)
      if (fpp[i]) fclose(fpp[i]);
   if (fp) fclose(fp);
   return VERROR;
FAIL_INVAL: errno = EINVAL; return VERROR;
}  /* end le_compressw() */

/**
 * Delete Ledger to a specified depth (inclusive).
 * @code le_delete(0); @endcode ... deletes the entire tree.
 * @param depth Depth at which to close the Ledger (inclusive)
*/
void le_delete(int depth)
{
   LSMTNode *node;
   char fname[FILENAME_MAX];

   /* close the ledger tree up to the specified depth */
   while (Ledger && Ledger->depth >= depth) {
      node = Ledger;
      Ledger = node->next;
      munmap(node->lmap, 0);
      munmap(node->tmap, 0);
      free(node);
      /* delete associated files */
      snprintf(fname, FILENAME_MAX, "%s.%d", Ledger_opt, node->depth);
      remove(fname);
      snprintf(fname, FILENAME_MAX, "%s.%d", Tagidx_opt, node->depth);
      remove(fname);
   }
}  /* end le_delete() */

/**
 * Extract a WOTS+ ledger from a WOTS+ neo-genesis block.
 * Checks address sort of neo-genesis block while processing.
 * Appends extracted data to a new Ledger data tree on success.
 * @param ngfname Filename of the WOTS+ neo-genesis block
 * @return (int) value representing extraction result
 * @retval VEBAD on invalid; check errno for details
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int le_extractw(char *ngfname)
{
   LENTRY_W le;            /* buffer for WOTS+ ledger entries */
   FILE *fp, *lfp;         /* FILE pointers */
   long long idx, remain;
   word32 hdrlen;          /* buffer for block header length */
   word8 paddr[TXWADDRLEN]; /* ledger address sort check */

   /* open the neo-genesis block for reading */
   fp = fopen(ngfname, "rb");
   if (fp == NULL) return VERROR;
   /* read header length */
   if (fread(&hdrlen, 4, 1, fp) != 1) goto FAIL_IO;
   /* check hdrlen value and alignment -- must have at least 1 entry */
   if ((hdrlen % sizeof(le)) != 4) goto FAIL_HDRLEN;
   if (hdrlen < (sizeof(le) + 4)) goto FAIL_HDRLEN;
   /* derive number of ledger entries */
   remain = ((long long) hdrlen - 4) / sizeof(le);

   /* open ledger and tag index (overwrite) for writing */
   lfp = fopen(Ledger_opt, "wb");
   if (lfp == NULL) goto FAIL_IO;
   /* reduce read/write system calls */
   setvbuf(lfp, NULL, _IOFBF, LERWBUFSZ);
   setvbuf(fp, NULL, _IOFBF, LERWBUFSZ);
   /* Read the ledger from fp and copy it to lfp,
    * extract tags and write index pairs to tfp.
    */
   if (fseek(fp, 4, SEEK_SET)) goto FAIL_IO2;
   for (idx = 0; remain > 0LL; remain--) {
      if (fread(&le, sizeof(le), 1, fp) != 1) goto FAIL_IO2;
      /* check ledger sort */
      if (idx && memcmp(le.addr, paddr, TXWADDRLEN) <= 0) goto FAIL_SORT;
      memcpy(paddr, le.addr, TXWADDRLEN);
      /* write ledger entries to ledger file, if more ledger data */
      if (fwrite(&le, sizeof(le), 1, lfp) != 1) goto FAIL_IO2;
   }  /* end for() */
   /* close files */
   fclose(lfp);
   fclose(fp);

   /* build tag index for extracted ledger data */
   if (tag_extractw(Ledger_opt, Tagidx_opt) != VEOK) return VERROR;

   /* acquire exclusive ledger lock */
   if ((errno = rwlock_wrlock(&Lelock))) return VERROR;
   /* delete existing ledger data */
   le_delete(0);
   /* append new ledger and tag index files */
   if (le_appendw(Ledger_opt, Tagidx_opt) != VEOK) goto FAIL_APPEND;
   /* release exclusive ledger lock */
   if ((errno = rwlock_wrunlock(&Lelock))) return VERROR;

   /* success */
   errno = 0;
   return VEOK;

/* error handling */
FAIL_APPEND: rwlock_wrunlock(&Lelock); return VERROR;
FAIL_SORT: errno = EMCM_LE_SORT;
FAIL_IO2: fclose(lfp); goto FAIL_IO;
FAIL_HDRLEN: errno = EMCM_HDRLEN;
FAIL_IO: fclose(fp); return VERROR;
}  /* end le_extractw() */

/**
 * Find a ledger entry by address. Requires an active Ledger tree.
 * @param addr Pointer an address to search for
 * @param len Length of address to compare in search
 * @return (void *) pointer to ledger entry, else NULL.
 * Check @a errno for more details.
 * @exception errno=0 No errors, address was not found in Ledger
 * @exception errno=EADDRNOTAVAIL The @a addr parameter is NULL
 * @exception errno=EMCMLENOTAVAIL The internal Ledger is NULL
*/
void *le_find(void *addr)
{
   LSMTNode *node;
   void *found;

   /* check valid ledger */
   if (Ledger == NULL) goto FAIL_LEDGER;
   if (addr == NULL) goto FAIL_ADDR;

   /* acquire shared read access */
   if ((errno = rwlock_rdlock(&Lelock))) return NULL;

   /* walk ledger nodes searching for data */
   for (found = NULL, node = Ledger; node; node = node->next) {
      found = (node->lsize == sizeof(LENTRY_W))
         ? bsearch(addr, node->lmap, node->lcount, node->lsize, le_cmpw)
         : bsearch(addr, node->lmap, node->lcount, node->lsize, le_cmpp);
      if (found) break;
   }

   /* release shared read access */
   if ((errno = rwlock_rdunlock(&Lelock))) return NULL;

   /* not found */
   errno = 0;
   return found;

FAIL_LEDGER: errno = EMCMLENOTAVAIL; return NULL;
FAIL_ADDR: errno = EADDRNOTAVAIL; return NULL;
}  /* end le_find() */

/**
 * Activate the Sanctuary Protocol to renew a ledger.
 * @param fee Pointer to fee threshold of the Sanctuary Protocol
 * @return (int) value representing operation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 */
int le_reneww(void *fee)
{
   LENTRY_W le;
   FILE *fp, *fpout;
   word32 in, out;
   word32 sanctuary[2];
   char lfname[FILENAME_MAX];
   char tfname[FILENAME_MAX];

   if (Sanctuary == 0) return 0;  /* success */
   if (Ledger == NULL) {
      errno = ENOENT;
      return VERROR;
   }

   sanctuary[0] = Sanctuary;
   sanctuary[1] = 0;

   print("Lastday 0x%0x.  Carousel begins...\n", Lastday);

   /* compress the ledger tree if multiple depths */
   if (Ledger->depth > 0) {
      if (le_compressw(Ledger_opt, 0, Ledger->depth) != VEOK) return VERROR;
      /* NOTE: compressed ledger parts are now at Ledger_opt (sorted) */
      snprintf(lfname, FILENAME_MAX, "%s", Ledger_opt);
   } else snprintf(lfname, FILENAME_MAX, "%s.%d", Ledger_opt, 0);

   /* open I/O ledger file pointers */
   fp = fopen(lfname, "rb");
   if (fp == NULL) return VERROR;
   snprintf(lfname, FILENAME_MAX, "%s.%d", Ledger_opt, Ledger->depth + 1);
   fpout = fopen(lfname, "wb");
   if (fpout == NULL) goto FAIL;

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

   /* build tag index for ledger data */
   snprintf(tfname, FILENAME_MAX, "%s.%d", Tagidx_opt, Ledger->depth + 1);
   if (tag_extractw(lfname, tfname) != VEOK) return VERROR;

   /* acquire exclusive ledger lock */
   if ((errno = rwlock_wrlock(&Lelock))) return VERROR;
   /* delete all existing ledger data */
   le_delete(0);
   /* append new ledger and tag index files */
   if (le_appendw(lfname, tfname) != VEOK) goto FAIL_APPEND;
   /* release exclusive ledger lock */
   if ((errno = rwlock_wrunlock(&Lelock))) return VERROR;

   /* success */
   print("%u citizens renewed out of %u\n", in - out, in);
   return VEOK;

FAIL_APPEND: rwlock_wrunlock(&Lelock); return VERROR;
FAIL_IO: fclose(fpout);
FAIL: fclose(fp);
   return VERROR;
}  /* end le_reneww() */

/**
 * Update the internal Ledger with a WOTS+ ledger update.
 * Consumes WOTS+ ledger file and (re)builds tag index.
 * @param filename Filename of ledger data to append
 * @param lsize size of each ledger entry in 
 * @returns (int) value representing operation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int le_update(char *filename, size_t count)
{
   LSMTNode *node;
   int depth, nextdepth;
   char lfname[FILENAME_MAX];
   char tfname[FILENAME_MAX];

   /* obtain next depth */
   nextdepth = Ledger ? Ledger->depth + 1 : 0;

   /* determine if, and how many, depths to compress */
   for (depth = nextdepth, node = Ledger; node; node = node->next) {
      if (node->lcount < (count * LECOMPRESS(node->depth))) {
         count += node->lcount;
         depth = node->depth;
      }
   }

   /* prepare ledger file for possible compression */
   snprintf(tfname, FILENAME_MAX, "%s.%d", Tagidx_opt, nextdepth);
   snprintf(lfname, FILENAME_MAX, "%s.%d", Ledger_opt, nextdepth);
   if (rename(filename, lfname) != 0) return VERROR;
   /* perform ledger tree compression on depth discrepency */
   if (depth != nextdepth) {
      /* compress and move ledger file to appropriate depth */
      if (le_compressw(Ledger_opt, depth, nextdepth) != VEOK) return VERROR;
      /* NOTE: compressed ledger parts are now at Ledger_opt (sorted) */
      if (rename(Ledger_opt, lfname) != 0) return VERROR;
   }

   /* build tag index for ledger data */
   if (tag_extractw(lfname, tfname) != VEOK) return VERROR;

   /* acquire exclusive ledger lock */
   if ((errno = rwlock_wrlock(&Lelock))) return VERROR;
   /* delete existing ledger data, up to nextdepth */
   le_delete(depth);
   /* append new ledger and tag index files */
   if (le_appendw(lfname, tfname) != VEOK) goto FAIL_APPEND;
   /* release exclusive ledger lock */
   if ((errno = rwlock_wrunlock(&Lelock))) return VERROR;

   /* success */
   errno = 0;
   return VEOK;

FAIL_APPEND: rwlock_wrunlock(&Lelock); return VERROR;
}  /* end le_update() */

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
 * @todo rethink empty tag file protection
 */
int tag_extractw(char *lfname, char *tfname)
{
   LENTRY_W le;      /* comparison buffer for ledger entries */
   TAGIDX ti;        /* tag index buffer pointer */
   FILE *lfp, *tfp;  /* FILE pointers */
   long long idx;

   /* open the ledger for reading -- reduce read/write system calls */
   lfp = fopen(lfname, "rb");
   tfp = fopen(tfname, "wb");
   if (lfp == NULL || tfp == NULL) goto FAIL_IO;
   if (setvbuf(lfp, NULL, _IOFBF, LERWBUFSZ) != 0) goto FAIL_IO;
   if (setvbuf(tfp, NULL, _IOFBF, LERWBUFSZ) != 0) goto FAIL_IO;

   /* Read ledger entries, extract tags and write index pairs */
   for (idx = 0; ; idx++) {
      if (fread(&le, sizeof(le), 1, lfp) != 1) {
         if (feof(lfp)) break;
         goto FAIL_IO;
      }
      /* if address includes a valid tag... */
      if (WOTS_HAS_TAG(le.addr)) {
         /* ... associate index and write to file */
         put64(ti.idx, &idx);
         memcpy(ti.tag, WOTS_TAGp(le.addr), TXTAGLEN);
         if (fwrite(&ti, sizeof(ti), 1, tfp) != 1) goto FAIL_IO;
      }
   }  /* end for() */
   /* close files */
   fclose(lfp);
   fclose(tfp);

   /* sort tag index file */
   if (filesort(tfname, sizeof(ti), tag_cmp) != 0) return VERROR;
   /* EMPTY TAG INDEX FILE PROTECTION */
   ftouch(tfname);

   /* success */
   return VEOK;

FAIL_IO:
   if (tfp) fclose(tfp);
   if (lfp) fclose(lfp);
   return VERROR;
}  /* end tag_extractw() */

/**
 * Find a tag and return the associated ledger entry.
 * Requires an active Ledger tree.
 * @param tag Pointer to tag to search for
 * @return (void *) pointer to ledger entry, or NULL if not found.
 */
void *tag_find(void *tag)
{
   LSMTNode *node;
   TAGIDX *f;
   long long idx;

   /* check valid ledger and tag index */
   if (Ledger == NULL) return NULL;

   /* walk ledger nodes searching for data */
   for (f = NULL, node = Ledger; node; node = node->next) {
      f = bsearch(tag, node->tmap, node->tcount, sizeof(*f), tag_cmp);
      if (f) {
         put64(&idx, f->idx);
         return (char *) node->lmap + (node->lsize * idx);
      }
   }

   /* not found */
   return NULL;
}  /* end tag_find() */

/* end include guard */
#endif
