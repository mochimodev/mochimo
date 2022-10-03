/**
 * @private
 * @headerfile sort.h <sort.h>
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_SORT_C
#define MOCHIMO_SORT_C


#include "sort.h"

/* internal support */
#include "types.h"
#include "error.h"

/* external support */
#include <string.h>

/**
 * Configurable global option for adjusting the size of the
 * external_merge_sort() memory buffer.
 */
size_t MaxSortBuffer_opt = DEFAULT_SORT_BUFFER;


int filesort_compare_tagidx(const void *a, const void *b)
{
   return memcmp(
      ((TAGIDX *) *((void **) a))->tag,
      ((TAGIDX *) *((void **) b))->tag,
      TXTAGLEN);
}

/**
 * Sort a file containing size length elements. Performs a single qsort()
 * where data fits in memory, or an external merge-sort where data is too
 * large for allocated memory buffer (of size, MaxSortBuffer_opt).
 * NOTE: The comparison function is passed (void **), and MUST be cast
 * from (const void *a) to (*((void **) a)) within the comparator function.
 * @param filename Name of file to sort
 * @param size Number of characters per element to sort by
 * @param comp Pointer to comparison function for comparing data
 * @returns VEOK on success, else non-zero value. Check errno for details.
*/
int filesort
   (char *filename, size_t size, int (*comp)(const void *, const void *))
{
   void **index;
   long long len, in;
   char *buffer, *minbuf, *nextbuf;
   char *wbuf, *rbuf[DEFAULT_SORT_FILES];
   FILE *fp, *fpa[DEFAULT_SORT_FILES];
   size_t rbufidx[DEFAULT_SORT_FILES];
   size_t rbuflen[DEFAULT_SORT_FILES];
   size_t wbufidx, buflen, minidx;
   size_t fidx, files, fmerge;
   size_t count, count_in, i;
   char fname[DEFAULT_SORT_FILES][FILENAME_MAX];
   char splitfname[FILENAME_MAX];

   /* BEGIN INIT PHASE */

   /* sanity checks */
   if (filename == NULL || size == 0 || comp == NULL) goto FAIL_INVAL;

   /* open file and check length matches sort_size */
   fp = fopen(filename, "rb");
   if (fp == NULL) return VERROR;
   if (fseek64(fp, 0LL, SEEK_END) != 0) goto FAIL_IO;
   len = ftell64(fp);
   if (len == EOF) goto FAIL_IO;
   if ((len % (long long) size) != 0) goto FAIL_SORT_LENGTH;
   rewind(fp);

   /* create sortand index buffers to read in and sort elements */
   if ((size_t) len > MaxSortBuffer_opt) {
      count = MaxSortBuffer_opt / size;
   } else count = (size_t) len / size;
   buflen = count * size;
   buffer = malloc(buflen);
   if (buffer == NULL) goto FAIL_IO;
   index = malloc(count * sizeof(void *));
   if (index == NULL) goto FAIL_IN;

   /* BEGIN SPLIT PHASE */

   files = 0;

   /* read data into buffer in "chunks" and sort into temp files */
   for (in = 0LL; in < len; in += (long long) (count_in * size)) {
      /* read in chunk of data and check failure */
      count_in = fread(buffer, size, count, fp);
      if (count_in < count && !feof(fp)) goto FAIL_IN2;
      /* initialize index pointers */
      for (i = 0; i < count_in; i++) index[i] = &buffer[i * size];
      /* perform sort on index pointers to buffer data */
      qsort(index, count_in, sizeof(void *), comp);
      /* write sorted "chunk" to numbered file */
      snprintf(splitfname, FILENAME_MAX, "%s.%zu", filename, files++);
      fpa[0] = fopen(splitfname, "wb");
      if (fpa[0] == NULL) goto FAIL_IN2;
      for (i = 0; i < count_in; i++) {
         if (fwrite(index[i], size, 1, fpa[0]) != 1) goto FAIL_IN3;
      }
      /* close output file and continue */
      fclose(fpa[0]);
   }  /* end for (in = 0LL... */
   /* close input file -- free index */
   fclose(fp);
   free(index);

   /* BEGIN MERGE PHASE */

   /* clear file pointers array */
   for (i = 0; i < DEFAULT_SORT_FILES; i++) fpa[i] = NULL;

   /* merge files in loop determined by SORT_FILES */
   for (fmerge = 0; (fmerge + 1) < files; ) {
      /* open multiple files for merge */
      for (fidx = 0; fmerge < files && fidx < DEFAULT_SORT_FILES; fidx++) {
         snprintf(fname[fidx], FILENAME_MAX, "%s.%zu", filename, fmerge++);
         fpa[fidx] = fopen(fname[fidx], "rb");
         if (fpa[fidx] == NULL) goto FAIL_MERGE;
      }
      /* open additional "split" file for merge destination */
      snprintf(splitfname, FILENAME_MAX, "%s.%zu", filename, files++);
      fp = fopen(splitfname, "wb");
      if (fp == NULL) goto FAIL_MERGE;
      /* prepare reusable space in sort buffer */
      count = (buflen / (fidx + 1)) / size;
      wbufidx = 0;
      wbuf = buffer + (fidx * count * size);
      for (i = 0; i < fidx; i++) {
         rbufidx[i] = rbuflen[i] = count;
         rbuf[i] = buffer + (i * count * size);
      }
      /* loop through all data and pass minimum values to write buffer */
      for (minbuf = NULL; ; minbuf = NULL) {
         /* search next minimum read buffer value -- fread as necessary */
         for (i = 0; i < fidx; i++) {
            if (rbufidx[i] >= rbuflen[i]) {
               if (fpa[i] == NULL) continue;
               /* perform read and set read buffer parameters */
               count_in = fread(rbuf[i], size, count, fpa[i]);
               if (count_in < count) {
                  if (feof(fpa[i])) {
                     fclose(fpa[i]);
                     fpa[i] = NULL;
                     /* remove temp file */
                     if (remove(fname[i]) != 0) goto FAIL_MERGE2;
                  } else goto FAIL_MERGE2;
                  if (count_in == 0) continue;
               }
               /* reset read buffer parameters */
               rbuflen[i] = count_in;
               rbufidx[i] = 0;
            }  /* end if (rbufidx[i]... */
            nextbuf = rbuf[i] + (rbufidx[i] * size);
            if (minbuf == NULL || memcmp(nextbuf, minbuf, size) < 0) {
               minbuf = nextbuf;
               minidx = i;
            }
         }  /* end for for (i = 0... */
         /* check minbuf -- place in write buffer */
         if (minbuf) {
            memcpy(wbuf + (wbufidx * size), minbuf, size);
            rbufidx[minidx]++;
            wbufidx++;
         }
         if (minbuf == NULL || wbufidx >= count) {
            /* move write buffer data to file */
            if (fwrite(wbuf, size, wbufidx, fp) < wbufidx) {
               goto FAIL_MERGE2;
            }
            wbufidx = 0;
         }
         if (minbuf == NULL) break;
      }  /* end for (minbuf = NULL... */
      /* ensure destination is closed */
      fclose(fp);
   }  /*  end for (fmerge = 0... */

   /* cleanup */
   free(buffer);

   /* overwrite the original file with the remaining (sorted) file */
   snprintf(splitfname, FILENAME_MAX, "%s.%zu", filename, fmerge);
   if (remove(filename) != 0) return VERROR;
   if (rename(splitfname, filename) != 0) return VERROR;

   /* sort successful */
   return (errno = VEOK);

/* merge phase failures */
FAIL_MERGE2: fclose(fp);
FAIL_MERGE:
   for (i = 0; i < DEFAULT_SORT_FILES; i++) {
      if (fpa[i]) {
         fclose(fpa[i]);
         remove(fname[i]);
         fpa[i] = NULL;
      }
   }
   free(buffer);
   return VERROR;

/* input phase failures */
FAIL_IN3: fclose(fpa[0]);
FAIL_IN2: free(index);
FAIL_IN: free(buffer);
   goto FAIL_IO;

/* init phase failures */
FAIL_INVAL: errno = EINVAL; return VERROR;
FAIL_SORT_LENGTH: errno = EMCM_SORT_LENGTH;
FAIL_IO: fclose(fp);
   return VERROR;
}  /* end mergesort_file() */

/* end include guard */
#endif
