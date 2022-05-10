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
#include "util.h"
#include "types.h"

/* external support */
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

static LTRAN *Ltrans;   /* malloc'd Ltrans[]: Nlt * sizeof(LTRAN) - sort */

/**
 * @private
 * Comparison function to sort index to ledger transactions, LTRAN Ltrans[]:
 * includes Ltrans[].trancode[0] in key: (+1)
*/
static int compare_lt(const void *va, const void *vb)
{
   word32 *a = (word32 *) va;
   word32 *b = (word32 *) vb;

   return memcmp(Ltrans[*a].addr, Ltrans[*b].addr, TXADDRLEN + 1);
}

/**
 * @private
 * Comparison function to sort index to TXIDs
*/
static int compare_tx(const void *va, const void *vb)
{
   word32 *a = (word32 *) va;
   word32 *b = (word32 *) vb;

   return memcmp(&Tx_ids[(*a) * HASHLEN], &Tx_ids[(*b) * HASHLEN], HASHLEN);
}

/**
 * Creates a malloc'd sort index and ledger transaction list:
 * - `word32 Ltidx[Nlt];`
 * - `LTRAN Ltrans[Nlt];`
 * ... writes sorted list back to same file.
 * @param fname Name of file to sort
 * @returns VEOK on success, else VERROR
 */
int sortlt(char *fname)
{
   FILE *fp;
   word32 *Ltidx;   /* malloc'd Ltidx[]: Nlt * sizeof(word32) */
   word32 Nlt;      /* number of ledger transactions */
   word32 j;
   size_t len;
   long offset;
   int ecode;

   /* open ltran file */
   fp = fopen(fname, "r+b");
   if (fp == NULL) {
      return perrno(errno, "sortlt(): failed to fopen(%s)", fname);
   }

   /* seek to file end */
   if (fseek(fp, 0, SEEK_END) != 0) {
      mErrno(FAIL_IO, "sortlt(): failed to fseek(END)");
   }
   /* obtain file offset (at file end), check valid size */
   offset = ftell(fp);
   if (offset == EOF) mErrno(FAIL_IO, "sortlt(): failed to ftell()");
   if ((offset % sizeof(LTRAN)) != 0) {
      mError(FAIL_IO, "sortlt(): invalid length: %ld", offset);
   }
   /* calc transactions */
   Nlt = offset / sizeof(LTRAN);
   if (Nlt == 0) mError(FAIL_IO, "sortlt(): 0 transactions");
   /* seek to file start */
   if (fseek(fp, 0, SEEK_SET) != 0) {
      mErrno(FAIL_IO, "sortlt(): failed to fseek(SET)");
   }
   /* allocate memory */
   Ltidx = malloc((len = Nlt * sizeof(word32)));
   if (Ltidx == NULL) {
      mError(FAIL_Ltidx, "sortlt(): failed to malloc(%zu) Ltidx", len);
   }
   Ltrans = malloc((len = Nlt * sizeof(LTRAN)));
   if (Ltrans == NULL) {
      mError(FAIL_Ltrans, "sortlt(): failed to malloc(%zu) Ltrans", len);
   }
   /* read-in transactions */
   if (fread(Ltrans, sizeof(LTRAN), Nlt, fp) != Nlt) {
      mError(FAIL_IO2, "sortlt(): failed to fread(Ltrans)");
   }
   /* initialize transaction index; Ltidx[] = 0,1,2,3,4,5,... */
   for (j = 0; j < Nlt; j++) Ltidx[j] = j;
   /* perform sort operation */
   qsort(Ltidx, Nlt, sizeof(word32), compare_lt);
   /* return to start of file */
   if (fseek(fp, 0, SEEK_SET) != 0) {
      mErrno(FAIL_IO2, "sortlt(): failed to fseek(SET) (pre-write)");
   }
   /* write sorted transactions */
   for (j = 0; j < Nlt; j++) {
      if (fwrite(&Ltrans[Ltidx[j]], sizeof(LTRAN), 1, fp) != 1) {
         mError(FAIL_IO2, "sortlt(): failed to fwrite()");
      }
   }

   /* success */
   ecode = VEOK;

   /* cleanup */
FAIL_IO2:
   free(Ltrans);
   Ltrans = NULL;
FAIL_Ltrans:
   free(Ltidx);
   Ltidx = NULL;
FAIL_Ltidx:
FAIL_IO:
   fclose(fp);
   Nlt = 0;

   return ecode;
}  /* end sortlt() */


/**
 * Creates a malloc'd sort index and transaction ID list:
 * - `word32 Txidx[Ntx];`
 * - `word8 Tx_ids[Ntx * HASHLEN];`
 * ... stores sorted list in memory.
 * @param fname Name of file to sort
 * @returns VEOK on success, else error code
 */
int sorttx(char *fname)
{
   FILE *fp;
   word8 *bp;
   size_t len;
   long offset;
   int ecode;
   word32 j;

   /* ensure pointers are free */
   if (Tx_ids) {
      free(Tx_ids);
      Tx_ids = NULL;
   }
   if (Txidx) {
      free(Txidx);
      Txidx = NULL;
   }

   /* open txclean file, and seek to end */
   fp = fopen(fname, "rb");
   if (fp == NULL) mErrno(FAIL, "sorttx(): failed to fopen(%s)", fname);
   if (fseek(fp, 0, SEEK_END) != 0) {
      mErrno(FAIL_IO, "sorttx(): failed to fseek(END)");
   }
   /* obtain file offset (at file end), check valid size */
   offset = ftell(fp);
   if (offset == EOF) mErrno(FAIL_IO, "sorttx(): failed to ftell(fp)");
   if ((offset % sizeof(TXQENTRY)) != 0) {
      mError(FAIL_IO, "sorttx(): invalid size: %ld bytes", offset);
   }
   /* calc transactions */
   Ntx = offset / sizeof(TXQENTRY);
   if (Ntx == 0) mError(FAIL_IO, "sorttx(): 0 transactions");
   /* seek to file start */
   if (fseek(fp, 0, SEEK_SET) != 0) {
      mErrno(FAIL_IO, "sorttx(): failed to fseek(SET)");
   }
   /* allocate memory */
   Txidx = malloc((len = Ntx * sizeof(word32)));
   if (Txidx == NULL) {
      mError(FAIL_TXIDX, "sorttx(): failed to malloc(%zu) Txidx", len);
   }
   Tx_ids = malloc((len = Ntx * HASHLEN));
   if (Tx_ids == NULL) {
      mError(FAIL_TXIDS, "sorttx(): failed to malloc(%zu) Tx_ids", len);
   }
   /* Read each (pre-computed) TXID into Tx_ids[Ntx][HASHLEN] */
   for(j = 0, bp = Tx_ids; j < Ntx; j++, bp += HASHLEN) {
      /* seek down in transaction record to txid[] */
      if (fseek(fp, sizeof(TXQENTRY) - HASHLEN, SEEK_CUR) != 0) {
         mErrno(FAIL_IO2, "sorttx(): failed to fseek(CUR)");
      }
      /* reading txid[] puts us at start of next record (TXQENTRY) */
      if (fread(bp, HASHLEN, 1, fp) != 1) {
         mError(FAIL_IO2, "sorttx(): failed to fread(bp)");
      }
      /* initialize index Txidx[] = 0,1,2,3,4,5,... */
      Txidx[j] = j;
   }

   fclose(fp);

   /* sort the index */
   qsort(Txidx, Ntx, sizeof(word32), compare_tx);

   /* success */
   return VEOK;

   /* failure / error handling */
FAIL_IO2:
   free(Tx_ids);
FAIL_TXIDS:
   Tx_ids = NULL;
   free(Txidx);
FAIL_TXIDX:
   Txidx = NULL;
FAIL_IO:
   fclose(fp);
FAIL:

   return ecode;
}  /* end sorttx() */

/* end include guard */
#endif
