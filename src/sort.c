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
#include "error.h"
#include "types.h"

/* external support */
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

static LTRAN *Ltrans;   /* malloc'd Ltrans[]: Nlt * sizeof(LTRAN) - sort */

word8 *Tx_ids;  /* malloc'd Tx_ids[] Ntx*32 bytes */
word32 *Txidx;  /* malloc'd Txidx[] Ntx*4 bytes */
word32 Ntx;     /* number of transactions in clean TX queue */

/**
 * @private
 * Comparison function to sort index to ledger transactions, LTRAN Ltrans[]:
 * includes Ltrans[].trancode[0] in key: (+1)
*/
static int compare_lt(const void *va, const void *vb)
{
   word32 *a = (word32 *) va;
   word32 *b = (word32 *) vb;

   return memcmp(Ltrans[*a].addr, Ltrans[*b].addr, TXWOTSLEN + 1);
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
 * Free resources allocated by sorttx().
*/
void sorttx_free(void)
{
   if (Tx_ids) free(Tx_ids);
   if (Txidx) free(Txidx);
   Tx_ids = NULL;
   Txidx = NULL;
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
      perrno("failed to fopen(%s)", fname);
      return VERROR;
   }

   /* seek to file end */
   if (fseek(fp, 0, SEEK_END) != 0) {
      perrno("failed to fseek(END)");
      goto FAIL_IO;
   }
   /* obtain file offset (at file end), check valid size */
   offset = ftell(fp);
   if (offset == EOF) {
      perrno("failed to ftell()");
      goto FAIL_IO;
   }
   if ((offset % sizeof(LTRAN)) != 0) {
      perr("invalid length: %ld", offset);
      goto FAIL_IO;
   }
   /* calc transactions */
   Nlt = offset / sizeof(LTRAN);
   if (Nlt == 0) {
      perr("0 transactions");
      goto FAIL_IO;
   }
   /* seek to file start */
   if (fseek(fp, 0, SEEK_SET) != 0) {
      perrno("failed to fseek(SET)");
      goto FAIL_IO;
   }
   /* allocate memory */
   Ltidx = malloc((len = Nlt * sizeof(word32)));
   if (Ltidx == NULL) {
      perr("failed to malloc(%zu) Ltidx", len);
      goto FAIL_Ltidx;
   }
   Ltrans = malloc((len = Nlt * sizeof(LTRAN)));
   if (Ltrans == NULL) {
      perr("failed to malloc(%zu) Ltrans", len);
      goto FAIL_Ltrans;
   }
   /* read-in transactions */
   if (fread(Ltrans, sizeof(LTRAN), Nlt, fp) != Nlt) {
      perr("failed to fread(Ltrans)");
      goto FAIL_IO2;
   }
   /* initialize transaction index; Ltidx[] = 0,1,2,3,4,5,... */
   for (j = 0; j < Nlt; j++) Ltidx[j] = j;
   /* perform sort operation */
   qsort(Ltidx, Nlt, sizeof(word32), compare_lt);
   /* return to start of file */
   if (fseek(fp, 0, SEEK_SET) != 0) {
      perrno("failed to fseek(SET) (pre-write)");
      goto FAIL_IO2;
   }
   /* write sorted transactions */
   for (j = 0; j < Nlt; j++) {
      if (fwrite(&Ltrans[Ltidx[j]], sizeof(LTRAN), 1, fp) != 1) {
         perr("failed to fwrite()");
         goto FAIL_IO2;
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
   word32 j;

   /* ensure pointers are free */
   sorttx_free();

   /* open txclean file, and seek to end */
   fp = fopen(fname, "rb");
   if (fp == NULL) {
      perrno("failed to fopen(%s)", fname);
      return VERROR;
   }
   if (fseek(fp, 0, SEEK_END) != 0) {
      perrno("failed to fseek(END)");
      goto CLEANUP;
   }
   /* obtain file offset (at file end), check valid size */
   offset = ftell(fp);
   if (offset == EOF) {
      perrno("failed to ftell(fp)");
      goto CLEANUP;
   }
   if ((offset % sizeof(TXQENTRY)) != 0) {
      perr("invalid size: %ld bytes", offset);
      goto CLEANUP;
   }
   /* calc transactions */
   Ntx = offset / sizeof(TXQENTRY);
   if (Ntx == 0) {
      perr("0 transactions");
      goto CLEANUP;
   }
   /* seek to file start */
   if (fseek(fp, 0, SEEK_SET) != 0) {
      perrno("failed to fseek(SET)");
      goto CLEANUP;
   }
   /* allocate memory */
   Txidx = malloc((len = Ntx * sizeof(word32)));
   if (Txidx == NULL) {
      perr("failed to malloc(%zu) Txidx", len);
      goto CLEANUP_FREE;
   }
   Tx_ids = malloc((len = Ntx * HASHLEN));
   if (Tx_ids == NULL) {
      perr("failed to malloc(%zu) Tx_ids", len);
      goto CLEANUP_FREE;
   }
   /* Read each (pre-computed) TXID into Tx_ids[Ntx][HASHLEN] */
   for(j = 0, bp = Tx_ids; j < Ntx; j++, bp += HASHLEN) {
      /* seek down in transaction record to txid[] */
      if (fseek(fp, sizeof(TXQENTRY) - HASHLEN, SEEK_CUR) != 0) {
         perrno("failed to fseek(CUR)");
         goto CLEANUP_FREE;
      }
      /* reading txid[] puts us at start of next record (TXQENTRY) */
      if (fread(bp, HASHLEN, 1, fp) != 1) {
         perr("failed to fread(bp)");
         goto CLEANUP_FREE;
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
CLEANUP_FREE:
   sorttx_free();
CLEANUP:
   fclose(fp);
   return VERROR;
}  /* end sorttx() */

/* end include guard */
#endif
