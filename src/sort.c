/**
 * @file sort.c
 * @brief Mochimo specific sorting routines.
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
 * @note The original Polymorphic Shell sort algorithm, shell(), was
 * deprecated in favour of qsort().
 * > For more details see <https://godbolt.org/z/YE7j57Po9>
*/

/* include guard */
#ifndef MOCHIMO_SORT_C
#define MOCHIMO_SORT_C


#include "extint.h"
#include "extprint.h"
#include "types.h"

#include <stdlib.h>  /* for malloc() & qsort() */
#include <string.h>  /* for memcmp() */
#include <errno.h>   /* for errno */

static LTRAN *Ltrans;   /* malloc'd Ltrans[]: Nlt * sizeof(LTRAN) */
static word32 *Ltidx;   /* malloc'd Ltidx[]: Nlt * sizeof(word32) */
static word32 Nlt;      /* number of ledger transactions */

word8 *Tx_ids;    /**< malloc'd Tx_ids[]: Ntx * HASHLEN */
word32 *Txidx;   /**< malloc'd Txidx[]: Ntx * sizeof(word32) */
word32 Ntx;      /**< number of transaction entries in "txclean" */

/**
 * @private
 * Comparison function to sort index to ledger transactions, LTRAN Ltrans[]:
 * includes Ltrans[].trancode[0] in key: (+1)
*/
static int compare_ltrans(const void *va, const void *vb)
{
   word32 *a = (word32 *) va;
   word32 *b = (word32 *) vb;

   return memcmp(Ltrans[*a].addr, Ltrans[*b].addr, TXADDRLEN + 1);
}

/**
 * @private
 * Comparison function to sort index to TXIDs
*/
static int compare_txids(const void *va, const void *vb)
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
   size_t len;
   long offset;
   int ecode;
   word32 j;

   /* open ltran file */
   fp = fopen(fname, "r+b");
   if (fp == NULL) {
      return perrno(errno, "sortlt(): fopen(%s) failed", fname);
   }

   /* seek to file end */
   ecode = fseek(fp, 0, SEEK_END);
   if (ecode != 0) {
      ecode = perrno(ecode, "sortlt(%s): fseek(END) failed", fname);
      goto FAIL_IO;
   }
   /* obtain file offset (at file end), check valid size */
   offset = ftell(fp);
   if (offset == EOF) {
      ecode = perrno(errno, "sortlt(%s): ftell() failed", fname);
      goto FAIL_IO;
   } else if ((offset % sizeof(LTRAN)) != 0) {
      ecode = perr("sortlt(%s): invalid size: %ld bytes", fname, offset);
      goto FAIL_IO;
   }
   /* calc transactions */
   Nlt = offset / sizeof(LTRAN);
   if (Nlt == 0) {
      ecode = perr("sortlt(%s): 0 transactions", fname);
      goto FAIL_IO;
   }
   /* seek to file start */
   ecode = fseek(fp, 0, SEEK_SET);
   if (ecode) {
      ecode = perrno(ecode, "sortlt(%s): fseek(SET) failed", fname);
      goto FAIL_IO;
   }
   /* allocate memory */
   len = Nlt * sizeof(word32);
   Ltidx = malloc(len);
   if (Ltidx == NULL) {
      ecode = perr("sortlt(%s): Ltidx malloc(%zu) failed", fname, len);
      goto FAIL_Ltidx;
   }
   len = Nlt * sizeof(LTRAN);
   Ltrans = malloc(len);
   if (Ltrans == NULL) {
      ecode = perr("sortlt(%s): Ltrans malloc(%zu) failed", fname, len);
      goto FAIL_Ltrans;
   }
   /* read-in transactions */
   if (fread(Ltrans, sizeof(LTRAN), Nlt, fp) != Nlt) {
      ecode = perr("sortlt(%s): fread() failed", fname);
      goto FAIL_IO2;
   }
   /* initialize transaction index; Ltidx[] = 0,1,2,3,4,5,... */
   for(j = 0; j < Nlt; j++) Ltidx[j] = j;
   /* perform sort operation */
   qsort(Ltidx, Nlt, sizeof(word32), compare_ltrans);
   /* return to start of file */
   ecode = fseek(fp, 0, SEEK_SET);
   if (ecode) {
      ecode = perrno(ecode, "sortlt(%s): fseek(SET) failed", fname);
      goto FAIL_IO2;
   }
   /* write sorted transactions */
   for(j = 0; j < Nlt; j++) {
      if(fwrite(&Ltrans[Ltidx[j]], 1, sizeof(LTRAN), fp) != sizeof(LTRAN)) {
         ecode = perr("sortlt(%s): fwrite() failed", fname);
         goto FAIL_IO2;
      }
   }

   /* success */
   ecode = VEOK;

   /* cleanup */
FAIL_IO2:
   free(Ltrans);
FAIL_Ltrans:
   Ltrans = NULL;
   free(Ltidx);
FAIL_Ltidx:
   Ltidx = NULL;
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
 * @returns VEOK on success, else VERROR
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

   /* open txclean file */
   fp = fopen(fname, "rb");
   if (fp == NULL) {
      perrno(errno, "sorttx(): fopen(%s) failed", fname);
      goto FAIL_FOPEN;
   }
   /* seek to file end */
   ecode = fseek(fp, 0, SEEK_END);
   if (ecode != 0) {
      perrno(ecode, "sorttx(%s): fseek(END) failed", fname);
      goto FAIL_IO;
   }
   /* obtain file offset (at file end), check valid size */
   offset = ftell(fp);
   if (offset == EOF) {
      perrno(errno, "sorttx(%s): ftell() failed", fname);
      goto FAIL_IO;
   } else if ((offset % sizeof(TXQENTRY)) != 0) {
      perr("sorttx(%s): invalid size: %ld bytes", fname, offset);
      goto FAIL_IO;
   }
   /* calc transactions */
   Ntx = offset / sizeof(TXQENTRY);
   if (Ntx == 0) {
      perr("sorttx(%s): 0 transactions", fname);
      goto FAIL_IO;
   }
   /* seek to file start */
   ecode = fseek(fp, 0, SEEK_SET);
   if (ecode) {
      perrno(ecode, "sorttx(%s): fseek(SET) failed", fname);
      goto FAIL_IO;
   }
   /* allocate memory */
   len = Ntx * sizeof(word32);
   Txidx = malloc(len);
   if (Txidx == NULL) {
      perr("sorttx(%s): Txidx malloc(%zu) failed", fname, len);
      goto FAIL_TXIDX;
   }
   len = Ntx * HASHLEN;
   Tx_ids = malloc(len);
   if (Tx_ids == NULL) {
      perr("sorttx(%s): Tx_ids malloc(%zu) failed", fname, len);
      goto FAIL_TXIDS;
   }
   /* Read each (pre-computed) TXID into Tx_ids[Ntx][HASHLEN] */
   for(j = 0, bp = Tx_ids; j < Ntx; j++, bp += HASHLEN) {
      /* seek down in transaction record to txid[] */
      ecode = fseek(fp, sizeof(TXQENTRY) - HASHLEN, SEEK_CUR);
      if (ecode != 0) {
         perrno(ecode, "sorttx(%s): fseek(SET) failed", fname);
         goto FAIL_IO2;
      }
      /* reading txid[] puts us at start of next record (TXQENTRY) */
      if (fread(bp, 1, HASHLEN, fp) != HASHLEN) {
         perr("sorttx(%s): fread() failed", fname);
         goto FAIL_IO2;
      }
      /* initialize index Txidx[] = 0,1,2,3,4,5,... */
      Txidx[j] = j;
   }
   /* sort the index */
   qsort(Txidx, Ntx, sizeof(word32), compare_txids);

   /* cleanup - success */
   fclose(fp);
   return VEOK;

   /* cleanup - fail */
FAIL_IO2:
   free(Tx_ids);
FAIL_TXIDS:
   Tx_ids = NULL;
   free(Txidx);
FAIL_TXIDX:
   Txidx = NULL;
FAIL_IO:
   fclose(fp);
FAIL_FOPEN:
   return VERROR;
}  /* end sorttx() */

/* end include guard */
#endif
