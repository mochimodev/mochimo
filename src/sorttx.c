/* sorttx.c  Sort the clean TX queue
 *
 * Copyright (c) 2018 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * The Mochimo Project System Software
 *
 * Date 10 January 2018
*/


word32 Ntx;     /* number of transactions in clean TX queue */
word32 *Txidx;  /* malloc'd Txidx[] Ntx*4 bytes */
byte *Tx_ids;   /* malloc'd Tx_ids[] Ntx*32 bytes */

/* for sorting unsigned indexes:
 * #define SHELLFUN  (a[k - *gap] > temp)
 */

/* To sort the TX_ID's:
 * a[] is array of word32 indexes, temp is a word32 index
 */
#define SHELLFUN (memcmp(&Tx_ids[a[k - *gap]*32], &Tx_ids[temp*32], 32) > 0)

#include "sort.c"


/* Creates a malloc'd sort index:
 * word32 Txidx[Ntx] and the TX_ID list:
 * byte Tx_ids[Ntx * 32]
 *
 * Returns VERROR on file errors, else VEOK.
 */
int sorttx(char *fname)
{
   FILE *fp;
   long offset;
   byte *bp;
   unsigned j;

   fp = fopen(fname, "rb");
   if(fp == NULL) return error("sorttx(): missing %s", fname);
   if(fseek(fp, 0, SEEK_END) != 0) {
bad:
      if(Tx_ids) free(Tx_ids);
      if(Txidx) free(Txidx);
      Tx_ids = NULL;
      Txidx = NULL;
      fclose(fp);
      return error("I/O error on %s", fname);
   }
   offset = ftell(fp);
   if((offset % sizeof(TXQENTRY)) != 0) goto bad;  /* check record sizes */

   /* compute number of transactions in file */
   Ntx = offset / sizeof(TXQENTRY);
   if(Ntx == 0) goto out;

   /* seek to first TX record of file */
   if(fseek(fp, 0, SEEK_SET) != 0) goto bad;

   /* Allocate array for sort */
   Txidx = malloc(Ntx * 4);  /* (word32 *) */
   Tx_ids = malloc(Ntx * HASHLEN);  /* (Ntx * 32) */

   if(Txidx == NULL || Tx_ids == NULL) return VERROR;

   /* Read each (pre-computed) TX_ID into Tx_ids[Ntx][32]
    * and initialise index Txidx[] = 0,1,2,3,4,5,...
    */
   for(j = 0, bp = Tx_ids; j < Ntx; j++, bp += HASHLEN) {
      /* seek down in transaction record to tx_id[] */
      if(fseek(fp, sizeof(TXQENTRY) - HASHLEN, SEEK_CUR) != 0) goto bad;
      /* reading tx_id[] puts us at start of next record */
      if(fread(bp, 1, HASHLEN, fp) != HASHLEN) goto bad;
      Txidx[j] = j;
   }

   /* sort the index */
   shell(Txidx, Ntx);
out:
   fclose(fp);
   return VEOK;
}  /* end sorttx() */
