/* sortlt.c  Sort the ledger transaction queue
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * The Mochimo Project System Software
 *
 * Date 10 January 2018
 *
 * exit() with 1 on errors, else 0.
*/

#include "config.h"
#include "mochimo.h"
#define closesocket(_sd) close(_sd)

#define EXCLUDE_NODES

byte Errorlog = 1;  /* since not including data.c */
byte Bgflag;
byte Monitor;
byte Running = 1;
word32 Trace = 1;
word32 Nsolved;
pid_t Mpid, Sendfound_pid;  /* in error.c */

#include "error.c"
#include "daemon.c"

word32 Nlt;     /* number of transactions in ltran.dat */
word32 *Ltidx;  /* malloc'd Ltidx[] Nlt * 4 bytes */
LTRAN *Ltrans;  /* malloc'd Ltrans[] Nlt * sizeof(LTRAN) bytes */

/* for sorting unsigned indexes:
 * #define SHELLFUN  (a[k - *gap] > temp)
 */

/* Comparison function to sort index to ledger transactions: LTRAN Ltrans[]:
 * a[] is array of word32 indexes, temp is a word32 index
 * include Ltrans[].trancode[0] in key: (+1)
 */
#define SHELLFUN \
  (memcmp(Ltrans[a[k - *gap]].addr, Ltrans[temp].addr, TXADDRLEN+1) > 0)

#include "sort.c"


/* get memory or exit */
void *tmalloc(size_t len) {
   void *ptr;

   ptr = malloc(len);
   if(ptr == NULL) {
      fatal2(1, "sortlt.c: No memory!");
   }
   return ptr;
}


/* Creates a malloc'd sort index:
 * word32 Ltidx[Nlt] and the transaction list:
 * LTRAN Ltrans[Nlt]
 *
 * Returns VERROR on file errors, else VEOK.
 */
int sortlt(char *fname)
{
   FILE *fp;
   long offset;
   unsigned j;

   fix_signals();
   close_extra();

   fp = fopen(fname, "r+b");
   if(fp == NULL) return error("sortlt(): missing %s", fname);
   if(fseek(fp, 0, SEEK_END) != 0) {
bad:
      if(Ltrans) free(Ltrans);
      if(Ltidx) free(Ltidx);
      Ltrans = NULL;
      Ltidx = NULL;
      fclose(fp);
      return error("I/O error on %s", fname);
   }
   offset = ftell(fp);
   if((offset % sizeof(LTRAN)) != 0) goto bad;  /* check record sizes */

   /* compute number of transactions in file */
   Nlt = offset / sizeof(LTRAN);
   if(Nlt == 0) goto out;

   /* seek to first TX record of file */
   if(fseek(fp, 0, SEEK_SET) != 0) goto bad;

   /* Allocate array for sort */
   Ltidx = tmalloc(Nlt * 4);  /* (word32 *) */
   Ltrans = tmalloc(Nlt * sizeof(LTRAN));  /* (LTRAN *) */

   /* Read in transaction file */
   if(fread(Ltrans, sizeof(LTRAN), Nlt, fp) != Nlt) goto bad;
      
   /* and initialise index Ltidx[] = 0,1,2,3,4,5,...  */
   for(j = 0; j < Nlt; j++)
      Ltidx[j] = j;

   /* sort the index */
   shell(Ltidx, Nlt);

   /* write the file back out in sorted order */
   if(fseek(fp, 0, SEEK_SET) != 0) goto bad;
   for(j = 0; j < Nlt; j++) {
      if(fwrite(&Ltrans[Ltidx[j]], 1, sizeof(LTRAN), fp) != sizeof(LTRAN))
         goto bad;
   }
out:
   fclose(fp);
   return VEOK;
}  /* end sortlt() */


int main(int argc, char **argv)
{
   if(argc != 2) {
      printf("\nusage:  sortlt ltran.dat\n"
             "Called before block update.\n\n");
      exit(1);
   }

   if(sortlt(argv[1]) == VEOK) return 0;
   return 1;
}
