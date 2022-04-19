/* ledger.c  Open, close, and search ledger.dat
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * The Mochimo Project System Software
*/

/* include guard */
#ifndef MOCHIMO_LEDGER_C
#define MOCHIMO_LEDGER_C


#include <errno.h>
#include "extint.h"
#include "extprint.h"
#include "types.h"

static FILE *Lefp;
static unsigned long Nledger;

/* Open ledger "ledger.dat" */
int le_open(char *ledger, char *fopenmode)
{
   unsigned long offset;

   /* Already open? */
   if(Lefp) return VEOK;
   Nledger = 0;
   Lefp = fopen(ledger, fopenmode);
   if(Lefp == NULL)
      return perrno(errno, "le_open(): Cannot open ledger");
   if(fseek(Lefp, 0, SEEK_END)) goto bad;
   offset = ftell(Lefp);
   if(offset < sizeof(LENTRY) || (offset % sizeof(LENTRY)) != 0) goto bad;
   Nledger = offset / sizeof(LENTRY);  /* number of ledger entries */
   return VEOK;
bad:
   fclose(Lefp);
   Lefp = NULL;
   return perr("le_open(): Bad ledger I/O format");
}  /* end le_open() */


void le_close(void)
{
   if(Lefp == NULL) return;
   fclose(Lefp);
   Lefp = NULL;
   Nledger = 0;
}


/* Binary search ledger.dat (Lefp) for addr.
 * input: addr
 * outputs: *le, *position, and return code.
 * Returns 1 if found, 0 if not found.
 * If found, le is filled in with ledger entry.
 * If position is non-NULL put the index of found LENTRY struct there,
 * else the index of where to insert addr in ledger.dat.
 */
int le_find(word8 *addr, LENTRY *le, long *position, word16 len)
{
   long cond, mid, hi, low;
   size_t addrlen;

   if(Lefp == NULL) {
      perr("le_find(): use le_open() first!");
      return 0;
   }

   low = 0;
   hi = Nledger - 1;
   addrlen = len < 2 ? TXADDRLEN : len;

   while(low <= hi) {
      mid = (hi + low) / 2;
      if(fseek(Lefp, mid * sizeof(LENTRY), SEEK_SET) != 0)
         { perr("le_find(): fseek");  break; }
      if(fread(le, 1, sizeof(LENTRY), Lefp) != sizeof(LENTRY))
         { perrno(errno, "le_find(): fread");  break; }
      cond = memcmp(addr, le->addr, addrlen);
      if(cond == 0) {
         if(position) *position = mid;
         return 1;  /* found target addr */
      }
      if(cond < 0) hi = mid - 1; else low = mid + 1;
   }  /* end while */
   /* Not found.
    * To add target addr, move ledger[position] up and insert target
    * at ledger[position].
    */
   if(position) *position = low;
   return 0;  /* not found */
}  /* end le_find() */


/* Extract the ledger from a neo-genesis block and
 * put it in ledger file lfile (ledger.dat)
 * Return VEOK on success, else VERROR.
 */
int le_extract(char *fname, char *lfile)
{
   word32 hdrlen;    /* to read-in block header length */
   FILE *fp, *lfp;
   LENTRY le;        /* buffer to read ledger entry */
   word8 prevaddr[TXADDRLEN];  /* to check block addr sort */
   word8 first;

   if(Trace) plog("le_extract() ledger from %s to %s", fname, lfile);

   /* open the neo-genesis block and read in file header length */
   fp = fopen(fname, "rb");
   if(!fp) return VERROR;;
   if(fread(&hdrlen, 1, 4, fp) != 4) goto ioerror;

   lfp = fopen(lfile, "wb");
   if(!lfp) {
      perr("le_extract(): Cannot open %s", lfile);
      goto ioerror;
   }

   /* Make sure that NG header contains at least
    * one ledger entry.
    */
   if(hdrlen < (sizeof(LENTRY) + 4)) {
      perr("le_extract(): Not a neo-genesis block: %s", fname);
      goto error2;
   }

   /*
    * Read the ledger from fp and copy it to lfp,
    * creating a new ledger.dat file.
    * NOTE: block trailer must be less than sizeof(LENTRY)
    */
   if(fseek(fp, 4, SEEK_SET)) goto error2;
   for(hdrlen -= 4, first = 1; ; first = 0) {
      if(fread(&le, 1, sizeof(LENTRY), fp) != sizeof(LENTRY))
         break;
      hdrlen -= sizeof(LENTRY);
      /* check ledger sort in NG block */
      if(!first && memcmp(le.addr, prevaddr, TXADDRLEN) <= 0) {
         perr("le_extract(): bad ledger sort in neo-genesis block");
         goto error2;
      }
      memcpy(prevaddr, le.addr, TXADDRLEN);
      if(fwrite(&le, 1, sizeof(LENTRY), lfp) != sizeof(LENTRY))
         goto error2;
   }
   if(hdrlen) {
      perr("le_extract(): bad neo-genesis block length");
      goto error2;
   }
   fclose(fp);
   fclose(lfp);
   return VEOK;
ioerror:
      fclose(fp);
      unlink(lfile);  /* remove bad ledger */
      return perr("le_extract() failed!");
error2:
   fclose(lfp);
   goto ioerror;
}  /* end le_extract() */

/*
   [<---2196 Bytes Address--->][<--12 Bytes Tag-->]

   12-byte tag:  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                 ^
              type byte
*/

word8 *Tagidx;    /* array of all 12-word8 tags in ledger order */
word32 Ntagidx;  /* number of tags in Tagidx[] */
#define BAIL(m) { message = m; goto bail; }


/* Release tag index */
void tag_free(void)
{
   if(Tagidx != NULL) free(Tagidx);
   Tagidx = NULL;
   Ntagidx = 0;
}


/* Build the tag index, Tagidx[].
 * Return VEOK if success, else error code.
 */
int tag_buildidx(void)
{
   FILE *fp;
   LENTRY le;
   int message;
   word32 n;
   word8 *tp;

   if(Trace) plog("tag_buildidx()");
   if(Tagidx != NULL) return VEOK;  /* index already made */

   fp = fopen("ledger.dat", "rb");
   if(fp == NULL) BAIL(1);
   fseek(fp, 0L, SEEK_END);
   Ntagidx = ftell(fp) / sizeof(le);
   Tagidx = (word8 *) malloc(Ntagidx * TXTAGLEN);
   if(Tagidx == NULL) BAIL(2);  /* no memory */
   fseek(fp, 0L, SEEK_SET);
   for(tp = Tagidx, n = 0; n < Ntagidx; n++, tp += TXTAGLEN) {
      if(fread(&le, sizeof(le), 1, fp) != 1) break;   /* EOF */
      memcpy(tp, ADDR_TAG_PTR(le.addr), TXTAGLEN);
   }
   if(n != Ntagidx) BAIL(3);  /* I/O error likely */
   fclose(fp);
   if(Trace) plog("tag_buildidx() success: Ntagidx = %u", Ntagidx);
   return VEOK;  /* index built */
bail:
   if(fp != NULL) fclose(fp);
   if(Tagidx != NULL) free(Tagidx);
   Tagidx = NULL;
   Ntagidx = 0;
   perr("tag_buildidx(): BAIL(%d)\007", message);  /* should not happen */
   return message;
}  /* end tag_buildidx() */


/* Search txq1.dat and txclean.dat for a tag matching tag of addr in
 * some pending TX's change address.
 * Return VEOK if the tag is found in a chg_addr, otherwise VERROR.
 */
int tag_qfind(word8 *addr)
{
   FILE *fp;
   TXQENTRY tx;
   word8 *tag, *txtag;

   tag = ADDR_TAG_PTR(addr);
   txtag = ADDR_TAG_PTR(tx.chg_addr);

   fp = fopen("txq1.dat", "rb");
   if(fp != NULL) {
      for(;;) {
         if(fread(&tx, 1, sizeof(TXQENTRY), fp) != sizeof(TXQENTRY)) break;
         if(memcmp(tag, txtag, TXTAGLEN) == 0) {
            fclose(fp);
            return VEOK;  /* found */
         }
      }  /* end for */
      fclose(fp);
   }  /* end if fp */

   /* Stop Block Constructor who uses txclean.dat. */
   if(Bcpid) {
      kill(Bcpid, SIGTERM);
      waitpid(Bcpid, NULL, 0);
      Bcpid = 0;
   }

   fp = fopen("txclean.dat", "rb");
   if(fp != NULL) {
      for(;;) {
         if(fread(&tx, 1, sizeof(TXQENTRY), fp) != sizeof(TXQENTRY)) break;
         if(memcmp(tag, txtag, TXTAGLEN) == 0) {
            fclose(fp);
            return VEOK;  /* found */
         }
      }  /* end for */
      fclose(fp);
   }  /* end if fp */
   return VERROR;  /* tag not found */
}  /* end tag_qfind() */


#if TXTAGLEN != 12
   TXTAGLEN must be 12 for tag code in tag.c tag_find()
#endif

/* Find the tag of addr in Tagidx[].
 * If foundaddr or balance is not NULL, copy the
 * full fields from ledger.dat to foundaddr and/or balance.
 * Return VEOK if tag found, VERROR if not found, or
 * some other internal error code.
 */
int tag_find(word8 *addr, word8 *foundaddr, word8 *balance, size_t len)
{
   FILE *fp;
   word8 *tag, *tp;
   LENTRY le;
   word32 n;
   int message;

   fp = NULL;  /* for bail */
   if(Tagidx == NULL) tag_buildidx();
   if(Tagidx == NULL) BAIL(2);  /* 2 > VERROR */

   tag = ADDR_TAG_PTR(addr);
   /* Search tag index, Tagidx[] for tag. */
   for(tp = Tagidx, n = 0; n < Ntagidx; n++, tp += TXTAGLEN) {
      /* compare tag in Tagidx[] to tag */
      if(  /* partial tag len search */
         (len > 1 && len < TXTAGLEN && memcmp(tp, tag, len) == 0)
         || (  /* full tag len search (about 9 instructions in asm) */
            *((word32 *) tp)       == *((word32 *) tag)
         && *((word32 *) (tp + 4)) == *((word32 *) (tag + 4))
         && *((word32 *) (tp + 8)) == *((word32 *) (tag + 8))) ) {
         /* tag found */
         if(foundaddr != NULL || balance != NULL) {
            /* and caller wants ledger entry... */
            fp = fopen("ledger.dat", "rb");
            if(fp == NULL) BAIL(3);
            /* n is record number in ledger.dat */
            if(fseek(fp, n * sizeof(le), SEEK_SET)) BAIL(4);
            if(fread(&le, sizeof(le), 1, fp) != 1) BAIL(5);
            if(memcmp(ADDR_TAG_PTR(le.addr), tag, len)) BAIL(6);
            fclose(fp);
            if(foundaddr != NULL) memcpy(foundaddr, le.addr, TXADDRLEN);
            if(balance != NULL) memcpy(balance, le.balance, TXAMOUNT);
         }  /* end if copy entry */
         return VEOK;  /* found tag! */
      }  /* end if memcmp found */
   }  /* end for tp -- search for tag */
   return VERROR;  /* tag not found */
bail:
   if(fp != NULL) fclose(fp);
   tag_free();  /* Erase the bad index */
   perr("tag_find(): BAIL(%d)\007", message);  /* should not happen */
   return message;
}  /* end tag_find() */

/* end include guard */
#endif
