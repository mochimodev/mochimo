/* ledger.c  Search, read, and write to ledger.dat
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * The Mochimo Project System Software
*/


FILE *Lefp;
unsigned long Nledger;
byte Lerror;  /* set if any errors on ledger -- sticky bit */


/* Open ledger "ledger.dat" */
int le_open(char *ledger, char *fopenmode)
{
   unsigned long offset;

   /* Already open? */
   if(Lefp) return VEOK;
   Nledger = 0;
   Lefp = fopen(ledger, fopenmode);
   if(Lefp == NULL)
      return (Lerror = error("le_open(): Cannot open ledger"));
   if(fseek(Lefp, 0, SEEK_END)) goto bad;
   offset = ftell(Lefp);
   if(offset < sizeof(LENTRY) || (offset % sizeof(LENTRY)) != 0) goto bad;
   Nledger = offset / sizeof(LENTRY);  /* number of ledger entries */
   return VEOK;
bad:
   fclose(Lefp);
   Lefp = NULL;
   return (Lerror = error("le_open(): Bad ledger I/O format"));
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
int le_find(byte *addr, LENTRY *le, long *position, int mode)
{
   long cond, mid, hi, low;

   if(Lefp == NULL) {
      Lerror = error("le_find(): use le_open() first!");
      return 0;
   }

   low = 0;
   hi = Nledger - 1;

   while(low <= hi) {
      mid = (hi + low) / 2;
      if(fseek(Lefp, mid * sizeof(LENTRY), SEEK_SET) != 0)
         { Lerror = error("le_find(): fseek");  break; }
      if(fread(le, 1, sizeof(LENTRY), Lefp) != sizeof(LENTRY))
         { Lerror = error("le_find(): fread");  break; }
      if(mode == 1) {
         cond = memcmp(addr, le->addr, TXADDRLEN-12);
      } else {
         cond = memcmp(addr, le->addr, TXADDRLEN);
      }
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
