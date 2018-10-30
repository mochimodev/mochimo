/* pink.c  pink list functions to screen IP addresses.
 *
 * Copyright (c) 2018 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 9 January 2018
*/


/* pink lists of EVIL IP addresses read in from disk */
word32 Cpinklist[CPINKLEN];
word32 Lpinklist[LPINKLEN];
word32 Epinklist[EPINKLEN];
word32 Cpinkidx, Lpinkidx, Epinkidx;

/* Re-read epoch pink list from init(). */
int readpink(void)
{
   if(Trace) plog("reading epoch pink list...");
   return readlist32(Epinklist, 4, EPINKLEN, "epink.lst", &Epinkidx);
}


/*
 * Save pink lists to disk.
 */
int savepink(void)
{
   int j;

   if(Trace) plog("saving epoch pink list...");

   /* save non-zero entries */
   for(j = 0; j < EPINKLEN; j++)
      if(Epinklist[j] == 0) break;

   write_data(Epinklist, j * 4, "epink.lst");
   return VEOK;
}  /* end savepink() */


int pinklisted(word32 ip)
{
   if(Disable_pink) return 0;   /* for debug */

   if(search32(ip, Cpinklist, CPINKLEN) != NULL
      || search32(ip, Lpinklist, LPINKLEN) != NULL
      || search32(ip, Epinklist, EPINKLEN) != NULL)
         return 1;
   return 0;
}


/* Add ip address to current pinklist.
 * Call pinklisted() first to check if already on list.
 */
int cpinklist(word32 ip)
{
   if(Cpinkidx >= CPINKLEN)
      Cpinkidx = 0;
   Cpinklist[Cpinkidx++] = ip;
   return VEOK;
}

/* Add ip address to current pinklist and remove it from
 * current and recent peer lists.
 * Checks the list first...
 */
int pinklist(word32 ip)
{
   if(Trace)
      plog("%s pink-listed", ntoa((byte *) &ip));

   if(!pinklisted(ip)) {
      if(Cpinkidx >= CPINKLEN)
         Cpinkidx = 0;
      Cpinklist[Cpinkidx++] = ip;
   }
   if(!Disable_pink) {
      remove32(ip, Rplist, RPLISTLEN, &Rplistidx);
      remove32(ip, Cplist, CPLISTLEN, &Cplistidx);
   }
   return VEOK;
}  /* end pinklist() */


/* Add ip address to last pinklist.
 * Caller checks if already on list.
 */
int lpinklist(word32 ip)
{
   if(Lpinkidx >= LPINKLEN)
      Lpinkidx = 0;
   Lpinklist[Lpinkidx++] = ip;
   return VEOK;
}


int epinklist(word32 ip)
{
   if(Epinkidx >= EPINKLEN) {
      if(Trace) plog("Epoch pink list overflow");
      Epinkidx = 0;
   }
   Epinklist[Epinkidx++] = ip;
   return VEOK;
}


/* Call after each epoch.
 * Merges current pink list into last pink list
 * and purges current pink list.
 */
void mergepinklists(void)
{
   int j;
   word32 ip, *ptr;

   for(j = 0; j < CPINKLEN; j++) {
      ip = Cpinklist[j];
      if(ip == 0) continue;  /* empty */
      ptr = search32(ip, Lpinklist, LPINKLEN);
      if(ptr == NULL) lpinklist(ip);  /* add to last bad list */
      Cpinklist[j] = 0;
   }
   Cpinkidx = 0;
}


/* Erase Epoch Pink List */
void purge_epoch(void)
{
   if(Trace) plog("   purging epoch pink list");
   unlink("epink.lst");
   memset(Epinklist, 0, sizeof(Epinklist));
   Epinkidx = 0;
}
