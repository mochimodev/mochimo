/**
 * @private
 * @headerfile peer.h <peer.h>
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

#ifndef MOCHIMO_PEER_C
#define MOCHIMO_PEER_C  /* include guard */


#include "peer.h"

/* internal support */
#include "error.h"

/* external support */
#include <string.h>
#include <stdlib.h>
#include "extlib.h"
#include "extinet.h"

/* Recent peers list */
word32 Rplist[RPLISTLEN] = {0};
word32 Rplistidx = 0;  /* Recent peer list */

/* pink lists of EVIL IP addresses read in from disk */
word32 Cpinklist[CPINKLEN] = {0};
word32 Cpinkidx = 0;
word32 Lpinklist[LPINKLEN] = {0};
word32 Lpinkidx = 0;
word32 Epinklist[EPINKLEN] = {0};
word32 Epinkidx = 0;

word8 Nopinklist = 0;  /* disable pinklist IP's when set */
word8 Noprivate = 0;   /* filter out private IP's when set v.28 */

/**
 * Search a list[] of 32-bit unsigned integers for a non-zero value.
 * A zero value marks the end of list (zero cannot be in the list).
 * Returns NULL if not found, else a pointer to value. */
word32 *search32(word32 val, word32 *list, unsigned len)
{
   for( ; len; len--, list++) {
      if(*list == 0) break;
      if(*list == val) return list;
   }
   return NULL;
}

/**
 * Remove bad from list[maxlen]
 * Returns 0 if bad is not in list, else bad.
 * NOTE: *idx queue index is adjusted if idx is non-NULL. */
word32 remove32(word32 bad, word32 *list, unsigned maxlen, word32 *idx)
{
   word32 *bp, *end;

   bp = search32(bad, list, maxlen);
   if(bp == NULL) return 0;
   if(idx && &list[*idx] > bp) idx[0]--;
   for(end = &list[maxlen - 1]; bp < end; bp++) bp[0] = bp[1];
   *bp = 0;
   return bad;
}

/**
 * Append a non-zero 32-bit unsigned integer to a list[].
 * Returns 0 if val was not added, else val.
 * NOTE: *idx queue index is always adjusted, as idx is required. */
word32 include32(word32 val, word32 *list, unsigned len, word32 *idx)
{
   if(idx == NULL || val == 0) return 0;
   if(search32(val, list, len) != NULL) return 0;
   if(idx[0] >= len) idx[0] = 0;
   list[idx[0]++] = val;
   return val;
}

/**
 * Shuffle a list of < 64k 32-bit unsigned integers using Durstenfeld's
 * implementation of the Fisher-Yates shuffling algorithm.
 * NOTE: the shuffling length limitation is due to rand16(). */
void shuffle32(word32 *list, word32 len)
{
   word32 *ptr, *p2, temp;

   if (len < 2) return; /* list length too short to shuffle, bail */
   while (list[--len] == 0 && len > 0);  /* get non-zero list length */
   for(ptr = &list[len]; len > 1; len--, ptr--) {
      p2 = &list[rand16() % len];
      temp = *ptr;
      *ptr = *p2;
      *p2 = temp;
   }
}

/**
 * Returns non-zero if ip is private, else 0. */
int isprivate(word32 ip)
{
   word8 *bp;

   bp = (word8 *) &ip;
   if(bp[0] == 10) return 1;  /* class A */
   if(bp[0] == 172 && bp[1] >= 16 && bp[1] <= 31) return 2;  /* class B */
   if(bp[0] == 192 && bp[1] == 168) return 3;  /* class C */
   if(bp[0] == 169 && bp[1] == 254) return 4;  /* auto */
   if (bp[0] == 127) return 5;  /* loopback */
   if (bp[0] == 0) return 6;  /* reserved */
   return 0;  /* public IP */
}

word32 addpeer(word32 ip, word32 *list, word32 len, word32 *idx)
{
   if(ip == 0) return 0;
   if(Noprivate && isprivate(ip)) return 0;  /* v.28 */
   if(search32(ip, list, len) != NULL) return 0;
   if(*idx >= len) *idx = 0;
   list[idx[0]++] = ip;
   return ip;
}

int loadpeers(word32 *dstipl, int dstlen, word32 *srcipl, int srclen)
{
   int i;

   for (i = 0; i < dstlen && i < srclen; i++) {
      if (srcipl[i] == 0) break;
      dstipl[i] = srcipl[i];
   }

   return i;
}

void print_ipl(word32 *list, word32 len)
{
   unsigned int j;

   for(j = 0; j < len && list[j]; j++) {
      if((j % 4) == 0) printf("\n");
      printf("   %-15.15s", ntoa(&list[j], NULL));
   }

   printf("\n\n");
}

/**
 * Save the Rplist[] list to disk.
 * Returns VEOK on success, else VERROR */
int save_ipl(char *fname, word32 *list, word32 len)
{
   const char preface[] = "# Peer list (saved by node)\n";
   char ipaddr[18];
   word32 j;
   FILE *fp;

   pdebug("saving %s...", fname);

   /* open file for writing */
   fp = fopen(fname, "w");
   if (fp == NULL) {
      perrno("fopen(%s) failed", fname);
      return VERROR;
   };

   /* write preface */
   if (fwrite(preface, strlen(preface), 1, fp) != 1) goto IOERROR;

   /* save non-zero entries */
   for(j = 0; j < len; j++) {
      if (list[j] == 0) continue;
      ntoa(&list[j], ipaddr);
      strncat(ipaddr, "\n", 2);
      if (fwrite(ipaddr, strlen(ipaddr), 1, fp) != 1) goto IOERROR;
   }

   fclose(fp);
   plog("%s saved", fname);
   return VEOK;
IOERROR:
   fclose(fp);
   remove(fname);
   perr("*** %s I/O write error", fname);
   return VERROR;
}  /* end save_ipl() */

/**
 * Read an IP list file, fname, into plist.
 * Valid lines in IP list include:
 *    host.domain.name
 *    1.2.3.4
 * @returns Number of peers read into list, else (-1) on error
*/
int read_ipl(char *fname, word32 *plist, word32 plistlen, word32 *plistidx)
{
   char buff[128];
   word32 count;
   FILE *fp;

   pdebug("reading %s...", fname);
   count = 0;

   /* check valid fname and open for reading */
   if (fname == NULL || *fname == '\0') return (-1);
   fp = fopen(fname, "r");
   if (fp == NULL) return (-1);

   /* read file line-by-line */
   while(fgets(buff, 128, fp)) {
      if (strtok(buff, " #\r\n\t") == NULL) break;
      if (*buff == '\0') continue;
      if (addpeer(aton(buff), plist, plistlen, plistidx)) {
         pdebug("added %s from %s", buff, fname);
         count++;
      }
   }
   /* check for read errors */
   if (ferror(fp)) perr("*** %s I/O error", fname);

   fclose(fp);
   return count;
}  /* end read_ipl() */


int pinklisted(word32 ip)
{
   if(Nopinklist) return 0;

   if(search32(ip, Cpinklist, CPINKLEN) != NULL
      || search32(ip, Lpinklist, LPINKLEN) != NULL
      || search32(ip, Epinklist, EPINKLEN) != NULL)
         return 1;
   return 0;
}

/**
 * Add ip address to current pinklist.
 * Call pinklisted() first to check if already on list.
 */
int cpinklist(word32 ip)
{
   if(Cpinkidx >= CPINKLEN)
      Cpinkidx = 0;
   Cpinklist[Cpinkidx++] = ip;
   return VEOK;
}

/**
 * Add ip address to current pinklist and remove it from
 * current and recent peer lists.
 * Checks the list first...
 */
int pinklist(word32 ip)
{
   pdebug("%s pink-listed", ntoa(&ip, NULL));

   if(!pinklisted(ip)) {
      if(Cpinkidx >= CPINKLEN)
         Cpinkidx = 0;
      Cpinklist[Cpinkidx++] = ip;
   }
   if(!Nopinklist) {
      remove32(ip, Rplist, RPLISTLEN, &Rplistidx);
   }
   return VEOK;
}  /* end pinklist() */


/**
 * Add ip address to last pinklist.
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
      pdebug("Epoch pink list overflow");
      Epinkidx = 0;
   }
   Epinklist[Epinkidx++] = ip;
   return VEOK;
}


/**
 * Call after each epoch.
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

/**
 * Erase Epoch Pink List */
void purge_epoch(void)
{
   pdebug("   purging epoch pink list");
   remove("epink.lst");
   memset(Epinklist, 0, sizeof(Epinklist));
   Epinkidx = 0;
}

/* end include guard */
#endif
