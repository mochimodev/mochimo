/**
 * @private
 * @headerfile peer.h <peer.h>
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_PEER_C
#define MOCHIMO_PEER_C


#include "peer.h"

/* external support */
#include <string.h>
#include "extlib.h"

/** Current pink list. Intended reset on block update */
word32 Cpinklist[CPINKLEN] = { 0 };
word32 Cpinkidx = 0;
/** Epoch peer list. Intended reset on every epoch (EPOCHMASK) */
word32 Epinklist[EPINKLEN] = { 0 };
word32 Epinkidx = 0;
/** Local peer list. Read in from disk, and preserved */
word32 Lplist[LPLISTLEN] = { 0 };
word32 Lplistidx = 0;
/** Recent peer list. Rotates with certain successful connections */
word32 Rplist[RPLISTLEN] = { 0 };
word32 Rplistidx = 0;

/** Disable pinklist IP's when set */
word8 Nopinklist_opt;
/** Filter out private IP's when set. No effect on Lplist (local) */
word8 Noprivate_opt;

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
   return 0;  /* public IP */
}

word32 addpeer(word32 ip, word32 *list, word32 len, word32 *idx)
{
   if(ip == 0) return 0;
   if(Noprivate_opt && isprivate(ip)) return 0;  /* v.28 */
   if(search32(ip, list, len) != NULL) return 0;
   if(*idx >= len) *idx = 0;
   list[idx[0]++] = ip;
   return ip;
}

word32 addpeer_d(word32 ip, word32 **dlist, word32 *len, word32 *idx)
{
   void *ptr;
   word32 i;

   if (ip == 0) return 0;
   if (Noprivate_opt && isprivate(ip)) return 0;  /* v.28 */
   if (search32(ip, *dlist, *len) != NULL) return 0;
   if (*idx >= *len) {
      ptr = realloc(*dlist, sizeof(word32) * (*len + 32));
      if (ptr == NULL) return 0;
      *len += 32;
      *dlist = ptr;
      /* zero new section of list */
      for (i = *idx; i < *len; i++) (*dlist)[i] = 0;
   }
   /* add peer to dynamic list */
   (*dlist)[(*idx)++] = ip;

   return ip;
}  /* end addpeer_d() */

/**
 * Save a peer list to disk.
 * @returns (int) value representing operation status
 * @retval VEOK on success
 * @retval VERROR on error, check errno for details
*/
int save_ipl(char *fname, word32 *list, word32 len)
{
   static char preface[] = "# Peer list (built by node)\n";
   char ipaddr[16];  /* for threadsafe ntoa() usage */
   word32 j;
   FILE *fp;

   /* open file for writing */
   fp = fopen(fname, "w");
   if (fp == NULL) return VERROR;

   /* save non-zero entries */
   for(j = 0; j < len && list[j] != 0; j++) {
      ntoa(&list[j], ipaddr);
      if ((j == 0 && fwrite(preface, strlen(preface), 1, fp) != 1) ||
         (fwrite(ipaddr, strlen(ipaddr), 1, fp) != 1) ||
         (fwrite("\n", 1, 1, fp) != 1)) {
         fclose(fp);
         remove(fname);
         return VERROR;
      }
   }

   fclose(fp);
   return VEOK;
}  /* end save_ipl() */

/**
 * Read an IP list file, fname, into plist.
 * Valid lines in IP list include:
 * - host.domain.name
 * - 1.2.3.4
 * @returns Number of peers read into list, else (-1) on error
*/
int read_ipl(char *fname, word32 *plist, word32 plistlen, word32 *plistidx)
{
   char buff[128];
   word32 count;
   FILE *fp;

   /* init */
   count = 0;

   /* check valid fname and open for reading */
   if (fname == NULL || *fname == '\0') return (-1);
   fp = fopen(fname, "r");
   if (fp == NULL) return (-1);
   /* read file line-by-line */
   while(fgets(buff, 128, fp)) {
      if (strtok(buff, " #\r\n\t") == NULL) break;
      if (*buff == '\0') continue;
      count += include32(aton(buff), plist, plistlen, plistidx);
   }
   /* check for read errors */
   if (ferror(fp)) {
      fclose(fp);
      return VERROR;
   }

   fclose(fp);
   return count;
}  /* end read_ipl() */


int pinklisted(word32 ip)
{
   if(Nopinklist_opt) return 0;

   if(search32(ip, Cpinklist, CPINKLEN) != NULL
      || search32(ip, Epinklist, EPINKLEN) != NULL)
         return 1;
   return 0;
}

/**
 * Add ip address to current pinklist and remove it from
 * current and recent peer lists.
 * Checks the list first...
 */
void pinklist(word32 ip)
{
   if(!pinklisted(ip)) {
      if(Cpinkidx >= CPINKLEN) Cpinkidx = 0;
      Cpinklist[Cpinkidx++] = ip;
   }
   if(!Nopinklist_opt) {
      remove32(ip, Rplist, RPLISTLEN, &Rplistidx);
   }
}  /* end pinklist() */


void epinklist(word32 ip)
{
   if(Epinkidx >= EPINKLEN) {
      /* set_errno(EMCM_EPINK_OVERFLOW); */
      Epinkidx = 0;
   }
   Epinklist[Epinkidx++] = ip;
}

/**
 * Purges current pink list.
 */
void purge_pinklist(void)
{
   memset(Cpinklist, 0, sizeof(Cpinklist));
   Cpinkidx = 0;
}

/**
 * Erase Epoch Pink List
*/
void purge_epinklist(void)
{
   remove("epink.lst");
   memset(Epinklist, 0, sizeof(Epinklist));
   Epinkidx = 0;
}

/* end include guard */
#endif
