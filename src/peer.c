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
#include "network.h"
#include "error.h"

/* external support */
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "exttime.h"
#include "extthrd.h"
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
   if (isprivate(ip)) {
      pdebug("%s is private", ntoa(&ip, NULL));
      pdebug("   not pink-listed");
      return VEOK;
   }

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
   if (isprivate(ip)) {
      pdebug("%s is private", ntoa(&ip, NULL));
      pdebug("   not pink-listed");
      return VEOK;
   }

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
   if (isprivate(ip)) {
      pdebug("%s is private", ntoa(&ip, NULL));
      pdebug("   not pink-listed");
      return VEOK;
   }

   if(Lpinkidx >= LPINKLEN)
      Lpinkidx = 0;
   Lpinklist[Lpinkidx++] = ip;
   return VEOK;
}


int epinklist(word32 ip)
{
   if (isprivate(ip)) {
      pdebug("%s is private", ntoa(&ip, NULL));
      pdebug("   not pink-listed");
      return VEOK;
   }

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

/* ----------------------------------------------------------------
 * Provisional Peer Management
 * ---------------------------------------------------------------- */

/* provisional peer list and synchronization */
static PROVPEER Provlist[PROVPEERSLEN];
static word32 Provcount = 0;
static RWLock Provlock = RWLOCK_INITIALIZER;

/* verification thread state */
static Thread Provthread;
static volatile int Provrunning = 0;

/**
 * @private
 * Check source reputation within the recent time window.
 * Must be called with at least a read lock held.
 * @param source_ip IP of the source to evaluate
 * @param now Current time
 * @return 1 if source has bad reputation, 0 if acceptable
 */
static int source_is_bad(word32 source_ip, word32 now)
{
   word32 i, total, failures, last_attempt;

   total = failures = 0;
   for (i = 0; i < Provcount; i++) {
      if (Provlist[i].source_ip != source_ip) continue;
      if (Provlist[i].status == PROVSTATUS_EXPIRED) {
         /* derive last_attempt from next_attempt and fail_count */
         last_attempt = Provlist[i].next_attempt
            - (PROVBACKOFF * Provlist[i].fail_count);
         /* only count entries within the reputation window */
         if (now > last_attempt && (now - last_attempt) > PROVREPUTIME) {
            continue;
         }
         total++;
         failures++;
      } else if (Provlist[i].status == PROVSTATUS_PENDING) {
         total++;
      }
   }
   if (total < PROVREPUTHR) return 0;
   if ((failures * 100 / total) >= PROVREPUFAIL) return 1;
   return 0;
}

/**
 * Add an IP to the provisional peer list.
 * Deduplicates against existing provisional entries and Rplist.
 * Checks source reputation -- silently drops IPs from sources with
 * high failure rates in the recent time window.
 * @param ip Candidate peer IP address
 * @param source_ip IP of the peer that advertised this candidate
 * @return 0 on success (appended or deduplicated), -1 if list full
 */
int addprovisional(word32 ip, word32 source_ip)
{
   word32 i, now;

   if (ip == 0) return 0;

   rwlock_wrlock(&Provlock);

   /* check if already in provisional list */
   for (i = 0; i < Provcount; i++) {
      if (Provlist[i].ip == ip) {
         rwlock_wrunlock(&Provlock);
         return 0;  /* duplicate */
      }
   }

   /* check if already in Rplist */
   if (search32(ip, Rplist, RPLISTLEN) != NULL) {
      rwlock_wrunlock(&Provlock);
      return 0;  /* already a known peer */
   }

   /* check source reputation */
   now = (word32) time(NULL);
   if (source_is_bad(source_ip, now)) {
      rwlock_wrunlock(&Provlock);
      return 0;  /* source has bad reputation */
   }

   /* check capacity */
   if (Provcount >= PROVPEERSLEN) {
      rwlock_wrunlock(&Provlock);
      return -1;  /* list full */
   }

   /* append new entry */
   memset(&Provlist[Provcount], 0, sizeof(PROVPEER));
   Provlist[Provcount].ip = ip;
   Provlist[Provcount].source_ip = source_ip;
   Provlist[Provcount].status = PROVSTATUS_PENDING;
   Provcount++;

   rwlock_wrunlock(&Provlock);
   return 0;
}  /* end addprovisional() */

/**
 * Harvest verified peers from the provisional list into Rplist.
 * Scans for entries with status=VERIFIED, calls addrecent() for each,
 * and clears them. Also compacts expired entries to reclaim space.
 * @return Number of peers promoted, or -1 on error
 */
int harvest_provisional(void)
{
   word32 i, j, promoted;

   rwlock_wrlock(&Provlock);

   promoted = 0;
   for (i = 0; i < Provcount; i++) {
      if (Provlist[i].status == PROVSTATUS_VERIFIED) {
         addrecent(Provlist[i].ip);
         promoted++;
         Provlist[i].status = PROVSTATUS_EXPIRED;  /* mark for removal */
      }
   }

   /* compact: remove expired entries by shifting remaining down */
   j = 0;
   for (i = 0; i < Provcount; i++) {
      if (Provlist[i].status == PROVSTATUS_EXPIRED) continue;
      if (i != j) memcpy(&Provlist[j], &Provlist[i], sizeof(PROVPEER));
      j++;
   }
   Provcount = j;

   rwlock_wrunlock(&Provlock);
   return (int) promoted;
}  /* end harvest_provisional() */

/**
 * @private
 * Verification thread routine.
 * Processes pending entries in batches, sleeping between passes. */
static ThreadProc provpeer_thread(void *arg)
{
   NODE node;
   word32 i, ip, now, batch;
   int ecode;

   (void) arg;

   while (Provrunning && Running) {
      batch = 0;

      for (i = 0; i < PROVPEERSLEN && batch < PROVBATCHSIZE; i++) {
         if (!Provrunning || !Running) break;

         /* read lock to check entry */
         rwlock_rdlock(&Provlock);
         if (i >= Provcount) {
            rwlock_rdunlock(&Provlock);
            break;
         }
         if (Provlist[i].status != PROVSTATUS_PENDING) {
            rwlock_rdunlock(&Provlock);
            continue;
         }
         now = (word32) time(NULL);
         if (Provlist[i].next_attempt > now) {
            rwlock_rdunlock(&Provlock);
            continue;
         }
         ip = Provlist[i].ip;
         rwlock_rdunlock(&Provlock);

         /* attempt connection (blocking -- OK in background thread) */
         memset(&node, 0, sizeof(NODE));
         ecode = callserver(&node, ip);
         if (ecode == VEOK) {
            sock_close(node.sd);
            node.sd = INVALID_SOCKET;
         }

         /* update entry with result */
         rwlock_wrlock(&Provlock);
         if (i < Provcount && Provlist[i].ip == ip) {
            if (ecode == VEOK) {
               Provlist[i].status = PROVSTATUS_VERIFIED;
            } else {
               Provlist[i].fail_count++;
               now = (word32) time(NULL);
               Provlist[i].next_attempt =
                  now + (PROVBACKOFF * Provlist[i].fail_count);
               if (Provlist[i].fail_count >= PROVMAXFAILS) {
                  Provlist[i].status = PROVSTATUS_EXPIRED;
               }
            }
         }
         rwlock_wrunlock(&Provlock);
         batch++;
      }  /* end for */

      /* sleep between passes (check Running every second) */
      for (i = 0; i < 30 && Provrunning && Running; i++) {
         millisleep(1000);
      }
   }  /* end while */

   return 0;
}  /* end provpeer_thread() */

/**
 * Start the background provisional peer verification thread.
 * The thread processes up to PROVBATCHSIZE entries per pass,
 * attempting callserver() on each pending entry whose
 * next_attempt time has passed. Sleeps between passes.
 * @return 0 on success, -1 on error
 */
int start_provisional_verifier(void)
{
   if (Provrunning) return 0;  /* already running */
   Provrunning = 1;
   if (thread_create(&Provthread, provpeer_thread, NULL) != 0) {
      Provrunning = 0;
      return -1;
   }
   return 0;
}  /* end start_provisional_verifier() */

/**
 * Stop the background provisional peer verification thread.
 * Signals the thread to exit and waits for it to finish. */
void stop_provisional_verifier(void)
{
   if (!Provrunning) return;
   Provrunning = 0;
   thread_join(Provthread);
}  /* end stop_provisional_verifier() */

/**
 * Clear the entire provisional peer list. */
void purge_provisional(void)
{
   rwlock_wrlock(&Provlock);
   memset(Provlist, 0, sizeof(Provlist));
   Provcount = 0;
   rwlock_wrunlock(&Provlock);
}  /* end purge_provisional() */

/* end include guard */
#endif
