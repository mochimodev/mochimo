

#include "extinet.h"
#include "extint.h"
#include "config.h"
#include "types.h"

/* The Node struct */
typedef struct {
   TX tx;            /* packet buffer */
   word32 ip;        /* source ip *//*
   word16 port;      // unused... */
   word16 id1, id2;  /* from tx handshake */
   char id[22];      /* "0.0.0.0 AB~EF" - for logging identification */
   pid_t pid;        /* process id of child -- zero if empty slot */
   volatile int ts;  /* thread status -- set by thread */
   SOCKET sd;
} NODE;


NODE Nodes[MAXNODES];  /* data structure for connected NODE's     */
NODE *Hi_node = Nodes; /* points one beyond last logged in NODE   */

word32 Rplist[RPLISTLEN];  /* recent peer list */
word32 Rplistidx;
word32 Cplist[CPLISTLEN];  /* current peer list */
word32 Cplistidx;
/* LAN peer list */
#define LPLISTLEN 32
word32 Lplist[LPLISTLEN] = { 0 };
word32 Splist[RPLISTLEN+LPLISTLEN] = {0};

#define CORELISTLEN 16
#if CORELISTLEN > RPLISTLEN
#error Fix CORELISTLEN
#endif
word32 Coreplist[CORELISTLEN] = {  /* ip's of the Core Network */
   0x0100007f,    /* local host  debug */
};

int Quorum = 4;         /* Number of peers in get_eon() gang[MAXQUORUM] */
word8 Ininit;            /* non-zero when init() runs */
word8 Insyncup;          /* non-zero when syncup() runs */
word8 Safemode;          /* Safe mode enable */
word8 Nominer;           /* Do not start miner if true -n */
word32 Watchdog;        /* enable watchdog timeout -wN */
time_t Utime;           /* update time for watchdog */
word8 Betabait;          /* betabait() display */
word8 Cbits = CBITS;     /* 8 capability bits */
time_t Pushtime;        /* time of last OP_MBLOCK */
word8 Allowpush;         /* set by -P flag in mochimo.c */

#define recentip(ip) search32(ip, Rplist, RPLISTLEN)
#define currentip(ip) search32(ip, Cplist, CPLISTLEN)

word8 Noprivate;  /* filter out private IP's when set v.28 */

/* Returns non-zero if ip is private, else 0. */
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


void addrecent(word32 ip)
{
   if(ip == 0) return;
   if(Noprivate && isprivate(ip)) return;  /* v.28 */
   if(search32(ip, Rplist, RPLISTLEN) != NULL) return;
   if(Rplistidx >= RPLISTLEN) Rplistidx = 0;
   Rplist[Rplistidx++] = ip;
}

void addcurrent(word32 ip)
{
   if(ip == 0) return;
   if(Noprivate && isprivate(ip)) return;  /* v.28 */
   if(search32(ip, Cplist, CPLISTLEN) != NULL) return;
   if(Cplistidx >= CPLISTLEN) Cplistidx = 0;
   Cplist[Cplistidx++] = ip;
}


/*
 * Save Rplist[] list to disk.
 */
int save_rplist(void)
{
   int j;

   if(Trace) plog("saving recent peer list...");

   /* save non-zero entries */
   for(j = 0; j < RPLISTLEN; j++)
      if(Rplist[j] == 0) break;

   write_data(Rplist, j * 4, "rplist.lst");
   return VEOK;
}  /* end save_rplist() */


/* Thanks David! */
int existsnz(char *fname)
{
   FILE *fp;
   long len;

   fp = fopen(fname, "rb");
   if(!fp) return 0;
   fseek(fp, 0, SEEK_END);
   len = ftell(fp);
   fclose(fp);
   if(len == 0) return 0;
   return 1;
}


/* Search an array list[] of word32's for a non-zero value.
 * A zero value marks the end of list (zero cannot be in the list).
 * Returns NULL if not found, else a pointer to value.
 */
word32 *search32(word32 val, word32 *list, unsigned len)
{
   for( ; len; len--, list++) {
      if(*list == 0) break;
      if(*list == val) return list;
   }
   return NULL;
}


/* Remove bad from list[maxlen]
 * Returns: 0 if bad is not in list.
 *          bad if removed.
 *
 * NOTE: *idx queue index is adjusted if idx is non-NULL.
*/
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


/* shuffle a list of < 64k word32's using Durstenfeld's algorithm */
void shuffle32(word32 *list, word32 len)
{
   word32 *ptr, *p2, temp, listlen = len;

   if(len < 2) return;
   for(ptr = &list[len - 1]; len > 1; len--, ptr--) {
      p2 = &list[rand16() % listlen];
      temp = *ptr;
      *ptr = *p2;
      *p2 = temp;
   }
}

