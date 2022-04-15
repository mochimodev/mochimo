/* bx.c  Block Explorer
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.TXT   **** NO WARRANTY ****
 *
 * Date: May 18 2018
 *
 * NOTE: compile with:  cc -o bx bx.c trigg.o sha256.o
*/

#include "extmath.h"    /* 64-bit math support */

#ifdef BX_MYSQL
#include <mysql.h>
#include <my_global.h>
#include <dirent.h>
#endif

#include "config.h"
#include "mochimo.h"
#include "rand.c"

#ifdef UNIXLIKE
#include <unistd.h>
#define CLEARSCR() system("clear")
#else
#define CLEARSCR() clrscr()
void clrscr(void);
typedef int pid_t;
#endif

#define ADDR_TAG_PTR(addr) (((byte *) addr) + 2196)
#define ADDR_TAG_LEN 12

/* Globals */
word32 Bnum;
FILE *Bfp;
word32 Hdrlen;
byte Maddr[TXADDRLEN];
byte Sigint;
word32 Txidx;
long Foffset;

void ctrlc(int sig)
{
   signal(SIGINT, ctrlc);
   Sigint = 1;
}


/* byte buffer access
 * little-endian compiler order
 */

word16 get16(void *buff)
{
   return *((word16 *) buff);
}

void put16(void *buff, word16 val)
{
   *((word16 *) buff) = val;
}

word32 get32(void *buff)
{
   return *((word32 *) buff);
}

void put32(void *buff, word32 val)
{
   *((word32 *) buff) = val;
}


/* buff<--val */
void put64(void *buff, void *val)
{
   ((word32 *) buff)[0] = ((word32 *) val)[0];
   ((word32 *) buff)[1] = ((word32 *) val)[1];
}


/* Prototypes */
char *trigg_check(byte *in, byte d, byte *bnum);


/* Find a binary string, s, of length, len, in file fp.
 * Caller sets seek position of fp before call.
 * If found, return offset of start of match, else return -1.
 */
long findtag(byte *s, int len, FILE *fp)
{
   byte *cp, c;
   int len2;

   for(cp = s, len2 = len; len2; ) {
      if(fread(&c, 1, 1, fp) != 1) return -1L;
      if(*cp != c) {
         cp = s;
         len2 = len;
         if(*cp == c) { cp++; len2--; }
         continue;
      }
      len2--;
      cp++;
   }
   return ftell(fp) - len;
}


/* Convert a hex ASCII string hex into a value. */
unsigned long htoul(char *hex)
{
   static char hextab[] = "0123456789abcdef";
   char *cp;
   unsigned long val;

   val = 0;
   for( ; *hex; hex++) {
      if(*hex == 'x' || *hex == 'X') continue;
      cp = strchr(hextab, tolower(*hex));
      if(!cp) break;
      val = (val * 16) + (cp - hextab);
   }
   return val;
}  /* end htoul() */


/* Convert ASCII string s into a value.
 * 0123 or 0x123 is hex, otherwise, s is decimal.
 */
unsigned long getval(char *s)
{
   if(s == NULL) return 0;
   while(*s && *s <= ' ') s++;
   if(*s == '\0') return 0;
   /* if(strchr(s, '.')) value is float */
   if(*s == '0') return htoul(s);
   return strtoul(s, NULL, 10);  /* for really big unsigned longs */
/*   return atol(s); */
}


/* bnum is little-endian on disk and core. */
char *bnum2hex(byte *bnum)
{
   static char buff[20];

   sprintf(buff, "%02x%02x%02x%02x%02x%02x%02x%02x",
                  bnum[7],bnum[6],bnum[5],bnum[4],
                  bnum[3],bnum[2],bnum[1],bnum[0]);
   return buff;
}


char *b2hex8(byte *amt)
{
   static char str[20];

   sprintf(str, "%02x%02x%02x%02x%02x%02x%02x%02x",
           amt[0], amt[1], amt[2], amt[3],
           amt[4], amt[5], amt[6], amt[7]);
   return str;
}


void b2hexch(byte *addr, int len, int lastchar)
{
   int n;

   for(n = 0; len; len--) {
      printf("%02x", *addr++);
      if(++n >= 36) {
         printf("\n");
         n = 0;
      }
   }
   if(lastchar)
      printf("%c", lastchar);
}

#define bytes2hex(addr, len) b2hexch(addr, len, '\n')

void disp_taddr(byte *addr)
{
   b2hexch(addr, 16, 0);
   printf("   Tag: ");
   bytes2hex(ADDR_TAG_PTR(addr), ADDR_TAG_LEN);
}


/* Seek to end of fname and read block trailer.
 * Return 0 on success, else error code.
 * fp is open and stays open.
 */
int readtrailer2(BTRAILER *trailer, FILE *fp)
{

   if(fseek(fp, -(sizeof(BTRAILER)), SEEK_END) != 0) {
bad:
      printf("Cannot read block trailer\n");
      return 1;
   }
   if(fread(trailer, 1, sizeof(BTRAILER), fp) != sizeof(BTRAILER))
      goto bad;
   return 0;
}


/* Return 0 on success.
 * Non-NULL filename over-rides bnum.
 */
int read_block(word32 bnum, BHEADER *bh, BTRAILER *bt, char *filename)
{
   char fnamebuff[100], *fname;
   int count;
   static byte bnum8[8];

   if(Bfp) fclose(Bfp);
   if(filename) fname = filename;
   else {
      fname = fnamebuff;
      put32(bnum8, bnum);
      sprintf(fname, "b%s.bc", bnum2hex(bnum8));
   }
   Bfp = fopen(fname, "rb");
   if(Bfp == NULL) {
      printf("Cannot open %s\n", fname);
      return 1;
   }
   Bnum = bnum;
   count = fread(&Hdrlen, 1, 4, Bfp);
   if(count != 4) {
err:
      printf("Error reading %s\n", fname);
      fclose(Bfp);
      Bfp = NULL;
      return 2;
   }
   memset(bh, 0, sizeof(BHEADER));
   put32(bh->hdrlen, Hdrlen);
   if(Hdrlen == sizeof(BHEADER) && Bnum != 0) {
      fseek(Bfp, 0, SEEK_SET);
      if(fread(bh, 1, sizeof(BHEADER), Bfp) != sizeof(BHEADER))
         goto err;
   }
   if(Bnum == 0) printf("%s is the Genesis Block.\n\n", fname);
   else {
      if((Bnum & 255) == 0)
         printf("%s is a neo-genesis block with %d ledger entries.\n\n",
                fname, (int) ((get32(bh->hdrlen) - 4) / sizeof(LENTRY)));
   }

   if(readtrailer2(bt, Bfp) != 0) goto err;
   return 0;  /* success */
}  /* end read_block() */


/* Convert nul-terminated hex string in[] to binary out[].
 * in and out may point to same space.
 * example: in[]   = { '0', '1', 'a', '0' }
 *          out[]: = { 1, 160 }
*/
int hex2bytes(char *in, char *out)
{
   char *hp;
   static char hextab[] = "0123456789abcdef";
   int j, len, val = 0;

   len = strlen(in);
   if(len & 1) return 0;  /* len should be even */
   for(j = 0; *in && len; in++, j++, len--) {
      hp = strchr(hextab, tolower(*in));
      if(!hp) break;  /* if non-hex */
      val = (val * 16) + (hp - hextab);  /* convert 4 bits per char */
      if(j & 1) *out++ = val;  /* done with this byte */
   }
   return j;  /* number of characters scanned */
}


#define I_ZSUP 1   /* zero suppress */

/* Format an 8-byte value into out for display. */
char *itoa64(void *val64, char *out, int dec, int flags)
{
   int count;
   static char s[24];
   char *cp, zflag = 1;
   word32 *tab;
   byte val[8];

   /* 64-bit little-endian */
   static word32 table[] = {
     0x89e80000, 0x8ac72304,      /* 1e19 */
     0xA7640000, 0x0DE0B6B3,      /* 1e18 */
     0x5D8A0000, 0x01634578,      /* 1e17 */
     0x6FC10000, 0x002386F2,      /* 1e16 */
     0xA4C68000, 0x00038D7E,      /* 1e15 */
     0x107A4000, 0x00005AF3,      /* 1e14 */
     0x4E72A000, 0x00000918,      /* 1e13 */
     0xD4A51000, 0x000000E8,      /* 1e12 */
     0x4876E800, 0x00000017,      /* 1e11 */
     0x540BE400, 0x00000002,      /* 1e10 */
     0x3B9ACA00, 0x00000000,      /* 1e09 */
     0x05F5E100, 0x00000000,      /* 1e08 */
     0x00989680, 0x00000000,      /* 1e07 */
     0x000F4240, 0x00000000,      /* 1e06 */
     0x000186A0, 0x00000000,      /* 1e05 */
     0x00002710, 0x00000000,      /* 1e04 */
     0x000003E8, 0x00000000,      /* 1e03 */
     0x00000064, 0x00000000,      /* 1e02 */
     0x0000000A, 0x00000000,      /* 1e01 */
     0x00000001, 0x00000000,      /*   1  */
   };

   if(out == NULL) cp = s; else cp = out;
   out = cp;  /* return value */
   if((flags & I_ZSUP) == 0) zflag = 0;  /* leading zero suppression flag */
   dec = 20 - (dec + 1);  /* where to put decimal point */
   put64(val, val64);

   for(tab = table; ; ) {
      count = 0;
      for(;;) {
         count++;
         if(sub64(val, tab, val) != 0) {
            count--;
            add64(val, tab, val);
            *cp = count + '0';
            if(*cp == '0' && zflag) *cp = ' '; else zflag = 0;
            cp++;
            if(dec-- == 0) *cp++ = '.';
            tab += 2;
            if(tab[0] == 1 && tab[1] == 0) {
               *cp = val[0] + '0';
               return out;
            }
            break;
         }
      }  /* end for */
   }  /* end for */
}  /* end itoa64() */


/* Left justify */
char *itoa64lj(void *val64, char *out, int dec, int flags)
{
   char *cp;

   cp = itoa64(val64, out, dec, flags);
   while(*cp && (*cp == ' ' || *cp == '.')) cp++;
   return cp;
}


/* Input a string to buff from stdin.
 * len > 2
 */
char *tgets(char *buff, int len)
{
   char *cp, fluff[16];

   *buff = '\0';
   fgets(buff, len, stdin);
   cp = strchr(buff, '\n');
   if(cp) *cp = '\0';
   else {
      for(;;) {
         if(fgets(fluff, 16, stdin) == NULL) break;
         if(strchr(fluff, '\n') != NULL) break;
      }
   }
   return buff;
}


void banner(void)
{
   printf("The Mochimo Block Explorer version 1.1\n\n");
}


char *timestr(word32 timeval)
{
  time_t t;
  static char out[32];
  char *cp;

  t = timeval;
  strcpy(out, asctime(gmtime(&t)));
  cp = strchr(out, '\n');
  if(cp) strcpy(cp, " GMT");
  return out;
}


void disp_bh(BHEADER *bh, BTRAILER *bt)
{
   char *haiku;

   CLEARSCR();
   banner();

   if(Bnum == 0) printf("This block is the Genesis Block.\n\n");
   else {
      if((Bnum & 255) == 0)
         printf("This block is a neo-genesis block.\n\n");
   }

   printf("Block:      %u  (0x%x)\n", get32(bt->bnum), get32(bt->bnum));
   printf("Hdrlen:     %u  (0x%x)\n", get32(bh->hdrlen), get32(bh->hdrlen));
   printf("Miner:      0x");  disp_taddr(bh->maddr);
   printf("Reward:     %s", itoa64lj(bh->mreward, NULL, 9, 1));
   printf("  [0x%s]\n", b2hex8(bh->mreward));
   printf("Hash:       0x");  bytes2hex(bt->bhash, HASHLEN);
   printf("Phash:      0x");  bytes2hex(bt->phash, HASHLEN);
   printf("TX count:   %u\n", get32(bt->tcount));
   printf("Solve time: %s\n", timestr(get32(bt->stime)));
   printf("Root:       0x");  bytes2hex(bt->mroot, HASHLEN);
   printf("Nonce:      0x");  bytes2hex(bt->nonce, HASHLEN);
   printf("Difficulty: %d\n", bt->difficulty[0]);

   haiku = trigg_check(bt->mroot, bt->difficulty[0], bt->bnum);
   if(haiku)
      printf("\n%s\n\n", haiku);
   else if(bt->bnum[0])
      printf("trigg_check() failed!\n");
}  /* end disp_bh() */


/* Hex converter */
void hexcon(void)
{
   char buff[81], *cp;
   unsigned long val;
   int n;

   for(;;) {
      printf("Enter value (e.g. 'string, or decimal 123, or hex 0123,"
             " p=previous):\n");
      tgets(buff, 80);
      if(*buff == '\'') {
         printf("0x");
         for(cp = buff + 1; *cp; cp++) printf("%02x", *cp);
         printf("\n");
         continue;
      }
      if(buff[0] < '0' || buff[0] > '9') break;
      val = getval(buff);
      printf("%lu  (0x%lx)  [0x%s]    ", val, val, b2hex8((byte *) &val));
      for(cp = (char *) &val, n = sizeof(val); n; n--, cp++) {
         if(*cp >= ' ' && *cp < 127) printf("%c", *cp); else printf(".");
      }
      printf("\n");
   }  /* end for */
}


/* tfile explorer */
int tfx(char *fname)
{
   FILE *fp;
   long offset, saveoff, lastfind;
   BTRAILER bt;
   int count;
   char buff[81];
   static char sbuff[81];
   int len = 2;
   word32 idx, j, nblocks;
   unsigned long flen;
   char *haiku;

   fp = fopen(fname, "rb");
   if(fp == NULL) return 1;

   fseek(fp, 0, SEEK_END);
   flen = ftell(fp);
   nblocks = flen / sizeof(BTRAILER);
   fseek(fp, 0, SEEK_SET);
   lastfind = 0;

   for(idx = 0; ; ) {
      CLEARSCR();
      printf("Tfile Explorer on %u block chain\n\n", nblocks);

      saveoff = ftell(fp);
      count = fread(&bt, 1, sizeof(BTRAILER), fp);
      if(count != sizeof(BTRAILER)) {
         memset(&bt, 0, sizeof(BTRAILER));
         printf("\nEnd of tfile.\n");
         goto getcmd;
      }

      printf("Block:      %u  (0x%x)\n", get32(bt.bnum), get32(bt.bnum));
      printf("Hash:       0x");  bytes2hex(bt.bhash, HASHLEN);
      printf("Phash:      0x");  bytes2hex(bt.phash, HASHLEN);
      printf("TX count:   %u\n", get32(bt.tcount));
      printf("Solve time: %s\n", timestr(get32(bt.stime)));
      printf("Root:       0x");  bytes2hex(bt.mroot, HASHLEN);
      printf("Nonce:      0x");  bytes2hex(bt.nonce, HASHLEN);
      printf("Difficulty: %d\n", bt.difficulty[0]);

      haiku = trigg_check(bt.mroot, bt.difficulty[0], bt.bnum);
      if(haiku)
         printf("\n%s\n\n", haiku);
      else if(bt.bnum[0])
         printf("trigg_check() failed!\n");

getcmd:
      printf("\nq=quit, g=goto, f=find, h=hex, b=back, RETURN=next: ");
      tgets(buff, 81);
      if(buff[0] == 'q') break;    /* quit */
      if(buff[0] == '\0') { idx++; continue; }  /* next index */
      if(buff[0] == 'h') {
         hexcon();
         goto getcmd;
      }
      if(buff[0] == 'b') {
         if(idx > 0) { j = idx - 1; goto goidx; }
         goto getcmd;
      }
 
      /* goto index command */
      if(buff[0] == 'g') {
getidx:
         printf("Enter tfile index [%u]: ", idx);
         tgets(buff, 81);
         if(buff[0]) j = getval(buff); else j = idx;
goidx:
         offset = j * sizeof(BTRAILER);
         if((unsigned long) offset > flen) {
            printf("Index out of range.\n");
            goto getidx;
         }
         lastfind = offset;
         fseek(fp, offset, SEEK_SET);
         idx = j;
         continue;
      }  /* end 'g' goto command */

      /* Find command */
      if(buff[0] == 'f') {
         printf("RETURN=find again or\n");
getsearch:
         printf("Enter up to 80 character hex string (e.g. 001a20):\n");
         bytes2hex((byte *) sbuff, len / 2);
         memset(buff, 0, 81);
         tgets(buff, 81);
         if(*buff) {
            len = strlen(buff);
            if(len & 1) {
               printf("Please enter an even number of characters.\n");
               goto getsearch;
            }
            memcpy(sbuff, buff, 81);
            hex2bytes(sbuff, sbuff);
            fseek(fp, 0, SEEK_SET);
         } else fseek(fp, lastfind, SEEK_SET);
         printf("Searching for %u byte(s) from offset %ld (0x%lx)...\n\n",
                len / 2, lastfind, lastfind);
         offset = findtag((byte *) sbuff, len / 2, fp);
         if(offset == -1) {
notfound:
            fseek(fp, saveoff, SEEK_SET);
            printf("Not found.\n\n");
            goto getcmd;
         }
         lastfind = offset + 1;
         idx = offset / sizeof(BTRAILER);
         printf("Found in tfile index %u at file offset %lu (0x%lx)\n",
                idx, offset, offset);
         fseek(fp, idx * sizeof(BTRAILER), SEEK_SET);
         goto getcmd;
      }  /* end if found command */
   }  /* end for */
   fclose(fp);
   return 0;
}  /* end tfx() */


/* Ledger explorer.
 * fp is open.
 * top = 0 means ledger.dat type files.
 * top = 4 means Genesis or NG blocks.
 * Returns error code or zero.
 */
int lx(FILE *fp, word32 top)
{
   long offset, saveoff, temp, lastfind;
   LENTRY le;
   int count;
   char buff[81];
   static char sbuff[81];
   int len = 2;
   word32 idx, j;
   unsigned long flen;

   if(fp == NULL) return 1;

   if(top == 4) {
      if(Bnum == 0) printf("Genesis Block ");
      else printf("neo-genesis block ");
   }
   printf("Ledger Search\n\n");

   fseek(fp, 0, SEEK_END);
   flen = ftell(fp);
   fseek(fp, top, SEEK_SET);
   lastfind = 0;

   for(idx = 0; ; ) {
      saveoff = ftell(fp);
      count = fread(&le, 1, sizeof(LENTRY), fp);
      if(count != sizeof(LENTRY)) {
         memset(&le, 0, sizeof(LENTRY));
         printf("\nEnd of ledger.\n");
         goto getcmd;
      }
      printf("Ledger index: %u\n", idx);
      printf("Address:\n");
      disp_taddr(le.addr);
      printf("Balance:   %s", itoa64lj(le.balance, NULL, 9, 1));
      printf("  [0x%s]\n", b2hex8(le.balance));
getcmd:
      printf("\nq=quit, g=goto, f=find, h=hex,"
             " p=previous menu, RETURN=next: ");
      tgets(buff, 81);
      if(buff[0] == 'q') exit(0);    /* quit */
      if(buff[0] == 'p') return 0;   /* previous menu */
      if(buff[0] == '\0') { idx++; continue; }  /* next index */
      if(buff[0] == 'h') {
         hexcon();
         goto getcmd;
      }
      /* goto index command */
      if(buff[0] == 'g') {
getidx:
         printf("Enter ledger index [%u]: ", idx);
         tgets(buff, 81);
         if(buff[0]) j = getval(buff); else j = idx;
         offset = (j * sizeof(LENTRY)) + top;
         if((unsigned long) offset > flen) {
            printf("Index out of range.\n");
            goto getidx;
         }
         lastfind = offset;
         fseek(fp, offset, SEEK_SET);
         idx = j;
         continue;
      }  /* end 'g' goto command */

      /* Find command */
      if(buff[0] == 'f') {
         printf("RETURN=find again or\n");
getsearch:
         printf("Enter up to 80 character hex string (e.g. 001a20):\n");
         bytes2hex((byte *) sbuff, len / 2);
         memset(buff, 0, 81);
         tgets(buff, 81);
         if(*buff) {
            len = strlen(buff);
            if(len & 1) {
               printf("Please enter an even number of characters.\n");
               goto getsearch;
            }
            memcpy(sbuff, buff, 81);
            hex2bytes(sbuff, sbuff);
            fseek(fp, top, SEEK_SET);
         } else fseek(fp, lastfind, SEEK_SET);
         printf("Searching for %u byte(s) from offset %ld (0x%lx)...\n\n",
                len / 2, lastfind, lastfind);
         offset = findtag((byte *) sbuff, len / 2, fp);
         if(offset == -1) {
notfound:
            fseek(fp, saveoff, SEEK_SET);
            printf("Not found.\n\n");
            continue;
         }
         lastfind = offset + 1;
         temp = offset - top;
         if(temp < 0) goto notfound;
         idx = temp / sizeof(LENTRY);
         printf("Found in entry index %u at file offset %lu (0x%lx)\n\n",
                idx, offset, offset);
         fseek(fp, (idx * sizeof(LENTRY)) + top, SEEK_SET);
         continue;
      }  /* end if found command */
   }  /* end for */
   return 0;
}  /* end lx() */


/* Explore a ledger.dat type file, lfile. */
int showledger(char *lfile)
{
   FILE *fp;
   int status;

   fp = fopen(lfile, "rb");
   if(!fp) {
      printf("Cannot open %s\n", lfile);
      return 1;
   }
   Bnum = 2;  /* not an NG block */
   status = lx(fp, 0);
   fclose(fp);
   return status;
}


int findmenu(BHEADER *bh, BTRAILER *bt)
{
   char buff[81];
   static char sbuff[81];
   static int len = 2;
   long offset, temp;
   word32 idx;
   byte again;

   offset = -1;
   for( ;; ) {
      Sigint = 0;
      printf("q=quit, p=previous menu, RETURN=find again\n");
      printf("Enter up to 80 character hex string (e.g. 001a20):\n");
      bytes2hex((byte *) sbuff, len / 2);
      memset(buff, 0, 81);
      tgets(buff, 81);
      if(buff[0] == 'q') exit(0);
      if(buff[0] == 'p') return 0;
      if(buff[0] == '\0') goto getb;
      len = strlen(buff);
      if(len & 1) {
         printf("Please enter an even number of characters.\n");
         continue;
      }
      memcpy(sbuff, buff, 81);
      hex2bytes(sbuff, sbuff);
getb:
      printf("Enter starting block number [%u (0x%x)]: ", Bnum, Bnum);
      tgets(buff, 80);
      again = 1;
      if(buff[0]) {
         Bnum = getval(buff);
         again = 0;
      }
readb:
      read_block(Bnum, bh, bt, NULL);
      if(Bfp == NULL) {
         printf("No match.\n");
         continue;  /* or error */
      }
      if(!again) offset = -1;
      fseek(Bfp, offset + 1, SEEK_SET);
      again = 0;
      offset = findtag((byte *) sbuff, len / 2, Bfp);
      if(offset == -1) {
         if(Sigint) continue;
         Bnum++;
         goto readb;
      }
      printf("Found at block %u (0x%x) offset %lu (0x%lx)\n",
             Bnum, Bnum,  offset, offset);

      /* Check if match is in TX array of non-NG block. */
      Foffset = offset;
      temp = offset - get32(bh->hdrlen);
      if(temp >= 0 && bt->bnum[0] != 0) {
         idx = temp / sizeof(TXQENTRY);
         if(idx < get32(bt->tcount)) {
            Txidx = idx;
            printf("Match is in TX idx %u\n", Txidx);
         }
      }

      continue;
   }  /* end for */
}  /* end findmenu(); */


/* Check if buff is all zeros */
int iszero(void *buff, int len)
{
   byte *bp;

   for(bp = buff; len; bp++, len--)
      if(*bp) return 0;

   return 1;
}

#ifdef BX_MYSQL
#include "bx-mysql/bx_mysql_export.c"
#endif

int txmenu(BHEADER *bh, BTRAILER *bt)
{
   char buff[80];
   TXQENTRY txq;
   word32 j, k;
   word32 tcount;
   MTX *mtx;

   CLEARSCR();
   banner();

   if(Bfp == NULL) {
      printf("Goto block first.\n");
      return 1;
   }
   /* check for neo-genesis block. */
   if(bt->bnum[0] == 0)
      return lx(Bfp, 4);

   tcount = get32(bt->tcount);
   if(tcount == 0) {
      printf("No transactions are in this block.\n");
      return 1;
   }
   if(Txidx >= tcount) Txidx = 0;

   for( ;; ) {
      CLEARSCR();
      fseek(Bfp, (sizeof(TXQENTRY) * Txidx) + get32(bh->hdrlen), SEEK_SET);
      fread(&txq, 1, sizeof(TXQENTRY), Bfp);
      printf("Transactions in block %u (0x%x)\n\n", Bnum, Bnum);

      printf("Tx index:   %d\n", Txidx);
      printf("Tx id:      0x");  bytes2hex(txq.tx_id, 32);
      printf("src_addr:   0x");  disp_taddr(txq.src_addr);

      if(ismtx(&txq)) {
         mtx = (MTX *) &txq;
         for(j = k = 0; j < NR_DST; j++) {
            if(iszero(mtx->dst[j].tag, ADDR_TAG_LEN)) break;
            if(++k >= 10) {
               k = 0;
               printf("Press RETURN or 'q': ");
               tgets(buff, 10);
               if(*buff == 'q') break;
            }
            printf("dst[%d] amount: %s",
                   j, itoa64(mtx->dst[j].amount, NULL, 9, 1));
            printf("  tag: 0x");
            b2hexch(mtx->dst[j].tag, ADDR_TAG_LEN, ' ');
            if(mtx->zeros[j]) printf("*");  /* tag was not found in ledger */
            printf("\n");
         }  /* end for j */
      } else {
         printf("dst_addr:   0x");  disp_taddr(txq.dst_addr);
      }

      printf("chg_addr:   0x");  disp_taddr(txq.chg_addr);
      printf("send total:   %s", itoa64lj(txq.send_total, NULL, 9, 1));
      printf("  [0x%s]\n", b2hex8(txq.send_total));
      printf("change total: %s", itoa64lj(txq.change_total, NULL, 9, 1));
      printf("  [0x%s]\n", b2hex8(txq.change_total));
      printf("fee:          %s\n", itoa64lj(txq.tx_fee, NULL, 9, 1));
      printf("sig:        0x");  bytes2hex(txq.tx_sig, 32);

      printf("\nq=quit, g=goto TX, RETURN=next, b=back, "
             "p=previous menu: "
      );
      tgets(buff, 80);
      switch(buff[0]) {
getidx:
         case 'g':
            printf("Enter TX index [%u]: ", Txidx);
            tgets(buff, 80);
            if(buff[0]) j = getval(buff); else j = Txidx;
            if(j >= get32(bt->tcount)) {
               printf("Index out of range.\n");
               goto getidx;
            }
            Txidx = j;
            break;
         case 0:
            if(Txidx < (get32(bt->tcount) - 1)) Txidx++;
            break;
         case 'b':
            if(Txidx > 0) Txidx--;
            break;
         case 'p': return -1;  /* previous menu */
         case 'q': exit(0);    /* exit */
      }  /* end switch */
   }  /* end for */
}  /* end txmenu(); */


void mainmenu(void)
{
   unsigned long bnum;
   char buff[80];
   BHEADER bh;
   BTRAILER bt;

   CLEARSCR();
   banner();

   signal(SIGINT, ctrlc);

   for( ;; ) {
      printf("\nq=quit, g=goto block, RETURN=next, b=back, "
             "f=find, h=hex, t=TX menu: "
      );
      tgets(buff, 80);
      switch(buff[0]) {
         case 'h':
            hexcon();
            break;
         case 'g':
            printf("Enter block number [%u (0x%x)]: ", Bnum, Bnum);
            tgets(buff, 80);
            if(buff[0]) Bnum = getval(buff);
getblock:
            CLEARSCR();
            if(read_block(Bnum, &bh, &bt, NULL) == 0)
               disp_bh(&bh, &bt);
            break;
         case 0:
            { Bnum++; goto getblock; }  /* RETURN key */
         case 'b':   
            if(Bnum > 0) { Bnum--;  goto getblock; }
            break;
         case 'f':
            findmenu(&bh, &bt);
            break;
         case 'n':   break;
         case 't':
            txmenu(&bh, &bt);
            break;
         case 'q':
         case 'Q':   return;
      }  /* end switch */
   }  /* end for */
}  /* end mainmenu(); */


void usage(void)
{
   printf("\nUsage: bx [-option] [file]\n"
      "options:\n"
      "           -l  file is a ledger to explore.\n"
      "           -t  file is a tfile to explore.\n\n"
#ifdef BX_MYSQL
      "           -e  export path for block files.\n"
#endif
      "\n"
      "If file is a block, dump its header and trailer to stdout.\n"
   );
   exit(1);
}


/* Display block from command line -- called from main() */
int display_block(char *fname)
{
   BHEADER bh;
   BTRAILER bt;
   int status;

   Bnum = 1;
   status = read_block(Bnum, &bh, &bt, fname);
   if(status == 0) {
      disp_bh(&bh, &bt);
      Bnum = get32(bt.bnum);
   }
   exit(status);
}


int main(int argc, char **argv)
{
   int j;
   static char lflag, tflag, eflag;

   printf("\n");
   banner();

   /*
    * Parse command line arguments.
    */
   for(j = 1; j < argc; j++) {
      if(argv[j][0] != '-') break;
      switch(argv[j][1]) {
         case 'l':  lflag = 1;       /* file is ledger */
                    break;
         case 't':  tflag = 1;       /* file is tfile */
                    break;
#ifdef BX_MYSQL
         case 'e':  eflag = 1;
                    break;
#endif
         default:   usage();
      }  /* end switch */
   }  /* end for j */
   if(j < argc || eflag) {
      if(tflag) exit(tfx(argv[j]));
      if(lflag) exit(showledger(argv[j]));
#ifdef BX_MYSQL
      if(eflag) exit(export(argv[j]));
#endif
      if(display_block(argv[j])) usage();
   }
   mainmenu();
}  /* end main() */
