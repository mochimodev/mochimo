/* wallet.c  Prototype wallet
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 15 March 2018
 * Revised: 1 Sep 2018
 *
 * NOTE: To compile without a script:
 *
 *       viz. Borland C++ 5.5.
 *       bcc32 -DWIN32 -c sha256.c wots/wots.c
 *       bcc32 -DWIN32 wallet.c wots.obj sha256.obj
 *
 *       Unix-like:
 *       cc -DUNIXLIKE -DLONG64 -c sha256.c wots/wots.c
 *       cc -o wallet -DUNIXLIKE -DLONG64 wallet.c wots.o sha256.o
 *                                   ^
 *                  Remove if your longs are not 64-bit.
*/

#include "extlib.h"     /* general support */
#include "extinet.h"    /* socket support */
#include "extmath.h"    /* 64-bit math support */

#include "config.h"

#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdarg.h>
#include <time.h>
#include <string.h>
#include <signal.h>
#include <errno.h>

#define VEOK        0      /* No error                    */
#define VERROR      1      /* General error               */
#define VEBAD       2      /* client was bad              */

#ifdef UNIXLIKE
#include <unistd.h>
#define CLEARSCR() system("clear")
#else
#define CLEARSCR() clrscr()
void clrscr(void);
typedef int pid_t;
unsigned sleep(unsigned seconds);
#endif

#define OP_HELLO          1
#define OP_HELLO_ACK      2
#define OP_TX             3
#define OP_GET_IPL         6
#define OP_BALANCE        12
#define OP_RESOLVE        14

#define W_TAG    1
#define W_SEC    2
#define W_SPENT  4
#define W_BAL    8
#define W_DEL    16
#define W_ALL    255

#define TXNETWORK 0x0539
#define TXEOT     0xabcd
#define TXADDRLEN 2208
#define TXAMOUNT  8
#define TXSIGLEN  2144  /* WOTS */
#define HASHLEN 32
#define TXAMOUNT 8
#define RNDSEEDLEN 64
#define TXBUFF(tx)   ((word8 *) tx)
#define TXBUFFLEN    ((2*5) + (8*2) + 32 + 32 + 32 + 2 \
                        + (TXADDRLEN*3) + (TXAMOUNT*3) + TXSIGLEN + (2+2) )
#define TRANBUFF(tx) ((tx)->src_addr)
#define TRANLEN      ( (TXADDRLEN*3) + (TXAMOUNT*3) + TXSIGLEN )
#define TXSIGHASH_COUNT (TRANLEN - TXSIGLEN)

#define CRC_BUFF(tx) TXBUFF(tx)
#define CRC_COUNT   (TXBUFFLEN - (2+2))  /* tx buff less crc and trailer */
#define CRC_VAL_PTR(tx)  ((tx)->crc16)

#define ADDR_TAG_PTR(addr) (((word8 *) addr) + 2196)
#define TXTAGLEN 12
#define ADDR_HAS_TAG(addr) \
   (((word8 *) (addr))[2196] != 0x42 && ((word8 *) (addr))[2196] != 0x00)

#include "crypto/hash/cpu/sha256.h"      /* also defines word32 */
#include "crypto/wots/wots.h"   /* TXADDRLEN */


/* Wallet header */
typedef struct {
   word8 sig[4];
   word8 lastkey[4];
   word8 seed[RNDSEEDLEN];   /* seed to generate random bytes for addresses */
   word8 naddr[4];
   word8 name[25];
} WHEADER;


/* Wallet entry */
typedef struct {
   word8 key[4];
   word8 flags[1];
   word8 balance[8];
   word8 name[25];
   word8 addr[TXADDRLEN];
   word8 secret[32];
} WENTRY;


/* Wallet index entry */
typedef struct {
   word8 key[4];
   word8 flags[1];
   word8 balance[8];
   word8 name[25];
   word8 tag[TXTAGLEN];
} WINDEX;


/* Multi-byte numbers are little-endian.
 * Structure is checked on start-up for byte-alignment.
 * HASHLEN is checked to be 32.
 */
typedef struct {
   word8 version[2];  /* 0x01, 0x00 PVERSION  */
   word8 network[2];  /* 0x39, 0x05 TXNETWORK */
   word8 id1[2];
   word8 id2[2];
   word8 opcode[2];
   word8 cblock[8];        /* current block num  64-bit */
   word8 blocknum[8];      /* block num for I/O in progress */
   word8 cblockhash[32];   /* sha-256 hash of our current block */
   word8 pblockhash[32];   /* sha-256 hash of our previous block */
   word8 weight[32];       /* sum of block difficulties */
   word8 len[2];  /* length of data in transaction buffer for I/O op's */
   /* start transaction buffer */
   word8 src_addr[TXADDRLEN];
   word8 dst_addr[TXADDRLEN];
   word8 chg_addr[TXADDRLEN];
   word8 send_total[TXAMOUNT];
   word8 change_total[TXAMOUNT];
   word8 tx_fee[TXAMOUNT];
   word8 tx_sig[TXSIGLEN];
   /* end transaction buffer */
   word8 crc16[2];
   word8 trailer[2];  /* 0xcd, 0xab */
} TX;


/* stripped-down NODE for rx2() and callserver(): */
typedef struct {
   TX tx;  /* transaction buffer */
   word16 id1;      /* from tx */
   word16 id2;      /* from tx */
   int opcode;      /* from tx */
   word32 src_ip;
   SOCKET sd;
   pid_t pid;     /* process id of child -- zero if empty slot */
} NODE;

#include "crypto/crc16.c"
#include "crypto/xo4.c"     /* crypto */

void crctx(TX *tx)
{
   put16(CRC_VAL_PTR(tx), crc16(CRC_BUFF(tx), CRC_COUNT));
}


/* bnum is little-endian on disk and core. */
char *bnum2hex(word8 *bnum)
{
   static char buff[20];

   sprintf(buff, "%02x%02x%02x%02x%02x%02x%02x%02x",
                  bnum[7],bnum[6],bnum[5],bnum[4],
                  bnum[3],bnum[2],bnum[1],bnum[0]);
   return buff;
}


/* Globals */
#define PASSWLEN 188
char Password[PASSWLEN];
#define WFNAMELEN 256
char Wfname[WFNAMELEN] = "wallet.wal";  /* wallet file name */
XO4CTX Xo4ctx;
WHEADER Whdr;        /* wallet header */
WINDEX *Windex;      /* wallet index */
word32 Nindex;       /* number of addresses in wallet */
word8 Needcleanup;    /* for Winsock */
word32 Mfee[2] = { MFEE, 0 };
word8 Zeros[8];
word32 Port = 2095;  /* default server port */
char *Peeraddr;  /* peer address string optional, set on command line */
char *Corefile = "startnodes.lst"; /* Uses startnodes.lst if available */
unsigned Nextcore;  /* index into Coreplist for callserver() */
word8 Verbose;       /* output trace messages */
word8 Default_tag[TXTAGLEN]
   = { 0x42, 0, 0, 0, 0x0e, 0, 0, 0, 1, 0, 0, 0 };
word8 Cblocknum[8];  /* set from network */

#define CORELISTLEN 9

#if CORELISTLEN > RPLISTLEN
#error Fix CORELISTLEN
#endif

/* ip's of the Core Network */
word32 Coreplist[RPLISTLEN] = {
   0x0100007f,    /* local host */
};


word8 Sigint;

void ctrlc(int sig)
{
   signal(SIGINT, ctrlc);
   Sigint = 1;
   printf("Received signal %i.\n", sig);
}


/* shuffle a list of < 64k word32's */
void shuffle32(word32 *list, word32 len)
{
   word32 *ptr, *p2, temp;

   if(len < 2) return;
   for(ptr = &list[len - 1]; len > 1; len--, ptr--) {
      p2 = &list[rand16fast() % len];
      temp = *ptr;
      *ptr = *p2;
      *p2 = temp;
   }
}


/* Search an array list[] of word32's for a non-zero value.
 * A zero value marks the end of list (zero cannot be in the list).
 */
word32 *search32(word32 val, word32 *list, unsigned len)
{
   for( ; len; len--, list++) {
      if(*list == 0) break;
      if(*list == val) return list;
   }
   return NULL;
}

/* Taken & modified from str2ip.c */
word32 str2ip(char *addrstr)
{
   struct hostent *host;
   struct sockaddr_in addr;

   if(addrstr == NULL) return 0;

   memset(&addr, 0, sizeof(addr));
   if(addrstr[0] < '0' || addrstr[0] > '9') {
      host = gethostbyname(addrstr);
      if(host == NULL)
         return 0;
      memcpy((char *) &(addr.sin_addr.s_addr),
             host->h_addr_list[0], host->h_length);
   }
   else
      addr.sin_addr.s_addr = inet_addr(addrstr);

   return addr.sin_addr.s_addr;
}  /* end str2ip() */


/* Read-in the core ip list text file
 * each line: 1.2.3.4  or  host.domain.name
 */
int read_coreipl(char *fname)
{
   FILE *fp;
   char buff[128];
   int j;
   char *addrstr;
   word32 ip;

   if(fname == NULL || *fname == '\0') return VERROR;
   fp = fopen(fname, "rb");
   if(fp == NULL) return VERROR;
   for(j = 0; j < CORELISTLEN; ) {
      if(fgets(buff, 128, fp) == NULL) break;
      if(*buff == '#') continue;
      addrstr = strtok(buff, " \r\n\t");
      if(addrstr == NULL) break;
      ip = str2ip(addrstr);
      if(!ip) continue;
      /* put ip in Coreplist[j] */
      Coreplist[j++] = ip;
   }
   fclose(fp);
   return VEOK;
}  /* end read_coreipl() */

void bytes2hex(word8 *addr, int len, int lastchar)
{
   int n;
   
   for(n = 0; len; len--) {
      printf("%02x", *addr++);
      if(++n >= 36) {
         printf("\n");
         n = 0;
      }
   }
   if(lastchar) printf("%c", lastchar);
}


/* Check if buff is all zeros */
int iszero(void *buff, int len)
{
   word8 *bp;

   for(bp = buff; len; bp++, len--)
      if(*bp) return 0;

   return 1;
}


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


/* Display terminal error message
 * and exit.
 */
void fatal(char *fmt, ...)
{
   va_list argp;

   fprintf(stdout, "wallet: ");
   va_start(argp, fmt);
   vfprintf(stdout, fmt, argp);
   va_end(argp);
   printf("\n");
#ifdef _WINSOCKAPI_
    if(Needcleanup)
       WSACleanup();
#endif
   exit(2);
}


#ifdef FIONBIO
/* Set socket sd to non-blocking I/O on Win32 */
int nonblock(SOCKET sd)
{
   u_long arg = 1L;

   return ioctlsocket(sd, FIONBIO, (u_long FAR *) &arg);
}

#else
#include <fcntl.h>

/* Set socket sd to non-blocking I/O
 * Returns -1 on error.
 */
int nonblock(SOCKET sd)
{
   int flags;

   flags = fcntl(sd, F_GETFL, 0);
   return fcntl(sd, F_SETFL, flags | O_NONBLOCK);
}

#endif


int disp_ecode(int ecode)
{
   switch(ecode) {
      case VEOK:    printf("\nOk!\n");        break;
      case VEBAD:   printf("\nBad peer!\n");  break;
      case VERROR:
      default:
                    printf("\n***\n");   break;
   }
   return ecode;
}


int badidx(unsigned idx)
{
   if(idx == 0 || idx > Nindex) {
      printf("\nInvalid index.\n");
      return 1;
   }
   return 0;
}


/* Modified from connect2.c to use printf()
 * Returns: sd = a valid socket number on successful connect, 
 *          else INVALID_SOCKET (-1)
 */
SOCKET connectip2(word32 ip, char *addrstr)
{
   SOCKET sd;
   struct hostent *host;
   struct sockaddr_in addr;
   word16 port;
   char *name;
   time_t timeout;

   if((sd = socket(AF_INET, SOCK_STREAM, 0)) == INVALID_SOCKET) {
      printf("connectip(): cannot open socket.\n");
      return INVALID_SOCKET;
   }

   port = Port;
   memset((char *) &addr, 0, sizeof(addr));
   name = "";
   if(addrstr) {
      name = addrstr;
      if(addrstr[0] < '0' || addrstr[0] > '9') {
         host = gethostbyname(addrstr);
         if(host == NULL) {
            printf("connectip(): gethostbyname() failed.\n");
            return INVALID_SOCKET;
         }
         memcpy((char *) &(addr.sin_addr.s_addr),
                host->h_addr_list[0], host->h_length);
      }
      else
         addr.sin_addr.s_addr = inet_addr(addrstr);
   } else {
      addr.sin_addr.s_addr = ip;
   }  /* end if NULL addrstr */

   addr.sin_family = AF_INET;  /* AF_UNIX */
   /* Convert short integer to network byte order */
   addr.sin_port = htons(port);

   if(Verbose) {
      if(name[0])
         printf("Trying %s port %d...  ", name, port);
      else
         printf("Trying %s port %d...  ", ntoa(&ip, NULL), port);
   }

   nonblock(sd);
   timeout = time(NULL) + 3;
   Sigint = 0;

retry:
   if(connect(sd, (struct sockaddr *) &addr, sizeof(struct sockaddr))) {
#ifdef WIN32
      errno = WSAGetLastError();
#endif
      if(errno == EISCONN) goto out;
      if(time(NULL) < timeout && Sigint == 0) goto retry;
      sock_close(sd);
      if(Verbose) printf("connectip(): cannot connect() socket.\n");
      return INVALID_SOCKET;
   }
   if(Verbose) printf("connected.");
   nonblock(sd);
out:
   if(Verbose) printf("\n");
   return sd;
}  /* end connectip2() */


/* Send transaction in np->tx */
int sendtx2(NODE *np)
{
   int count;
   TX *tx;

   tx = &np->tx;

   put16(tx->version, PVERSION);
   put16(tx->network, TXNETWORK);
   put16(tx->trailer, TXEOT);
   put16(tx->id1, np->id1);
   put16(tx->id2, np->id2);
   crctx(tx);
   count = send(np->sd, TXBUFF(tx), TXBUFFLEN, 0);
   if(count != TXBUFFLEN)
      return VERROR;
   return VEOK;
}  /* end sendtx2() */


int send_op(NODE *np, int opcode)
{
   put16(np->tx.opcode, opcode);
   return sendtx2(np);
}


/* Receive next packet from NODE *np
 * Returns: VEOK=good, else error code.
 * Check id's if checkids is non-zero.
 * Set checkid to zero during handshake.
 */
int rx2(NODE *np, int checkids)
{
   int count, n;
   time_t timeout;
   TX *tx;

   tx = &np->tx;
   timeout = time(NULL) + 3;

   Sigint = 0;
   for(n = 0; ; ) {
      count = recv(np->sd, TXBUFF(tx) + n, TXBUFFLEN - n, 0);
      if(Sigint) return VERROR;
      if(count == 0) return VERROR;
      if(count < 0) {
         if(time(NULL) >= timeout) return -1;
         continue;
      }
      n += count;
      if(n == TXBUFFLEN) break;
   }  /* end for */

   /* check tx and return error codes or count */
   if(get16(tx->network) != TXNETWORK)
      return 2;
   if(get16(tx->trailer) != TXEOT)
      return 3;
   if(crc16(CRC_BUFF(tx), CRC_COUNT) != get16(tx->crc16))
      return 4;
   if(checkids && (np->id1 != get16(tx->id1) || np->id2 != get16(tx->id2)))
      return 5;
   return VEOK;
}  /* end rx2() */


/* Call peer and complete Three-Way */
int callserver(NODE *np, word32 ip, char *addrstr)
{
   int ecode, j;
   word16 opcode;

   Sigint = 0;

   memset(np, 0, sizeof(NODE));   /* clear structure */
   np->sd = INVALID_SOCKET;
   for(j = 0; j < RPLISTLEN && Sigint == 0; j++) {
      if(Nextcore >= RPLISTLEN) Nextcore = 0;
      ip = Coreplist[Nextcore];
      if(ip == 0 && addrstr == NULL) break;
      np->sd = connectip2(ip, addrstr);
      if(np->sd != INVALID_SOCKET) break;
      if(addrstr == NULL) Nextcore++;
      addrstr = NULL;
   }
   if(np->sd == INVALID_SOCKET) {
      Nextcore = 0;
      return VERROR;
   }
   np->ip = ip;
   np->id1 = rand16fast();
   send_op(np, OP_HELLO);

   ecode = rx2(np, 0);
   if(ecode != VEOK) {
      if(Verbose) printf("*** missing HELLO_ACK packet (%d)\n", ecode);
bad:
      sock_close(np->sd);
      np->sd = INVALID_SOCKET;
      Nextcore++;
      return VERROR;
   }
   np->id2 = get16(np->tx.id2);
   opcode = get16(np->tx.opcode);
   if(opcode != OP_HELLO_ACK) {
      if(Verbose) printf("*** HELLO_ACK is wrong: %d\n", opcode);
      goto bad;
   }
   put64(Cblocknum, np->tx.cblock);
   return VEOK;
}  /* end callserver() */


/* Call with:
 * opcode = OP_BALANCE tx->src_addr is address to query and
 *                     balance returned in tx->send_total.
 *
 * opcode = OP_GET_IPL  returns IP list in TRANBUFF(tx) of tx->len bytes.
 *
 * Returns VOEK on success, else VERROR.
 */
int get_tx(TX *tx, word32 ip, char *addrstr, int opcode)
{
   NODE node;

   if(callserver(&node, ip, addrstr) != VEOK)
      return VERROR;
   memcpy(&node.tx, tx, sizeof(TX));
   put16(node.tx.len, 1);  /* signal server that we are a wallet */
   send_op(&node, opcode);
   if(rx2(&node, 1) != VEOK) {
      sock_close(node.sd);
      return VERROR;
   }
   sock_close(node.sd);
   memcpy(tx, &node.tx, sizeof(TX));  /* return tx to caller's space */
   return VEOK;
}  /* end get_tx() */


/* Get a peer list from a Mochimo server at addrstr.
 * Return VEOK or VERROR.
 */
int get_ipl(char *addrstr)
{
   TX tx;
   int status, j, k;
   unsigned len;
   word32 ip, *ipp;

   Sigint = 0;
   memset(&tx, 0, sizeof(TX));
   for(j = 0; j < RPLISTLEN && Sigint == 0; j++) {
      ip = Coreplist[j];
      if(ip == 0 && addrstr == NULL) break;
      status = get_tx(&tx, ip, addrstr, OP_GET_IPL);
      len = get16(tx.len);
      /*
       * Insert the peer list after the core ip's in Coreplist[]
       */
      if(status == VEOK && len <= TRANLEN) {
         for(ipp = (word32 *) TRANBUFF(&tx), k = CORELISTLEN;
             k < RPLISTLEN && len > 0;
             ipp++, len -= 4) {
                if(*ipp == 0) continue;
                if(search32(*ipp, Coreplist, k)) continue;
                Coreplist[k++] = *ipp;
         }
         if(k > CORELISTLEN) {
            shuffle32(Coreplist, k);
            Nextcore = 0;
            printf("Addresses added: %d", k - CORELISTLEN);
         }
         return VEOK;
      }  /* end if copy ip list */
      addrstr = NULL;
   }  /* end for try again */
   return VERROR;
}  /* end get_ipl() */


int send_tx(TX *tx, word32 ip, char *addrstr)
{
   NODE node;
   int status;

   if(callserver(&node, ip, addrstr) != VEOK)
      return VERROR;
   memcpy(&node.tx, tx, sizeof(TX));
   put16(node.tx.len, 1);  /* signal server that we are a wallet */
   status = send_op(&node, OP_TX);
   sock_close(node.sd);
   return status;
}  /* end send_tx() */


/* Very special for Shylock: */
void shy_setkey(XO4CTX *ctx, word8 *salt, word8 *password, unsigned len)
{
   word8 key[256];

   memset(key, 0, 256);
   if(len > 188) len = 188;   /* 256-64-4 */
   memcpy(&key[64], salt, 4);
   memcpy(&key[68], password, len);
   sha256(&key[64], (256-64), key);
   sha256(key, 256, &key[32]);
   xo4_init(ctx, key, 64);
}


word8 *fuzzname(word8 *name, int len)
{
   word8 *bp;

   for(bp = name; len; len--)
      *bp++ |= ((rand16fast() & 0x8000) ? 128 : 0);
   return name;
}


word8 *unfuzzname(word8 *name, int len)
{
   word8 *bp;

   for(bp = name; len; len--)
      *bp++ &= 127;
   return name;
}

   
/* Copy outlen random bytes to out.
 * 64-byte seed is incremented.
 */
void rndbytes(word8 *out, word32 outlen, word8 *seed)
{
   static word8 state;
   static word8 rnd[PASSWLEN+64];
   word8 hash[32];  /* output for sha256() */
   int n;

   if(state == 0) {
      memcpy(rnd, seed, 64);
      memcpy(&rnd[64], Password, PASSWLEN);
      state = 1;
   }
   for( ; outlen; ) {
      /* increment big number in rnd and seed */
      for(n = 0; n < 64; n++) {
         if(++seed[n] != 0) break;
      }       
      for(n = 0; n < 64; n++) {
         if(++rnd[n] != 0) break;
      }       
      sha256(rnd, PASSWLEN+64, hash);
      if(outlen < 32) n = outlen; else n = 32;
      memcpy(out, hash, n);
      out += n;
      outlen -= n;
   }  /* end for outlen */
}  /* end rndbytes() */


/*
 * Create an address that can be later signed with wots_sign():
 * It calls the function wots_pkgen() which creates the address.
*/

/* Make up a random address that can be signed...
 * Outputs:
 *          addr[TXADDRLEN] takes the address (2208 bytes)
 *          secret[32]      needed for wots_sign()
 */
void create_addr(word8 *addr, word8 *secret, word8 *seed)
{
   word8 rnd2[32];

   rndbytes(secret, 32, seed);  /* needed later to use wots_sign() */

   rndbytes(addr, TXADDRLEN, seed);
   /* rnd2 is modified by wots_pkgen() */
   memcpy(rnd2, &addr[TXSIGLEN + 32], 32);
   /* generate a good addr */
   wots_pkgen(addr,              /* output first 2144 bytes */
              secret,            /* 32 */
              &addr[TXSIGLEN],   /* rnd1 32 */
              (word32 *) rnd2    /* rnd2 32 (modified) */
   );
   memcpy(&addr[TXSIGLEN+32], rnd2, 32);
}  /* end create_addr() */


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
}  /* end tgets() */


/* Open file, or if fatalflag, call fatal() else return fp. */
FILE *fopen2(char *fname, char *mode, int fatalflag)
{
   FILE *fp;

   fp = fopen(fname, mode);
   if(!fp && fatalflag) fatal("Cannot open %s", fname);
   return fp;
}


/* e.g. fp = fopen3(buff, "wb", "Overwrite");
 * or   fp = fopen3(buff, "ab", "Append");
 * or   fp = fopen3(buff, "wb", NULL);
 * Returns NULL on bad open, 1 on user cancel, or open fp.
 */
FILE *fopen3(char *fname, char *mode, char *safe)
{
   char buff[20];
   FILE *fp;

   fp = fopen(fname, "rb");
   if(fp && safe) {
      fclose(fp);
      printf("%s exists.  %s (y/n): ", fname, safe);
      tgets(buff, 20);
      if(*buff != 'y' && *buff != 'Y') return (FILE *) 1;
   }
   fp = fopen(fname, mode);
   return fp;
}


/* Returns 0 on success, else error code. */
int init_password(void)
{
   char temp[PASSWLEN];
   char temp2[PASSWLEN];

   memset(temp, 0, PASSWLEN);
   memset(temp2, 0, PASSWLEN);

   for(;;) {
      printf("Enter pass phrase of up to %d characters:\n"
             "(You need to remember this.)\n",
             PASSWLEN);
      tgets(temp, PASSWLEN);
      CLEARSCR();
      printf("Re-enter phrase:\n");
      tgets(temp2, PASSWLEN);
      if(strcmp(temp, temp2) == 0) break;
      CLEARSCR();
      printf("Phrases do not match.  Try again (y/n)? ");
      tgets(temp, PASSWLEN);
      if(*temp != 'y' && *temp != 'Y') return 1;  /* Password not changed */
   }
   strcpy(Password, temp);
   CLEARSCR();
   return 0;
}  /* end init_password() */


/* Set random seed for address generation. */
void init_seed(char *seed, unsigned len)
{
   FILE *fp;
   word8 b;

   if(len < 5) fatal("init_seed()");

   ((word16 *) seed)[0] = rand16fast();
   ((word16 *) seed)[1] = rand16fast();

   len -= 4;
   seed += 4;
   printf("Enter a random string of up to %d characters:\n"  
          "(You DO NOT need to remember this.)\n", len);
   tgets(seed, len);
   fp = fopen("/dev/random", "rb");
   if(fp) {
      for( ; len; len--) {
         if(fread(&b, 1, 1, fp) != 1) break;
         *seed++ ^= b;
      }
      fclose(fp);
   }
   CLEARSCR();
}  /* init_seed() */


void init_wallet(WHEADER *wh)
{
   FILE *fp;
   char fname[WFNAMELEN];
   char lbuff[100];
   static word8 salt[4];  /* salt is always zero for the header */

   memset(wh, 0, sizeof(WHEADER));
   printf("\nPress ctrl-c at any time to cancel.\n\n");

   printf("Enter the name of this wallet: ");
   tgets(lbuff, 100);
   memcpy(wh->name, lbuff, 25);
   if(init_password()) fatal("no pass phrase");
   init_seed((char *) wh->seed, RNDSEEDLEN);
   rndbytes(wh->sig, 4, wh->seed);

get_name:
   printf("Save to file (ctrl-c to quit) [%s]: ", Wfname);
   tgets(fname, WFNAMELEN);
   if(*fname) strncpy(Wfname, fname, WFNAMELEN-1);
   fp = fopen3(Wfname, "wb", "Overwrite");
   if(!fp) fatal("Cannot create %s", Wfname);  /* I/O error */
   if(fp == (FILE *) 1) goto get_name;  /* file existed */

   /* encrypt disk image */
   fuzzname(wh->name, 25);
   shy_setkey(&Xo4ctx, salt, (word8 *) Password, PASSWLEN);
   xo4_crypt(&Xo4ctx, wh, wh, sizeof(Whdr));

   if(fwrite(wh, 1, sizeof(WHEADER), fp) != sizeof(WHEADER))
      fatal("error writing %s", Wfname);
   fclose(fp);

   /* decrypt for use */
   shy_setkey(&Xo4ctx, salt, (word8 *) Password, PASSWLEN);
   xo4_crypt(&Xo4ctx, wh, wh, sizeof(WHEADER));
   unfuzzname(wh->name, 25);
   printf("Wallet '%-1.25s' id: 0x%x created to file %s\n", wh->name,
          *((int *) wh->sig), Wfname);
}  /* end init_wallet() */


/* Delete in-core wallet index and free memory. */
void delete_windex(void)
{
   if(Windex) {
      if(Nindex)
         memset(Windex, 0, Nindex * sizeof(WINDEX));  /* security */
      free(Windex);
      Windex = NULL;
   }
   Nindex = 0;
}


int read_wheader(WHEADER *whdr)
{
   FILE *fp;
   fp = fopen2(Wfname, "rb", 1);
   if(fread(whdr, 1, sizeof(WHEADER), fp) != sizeof(WHEADER))
      fatal("Cannot read %s", Wfname);
   fclose(fp);
   return 0;
}  /* end read_wheader() */


int decrypt_wheader()
{
   static word8 salt[4];
   shy_setkey(&Xo4ctx, salt, (word8 *) Password, PASSWLEN);
   xo4_crypt(&Xo4ctx, &Whdr, &Whdr, sizeof(WHEADER));
   unfuzzname(Whdr.name, 25);
   printf("Loaded wallet '%-1.25s' from %s\n", Whdr.name, Wfname);
   return 0;
}  /* end decrypt_wheader() */


/* Read a wallet address entry and de-crypt to entry.
 * idx is zero based here.
 * Return VEOK on success.
 */
int read_wentry(WENTRY *entry, unsigned idx)
{
   FILE *fp;

   if(idx >= Nindex) return VERROR;
   fp = fopen2(Wfname, "rb", 1);
   if(fseek(fp, sizeof(WHEADER) + (idx * sizeof(WENTRY)), SEEK_SET) != 0) {
bad:
      printf("\nI/O error\n");
      fclose(fp);
      return VERROR;
   }
   if(fread(entry, 1, sizeof(WENTRY), fp) != sizeof(WENTRY)) goto bad;
   shy_setkey(&Xo4ctx, entry->key, (word8 *) Password, PASSWLEN);
   /* entry->flags is first field after entry.key salt */
   xo4_crypt(&Xo4ctx, entry->flags, entry->flags,
             sizeof(WENTRY) - sizeof(entry->key));
   unfuzzname(entry->name, sizeof(entry->name));
   fclose(fp);
   return VEOK;
}  /* end read_wentry() */


/* Encrypt and write wallet entry to disk.
 * Returns VEOK on success with entry left encrypted,
 * else error code and entry indeterminate.
 * idx is zero based here.
 */
int write_wentry(WENTRY *entry, unsigned idx)
{
   FILE *fp;

   if(idx >= Nindex) return VERROR;
   fp = fopen2(Wfname, "r+b", 1);
   if(fseek(fp, sizeof(WHEADER) + (idx * sizeof(WENTRY)), SEEK_SET) != 0) {
bad:
      printf("\nI/O error\n");
      fclose(fp);
      return VERROR;
   }
   fuzzname(entry->name, sizeof(entry->name));
   shy_setkey(&Xo4ctx, entry->key, (word8 *) Password, PASSWLEN);
   /* entry->flags is first field after entry.key salt */
   xo4_crypt(&Xo4ctx, entry->flags, entry->flags,
             sizeof(WENTRY) - sizeof(entry->key));
   if(fwrite(entry, 1, sizeof(WENTRY), fp) != sizeof(WENTRY)) goto bad;
   fclose(fp);
   return VEOK;
}  /* end write_wentry() */


/* Open and read wallet entries to build malloc'd index Windex[].
 * Addresses and secrets are left encrypted on disk.
 * Only tag is read into core.
 */
word32 read_widx(void)
{
   FILE *fp;
   long fsize;
   word32 j;
   WINDEX *ip;
   WENTRY entry;

   /* If index already exists, delete it. */
   delete_windex();
   Nindex = 0;

   fp = fopen2(Wfname, "rb", 1);  /* open file or fatal() */
   fseek(fp, 0, SEEK_END);
   fsize = ftell(fp);   
   fseek(fp, sizeof(WHEADER), SEEK_SET);
   /* check file size */
   if(((fsize - sizeof(WHEADER)) % sizeof(WENTRY)) != 0)
      fatal("Invalid wallet file size on %s", Wfname);
   /* compute number of address entries in wallet */
   Nindex = (fsize - sizeof(WHEADER)) / sizeof(WENTRY);
   if(Nindex == 0) return Nindex;  /* new file has no entries */

   /* allocate array of WINDEX struct's */
   Windex = malloc(Nindex * sizeof(WINDEX));
   if(!Windex) fatal("No memory to read wallet index!");

   for(j = 0, ip = Windex; j < Nindex; j++, ip++) {
      if(fread(&entry, 1, sizeof(WENTRY), fp) != sizeof(WENTRY)) break;
      shy_setkey(&Xo4ctx, entry.key, (word8 *) Password, PASSWLEN);
      /* entry.flags is first field after entry.key salt */
      xo4_crypt(&Xo4ctx, entry.flags, entry.flags,
                sizeof(WENTRY) - sizeof(entry.key));
      unfuzzname(entry.name, sizeof(entry.name));
      memcpy(ip->key, entry.key, sizeof(ip->key));
      ip->flags[0] = entry.flags[0];
      memcpy(ip->balance, entry.balance, sizeof(ip->balance));
      memcpy(ip->name, entry.name, sizeof(ip->name));
      memcpy(ip->tag, ADDR_TAG_PTR(entry.addr), TXTAGLEN);
      ip->flags[0] &= ~(W_TAG);
      if(ADDR_HAS_TAG(entry.addr)) ip->flags[0] |= W_TAG;
      if(cmp64(ip->balance, Zeros)) {
         ip->flags[0] |= W_BAL;
         ip->flags[0] &= ~(W_DEL);
      }
      else ip->flags[0] &= ~(W_BAL);
      if(!iszero(entry.secret, 32)) ip->flags[0] |= W_SEC;
      else ip->flags[0] &= ~(W_SEC);
   }  /* end for j */
   if(j != Nindex || ferror(fp))
      fatal("I/O error reading wallet index");
   fclose(fp);
   memset(&entry, 0, sizeof(WENTRY));  /* security */
   return Nindex;
}  /* end read_widx() */


/* Find a duplicate of addr or its tag in wallet,
 * copy the entry to outentry, and return its 1-based index.
 * If not found, return 0 with addr and outentry unchanged.
 * Call fatal() on I/O errors.
 */
unsigned find_dup(word8 *addr, WENTRY *outentry)
{
   WINDEX *ip;
   unsigned idx;
   word8 *tag;
   WENTRY entry;

   read_widx();
   tag = NULL;
   if(ADDR_HAS_TAG(addr)) tag = ADDR_TAG_PTR(addr);
   for(idx = 0, ip = Windex; idx < Nindex; idx++, ip++) {
      if(read_wentry(&entry, idx) != VEOK) fatal("bad disk read");
      if((tag && memcmp(tag, ADDR_TAG_PTR(entry.addr), TXTAGLEN) == 0)
         || memcmp(addr, entry.addr, TXADDRLEN - TXTAGLEN) == 0) {
            memcpy(outentry, &entry, sizeof(WENTRY));
            return idx + 1;
      }
   }
   memset(&entry, 0, sizeof(WENTRY));  /* security */
   return 0;  /* not found */
}  /* end find_dup() */


/* Find a wallet address with the tag in addr.
 * If found, copy wallet entry to entry and return a
 * 1 based index, else return 0 with addr unchanged.
 * (entry is indeterminate on I/O error.)
 */
unsigned find_tag(word8 *addr, WENTRY *entry)
{
   WINDEX *ip;
   unsigned idx;
   word8 *tag;

   read_widx();
   tag = ADDR_TAG_PTR(addr);
   for(idx = 0, ip = Windex; idx < Nindex; idx++, ip++) {
      if(memcmp(tag, ip->tag, TXTAGLEN) == 0) {
         if(read_wentry(entry, idx) != VEOK) return 0;  /* error */
         return idx + 1;
      }
   }
   return 0;  /* not found */
}  /* end find_tag() */


/* Export an address to an un-encrypted disk file... */
int ext_addr(unsigned idx)
{
   char buff[80];
   FILE *fp;
   int ecode;
   WENTRY *entry, entryst;

   entry = &entryst;
   if(idx == 0) {
      if(Nindex == 0) {
         printf("\nNo addresses to export.\n");
         return VERROR;
      }
      printf("Export address index (1-%d): ", Nindex);
      tgets(buff, 80);
      idx = atoi(buff);
   }
   if(badidx(idx)) return VERROR;
   ecode = read_wentry(entry, idx-1);
   if(ecode != VEOK) goto out2;
   printf("%-1.25s\n", entry->name);
   printf("Write to file name: ");
   tgets(buff, 80);
   ecode = VEOK;
   if(*buff == '\0') goto out2;     /* cancel */
   fp = fopen3(buff, "wb", "Overwrite");
   if(fp == (FILE *) 1) goto out2;  /* user cancelled */
   ecode = VERROR;
   if(!fp) goto out2;
   if(ADDR_HAS_TAG(entry->addr)) {
      printf("Export tag (y/n)? ");
      tgets(buff, 80);
      if(*buff != 'y' && *buff != 'Y')
         memcpy(ADDR_TAG_PTR(entry->addr), Default_tag, TXTAGLEN);
   }
   if(fwrite(entry->addr, 1, TXADDRLEN, fp) != TXADDRLEN) goto out;
   printf("Write balance (y/n)? ");
   tgets(buff, 80);
   if(*buff == 'y' || *buff == 'Y') {
      if(fwrite(entry->balance, 1, 8, fp) != 8) goto out;
   } else goto okout;
   printf("Write secret (y/n)? ");
   tgets(buff, 80);
   if(*buff != 'y' && *buff != 'Y') goto okout;
   if(fwrite(entry->secret, 1, 32, fp) != 32) goto out;
okout:
   disp_ecode(VEOK);
   ecode = VEOK;
out:
   fclose(fp);
out2:
   if(ecode != VEOK) printf("\n*** Write error\n");
   memset(entry, 0, sizeof(WENTRY));  /* security */
   return ecode;
}  /* end ext_addr() */


int update_wheader(WHEADER *whdr)
{
   FILE *fp;
   static word8 salt[4];
   WHEADER whout;

   /* open wallet header */
   fp = fopen2(Wfname, "r+b", 1);
   /* encrypt it for write to disk */
   memcpy(&whout, whdr, sizeof(WHEADER));
   fuzzname(whout.name, 25);
   shy_setkey(&Xo4ctx, salt, (word8 *) Password, PASSWLEN);
   xo4_crypt(&Xo4ctx, &whout, &whout, sizeof(WHEADER));
   if(fwrite(&whout, 1, sizeof(WHEADER), fp) != sizeof(WHEADER))
      fatal("Cannot update wallet header");
   fclose(fp);
   return 0;
}  /* end update_wheader() */


/* Fetch an address with the tag field in addr.
 * Returns VEOK if found with full address in addr,
 * else error code.
 */
int get_tag(word8 *addr, word8 found[1])
{
   int ecode;
   TX tx;

   memset(&tx, 0, sizeof(TX));
   memcpy(tx.dst_addr, addr, TXADDRLEN);
   ecode = get_tx(&tx, 0, Peeraddr, OP_RESOLVE);
   if(ecode != VEOK || get16(tx.opcode) != OP_RESOLVE) {
      found[0] = 0;
      return VERROR;
   }

   if(tx.send_total[0])
   {
	   /* check if the tag on resolved address is the one we asked */
	   if(memcmp(ADDR_TAG_PTR(addr), ADDR_TAG_PTR(tx.dst_addr), TXTAGLEN) != 0)
	   {
		   found[0] = 0;
		   return VERROR;
	   }
   }

   memcpy(addr, tx.dst_addr, TXADDRLEN);
   found[0] = tx.send_total[0];  /* 1 if found, else 0 */
   return VEOK;
}  /* end get_tag() */


/* Add address to wallet.  Call after successful read_wheader().
 * If name is not NULL, entry is appended to wallet.
 * Set name to NULL to create import dummy entry.
 * Returns index of added address entry.
 */
int add_addr(WENTRY *entry, char *name)
{
   FILE *fp;
   word32 lastkey;
   char buff[80];
   long last_idx;

   fp = fopen2(Wfname, "ab", 1);  /* open file or fatal() */
   memset(entry, 0, sizeof(WENTRY));

   if(name) {  /* not import dummy */
      create_addr(entry->addr, entry->secret, Whdr.seed);
      printf("Add tag (y/n)? ");   /* tag */
      tgets(buff, 80);
      if(*buff == 'y' || *buff == 'Y') {
         ADDR_TAG_PTR(entry->addr)[0] = 1;        /* Create type-1 */
         rndbytes(ADDR_TAG_PTR(entry->addr) + 1,  /*  random tag field. */
                  TXTAGLEN - 1, Whdr.seed);
      }  /* tag */
      bytes2hex((word8 *) entry->addr, 32, '\n');
      if(ADDR_HAS_TAG(entry->addr)) {
         printf("Tag: ");
         bytes2hex(ADDR_TAG_PTR(entry->addr), TXTAGLEN, '\n');
      }
   }  /* end if not dummy */

   lastkey = get32(Whdr.lastkey);  /* salt */
   lastkey++;
   put32(Whdr.lastkey, lastkey);   /* save updated salt */
   put32(entry->key, lastkey);
   if(name)
      memcpy(entry->name, name, sizeof(entry->name));
   fuzzname(entry->name, sizeof(entry->name));
   shy_setkey(&Xo4ctx, entry->key, (word8 *) Password, PASSWLEN);
   /* entry->flags is first field after entry->key salt */
   xo4_crypt(&Xo4ctx, entry->flags, entry->flags,
             sizeof(WENTRY) - sizeof(entry->key));
   if(fwrite(entry, 1, sizeof(WENTRY), fp) != sizeof(WENTRY))
      fatal("I/O error");
   last_idx = ftell(fp);
   last_idx = (last_idx - sizeof(WHEADER)) / sizeof(WENTRY);
   fclose(fp);
   update_wheader(&Whdr);
   return last_idx;
}  /* end add_addr() */


/* Add or update imported tag to wallet.
 * Return VEOK on success, else VERROR.
 */
int add_tag_addr(void)
{
   FILE *fp;
   word32 lastkey;
   char buff[80];
   WENTRY *entry, entst;
   word8 addr[TXADDRLEN];
   word8 found;
   unsigned idx;

   entry = &entst;

   for(;;) {
      printf("Enter tag in hex: ");
      memset(buff, 0, 80);
      tgets(buff, 80);
      if(*buff == '\0') return VERROR;
      if(strlen(buff) == 24) break;
      printf("Please enter 24 hex digits.\n");
   }
   /* Convert buff to binary and */
   /* put buff into dst_addr of dummy TX
    * to query server.
    * If tag not found, return VERROR, else
    * add or update the found addr in wallet.
    */
   printf("Checking network...\n");
   hex2bytes(buff, buff);
   memcpy(ADDR_TAG_PTR(addr), buff, TXTAGLEN);
   if(get_tag(addr, &found) != VEOK) {
      printf("\nTag server not connected... Try again.\n");
      Nextcore++;
      return VERROR;
   }      
   if(!found) {
      printf("\nTag not found.\n");
      return VERROR;
   }
   if((idx = find_tag(addr, entry)) != 0) {
      /* tag already in wallet so just update address: */
      memcpy(entry->addr, addr, TXADDRLEN);
      if(write_wentry(entry, idx-1) != VEOK) {
         printf("*** Disk write error.\n");
         return VERROR;
      }
      printf("Tag updated from network.  New address:\n");
      bytes2hex(addr, 32, '\n');
      return VEOK;
   }

   printf("Enter address name: ");
   tgets(buff, 80);

   fp = fopen2(Wfname, "ab", 1);  /* open file or fatal() */
   memset(entry, 0, sizeof(WENTRY));
   strncpy((char *) entry->name, buff, 16);
   memcpy(entry->addr, addr, TXADDRLEN);
   lastkey = get32(Whdr.lastkey);  /* salt */
   lastkey++;
   put32(Whdr.lastkey, lastkey);   /* save updated salt */
   put32(entry->key, lastkey);
   fuzzname(entry->name, sizeof(entry->name));
   shy_setkey(&Xo4ctx, entry->key, (word8 *) Password, PASSWLEN);
   /* entry->flags is first field after entry->key salt */
   xo4_crypt(&Xo4ctx, entry->flags, entry->flags,
             sizeof(WENTRY) - sizeof(entry->key));
   if(fwrite(entry, 1, sizeof(WENTRY), fp) != sizeof(WENTRY))
      fatal("I/O error");
   fclose(fp);
   update_wheader(&Whdr);
   printf("Address imported.\n");
   return VEOK;
}  /* end add_tag_addr() */


/* Add address to wallet.  Call after successful read_wheader()
 * Prompt user.
 */
void add_addr2(int promptf)
{
   WENTRY entry;
   char name[80];
   unsigned idx;

   memset(name, 0, 80);
   if(promptf) {
      printf("Enter address name: ");
      tgets(name, 80);
      if(name[0] == '\0') return;
   }
   idx = add_addr(&entry, name);
   if(idx == 0)
      printf("\nAddress NOT created.\n");
   else
      printf("\nAddress index %u, '%s', created.\n", idx, name);
   read_widx();
   memset(&entry, 0, sizeof(WENTRY));  /* security */
}  /* end add_addr2() */


#define I_ZSUP 1   /* zero suppress */

/* Convert 64-bit little-endian int to a char string in out.
 * dec is where to put decimal point from left.
 */
char *itoa64(void *val64, char *out, int dec, int flags)
{
   int count;
   static char s[24];
   char *cp, zflag = 1;
   word32 *tab;
   word8 val[8];

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


/* Convert a decimal ASCII string to a 64-bit value out.
 * If string ends in 'c' or 'C' convert to Chi, else
 * leave as Satoshi.
 */
int atoi64(char *string, word8 *out)
{
   static word8 addin[8];
   static word32 ten[2] = { 10 };
   static word32 tene9[2] = { 1000000000 };  /* Satoshi per Chi */
   int overflow = 0;

   put64(out, Zeros);
   for( ;; string++) {
      if(*string < '0' || *string > '9') {
         if(*string == 'c' || *string == 'C') {
            overflow |= mult64(out, tene9, out);
         }
         return overflow;
      }
      overflow |= mult64(out, ten, out);
      addin[0] = *string - '0';
      overflow |= add64(out, addin, out);  /* add in this digit */
   }  /* end for */
}  /* end atoi64() */


/* The next 3 functions display wallet entries to user. */

void disp_header(void)
{
   printf("INDEX       NAME                    TAG                     "
          "     BALANCE\n");
}


void disp_line(WINDEX *ip, unsigned j)
{
   printf("%-6d %-25.25s ", j+1, ip->name);
   if(!Verbose && (ip->flags[0] & W_TAG) == 0) {
      for(j = 0; j < TXTAGLEN; j++) printf("  ");
   } else bytes2hex(ip->tag, TXTAGLEN, 0);
   printf(" %s\n", itoa64(ip->balance, NULL, 9, 1));
}


/* Flags (W_xxx) are already set in Windex[].
 * Returns number of entries displayed.
 * mode:
 *   0 src
 *   1 dst
 *   2 chg
 *   3 all not W_DEL
*/
int display_wallet(int mode, unsigned notidx)
{
   word32 j;
   WINDEX *ip;
   int line;
   char lbuff[10];
   word8 flags;
   int nout;

   read_widx();
   if(Nindex == 0 || Windex == NULL) {
noent:
      printf("No entries.\n");
      return 0;
   }

   nout = line = 0;
   for(j = 0, ip = Windex; j < Nindex; j++, ip++) {
      flags = ip->flags[0];
      if(flags & W_DEL) continue;
      switch(mode) {
         case 0: if((flags & W_SEC) == 0 || (flags & W_BAL) == 0) continue;
                 break;
         case 1: if(flags & W_SEC) continue;
                 break;
         case 2: if((flags & W_SEC) == 0 || (flags & W_SPENT)) continue;
                 break;
      }
      if(notidx && (j+1) == notidx) continue;
      if(line == 0) disp_header();
      if(++line > 19) {  /* page display */
         printf("ENTER=next, q=quit: ");
         tgets(lbuff, 10);
         line = 0;
         if(lbuff[0] == 'q') break;
      }
      disp_line(ip, j);
      nout++;
   }  /* end for */
   if(nout == 0) goto noent;
   if(line > 8) {
      printf("Press return...\n");
      tgets(lbuff, 10);
   }
   return nout;
}  /* end display_wallet() */


int archive_addr(void)
{
   char lbuff[80];
   unsigned idx;
   int ecode;
   WENTRY entry;

   if(Nindex == 0) {
      printf("\nNo addresses to archive.\n");
      return VERROR;
   }
   printf("Archive address index (1-%d): ", Nindex);
   tgets(lbuff, 80);
   idx = atoi(lbuff);
   if(badidx(idx)) return VERROR;
   ecode = read_wentry(&entry, idx-1);
   if(ecode != VEOK) return ecode;
   if(cmp64(entry.balance, Zeros) != 0)
      printf("Balance is not zero.\n");
   printf("Archive '%-1.25s' (y/n)? ", entry.name);
   tgets(lbuff, 80);
   if(lbuff[0] != 'y' && lbuff[0] != 'Y') return VEOK;
   put64(entry.balance, Zeros);
   entry.flags[0] |= W_DEL;
   ecode = write_wentry(&entry, idx-1);
   return ecode;
}  /* end archive_addr() */


/* Return 1 if tag unusable, else 0. */
int bad_tag(word8 *addr)
{
   word8 found;
   int status;

   Sigint = 0;
   for(;;) {
      printf("\nChecking that change tag is unused.  "
             "Press ctrl-c to cancel...\n");
      status = get_tag(addr, &found);
      sleep(1);
      if(status != VEOK || Sigint) {
         printf("Tag server not found.  Try again.\n");
         return 1;
      }
      if(found) {
         printf("Tag already in use.  Make new address.\n");
         return 1;
      }
      return 0;
   }
}  /* bad_tag() */


/* Check and address balance with network.
 * Parameter idx is 1-based wallet index.
 * Return VEOK on success, else error code.
 */
int check_bal(unsigned idx)
{
   int ecode;
   WENTRY entry;
   WINDEX *ip;
   TX tx;

   if(badidx(idx)) return VERROR;
   ecode = read_wentry(&entry, idx-1);
   if(ecode != VEOK) goto out;
   memset(&tx, 0, sizeof(TX));
   memcpy(tx.src_addr, entry.addr, TXADDRLEN);
   tx.send_total[0] = 1;
   ecode = get_tx(&tx, 0, Peeraddr, OP_BALANCE);
   if(ecode != VEOK) goto out;
   ip = &Windex[idx-1];
   put64(ip->balance, tx.send_total);
   put64(entry.balance, tx.send_total);
   if(cmp64(ip->balance, Zeros) != 0) {
      ip->flags[0] &= ~(W_SPENT | W_DEL);
      ip->flags[0] |= W_BAL;
   }
   ecode = write_wentry(&entry, idx-1);
out:
   if(ecode != VEOK)
      printf("*** Balance check failed.\n");
   memset(&tx, 0, sizeof(TX));         /* security */
   memset(&entry, 0, sizeof(WENTRY));
   return ecode;
}  /* end check_bal() */


/* Spend an address.
 * Return VEOK on success, else error code.
 * (Does many things...)
 */
int spend_addr(void)
{
   char buff[128];
   unsigned sidx, didx, cidx;
   int ecode;
   WENTRY sentry, dentry, centry;
   TX tx;
   word32 total[2], change[2];
   word8 message[32], rnd2[32];
   word8 val[8];
   word8 found;
   word8 olddst[TXADDRLEN];

   memset(&tx, 0, sizeof(TX));
   memset(&sentry, 0, sizeof(WENTRY));
   memset(&dentry, 0, sizeof(WENTRY));
   memset(&centry, 0, sizeof(WENTRY));
   put64(val, Zeros);
getsrc:
   printf("\nAddresses to spend:\n");
   if(display_wallet(0, 0) < 1) goto out2;
   printf("Spend address index (1-%d, or 0 to cancel): ", Nindex);
   tgets(buff, 80);
   sidx = atoi(buff);
   if(sidx == 0) {
      printf("\nCanceled.\n");
      goto out2;
   }
   if(badidx(sidx)) goto getsrc;
   ecode = read_wentry(&sentry, sidx-1);
   if(ecode != VEOK) {
ioerror:
      printf("\nI/O error.\n");
out2:
       ecode = VERROR;
out:
      memset(&tx, 0, sizeof(TX));          /* clear for security */
      memset(&sentry, 0, sizeof(WENTRY));
      memset(&dentry, 0, sizeof(WENTRY));
      memset(&centry, 0, sizeof(WENTRY));
      return ecode;
   }  /* end if error */

   memcpy(tx.src_addr, sentry.addr, TXADDRLEN);
getdst:
   printf("Foreign addresses:\n");
   memcpy(ADDR_TAG_PTR(dentry.addr), Default_tag, TXTAGLEN);
   display_wallet(1, sidx);
   printf("Destination address index (1-%d, or 0 for none): ", Nindex);
   tgets(buff, 80);
   didx = atoi(buff);
   if(didx == 0) goto skipdst;
   if(didx > Nindex) goto getdst;
   if(didx == sidx) goto getdst;
   ecode = read_wentry(&dentry, didx-1);
   if(ecode != VEOK) goto ioerror;
   /* if dst has tag, get the full address from network */
   memcpy(olddst, dentry.addr, TXADDRLEN);
   if(ADDR_HAS_TAG(dentry.addr)) {
      printf("Checking network...\n");
      get_tag(dentry.addr, &found);
      if(!found) {
         printf("\nTag not found.\n");
         goto out2;
      }
      if(memcmp(dentry.addr, olddst, TXADDRLEN) != 0)
         printf("Address updated.\n");
   }
getamt:
   printf("Enter send amount in Satoshi (or append c for Chi):\n");
   tgets(buff, 80);
   if(atoi64(buff, val)) {
      printf("Overflow.  Try again.\n");
      goto getamt;
   }
skipdst:
   printf("Send amount: %s\n", itoa64(val, NULL, 9, 1));
   put64(tx.send_total, val);
   memcpy(tx.dst_addr, dentry.addr, TXADDRLEN);
   add64(dentry.balance, tx.send_total, dentry.balance);

getchg:
   Sigint = 0;
   printf("Change addresses:\n");
   if(display_wallet(2, sidx) < 1) {
      printf("Create a new change address...\n");
      add_addr2(1);
      if(Sigint) goto ioerror;
      goto getchg;
   }
   printf("Create another change address (y/n)? ");
   tgets(buff, 80);
   if(*buff == 'y' || *buff == 'Y') {
      add_addr2(1);
      display_wallet(2, sidx);
   }
   printf("Change address index (1-%d, or 0 to cancel): ", Nindex);
   tgets(buff, 80);
   cidx = atoi(buff);
   if(cidx == 0) goto out2;
   if(cidx > Nindex) goto getchg;
   if(cidx == sidx || cidx == didx) goto getchg;
   ecode = read_wentry(&centry, cidx-1);
   if(ecode != VEOK) goto ioerror;
   memcpy(tx.chg_addr, centry.addr, TXADDRLEN);
   /* Calculate change and check source funds. */
   add64(Mfee, tx.send_total, total);
   printf("Checking source balance...\n");
   ecode = check_bal(sidx);
   if(ecode != VEOK) {
      printf("Cannot connect.\n");
      goto out2;
   }
   if(cmp64(total, sentry.balance) > 0) {
nofunds:
      printf("\nInsufficient funds.\n");
      goto out2;
   }
   if(sub64(sentry.balance, total, change) != 0) goto nofunds;
   put64(tx.change_total, change);
   add64(centry.balance, change, centry.balance);
   memcpy(tx.chg_addr, centry.addr, TXADDRLEN);
   put64(tx.tx_fee, Mfee);

   if(ADDR_HAS_TAG(tx.chg_addr))
      printf("Change address has tag.\n");
   if(ADDR_HAS_TAG(tx.src_addr)) {
      printf("Source address has tag.  Transfer tag (y/n)? ");
      tgets(buff, 80);
      if(*buff == 'y' || *buff == 'Y') {
         memcpy(ADDR_TAG_PTR(tx.chg_addr),
                ADDR_TAG_PTR(tx.src_addr), TXTAGLEN);
         memcpy(ADDR_TAG_PTR(centry.addr),
                ADDR_TAG_PTR(tx.chg_addr), TXTAGLEN);
      } else {
         memcpy(ADDR_TAG_PTR(tx.chg_addr), Default_tag, TXTAGLEN);
         memcpy(ADDR_TAG_PTR(centry.addr), Default_tag, TXTAGLEN);
      }
   }

   if(memcmp(tx.src_addr, tx.dst_addr, TXADDRLEN - TXTAGLEN) == 0) {
      printf("\nFrom and to address are the same.\n");
      goto out2;
   }
   if(memcmp(tx.src_addr, tx.chg_addr, TXADDRLEN - TXTAGLEN) == 0) {
      printf("\nFrom and change address are the same.\n");
      goto out2;
   }
   if(memcmp(tx.dst_addr, tx.chg_addr, TXADDRLEN - TXTAGLEN) == 0) {
      printf("\nDestination and change address are the same.\n");
      goto out2;
   }

   /* hash tx to message*/
   sha256(tx.src_addr,  TXSIGHASH_COUNT, message);
   /* sign TX with secret key for src_addr*/
   memcpy(rnd2, &tx.src_addr[TXSIGLEN+32], 32);  /* temp for wots_sign() */
   wots_sign(tx.tx_sig,  /* output 2144 */
             message,    /* hash 32 */
             sentry.secret,     /* random secret key 32 */
             &tx.src_addr[TXSIGLEN],    /* rnd1 32 */
             (word32 *) rnd2            /* rnd2 32 (maybe modified) */
   );

   if(get32(Mfee) < 100000)
      printf("Transaction fee is 0.0000%05u Chi\n", get32(Mfee));
   printf("Confirm send transaction (y/n)? ");
   tgets(buff, 80);
   if(buff[0] != 'y' && buff[0] != 'Y') {
notsent:
      printf("*** Not sent.\n");
      goto out2;
   }
   if(ADDR_HAS_TAG(tx.chg_addr) && bad_tag(tx.chg_addr)) goto notsent;

   /* transmit TX */
   printf("Trying connection.  Press ctrl-c to stop...\n");
   ecode = send_tx(&tx, 0, Peeraddr);
   if(ecode == VEOK)
      printf("Sent!\n");
   else {
      printf("*** Host not found\n");
      goto out2;
   }
   sentry.flags[0] |= W_SPENT;
   ecode = write_wentry(&sentry, sidx-1);   /* update wallet */
   if(didx)
      ecode |= write_wentry(&dentry, didx-1);
   ecode |= write_wentry(&centry, cidx-1);
   if(ecode != VEOK) goto ioerror;
   goto out;
}  /* end spend_addr(void) */


/* Import an address from disk file.
 * (Checks for tags.)
 */
int import_addr(void)
{
   char buff[80];
   FILE *fp;
   int ecode;
   WENTRY *entry, entryst, newentry;
   unsigned idx;

   entry = &entryst;

   ecode = VERROR;
   fp = NULL;
   memset(entry, 0, sizeof(WENTRY));
   printf("Import file name: ");
   tgets(buff, 80);
   if(buff[0] == '\0') goto out;
   fp = fopen(buff, "rb");
   if(!fp) {
      printf("Cannot open %s\n", buff);
      goto out;
   }
   memset(entry->secret, 0, 32);
   put64(entry->balance, Zeros);
   if(fread(entry->addr, 1, TXADDRLEN, fp) != TXADDRLEN) {
      printf("Cannot read address\n");
      goto out;
   }
   if(find_dup(entry->addr, &newentry)) {
      printf("WARNING: address or tag is already in wallet."
             "  Import anyway (y/n)? ");
      tgets(buff, 80);
      if(*buff != 'y' && *buff != 'Y') goto out;
   }
   if(fread(entry->balance, 1, 8, fp) != 8) {
      goto getname;
   }
   printf("Import secret (y/n)? ");
   tgets(buff, 80);
   if(*buff == 'y' || *buff == 'Y') {
      if(fread(entry->secret, 1, 32, fp) != 32) goto out;
   }
getname:
   printf("Enter address name: ");
   memset(buff, 0, 80);
   tgets(buff, 80);
   if(*buff == '\0') goto out;
   memcpy(entry->name, buff, 25);
   idx = add_addr(&newentry, NULL);
   if(idx == 0) goto out;
   read_widx();
   if(badidx(idx)) goto out;
   if(read_wentry(&newentry, idx-1) != VEOK) goto out;
   memcpy(newentry.name, entry->name, 25);
   memcpy(newentry.addr, entry->addr, TXADDRLEN);
   memcpy(newentry.secret, entry->secret, 32);
   put64(newentry.balance, entry->balance);
   ecode = VEOK;
out:
   if(ecode == VEOK)
      disp_ecode(VEOK);
   if(fp) fclose(fp);
   if(ecode == VEOK) {
      ecode = write_wentry(&newentry, idx-1);   /* update wallet */
      printf("Address imported.\n");
   } else
      printf("*** Not imported\n");
   memset(entry, 0, sizeof(WENTRY));
   memset(&newentry, 0, sizeof(WENTRY));  /* security */
   return ecode;
}  /* end import_addr() */


/* Check all balances. */
int query_all(void)
{
   unsigned j;
   int ecode;
   WINDEX *ip;

   printf("\nChecking balances, press ctrl-c to stop...\n\n");

   Sigint = 0;
   ecode = VEOK;
   for(ip = Windex, j = 1; j <= Nindex; ip++, j++) {
      if(Sigint) break;
      ecode = check_bal(j);
      if(ecode != VEOK) break;
   }
   return ecode;
}  /* end query_all(() */


/* Edit the name of an address. */
int edit_name(void)
{
   char lbuff[100];
   unsigned idx;
   int ecode;
   WENTRY entry;

   if(Nindex == 0) {
      printf("\nNo addresses to edit.\n");
      return VERROR;
   }
   printf("Change address name.\nindex (1-%d): ", Nindex);
   tgets(lbuff, 100);
   idx = atoi(lbuff);
   if(badidx(idx)) return VERROR;
   ecode = read_wentry(&entry, idx-1);
   if(ecode != VEOK) return ecode;
   printf("%-25.25s\nEnter new name or press ENTER to cancel:\n",
          entry.name);
   tgets(lbuff, 100);
   if(lbuff[0])
      memcpy(entry.name, lbuff, 25);
   ecode = write_wentry(&entry, idx-1);
   if(ecode == VEOK)
      disp_ecode(VEOK);
   return ecode;
}  /* end edit_name() */


void nstatus(void)
{
   if(cmp64(Cblocknum, Zeros) == 0)
      printf("\nWaiting on Mochimo HQ...\n");
   else
      printf("\nBlock: 0x%s\n", bnum2hex(Cblocknum));
}


void get_peers(char *peeraddr)
{
   int status;

   printf("Fetching peer list\n");
   printf("Press ctrl-c to stop...\n");
   status = get_ipl(peeraddr);
   if(Sigint) return;
   disp_ecode(status); 
}


int display_hex(void)
{
   char lbuff[100];
   unsigned idx;
   int ecode;
   WENTRY entry;

   if(Nindex == 0) {
      printf("\nNo addresses to display.\n");
      return VERROR;
   }
   printf("Display address in hexadecimal.\nindex (1-%d): ", Nindex);
   tgets(lbuff, 100);
   idx = atoi(lbuff);
   if(badidx(idx)) return VERROR;
   ecode = read_wentry(&entry, idx-1);
   if(ecode != VEOK) return ecode;
   printf("\n%-25.25s\n", entry.name);
   bytes2hex(entry.addr, 32, '\n');
   printf("(more)\nTag: ");
   bytes2hex(ADDR_TAG_PTR(entry.addr), TXTAGLEN, '\n');
   printf("Press return...");
   tgets(lbuff, 100);
   return VEOK;
}  /* end display_hex() */


/* Import case dispatcher. */
int import2(void)
{
   char buff[20];

   printf("Import foreign tag (y/n)? ");
   tgets(buff, 20);
   if(*buff == 'y' || *buff == 'Y')
      return add_tag_addr();
   else
      return import_addr();  /* with create */
}


void display_change(void)
{
   printf("\nChange addresses:\n");
   display_wallet(2, 0);
}


void display_import(void)
{
   printf("\nForeign addresses:\n");
   display_wallet(1, 0);
}


int menu2(void)
{
   char buff[20];

   CLEARSCR();

   for( ;; ) {
      printf("\n          Menu 2\n\n"
             "  1. Edit address name\n"
             "  2. Display change addresses\n"
             "  3. Display foreign addresses\n"
             "  4. Display address in hex\n"
             "  5. Get a Mochimo peer list\n"
             "  6. N/A\n"
             "  7. N/A\n"
             "  8. N/A\n"
             "  9. Main Menu\n"
             "  0. Exit\n\n"
             "  Select: "
      );
      tgets(buff, 20);
      switch(buff[0]) {
         case '1':  edit_name();          break;
         case '2':  CLEARSCR();  display_change();  break;
         case '3':  CLEARSCR();  display_import();  break;
         case '4':  display_hex();        break;
         case '5':  get_peers(Peeraddr);  break;
         case '6':  break;
         case '7':  break;
         case '8':  break;
         case '9':  CLEARSCR(); return -1;  /* previous menu */
         case '0':  return 0;   /* exit */
      }  /* end switch */
   }  /* end for */
}  /* end menu2(); */


void mainmenu(void)
{
   char buff[20];

   CLEARSCR();
   printf("\nMochimo Wallet (Build 31)\n"
          "Copyright (c) 2019 by Adequate Systems, LLC."
          "  All Rights Reserved.\n\n");
   read_widx();

   signal(SIGINT, ctrlc);

   if(Corefile && *Corefile) {
      if(read_coreipl(Corefile) != VEOK) {
         if(!Peeraddr || !*Peeraddr)
            printf("Cannot open %s... Defaulting to Localhost*\n"
                   "NOTE: If you aren't running a Mochimo node on this machine,\n"
                   "      the wallet will not operate correctly.\n", Corefile);
      } else if(Verbose) printf("%s loaded...\n", Corefile);
   }
   if(Peeraddr && *Peeraddr && Verbose)
      printf("Prioritising %s for connection...\n", Peeraddr);

   query_all();  /* check all old balances */
   display_wallet(0, 0);

   for( ;; ) {
      printf("\n          Main Menu\n\n"
         "  1. Network status   2. Display          3. Import address\n"
         "  4. Create address   5. Spend address    6. Check balances\n"
         "  7. Export address   8. Archive address  9. Menu 2\n"
         "  0. Exit\n\n"
         "  Select: "
      );
      tgets(buff, 20);
      switch(buff[0]) {
         case '1': nstatus();              break;
         case '2': CLEARSCR();  printf("\nMy addresses:\n");
                   display_wallet(0, 0);   break;
         case '3': import2();
                   read_widx();            break;
         case '4': add_addr2(1);
                   break;
         case '5': spend_addr();
                   read_widx();
                   break;
         case '6': query_all();
                   printf("\n");
                   display_wallet(0, 0);   break;
         case '7': ext_addr(0);            break;
         case '8': archive_addr();          break;
         case '9': if(menu2() == 0) return;
                   break;
         case '0': return;
      }  /* end switch */
   }  /* end for */
}  /* end mainmenu(); */


void usage(void)
{
   printf("\nUsage: wallet [-option -option2 . . .] [wallet_file]\n"
      "options:\n"
      "           -cFNAME  set alternate IP list to FNAME\n"
      "           -aS      set address string to S\n"
      "           -pN      set TCP port to N\n"
      "           -v       verbose output\n"
      "           -n       create new wallet\n\n"
   );
   exit(1);
}


int main(int argc, char **argv)
{
   int j;
   static word8 newflag;

#ifdef _WINSOCKAPI_
   static WORD wsaVerReq;
   static WSADATA wsaData;

   wsaVerReq = 0x0101;	/* version 1.1 */
   if(WSAStartup(wsaVerReq, &wsaData) == SOCKET_ERROR)
      fatal("WSAStartup()");
   Needcleanup = 1;
#endif

#ifndef __BORLANDC__
   if(sizeof(WINDEX) != 50)
      fatal("struct size error.\nSet compiler options for byte alignment.");
#endif

   for(j = 1; j < argc; j++) {
      if(argv[j][0] != '-') break;
      switch(argv[j][1]) {
         case 'p':  Port = atoi(&argv[j][2]);   /* TCP port */
                    break;
         case 'a':  if(argv[j][2]) Peeraddr = &argv[j][2];
                    break;
         case 'c':  if(argv[j][2]) Corefile = &argv[j][2];
                    break;
         case 'v':  Verbose = 1;
                    break;
         case 'n':  newflag = 1;
                    break;
         default:   usage();
      }  /* end switch */
   }  /* end for j */

   srand16fast(time(NULL));

   if(newflag) init_wallet(&Whdr);
   else if(argv[j]) {
      strncpy(Wfname, argv[j], WFNAMELEN-1);
      read_wheader(&Whdr);
      printf("Password:\n");
      tgets(Password, PASSWLEN);
      CLEARSCR();
      decrypt_wheader();
      printf("Press RETURN to continue or ctrl-c to cancel...\n");
      getchar();
      mainmenu();
   } else usage();

   delete_windex();
   memset(&Whdr, 0, sizeof(Whdr));

#ifdef _WINSOCKAPI_
    if(Needcleanup)
       WSACleanup();
#endif

   return 0;
}  /* end main() */
