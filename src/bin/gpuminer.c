/**
 * @private gpuminer.c
 * @brief Mochimo GPU miner binary.
 * @copyright Adequate Systems LLC, 2025. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

#ifndef MOCHIMO_MINER_C
#define MOCHIMO_MINER_C


/* requirements */
#ifndef _OPENMP
   #error "This program requires OpenMP support (-fopenmp) to compile."
#endif

/* device support */
#include "device.h"

/* internal support */
#include "types.h"   /* for standard mochimo datatypes */
#include "tfile.h"   /* for merkle_root() */
#include "network.h" /* for mochimo communication protocols */
#include "peach.h"   /* for peach algorithm */
#include "ledger.h"  /* for ledger support */
#include "error.h"   /* for error codes */
#include "bup.h"
#include "bcon.h"

/* external support */
#include "base58.h"
#include "crc16.h"
#include "extint.h"
#include "extio.h"
#include "extmath.h"
#include "extthrd.h"
#include "exttime.h"

/* system support */
#include <ctype.h>
#include <omp.h>
#include <signal.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#ifndef VERSION
   #define VERSION "<no-version>"

#endif

/* define MACRO abort() procedures on mutex failure */
#ifndef MUTEX_UNLOCK_OR_ABORT
#define MUTEX_UNLOCK_OR_ABORT(LOCKP) \
   do { \
      if (mutex_unlock(LOCKP) != 0) { \
         perrno("FATAL MUTEX UNLOCK FAILURE"); \
         palert("Unrecoverable // Aborting..."); \
         abort(); \
      } \
   } while (0)
#endif
#ifndef MUTEX_LOCK_OR_ABORT
#define MUTEX_LOCK_OR_ABORT(LOCKP) \
   do { \
      if (mutex_lock(LOCKP) != 0) { \
         perrno("FATAL MUTEX LOCK FAILURE"); \
         palert("Unrecoverable // Aborting..."); \
         abort(); \
      } \
   } while (0)
#endif

/* Windows support */
#ifdef _WIN32
   #pragma comment(lib, "Comdlg32.lib")
   #pragma comment(lib, "Advapi32.lib")
   #include <commdlg.h> /* for GetOpenFileName() */
   #include "win32lean.h"
   #include <wincrypt.h>

#else
   #include <execinfo.h>   /* for backtrace() */
   #include <unistd.h>
   #include <fcntl.h>

#endif

#define BLOCKLEN_MIN  ( sizeof(BHEADER) + TXLEN_DSK_MIN + sizeof(BTRAILER) )

#define GPUMAX 64

enum mcm_miner_mode_t {
   HEADLESS_MODE = 0,
   NODE_MODE,
   POOL_MODE,
};

/* solve handling thread lock/alarm */
Condition Salarm = CONDITION_INITIALIZER;
Mutex Slock = MUTEX_INITIALIZER;
/* store current/previous block data */
BTRAILER BT_solve = {0};
BTRAILER BT_curr = {0};
BTRAILER BT_prev = {0};
FILE *FP_curr = NULL;
FILE *FP_prev = NULL;

/**
 * @private
 * Fills a buffer with random data from urandom or CryptGenRandom (WIN32).
 * Obtain a random unsigned value from urandom or CryptGenRandom (WIN32).
 * @returns Random unsigned value
 */
unsigned urandom(void *buf, size_t bufsz) {
   unsigned seed = 123456789;

   /* use local seed if no buffer */
   if (!buf || !bufsz) {
      bufsz = sizeof(seed);
      buf = &seed;
   }

#ifdef _WIN32
   HCRYPTPROV prov = 0;
   if (CryptAcquireContext(&prov, NULL, NULL, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT | CRYPT_SILENT)) {
      CryptGenRandom(prov, (DWORD) bufsz, (BYTE *) buf);
      CryptReleaseContext(prov, 0);
   }
#else
   int fd = open("/dev/urandom", O_RDONLY);
   if (fd != -1) {
      read(fd, buf, bufsz);
      close(fd);
   }
#endif

   /* return unsigned value from buffer where available */
   if (buf && bufsz >= sizeof(unsigned)) {
      return *((unsigned *) buf);
   }

   return seed;
}

void open_dialog(char *filepath, size_t len)
{
   char *crlf;

#ifdef _WIN32
   OPENFILENAME ofn = { 0 };
   char dirpath[BUFSIZ] = ".";

   /* store current working directory for later restoration */
   GetCurrentDirectory(BUFSIZ, dirpath);
   /* ... on Windows, the file dialog function sets the working
    * directory to that of the selected file...
    */

   filepath[0] = '\0';
   ofn.lStructSize = sizeof(OPENFILENAME);
   ofn.lpstrFile = filepath;
   ofn.nMaxFile = (DWORD) len;
   ofn.lpstrTitle = "Select Mochimo address file...";
   ofn.Flags = OFN_FILEMUSTEXIST | OFN_HIDEREADONLY;
   GetOpenFileName((LPOPENFILENAME) &ofn);

   /* ... restore working directory */
   SetCurrentDirectory(dirpath);

#else
   static const char *openfilewsl = "powershell.exe -Command \"Add-Type -AssemblyName 'System.Windows.Forms'; \\$ofd = New-Object Windows.Forms.OpenFileDialog; \\$ofd.Title = 'Select Mochimo address file...'; if (\\$ofd.ShowDialog() -eq 'OK') { \\$ofd.FileName }\"";
   static const char *openfilecmd = "zenity --file-selection --title=\"Select Mochimo address file...\" 2>/dev/null";
   FILE *fp;
   size_t j;
   int iswsl;
   char c;

   /* check Windows Subsystem for Linux */
   char temp[256] = {0};
   fp = fopen("/proc/sys/kernel/osrelease", "r");
   if (fp == NULL) {
      perrno("Failed osrelease check");
      return;
   }
   fgets(temp, (int) len, fp);
   fclose(fp);

   /* check release for WSL and execute appropriate command */
   iswsl = strstr(temp, "microsoft") && strstr(temp, "WSL");
   fp = popen(iswsl ? openfilewsl : openfilecmd, "r");
   if (fp == NULL) {
      perrno("Failed to open file select dialog");
      return;
   }
   fgets(filepath, (int) len, fp);
   pclose(fp);

   /* cleanse Windows file structure, if space to do so */
   if (iswsl && len > 6 && strlen(filepath) < (len - 4)) {
      c = filepath[0];
      memmove(filepath + 6, filepath + 2, strlen(filepath) + 1);
      memcpy(filepath, "/mnt/", 5);
      filepath[5] = tolower(c);
      for (j = 6; j < len; j++) {
         if (filepath[j] == '\\') {
            filepath[j] = PREFERRED_PATH_SEP[0];
         }
      }
   }

#endif

   /* end at first carriage return or line feed */
   crlf = strpbrk(filepath, "\r\n");
   if (crlf) *crlf = '\0';
}  /* end open_dialog() */

void check_push_peers(void)
{
   /* function scope */
   word32 allpeers[RPLISTLEN], allidx, scanidx;
   word32 pushpeers[RPLISTLEN], pushidx, idx;

   /* thread local scope */
   NODE node;
   word32 peer;
   word16 len;
   char ipstr[16];

   /* read core peers list if no peers */
   if (Rplistidx == 0) {
      read_ipl("coreip.lst", Rplist, RPLISTLEN, &Rplistidx);
      if (Rplistidx == 0) {
         /* use localhost fallback if none found */
         pwarn("Peers Unavailable! Using localhost...");
         addpeer((word32) aton("127.0.0.1"), Rplist, RPLISTLEN, &Rplistidx);
      }
   }

   memset(pushpeers, 0, sizeof(pushpeers));
   scanidx = pushidx = 0;

   /* get IP list from recent peers */
   pdebug("Expanding IP list from recent peers...");
   memcpy(allpeers, Rplist, sizeof(Rplist));
   allidx = Rplistidx;

   /* scan allpeers */
   while (scanidx < allidx) {
      pdebug("Scan progress %u/%u...", scanidx, allidx);

      /* prepare parallel processing scope */
      #pragma omp for private(node, peer, len, ipstr)
      for (idx = scanidx; idx < allidx; idx++) {
         /* get IP list from peer */
         if (get_ipl(&node, allpeers[idx]) == VEOK) {
            /* check compatibility */
            if (node.tx.version[1] & C_PUSH) {
               /* add peer to push list */
               peer = allpeers[idx];
               #pragma omp critical
               if (addpeer(peer, pushpeers, RPLISTLEN, &pushidx)) {
                  pdebug("Added %s to push list", ntoa(&peer, ipstr));
               }
            }
            /* inspect peer list */
            for (len = 0; len < get16(node.tx.len); len += 4) {
               peer = *((word32 *) &node.tx.buffer[len]);
               #pragma omp critical
               if (addpeer(peer, allpeers, RPLISTLEN, &allidx)) {
                  pdebug("Added %s to scan list", ntoa(&peer, ipstr));
               }
            }
         }  /* end if get_ipl() */

         /* atomic increment scan index */
         #pragma omp atomic
            scanidx++;
      }  /* end (omp) for */
   }  /* end while */

   /* add push peers to recent peers list */
   if (pushidx > 0) {
      Rplistidx = 0;
      memset(Rplist, 0, sizeof(Rplist));
      /* add push peers to recent peers list Rplist using for loop */
      for (scanidx = 0; scanidx < pushidx; scanidx++) {
         peer = pushpeers[scanidx];
         if (addpeer(peer, Rplist, RPLISTLEN, &Rplistidx)) {
            pdebug("Added %s to recent peers list", ntoa(&peer, ipstr));
         }
      }
   }

   /* print recent peers list */
   print_ipl(Rplist, Rplistidx);
}  /* end check_push_peers() */

/**
 * Receive packets from NODE *np, and write to file, fname.
 * SOCKET np->sd is set non-blocking, ready to recv data.
 * Returns: VEOK (0) = good, else error code. */
int recv_fp(NODE *np, FILE *fp)
{
   TX *tx;
   time_t prevtime;
   word16 len;

   if (fp == NULL) {
      set_errno(EINVAL);
      return VERROR;
   }

   /* init recv_file() */
   time(&prevtime);
   tx = &(np->tx);

   /* receive packets and write */
   while (recv_tx(np, STD_TIMEOUT) == VEOK) {
      /* check recv'd packet */
      if (get16(tx->opcode) != OP_SEND_FILE) {
         pdebug("(%s) *** invalid opcode", np->id);
         break;
      }
      len = get16(tx->len);
      if (len && fwrite(tx->buffer, len, 1, fp) != 1) {
         pdebug("(%s) *** I/O error", np->id);
         break;
      }
      /* check EOF */
      if (len < sizeof(tx->buffer)) {
         return VEOK;
      }
   }  /* end for */

   return VERROR;
}  /* end recv_fp() */

/**
 * Send packets to NODE *np, and write to file, fname.
 * SOCKET np->sd is set non-blocking, ready to recv data.
 * Set fname NULL send np->tx.blocknum request.
 * Returns: VEOK (0) = good, else error code. */
int send_fp(NODE *np, FILE *fp)
{
   size_t count;
   int ecode;
   TX *tx;

   if (fp == NULL) {
      set_errno(EINVAL);
      return VERROR;
   }

   /* init send_file() */
   tx = &(np->tx);

   /* read and send packets */
   do {
      /* read file data and break on error */
      count = fread(tx->buffer, 1, sizeof(tx->buffer), fp);
      if (count != sizeof(tx->buffer) && ferror(fp)) {
         perr("(%s) *** I/O error", np->id);
         ecode = VERROR;
         break;
      }
      /* send file data and break on EOF */
      put16(tx->len, (word16) count);
      ecode = send_op(np, OP_SEND_FILE);
      if (count != sizeof(tx->buffer)) {
         break;
      }
   } while (ecode == VEOK);

   return ecode;
}  /* end send_fp() */

int network_recv_cblock(void)
{
   BTRAILER bt;
   NODE node;
   FILE *fp;
   long llen;
   int ecode;

   /* obtain latest cblock */
   fp = tmpfile();
   if (fp == NULL) goto ERROR_CLEANUP;
   ecode = callserver(&node, Rplist[0]);
   if (ecode == VEOK) ecode = send_op(&node, OP_GET_CBLOCK);
   if (ecode == VEOK) ecode = recv_fp(&node, fp);
   if (ecode != VEOK) goto ECODE_CLEANUP;
   /* socket cleanup */
   sock_close(node.sd);
   node.sd = INVALID_SOCKET;

   llen = ftell(fp);
   if (llen == (-1)) goto ERROR_CLEANUP;
   if (llen < (long) BLOCKLEN_MIN) {
      set_errno(EMCM_FILEDATA);
      goto ERROR_CLEANUP;
   }

   /* reconstruct block with own maddr */
   ecode = b_adjust_maddr_fp(fp);
   if (ecode != VEOK) goto ECODE_CLEANUP;

   if (fseek(fp, -((long) sizeof(BTRAILER)), SEEK_CUR) != 0) {
      goto ERROR_CLEANUP;
   }
   if (fread(&bt, sizeof(BTRAILER), 1, fp) != 1) goto ERROR_CLEANUP;
   /* check block not already solved (phash+bnum) */
   if (memcmp(bt.phash, BT_solve.phash, HASHLEN) == 0) {
      pdebug("cblock already solved");
      goto OK_CLEANUP;
   }
   /* check diff */
   if (memcmp(&bt, &BT_curr, sizeof(BTRAILER)) == 0) {
      pdebug("cblock unchanged");
      goto OK_CLEANUP;
   }

   /* update current block */
   memset(&BT_solve, 0, sizeof(BTRAILER));
   memcpy(&BT_prev, &BT_curr, sizeof(BTRAILER));
   memcpy(&BT_curr, &bt, sizeof(BTRAILER));
   if (FP_prev) fclose(FP_prev);
   FP_prev = FP_curr;
   FP_curr = fp;

   return VEOK;

OK_CLEANUP:
   if (node.sd != INVALID_SOCKET) sock_close(node.sd);
   if (fp) fclose(fp);
   return VETIMEOUT;

ERROR_CLEANUP:
   ecode = VERROR;
ECODE_CLEANUP:
   if (node.sd != INVALID_SOCKET) sock_close(node.sd);
   if (fp) fclose(fp);
   return ecode;
}  /* end network_recv_cblock() */

int network_send_solve(void)
{
   NODE node;
   FILE *fp;

   /* check solve exists to send */
   if (BT_solve.nonce[0] == 0) return VEOK;

   /* apply trailer to appropriate file */
   if (memcmp(&BT_solve, &BT_curr, 92) == 0) {
      if (FP_curr) fp = FP_curr;
      else return VEOK;
   } else if (memcmp(&BT_solve, &BT_prev, 92) == 0) {
      if (FP_prev) fp = FP_prev;
      else return VEOK;
   } else return VEOK;
   if (fseek(fp, -((long) sizeof(BTRAILER)), SEEK_END) != 0) {
      return VERROR;
   }
   if (fwrite(&BT_solve, sizeof(BTRAILER), 1, fp) != 1) {
      return VERROR;
   }
   /* connect to server for send */
   if (callserver(&node, Rplist[0]) != VEOK) {
      return VERROR;
   }
   put16(node.tx.len, 0);
   if (send_op(&node, OP_MBLOCK) != VEOK) {
      sock_close(node.sd);
      return VERROR;
   }
   rewind(fp);
   if (send_fp(&node, fp) != VEOK) {
      sock_close(node.sd);
      return VERROR;
   }
   print_bup(&BT_solve);
   /* remove temporary files containing block data */
   if (FP_curr) {
      fclose(FP_curr);
      FP_curr = NULL;
   }
   if (FP_prev) {
      fclose(FP_prev);
      FP_prev = NULL;
   }

   return VEOK;
}  /* end network_send_solve() */

/* print (and more importantly, log) server host information */
void phostinfo(void)
{
   char hostname[64];
   char addrname[18];

   /* get local machine name and IP address */
   gethostname(hostname, sizeof(hostname));
   gethostip(addrname, sizeof(addrname));
   /* print (and log) host information */
   plog("Network Host Information...");
   plog("  Machine name: %s", *hostname ? hostname : "unknown");
   plog("  IPv4 address: %s", *addrname ? addrname : "0.0.0.0");
   printf("\n");
}

/**
 * Segmentation fault handler.
 * @note compile with "-g -rdynamic" for readable backtrace
 * @note Not compatible with Windows at this stage
*/
void segfault(int sig)
{
#ifndef _WIN32
   void *array[10];
   size_t size;

   /* get void*'s for all entries on the stack */
   size = backtrace(array, 10);
#else
   fprintf(stderr, "*no backtrace on this system*\n");
#endif

   /* print out all the frames to stderr */
   fprintf(stderr, "Error: signal %d:\n", sig);

#ifndef _WIN32
   backtrace_symbols_fd(array, size, STDERR_FILENO);
#endif

   exit(1);
}

/*
 * Signal handlers
 *
 * Enter monitor on ctrl-C
 */
void ctrlc(int sig)
{
   printf("\n");
   pdebug("Got signal %i", sig);
   signal(SIGINT, ctrlc);
   if (Ininit) Running = 0;
   else Monitor = 1;
}

/*
 * Clear run flag, Running on SIGTERM
 */
void sigterm(int sig)
{
   printf("\n");
   pdebug("Got signal %i", sig);
   signal(SIGTERM, sigterm);
   sock_cleanup();
   Running = 0;
}

void print_usage(void)
{
   fprintf(stdout,
      "usage: mochiminer [options]\n"
      "   -d, --device-interval <num>  device polling interval (ms)\n"
      "   -h, --host <HOST[,HOST]>     list of Headless Mining hosts\n"
      "   -i, --interval <num>         work polling time, in seconds\n"
      "   -l, --log-level <num>        level of detail in logging (0-5)\n"
      "   -m, --maddr <ADDR>           set or select mining address\n"
      "   -N, --node <HOST[,HOST]>     list of Node Mining target hosts\n"
      "   -P, --pool <HOST[,HOST]>     list of Pool Mining target hosts\n"
      "   -p, --port <num>             port number of target\n\n"
   );
}

int main(int argc, char *argv[])
{
   unsigned seeds[8];
   unsigned long argu;
   char *argp;
   int argi;

   DEVICE_CTX device[GPUMAX];
   FILENAME maddrfile = {0};
   int device_count;
   int interval_ms;
MCM_DECL_UNUSED
   int miner_mode;

   {
      /* little endian check -- executed in isolation */
      STATIC_ASSERT(sizeof(word32) == 4, word32_size);
      if (get16((word8[2]) { 0x34, 0x12 }) != 0x1234u) {
         perr("incompatible endian type");
         return EXIT_FAILURE;
      }
   }

   /* logging setup */
   setploglevel(PLOG_INFO);
   /* Ignore all signals. */
   for (argi = 0; argi <= 23; argi++) {
      signal(argi, SIG_IGN);
   }
   /*signal(SIGINT, ctrlc);*/ /* then install ctrl-C handler */
   signal(SIGINT, sigterm);   /* ...and ctrl-C termination */
   signal(SIGTERM, sigterm);  /* ...and software termination */
   signal(SIGSEGV, segfault); /* segmentation fault handler */
#ifndef _WIN32
   signal(SIGCHLD, SIG_DFL);  /* so waitpid() works */
#endif
   /* seed random generators with urandom (or equivalent) */
   srand16fast(urandom(seeds, sizeof(seeds)));
   srand16(seeds[1], seeds[2], seeds[3]);
   srand32(*((unsigned long long *) &seeds[4]));
   /* enable socket support */
   sock_startup();

   /* init - defaults */
   miner_mode = NODE_MODE;
   Port = Dstport = PORT1;
   interval_ms = 10000; /* ms */
   Dynasleep = 10; /* ms */

/* ARGUMENT MACROs */
#define GET_ARGP_OR_EXIT_FAILURE(ARGP) \
   do { \
      ARGP = argvalue(&argi, argc, argv); \
      if (ARGP == NULL) { \
         perr("missing value"); \
         return EXIT_FAILURE; \
      } \
   } while (0)
#define GET_ARGU_OR_EXIT_FAILURE(ARGP, ARGU) \
   do { \
      GET_ARGP_OR_EXIT_FAILURE(ARGP); \
      pdebug("    parsing value: %s", ARGP); \
      ARGU = strtoul(ARGP, NULL, 0); \
      if (errno == ERANGE) { \
         perrno("invalid value"); \
         return EXIT_FAILURE; \
      } \
   } while (0)
#define PARSE_PEERLIST_TOKENS(ARGP) \
   do { \
      char *token = strtok(ARGP, ","); \
      while (token) { \
         pdebug("    parsing token: %s", token); \
         set_errno(0); \
         word32 peer = aton(token); \
         if (peer) addpeer(peer, Rplist, RPLISTLEN, &Rplistidx); \
         else if (errno) perrno("hostname resolution failure"); \
         token = strtok(NULL, ","); \
      } \
   } while (0)

   /* parse command line arguments */
   pdebug("... skipping 0th argument (program name): %s", argv[0]);
   for (argi = 1; Running && argi < argc; argi++) {
      pdebug("... parsing argument: %s", argv[argi]);
      /* ARGUMENT OPTIONS */
      if (argv[argi][0] == '-') {
         if (argument(argv[argi], "-d", "--device-interval")) {
            /* obtain interval value (auto-base) */
            GET_ARGU_OR_EXIT_FAILURE(argp, argu);
            Dynasleep = (word32) argu;
            continue; /* next arg */
         }
         if (argument(argv[argi], "-h", "--host")) {
            /* obtain comma separated hosts value */
            GET_ARGP_OR_EXIT_FAILURE(argp);
            PARSE_PEERLIST_TOKENS(argp);
            continue; /* next arg */
         }
         if (argument(argv[argi], "-i", "--interval")) {
            /* obtain interval value (auto-base) */
            GET_ARGU_OR_EXIT_FAILURE(argp, argu);
            if (argu < 1) {
               perr("invalid interval value");
               return EXIT_FAILURE;
            }
            interval_ms = (word8) argu;
            interval_ms *= 1000;
            continue; /* next arg */
         }
         if (argument(argv[argi], "-l", "--log-level")) {
            /* obtain log value (auto-base) */
            GET_ARGU_OR_EXIT_FAILURE(argp, argu);
            setploglevel((int) argu);
            continue; /* next arg */
         }
         if (argument(argv[argi], "-m", "--maddr")) {
            word8 maddr_chk[ADDR_TAG_LEN + 2];
            /* obtain maddr data or ask for file */
            argp = argvalue(&argi, argc, argv);
            if (argp == NULL) {
               plog("Select a Mining Address file...");
               open_dialog(maddrfile, FILENAME_MAX);
               if (*maddrfile == '\0') {
                  perr("Unspecified Mining Address...");
                  return EXIT_FAILURE;
               }
               /* read mining address */
               pdebug("read mochimo address file: %s", maddrfile);
               if (addr_tag_readfile(maddr_chk, maddrfile) != VEOK) {
                  perrno("Failed to read mining address...");
                  return EXIT_FAILURE;
               }
               /* print (and log) mining address */
            } else {
               /* interpret Base58 mining address (+checksum) */
               if (base58_decode(argp, NULL) != (ADDR_TAG_LEN + 2)) {
                  perr("invalid Mochimo Address length");
                  return EXIT_FAILURE;
               }
               /* convert Base58 to binary mining address */
               if (base58_decode(argp, maddr_chk) != 0) {
                  perrno("base58(Mochimo Address) decode FAILURE");
                  return EXIT_FAILURE;
               }
               /* ensure integrity of data */
               word16 crc = crc16(maddr_chk, ADDR_TAG_LEN);
               if (get16(maddr_chk + ADDR_TAG_LEN) != crc) {
                  perr("invalid Mochimo Address");
                  return EXIT_FAILURE;
               }
            }
            /* set and display mining address */
            set_maddr(maddr_chk);
            plog("Mining Address: %s", argp);
            plog("Mining Address(hex): %02x%02x%02x%02x...%02x%02x%02x%02x",
               maddr_chk[0], maddr_chk[1], maddr_chk[2], maddr_chk[3],
               maddr_chk[16], maddr_chk[17], maddr_chk[18], maddr_chk[19]);
            continue; /* next arg */
         }
         if (argument(argv[argi], "-N", "--node")) {
            /* obtain comma separated hosts value */
            GET_ARGP_OR_EXIT_FAILURE(argp);
            PARSE_PEERLIST_TOKENS(argp);
            miner_mode = NODE_MODE;
            continue; /* next arg */
         }
         if (argument(argv[argi], "-P", "--pool")) {
            /* obtain comma separated hosts value */
            GET_ARGP_OR_EXIT_FAILURE(argp);
            PARSE_PEERLIST_TOKENS(argp);
            miner_mode = POOL_MODE;
            continue; /* next arg */
         }
         if (argument(argv[argi], "-p", "--port")) {
            /* obtain port value (auto-base) */
            GET_ARGU_OR_EXIT_FAILURE(argp, argu);
            /* check port value range */
            if (argu < 1 || argu > 65535) {
               perr("invalid port value");
               return EXIT_FAILURE;
            }
            Port = Dstport = (word16) argu;
            continue; /* next arg */
         }
      }  /* end if (argv[argi][0] == '-') */
      /* unrecognised argument, check usage */
      perr("unrecognised argument");
      print_usage();
      return EXIT_FAILURE;
   }  /* end command line arguments */

   /* print (and log) copyright and version information */
   plog("Mochimo Miner " VERSION ", built " __DATE__ " " __TIME__);
   plog("Copyright (c) 2024 Adequate Systems, LLC.  All Rights Reserved.");
   plog("See the License Agreement at the links below:");
   plog("   https://mochimo.org/license.pdf (PDF version)");
   plog("   https://mochimo.org/license (TEXT version)");
   printf("\n");

   device_count = init_cuda_devices(device, GPUMAX);
   if (device_count < 1) {
      perr("No CUDA devices found.");
      plog("Mining will not be possible...");
      return EXIT_FAILURE;
   }
   plog("Cuda Devices (%d)...", device_count);
   for (int idx = 0; Running && idx < device_count; idx++) {
      plog(" - %s", device[idx].info);
      pdebug("initilizing device...");
      if (peach_init_cuda_device(&device[idx]) != VEOK) {
         perrno("peach initialization FAILURE");
         pwarn("%s will not be utilized...", device[idx].info);
      }
   }

   int paused = 1;
   int thread_idx = 1;

   /* enter (parallel) mining loop */
   #pragma omp parallel num_threads(3)
   {
      BTRAILER *bt = NULL;
      int task_idx = -1;
      int ecode;

      /* master thread takes first index (0) */
      #pragma omp master
         task_idx = 0;
      /* distribute index to remaining threads */
      if (task_idx < 0) {
         #pragma omp atomic capture
            task_idx = thread_idx++;
      }

      /* determine task */
      switch (task_idx) {
         case 2: {
            BTRAILER bt_solve = {0};
            time_t now;

            /* set working block trailer to current */
            bt = &BT_curr;

            /* Task 2: Device handler */
            thread_setname(thread_self(), "device_handler");
            /* Device management loop */
            for (time(&now); Running; millisleep(Dynasleep), time(&now)) {
               /* pause solving when appropriate */
               if (difftime(now, get32(bt->time0)) >= BRIDGEv3 ||
                     BT_solve.nonce[0] || get32(bt->tcount) == 0) {
                  if (!paused) {
                     /* report reason for pause */
                     if (difftime(now, get32(bt->time0)) >= BRIDGEv3) {
                        plog("Work Expired; waiting for work...");
                     } else if (get32(bt->tcount) == 0) {
                        plog("No Transactions; waiting for work...");
                     } else if (BT_solve.nonce[0]) {
                        plog("Work Solved; waiting for work...");
                     }
                     /* pause solving */
                     paused = 1;
                  }  /* end if paused == 0 */
               } else {
                  /* reset paused and status time */
                  if (paused) {
                     pdebug("Resuming work...");
                     paused = 0;
                  }
               }
               /* manage devices solving */
               for (int idx = 0; idx < device_count && !paused; idx++) {
                  /* execute solve protocol per device type */
                  switch (device[idx].type) {
                     case CUDA_DEVICE:
                        ecode = peach_solve_cuda(&device[idx], &BT_curr, 0, &bt_solve);
                        break;
                  /* case OPENCL_DEVICE:
                        solve = peach_solve_opencl(&device[idx], &BT_curr, 0, &bt);
                        break; */
                     default:
                        /* skip */
                        continue;
                  }
                  /* check for solve */
                  if (ecode == VEOK) {
                     /* (double) check solve is valid */
                     if (peach_check(&bt_solve) != VEOK) {
                        perr("peach_check() failed to verify solve!");
                        continue;  /* ... device loop */
                     }
                     /* acquire (exclusive) lock */
                     MUTEX_LOCK_OR_ABORT(&Slock);
                     /* embed (valid) solve time and block hash */
                     put32(bt_solve.stime, (word32) time(NULL));
                     if (get32(bt_solve.time0) == get32(bt_solve.stime)) {
                        put32(bt_solve.stime, (word32) time(NULL) + 1);
                     }
                     sha256(&bt_solve, sizeof(BTRAILER) - HASHLEN, bt_solve.bhash);
                     memcpy(&BT_solve, &bt_solve, sizeof(BTRAILER));
                     pdebug("solve handed to network thread...");
                     /* alert (sleeping) network thread */
                     condition_signal(&Salarm);
                     /* release (exclusive) lock */
                     MUTEX_UNLOCK_OR_ABORT(&Slock);
                  }  /* end if solve */
               }  /* end device loop */
            }  /* end while */
            break;
         }  /* end Device Handler */
         case 1: {
            double hps, total;
            const char *m;

            /* Task 1: Network handler */
            thread_setname(thread_self(), "network_handler");
            /* exclusive "Running" loop... */
            MUTEX_LOCK_OR_ABORT(&Slock);
            while (Running) {
               /* send solve or check network */
               if (network_send_solve() == VEOK) {
                  if (network_recv_cblock() == VEOK) {
                     /* ... currently there is a period of time between
                      * a block transition where a Node does not have
                      * transactions to produce a candidate block and
                      * will simply abort the connection. This results
                      * in miners continuing to mine the previous block
                      * until transactions appear on the next block, or
                      * the BRIDGE time is reached... */
                     /* report stats, or set block trailer to previous */
                     if (bt) {
                        /* only report on block changes */
                        if (get32(bt->bnum) != get32(BT_curr.bnum)) {
                           total = 0.0;
                           /* report block summary */
                           plog("Work summary; block %u(0x%x), difficulty %u",
                              get32(bt->bnum), get32(bt->bnum), bt->difficulty[0]);
                           /* print block work stats and hashrate per device */
                           for (int idx = 0; idx < device_count; idx++) {
                              if (device[idx].status <= DEV_NULL) {
                                 plog(" - %s failure...", device[idx].info);
                                 continue;
                              }
                              total += (double) device[idx].hps;
                              hps = (double) device[idx].hps;
                              m = metric_reduce(&hps);
                              plog(" - %s %.02lf%sH/s", device[idx].info, hps, m);
                           }  /* end device loop */
                           /* repoort total hashrate if device count > 1 */
                           if (device_count > 1) {
                              m = metric_reduce(&total);
                              plog(" - Total %.02lf%sH/s", total, m);
                           }
                        }  /* end if bt */
                     } else bt = &BT_prev;
                     plog("New work; %s:%"P16u" %u(0x%x):%u:%s...",
                        ntoa(Rplist, NULL), Dstport, get32(BT_curr.bnum),
                        get32(BT_curr.bnum), BT_curr.difficulty[0],
                        hash2hex32(BT_curr.mroot, NULL));
                  } else {
                     ecode = sock_errno;
                     if (ecode && !sock_waiting(ecode)) {
                        perrno("network_recv_cblock() FAILURE");
                     }
                  }
               } else {
                  perrno("network_send_solve() FAILURE");
               }
               /* wait for work, sleepy time (5 second timeout)... */
               if (condition_timedwait(&Salarm, &Slock, interval_ms)) {
                  perrno("CONDITION FAILURE");
                  Running = 0;
                  break;
               }
            }  /* end while */
            pdebug("network thread finished...");
            /* release (exclusive) lock */
            MUTEX_UNLOCK_OR_ABORT(&Slock);
            break;
         }  /* end Network Handler */
         default: {
            /* Remaining Threads: Idle */
         /* thread_setname(thread_self(), "idle"); */
            while (Running) millisleep(100);
            break;
         }  /* end Idle */
      }  /* end switch */
      /* acquire (exclusive) lock */
      MUTEX_LOCK_OR_ABORT(&Slock);
      /* alert sleeping thread */
      condition_signal(&Salarm);
      /* release (exclusive) lock */
      MUTEX_UNLOCK_OR_ABORT(&Slock);
   }  /* end parallel */
   pdebug("all threads finished...");

   printf("\n\n");
   return EXIT_SUCCESS;
}  /* end main() */

/* end include guard */
#endif
