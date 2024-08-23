/**
 * @file mcmd.c
 * @brief Mochimo Server Daemon (mcmd) binary source file.
 * @copyright Adequate Systems LLC, 2018-2024. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
 * @note This binary uses INET support from the extended-c extinet header,
 * requiring the use of wsa_startup() to activate socket support on Windows.
*/


/* define GIT_VERSION blank (if not defined) */
#ifndef GIT_VERSION
   #define GIT_VERSION ""

#endif

/* internal support */
#include "error.h"
#include "peer.h"
#include "trigg.h"

/* external support */
#include "extinet.h"    /* socket support */
#include "extlib.h"     /* general support */

/* system support */
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <signal.h>
#ifndef _WIN32
   #include <execinfo.h>
   #include <unistd.h>

#endif

/* Mochimo run-time options */
char *Opt_cplfile = "coreip.lst";
char *Opt_rplfile = "recent.lst";
char *Opt_wdir = "d";

unsigned long Opt_trustblock = 0;

int Opt_addr = INADDR_ANY;
int Opt_backlog = 256;
int Opt_family = AF_INET;
int Opt_port = PORT1;
int Opt_quorum = 4;
int Opt_reuse = 0;
int Opt_v3 = 0;

/* source file extensions */
#include "mcmd/processor.c"

/* enable socket option helper */
static inline int enablesockopt(SOCKET sd, int level, int optname)
{
   const int optval = 1;
   return setsockopt(sd, level, optname, &optval, sizeof(optval));
}

int check_directory(char *dirname)
{
   char fname[FILENAME_MAX];

   mkdir_p(dirname);
   snprintf(fname, FILENAME_MAX, "%s/chkfile", dirname);
   if (ftouch(fname) == VEOK) return remove(fname);
   perrno("Permission failure, %s", dirname);
   return VERROR;
}

int warn_unreadable(const char *filename)
{
   FILE *fp;
   char errmsg[128];

   /* try read-only open */
   fp = fopen(filename, "rb");
   if (fp == NULL) {
      mcm_strerror(errno, errmsg, sizeof(errmsg));
      pwarn("Permission failure, %s: %s", filename, errmsg);
      return VERROR;
   }

   fclose(fp);
   return VEOK;
}

int veronica(void)
{
   char haiku[256], buff[16];

   trigg_generate(buff);
   trigg_expand(buff, haiku);
   printf("\n%s\n\n", haiku);

   return EXIT_SUCCESS;
}

void debug_dump(void)
{
   char hex[65];

   pdebug("");
   pdebug("LOCAL STATE...");
   pdebug("Cbits = 0x%x", Cbits);
   pdebug("Cblocknum = 0x%s", bnum2hex(Cblocknum, hex));
   pdebug("Cblockhash = 0x%s", hash2hex(Cblockhash, 32, hex));
   pdebug("Prevhash = 0x%s", hash2hex(Prevhash, 32, hex));
   pdebug("Weight = 0x%s", weight2hex(Weight, hex));
   pdebug("");

/* pdebug("GLOBAL COUNTERS...");
   pdebug("Nbalance = %u", Nbalance);
   pdebug("Nhashes = %u", Nhashes);
   pdebug("Niplist = %u", Niplist);
   pdebug("Nnacks = %u", Nnacks);
   pdebug("Nrecvs = %u", Nrecvs);
   pdebug("Nrecverrs = %u", Nrecverrs);
   pdebug("Nrecvsbad = %u", Nrecvsbad);
   pdebug("Nsends = %u", Nsends);
   pdebug("Nsenderrs = %u", Nsenderrs); */
}  /* end debug_dump() */

void print_signal(int sig)
{
   fprintf(stderr, "\n");
   switch (sig) {
		case SIGABRT: fprintf(stderr, "Caught SIGABRT"); break;
		case SIGFPE: fprintf(stderr, "Caught SIGFPE"); break;
		case SIGILL: fprintf(stderr, "Caught SIGILL"); break;
		case SIGINT: fprintf(stderr, "Caught SIGINT"); break;
		case SIGSEGV: fprintf(stderr, "Caught SIGSEGV"); break;
		case SIGTERM: fprintf(stderr, "Caught SIGTERM"); break;
		default: fprintf(stderr, "Caught SIG(%d)", sig);
	}
}

void signal_shutdown(int sig)
{
   print_signal(sig);
   if (Shutdown) {
      fprintf(stderr, "Server (violently) terminated");
      exit(0);
   }
   fprintf(stderr, "Server exiting, please wait...");
   Shutdown = 1;
}

/**
 * Termination signal handler. Also prints backtrace on SIGSEGV.
 * @note compile with "-g -rdynamic" for readable backtrace
 */
void signal_terminate(int sig)
{
   print_signal(sig);

   if (sig == SIGSEGV) {
   #ifdef _WIN32
      fprintf(stderr, "*WIN32 backtrace unavailable*\n");
   #else
      void *array[10];
      size_t size;

      /* get void*'s for all entries on the stack */
      size = backtrace(array, 10);
      backtrace_symbols_fd(array, size, STDERR_FILENO);
   #endif
   }

   abort();
}  /* end signal_terminate() */

void signal_setup(void)
{
#ifdef _WIN32
   /* intialize Windows signal handling */
   signal(SIGFPE, signal_terminate);
   signal(SIGILL, signal_terminate);
   signal(SIGINT, signal_shutdown);
   signal(SIGSEGV, signal_terminate);
   signal(SIGTERM, signal_terminate);
#else  /* ... assume POSIX */
   /* initialize POSIX signal handling */
   struct sigaction ignore, shutdown, terminate;

   /* ignore SIGPIPE - recv()/send() errors handled internally */
   ignore.sa_flags = 0;
   ignore.sa_handler = SIG_IGN;
   sigemptyset(&(ignore.sa_mask));
   sigaction(SIGPIPE, &ignore, NULL);

   /* shutdown signals */
   shutdown.sa_flags = 0;
   shutdown.sa_handler = signal_shutdown;
   sigemptyset(&(shutdown.sa_mask));
   sigaction(SIGINT, &shutdown, NULL);

   /* termination signals */
   terminate.sa_flags = 0;
   terminate.sa_handler = signal_terminate;
   sigemptyset(&(terminate.sa_mask));
   sigaction(SIGFPE, &terminate, NULL);
   sigaction(SIGILL, &terminate, NULL);
   sigaction(SIGSEGV, &terminate, NULL);
   sigaction(SIGTERM, &terminate, NULL);
#endif

#ifndef NDEBUG
   /* debug (data) dump on exit */
   atexit(debug_dump);
#endif
}  /* end signal_setup() */

/* print (and more importantly, log) copyright and host information */
void splashscreen(void)
{
   char hostname[64];
   char addrname[18];

   /* get local machine name and IP address */
   gethostname(hostname, sizeof(hostname));
   gethostip(addrname, sizeof(addrname));

   /* print (and log) copyright information */
   plog("Mochimo Server Daemon %s, built on %s at %s",
      GIT_VERSION, __DATE__, __TIME__);
   plog("Copyright (c) 2018-2024 Adequate Systems, LLC.  All Rights Reserved.");
   plog("See the License Agreement at the links below:");
   plog("   https://mochimo.org/license.pdf (PDF version)");
   plog("   https://mochimo.org/license (TEXT version)");
   printf("\n");
   /* print (and log) host information */
   plog("Network Host Information...");
   plog("  Machine name: %s", *hostname ? hostname : "unknown");
   plog("  IPv4 address: %s", *addrname ? addrname : "0.0.0.0");
   printf("\n");
   sleep(3);
}  /* end splashscreen() */

int usage(void)
{
   printf(
      "\nUSAGE: mcmd [OPTIONS]... [--] [DIRECTORY]"

      "\n\nDIRECTORY:"
      "\n   Defaults to \"d/\""

      "\n\nOPTIONS:"
      "\n       --                     Forces the end of OPTIONS arguments"
      "\n   -c, --cplist=<FILE>        Set core peer list to <FILE>"
      "\n       --dir-bc=<DIR>         Set block archive directory to <DIR>"
      "\n       --disable-api          Disable built in API server"
      "\n       --disable-pinklist     Disable the pinklist of evil peers"
      "\n       --dst-port=<NUM>       Set destination port number to <NUM>"
      "\n   -e, --eplist=<FILE>        Set epoch (peer) pinklist to <FILE>"
      "\n   -h, --help                 Print this usage information"
      "\n       --log-functions        Enhanced Debugging: Trace functions"
      "\n   -l, --log-level=<LL>       Set log level to <LL>"
      "\n      Log levels (0-5) represent the level of detail in logs."
      "\n      Each trace level includes the logs of all lower levels."
      "\n      0: Alert, 2: Error, 4: Info, 5: Debug"
      "\n       --log-timestamps       Enhanced Debugging: Trace timestamps"
      "\n   -m, --mfee=<NUM>           Set minimum fee to <NUM> nMCM (>500)"
      "\n   -p, --port=<NUM>           Set server port number to <NUM>"
      "\n   -q, --quorum=<NUM>         Set network quorum size to <NUM>"
      "\n       --reuse-addr           Enable SO_REUSEADDR on server socket"
      "\n   -r, --rplist=<FILE>        Set recent peer list to <FILE>"
      "\n   -T, --testnet=<PORT>"
      "\n      Testnet initialization requires a donor blockchain."
      "\n      Donor blockchains can be obtained via network sync."
      "\n   -3, --v3-bnum=<NUM>        Enable v3.0 reboot for block <NUM>"
      "\n       --version              Print the current software version"
      "\n\n"
   );

   return EXIT_SUCCESS;
}  /* end usage() */

/*
 * Initialise data and call the server.
 */
int main(int argc, char **argv)
{
   char *argp, *cp;     /* argument pointer (for argv) */
   unsigned long argu;  /* argument unsigned value */
   int eoa, j, k;       /* argument indices */
   int ecode = VEOK;
   int nonopt_args = 0;

   /* sanity checks are executed in isolation */
   {
      /* ensure struct integrity at compile time */
      STATIC_ASSERT(sizeof(word32) == 4, word32_size);
      STATIC_ASSERT(sizeof(MDST) == 20, MDST_struct_size);
      STATIC_ASSERT(sizeof(TXQENTRY) == 2412, TXQENTRY_struct_size);
      STATIC_ASSERT(sizeof(NGHEADER) == 12, NGHEADER_struct_size);
      STATIC_ASSERT(sizeof(BHEADER) == 56, BHEADER_struct_size);
      STATIC_ASSERT(sizeof(BTRAILER) == 160, BTRAILER_struct_size);
      STATIC_ASSERT(sizeof(LENTRY) == 52, LENTRY_struct_size);
      STATIC_ASSERT(sizeof(LTRAN) == 53, LTRAN_struct_size);
      /* ensure little endian architecture at run time */
      if (get16((word8[2]) { 0x34, 0x12 }) != 0x1234u) {
         perr("incompatible endian type");
         return EXIT_FAILURE;
      }
   }

   /* setup signal operations */
   signal_setup();

   /* force debug logging */
   setploglevel(PLOG_DEBUG);

   /* multiple sources of entropy for improved prng */
   srand((unsigned int) time(NULL));
   srand16fast(time(NULL) ^ rand() ^ getpid());
   srand16(time(NULL), rand(), getpid());
   srand32(time(NULL) ^ rand() ^ getpid());

#ifdef _WIN32

   /* request winsock dll version 2.2 for Win32 */
   if (wsa_startup(2, 2) != 0) {
      perrno("wsa_startup() FAILURE");
      return EXIT_FAILURE;
   }

   /* NOTE: Although considered good practice, wsa_cleanup() is NOT
    * called at the end of the process. Since wsa_startup() is only
    * called ONCE at the start of the process, a subsequent call to
    * wsa_cleanup() under all exit conditions is needlessly complex.
    */

#endif

#define OBTAIN_ARGUMENT_POINTER(J, ARGC, ARGV, ARGP) \
   do { \
      ARGP = argvalue(&J, ARGC, ARGV); \
      if (ARGP == NULL) { \
         perr("missing argument value"); \
         return EXIT_FAILURE; \
      } \
      pdebug("    argument pointer: %s", ARGP); \
   } while (0);

#define OBTAIN_ARGUMENT_UNSIGNED(J, ARGC, ARGV, ARGP, ARGU) \
   do { \
      OBTAIN_ARGUMENT_POINTER(J, ARGC, ARGV, ARGP); \
      ARGU = strtoul(ARGP, NULL, 0); \
      if (ARGU == 0 || errno == ERANGE) { \
         perr("invalid argument value \"%s\"", ARGP); \
         return EXIT_FAILURE; \
      } \
      pdebug("    argument unsigned: %lu", ARGU); \
   } while (0);

   /* Parse command line arguments. */
   pdebug("... 0th argument: %s", argv[0]);
   for (eoa = 0, j = 1; !eoa && j < argc; j++) {
      pdebug("... parsing argument: %s", argv[j]);
      if (argv[j][0] == '-') {
         /* check for End Of Arguments indicator */
         if (argument(argv[j], "--", NULL)) {
            /* finish processing argument options */
            if (eoa++ == 0) pdebug("... end of arguments");
            continue;
         }
         /*******************/
         /* RUNTIME OPTIONS */
         if (argument(argv[j], "-c", "--cplist")) {
            /* obtain char pointer */
            OBTAIN_ARGUMENT_POINTER(j, argc, argv, argp);
            warn_unreadable(argp);
            Opt_cplfile = argp;
            continue;
         }
         if (argument(argv[j], NULL, "--disable-pinklist")) {
            /* flag disabled pinklist */
            Nopinklist = 1;
            continue;
         }
         if (argument(argv[j], "-h", "--help")) return usage();
         if (argument(argv[j], "-l", "--log-level")) {
            /* obtain unsigned value -- auto-detect base-n */
            OBTAIN_ARGUMENT_UNSIGNED(j, argc, argv, argp, argu);
            setploglevel(argu);
            continue;
         }
         if (argument(argv[j], "-m", "--mfee")) {
            /* obtain unsigned value -- auto-detect base-n */
            OBTAIN_ARGUMENT_UNSIGNED(j, argc, argv, argp, argu);
            Myfee[0] = argu;
            if (Myfee[0] < Mfee[0]) Myfee[0] = Mfee[0];
            else Cbits |= C_MFEE;
            continue;
         }
         if (argument(argv[j], "-p", "--port")) {
            /* obtain unsigned value -- auto-detect base-n */
            OBTAIN_ARGUMENT_UNSIGNED(j, argc, argv, argp, argu);
            Opt_port = argu;
            continue;
         }
         if (argument(argv[j], "-q", "--quorum")) {
            /* obtain unsigned value -- auto-detect base-n */
            OBTAIN_ARGUMENT_UNSIGNED(j, argc, argv, argp, argu);
            Opt_quorum = argu;
            if (Opt_quorum > MAXQUORUM) {
               perr("quorum exceeds MAXQUORUM=%u", MAXQUORUM);
               return EXIT_FAILURE;
            }
            continue;
         }
         if (argument(argv[j], NULL, "--reuse-addr")) {
            /* flag socket for SO_REUSEADDR */
            Opt_reuse = 1;
            continue;
         }
         if (argument(argv[j], "-r", "--rplist")) {
            /* obtain char pointer */
            OBTAIN_ARGUMENT_POINTER(j, argc, argv, argp);
            warn_unreadable(argp);
            Opt_rplfile = argp;
            continue;
         }
         if (argument(argv[j], "--veronica", "--Veronica")) {
            return veronica();
         }
         if (argument(argv[j], NULL, "--version")) {
            /* print (only) GIT_VERSION information */
            printf(GIT_VERSION "\n");
            return EXIT_SUCCESS;
         }
         if (argument(argv[j], "-3", "--v3-bnum")) {
            /* obtain unsigned value -- auto-detect base-n */
            OBTAIN_ARGUMENT_UNSIGNED(j, argc, argv, argp, argu);
            /* set v3.0 trigger block and continue */
            pdebug("    argument unsigned: %lu", argu);
            V30TRIGGER = argu - 1;
            Opt_v3 = 1;
            continue;
         }
         /*********************/
         /* DEVELOPER OPTIONS */
         if (argument(argv[j], NULL, "--trust-block")) {
            /* Set a trusted block number (for development use only).
             * Skips PoW validation up to specified block (inclusive).
             */

            /* obtain unsigned value -- auto-detect base-n */
            OBTAIN_ARGUMENT_UNSIGNED(j, argc, argv, argp, argu);
            Opt_trustblock = argu;
            continue;
         }
      } else if (argv[j][0]) {
         switch (nonopt_args++) {
            case 0:
            /* set core ip list */
            continue;
               /* first non-option argument -- working directory */
               Opt_wdir = argv[j];
               plog("-- working directory = %s", Opt_wdir);
               /* change working directory -- check location */
               if (check_directory(NULL) != VEOK || cd(Opt_wdir) != 0) {
                  perrno("Cannot change directory to \"%s\"", Opt_wdir);
                  plog("Working directory unavailable...");
                  return EXIT_FAILURE;
               }
               break;
            default:
               perr("Unknown non-option argument, %s", argv[j]);
               return EXIT_FAILURE;
         }  /* end switch nonopt_args */
      }  /* end if arguments... */
   }  /* end for j */

#undef OBTAIN_ARGUMENT_POINTER
#undef OBTAIN_ARGUMENT_UNSIGNED

   /* check extended directory tree permissions */
   //if (check_directory(Bcdir_opt) != VEOK) return EXIT_FAILURE;

   /* splash BEFORE meaningful operations */
   splashscreen();

   /* prepare node server context */
   Node.backlog = Opt_backlog;
   Node.logger = mcmd_logger;
   Node.on_accept = node_accept;
   Node.on_cleanup = mcmd_cleanup;
   Node.on_io = mcmd_io;

   /* prepare node server socket */
   Node.sd = socket(Opt_family, SOCK_STREAM, 0);
   if (Node.sd == INVALID_SOCKET) {
      perrno("socket() FAILURE");
      return EXIT_FAILURE;
   }

   /* Opt_reuse enables SO_REUSEADDR on node socket */
   if (Opt_reuse) {
      ecode = enablesockopt(Node.sd, SOL_SOCKET, SO_REUSEADDR);
      if (ecode == SOCKET_ERROR) {
         perrno("enablesockopt() FAILURE");
         return EXIT_FAILURE;
      }
   }

   /* queue intial network scan */
   if (initial_network_scan(&Node) != VEOK) {
      perr("initial_network_scan() FAILURE");
      return EXIT_FAILURE;
   }

   /* build arguments to pass to server thread */
   MCMD_SERVER_ARGS args = { 0 };
   args.addr.sin_addr.s_addr = htonl(Opt_addr);
   args.addr.sin_family = Opt_family;
   args.addr.sin_port = htons(Opt_port);
   args.sp = &Node;

   /* start worker thread for NODE processing */
   ecode = thread_create(&(args.tid), mcmd_server_thread, &args);
   if (ecode != 0) {
      perrno("mcmd server thread creation FAILURE");
      return EXIT_FAILURE;
   }

   /* run mcmd_processor() loop for completed server work */
   ecode = mcmd_processor(&Node);
   /* ... mcmd_processor() ends on global Shutdown flag or failure */
   if (ecode != VEOK) perr("mcmd_processor() FAILURE");

   /* wait 10 seconds for shutdown */
   if (server_shutdown(&Node, 10) == 0) {
      ecode = thread_join(args.tid);
      if (ecode != 0) perrno("thread_join(mcmd_server) FAILURE");
   } else perrno("server_shutdown() FAILURE");

   return ecode != VEOK ? EXIT_FAILURE : EXIT_SUCCESS;
}  /* end main() */
