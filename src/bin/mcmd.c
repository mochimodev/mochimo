
#include "mcmd.h"

#ifndef SERVER_THREADS
   #define SERVER_THREADS  ( cpu_cores() )   /* system dependant */

#endif

/* exclusive support */
#include "mcmd_server.c"

/* define EXEC_NAME and GIT_VERSION (if not defined) */

#ifndef EXEC_NAME
   #define EXEC_NAME "Mochimo Server Daemon"

#endif

#ifndef GIT_VERSION
   #define GIT_VERSION "v-unknown"

#endif

#define NODE_IS_WALLET(np)   ( \
   ((np)->pkt.version[0] == 4 && get16((np)->pkt.len) == 1) || \
   ((np)->pkt.version[0] == 5 && ((np)->pkt.version[1] & C_WALLET)) )

int Running = 0;
int Nopush_opt = 0;
word32 Quorum_opt = 4;
word64 Trustblock_opt = 0;
word16 Dstport_opt = PORT1;
word16 Port_opt = PORT1;

word32 Mfee[2] = { MFEE, 0 };
word32 Myfee[2] = { MFEE, 0 };

/* Recent peers lock */
RWLock RplistLock = RWLOCK_INITIALIZER;

/* Server Task List's, and related Mutex's */
Condition ActiveIOAlarm = CONDITION_INITIALIZER;
Mutex ActiveIOLock = MUTEX_INITIALIZER;
Mutex InactiveIOLock = MUTEX_INITIALIZER;
Mutex SyncIOLock = MUTEX_INITIALIZER;
LinkedList ActiveIO = { 0 };
LinkedList InactiveIO = { 0 };
LinkedList SyncIO = { 0 };

/* Server status flags */
int ServerSyncup = 0;
int ServerInit = 0;
int ServerOk = 0;

/* cli options for peerlist filenames */
char *Coreip_opt = "coreip.lst";
char *Epinkip_opt = "epink.lst";
char *Localip_opt = "local.lst";
char *Recentip_opt = "recent.lst";
char *Startip_opt = "start.lst";

/* cli options for peerlist web address */
char *Starthttp_opt = "https://mochimo.org/peers/start";

void signal_handler(int sig)
{
   print("\n");
   switch (sig) {
		case SIGABRT: plog("Caught SIGABRT"); break;
		case SIGFPE:  plog("Caught SIGFPE"); break;
		case SIGILL:  plog("Caught SIGILL"); break;
		case SIGINT:  plog("Caught SIGINT"); break;
		case SIGSEGV: plog("Caught SIGINT"); break;
		case SIGTERM: plog("Caught SIGTERM"); break;
		default: plog("Caught signal %d", sig);
	}
   /* signal server shutdown -- or exit */
   if (Running) Running = 0; else exit(1);
}

void redirect_signals(void)
{
   signal(SIGABRT, signal_handler);
	signal(SIGFPE,  signal_handler);
	signal(SIGILL,  signal_handler);
	signal(SIGINT,  signal_handler);
	signal(SIGSEGV, signal_handler);
	signal(SIGTERM, signal_handler);
}

int check_directory(const char *dir) {
   char permchk[FILENAME_MAX];

   /* build permission check file path */
   snprintf(permchk, FILENAME_MAX, "%s%sperm.chk",
      dir ? dir : "", dir ? PATH_SEPARATOR : "");
   /* touch the file and remove it, checking for failures */
   if (ftouch(permchk) || remove(permchk)) {
      return perrno(errno, "%s%spermission FAILURE",
         dir ? dir : "", dir ? PATH_SEPARATOR " " : "");
   }

   return VEOK;
}

int veronica(void)
{
   char haiku[256], buff[16];
   print("\n%s\n\n", trigg_expand(trigg_generate(buff), haiku));
   return VEOK;
}

int init(void)
{
   /* word8 highblock[8]; */
   word8 genbnum[8] = { 0 };
   word8 genbhash[4] = { 0x00, 0x17, 0x0c, 0x67 };
   char fname[FILENAME_MAX], fpath[FILENAME_MAX];
   char weightstr[65], bnumstr[17];
   int nochaindata;

#undef FnMSG
#define FnMSG(x) "init()" x

   /* read stored peers from disk */
   read_ipl(Epinkip_opt, Epinklist, EPINKLEN, &Epinkidx);
   read_ipl(Localip_opt, Lplist, LPLISTLEN, &Lplistidx);
   if (read_ipl(Recentip_opt, Rplist, RPLISTLEN, &Rplistidx) < 1) {
      /* if no recent peers, try (downloading) start peers */
      remove(Startip_opt);
      // http_get(Starthttp_opt, Startip_opt, 3);
      if (read_ipl(Startip_opt, Rplist, RPLISTLEN, &Rplistidx) < 1) {
         cwd(fpath, sizeof(fpath));
         path_join(fname, fpath, Startip_opt);
         pwarn("No start peers. Consider refreshing %s", fname);
         /* if no start peers, try core peers */
         if (read_ipl(Coreip_opt, Rplist, RPLISTLEN, &Rplistidx) < 1) {
            pwarn("Failed to load core peers. Check installation.");
         }
      }
   }

   /* initialize genesis block filename */
   bc_fqan(fname, genbnum, genbhash);
   path_join(fpath, Bcdir_opt, fname);

   /* derive Ledger filename */
   snprintf(fname, FILENAME_MAX, "%s.0", Lefname_opt);

   /* determine appropriate chain data exists */
   nochaindata = !fexists("tfile.dat");
   nochaindata |= !fexists(fpath);
   nochaindata |= !fexists(fname);

   /* (try) restore core chain files if any do not exist */
   if (nochaindata) {
      pdebug("Core chain files missing, attempting restoration...");
      if (!fexists("genblock.bc")) {
         return perr("Genesis block missing, cannot restore files!");
      } else if (fcopy("genblock.bc", fpath) != VEOK) {
         return perr("Failed to restore %s from Genesis block", fpath);
      }
      remove("tfile.dat");
      if (append_tfile(fpath, "tfile.dat") != VEOK) {
         return perr("Failed to restore Tfile from Genesis block");
      } else if (le_extract("genblock.bc") != VEOK) {
         return perr("Failed to restore Ledger from Genesis block");
      } else pdebug("Restoration of core chain files successful!");
   }

   pdebug("DISABLED: Initializing local chain data...");
   /* Find the last block in bc/ and reset Time0, and Difficulty */
   /* DISABLED FOR NOW...
   if (bc_reset(BCDIR) != VEOK) return perr("bc_reset(BCDIR) failed");
   */
   /* validate our own tfile.dat to compute Weight, check bnum */
   /* DISABLED FOR NOW...
   if (validate_tfile("tfile.dat", highblock, Weight)) return perr("Tfile bad!");
   if (cmp64(Cblocknum, highblock) != 0) return perr("Tfile mismatch!");
   */

   pdebug("Block Number / Weight: 0x%s / 0x%s",
      bnum2hex(Cblocknum, bnumstr), weight2hex(Weight, weightstr));

   return VEOK;
}  /* end init_chain() */

int usage(void)
{
   print(
      "\nUSAGE: mochimo [OPTIONS]... [DIRECTORY]"

      "\n\nDIRECTORY:"
      "\n   Defaults to \"d/\""

      "\n\nOPTIONS:"
      "\n   --                         Forces the end of OPTIONS arguments."
      "\n   -bc, --bc-archive=<dir>    Set block archive directory to <dir>"
      "\n   -h, --help                 Print this usage information"
      "\n   --no-pinklist              Disable the pinklist of evil peers"
      "\n   --no-pushblock             Disable block push capability"
      "\n   -p, --port=<num>"
      "\n      Set server port number to <num>. Valid range is (1-65535)."
      "\n      The operating system may impose additional restrictions..."
      "\n   --private-peers            Allow private peers in peer lists"
      "\n   -q, --quorum=<num>         Set network quorum size to <num>"
      "\n   --version                  Print the current software version"

      "\n\nOPTIONS (logging):"
      "\n   Logging levels (0-5) represent the logging level of detail."
      "\n   A log level includes the logs of all levels below it."
      "\n      5 = debug logs"
      "\n      4 = fine logs"
      "\n      3 = general logs"
      "\n      2 = warnings"
      "\n      1 = errors"
      "\n      0 = none"
      "\n   -ll, --log-level=<num>     Set screen log level to <num>."
      "\n   -o, --output-file=<file>   Set output log file to <file>"
      "\n   -ol, --output-level=<num>  Set output log level to <num>."

      "\n\nOPTIONS (peerlist):"
      "\n   -cp, --core-plist=<file>   Set core fallback list to <file>"
      "\n   -ep, --epink-plist=<file>  Set epoch pinklist to <file>"
      "\n   -lp, --local-plist=<file>  Set local peer list to <file>"
      "\n   -rp, --recent-plist=<file> Set recent peer list to <file>"
      "\n   -sp, --start-plist=<file>  Set start peer list to <file>"
      "\n   -sw, --start-web=<http>    Download start peers from <http>"

      "\n\nOPTIONS (advanced):"
      "\n   -A, --API"
      "\n      API mode enables mysql export of blockchain data and"
      "\n      a simple http server for query and analysis of data."
      "\n   -M, --Mfee=<num>"
      "\n      Set the minimum mining fee threshold to <num> nMCM."
      "\n      All transactions received that do not meet this"
      "\n      threshold will be ignored. Cannot be < 500 nMCM."
      "\n   -S, --Sanctuary=<num>,<last>"
      "\n      Set Sanctuary to <num> and Lastday to the neogenesis"
      "\n      block proceeding <last>. All ledger addresses not in"
      "\n      compliance with network consensus will be dropped."
      "\n   -T, --Testnet=<port>"
      "\n      Testnet initialization requires a donor blockchain."
      "\n      Donor blockchains can be obtained via network sync."
      "\n   -V, --Virtual=<num>"
      "\n      Virtual mode uses alternate ports for receiving and"
      "\n      and sending network communications; exact ports are"
      "\n      defined as PORT1 and PORT2 at compile time. Virtual"
      "\n      modes 1 and 2 (as <num>) may be specified."
      "\n\n"
   );

   return 0;
}  /* end usage() */

int main (int argc, char *argv[])
{
   static int j, eoa, ecode;
   static int int_opt;                  /* integer for cli options */
   static unsigned uint_opt;            /* unsigned for cli options */
   static char *char_opt, *char_opt2;   /* char pointers for cli options */
   static char *proc_name, *working_dir;

   /* redirect signals */
   signal(SIGABRT, signal_handler);
	signal(SIGFPE,  signal_handler);
	signal(SIGILL,  signal_handler);
	signal(SIGINT,  signal_handler);
	signal(SIGSEGV, signal_handler);
	signal(SIGTERM, signal_handler);

   /* derive process name, check for duplicates */
   proc_name = strrchr(argv[0], '/');
   if (proc_name == NULL) proc_name = strrchr(argv[0], '\\');
   if (proc_name) proc_name++; else proc_name = argv[0];
   if (proc_dups(proc_name)) return perr("Process is already running!");
   /* use multiple sources of entropy for improved prng */
   srand((unsigned int) time(NULL));
   srand16((word32) time(NULL), (word32) rand(), (word32) getpid());
   srand16fast((word32) time(NULL) ^ rand() ^ getpid());
   /* init defaults */
   set_print_level(PLEVEL_LOG);
   Noprivate_opt = 1;
   Cbits |= C_PUSH;
   Running = 1;

   /* enable socket support */
   sock_startup();

   /* parse command line arguments */
   for (j = 1, eoa = 0, ecode = VEOK; Running && j < argc; j++) {
      if (argv[j][0] == '-') {
         /***********/
         /* OPTIONS */
         if (eoa || argument(argv[j], NULL, "--")) {
            /* flag to skip remaining options with leading '-' */
            if (eoa++ == 0) pfine("... end of arguments");
         }
         else if (argument(argv[j], "-h", "--help")) {
            /* print usage information and exit */
USAGE:      usage();
            goto EXIT;
         }
         else if (argument(argv[j], NULL, "--no-pinklist")) {
            pfine("... pinklist disabled");
            /* set "nopinklist" flag */
            Nopinklist_opt = 1;
         }
         else if (argument(argv[j], NULL, "--no-pushblock")) {
            pfine("... push blocks disabled");
            /* unset PUSH capability bit and set "nopush" flag */
            Cbits &= ~(C_PUSH);
            Nopush_opt = 1;
         }
         else if (argument(argv[j], "-p", "--port")) {
            /* obtain/check port number as integer */
            char_opt = argvalue(&j, argc, argv);
            if (char_opt == NULL) goto_perr(USAGE, "Missing port number");
            int_opt = atoi(char_opt);
            if (int_opt < 1 || int_opt > 65535)
               goto_perr(USAGE, "Invalid port number");
            pfine("... port = %s (%d)", char_opt, int_opt);
            /* set port number for receive and destination ports */
            Port_opt = Dstport_opt = (word16) int_opt;
         }
         else if (argument(argv[j], NULL, "--private-peers")) {
            pfine("... private peers enabled");
            /* set "noprivate" flag */
            Noprivate_opt = 0;
         }
         else if (argument(argv[j], "-q", "--quorum")) {
            /* obtain/check quorum number as integer */
            char_opt = argvalue(&j, argc, argv);
            if (char_opt == NULL) goto_perr(USAGE, "Missing quorum size");
            uint_opt = strtoul(char_opt, NULL, 0);
            if (uint_opt < 1) goto_perr(USAGE, "Invalid quorum size");
            pfine("... quorum = %s (%u)", char_opt, uint_opt);
            /* set quorum number */
            Quorum_opt = uint_opt;
         }
         else if (argument(argv[j], NULL, "--Veronica")) {
            veronica();
            goto EXIT;
         }
         else if (argument(argv[j], NULL, "--version")) {
            /* print GIT_VERSION information */
            print(GIT_VERSION);
            goto EXIT;
         }
         /*********************/
         /* LOG LEVEL OPTIONS */
         else if (argument(argv[j], "-ll", "--log-level")) {
            /* obtain/check log level as integer */
            char_opt = argvalue(&j, argc, argv);
            if (char_opt == NULL) goto_perr(USAGE, "Missing log level");
            int_opt = atoi(char_opt);
            if (int_opt < 0) goto_perr(USAGE, "Invalid log level");
            pfine("... log level = %s (%d)", char_opt, int_opt);
            /* set (printed) log level */
            set_print_level(int_opt);
         }
         else if (argument(argv[j], "-o", "--output-file")) {
            /* obtain output log filename, use LOGNAME if not specified */
            char_opt = argvalue(&j, argc, argv);
            if (char_opt == NULL) char_opt = LOGNAME;
            pfine("... output log file = %s", char_opt);
            /* set LOGGING capability bit and open output log file */
            Cbits |= C_LOGGING;
            set_output_file(char_opt, "a");
         }
         else if (argument(argv[j], "-ol", "--output-level")) {
            /* obtain/check (output) log level as integer */
            char_opt = argvalue(&j, argc, argv);
            if (char_opt == NULL) goto_perr(USAGE, "Missing output level");
            int_opt = atoi(char_opt);
            if (int_opt < 0) goto_perr(USAGE, "Invalid output level");
            pfine("... output level = %s (%d)", char_opt, int_opt);
            /* set (output) log level */
            set_output_level(int_opt);
         }
         /********************/
         /* PEERLIST OPTIONS */
         else if (argument(argv[j], "-cp", "--core-plist")) {
            /* obtain peerlist filename */
            Coreip_opt = argvalue(&j, argc, argv);
            if (Coreip_opt == NULL) goto_perr(USAGE, "Missing peerlist");
            pfine("... core peerlist = %s", Coreip_opt);
         }
         else if (argument(argv[j], "-ep", "--epink-plist")) {
            /* obtain peerlist filename */
            Epinkip_opt = argvalue(&j, argc, argv);
            if (Epinkip_opt == NULL) goto_perr(USAGE, "Missing peerlist");
            pfine("... epoch pinklist = %s", Epinkip_opt);
         }
         else if (argument(argv[j], "-lp", "--local-plist")) {
            /* obtain peerlist filename */
            Localip_opt = argvalue(&j, argc, argv);
            if (Localip_opt == NULL) goto_perr(USAGE, "Missing peerlist");
            pfine("... local peerlist = %s", Localip_opt);
         }
         else if (argument(argv[j], "-rp", "--recent-plist")) {
            /* obtain peerlist filename */
            Recentip_opt = argvalue(&j, argc, argv);
            if (Recentip_opt == NULL) goto_perr(USAGE, "Missing peerlist");
            pfine("... recent peerlist = %s", Recentip_opt);
         }
         else if (argument(argv[j], "-sp", "--start-plist")) {
            /* obtain peerlist filename */
            Startip_opt = argvalue(&j, argc, argv);
            if (Startip_opt == NULL) goto_perr(USAGE, "Missing peerlist");
            pfine("... start peerlist = %s", Startip_opt);
         }
         else if (argument(argv[j], "-sw", "--start-weblist")) {
            /* obtain peerlist address */
            Starthttp_opt = argvalue(&j, argc, argv);
            if (Starthttp_opt == NULL) goto_perr(USAGE, "Missing address");
            pfine("... start weblist = %s", Starthttp_opt);
         }
         /********************/
         /* ADVANCED OPTIONS */
         else if (argument(argv[j], "-mf", "--mining-fee")) {
            /* obtain/check mining fee as integer */
            char_opt = argvalue(&j, argc, argv);
            if (char_opt == NULL) goto_perr(USAGE, "Missing fee value");
            int_opt = atoi(char_opt);
            if (int_opt < MFEE) goto_perr(USAGE, "Invalid fee value");
            pfine("... mining fee (Myfee) = %s (%d)", char_opt, int_opt);
            /* set Myfee, and MFEE capability bit if non-standard */
            Myfee[0] = int_opt;
            if (cmp64(Myfee, Mfee)) Cbits |= C_MFEE;
         }
         else if (argument(argv[j], NULL, "--Sanctuary")) {
            /* obtain/check Sanctuary/Lastday as unsigned long */
            char_opt = argvalue(&j, argc, argv);
            if (char_opt == NULL) goto_perr(USAGE, "Missing protocol data");
            char_opt2 = strchr(char_opt, ',');
            if (char_opt2) *(char_opt2++) = '\0';  /* create separation */
            else goto_perr(USAGE, "Malformed protocol data");
            Sanctuary_opt = strtoul(char_opt, NULL, 0);
            Lastday_opt = (strtoul(char_opt2, NULL, 0) + 255) & 0xffffff00;
            pfine("... Sanctuary %s (%lu), %s (%lu)",
               char_opt, (unsigned long) Sanctuary_opt,
               char_opt2, (unsigned long) Lastday_opt);
         }
         else if (argument(argv[j], "-V", "--Virtual")) {
            /* obtain/check virtual mode as integer */
            char_opt = argvalue(&j, argc, argv);
            if (*char_opt == '2') { Dstport_opt = PORT1; Port_opt = PORT2; }
            else { char_opt = "1"; Dstport_opt = PORT2; Port_opt = PORT1; }
            pfine("... virtual mode = %c", *char_opt);
         }
         /*********************/
         /* DEVELOPER OPTIONS */
         else if (argument(argv[j], "-tb", "--trust-block")) {
            /* Set a trusted block number (for development use only).
             * Skips PoW validation up to specified block (inclusive). */
            /* obtain trust block as unsigned long */
            char_opt = argvalue(&j, argc, argv);
            if (char_opt == NULL) goto_perr(USAGE, "Missing block number");
            Trustblock_opt = strtoul(char_opt, NULL, 0);
            pfine("... trust block = %s (%lu)", char_opt,
               (unsigned long) Trustblock_opt);
         }
         /******************/
         /* UNKNOWN OPTION */
         else goto_perr(USAGE, "Unknown argument, %s", argv[j]);
      } else if (argv[j][0]) {
         /* additional non-option arguments */
         if (working_dir == NULL) {
            working_dir = argv[j];
            pfine("-- working directory = %s", working_dir);
         }
      }  /* end if arguments... */
   }  /* end for j */

   /* print splashscreen -- 2 seconds */
   psplash(EXEC_NAME, GIT_VERSION, 1);
   millisleep(2000);
   /* Running check */
   if (!Running) goto EXIT;

   /* print host info -- 1 second */
   phostinfo();
   millisleep(1000);
   /* Running check */
   if (!Running) goto EXIT;

   /* change working directory -- check location */
   if (working_dir == NULL) working_dir = "d";
   if (cd(working_dir) != 0) {
      perrno(errno, "Cannot change DIRECTORY to \"%s\"", working_dir);
      plog("Working directory unavailable. Check installation.");
      goto EXIT;
   } else if (fexists(proc_name)) {
      perr("Found executing binary '%s' in working directory", proc_name);
      plog("Cowardly refusing to work in specified directory");
      goto EXIT;
   }

   /* check directory structure and permissions */
   if (check_directory("") != VEOK) goto EXIT;
   if (check_directory(Bcdir_opt) != VEOK) goto EXIT;

   /* intialize peers, chain files -- start server */
   if (init() == VEOK) {
      if (!Running) goto EXIT;
      ecode = server(Port_opt, PORTA);
      /* save dynamic peer lists */
      save_ipl(Recentip_opt, Rplist, RPLISTLEN);
      save_ipl(Epinkip_opt, Epinklist, EPINKLEN);
   } else goto EXIT;

EXIT:
   /* shutdown active sockets */
   sock_cleanup();
   plog("");

   return ecode;
}  /* end main() */
