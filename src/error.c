/**
 * @private
 * @headerfile error.h <error.h>
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef EXTENDED_ERROR_C
#define EXTENDED_ERROR_C


#include "error.h"

/* internal support */
#include "types.h"

/* external support */
#include "extstring.h"
#include <stdarg.h>  /* for va_list support */

/* Initialize default runtime configuration */
static Mutex Printlock = MUTEX_INITIALIZER;
static Mutex Outputlock = MUTEX_INITIALIZER;
static FILE *Outputfp;
static int Printlevel;
static int Outputlevel;
static unsigned int Nprinterrs;
static unsigned int Nprintlogs;

/* Windows compatibility */
#ifdef _WIN32
   #include <tlhelp32.h>  /* for CreateToolhelp32Snapshot in proc_dups() */
   /* localtime_r() is not specified by Windows.. */
   #define localtime_r(t, tm) localtime_s(tm, t)

#endif

/**
 * Check argument list for options. @a chk1 and/or @a chk2 can be NULL.
 * Compatible with appended values using " " or ":" or "=".<br/>
 * e.g. `--arg <value>1 or `--arg:<value>` or `--arg=<value>`
 * @param argv Pointer to argument list item to check
 * @param chk1 First option to check against @a argv
 * @param chk2 Second option to check against @a argv
 * @returns 1 if either options match argument, else 0 for no match.
*/
int argument(char *argv, char *chk1, char *chk2)
{
   int result = 0;
   char *vp;

   /* remove value identifier, temporarily */
   vp = strchr(argv, '=');
   if (vp) *vp = '\0';
   /* check argv for match */
   if (argv != NULL && *argv) {
      if (chk1 != NULL && strcmp(argv, chk1) == 0) result = 1;
      else if (chk2 != NULL && strcmp(argv, chk2) == 0) result = 1;
   }
   /* replace value identifier */
   if (vp) *vp = '=';

   return result;
}  /* end argument() */

/**
 * Obtain the value associated with the current argument index.
 * Compatible with appended values using " " or "=".<br/>
 * e.g. `--arg <value>` or `--arg=<value>`
 * @param idx Pointer to current argument index (i.e. argv[*idx])
 * @param argc Number of total arguments
 * @param argv Pointer to argument list
 * @returns Char pointer to argument value, else NULL for no value.
*/
char *argvalue(int *idx, int argc, char *argv[])
{
   char *vp = NULL;

   /* check index */
   if (*idx >= argc) return NULL;
   /* scan for value identifier */
   vp = strchr(argv[*idx], '=');
   if (vp) vp++;
   else if (++(*idx) < argc && argv[*idx][0] != '-') {
      vp = argv[*idx];
   } else --(*idx);

   return vp;
}  /* end argvalue() */

/**
 * Convert an address to a hexidecimal string (with ellipsis).
 * Places the first 4 bytes of @a addr, into @a hex as a hexidecimal string,
 * followed by an ellipsis "...".
 * @param addr Pointer to address or array of at least 4 bytes
 * @param hex Pointer to character array of at least 12 bytes
 * @returns Pointer to @a hex.
*/
char *addr2hex(void *addr, char *hex)
{
   word8 *bp = (word8 *) addr;

   snprintf(hex, 12, "%02x%02x%02x%02x...", bp[0], bp[1], bp[2], bp[3]);

   return hex;
}  /* end addr2hex() */

/**
 * Convert a hash to a hexadecimal string.
 * @param hash Pointer to hash
 * @param count Number of hash bytes to convert
 * @param hex Pointer to character array
 * @returns Pointer to @a hex.
*/
char *hash2hex(void *hash, int count, char *hex)
{
   int i;
   word8 *bp;

   bp = (word8 *) hash;
   for (hex[0] = '\0', i = 0; i < count; i++) {
      snprintf(&hex[i * 2], 3, "%02x", bp[i]);
   }

   return hex;
}  /* end hash2hex() */

/**
 * Convert a 64-bit block number to a hexadecimal string.
 * Leading zeros are omitted from hexidecimal string result.
 * @param bnum Pointer to 64-bit block number
 * @param hex Pointer to character array of at least 17 bytes
 * @returns Pointer to @a hex.
*/
char *bnum2hex(void *bnum, char *hex)
{
   word32 *b32 = (word32 *) bnum;

   if (b32[1]) snprintf(hex, 17, "%" P32x "%08" P32x, b32[1], b32[0]);
   else snprintf(hex, 17, "%" P32x, b32[0]);

   return hex;
}  /* end bnum2hex() */

/**
 * Convert a 64-bit block number to a full 64-bit hexadecimal string.
 * @param bnum Pointer to 64-bit block number
 * @param hex Pointer to character array of at least 17 bytes
 * @returns Pointer to @a hex.
*/
char *bnum2hex64(void *bnum, char *hex)
{
   word8 *bp = (word8 *) bnum;

   snprintf(hex, 17, "%02x%02x%02x%02x%02x%02x%02x%02x",
      bp[7], bp[6], bp[5], bp[4], bp[3], bp[2], bp[1], bp[0]);

   return hex;
}  /* end bnum2hex64() */

/**
 * Convert a block (number and hash) to an identifying string.
 * Builds a block identifying string from a block number and hash in the
 * format: `0x<bnum> #<bhash>...`
 * @param bnum Pointer to 64-bit block number
 * @param bhash Pointer to block hash or byte array of at least 4 bytes
 * @param id Pointer to character array of at least 32 bytes
 * @returns Pointer to @a id.
*/
char *block2id(void *bnum, void *bhash, char *id)
{
   char hexstr[17], hashstr[12];

   bnum2hex(bnum, hexstr);
   addr2hex(bhash, hashstr);
   snprintf(id, 32, "0x%s #%s", hexstr, hashstr);

   return id;
}  /* end block2id() */

/**
 * Convert an operation code to an identifying string.
 * @param op Value of the opcode to convert
 * @returns Character pointer to identifying string
*/
char *op2str(unsigned op)
{
   switch (op) {
      case OP_NULL: return "OP_NULL";
      case OP_HELLO: return "OP_HELLO";
      case OP_HELLO_ACK: return "OP_HELLO_ACK";
      case OP_TX: return "OP_TX";
      case OP_FOUND: return "OP_FOUND";
      case OP_GET_BLOCK: return "OP_GET_BLOCK";
      case OP_GET_IPL: return "OP_GET_IPL";
      case OP_SEND_FILE: return "OP_SEND_FILE";
      case OP_SEND_IPL: return "OP_SEND_IPL";
      case OP_BUSY: return "OP_BUSY";
      case OP_NACK: return "OP_NACK";
      case OP_GET_TFILE: return "OP_GET_TFILE";
      case OP_BALANCE: return "OP_BALANCE";
      case OP_SEND_BAL: return "OP_SEND_BAL";
      case OP_RESOLVE: return "OP_RESOLVE";
      case OP_GET_CBLOCK: return "OP_GET_CBLOCK";
      case OP_MBLOCK: return "OP_MBLOCK";
      case OP_HASH: return "OP_HASH";
      case OP_TF: return "OP_TF";
      case OP_IDENTIFY: return "OP_IDENTIFY";
      default: return "OP_UNKNOWN";
   }  /* end switch (op) */
}  /* end op2str() */

/**
 * Convert a 256-bit chain weightto a hexadecimal string.
 * Leading zeros are ommited from hexidecimal string result.
 * @param weight Pointer to 256-bit chain weight (or equivalent value)
 * @param hex Pointer to character array of at least 65 bytes
 * @returns Pointer to @a hex
*/
char *weight2hex(void *weight, char *hex)
{
   word32 *w32 = (word32 *) weight;
   int count, p;

   for (count = 0, p = 7; p >= 0; p--) {
      if (p && w32[p] == 0) continue;
      if (count == 0) count = snprintf(hex, 65, "%" P32x, w32[p]);
      else count += asnprintf(hex, 65, "%08" P32x, w32[p]);
   }

   return hex;
}  /* end weight2hex() */

/**
 * Writes a Mochimo "fully qualified archive name" (FQAN) into a buffer.
 * @param buf Pointer to buffer to write to
 * @param bnum Pointer to a 64-bit block number
 * @param bhash Pointer to a block hash
 * @param ext Name extension
 * @see bc_fqan()
 * @see lt_fqan()
*/
int mcm_fqan(char *buf, char *pre, char *ext, void *bnum, void *bhash)
{
   char bnumstr[17] = "";
   char bhashstr[9] = "";

   if (bnum) bnum2hex64(bnum, bnumstr);
   if (bhash) hash2hex(bhash, 4, bhashstr);

   return (
      snprintf(
         buf, FILENAME_MAX, "%s%s%s%s%s%s",
         pre ? pre : "", bnumstr[0] ? bnumstr : "",
         bhashstr[0] ? "." : "", bhashstr[0] ? bhashstr : "",
         ext ? "." : "", ext ? ext : ""
      )
   );
}  /* end mcm_fqan() */

/**
 * Join multiple strings into a file path written to a buffer.
 * Consider using path_join() instead.
 * @param buf Pointer to a buffer to write to
 * @param count Number of strings to join
 * @param ... Strings to join together
 * @returns (int) value respresenting operation result.
 * 0 for success, or non-zero value representing the number at which count
 * was reduced before an error occurred.
*/
int path_count_join(char *buf, int count, ...)
{
   va_list args;

   *buf = '\0';
   if (count > 0) {
      va_start(args, count);
      strncat(buf, va_arg(args, char *), FILENAME_MAX - strlen(buf));
      for (count--; count > 0; count--) {
         strncat(buf, PATH_SEPARATOR, FILENAME_MAX - strlen(buf));
         strncat(buf, va_arg(args, char *), FILENAME_MAX - strlen(buf));
      }
      va_end(args);
   }

   return count;
}  /* end path_count_join() */

/**
 * Move the cursor position around the screen.
 * @param x number of x axis steps to move, negative for left
 * @param y number of y axis steps to move, negative for up
*/
void move_cursor(int x, int y)
{
#ifdef _WIN32
   COORD coord;
   CONSOLE_SCREEN_BUFFER_INFO csbi;
   HANDLE handle = GetStdHandle(STD_OUTPUT_HANDLE);

   if (handle == INVALID_HANDLE_VALUE) return;
   if (!GetConsoleScreenBufferInfo(handle, &csbi)) return;

   coord.X = csbi.dwCursorPosition.X + x;
   coord.Y = csbi.dwCursorPosition.Y + y;
   SetConsoleCursorPosition(handle, coord);

#else
   if (x < 0) printf("\x1b[%dD", -(x));
   else if (x) printf("\x1b[%dC", x);
   if (y < 0) printf("\x1b[%dA", -(y));
   else if (y) printf("\x1b[%dB", y);

#endif
}

/**
 * Clear line from cursor right.
 * @param fp file pointer to location of clear
*/
void clear_right(FILE *fp)
{
#ifdef _WIN32
   COORD coord;
   HANDLE handle;
   DWORD cells, count;
   CONSOLE_SCREEN_BUFFER_INFO csbi;

   if (fp == stdout) handle = GetStdHandle(STD_OUTPUT_HANDLE);
   else if (fp == stderr) handle = GetStdHandle(STD_ERROR_HANDLE);
   else return;

   if (handle == INVALID_HANDLE_VALUE) return;
   if (!GetConsoleScreenBufferInfo(handle, &csbi)) return;

   coord = csbi.dwCursorPosition;
   cells = csbi.dwSize.X - csbi.dwCursorPosition.X;
   FillConsoleOutputCharacter(handle, (TCHAR) ' ', cells, coord, &count);

#else
   if (fp == stdout || fp == stderr) fprintf(fp, "\x1b[K");

#endif
}

/**
 * Get the number of errors logged by print_extended().
 * @returns unsigned int -- Number of error logs
*/
unsigned int get_num_errs(void)
{
   unsigned int counter;

   mutex_lock(&Outputlock);
   counter = Nprinterrs;
   mutex_unlock(&Outputlock);

   return counter;
}

/**
 * Get the number of logs counted by print_extended().
 * @returns unsigned int -- Number of logs
*/
unsigned int get_num_logs(void)
{
   unsigned int counter;

   mutex_lock(&Outputlock);
   counter = Nprintlogs;
   mutex_unlock(&Outputlock);

   return counter;
}

/**
 * Set the output file for printing logs to file.
 * If @a fname or @a mode are `NULL`, output file is closed.
 * @param fname file name of output file
 * @param mode file mode of output file (e.g. "w" or "a")
 * @returns 0 on success, else 1 on error (see @a errno for details)
*/
int set_output_file(char *fname, char *mode)
{
   int ecode = 0;

   mutex_lock(&Outputlock);
   if (Outputfp) fclose(Outputfp);
   if (fname && mode) {
      Outputfp = fopen(fname, mode);
      if (Outputfp == NULL) ecode = 1;
   } else Outputfp = NULL;
   mutex_unlock(&Outputlock);

   return ecode;
}

/**
 * Set the print level for printing to output.
 * @param level The print level allowed to print to output file
*/
void set_output_level(int level)
{
   mutex_lock(&Outputlock);
   Outputlevel = level;
   mutex_unlock(&Outputlock);
}

/**
 * Set the print level for printing to screen.
 * @param level The print level allowed to print to screen
*/
void set_print_level(int level)
{
   mutex_lock(&Printlock);
   Printlevel = level;
   mutex_unlock(&Printlock);
}

/**
 * Get a textual description of an error code.
 * The error code may be an error of the Mochimo API, or the C API.
 * @param errnum Value of the error number to get description for
 * @return (const char *) representing textual description of a
 * Mochimo ecosystem error, or NULL if none is available;
*/
const char *mcm_errno_text(int errnum)
{
   switch (errnum) {
      case EMCMLEDEBIT:
         return "Ledger entry debit did not match balance";
      case EMCMLECREDITOVERFLOW:
         return "Ledger entry credit overflowed the balance";
      case EMCMLEDEPTH:
         return "Maximum ledger depth reached";
      case EMCMLENOTAVAIL:
         return "Internal Ledger is not available";
      case EMCMLETRANCODE:
         return "Unknown ledger entry transaction code";
      case EMCMNOTXS:
         return "No transactions to handle";
      case EMCMOPCODE:
         return "Unhandled operation code";
      case EMCMOPHELLO:
         return "Missing OP_HELLO packet";
      case EMCMOPHELLOACK:
         return "Missing OP_HELLO_ACK packet";
      case EMCMOPNVAL:
         return "Invalid operation code";
      case EMCMOPRECV:
         return "Received unexpected operation code";
      case EMCMPKTCRC:
         return "Invalid CRC16 packet hash";
      case EMCMPKTIDS:
         return "Unexpected packet identification";
      case EMCMPKTNACK:
         return "Unexpected negative acknowledgement";
      case EMCMPKTNET:
         return "Incompatible packet network";
      case EMCMPKTOPCODE:
         return "Invalid packet opcode";
      case EMCMPKTTLR:
         return "Invalid packet trailer";
      case EMCM_MATH64_OVERFLOW:
         return "Unspecified 64-bit math overflow";
      case EMCM_MATH64_UNDERFLOW:
         return "Unspecified 64-bit math underflow";
      case EMCM_SORT_LENGTH:
         return "Unexpected file length during sort";
      case EMCM_EOF:
         return "Unexpected end-of-file";
      case EMCM_FILECOUNT:
         return "Unexpected number of items in file";
      case EMCM_FILELEN:
         return "Unexpected length of file";
      case EMCM_BHASH:
         return "Bad block hash";
      case EMCM_BNUM:
         return "Bad block number";
      case EMCM_DIFF:
         return "Bad difficulty";
      case EMCM_HDRLEN:
         return "Bad header length";
      case EMCM_MADDR:
         return "Bad miner address";
      case EMCM_MFEE:
         return "Bad miner fee";
      case EMCM_MFEES_OVERFLOW:
         return "Overflow of miner fees";
      case EMCM_MREWARD:
         return "Bad miner reward";
      case EMCM_MREWARDS_OVERFLOW:
         return "Overflow of miner rewards";
      case EMCM_MROOT:
         return "Bad merkle root";
      case EMCM_NONCE:
         return "Bad nonce";
      case EMCM_PHASH:
         return "Bad (previous) block hash";
      case EMCM_STIME:
         return "Bad solve time";
      case EMCM_TCOUNT:
         return "Bad TX count";
      case EMCM_TIME0:
         return "Bad start time";
      case EMCM_TLRLEN:
         return "Bad trailer length";
      case EMCM_TRAILER:
         return "Bad trailer data";
      case EMCM_LE_AMOUNTS_OVERFLOW:
         return "Overflow of ledger amounts";
      case EMCM_LE_AMOUNTS_SUM:
         return "Bad sum of ledger amounts";
      case EMCM_LE_EMPTY:
         return "No records written to ledger file";
      case EMCM_LE_NON_NG:
         return "Ledger cannot be extracted from a non-NG block";
      case EMCM_LE_SORT:
         return "Bad ledger sort";
      case EMCM_LE_TAG_REF:
         return "Bad tag reference to ledger entry";
      case EMCM_LT_CODE:
         return "Bad ledger tx code";
      case EMCM_LT_DEBIT:
         return "Ledger tx debit, does not match ledger entry balance";
      case EMCM_LT_NOT_CREDIT:
         return "Unexpected ledger tx code for ledger entry creation";
      case EMCM_LT_SORT:
         return "Bad ledger tx sort";
      case EMCM_POW_TRIGG:
         return "Bad PoW (Trigg)";
      case EMCM_POW_PEACH:
         return "Bad PoW (Peach)";
      case EMCM_POW_ANOMALY:
         return "Bad PoW Anomaly (bugfix)";
      case EMCM_GENHASH:
         return "Bad Genesis hash";
      case EMCM_NZGEN:
         return "Non-zero Genesis data";
      case EMCM_TX_AMOUNTS_OVERFLOW:
         return "Overflow of TX amounts";
      case EMCM_TX_CHG_ADDR:
         return "Bad TX change address";
      case EMCM_TX_CHG_TAG:
         return "Bad TX change tag";
      case EMCM_TX_DST_ADDR:
         return "Bad TX destination address";
      case EMCM_TX_DST_TAG:
         return "Bad TX destination tag";
      case EMCM_TX_DUP:
         return "Duplicate TX ID";
      case EMCM_TX_FEE:
         return "Bad TX fee";
      case EMCM_TX_ID:
         return "Bad TX ID";
      case EMCM_TX_SIG:
         return "Bad TX signature";
      case EMCM_TX_SORT:
         return "Bad TX sort";
      case EMCM_TX_SRC_ADDR:
         return "Bad TX source address";
      case EMCM_TX_SRC_LE_BALANCE:
         return "Bad TX amounts, not equal to src ledger balance";
      case EMCM_TX_SRC_NOT_FOUND:
         return "Bad TX source, not found in ledger";
      case EMCM_TX_SRC_TAG:
         return "Bad TX source tag";
      case EMCM_TX_SRC_TAGGED:
         return "Bad TX, src tag != chg tag, and src tag non-default";
      case EMCM_TXMDST_AMOUNTS:
         return "Bad multi-destination TX amounts do not match total";
      case EMCM_TXMDST_AMOUNTS_OVERFLOW:
         return "Bad multi-destination TX amounts overflowed";
      case EMCM_TXMDST_CHG_DISSOLVE:
         return "Bad multi-destination TX change tag will dissolve";
      case EMCM_TXMDST_DST_AMOUNT:
         return "Bad multi-destination TX destination amount is zero";
      case EMCM_TXMDST_DST_IS_SRC:
         return "Bad multi-destination TX destination tag is source tag";
      case EMCM_TXMDST_FEES:
         return "Bad multi-destination TX fees do not cover tx fee";
      case EMCM_TXMDST_FEES_OVERFLOW:
         return "Bad multi-destination TX fees overflowed";
      case EMCM_TXMDST_SRC_NOT_CHG:
         return "Bad multi-destination TX src tag != chg tag";
      case EMCM_TXMDST_SRC_TAG:
         return "Bad multi-destination TX missing src tag";
      case EMCM_TXWOTS_SIG:
         return "Bad TX, WOTS+ signature invalid";
      case EMCM_XTX_NZTPADDING:
         return "eXtended TX contains non-zero trailing padding";
      case EMCM_XTX_UNDEFINED:
         return "eXtended TX type is not defined";
      default: return NULL;
   }  /* end switch (errnum) */
}  /* end mcm_errno_text() */

/**
 * Get a textual description of an error code.
 * The error code may be a standard C errno, a Mochimo errno,
 * or an "alternate" errno handled by the extended-c module.
 * @param errnum Value of the error number to get description for
 * @param buf Pointer to a buffer to place the textual description
 * @param bufsz Size of the buffer
 * @return (char *) containing a textual description of error
*/
char *strerror_mcm(int errnum, char *buf, size_t bufsz)
{
   const char *cp;

   /* check if error originates from the Mochimo ecosystem */
   cp = mcm_errno_text(errnum);
   /* if it DOES NOT, use extended-c strerror_ext() */
   if (cp == NULL) return strerror_ext(errnum, buf, bufsz);
   /* copy error description and ensure buf is nul-terminated */
   strncpy(buf, cp, bufsz);
   buf[bufsz - 1] = '\0';
   return buf;
}  /* end strerror_mcm() */

/**
 * Print a message to stdout.
 * @param fmt A string format (or message) for printing
 * @param ... Variable arguments supporting the format string
 * @note Prints to screen, regardless of specified print level.
*/
void print(const char *fmt, ...)
{
   va_list args;

   mutex_lock(&Printlock);

   va_start(args, fmt);
   vfprintf(stdout, fmt, args);
   va_end(args);

   mutex_unlock(&Printlock);
}

/**
 * Print to screen and log to file.
 * If not using a custom errno and log level pair, consider using:
 * perrno(), perr(), pwarn(), plog(), pfine(), or pdebug()
 * @param e error number ( @a errno ) associated with log
 * @param ll print level of log
 * @param fmt A string format (or message) to log
 * @param ... Variable arguments supporting @a fmt
 * @returns int -- VERROR for PLEVEL_ERROR, else VEOK
*/
int pcustom(int e, int ll, const char *fmt, ...)
{
   static const char *PLEVEL_PREFIX[NUM_PLEVELS] = {
      "", "Error. ", "Warning... ", "", "FINE: ", "DEBUG: "
   };

   struct tm dt;
   time_t t;
   FILE *fp;
   va_list args;
   int return_code;
   char error[64] = "";
   char timestamp[32] = "";

   /* set return code */
   return_code = (ll == PLEVEL_ERROR);

   /* ignore NULL fmt's and insufficient print levels */
   if (fmt == NULL) return return_code;
   if ((Outputfp == NULL || Outputlevel < ll) && Printlevel < ll) {
      return return_code;
   }

   /* determine std print location -- per print level */
   fp = (ll == PLEVEL_ERROR) ? stderr : stdout;

   /* print to screen */
   if (Printlevel >= ll) {
      mutex_lock(&Printlock);

      fprintf(fp, "%s", PLEVEL_PREFIX[ll]);
      va_start(args, fmt);
      vfprintf(fp, fmt, args);
      va_end(args);
      if (e != INVALID_ERRNO) {
         fprintf(fp, ": (%d) %s\n", e, strerror_mcm(e, error, 64));
      } else fprintf(fp, "\n");

      mutex_unlock(&Printlock);
   }

   mutex_lock(&Outputlock);

   /* print to output, timestamp: "yyyy-mm-ddThh:mm:ss+0000" */
   if (Outputfp && Outputlevel >= ll) {
      time(&t);
      localtime_r(&t, &dt);
      strftime(timestamp, sizeof(timestamp) - 1, "%FT%T%z - ", &dt);
      fprintf(Outputfp, "%s%s", timestamp, PLEVEL_PREFIX[ll]);
      va_start(args, fmt);
      vfprintf(Outputfp, fmt, args);
      va_end(args);
      if (e != INVALID_ERRNO) {
         fprintf(Outputfp, ": (%d) %s\n", e, strerror_mcm(e, error, 64));
      } else fprintf(Outputfp, "\n");
   }

   /* increment appropriate print counter */
   if (ll == PLEVEL_ERROR) Nprinterrs++;
   Nprintlogs++;

   mutex_unlock(&Outputlock);

   return return_code;
}  /* end pcustom() */

/**
 * Print local host info on stdout.
 * @returns 0 on succesful operation, or (-1) on error.
*/
void phostinfo(void)
{
   char hostname[64] = "";
   char addrname[64] = "";

   /* get local machine name and IP address */
   gethostname(hostname, sizeof(hostname));
   gethostip(addrname, sizeof(addrname));
   print("Local Machine Info\n");
   print("  Machine name: %s\n", *hostname ? hostname : "unknown");
   print("  IPv4 address: %s\n", *addrname ? addrname : "0.0.0.0");
   print("\n");
}  /* end phostinfo() */

/**
 * Check for duplicate running processes, by @a name. Checks running dups
 * by counting the number of running processes matching specified @a name.
 * @param name String repesenting process name to search for
 * @returns The number of duplicate processes, or (-1) on error. The number
 * of "duplicate" processes is considered 0 if One (1) process is found.
*/
int proc_dups(const char *name)
{
   int result = -1;

#ifdef _WIN32
   PROCESSENTRY32 pe = { 0 };
   HANDLE pss;

   /* init */
   pe.dwSize = sizeof(pe);
   /* obtain snapshot to scan processes */
   pss = CreateToolhelp32Snapshot(TH32CS_SNAPALL, 0);
   if (pss != NULL) {
      /* initiate first process scan */
      if (Process32First(pss, &pe)) {
         do {
            /* compare szExeFile with process name */
            if (strncmp(name, pe.szExeFile, MAX_PATH) == 0) result++;
            /* iterate remaining processes in snapshot */
         } while(Process32Next(pss, &pe));
         if (result < 0) result = 0;
      }
      /* close snapshot handle */
      CloseHandle(pss);
   }

/* end _WIN32 routine */
#else  /* assume UNIXLIKE */
   FILE *fd;
   char cmd[48];

   /* use POSIX "pgrep" to list processes by name */
   sprintf(cmd, "pgrep %.32s", name);
   fd = popen(cmd, "r");
   if (fd != NULL) {
      /* count lines of output */
      while (fgets(cmd, sizeof(cmd), fd)) result++;
      if (result < 0) result = 0;
      pclose(fd);
   }

/* end UNIXLIKE routine */
#endif

   return result;
}  /* end proc_dups() */

/**
 * Print (and log) a splashscreen with version information to screen
 * @param execname Name of running process
 * @param version Version string of process
 * @param copy_details Flag to print copyright details when set
*/
void psplash(char *execname, char *version, int copy_details)
{
   plog("");
   plog("%s %s, built " __DATE__ " " __TIME__, execname, version);
   plog("Copyright (c) 2022 Adequate Systems, LLC. All Rights Reserved.");
   if (copy_details) {
      plog("See the License Agreement at the links below:");
      plog("   https://mochimo.org/license.pdf (PDF version)");
      plog("   https://mochimo.org/license (TEXT version)");
      plog("");
   }
}

/* end include guard */
#endif
