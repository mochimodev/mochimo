/**
 * @private
 * @headerfile error.h <error.h>
 * @copyright Adequate Systems LLC, 2018-2023. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_ERROR_C
#define MOCHIMO_ERROR_C


#include "error.h"

/* internal support */
#include "types.h"

/* external support */
#include <stdarg.h>  /* for va_list support */
#include <math.h>    /* for isnan/isinf/log10/pow */
#include "extstring.h"

/* Initialize default runtime configuration */
static unsigned int Nerrs;
static unsigned int Nlogs;
static int Loglevel = PLOG_INFO;
static int Logfunc;
static int Logtime;

/* Windows compatibility */
#ifdef _WIN32
   /* localtime_r() is not specified by Windows... */
   #define localtime_r(t, tm) localtime_s(tm, t)

#endif

/**
 * Check argument list for options. @a chk1 and/or @a chk2 can be NULL.
 * Compatible with values separated by " " or "=".<br/>
 * e.g. `--arg <value> or `--arg=<value>`
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
 * Convert a block number into a blockchain filename (w/ 64-bit hex).
 * @param bnum Pointer to 64-bit block number
 * @param fname Pointer to output character array, or NULL
 * @returns Pointer to provided @a fname or (if not provided)
 * internal static buffer containing the resulting output.
*/
char *bnum2fname(word8 bnum[8], char fname[21])
{
   word8 *bp;
   static char sbuf[21];

   /* static buffer check */
   if (fname == NULL) fname = sbuf;

   bp = (word8 *) bnum;
   snprintf(fname, 21, "b%02x%02x%02x%02x%02x%02x%02x%02x.bc",
      bp[7], bp[6], bp[5], bp[4], bp[3], bp[2], bp[1], bp[0]);

   return fname;
}

/**
 * Convert a 64-bit block number to a hexadecimal string.
 * Leading zeros are omitted from hexidecimal string result.
 * @param bnum Pointer to 64-bit block number
 * @param hex Pointer to output character array, or NULL
 * @returns Pointer to provided @a hex or (if not provided)
 * internal static buffer containing the resulting output.
*/
char *bnum2hex(word8 bnum[8], char hex[17])
{
   word32 *b32;
   static char sbuf[17];

   /* static buffer check */
   if (hex == NULL) hex = sbuf;

   b32 = (word32 *) bnum;
   if (b32[1]) snprintf(hex, 17, "%" P32x "%08" P32x, b32[1], b32[0]);
   else snprintf(hex, 17, "%" P32x, b32[0]);

   return hex;
}

/**
 * Convert a 64-bit block number to a full 64-bit hexadecimal string.
 * @param bnum Pointer to 64-bit block number
 * @param hex Pointer to output character array, or NULL
 * @returns Pointer to provided @a hex or (if not provided)
 * internal static buffer containing the resulting output.
*/
char *bnum2hex64(word8 bnum[8], char hex[17])
{
   word8 *bp;
   static char sbuf[17];

   /* static buffer check */
   if (hex == NULL) hex = sbuf;

   bp = (word8 *) bnum;
   snprintf(hex, 17, "%02x%02x%02x%02x%02x%02x%02x%02x",
      bp[7], bp[6], bp[5], bp[4], bp[3], bp[2], bp[1], bp[0]);

   return hex;
}

double diffclocktime(clock_t prev)
{
   return (double) (clock() - prev) / CLOCKS_PER_SEC;
}

/**
 * Convert 32 bits of a hash to a hexadecimal string.
 * @param hash Pointer to 32-bits of any byte array
 * @param hex Pointer to output character array, or NULL
 * @returns Pointer to provided @a hex or (if not provided)
 * internal static buffer containing the resulting output.
*/
char *hash2hex32(word8 hash[4], char hex[9])
{
   word8 *bp;
   static char sbuf[17];

   /* static buffer check */
   if (hex == NULL) hex = sbuf;

   bp = (word8 *) hash;
   snprintf(hex, 9, "%02x%02x%02x%02x", bp[0], bp[1], bp[2], bp[3]);

   return hex;
}

char *metric_reduce(double *value)
{
   static char M[8][3] = { "", "K", "M", "G", "T", "P", "E", "Z" };
   static int MLEN = sizeof(M) / sizeof (*M);
   int m;

   /* check value for sanity */
   if (isnan(*value) || isinf(*value) || *value < 1.0) return M[0];

   /* calculate metric number and reduce value */
   m = (int) (log10(*value) / 3);
   if (m >= MLEN) m = MLEN - 1;
   *value /= pow(1000.0, (double) m);

   return M[m];
}

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
 * Convert a function return code to an identifying string.
 * @param ve Function return code to convert
 * @returns Character pointer to identifying string
*/
char *ve2str(int ve)
{
   switch (ve) {
      case VEWAITING: return "VEWAITING";
      case VETIMEOUT: return "VETIMEOUT";
      case VEOK: return "VEOK";
      case VERROR: return "VERROR";
      case VEBAD: return "VEBAD";
      case VEBAD2: return "VEBAD2";
      default: return "(unknown)";
   }  /* end switch (ve) */
}  /* end ve2str() */

/**
 * CONSIDER USING THE path_join() MACRO;
 * Join multiple strings into a file path written to a buffer.
 * @param path Pointer to character array[FILENAME_MAX], or NULL
 * @param count Number of string parameters to join
 * @param ... Strings to join together
 * @returns Pointer to provided @a path or (if not provided)
 * internal static buffer containing the resulting output.
*/
char *path_join_count(char path[FILENAME_MAX], int count, ...)
{
   va_list args;
   char *next;
   static char sbuf[FILENAME_MAX];

   /* static buffer and usage check */
   if (path == NULL) path = sbuf;

   /* join variable arguments together */
   va_start(args, count);
   for (*path = '\0'; count > 0; count --) {
      next = va_arg(args, char *);
      if (next == NULL || *next == '\0') continue;
      if (*path != '\0') {
         strncat(path, PATH_SEP, FILENAME_MAX - strlen(path) - 1);
      }
      strncat(path, next, FILENAME_MAX - strlen(path) - 1);
   }
   va_end(args);

   return path;
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
      /* core relateed errors... */
      case EMCM_MATH64_OVERFLOW:
         return "Unspecified 64-bit math overflow";
      case EMCM_MATH64_UNDERFLOW:
         return "Unspecified 64-bit math underflow";

      /* file related errors... */
      case EMCM_EOF:
         return "Unexpected end-of-file";
      case EMCM_FILECOUNT:
         return "Unexpected number of items in file";
      case EMCM_FILEDATA:
         return "Unexpected file data";
      case EMCM_FILELEN:
         return "Unexpected length of file";
      case EMCM_SORTLEN:
         return "Unexpected file length during sort";

      /* block related errors... */
      case EMCM_BHASH:
         return "Bad block hash";
      case EMCM_BNUM:
         return "Bad block number";
      case EMCM_DIFF:
         return "Bad difficulty";
      case EMCM_GENHASH:
         return "Bad Genesis hash";
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
      case EMCM_NZGEN:
         return "Non-zero Genesis data";
      case EMCM_PHASH:
         return "Bad (previous) block hash";
      case EMCM_PTIME:
         return "Bad TOT time";
      case EMCM_STIME:
         return "Bad solve time";
      case EMCM_TCOUNT:
         return "Bad TX count";
      case EMCM_TIME0:
         return "Bad start time";
      case EMCM_TLRLEN:
         return "Bad trailer length";
      case EMCM_TMAX:
         return "Too many transactions";
      case EMCM_TRAILER:
         return "Bad trailer data";

      /* ledger entry related errors... */
      case EMCM_LEOVERFLOW:
         return "Overflow of ledger amounts";
      case EMCM_LEEMPTY:
         return "No records written to ledger file";
      case EMCM_LEEXTRACT:
         return "Ledger cannot be extracted from a non-NG block";
      case EMCM_LESORT:
         return "Bad ledger sort";
      case EMCM_LESUM:
         return "Bad sum of ledger amounts";
      case EMCM_LETAG:
         return "Bad tag reference to ledger entry";

      /* ledger transaction related errors... */
      case EMCM_LTCODE:
         return "Bad ledger transaction code";
      case EMCM_LTCREDIT:
         return "Unexpected ledger transaction code for ledger entry creation";
      case EMCM_LTDEBIT:
         return "Ledger transaction debit, does not match ledger entry balance";
      case EMCM_LTSORT:
         return "Bad ledger transactions sort";

      /* network related errors... */
      case EMCM_OPCODE:
         return "Unhandled operation code";
      case EMCM_OPHELLO:
         return "Missing OP_HELLO packet";
      case EMCM_OPHELLOACK:
         return "Missing OP_HELLO_ACK packet";
      case EMCM_OPNVAL:
         return "Invalid operation code";
      case EMCM_OPRECV:
         return "Received unexpected operation code";
      case EMCM_PKTCRC:
         return "Invalid CRC16 packet hash";
      case EMCM_PKTIDS:
         return "Unexpected packet identification";
      case EMCM_PKTNACK:
         return "Unexpected negative acknowledgement";
      case EMCM_PKTNET:
         return "Incompatible packet network";
      case EMCM_PKTOPCODE:
         return "Invalid packet opcode";
      case EMCM_PKTTLR:
         return "Invalid packet trailer";

      /* POW related errors... */
      case EMCM_POWTRIGG:
         return "Bad PoW (Trigg)";
      case EMCM_POWPEACH:
         return "Bad PoW (Peach)";
      case EMCM_POWANOMALY:
         return "Bad PoW Anomaly (bugfix)";

      /* transaction related errors... */
      case EMCM_TX0:
         return "No transactions to handle";
      case EMCM_TXCHGEXISTS:
         return "Change address is not in Ledger";
      case EMCM_TXCHGNOLE:
         return "Change address is not in Ledger";
      case EMCM_TXCHGNOTAG:
         return "Change address is not Tagged";
      case EMCM_TXDSTNOLE:
         return "Destination address is not in Ledger";
      case EMCM_TXDSTNOTAG:
         return "Destination address is not Tagged";
      case EMCM_TXFEE:
         return "Fee is invalid";
      case EMCM_TXFEE_OVERFLOW:
         return "Overflow of transaction feees";
      case EMCM_TXID:
         return "Bad transaction ID";
      case EMCM_TXIDDUP:
         return "Duplicate transaction ID";
      case EMCM_TXINVAL:
         return "Invalid transaction";
      case EMCM_TXOVERFLOW:
         return "Overflow of transaction amounts";
      case EMCM_TXSORT:
         return "Bad transaction sort";
      case EMCM_TXCHG:
         return "Source address is change address";
      case EMCM_TXDST:
         return "Source address is destination address";
      case EMCM_TXSRCDUP:
         return "Duplicate transaction source address";
      case EMCM_TXSRCLE:
         return "Source address is not in Ledger";
      case EMCM_TXSRCNOTAG:
         return "Source address is not Tagged";
      case EMCM_TXTAGCHG:
         return "Invalid Tag activation (change address already exists)";
      case EMCM_TXTAGSRC:
         return "Invalid Tag activation (source address is tagged)";
      case EMCM_TXTOTAL:
         return "Transaction total does not match ledger balance";
      case EMCM_TXWOTS:
         return "WOTS+ signature invalid";

      /* eXtended transaction related errors... */
      case EMCM_XTXCHGTOTAL:
         return "eXtended TX change total is less than fee";
      case EMCM_XTXDSTAMOUNT:
         return "eXtended TX destination amount is zero";
      case EMCM_XTXFEES:
         return "eXtended TX fee does not match tally";
      case EMCM_XTXHASPUNCT:
         return "eXtended TX MEMO contains punctuation character";
      case EMCM_XTXNONPRINT:
         return "eXtended TX MEMO contains non-printable character";
      case EMCM_XTXNOTERM:
         return "eXtended TX MEMO is missing a null terminator";
      case EMCM_XTXNZTPADDING:
         return "eXtended TX contains non-zero trailing padding";
      case EMCM_XTXSENDTOTAL:
         return "eXtended TX send total is zero";
      case EMCM_XTXTAGMATCH:
         return "eXtended TX destination tag matches source tag";
      case EMCM_XTXTAGMISMATCH:
         return "eXtended TX source tag does not match change tag";
      case EMCM_XTXTAGNOLE:
         return "eXtended TX destination tag is not in Ledger";
      case EMCM_XTXTOTALS:
         return "eXtended TX total does not match tally";
      case EMCM_XTXUNDEF:
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
char *mcm_strerror(int errnum, char *buf, size_t bufsz)
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
}  /* end mcm_strerror() */

/**
 * Get the number of errors printed.
 * @returns Number of errors
*/
unsigned int perrcount(void)
{
   return Nerrs;
}

/**
 * Get the number of logs printed.
 * @returns Number of logs
*/
unsigned int plogcount(void)
{
   return Nlogs;
}

/**
 * Print a log to screen.
 * @param ll level of log to be printed
 * @param func function name where log occurrred
 * @param line line number where log occurrred
 * @param fmt A string format (or message) to log
 * @param ... Variable arguments supporting @a fmt
*/
void plogx(int ll, const char *func, int line, const char *fmt, ...)
{
   time_t t;
   struct tm dt;
   va_list args;
   FILE *stream;
   int ecode;
   char error[64];
   char timestamp[28];

   /* ignore empty format and higher log levels */
   if (fmt == NULL || fmt[0] == 0 || Loglevel < ll) return;

   /* determine appropriate stream and save errno */
   stream = (ll < PLOG_WARN) ? stderr : stdout;
   ecode = errno;

   /* THREADSAFE atomic lock would start here... */

   /* print timestamp */
   if (Logtime) {
      time(&t);
      localtime_r(&t, &dt);
      strftime(timestamp, sizeof(timestamp), "[%F %T%z] ", &dt);
      fprintf(stream, "%s ", timestamp);
   }
   /* print prefix */
   switch (ll) {
      case PLOG_ALERT: fprintf(stream, "CRITICAL!!! "); break;
      case PLOG_ERRNO: /* fallthrough */
      case PLOG_ERROR: fprintf(stream, "ERROR! "); break;
      case PLOG_WARN: fprintf(stream, "Warning... "); break;
      case PLOG_DEBUG: fprintf(stream, "DEBUG "); break;
   }
   /* print function reference (always on DEBUG) */
   if (Logfunc || ll == PLOG_DEBUG) {
      fprintf(stream, "<%s:%d> ", func, line);
   }
   /* print information */
   va_start(args, fmt);
   vfprintf(stream, fmt, args);
   va_end(args);
   /* print error details (if errno) */
   if (ll == PLOG_ERRNO) {
      mcm_strerror(ecode, error, sizeof(error));
      fprintf(stream, ": (%d) %s", ecode, error);
   }
   /* newline and flush to stream */
   fprintf(stream, "\n");
   fflush(stream);

   /* increment log counter/s */
   if (ll <= PLOG_ERROR) Nerrs++;
   Nlogs++;

   /* THREADSAFE atomic lock would end here... */
}  /* end plogx() */

/**
 * Set logging function references.
 * @param ll Value to set option (boolean)
*/
void setplogfunctions(int val)
{
   Logfunc = val;
}

/**
 * Set the logging level cap.
 * @param ll Log level to print (inclusive)
*/
void setploglevel(int ll)
{
   Loglevel = ll;
}

/**
 * Set logging timestamps option.
 * @param val Value to set option (boolean)
*/
void setplogtime(int val)
{
   Logtime = val;
}

/**
 * Convert a 256-bit chain weight to a hexadecimal string.
 * Leading zeros are ommited from hexidecimal string result.
 * @param weight Pointer to 256-bit chain weight (or equivalent value)
 * @param hex Pointer to 65 byte character array, or NULL
 * @returns Pointer to provided @a hex or (if not provided)
 * internal static buffer containing the resulting output.
*/
char *weight2hex(word8 weight[32], char hex[65])
{
   word32 *dp;
   char *cp;
   int i;
   static char sbuf[65];

   /* static buffer check */
   if (hex == NULL) hex = sbuf;

   /* skip empty values -- print initial hex value w/o leading zeros */
   dp = (word32 *) weight;
   for (i = 7; i > 0 && dp[i] == 0; i--);
   snprintf(hex, 65, "%x", dp[i--]);
   for (cp = hex + strlen(hex); i >= 0; i--, cp += strlen(cp)) {
      snprintf(cp, 64 - (cp - hex), "%08x", dp[i]);
   }

   return hex;
}

/* end include guard */
#endif
