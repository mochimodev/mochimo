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
#include "extstring.h"

/* system support */
#include <stdarg.h>  /* for va_list */
#include <math.h>    /* for metric_reduce() */
#ifdef _WIN32  /* Windows compatibility */
   /* localtime_r() is not specified by Windows.. */
   #define localtime_r(t, tm) localtime_s(tm, t)

#endif

/* Initialize default runtime configuration */
static unsigned int Nlogs;
static int Loglevel = LL_INFO;
static int Logtime;

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
}

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
}

/**
 * Append to a character string buffer. Calculates the remaining available
 * space in @a buf using @a bufsz and `strlen(buf)`, and appends at most
 * the remaining available space, less 1 (for the null terminator).
 * @param buf Pointer to character string buffer
 * @param bufsz Size of @a buf, in bytes
 * @param fmt Pointer to null-terminated string specifying how to interpret
 * the data
 * @param ... arguments specifying data to print
 * @returns Number of characters (not insluding the null terminator) which
 * would have been written to @a buf if @a bufsz was ignored.
*/
int asnprintf(char *buf, size_t bufsz, const char *fmt, ...)
{
   va_list args;
   size_t cur;
   int count;

   cur = strlen(buf);
   va_start(args, fmt);
   count = vsnprintf(&buf[cur], bufsz > cur ? bufsz - cur : 0, fmt, args);
   va_end(args);

   return count;
}

/**
 * Convert a 64-bit block number to a hexadecimal string.
 * Leading zeros are omitted from hexidecimal string result.
 * @param bnum Pointer to 64-bit block number
 * @param hex Pointer to character array of at least 17 bytes
 * @returns Pointer to @a hex.
*/
char *bnum2hex(void *bnum, char *hex)
{
   word32 *b32;

   if (bnum) {
      b32 = (word32 *) bnum;
      if (b32[1]) snprintf(hex, 17, "%" P32x "%08" P32x, b32[1], b32[0]);
      else snprintf(hex, 17, "%" P32x, b32[0]);
   } else snprintf(hex, 17, "(null)");

   return hex;
}

/**
 * Convert a 64-bit block number to a full 64-bit hexadecimal string.
 * @param bnum Pointer to 64-bit block number
 * @param hex Pointer to character array of at least 17 bytes
 * @returns Pointer to @a hex.
*/
char *bnum2hex64(void *bnum, char *hex)
{
   word8 *bp;

   if (bnum) {
      bp = (word8 *) bnum;
      snprintf(hex, 17, "%02x%02x%02x%02x%02x%02x%02x%02x",
         bp[7], bp[6], bp[5], bp[4], bp[3], bp[2], bp[1], bp[0]);
   } else snprintf(hex, 17, "(null)");

   return hex;
}

double diffclocktime(clock_t prev)
{
   return (double) (clock() - prev) / CLOCKS_PER_SEC;
}

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
   }
}

/**
 * Convert a 256-bit chain weight to a hexadecimal string.
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
}

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
      case EMCMTXCHGEXISTS:
         return "Change address is not in Ledger";
      case EMCMTXCHGNOLE:
         return "Change address is not in Ledger";
      case EMCMTXCHGNOTAG:
         return "Change address is not Tagged";
      case EMCMTXDSTNOLE:
         return "Destination address is not in Ledger";
      case EMCMTXDSTNOTAG:
         return "Destination address is not Tagged";
      case EMCMTXDUP:
         return "Duplicate transaction ID";
      case EMCMTXFEE:
         return "Fee is invalid";
      case EMCMTXFEEOVERFLOW:
         return "Overflow of transaction feees";
      case EMCMTXID:
         return "Bad transaction ID";
      case EMCMTXOVERFLOW:
         return "Overflow of transaction amounts";
      case EMCMTXSORT:
         return "Bad transaction sort";
      case EMCMTXSRCISCHG:
         return "Source address is change address";
      case EMCMTXSRCISDST:
         return "Source address is destination address";
      case EMCMTXSRCNOLE:
         return "Source address is not in Ledger";
      case EMCMTXSRCNOTAG:
         return "Source address is not Tagged";
      case EMCMTXTAGCHG:
         return "Invalid Tag activation (change address already exists)";
      case EMCMTXTAGSRC:
         return "Invalid Tag activation (source address is tagged)";
      case EMCMTXTOTAL:
         return "Transaction total does not match ledger balance";
      case EMCMTXWOTS:
         return "WOTS+ signature invalid";
      case EMCMXTXCHGTOTAL:
         return "eXtended TX change total is less than fee";
      case EMCMXTXDSTAMOUNT:
         return "eXtended TX destination amount is zero";
      case EMCMXTXFEES:
         return "eXtended TX fee does not match tally";
      case EMCMXTXHASPUNCT:
         return "eXtended TX MEMO contains punctuation character";
      case EMCMXTXNONPRINT:
         return "eXtended TX MEMO contains non-printable character";
      case EMCMXTXNOTERM:
         return "eXtended TX MEMO is missing a null terminator";
      case EMCMXTXNZTPADDING:
         return "eXtended TX contains non-zero trailing padding";
      case EMCMXTXSENDTOTAL:
         return "eXtended TX send total is zero";
      case EMCMXTXTAGMATCH:
         return "eXtended TX destination tag matches source tag";
      case EMCMXTXTAGMISMATCH:
         return "eXtended TX source tag does not match change tag";
      case EMCMXTXTAGNOLE:
         return "eXtended TX destination tag is not in Ledger";
      case EMCMXTXTOTALS:
         return "eXtended TX total does not match tally";
      case EMCMXTXNODEF:
         return "eXtended TX type is not defined";
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
   char error[64];
   char timestamp[28];
   int ecode = errno;

   /* ignore empty format and higher trace levels */
   if (fmt == NULL || fmt[0] == 0 || Loglevel < ll) return;

   /* THREADSAFE atomic lock would start here... */

   /* print timestamp */
   if (Logtime) {
      time(&t);
      localtime_r(&t, &dt);
      strftime(timestamp, sizeof(timestamp), "[%F %T%z] ", &dt);
      printf("%s ", timestamp);
   }
   /* print prefix */
   switch (ll) {
      case LL_ALERT: printf("CRITICAL!!! "); break;
      case LL_ERRNO: /* fallthrough */
      case LL_ERROR: printf("ERROR! "); break;
      case LL_WARN: printf("Warning... "); break;
   /* case LL_DEBUG: */
      default: printf("DEBUG<%s:%d> ", func, line);
   }
   /* print information */
   va_start(args, fmt);
   vprintf(fmt, args);
   va_end(args);
   /* print error details (if errno) */
   if (ll == LL_ERRNO) {
      mcm_strerror(ecode, error, sizeof(error));
      printf(": (%d) %s", ecode, error);
   }
   /* newline and flush to stream */
   printf("\n");
   fflush(stdout);

   /* increment trace log counter */
   Nlogs++;

   /* THREADSAFE atomic lock would end here... */
}  /* end plogx() */

/**
 * Set the logging level cap.
 * @param ll Trace logging level to print (inclusive)
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

/* end include guard */
#endif
