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
static FILE *Logfile;
static unsigned int Nerrs;
static unsigned int Nlogs;
static int Loglevel = PLOG_INFO;
static int Logtime;

/* Windows compatibility */
#ifdef _WIN32
   /* localtime_r() is not specified by Windows... */
   #define localtime_r(t, tm) localtime_s(tm, t)

#endif

/* set default "preferred path separator" per OS */
#ifndef PREFERRED_PATH_SEP
   #ifdef _WIN32
      #define PREFERRED_PATH_SEP  "\\"
   #else
      #define PREFERRED_PATH_SEP  "/"
   #endif
#endif

#ifndef MIN
   #define MIN(a, b) ((a) < (b) ? (a) : (b))

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
int argument(char *argv, const char *chk1, const char *chk2)
{
   size_t len;

   /* check argument input array */
   if (argv == NULL || *argv == '\0') return 0;

   /* ignore value separation character '=' */
   len = strcspn(argv, "=");

   /* check for a match in either check options */
   if (chk1 && strncmp(argv, chk1, MIN(len, strlen(chk1))) == 0) return 1;
   if (chk2 && strncmp(argv, chk2, MIN(len, strlen(chk2))) == 0) return 1;

   return 0;
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
   char *vp;

   /* check index count */
   if (*idx >= argc) return NULL;
   /* return characters proceeding '=' ... */
   vp = strchr(argv[*idx], '=');
   if (vp) return ++vp;
   /* ... or next available argument */
   if ((*idx + 1) < argc) {
      return vp = argv[++(*idx)];
   }

   return NULL;
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
         strncat(path, PREFERRED_PATH_SEP, FILENAME_MAX - strlen(path) - 1);
      }
      strncat(path, next, FILENAME_MAX - strlen(path) - 1);
   }
   va_end(args);

   return path;
}

/* inline helper to avoid error description spam in the documentation */
static inline char *mcm__strerror(int errnum, char *buf, size_t bufsz)
{
   const char *cp;

   /* check if error is one of Mochimo's... */
   switch (errnum) {
   /* "EMCM__DESC" is provided to "EMCM__TABLE" for extraction of Mochimo
    * error IDs and descriptions as case values and results, respectively.
    */
#define EMCM__DESC(DESC, ID) case ID: cp = DESC; break;
      EMCM__TABLE(EMCM__DESC)

      default:
         /* ... if NOT, rely on strerror_ext() */
         return strerror_ext(errnum, buf, bufsz);
   }  /* end switch (errnum) */

   /* "copy" to buf (snprintf ensures termination) */
   snprintf(buf, bufsz, "%s", cp);

   return buf;
}  /* end mcm__strerror() */

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
   return mcm__strerror(errnum, buf, bufsz);
}  /* end mcm_strerror() */

/* inline helper to avoid error description spam in the documentation */
static inline char *mcm__strerrorname(int errnum, char *buf, size_t bufsz)
{
   const char *cp = "INTERNAL_ERROR";

   /* check if error is one of Mochimo's... */
   switch (errnum) {
   /* "EMCM__NAME" is provided to "EMCM__TABLE" for extraction of Mochimo
    * error IDs as case values and re-interpretted as textual names.
    */
#define EMCM__NAME(_, NAME) case NAME: cp = #NAME; break;
      EMCM__TABLE(EMCM__NAME)

   #ifdef _GNU_SOURCE
      default:
         /* ... if NOT, use (GNU only) strerrorname_np */
         cp = strerrorname_np(errnum);
         if (cp == NULL) cp = "UNKNOWN_ERROR";
   #endif
   }  /* end switch (errnum) */

   /* "copy" to buf (snprintf ensures termination) */
   snprintf(buf, bufsz, "%s", cp);

   return buf;
}  /* end mcm__strerrorname() */

/**
 * Get a textual name of a Mochimo error code.
 * All Mochimo error codes will return a name representing the @a errnum.
 * All other error codes, either of standard C errno, or an "alternate"
 * errno handled by the extended-c module, return "UNREGISTERED_ERROR".
 * @param errnum Value of the error number to get name
 * @param buf Pointer to a buffer to place the textual description
 * @param bufsz Size of the buffer
 * @return (char *) containing a textual name of error
 */
char *mcm_strerrorname(int errnum, char *buf, size_t bufsz)
{
   return mcm_strerrorname(errnum, buf, bufsz);
}  /* end mcm_strerrorname() */

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
 * @param file file name where log occurrred
 * @param line line number where log occurrred
 * @param fmt A string format (or message) to log
 * @param ... Variable arguments supporting @a fmt
*/
void plogx(int ll, const char *file, int line, const char *fmt, ...)
{
   time_t t;
   struct tm dt;
   va_list args;
   FILE *stream;
   int ecode;
   char error[64];
   char timestamp[28];
   char *filename;

   /* ignore empty format and higher log levels */
   if (fmt == NULL || fmt[0] == 0 || Loglevel < ll) return;

   /* store errno for later */
   ecode = errno;

   /* THREADSAFE atomic lock would start here... */

   /* check for specified output file... */
   if (Logfile) {
      /* ... and print timestamp for specific log file */
      time(&t);
      localtime_r(&t, &dt);
      strftime(timestamp, sizeof(timestamp), "[%F %T%z] ", &dt);
      fprintf(Logfile, "%s ", timestamp);
      stream = Logfile;
      /* ... otherwise, set stream appropriately */
   } else stream = (ll <= PLOG_ERROR) ? stderr : stdout;

   /* print log type prefix */
   switch (ll) {
      case PLOG_ALERT: fprintf(stream, "!!!!!"); break;
      case PLOG_ERRNO: /* fallthrough */
      case PLOG_ERROR: fprintf(stream, "ERROR"); break;
      case PLOG_WARN:  fprintf(stream, "Warn... "); break;
      case PLOG_DEBUG: fprintf(stream, "DEBUG"); break;
   }
   /* print file reference on error or debug type logs */
   if (ll <= PLOG_ERROR || ll == PLOG_DEBUG) {
      /* __FILE__ MAY contain a filepath */
      filename = strrchr(file, PREFERRED_PATH_SEP[0]);
      if (filename) filename++; else filename = (char *) file;
      fprintf(stream, "[%s:%d] ", filename, line);
   }
   /* print log information */
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

void setplogfile(FILE* fp)
{
   Logfile = fp;
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
