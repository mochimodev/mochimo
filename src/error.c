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

#include "types.h"
#include <string.h>  /* for string support */
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
#if OS_WINDOWS
   #include <tlhelp32.h>  /* for CreateToolhelp32Snapshot in proc_dups() */
   /* localtime_r() is not specified by Windows.. */
   #define localtime_r(t, tm) localtime_s(tm, t)

#endif

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
 * @param pre (optional) Name prefix
 * @param ext (optional) Name extension
 * @param bnum (optional) Pointer to a 64-bit block number
 * @param bhash (optional) Pointer to a block hash
 * @see bc_fqan()
 * @see lt_fqan()
*/
int mcm_fqan(char *buf, char *pre, char *ext, void *bnum, void *bhash)
{
   char bnumstr[17] = "0000000000000000";
   char bhashstr[9] = "00170c67";

   return (
      snprintf(
         buf, FILENAME_MAX, "%s%s%s%s%s%s",
         pre ? pre : "", bnum ? bnum2hex64(bnum, bnumstr) : bnumstr,
         bhash ? "." : "", bhash ? hash2hex(bhash, 4, bhashstr) : bhashstr,
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
 * Resolve conflicting error codes between POSIX and Windows Sockets.
 * These error codes "may" share a base value with one of the POSIX errno
 * codes defined by the system. This function resolves potential conflicts,
 * so that a print function that derives a textual description from the
 * error code can provide a reliable description to any error code.
 * - WSA_INVALID_HANDLE = 6L
 * - WSA_NOT_ENOUGH_MEMORY = 8L
 * - WSA_INVALID_PARAMETER = 87L
 * @param errnum Error code obtain from @a sock_errno of extinet.h
 * @returns (int) reliable error number for use in perrno() (if desired)
*/
int resolve_wsa_conflicts(long errnum)
{
#if OS_WINDOWS
   switch (errnum) {
      case WSA_INVALID_HANDLE: return EINVAL;
      case WSA_NOT_ENOUGH_MEMORY: return ENOMEM;
      case WSA_INVALID_PARAMETER: return EINVAL;
      default: return (int) errnum;
   }
#else
   return (int) errnum;
#endif
}

/**
 * Move the cursor position around the screen.
 * @param x number of x axis steps to move, negative for left
 * @param y number of y axis steps to move, negative for up
*/
void move_cursor(int x, int y)
{
#if OS_WINDOWS
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
#if OS_WINDOWS
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
      if (e >= 0) {
         fprintf(fp, ": (%d) %s\n", e, errno_text(e));
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
      if (e >= 0) {
         fprintf(Outputfp, ": (%d) %s\n", e, errno_text(e));
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
