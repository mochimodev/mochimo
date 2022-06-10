/**
 * @file util.h
 * @brief Mochimo utilities support.
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
 * @note The original Polymorphic Shell sort algorithm, shell(), was
 * deprecated in favour of qsort().
 * > For more details see <https://godbolt.org/z/YE7j57Po9>
*/

/* include guard */
#ifndef MOCHIMO_UTILITIES_H
#define MOCHIMO_UTILITIES_H


/* internal support*/
#include "types.h"
#include "network.h"

/* external support */
#include "extos.h"   /* includes <unistd.h> on UNIX */
#ifdef OS_UNIX
   #include <sys/wait.h>
   #include <sys/file.h>
   #include <execinfo.h>

#endif
#include "extprint.h"
#include <errno.h>

#define BAIL(m)   do { message = m; goto bail; } while(0)

/**
 * Mochimo error code. Sets @a ecode to given value and jumps to label.
 * Example: @code mEcode(FAIL_LABEL, VETIMEOUT); @endcode
 * Requires: @code int ecode; @endcode
 * @param _lbl Label to jump to
 * @param _e Error code to set ecode to
*/
#define mEcode(_lbl, _e)   do { ecode = _e; goto _lbl; } while(0)

/**
 * Mochimo protocol violation. Calls perr(...) with variable arguments,
 * sets @a ecode to VEBAD2 (indicating that a peer is in violation of
 * protocol and may need pinklisting), and jumps to label.
 * Example: @code mEdrop(FAIL_LABEL, "Violation of protocol"); @endcode
 * Requires: @code int ecode; @endcode
 * @param _lbl Label to jump to
 * @param ... arguments passed to perr()
*/
#define mEdrop(_lbl, ...) \
   do { perr(__VA_ARGS__); mEcode(_lbl, VEBAD2); } while(0)

/**
 * Mochimo error w/ error number. Calls perrno(...) with variable
 * arguments, sets @a ecode to VERROR, and jumps to label.
 * Example: @code mErrno(FAIL_LABEL, errno, "Failure message"); @endcode
 * Requires: @code int ecode; @endcode
 * @param _lbl Label to jump to
 * @param ... arguments passed to perr() */
#define mErrno(_lbl, ...)  mEcode(_lbl, perrno(errno, __VA_ARGS__))

/**
 * Mochimo error. Calls perr(...) with variable arguments, sets
 * @a ecode to VERROR, and jumps to label.
 * Example: @code mError(FAIL_LABEL, "Failure message"); @endcode
 * Requires: @code int ecode; @endcode
 * @param _lbl Label to jump to
 * @param ... arguments passed to perr()
*/
#define mError(_lbl, ...)  mEcode(_lbl, perr(__VA_ARGS__))

/**
 * Display a fatal @a msg, terminate services and exit with @a ecode.
 * @param ecode value to supply to exit()
 * @param msg message to print with trace
 * @note Non-zero exit codes should expect a restart.
 * If a restart is not desireable, use exitcode Zero (0).
*/
#define fatal(ecode, msg)                          \
   do {                                            \
      pfatal("%s", msg == NULL ? "<nomsg>" : msg); \
      pdebug("Terminating services...");           \
      if (Found_pid) kill(Found_pid, SIGTERM);     \
      if (Bcon_pid) kill(Bcon_pid, SIGTERM);       \
      if (Mqpid) kill(Mqpid, SIGTERM);             \
      if (Mpid) kill(Mpid, SIGTERM);               \
      sock_cleanup();                              \
      Running = 0;                                 \
      while (waitpid(-1, NULL, 0) != -1);          \
      exit(ecode);                                 \
   } while(0)

/**
 * Resign the process (no restart).
 * @param msg Reason for resigning the process
*/
#define resign(msg)  fatal(0, msg)

/**
 * Restart the process.
 * @param msg Reason for restarting the process
*/
#define restart(msg) fatal(1, msg)

/* bnum is little-endian on disk and core. */
#define weight2hex(_weight)   val2hex(_weight, 32, NULL, 0)

typedef struct {
   char *id;
   char *idl;
} OPTIONS;

/* C/C++ compatible function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

char *show(char *state);
void phostinfo(void);
int proc_dups(const char *name);
int argument(char *argv, char *chk1, char *chk2);
char *argvalue(int *idx, int argc, char *argv[]);
char *metric_reduce(double *value);
int stop_bcon(void);
int stop_found(void);
int stop_miner(void);
void stop_mirror(void);
void stop4update(void);
double diffclocktime(clock_t to, clock_t from);
int check_directory(char *dirname);
int clear_directory(char *dname);
void crctx(TX *tx);
word32 gethdrlen(char *fname);
int readtrailer(BTRAILER *trailer, char *fname);
char *val2hex64(void *val, char hex[]);
char *bnum2hex(void *bnum);
char *val2hex(void *val, int len, char *buf, int bufsize);
char *addr2str(void *addr);
char *hash2str(word8 *hash);
char *block2str(void *bnum, void *bhash, char *buf, size_t bufsz);
char *tgets(char *buff, int len);
int accept_block(char *ublock, word8 *newnum);
int read_global(void);
int write_global(void);
void add_weight(word8 *weight, word8 difficulty, word8 *bnum);
void get_mreward(word32 *reward, word32 *bnum);
int append_tfile(char *fname, char *tfile);
word32 set_difficulty(BTRAILER *btp);

#ifdef OS_UNIX
   int lock(char *lockfile, int seconds);
   int unlock(int fd);
   void segfault(int sig);

#endif

void print_bup(BTRAILER *bt, char *solvestr);
void print_splash(char *execname, char *version);
void ctrlc(int sig);
void sigterm(int sig);
void fix_signals(void);
void close_extra(void);

#ifdef __cplusplus
}  /* end extern "C" */
#endif

/* end include guard */
#endif
