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

#define BAIL(m) { message = m; goto bail; }

/**
 * Mochimo error code. Sets @a ecode to given value and jumps to label.
 * Example: @code mEcode(FAIL_LABEL, VETIMEOUT); @endcode
 * Requires: @code int ecode; @endcode
 * @param _lbl Label to jump to
 * @param _e Error code to set ecode to
*/
#define mEcode(_lbl, _e)   { ecode = _e; goto _lbl; }

/**
 * Mochimo protocol violation. Calls perr(...) with variable arguments,
 * sets @a ecode to VEBAD2 (indicating that a peer is in violation of
 * protocol and may need pinklisting), and jumps to label.
 * Example: @code mEdrop(FAIL_LABEL, "Violation of protocol"); @endcode
 * Requires: @code int ecode; @endcode
 * @param _lbl Label to jump to
 * @param ... arguments passed to perr()
*/
#define mEdrop(_lbl, ...)  { perr(__VA_ARGS__); mEcode(_lbl, VEBAD2); }

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
int get_option_idx(OPTIONS *opts, int len, char *search);
char *get_option_value(int *idx, char *argv[], int argc);
int stop_bcon(void);
int stop_found(void);
int stop_miner(int make_idle);
void stop_mirror(void);
void stop4update(void);
void fatal2(int exitcode, char *message);
void resign(char *mess);
void restart(char *mess);
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

void ctrlc(int sig);
void sigterm(int sig);
void fix_signals(void);
void close_extra(void);

#ifdef __cplusplus
}  /* end extern "C" */
#endif

/* end include guard */
#endif
