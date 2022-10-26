/**
 * @file error.h
 * @brief Mochimo error codes, logging and string support.
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef EXTENDED_ERROR_H
#define EXTENDED_ERROR_H


#include "extio.h"
#include <errno.h>

/* internal helper MACROs */

#define VA_NUMBER 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, \
   50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, \
   33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, \
   16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0

#define VA_SELECT(_1,  _2,  _3,  _4,  _5,  _6,  _7,  _8,  _9, _10, _11, \
   _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, \
   _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, \
   _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, \
   _54, _55, _56, _57, _58, _59, _60, _61, _62, _63, N, ...) N

#define VA_SHIFT(...) VA_SELECT(__VA_ARGS__)

#define VA_COUNT(...) VA_SHIFT(__VA_ARGS__, VA_NUMBER)

#define makeSTR_(x) #x

#define makeSTR(x) makeSTR_(x)

#define goto_perr(_lbl, ...) \
   { perr(__VA_ARGS__); goto _lbl; }

#define goto_pecode(_lbl, ...) \
   { perrno(ecode, __VA_ARGS__); goto _lbl; }

#define goto_perrno(_lbl, ...) \
   { perrno(errno, __VA_ARGS__); goto _lbl; }

#define on_ecode_goto_perr(_cmd, _lbl, ...) \
   if ((ecode = (_cmd))) goto_perr(_lbl, __VA_ARGS__)

#define on_ecode_goto_pecode(_cmd, _lbl, ...) \
   if ((ecode = (_cmd))) goto_pecode(_lbl, __VA_ARGS__)

#define on_ecode_goto_perrno(_cmd, _lbl, ...) \
   if ((ecode = (_cmd))) goto_perrno(_lbl, __VA_ARGS__)

#define lock_on_ecode_goto_pecode(_lock, _lbl, _code_block) \
   if ((ecode = mutex_lock(&(_lock))) == 0) { _code_block; \
      on_ecode_goto_pecode( mutex_unlock(&(_lock)), _lbl, \
         FnMSG(makeSTR(_lock) " UNLOCK FAILURE")); \
   } else { perrno(ecode, FnMSG(makeSTR(_lock) " LOCK FAILURE")); goto _lbl; }

#define lock_on_ecode_goto_perr(_lock, _lbl, _code_block) \
   if ((ecode = mutex_lock(&(_lock))) == 0) { _code_block; \
      on_ecode_goto_perr( mutex_unlock(&(_lock)), _lbl, \
         FnMSG(makeSTR(_lock) " UNLOCK FAILURE")); \
   } else { perr(FnMSG(makeSTR(_lock) " LOCK FAILURE")); goto _lbl; }

#define trylock_on_ecode_goto_pecode(_lock, _lbl, _code_block) \
   if ((ecode = mutex_trylock(&(_lock))) == 0) { _code_block; \
      on_ecode_goto_pecode( mutex_unlock(&(_lock)), _lbl, \
         FnMSG(makeSTR(_lock) " UNLOCK FAILURE")); \
   } else if (ecode != EBUSY) { \
      perrno(ecode, FnMSG(makeSTR(_lock) " TRYLOCK FAILURE")); goto _lbl; }

#define tryrdlock_on_ecode_goto_pecode(_lock, _lbl, _code_block) \
   if ((ecode = rwlock_tryrdlock(&(_lock))) == 0) { _code_block; \
      on_ecode_goto_pecode( rwlock_rdunlock(&(_lock)), _lbl, \
         FnMSG(makeSTR(_lock) " UNLOCK FAILURE")); \
   } else if (ecode != EBUSY) { \
      perrno(ecode, FnMSG(makeSTR(_lock) " TRYLOCK FAILURE")); goto _lbl; }

#define wrlock_on_ecode_goto_pecode(_lock, _lbl, _code_block) \
   if ((ecode = rwlock_wrlock(&(_lock))) == 0) { _code_block; \
      on_ecode_goto_pecode( rwlock_wrunlock(&(_lock)), _lbl, \
         FnMSG(makeSTR(_lock) " UNLOCK FAILURE")); \
   } else { perrno(ecode, FnMSG(makeSTR(_lock) " LOCK FAILURE")); goto _lbl; }

#define rdlock_on_ecode_goto_pecode(_lock, _lbl, _code_block) \
   if ((ecode = rwlock_rdlock(&(_lock))) == 0) { _code_block; \
      on_ecode_goto_pecode( rwlock_rdunlock(&(_lock)), _lbl, \
         FnMSG(makeSTR(_lock) " UNLOCK FAILURE")); \
   } else { perrno(ecode, FnMSG(makeSTR(_lock) " LOCK FAILURE")); goto _lbl; }

/** No print/log level (blank) */
#define PLEVEL_NONE  0
/** Error print/log level */
#define PLEVEL_ERROR 1
/** Warning print/log level */
#define PLEVEL_WARN  2
/** Standard print/log level */
#define PLEVEL_LOG   3
/** Fine print/log level */
#define PLEVEL_FINE  4
/** Debug print/log level */
#define PLEVEL_DEBUG 5

/**
 * @private
 * Number of print levels. Increment when adding more.
*/
#define NUM_PLEVELS  6

#define INVALID_ERRNO   ( (0x7fffffff) )

/**
 * Print/log an error message, with description of @a errnum.
 * @param E @a errno associated with error log message
 * @param ... arguments you would normally pass to printf()
 * @returns (int) VERROR, per pcustom()
*/
#define perrno(E, ...)  pcustom(E, PLEVEL_ERROR, __VA_ARGS__)

/**
 * Print/log an error message.
 * @param ... arguments you would normally pass to printf()
 * @returns (int) VERROR, per pcustom()
*/
#define perr(...)       pcustom(INVALID_ERRNO, PLEVEL_ERROR, __VA_ARGS__)

/**
 * Print/log a warning message.
 * @param ... arguments you would normally pass to printf()
 * @returns (int) VEOK, per pcustom()
*/
#define pwarn(...)      pcustom(INVALID_ERRNO, PLEVEL_WARN, __VA_ARGS__)

/**
 * Print/log a standard message.
 * @param ... arguments you would normally pass to printf()
 * @returns (int) VEOK, per pcustom()
*/
#define plog(...)       pcustom(INVALID_ERRNO, PLEVEL_LOG, __VA_ARGS__)

/**
 * Print/log a fine message.
 * @param ... arguments you would normally pass to printf()
 * @returns (int) VEOK, per pcustom()
*/
#define pfine(...)      pcustom(INVALID_ERRNO, PLEVEL_FINE, __VA_ARGS__)

/**
 * Print/log a debug message.
 * @param ... arguments you would normally pass to printf()
 * @returns (int) VEOK, per pcustom()
*/
#define pdebug(...)     pcustom(INVALID_ERRNO, PLEVEL_DEBUG, __VA_ARGS__)

/**
 * Write a file path into a buffer by joining multiple strings together
 * with the PATH_SEPARATOR.
 * @param buf Pointer to a buffer to write to
 * @param ... Strings to join together
*/
#define path_join(_buf, ...) \
   path_count_join(_buf, VA_COUNT(__VA_ARGS__), __VA_ARGS__)

/**
 * Write a blockchain filename into a buffer.
 * @param buf Pointer to buffer to write to
 * @param bnum (optional) Pointer to a 64-bit block number
 * @param bhash (optional) Pointer to the first 4 bytes of a block hash
*/
#define bc_fqan(buf, bnum, bhash)   mcm_fqan(buf, "b", "bc", bnum, bhash)

/**
 * Write a ledger transaction filename into a buffer.
 * @param buf Pointer to buffer to write to
 * @param bnum (optional) Pointer to a 64-bit block number
 * @param bhash (optional) Pointer to the first 4 bytes of a block hash
*/
#define lt_fqan(buf, bnum, bhash)   mcm_fqan(buf, "l", "lt", bnum, bhash)

/**
 * Mochimo error number types
*/
enum mcm_errno_t {
   /* Force integer enum for compatibility with POSIX standard errno */
   EMCMFORCEINTEGER = -0x7fffffff - 1,
   /* initialize errors well above any existing error numbers */
   EMCMFIRST = 0x8000,

   /** Internal Ledger is not available */
   EMCMLENOTAVAIL,
   /** Maximum ledger depth reached */
   EMCMLEDEPTH,

   /** Unspecified 64-bit math overflow */
   EMCM_MATH64_OVERFLOW,
   /** Unspecified 64-bit math underflow */
   EMCM_MATH64_UNDERFLOW,

   /** Unexpected file length during sort */
   EMCM_SORT_LENGTH,

   /** Unexpected end-of-file */
   EMCM_EOF,
   /** Unexpected number of items in file */
   EMCM_FILECOUNT,
   /** Unexpected length of file */
   EMCM_FILELEN,

   /** Bad block hash */
   EMCM_BHASH,
   /** Bad block number */
   EMCM_BNUM,
   /** Bad difficulty */
   EMCM_DIFF,
   /** Bad header length */
   EMCM_HDRLEN,
   /** Bad miner address */
   EMCM_MADDR,
   /** Bad miner fee */
   EMCM_MFEE,
   /** Overflow of miner fees */
   EMCM_MFEES_OVERFLOW,
   /** Bad miner reward */
   EMCM_MREWARD,
   /** Overflow of miner rewards */
   EMCM_MREWARDS_OVERFLOW,
   /** Bad merkle root */
   EMCM_MROOT,
   /** Bad nonce */
   EMCM_NONCE,
   /** Bad (previous) block hash */
   EMCM_PHASH,
   /** Bad solve time */
   EMCM_STIME,
   /** Bad TX count */
   EMCM_TCOUNT,
   /** Bad start time */
   EMCM_TIME0,
   /** Bad trailer length */
   EMCM_TLRLEN,
   /** Bad trailer data */
   EMCM_TRAILER,

   /** Overflow of ledger amounts */
   EMCM_LE_AMOUNTS_OVERFLOW,
   /** Bad sum of ledger amounts */
   EMCM_LE_AMOUNTS_SUM,
   /** No records written to ledger file */
   EMCM_LE_EMPTY,
   /** Ledger cannot be extracted from a non-NG block */
   EMCM_LE_NON_NG,
   /** Bad ledger sort */
   EMCM_LE_SORT,
   /** Bad tag reference to ledger entry */
   EMCM_LE_TAG_REF,

   /** Bad ledger transaction code */
   EMCM_LT_CODE,
   /** Ledger transaction debit, does not match ledger entry balance */
   EMCM_LT_DEBIT,
   /** Unexpected ledger transaction code for ledger entry creation */
   EMCM_LT_NOT_CREDIT,
   /** Bad ledger transactions sort */
   EMCM_LT_SORT,

   /** Bad PoW (Trigg) */
   EMCM_POW_TRIGG,
   /** Bad PoW (Peach) */
   EMCM_POW_PEACH,
   /** Bad PoW Anomaly (bugfix) */
   EMCM_POW_ANOMALY,

   /** Bad Genesis hash */
   EMCM_GENHASH,
   /** Non-zero Genesis data */
   EMCM_NZGEN,

   /** Missing OP_HELLO packet */
   EMCM_NOHELLO,
   /** Missing OP_HELLO_ACK packet */
   EMCM_NOHELLOACK,
   /** Unrecognised operation code */
   EMCM_OPCODE,
   /** Invalid operation code */
   EMCM_OPINVAL,

   /** Invalid CRC16 packet hash */
   EMCM_PKTCRC,
   /** Unexpected packet identification */
   EMCM_PKTIDS,
   /** Unexpected negative acknowledgement */
   EMCM_PKTNACK,
   /** Incompatible packet network */
   EMCM_PKTNET,
   /** Invalid packet opcode */
   EMCM_PKTOPCODE,
   /** Invalid packet trailer */
   EMCM_PKTTLR,


   /** Overflow of TX amounts */
   EMCM_TX_AMOUNTS_OVERFLOW,
   /** Bad TX change address */
   EMCM_TX_CHG_ADDR,
   /** Bad TX change tag */
   EMCM_TX_CHG_TAG,
   /** Bad TX destination address */
   EMCM_TX_DST_ADDR,
   /** Bad TX destination tag */
   EMCM_TX_DST_TAG,
   /** Duplicate TX ID */
   EMCM_TX_DUP,
   /** Bad TX fee */
   EMCM_TX_FEE,
   /** Bad TX ID */
   EMCM_TX_ID,
   /** Bad TX signature */
   EMCM_TX_SIG,
   /** Bad TX sort */
   EMCM_TX_SORT,
   /** Bad TX source address */
   EMCM_TX_SRC_ADDR,
   /** Bad TX amounts, not equal to src ledger balance */
   EMCM_TX_SRC_LE_BALANCE,
   /** Bad TX source, not found in ledger */
   EMCM_TX_SRC_NOT_FOUND,
   /** Bad TX source tag */
   EMCM_TX_SRC_TAG,
   /** Bad TX, src tag != chg tag, and src tag non-default */
   EMCM_TX_SRC_TAGGED,

   /** Bad multi-destination TX amounts do not match total */
   EMCM_TXMDST_AMOUNTS,
   /** Bad multi-destination TX amounts overflowed */
   EMCM_TXMDST_AMOUNTS_OVERFLOW,
   /** Bad multi-destination TX change tag will dissolve */
   EMCM_TXMDST_CHG_DISSOLVE,
   /** Bad multi-destination TX destination amount is zero */
   EMCM_TXMDST_DST_AMOUNT,
   /** Bad multi-destination TX destination tag is source tag */
   EMCM_TXMDST_DST_IS_SRC,
   /** Bad multi-destination TX fees do not cover tx fee */
   EMCM_TXMDST_FEES,
   /** Bad multi-destination TX fees overflowed */
   EMCM_TXMDST_FEES_OVERFLOW,
   /** Bad multi-destination TX src tag != chg tag */
   EMCM_TXMDST_SRC_NOT_CHG,
   /** Bad multi-destination TX missing src tag */
   EMCM_TXMDST_SRC_TAG,

   /** Bad TX, WOTS+ signature invalid */
   EMCM_TXWOTS_SIG,

   /** eXtended TX contains non-zero trailing padding */
   EMCM_XTX_NZTPADDING,
   /** eXtended TX type is not defined */
   EMCM_XTX_UNDEFINED,
};

/* C/C++ compatible function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

char *addr2hex(void *addr, char *hex);
char *hash2hex(void *hash, int count, char *hex);
char *bnum2hex(void *bnum, char *hex);
char *bnum2hex64(void *bnum, char *hex);
char *block2id(void *bnum, void *bhash, char *id);
char *weight2hex(void *weight, char *hex);
int mcm_fqan(char *buf, char *pre, char *ext, void *bnum, void *bhash);
int path_count_join(char *buf, int count, ...);
void move_cursor(int x, int y);
void clear_right(FILE *fp);
unsigned int get_num_errs(void);
unsigned int get_num_logs(void);
int set_output_file(char *fname, char *mode);
void set_output_level(int level);
void set_print_level(int level);
const char *mcm_errno_text(int errnum);
char *strerror_mcm(int errnum, char *buf, size_t bufsz);
void print(const char *fmt, ...);
int pcustom(int e, int ll, const char *fmt, ...);
void phostinfo(void);
int proc_dups(const char *name);
void psplash(char *execname, char *version, int copy_details);

#ifdef __cplusplus
}  /* end extern "C" */
#endif

/* end include guard */
#endif
