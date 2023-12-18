/**
 * @file error.h
 * @brief Mochimo error codes, logging and util support.
 * @copyright Adequate Systems LLC, 2018-2023. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_ERROR_H
#define MOCHIMO_ERROR_H


/* external support */
#include "exterrno.h"
#include "extio.h"
#include <time.h>

/* log level constants */
#define LL_ALERT  0
#define LL_ERRNO  1
#define LL_ERROR  2
#define LL_WARN   3
#define LL_INFO   4
#define LL_DEBUG  5

/* internal helper MACROs */
#define makeSTR_(x) #x
#define makeSTR(x) makeSTR_(x)
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

/**
 * Write a file path into a buffer by joining multiple strings together
 * with the PATH_SEPARATOR.
 * @param buf Pointer to a buffer to write to
 * @param ... Strings to join together
*/
#define path_join(_buf, ...) \
   path_count_join(_buf, VA_COUNT(__VA_ARGS__), __VA_ARGS__)

/**
 * Print an alert level trace log.
 * @param ... arguments you would normally pass to printf()
*/
#define palert(...) \
   plogx(LL_ALERT, __FUNCTION__, __LINE__, "" __VA_ARGS__)

/**
 * Print an error level trace log, with description of @a errnum.
 * @param ... arguments you would normally pass to printf()
*/
#define perrno(...) \
   plogx(LL_ERRNO, __FUNCTION__, __LINE__, "" __VA_ARGS__)

/**
 * Print an error level trace log.
 * @param ... arguments you would normally pass to printf()
*/
#define perr(...) \
   plogx(LL_ERROR, __FUNCTION__, __LINE__, "" __VA_ARGS__)

/**
 * Print a warning level trace log.
 * @param ... arguments you would normally pass to printf()
*/
#define pwarn(...) \
   plogx(LL_WARN, __FUNCTION__, __LINE__, "" __VA_ARGS__)

/**
 * Print an information level trace log.
 * @param ... arguments you would normally pass to printf()
*/
#define plog(...) \
   plogx(LL_INFO, __FUNCTION__, __LINE__, "" __VA_ARGS__)

/**
 * Print a debugging level trace log.
 * @param ... arguments you would normally pass to printf()
*/
#define pdebug(...) \
   plogx(LL_DEBUG, __FUNCTION__, __LINE__, "" __VA_ARGS__)

/**
 * Mochimo error number types
*/
enum mcm_errno_t {
   /* Force integer enum for compatibility with POSIX standard errno */
   EMCMFORCEINTEGER = -0x7fffffff - 1,
   /* initialize errors well above any existing POSIX errno */
   EMCMFIRST = 0x8000,

   /** Ledger entry debit did not match balance */
   EMCMLEDEBIT,
   /** Ledger entry credit overflowed the balance */
   EMCMLECREDITOVERFLOW,
   /** Maximum ledger depth reached */
   EMCMLEDEPTH,
   /** Internal Ledger is not available */
   EMCMLENOTAVAIL,
   /** Unknown ledger entry transaction code */
   EMCMLETRANCODE,

   /** No transactions to handle */
   EMCMNOTXS,

   /** Unhandled operation code */
   EMCMOPCODE,
   /** Missing OP_HELLO packet */
   EMCMOPHELLO,
   /** Missing OP_HELLO_ACK packet */
   EMCMOPHELLOACK,
   /** Invalid operation code */
   EMCMOPNVAL,
   /** Received unexpected operation code */
   EMCMOPRECV,
   /** Invalid CRC16 packet hash */
   EMCMPKTCRC,
   /** Unexpected packet identification */
   EMCMPKTIDS,
   /** Unexpected negative acknowledgement */
   EMCMPKTNACK,
   /** Incompatible packet network */
   EMCMPKTNET,
   /** Invalid packet opcode */
   EMCMPKTOPCODE,
   /** Invalid packet trailer */
   EMCMPKTTLR,

   /** Change address is not in Ledger */
   EMCMTXCHGEXISTS,
   /** Change address is not in Ledger */
   EMCMTXCHGNOLE,
   /** Change address is not Tagged */
   EMCMTXCHGNOTAG,
   /** Destination address is not in Ledger */
   EMCMTXDSTNOLE,
   /** Destination address is not Tagged */
   EMCMTXDSTNOTAG,
   /** Duplicate transaction ID */
   EMCMTXDUP,
   /** Fee is invalid */
   EMCMTXFEE,
   /** Overflow of transaction feees */
   EMCMTXFEEOVERFLOW,
   /** Bad transaction ID */
   EMCMTXID,
   /** Overflow of transaction amounts */
   EMCMTXOVERFLOW,
   /** Bad transaction sort */
   EMCMTXSORT,
   /** Source address is change address */
   EMCMTXSRCISCHG,
   /** Source address is destination address */
   EMCMTXSRCISDST,
   /** Source address is not in Ledger */
   EMCMTXSRCNOLE,
   /** Source address is not Tagged */
   EMCMTXSRCNOTAG,
   /** Invalid Tag activation (change address already exists) */
   EMCMTXTAGCHG,
   /** Invalid Tag activation (source address is tagged) */
   EMCMTXTAGSRC,
   /** Transaction total does not match ledger balance */
   EMCMTXTOTAL,
   /** WOTS+ signature invalid */
   EMCMTXWOTS,

   /** eXtended TX change total is less than fee */
   EMCMXTXCHGTOTAL,
   /** eXtended TX destination amount is zero */
   EMCMXTXDSTAMOUNT,
   /** eXtended TX fee does not match tally */
   EMCMXTXFEES,
   /** eXtended TX MEMO contains punctuation character */
   EMCMXTXHASPUNCT,
   /** eXtended TX MEMO contains non-printable character */
   EMCMXTXNONPRINT,
   /** eXtended TX MEMO is missing a null terminator */
   EMCMXTXNOTERM,
   /** eXtended TX contains non-zero trailing padding */
   EMCMXTXNZTPADDING,
   /** eXtended TX send total is zero */
   EMCMXTXSENDTOTAL,
   /** eXtended TX destination tag matches source tag */
   EMCMXTXTAGMATCH,
   /** eXtended TX source tag does not match change tag */
   EMCMXTXTAGMISMATCH,
   /** eXtended TX destination tag is not in Ledger */
   EMCMXTXTAGNOLE,
   /** eXtended TX total does not match tally */
   EMCMXTXTOTALS,
   /** eXtended TX type is not defined */
   EMCMXTXNODEF,

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
};

/* C/C++ compatible function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

int argument(char *argv, char *chk1, char *chk2);
char *argvalue(int *idx, int argc, char *argv[]);
int asnprintf(char *buf, size_t bufsz, const char *fmt, ...);
char *bnum2hex(void *bnum, char *hex);
char *bnum2hex64(void *bnum, char *hex);
double diffclocktime(clock_t prev);
char *hash2hex(void *hash, int count, char *hex);
char *metric_reduce(double *value);
char *op2str(unsigned op);
char *ve2str(int ve);
char *weight2hex(void *weight, char *hex);
int path_count_join(char *buf, int count, ...);
const char *mcm_errno_text(int errnum);
char *mcm_strerror(int errnum, char *buf, size_t bufsz);
unsigned int plogcount(void);
void plogx(int ll, const char *func, int line, const char *fmt, ...);
void setploglevel(int ll);
void setplogtime(int val);

#ifdef __cplusplus
}  /* end extern "C" */
#endif

/* end include guard */
#endif
