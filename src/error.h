/**
 * @file error.h
 * @brief Mochimo error codes, logging and associated support.
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

/* print log levels */
#define PLOG_ALERT 0
#define PLOG_ERRNO 1
#define PLOG_ERROR 2
#define PLOG_WARN  3
#define PLOG_INFO  4
#define PLOG_DEBUG 5

/**
 * Write a file path into a buffer by joining multiple strings together
 * with the PATH_SEP.
 * @param buf Pointer to a buffer to write to
 * @param bufsz Size of buffer to write to, in bytes
 * @param ... Strings to join together
*/
#define path_join(buf, bufsz, ...) \
   path_join_count(buf, bufsz, VA_COUNT(__VA_ARGS__), __VA_ARGS__)

/**
 * Print an alert level log.
 * @param ... arguments you would normally pass to printf()
*/
#define palert(...) \
   plogx(PLOG_ALERT, __FUNCTION__, __LINE__, __VA_ARGS__)

/**
 * Print an error level log, with description of @a errnum.
 * @param ... arguments you would normally pass to printf()
*/
#define perrno(...) \
   plogx(PLOG_ERRNO, __FUNCTION__, __LINE__, __VA_ARGS__)

/**
 * Print an error level log.
 * @param ... arguments you would normally pass to printf()
*/
#define perr(...) \
   plogx(PLOG_ERROR, __FUNCTION__, __LINE__, __VA_ARGS__)

/**
 * Print a warning level log.
 * @param ... arguments you would normally pass to printf()
*/
#define pwarn(...) \
   plogx(PLOG_WARN, __FUNCTION__, __LINE__, __VA_ARGS__)

/**
 * Print an information level log.
 * @param ... arguments you would normally pass to printf()
*/
#define plog(...) \
   plogx(PLOG_INFO, __FUNCTION__, __LINE__, __VA_ARGS__)

#ifndef NDEBUG
   /**
    * Print a debugging level log.
    * @param ... arguments you would normally pass to printf()
   */
   #define pdebug(...) \
      plogx(PLOG_DEBUG, __FUNCTION__, __LINE__, __VA_ARGS__)

#else
   /* To avoid (potential) compiler warnings about unused variables,
    * we pass the variables into a dummy function to immitate "use".
    * The compiler will (see "should") remove the erroneous code. */
   static void voidify(const char *_, ...) { (void)_; }
   #define pdebug(...) voidify(NULL, __VA_ARGS__)

#endif

/**
 * Mochimo error number types
*/
enum mcm_errno_t {
   /* Force integer enum for compatibility with POSIX standard errno */
   EMCMFORCEINTEGER = -0x7fffffff - 1,
   /* initialize errors well above any existing POSIX errno */
   EMCMFIRST = 0x8000,

   /* core relateed errors... */
   /** Unspecified 64-bit math overflow */
   EMCM_MATH64_OVERFLOW,
   /** Unspecified 64-bit math underflow */
   EMCM_MATH64_UNDERFLOW,

   /* file related errors... */
   /** Unexpected end-of-file */
   EMCM_EOF,
   /** Unexpected number of items in file */
   EMCM_FILECOUNT,
   /** Unexpected file data */
   EMCM_FILEDATA,
   /** Unexpected length of file */
   EMCM_FILELEN,
   /** Unexpected file length during sort */
   EMCM_SORTLEN,

   /* block related errors... */
   /** Bad block hash */
   EMCM_BHASH,
   /** Bad block number */
   EMCM_BNUM,
   /** Bad difficulty */
   EMCM_DIFF,
   /** Bad Genesis hash */
   EMCM_GENHASH,
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
   /** Non-zero Genesis data */
   EMCM_NZGEN,
   /** Bad (previous) block hash */
   EMCM_PHASH,
   /** Bad TOT time */
   EMCM_PTIME,
   /** Bad solve time */
   EMCM_STIME,
   /** Bad TX count */
   EMCM_TCOUNT,
   /** Bad start time */
   EMCM_TIME0,
   /** Bad trailer length */
   EMCM_TLRLEN,
   /** Too many transactions */
   EMCM_TMAX,
   /** Bad trailer data */
   EMCM_TRAILER,

   /* ledger entry related errors... */
   /** Overflow of ledger amounts */
   EMCM_LEOVERFLOW,
   /** No records written to ledger file */
   EMCM_LEEMPTY,
   /** Ledger cannot be extracted from a non-NG block */
   EMCM_LEEXTRACT,
   /** Bad ledger sort */
   EMCM_LESORT,
   /** Bad sum of ledger amounts */
   EMCM_LESUM,
   /** Bad tag reference to ledger entry */
   EMCM_LETAG,

   /* ledger transaction related errors... */
   /** Bad ledger transaction code */
   EMCM_LTCODE,
   /** Unexpected ledger transaction code for ledger entry creation */
   EMCM_LTCREDIT,
   /** Ledger transaction debit, does not match ledger entry balance */
   EMCM_LTDEBIT,
   /** Bad ledger transactions sort */
   EMCM_LTSORT,

   /* network related errors... */
   /** Unhandled operation code */
   EMCM_OPCODE,
   /** Missing OP_HELLO packet */
   EMCM_OPHELLO,
   /** Missing OP_HELLO_ACK packet */
   EMCM_OPHELLOACK,
   /** Invalid operation code */
   EMCM_OPNVAL,
   /** Received unexpected operation code */
   EMCM_OPRECV,
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

   /* POW related errors... */
   /** Bad PoW (Trigg) */
   EMCM_POWTRIGG,
   /** Bad PoW (Peach) */
   EMCM_POWPEACH,
   /** Bad PoW Anomaly (bugfix) */
   EMCM_POWANOMALY,

   /* transaction related errors... */
   /** No transactions to handle */
   EMCM_TX0,
   /** Change address is not in Ledger */
   EMCM_TXCHGEXISTS,
   /** Change address is not in Ledger */
   EMCM_TXCHGNOLE,
   /** Change address is not Tagged */
   EMCM_TXCHGNOTAG,
   /** Destination address is not in Ledger */
   EMCM_TXDSTNOLE,
   /** Destination address is not Tagged */
   EMCM_TXDSTNOTAG,
   /** Duplicate transaction ID */
   EMCM_TXDUP,
   /** Fee is invalid */
   EMCM_TXFEE,
   /** Overflow of transaction feees */
   EMCM_TXFEE_OVERFLOW,
   /** Bad transaction ID */
   EMCM_TXID,
   /** Overflow of transaction amounts */
   EMCM_TXOVERFLOW,
   /** Bad transaction sort */
   EMCM_TXSORT,
   /** Source address is change address */
   EMCM_TXCHG,
   /** Source address is destination address */
   EMCM_TXDST,
   /** Source address is not in Ledger */
   EMCM_TXSRCLE,
   /** Source address is not Tagged */
   EMCM_TXSRCNOTAG,
   /** Invalid Tag activation (change address already exists) */
   EMCM_TXTAGCHG,
   /** Invalid Tag activation (source address is tagged) */
   EMCM_TXTAGSRC,
   /** Transaction total does not match ledger balance */
   EMCM_TXTOTAL,
   /** WOTS+ signature invalid */
   EMCM_TXWOTS,

   /* eXtended transaction related errors... */
   /** eXtended TX change total is less than fee */
   EMCM_XTXCHGTOTAL,
   /** eXtended TX destination amount is zero */
   EMCM_XTXDSTAMOUNT,
   /** eXtended TX fee does not match tally */
   EMCM_XTXFEES,
   /** eXtended TX MEMO contains punctuation character */
   EMCM_XTXHASPUNCT,
   /** eXtended TX MEMO contains non-printable character */
   EMCM_XTXNONPRINT,
   /** eXtended TX MEMO is missing a null terminator */
   EMCM_XTXNOTERM,
   /** eXtended TX contains non-zero trailing padding */
   EMCM_XTXNZTPADDING,
   /** eXtended TX send total is zero */
   EMCM_XTXSENDTOTAL,
   /** eXtended TX destination tag matches source tag */
   EMCM_XTXTAGMATCH,
   /** eXtended TX source tag does not match change tag */
   EMCM_XTXTAGMISMATCH,
   /** eXtended TX destination tag is not in Ledger */
   EMCM_XTXTAGNOLE,
   /** eXtended TX total does not match tally */
   EMCM_XTXTOTALS,
   /** eXtended TX type is not defined */
   EMCM_XTXUNDEF,
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
char *path_join_count(char *buf, size_t bufsz, int count, ...);
char *mcm_strerror(int errnum, char *buf, size_t bufsz);
unsigned int plogcount(void);
void plogx(int ll, const char *func, int line, const char *fmt, ...);
void setplogfunctions(int val);
void setploglevel(int ll);
void setplogtime(int val);
char *sprintbnum(char *buffer, const char *dir, void *bnum);

#ifdef __cplusplus
}  /* end extern "C" */
#endif

/* end include guard */
#endif
