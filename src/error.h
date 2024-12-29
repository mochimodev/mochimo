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
#include "extint.h"
#include "extio.h"
#include <time.h>

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

/* Mochimo declaration MACROs; compiler specific */
#if defined(__GNUC__) || defined(__clang__)
   #define MCM_DECL_ALIGNED(X) __attribute__((aligned(X)))
   #define MCM_DECL_DEPRECATED __attribute__((deprecated))
   #define MCM_DECL_UNUSED __attribute__((unused))
#elif defined(_MSC_VER)
   #define MCM_DECL_ALIGNED(X) __declspec(align(X))
   #define MCM_DECL_DEPRECATED __declspec(deprecated)
   #define MCM_DECL_UNUSED
#else
   #define MCM_DECL_ALIGNED(X)
   #define MCM_DECL_DEPRECATED
   #define MCM_DECL_UNUSED
#endif

/* STATIC ASSERTION MACRO, for compile time assertion. */
#define STATIC_ASSERT(EXPR, MSG) MCM_DECL_UNUSED static char \
   STATIC_ASSERTION_FAILURE__##MSG[(2*(!!(EXPR)))-1]

/* print log levels */
#define PLOG_ALERT 0
#define PLOG_ERRNO 1
#define PLOG_ERROR 2
#define PLOG_WARN  3
#define PLOG_INFO  4
#define PLOG_DEBUG 5

/**
 * Write a file path into a buffer by joining multiple strings together.
 * Assumes path is a buffer of at least FILENAME_MAX bytes in length.
 * Uses "\\" to separate paths in Windows, otherwise uses "/".
 * @param path Pointer to a buffer to write to
 * @param ... Strings to join together
*/
#define path_join(path, ...) \
   path_join_count(path, VA_COUNT(__VA_ARGS__), __VA_ARGS__)

/**
 * Print an alert level log.
 * @param ... arguments you would normally pass to printf()
*/
#define palert(...) \
   plogx(PLOG_ALERT, __FILE__, __LINE__, __VA_ARGS__)

/**
 * Print an error level log, with description of @a errnum.
 * @param ... arguments you would normally pass to printf()
*/
#define perrno(...) \
   plogx(PLOG_ERRNO, __FILE__, __LINE__, __VA_ARGS__)

/**
 * Print an error level log.
 * @param ... arguments you would normally pass to printf()
*/
#define perr(...) \
   plogx(PLOG_ERROR, __FILE__, __LINE__, __VA_ARGS__)

/**
 * Print a warning level log.
 * @param ... arguments you would normally pass to printf()
*/
#define pwarn(...) \
   plogx(PLOG_WARN, __FILE__, __LINE__, __VA_ARGS__)

/**
 * Print an information level log.
 * @param ... arguments you would normally pass to printf()
*/
#define plog(...) \
   plogx(PLOG_INFO, __FILE__, __LINE__, __VA_ARGS__)

#ifndef NDEBUG
   /**
    * Print a debugging level log.
    * @param ... arguments you would normally pass to printf()
   */
   #define pdebug(...) \
      plogx(PLOG_DEBUG, __FILE__, __LINE__, __VA_ARGS__)

#else
   /* To avoid (potential) compiler warnings about unused variables,
    * we pass the variables into a dummy function to immitate "use".
    * The compiler will (see "should") remove the erroneous code. */
   static void voidify(const char *_, ...) { (void)_; }
   #define pdebug(...) voidify(NULL, __VA_ARGS__)

#endif

/* Table of Mochimo error codes (names) and descriptions (documented).
 * Although initially, a dirty solution to a debatable problem, the use of
 * this MACRO has since evolved and seems a fitting way to manage multiple
 * related lookup tables of the same data, in a single location.
 */

#define EMCM__TABLE(EMCM__ITEM) \
/* math errors... */ \
   EMCM__ITEM(EMCM_MATH64_OVERFLOW, "Unspecified 64-bit math overflow") \
   EMCM__ITEM(EMCM_MATH64_UNDERFLOW, "Unspecified 64-bit math underflow") \
/* file related errors... */ \
   EMCM__ITEM(EMCM_EOF, "Unexpected end-of-file") \
   EMCM__ITEM(EMCM_FILECOUNT, "Unexpected number of items in file") \
   EMCM__ITEM(EMCM_FILEDATA, "Unexpected file data") \
   EMCM__ITEM(EMCM_FILELEN, "Unexpected length of file") \
   EMCM__ITEM(EMCM_SORTLEN, "Unexpected file length during sort") \
/* block related errors... */ \
   EMCM__ITEM(EMCM_BHASH, "Bad block hash") \
   EMCM__ITEM(EMCM_BNUM, "Bad block number") \
   EMCM__ITEM(EMCM_DIFF, "Bad difficulty") \
   EMCM__ITEM(EMCM_GENHASH, "Bad Genesis hash") \
   EMCM__ITEM(EMCM_HDRLEN, "Bad header length") \
   EMCM__ITEM(EMCM_MADDR, "Bad miner address") \
   EMCM__ITEM(EMCM_MFEE, "Bad miner fee") \
   EMCM__ITEM(EMCM_MFEES_OVERFLOW, "Overflow of miner fees") \
   EMCM__ITEM(EMCM_MREWARD, "Bad miner reward") \
   EMCM__ITEM(EMCM_MREWARDS_OVERFLOW, "Overflow of miner rewards") \
   EMCM__ITEM(EMCM_MROOT, "Bad merkle root") \
   EMCM__ITEM(EMCM_NONCE, "Bad nonce") \
   EMCM__ITEM(EMCM_NZGEN, "Non-zero Genesis data") \
   EMCM__ITEM(EMCM_PHASH, "Bad (previous) block hash") \
   EMCM__ITEM(EMCM_PTIME, "Bad TOT time") \
   EMCM__ITEM(EMCM_STIME, "Bad solve time") \
   EMCM__ITEM(EMCM_TCOUNT, "Bad TX count") \
   EMCM__ITEM(EMCM_TIME0, "Bad start time") \
   EMCM__ITEM(EMCM_TLRLEN, "Bad trailer length") \
   EMCM__ITEM(EMCM_TMAX, "Too many transactions") \
   EMCM__ITEM(EMCM_TRAILER, "Bad trailer data") \
/* ledger entry related errors... */ \
   EMCM__ITEM(EMCM_LEOVERFLOW, "Overflow of ledger amounts") \
   EMCM__ITEM(EMCM_LECLOSED, "Ledger operation attempted while ledger is closed") \
   EMCM__ITEM(EMCM_LEEMPTY, "No records written to ledger file") \
   EMCM__ITEM(EMCM_LEEXTRACT, "Ledger cannot be extracted from a non-NG block") \
   EMCM__ITEM(EMCM_LESORT, "Bad ledger sort") \
   EMCM__ITEM(EMCM_LESUM, "Bad sum of ledger amounts") \
   EMCM__ITEM(EMCM_LETAG, "Bad tag reference to ledger entry") \
/* ledger transaction related errors... */ \
   EMCM__ITEM(EMCM_LTCODE, "Bad ledger transaction code") \
   EMCM__ITEM(EMCM_LTCREDIT, "Unexpected ledger transaction code for ledger entry creation") \
   EMCM__ITEM(EMCM_LTDEBIT, "Ledger transaction debit, does not match ledger entry balance") \
   EMCM__ITEM(EMCM_LTSORT, "Bad ledger transactions sort") \
/* network related errors... */ \
   EMCM__ITEM(EMCM_OPCODE, "Unhandled operation code") \
   EMCM__ITEM(EMCM_OPHELLO, "Missing OP_HELLO packet") \
   EMCM__ITEM(EMCM_OPHELLOACK, "Missing OP_HELLO_ACK packet") \
   EMCM__ITEM(EMCM_OPNVAL, "Invalid operation code") \
   EMCM__ITEM(EMCM_OPRECV, "Received unexpected operation code") \
   EMCM__ITEM(EMCM_PKTCRC, "Invalid CRC16 packet hash") \
   EMCM__ITEM(EMCM_PKTIDS, "Unexpected packet identification") \
   EMCM__ITEM(EMCM_PKTNACK, "Unexpected negative acknowledgement") \
   EMCM__ITEM(EMCM_PKTNET, "Incompatible packet network") \
   EMCM__ITEM(EMCM_PKTOPCODE, "Invalid packet opcode") \
   EMCM__ITEM(EMCM_PKTTLR, "Invalid packet trailer") \
/* POW related errors... */ \
   EMCM__ITEM(EMCM_POWTRIGG, "Bad PoW (Trigg)") \
   EMCM__ITEM(EMCM_POWPEACH, "Bad PoW (Peach)") \
   EMCM__ITEM(EMCM_POWANOMALY, "Bad PoW Anomaly (bugfix)") \
/* transaction related errors... */ \
   EMCM__ITEM(EMCM_TX0, "No transactions to handle") \
   EMCM__ITEM(EMCM_TXADRS, "Invalid address scheme data") \
   EMCM__ITEM(EMCM_TXBTL, "Transaction block-to-live out of range") \
   EMCM__ITEM(EMCM_TXCHGEXISTS, "Change address is not in Ledger") \
   EMCM__ITEM(EMCM_TXCHGNOLE, "Change address is not in Ledger") \
   EMCM__ITEM(EMCM_TXCHGNOTAG, "Change address is not Tagged") \
   EMCM__ITEM(EMCM_TXCHGTAGDUP, "Duplicate change address tag") \
   EMCM__ITEM(EMCM_TXDSTNOLE, "Destination address is not in Ledger") \
   EMCM__ITEM(EMCM_TXDSTNOTAG, "Destination address is not Tagged") \
   EMCM__ITEM(EMCM_TXFEE, "Fee is invalid") \
   EMCM__ITEM(EMCM_TXFEE_OVERFLOW, "Overflow of transaction feees") \
   EMCM__ITEM(EMCM_TXID, "Bad transaction ID") \
   EMCM__ITEM(EMCM_TXIDDUP, "Duplicate transaction ID") \
   EMCM__ITEM(EMCM_TXINVAL, "Invalid transaction") \
   EMCM__ITEM(EMCM_TXOVERFLOW, "Overflow of transaction amounts") \
   EMCM__ITEM(EMCM_TXSORT, "Bad transaction sort") \
   EMCM__ITEM(EMCM_TXCHG, "Source address is change address") \
   EMCM__ITEM(EMCM_TXDSA, "Invalid Digital Signature Algorithm") \
   EMCM__ITEM(EMCM_TXDST, "Source address is destination address") \
   EMCM__ITEM(EMCM_TXNONCE, "Invalid transaction nonce") \
   EMCM__ITEM(EMCM_TXSRCDUP, "Duplicate transaction source address") \
   EMCM__ITEM(EMCM_TXSRCLE, "Source address is not in Ledger") \
   EMCM__ITEM(EMCM_TXSRCNOTAG, "Source address is not Tagged") \
   EMCM__ITEM(EMCM_TXTAGCHG, "Invalid Tag activation (change address already exists)") \
   EMCM__ITEM(EMCM_TXTAGSRC, "Invalid Tag activation (source address is tagged)") \
   EMCM__ITEM(EMCM_TXTOTAL, "Transaction total does not match ledger balance") \
   EMCM__ITEM(EMCM_TXWOTS, "WOTS+ signature invalid") \
/* eXtended transaction related errors... */ \
   EMCM__ITEM(EMCM_XTXCHGTOTAL, "eXtended TX change total is less than fee") \
   EMCM__ITEM(EMCM_XTXDSTAMOUNT, "eXtended TX destination amount is zero") \
   EMCM__ITEM(EMCM_XTXFEES, "eXtended TX fee does not match tally") \
   EMCM__ITEM(EMCM_XTXREF, "Invalid reference format in eXtended Transaction") \
   EMCM__ITEM(EMCM_XTXSENDTOTAL, "eXtended TX send total is zero") \
   EMCM__ITEM(EMCM_XTXSRCNOTAG, "eXtended TX source is not tagged") \
   EMCM__ITEM(EMCM_XTXTAGMATCH, "eXtended TX destination tag matches source tag") \
   EMCM__ITEM(EMCM_XTXTAGMISMATCH, "eXtended TX source tag does not match change tag") \
   EMCM__ITEM(EMCM_XTXTAGNOLE, "eXtended TX destination tag is not in Ledger") \
   EMCM__ITEM(EMCM_XTXTOTALS, "eXtended TX total does not match tally") \
   EMCM__ITEM(EMCM_XTXUNDEF, "eXtended TX type is not defined")

/**
 * Mochimo error number type. This is a signed integer type (by force).
*/
enum mcm_errno_t {
   /* The intent of enum mcm_errno_t is interoperability with POSIX errno.
    *
    * C99 7.5/2; errno expands to a modifiable lvalue of type int.
    * C99 6.7.2.2/3-4; enum identifiers are declared constants of type int;
    * enum type is implementation-defined, but shall be capable of
    * representing the values of all the members of the enumeration.
    * "... ruh roh raggy ..."
    *
    * GCC seems to declare enum types as unsigned integers by default.
    * Along comes a wise guy... declares "enum mcm_error_t" variable...
    * encounters check for negative... and the compiler starts wailing.
    *
    * Therefore; force enum type int (signed) and call it a day.
    */
   EMCM__FORCEINT = -0x7fffffff - 1,
   /* initialize above any existing errno (IBM has some at 20k -_-) */
   EMCM__INIT = 0x6000,

   /* "EMCM__ENUM" is provided to "EMCM__TABLE" for extraction of Mochimo
    * error IDs and associated docs into the body of "enum mcm_errno_t".
    * Doxygen (our intended target) does not interpret comments as
    * whitespace during MACRO expansion.
    */
#define EMCM__ENUM(ID, DESC) /** DESC */ ID,
   EMCM__TABLE(EMCM__ENUM)
};

/* C/C++ compatible function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

int argument(char *argv, char *chk1, char *chk2);
char *argvalue(int *idx, int argc, char *argv[]);
char *bnum2fname(word8 bnum[8], char fname[21]);
char *bnum2hex(word8 bnum[8], char hex[17]);
char *bnum2hex64(word8 bnum[8], char hex[17]);
double diffclocktime(clock_t prev);
char *hash2hex32(word8 hash[4], char hex[9]);
char *metric_reduce(double *value);
char *op2str(unsigned op);
char *ve2str(int ve);
char *path_join_count(char path[FILENAME_MAX], int count, ...);
char *mcm_strerror(int errnum, char *buf, size_t bufsz);
char *mcm_strerrorname(int errnum, char *buf, size_t bufsz);
unsigned int perrcount(void);
unsigned int plogcount(void);
void plogx(int ll, const char *func, int line, const char *fmt, ...);
void setplogfile(FILE *fp);
void setploglevel(int ll);
void setplogtime(int val);
char *weight2hex(word8 weight[32], char hex[65]);

#ifdef __cplusplus
}  /* end extern "C" */
#endif

/* end include guard */
#endif
