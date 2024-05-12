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

/* Table of Mochimo error descriptions, and (documented) names.
 * NOTE: Our insanity is directly proportional to the number of errors and
 * associated trivial information we must implement raised to the power of
 * the number of places we have to repeat such information in our codebase.
 * These MACROs serve as arguably unpleasant but necessary boilerplate to
 * avoid such a demise of sanity. C Gods, please forgive me...
 */

#define EMCM__TABLE(fn) \
/* math errors... */ \
   fn("Unspecified 64-bit math overflow", \
   /** Unspecified 64-bit math overflow */ \
      EMCM_MATH64_OVERFLOW) \
   fn("Unspecified 64-bit math underflow", \
   /** Unspecified 64-bit math underflow */ \
      EMCM_MATH64_UNDERFLOW) \
/* file related errors... */ \
   fn("Unexpected end-of-file", \
   /** Unexpected end-of-file */ \
      EMCM_EOF) \
   fn("Unexpected number of items in file", \
   /** Unexpected number of items in file */ \
      EMCM_FILECOUNT) \
   fn("Unexpected file data", \
   /** Unexpected file data */ \
      EMCM_FILEDATA) \
   fn("Unexpected length of file", \
   /** Unexpected length of file */ \
      EMCM_FILELEN) \
   fn("Unexpected file length during sort", \
   /** Unexpected file length during sort */ \
      EMCM_SORTLEN) \
/* block related errors... */ \
   fn("Bad block hash", \
   /** Bad block hash */ \
      EMCM_BHASH) \
   fn("Bad block number", \
   /** Bad block number */ \
      EMCM_BNUM) \
   fn("Bad difficulty", \
   /** Bad difficulty */ \
      EMCM_DIFF) \
   fn("Bad Genesis hash", \
   /** Bad Genesis hash */ \
      EMCM_GENHASH) \
   fn("Bad header length", \
   /** Bad header length */ \
      EMCM_HDRLEN) \
   fn("Bad miner address", \
   /** Bad miner address */ \
      EMCM_MADDR) \
   fn("Bad miner fee", \
   /** Bad miner fee */ \
      EMCM_MFEE) \
   fn("Overflow of miner fees", \
   /** Overflow of miner fees */ \
      EMCM_MFEES_OVERFLOW) \
   fn("Bad miner reward", \
   /** Bad miner reward */ \
      EMCM_MREWARD) \
   fn("Overflow of miner rewards", \
   /** Overflow of miner rewards */ \
      EMCM_MREWARDS_OVERFLOW) \
   fn("Bad merkle root", \
   /** Bad merkle root */ \
      EMCM_MROOT) \
   fn("Bad nonce", \
   /** Bad nonce */ \
      EMCM_NONCE) \
   fn("Non-zero Genesis data", \
   /** Non-zero Genesis data */ \
      EMCM_NZGEN) \
   fn("Bad (previous) block hash", \
   /** Bad (previous) block hash */ \
      EMCM_PHASH) \
   fn("Bad TOT time", \
   /** Bad TOT time */ \
      EMCM_PTIME) \
   fn("Bad solve time", \
   /** Bad solve time */ \
      EMCM_STIME) \
   fn("Bad TX count", \
   /** Bad TX count */ \
      EMCM_TCOUNT) \
   fn("Bad start time", \
   /** Bad start time */ \
      EMCM_TIME0) \
   fn("Bad trailer length", \
   /** Bad trailer length */ \
      EMCM_TLRLEN) \
   fn("Too many transactions", \
   /** Too many transactions */ \
      EMCM_TMAX) \
   fn("Bad trailer data", \
   /** Bad trailer data */ \
      EMCM_TRAILER) \
/* ledger entry related errors... */ \
   fn("Overflow of ledger amounts", \
   /** Overflow of ledger amounts */ \
      EMCM_LEOVERFLOW) \
   fn("Ledger operation attempted while ledger is closed", \
   /** Ledger operation attempted while ledger is closed */ \
      EMCM_LECLOSED) \
   fn("No records written to ledger file", \
   /** No records written to ledger file */ \
      EMCM_LEEMPTY) \
   fn("Ledger cannot be extracted from a non-NG block", \
   /** Ledger cannot be extracted from a non-NG block */ \
      EMCM_LEEXTRACT) \
   fn("Bad ledger sort", \
   /** Bad ledger sort */ \
      EMCM_LESORT) \
   fn("Bad sum of ledger amounts", \
   /** Bad sum of ledger amounts */ \
      EMCM_LESUM) \
   fn("Bad tag reference to ledger entry", \
   /** Bad tag reference to ledger entry */ \
      EMCM_LETAG) \
/* ledger transaction related errors... */ \
   fn("Bad ledger transaction code", \
   /** Bad ledger transaction code */ \
      EMCM_LTCODE) \
   fn("Unexpected ledger transaction code for ledger entry creation", \
   /** Unexpected ledger transaction code for ledger entry creation */ \
      EMCM_LTCREDIT) \
   fn("Ledger transaction debit, does not match ledger entry balance", \
   /** Ledger transaction debit, does not match ledger entry balance */ \
      EMCM_LTDEBIT) \
   fn("Bad ledger transactions sort", \
   /** Bad ledger transactions sort */ \
      EMCM_LTSORT) \
/* network related errors... */ \
   fn("Unhandled operation code", \
   /** Unhandled operation code */ \
      EMCM_OPCODE) \
   fn("Missing OP_HELLO packet", \
   /** Missing OP_HELLO packet */ \
      EMCM_OPHELLO) \
   fn("Missing OP_HELLO_ACK packet", \
   /** Missing OP_HELLO_ACK packet */ \
      EMCM_OPHELLOACK) \
   fn("Invalid operation code", \
   /** Invalid operation code */ \
      EMCM_OPNVAL) \
   fn("Received unexpected operation code", \
   /** Received unexpected operation code */ \
      EMCM_OPRECV) \
   fn("Invalid CRC16 packet hash", \
   /** Invalid CRC16 packet hash */ \
      EMCM_PKTCRC) \
   fn("Unexpected packet identification", \
   /** Unexpected packet identification */ \
      EMCM_PKTIDS) \
   fn("Unexpected negative acknowledgement", \
   /** Unexpected negative acknowledgement */ \
      EMCM_PKTNACK) \
   fn("Incompatible packet network", \
   /** Incompatible packet network */ \
      EMCM_PKTNET) \
   fn("Invalid packet opcode", \
   /** Invalid packet opcode */ \
      EMCM_PKTOPCODE) \
   fn("Invalid packet trailer", \
   /** Invalid packet trailer */ \
      EMCM_PKTTLR) \
/* POW related errors... */ \
   fn("Bad PoW (Trigg)", \
   /** Bad PoW (Trigg) */ \
      EMCM_POWTRIGG) \
   fn("Bad PoW (Peach)", \
   /** Bad PoW (Peach) */ \
      EMCM_POWPEACH) \
   fn("Bad PoW Anomaly (bugfix)", \
   /** Bad PoW Anomaly (bugfix) */ \
      EMCM_POWANOMALY) \
/* transaction related errors... */ \
   fn("No transactions to handle", \
   /** No transactions to handle */ \
      EMCM_TX0) \
   fn("Invalid address scheme data", \
   /** Invalid address scheme data */ \
      EMCM_TXADRS) \
   fn("Transaction block-to-live out of range", \
   /** Transaction block-to-live out of range */ \
      EMCM_TXBTL) \
   fn("Change address is not in Ledger", \
   /** Change address is not in Ledger */ \
      EMCM_TXCHGEXISTS) \
   fn("Change address is not in Ledger", \
   /** Change address is not in Ledger */ \
      EMCM_TXCHGNOLE) \
   fn("Change address is not Tagged", \
   /** Change address is not Tagged */ \
      EMCM_TXCHGNOTAG) \
   fn("Duplicate change address tag", \
   /** Duplicate change address tag */ \
      EMCM_TXCHGTAGDUP) \
   fn("Destination address is not in Ledger", \
   /** Destination address is not in Ledger */ \
      EMCM_TXDSTNOLE) \
   fn("Destination address is not Tagged", \
   /** Destination address is not Tagged */ \
      EMCM_TXDSTNOTAG) \
   fn("Fee is invalid", \
   /** Fee is invalid */ \
      EMCM_TXFEE) \
   fn("Overflow of transaction feees", \
   /** Overflow of transaction feees */ \
      EMCM_TXFEE_OVERFLOW) \
   fn("Bad transaction ID", \
   /** Bad transaction ID */ \
      EMCM_TXID) \
   fn("Duplicate transaction ID", \
   /** Duplicate transaction ID */ \
      EMCM_TXIDDUP) \
   fn("nvalid transaction", \
   /* Invalid transaction */ \
      EMCM_TXINVAL) \
   fn("Overflow of transaction amounts", \
   /** Overflow of transaction amounts */ \
      EMCM_TXOVERFLOW) \
   fn("Bad transaction sort", \
   /** Bad transaction sort */ \
      EMCM_TXSORT) \
   fn("Source address is change address", \
   /** Source address is change address */ \
      EMCM_TXCHG) \
   fn("Invalid Digital Signature Algorithm", \
   /** Invalid Digital Signature Algorithm */ \
      EMCM_TXDSA) \
   fn("Source address is destination address", \
   /** Source address is destination address */ \
      EMCM_TXDST) \
   fn("Invalid transaction nonce", \
   /** Invalid transaction nonce */ \
      EMCM_TXNONCE) \
   fn("Duplicate transaction source address", \
   /** Duplicate transaction source address */ \
      EMCM_TXSRCDUP) \
   fn("Source address is not in Ledger", \
   /** Source address is not in Ledger */ \
      EMCM_TXSRCLE) \
   fn("Source address is not Tagged", \
   /** Source address is not Tagged */ \
      EMCM_TXSRCNOTAG) \
   fn("Invalid Tag activation (change address already exists)", \
   /** Invalid Tag activation (change address already exists) */ \
      EMCM_TXTAGCHG) \
   fn("Invalid Tag activation (source address is tagged)", \
   /** Invalid Tag activation (source address is tagged) */ \
      EMCM_TXTAGSRC) \
   fn("Transaction total does not match ledger balance", \
   /** Transaction total does not match ledger balance */ \
      EMCM_TXTOTAL) \
   fn("WOTS+ signature invalid", \
   /** WOTS+ signature invalid */ \
      EMCM_TXWOTS) \
/* eXtended transaction related errors... */ \
   fn("eXtended TX change total is less than fee", \
   /** eXtended TX change total is less than fee */ \
      EMCM_XTXCHGTOTAL) \
   fn("eXtended TX destination amount is zero", \
   /** eXtended TX destination amount is zero */ \
      EMCM_XTXDSTAMOUNT) \
   fn("eXtended TX fee does not match tally", \
   /** eXtended TX fee does not match tally */ \
      EMCM_XTXFEES) \
   fn("Invalid reference format in eXtended Transaction", \
   /** Invalid reference format in eXtended Transaction */ \
      EMCM_XTXREF) \
   fn("eXtended TX send total is zero", \
   /** eXtended TX send total is zero */ \
      EMCM_XTXSENDTOTAL) \
   fn("eXtended TX source is not tagged", \
   /** eXtended TX source is not tagged */ \
      EMCM_XTXSRCNOTAG) \
   fn("eXtended TX destination tag matches source tag", \
   /** eXtended TX destination tag matches source tag */ \
      EMCM_XTXTAGMATCH) \
   fn("eXtended TX source tag does not match change tag", \
   /** eXtended TX source tag does not match change tag */ \
      EMCM_XTXTAGMISMATCH) \
   fn("eXtended TX destination tag is not in Ledger", \
   /** eXtended TX destination tag is not in Ledger */ \
      EMCM_XTXTAGNOLE) \
   fn("eXtended TX total does not match tally", \
   /** eXtended TX total does not match tally */ \
      EMCM_XTXTOTALS) \
   fn("eXtended TX type is not defined", \
   /** eXtended TX type is not defined */ \
      EMCM_XTXUNDEF)

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
#define EMCM__ENUM(_, ID) ID,
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
