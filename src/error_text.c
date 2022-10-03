/**
 * @private
 * @headerfile error.h <error.h>
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef EXTENDED_ERROR_TEXT_C
#define EXTENDED_ERROR_TEXT_C


#include "error.h"

/**
 * Get a textual description of an error code.
 * The error code may be an error of the Mochimo API, or the C API.
 * @param errnum Value of the error number to get description for
 * @return char * -- textual description of error, or "Unknown error..."
 * @note Why not use strerror_r()? Because the GNU-specific version
 * "may or may not" fill the provided buffer with an error string,
 * so you have to use the returned (char *), but the XSI-compliant
 * version of strerror_r() and Windows' version (strerror_s()) don't
 * return a (char *), and I just don't need this kind of uncertain
 * hormonal behaviour from a logical entity.
 */
const char *errno_text(int errnum)
{
   switch (errnum) {

/* BEGIN MOCHIMO ERRORS */

   case EMCM_MATH64_OVERFLOW:
      return "Unspecified 64-bit math overflow";
   case EMCM_MATH64_UNDERFLOW:
      return "Unspecified 64-bit math underflow";
   case EMCM_SORT_LENGTH:
      return "Unexpected file length during sort";
   case EMCM_EOF:
      return "Unexpected end-of-file";
   case EMCM_FILELEN:
      return "Bad file length";
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
      return "Bad ledger transaction code";
   case EMCM_LT_DEBIT:
      return "Ledger transaction debit, does not match ledger entry balance";
   case EMCM_LT_NOT_CREDIT:
      return "Unexpected ledger transaction code for ledger entry creation";
   case EMCM_LT_SORT:
      return "Bad ledger transactions sort";
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
   case EMCM_NOHELLO:
      return "Missing OP_HELLO packet";
   case EMCM_NOHELLOACK:
      return "Missing OP_HELLO_ACK packet";
   case EMCM_OPCODE:
      return "Unrecognised operation code";
   case EMCM_OPINVAL:
      return "Invalid operation code";
   case EMCM_PKTCRC:
      return "Invalid CRC16 packet hash";
   case EMCM_PKTIDS:
      return "Unexpected packet identification";
   case EMCM_PKTNACK:
      return "Unexpected negative acknowledgement";
   case EMCM_PKTNET:
      return "Incompatible packet network";
   case EMCM_PKTOPCODE:
      return "Invalid packet opcode";
   case EMCM_PKTTLR:
      return "Invalid packet trailer";
   case EMCM_TX_AMOUNTS_OVERFLOW:
      return "Overflow of TX amounts";
   case EMCM_TX_CHG_ADDR:
      return "Bad TX change address";
   case EMCM_TX_CHG_TAG:
      return "Bad TX change tag";
   case EMCM_TX_DST_ADDR:
      return "Bad TX destination address";
   case EMCM_TX_DST_TAG:
      return "Bad TX destination tag";
   case EMCM_TX_DUP:
      return "Duplicate TX ID";
   case EMCM_TX_FEE:
      return "Bad TX fee";
   case EMCM_TX_ID:
      return "Bad TX ID";
   case EMCM_TX_SIG:
      return "Bad TX signature";
   case EMCM_TX_SORT:
      return "Bad TX sort";
   case EMCM_TX_SRC_ADDR:
      return "Bad TX source address";
   case EMCM_TX_SRC_LE_BALANCE:
      return "Bad TX amounts, not equal to src ledger balance";
   case EMCM_TX_SRC_NOT_FOUND:
      return "Bad TX source, not found in ledger";
   case EMCM_TX_SRC_TAG:
      return "Bad TX source tag";
   case EMCM_TX_SRC_TAGGED:
      return "Bad TX, src tag != chg tag, and src tag non-default";
   case EMCM_TXMDST_AMOUNTS:
      return "Bad multi-destination TX amounts do not match total";
   case EMCM_TXMDST_AMOUNTS_OVERFLOW:
      return "Bad multi-destination TX amounts overflowed";
   case EMCM_TXMDST_CHG_DISSOLVE:
      return "Bad multi-destination TX change tag will dissolve";
   case EMCM_TXMDST_DST_AMOUNT:
      return "Bad multi-destination TX destination amount is zero";
   case EMCM_TXMDST_DST_IS_SRC:
      return "Bad multi-destination TX destination tag is source tag";
   case EMCM_TXMDST_FEES:
      return "Bad multi-destination TX fees do not cover tx fee";
   case EMCM_TXMDST_FEES_OVERFLOW:
      return "Bad multi-destination TX fees overflowed";
   case EMCM_TXMDST_SRC_NOT_CHG:
      return "Bad multi-destination TX src tag != chg tag";
   case EMCM_TXMDST_SRC_TAG:
      return "Bad multi-destination TX missing src tag";
   case EMCM_TXWOTS_SIG:
      return "Bad TX, WOTS+ signature invalid";
   case EMCM_XTX_NZTPADDING:
      return "eXtended TX contains non-zero trailing padding";
   case EMCM_XTX_UNDEFINED:
      return "eXtended TX type is not defined";

/* END MOCHIMO ERRORS */

/* BEGIN POSIX ERRNO ERRORS */

#if defined(E2BIG)
      case E2BIG:  /* (POSIX.1-2001) */
         return "Argument list too long";
#endif
#if defined(WSAEACCES)
      case WSAEACCES:  /* (Windows Sockets) fallthrough */
#endif
#if defined(EACCES)
      case EACCES:  /* (POSIX.1-2001) */
#endif
#if defined(WSAEACCES) || defined(EACCES)
         return "Permission denied";
#endif
#if defined(WSAEADDRINUSE)
      case WSAEADDRINUSE:  /* (Windows Sockets) fallthrough */
#endif
#if defined(EADDRINUSE)
      case EADDRINUSE:  /* (POSIX.1-2001) */
#endif
#if defined(WSAEADDRINUSE) || defined(EADDRINUSE)
         return "Address already in use";
#endif
#if defined(WSAEADDRNOTAVAIL)
      case WSAEADDRNOTAVAIL:  /* (Windows Sockets) fallthrough */
#endif
#if defined(EADDRNOTAVAIL)
      case EADDRNOTAVAIL:  /* (POSIX.1-2001) */
#endif
#if defined(WSAEADDRNOTAVAIL) || defined(EADDRNOTAVAIL)
         return "Address not available";
#endif
#if defined(WSAEAFNOSUPPORT)
      case WSAEAFNOSUPPORT:  /* (Windows Sockets) fallthrough */
#endif
#if defined(EAFNOSUPPORT)
      case EAFNOSUPPORT:  /* (POSIX.1-2001) */
#endif
#if defined(WSAEAFNOSUPPORT) || defined(EAFNOSUPPORT)
         return "Address family not supported";
#endif
/* EAGAIN may be same value as EWOULDBLOCK, but EAGAIN takes precedence */
#if defined(EAGAIN)
      case EAGAIN:  /* (POSIX.1-2001) */
         return "Resource temporarily unavailable";
#endif
#if defined(WSAEALREADY)
      case WSAEALREADY:  /* (Windows Sockets) fallthrough */
#endif
#if defined(EALREADY)
      case EALREADY:  /* (POSIX.1-2001) */
#endif
#if defined(WSAEALREADY) || defined(EALREADY)
         return "Connection already in progress";
#endif
#if defined(EBADE)
      case EBADE:
         return "Invalid exchange";
#endif
#if defined(WSAEBADF)
      case WSAEBADF:  /* (Windows Sockets) fallthrough */
#endif
#if defined(EBADF)
      case EBADF:  /* (POSIX.1-2001) */
#endif
#if defined(WSAEBADF) || defined(EBADF)
         return "Bad file descriptor";
#endif
#if defined(EBADFD)
      case EBADFD:
         return "File descriptor in bad state";
#endif
#if defined(EBADMSG)
      case EBADMSG:  /* (POSIX.1-2001) */
         return "Bad message";
#endif
#if defined(EBADR)
      case EBADR:
         return "Invalid request descriptor";
#endif
#if defined(EBADRQC)
      case EBADRQC:
         return "Invalid request code";
#endif
#if defined(EBADSLT)
      case EBADSLT:
         return "Invalid slot";
#endif
#if defined(EBUSY)
      case EBUSY:  /* (POSIX.1-2001) */
         return "Device or resource busy";
#endif
#if defined(ECANCELED)
      case ECANCELED:  /* (POSIX.1-2001) */
         return "Operation canceled";
#endif
#if defined(ECHILD)
      case ECHILD:  /* (POSIX.1-2001) */
         return "No child processes";
#endif
#if defined(ECHRNG)
      case ECHRNG:
         return "Channel number out of range";
#endif
#if defined(ECOMM)
      case ECOMM:
         return "Communication error on send";
#endif
#if defined(WSAECONNABORTED)
      case WSAECONNABORTED:  /* (Windows Sockets) fallthrough */
#endif
#if defined(ECONNABORTED)
      case ECONNABORTED:  /* (POSIX.1-2001) */
#endif
#if defined(WSAECONNABORTED) || defined(ECONNABORTED)
         return "Connection aborted";
#endif
#if defined(WSAECONNREFUSED)
      case WSAECONNREFUSED:  /* (Windows Sockets) fallthrough */
#endif
#if defined(ECONNREFUSED)
      case ECONNREFUSED:  /* (POSIX.1-2001) */
#endif
#if defined(WSAECONNREFUSED) || defined(ECONNREFUSED)
         return "Connection refused";
#endif
#if defined(WSAECONNRESET)
      case WSAECONNRESET:  /* (Windows Sockets) fallthrough */
#endif
#if defined(ECONNRESET)
      case ECONNRESET:  /* (POSIX.1-2001) */
#endif
#if defined(WSAECONNRESET) || defined(ECONNRESET)
         return "Connection reset";
#endif
#if defined(EDEADLK)
      case EDEADLK:  /* (POSIX.1-2001) */
         return "Resource deadlock avoided";
#endif
/* On most architectures, EDEADLOCK is a synonym for EDEADLK.
 * On some architectures (e.g., Linux MIPS, PowerPC, SPARC),
 * it is a separate error code. */
#if defined(EDEADLOCK) && (!defined(EDEADLK) || (EDEADLOCK != EDEADLK))
      case EDEADLOCK:
         return "File locking deadlock error";
#endif
#if defined(WSAEDESTADDRREQ)
      case WSAEDESTADDRREQ:  /* (Windows Sockets) fallthrough */
#endif
#if defined(EDESTADDRREQ)
      case EDESTADDRREQ:  /* (POSIX.1-2001) */
#endif
#if defined(WSAEDESTADDRREQ) || defined(EDESTADDRREQ)
         return "Destination address required";
#endif
#if defined(EDOM)
      case EDOM:
         return "Mathematics argument out of domain of function";
         /* (POSIX.1, C99) */
#endif
#if defined(WSAEDQUOT)
      case WSAEDQUOT:  /* (Windows Sockets) fallthrough */
#endif
#if defined(EDQUOT)  /* (POSIX.1-2001) */
      case EDQUOT:
#endif
#if defined(WSAEDQUOT) || defined(EDQUOT)
         return "Disk quota exceeded";
#endif
#if defined(EEXIST)
      case EEXIST:  /* (POSIX.1-2001) */
         return "File exists";
#endif
#if defined(WSAEFAULT)
      case WSAEFAULT:  /* (Windows Sockets) fallthrough */
#endif
#if defined(EFAULT)
      case EFAULT:  /* (POSIX.1-2001) */
#endif
#if defined(WSAEFAULT) || defined(EFAULT)
         return "Bad address";
#endif
#if defined(EFBIG)
      case EFBIG:  /* (POSIX.1-2001) */
         return "File too large";
#endif
#if defined(WSAEHOSTDOWN)
      case WSAEHOSTDOWN:  /* (Windows Sockets) fallthrough */
#endif
#if defined(EHOSTDOWN)
      case EHOSTDOWN:
#endif
#if defined(WSAEHOSTDOWN) || defined(EHOSTDOWN)
         return "Host is down";
#endif
#if defined(WSAEHOSTUNREACH)
      case WSAEHOSTUNREACH:  /* (Windows Sockets) fallthrough */
#endif
#if defined(EHOSTUNREACH)
      case EHOSTUNREACH:  /* (POSIX.1-2001) */
#endif
#if defined(WSAEHOSTUNREACH) || defined(EHOSTUNREACH)
         return "Host is unreachable";
#endif
#if defined(EHWPOISON)
      case EHWPOISON:
         return "Memory page has hardware error";
#endif
#if defined(EIDRM)
      case EIDRM:  /* (POSIX.1-2001) */
         return "Identifier removed";
#endif
#if defined(EILSEQ)
      case EILSEQ:
         return "Invalid or incomplete multibyte or wide character";
         /* (POSIX.1, C99) The text shown here is the glibc error
          * description; in POSIX.1, this error is described as
          * "Illegal byte sequence" */
#endif
#if defined(WSAEINPROGRESS)
      case WSAEINPROGRESS:  /* (Windows Sockets) fallthrough */
#endif
#if defined(EINPROGRESS)
      case EINPROGRESS:  /* (POSIX.1-2001) */
#endif
#if defined(WSAEINPROGRESS) || defined(EINPROGRESS)
         return "Operation in progress";
#endif
#if defined(WSAEINTR)
      case WSAEINTR:  /* (Windows Sockets) fallthrough */
#endif
#if defined(EINTR)
      case EINTR:  /* (POSIX.1-2001) */
#endif
#if defined(WSAEINTR) || defined(EINTR)
         return "Interrupted function call";
#endif
#if defined(WSAEINVAL)
      case WSAEINVAL:  /* (Windows Sockets) fallthrough */
#endif
#if defined(EINVAL)  /* (POSIX.1-2001) */
      case EINVAL:
#endif
#if defined(WSAEINVAL) || defined(EINVAL)
         return "Invalid argument";
#endif
#if defined(EIO)
      case EIO:  /* (POSIX.1-2001) */
         return "Input/output error";
#endif
#if defined(WSAEISCONN)
      case WSAEISCONN:  /* (Windows Sockets) fallthrough */
#endif
#if defined(EISCONN)
      case EISCONN:  /* (POSIX.1-2001) */
#endif
#if defined(WSAEISCONN) || defined(EISCONN)
         return "Socket is connected";
#endif
#if defined(EISDIR)
      case EISDIR:  /* (POSIX.1-2001) */
         return "Is a directory";
#endif
#if defined(EISNAM)
      case EISNAM:
         return "Is a named type file";
#endif
#if defined(EKEYEXPIRED)
      case EKEYEXPIRED:
         return "Key has expired";
#endif
#if defined(EKEYREJECTED)
      case EKEYREJECTED:
         return "Key was rejected by service";
#endif
#if defined(EKEYREVOKED)
      case EKEYREVOKED:
         return "Key has been revoked";
#endif
#if defined(EL2HLT)
      case EL2HLT:
         return "Level 2 halted";
#endif
#if defined(EL2NSYNC)
      case EL2NSYNC:
         return "Level 2 not synchronized";
#endif
#if defined(EL3HLT)
      case EL3HLT:
         return "Level 3 halted";
#endif
#if defined(EL3RST)
      case EL3RST:
         return "Level 3 reset";
#endif
#if defined(ELIBACC)
      case ELIBACC:
         return "Cannot access a needed shared library";
#endif
#if defined(ELIBBAD)
      case ELIBBAD:
         return "Accessing a corrupted shared library";
#endif
#if defined(ELIBMAX)
      case ELIBMAX:
         return "Attempting to link in too many shared libraries";
#endif
#if defined(ELIBSCN)
      case ELIBSCN:
         return ".lib section in a.out corrupted";
#endif
#if defined(ELIBEXEC)
      case ELIBEXEC:
         return "Cannot exec a shared library directly";
#endif
#if defined(ELNRANGE)
      case ELNRANGE:
         return "Link number out of range";
#endif
#if defined(ELOOP)
      case ELOOP:  /* (POSIX.1-2001) */
         return "Too many levels of symbolic links";
#endif
#if defined(EMEDIUMTYPE)
      case EMEDIUMTYPE:
         return "Wrong medium type";
#endif
#if defined(WSAEMFILE)
      case WSAEMFILE:  /* (Windows Sockets) fallthrough */
#endif
#if defined(EMFILE)
      case EMFILE:  /* (POSIX.1-2001) */
#endif
#if defined(WSAEMFILE) || defined(EMFILE)
         return "Too many open files";
#endif
#if defined(EMLINK)
      case EMLINK:  /* (POSIX.1-2001) */
         return "Too many links";
#endif
#if defined(WSAEMSGSIZE)
      case WSAEMSGSIZE:  /* (Windows Sockets) fallthrough */
#endif
#if defined(EMSGSIZE)
      case EMSGSIZE:  /* (POSIX.1-2001) */
#endif
#if defined(WSAEMSGSIZE) || defined(EMSGSIZE)
         return "Message too long";
#endif
#if defined(EMULTIHOP)
      case EMULTIHOP:  /* (POSIX.1-2001) */
         return "Multihop attempted";
#endif
#if defined(WSAENAMETOOLONG)
      case WSAENAMETOOLONG:  /* (Windows Sockets) fallthrough */
#endif
#if defined(ENAMETOOLONG)
      case ENAMETOOLONG:  /* (POSIX.1-2001) */
#endif
#if defined(WSAENAMETOOLONG) || defined(ENAMETOOLONG)
         return "Filename too long";
#endif
#if defined(WSAENETDOWN)
      case WSAENETDOWN:  /* (Windows Sockets) fallthrough */
#endif
#if defined(ENETDOWN)
      case ENETDOWN:  /* (POSIX.1-2001) */
#endif
#if defined(WSAENETDOWN) || defined(ENETDOWN)
         return "Network is down";
#endif
#if defined(WSAENETRESET)
      case WSAENETRESET:  /* (Windows Sockets) fallthrough */
#endif
#if defined(ENETRESET)
      case ENETRESET:  /* (POSIX.1-2001) */
#endif
#if defined(WSAENETRESET) || defined(ENETREWSAENETRESET)
         return "Connection aborted by network";
#endif
#if defined(WSAENETUNREACH)
      case WSAENETUNREACH:  /* (Windows Sockets) fallthrough */
#endif
#if defined(ENETUNREACH)
      case ENETUNREACH:  /* (POSIX.1-2001) */
#endif
#if defined(WSAENETUNREACH) || defined(ENETUNREACH)
         return "Network unreachable";
#endif
#if defined(ENFILE)
      case ENFILE:  /* (POSIX.1-2001) */
         return "Too many open files in system";
#endif
#if defined(ENOANO)
      case ENOANO:
         return "No anode";
#endif
#if defined(WSAENOBUFS)
      case WSAENOBUFS:  /* (Windows Sockets) fallthrough */
#endif
#if defined(ENOBUFS)
      case ENOBUFS:  /* (POSIX.1 (XSI STREAMS option)) */
#endif
#if defined(WSAENOBUFS) || defined(ENOBUFS)
         return "No buffer space available";
#endif
#if defined(ENODATA)
      case ENODATA:
         return "The named attribute does not exist, or "
            "the process has no access to this attribute";
         /* In POSIX.1-2001 (XSI STREAMS option), this error was described
          * as "No message is available on the STREAM head read queue". */
#endif
#if defined(ENODEV)
      case ENODEV:  /* (POSIX.1-2001) */
         return "No such device";
#endif
#if defined(ENOENT)
      case ENOENT:  /* (POSIX.1-2001) */
         return "No such file or directory";
#endif
#if defined(ENOEXEC)
      case ENOEXEC:  /* (POSIX.1-2001) */
         return "Exec format error";
#endif
#if defined(ENOKEY)
      case ENOKEY:
         return "Required key not available";
#endif
#if defined(ENOLCK)
      case ENOLCK:  /* (POSIX.1-2001) */
         return "No locks available";
#endif
#if defined(ENOLINK)
      case ENOLINK:  /* (POSIX.1-2001) */
         return "Link has been severed";
#endif
#if defined(ENOMEDIUM)
      case ENOMEDIUM:
         return "No medium found";
#endif
#if defined(ENOMEM)
      case ENOMEM:  /* (POSIX.1-2001) */
         return "Not enough space/cannot allocate memory";
#endif
#if defined(ENOMSG)
      case ENOMSG:  /* (POSIX.1-2001) */
         return "No message of the desired type";
#endif
#if defined(ENONET)
      case ENONET:
         return "Machine is not on the network";
#endif
#if defined(ENOPKG)
      case ENOPKG:
         return "Package not installed";
#endif
#if defined(WSAENOPROTOOPT)
      case WSAENOPROTOOPT:  /* (Windows Sockets) fallthrough */
#endif
#if defined(ENOPROTOOPT)
      case ENOPROTOOPT:  /* (POSIX.1-2001) */
#endif
#if defined(WSAENOPROTOOPT) || defined(ENOPROTOOPT)
         return "Protocol not available";
#endif
#if defined(ENOSPC)
      case ENOSPC:  /* (POSIX.1-2001) */
         return "No space left on device";
#endif
#if defined(ENOSR)
      case ENOSR:  /* (POSIX.1 (XSI STREAMS option)) */
         return "No STREAM resources";
#endif
#if defined(ENOSTR)
      case ENOSTR:  /* (POSIX.1 (XSI STREAMS option)) */
         return "Not a STREAM";
#endif
#if defined(ENOSYS)
      case ENOSYS:  /* (POSIX.1-2001) */
         return "Function not implemented";
#endif
#if defined(ENOTBLK)
      case ENOTBLK:
         return "Block device required";
#endif
#if defined(WSAENOTCONN)
      case WSAENOTCONN:  /* (Windows Sockets) fallthrough */
#endif
#if defined(ENOTCONN)
      case ENOTCONN:  /* (POSIX.1-2001) */
#endif
#if defined(WSAENOTCONN) || defined(ENOTCONN)
         return "The socket is not connected";
#endif
#if defined(ENOTDIR)
      case ENOTDIR:  /* (POSIX.1-2001) */
         return "Not a directory";
#endif
#if defined(WSAENOTEMPTY)
      case WSAENOTEMPTY:  /* (Windows Sockets) fallthrough */
#endif
#if defined(ENOTEMPTY)
      case ENOTEMPTY:  /* (POSIX.1-2001) */
#endif
#if defined(WSAENOTEMPTY) || defined(ENOTEMPTY)
         return "Directory not empty";
#endif
#if defined(WSANO_RECOVERY)
      case WSANO_RECOVERY:  /* (Windows Sockets) fallthrough */
#endif
#if defined(ENOTRECOVERABLE)
      case ENOTRECOVERABLE:  /* (POSIX.1-2008) */
#endif
#if defined(WSANO_RECOVERY) || defined(ENOTRECOVERABLE)
         return "State not recoverable";
#endif
#if defined(WSAENOTSOCK)
      case WSAENOTSOCK:  /* (Windows Sockets) fallthrough */
#endif
#if defined(ENOTSOCK)
      case ENOTSOCK:  /* (POSIX.1-2001) */
#endif
#if defined(WSAENOTSOCK) || defined(ENOTSOCK)
         return "Not a socket";
#endif
#if defined(ENOTSUP)
      case ENOTSUP:  /* (POSIX.1-2001) */
         return "Operation not supported";
#endif
#if defined(ENOTTY)
      case ENOTTY:  /* (POSIX.1-2001) */
         return "Inappropriate I/O control operation";
#endif
#if defined(ENOTUNIQ)
      case ENOTUNIQ:
         return "Name not unique on network";
#endif
#if defined(ENXIO)
      case ENXIO:  /* (POSIX.1-2001) */
         return "No such device or address";
#endif
#if defined(WSAEOPNOTSUPP)
      case WSAEOPNOTSUPP:  /* (Windows Sockets) fallthrough */
#endif
/* ENOTSUP and EOPNOTSUPP may have the same value on Linux, but
 * according to POSIX.1 these error values should be distinct */
#if defined(EOPNOTSUPP) && (!defined(ENOTSUP) || (EOPNOTSUPP != ENOTSUP))
      case EOPNOTSUPP:  /* (POSIX.1-2001) */
#endif
#if defined(WSAEOPNOTSUPP) || \
   (defined(EOPNOTSUPP) && (!defined(ENOTSUP) || (EOPNOTSUPP != ENOTSUP)))
         return "Operation not supported on socket";
#endif
#if defined(EOVERFLOW)
      case EOVERFLOW:  /* (POSIX.1-2001) */
         return "Value too large to be stored in data type";
#endif
#if defined(EOWNERDEAD)
      case EOWNERDEAD:  /* (POSIX.1-2008) */
         return "Owner died";
#endif
#if defined(EPERM)
      case EPERM:  /* (POSIX.1-2001) */
         return "Operation not permitted";
#endif
#if defined(WSAEPFNOSUPPORT)
      case WSAEPFNOSUPPORT:  /* (Windows Sockets) fallthrough */
#endif
#if defined(EPFNOSUPPORT)
      case EPFNOSUPPORT:
#endif
#if defined(WSAEPFNOSUPPORT) || defined(EPFNOSUPPORT)
         return "Protocol family not supported";
#endif
#if defined(EPIPE)
      case EPIPE:  /* (POSIX.1-2001) */
         return "Broken pipe";
#endif
#if defined(EPROTO)
      case EPROTO:  /* (POSIX.1-2001) */
         return "Protocol error";
#endif
#if defined(WSAEPROTONOSUPPORT)
      case WSAEPROTONOSUPPORT:  /* (Windows Sockets) fallthrough */
#endif
#if defined(EPROTONOSUPPORT)
      case EPROTONOSUPPORT:  /* (POSIX.1-2001) */
#endif
#if defined(WSAEPROTONOSUPPORT) || defined(EPROTONOSUPPORT)
         return "Protocol not supported";
#endif
#if defined(WSAEPROTOTYPE)
      case WSAEPROTOTYPE:  /* (Windows Sockets) fallthrough */
#endif
#if defined(EPROTOTYPE)
      case EPROTOTYPE:  /* (POSIX.1-2001) */
#endif
#if defined(WSAEPROTOTYPE) || defined(EPROTOTYPE)
         return "Protocol wrong type for socket";
#endif
#if defined(ERANGE)
      case ERANGE:  /* (POSIX.1, C99) */
         return "Result too large";
#endif
#if defined(EREMCHG)
      case EREMCHG:
         return "Remote address changed";
#endif
#if defined(WSAEREMOTE)
      case WSAEREMOTE:  /* (Windows Sockets) fallthrough */
#endif
#if defined(EREMOTE)
      case EREMOTE:
#endif
#if defined(WSAEREMOTE) || defined(EREMOTE)
         return "Object is remote";
#endif
#if defined(EREMOTEIO)
      case EREMOTEIO:
         return "Remote I/O error";
#endif
#if defined(ERESTART)
      case ERESTART:
         return "Interrupted system call should be restarted";
#endif
#if defined(ERFKILL)
      case ERFKILL:
         return "Operation not possible due to RF-kill";
#endif
#if defined(EROFS)
      case EROFS:  /* (POSIX.1-2001) */
         return "Read-only filesystem";
#endif
#if defined(WSAESHUTDOWN)
      case WSAESHUTDOWN:  /* (Windows Sockets) fallthrough */
#endif
#if defined(ESHUTDOWN)
      case ESHUTDOWN:
#endif
#if defined(WSAESHUTDOWN) || defined(ESHUTDOWN)
         return "Cannot send after transport endpoint shutdown";
#endif
#if defined(ESPIPE)
      case ESPIPE:  /* (POSIX.1-2001) */
         return "Invalid seek";
#endif
#if defined(WSAESOCKTNOSUPPORT)
      case WSAESOCKTNOSUPPORT:  /* (Windows Sockets) fallthrough */
#endif
#if defined(ESOCKTNOSUPPORT)
      case ESOCKTNOSUPPORT:
#endif
#if defined(WSAESOCKTNOSUPPORT) || defined(ESOCKTNOSUPPORT)
         return "Socket type not supported";
#endif
#if defined(ESRCH)
      case ESRCH:  /* (POSIX.1-2001) */
         return "No such process";
#endif
#if defined(WSAESTALE)
      case WSAESTALE:  /* (Windows Sockets) fallthrough */
#endif
#if defined(ESTALE)
      case ESTALE:  /* (POSIX.1-2001) */
#endif
#if defined(WSAESTALE) || defined(ESTALE)
         return "Stale file handle";
#endif
#if defined(ESTRPIPE)
      case ESTRPIPE:
         return "Streams pipe error";
#endif
#if defined(ETIME)
      case ETIME:  /* (POSIX.1 (XSI STREAMS option)) */
         return "Timer expired";
         /* POSIX.1 says "STREAM ioctl(2) timeout" */
#endif
#if defined(WSAETIMEDOUT)
      case WSAETIMEDOUT:  /* (Windows Sockets) fallthrough */
#endif
#if defined(ETIMEDOUT)
      case ETIMEDOUT:  /* (POSIX.1-2001) */
#endif
#if defined(WSAETIMEDOUT) || defined(ETIMEDOUT)
         return "Connection timed out";
#endif
#if defined(WSAETOOMANYREFS)
      case WSAETOOMANYREFS:  /* (Windows Sockets) fallthrough */
#endif
#if defined(ETOOMANYREFS)
      case ETOOMANYREFS:
#endif
#if defined(WSAETOOMANYREFS) || defined(ETOOMANYREFS)
         return "Too many references: cannot splice";
#endif
#if defined(ETXTBSY)
      case ETXTBSY:  /* (POSIX.1-2001) */
         return "Text file busy";
#endif
#if defined(EUCLEAN)
      case EUCLEAN:
         return "Structure needs cleaning";
#endif
#if defined(EUNATCH)
      case EUNATCH:
         return "Protocol driver not attached";
#endif
#if defined(WSAEUSERS)
      case WSAEUSERS:  /* (Windows Sockets) fallthrough */
#endif
#if defined(EUSERS)
      case EUSERS:
#endif
#if defined(WSAEUSERS) || defined(EUSERS)
         return "Too many users";
#endif
#if defined(WSAEWOULDBLOCK)
      case WSAEWOULDBLOCK:  /* (Windows Sockets) fallthrough */
#endif
/* EWOULDBLOCK may be same value as EAGAIN, but EAGAIN takes precedence */
#if defined(EWOULDBLOCK) && (!defined(EAGAIN) || (EWOULDBLOCK != EAGAIN))
      case EWOULDBLOCK:  /* (POSIX.1-2001) */
#endif
#if defined(WSAEWOULDBLOCK) || \
   (defined(EWOULDBLOCK) && (!defined(EAGAIN) || (EWOULDBLOCK != EAGAIN)))
        return "Operation would block";
#endif
#if defined(EXDEV)
      case EXDEV:  /* (POSIX.1-2001) */
         return "Improper link";
#endif
#if defined(EXFULL)
      case EXFULL:
         return "Exchange full";
#endif

/* END ERRNO ERRORS */

/* BEGIN WSA ERRNO ERRORS */

#if defined(WSA_OPERATION_ABORTED)
      case WSA_OPERATION_ABORTED:  /* (Windows Sockets) */
         return "Overlapped operation aborted";
#endif
#if defined(WSA_IO_INCOMPLETE)
      case WSA_IO_INCOMPLETE:  /* (Windows Sockets) */
         return "Overlapped I/O event object not in signaled state";
#endif
#if defined(WSA_IO_PENDING)
      case WSA_IO_PENDING:  /* (Windows Sockets) */
         return "Overlapped operations will complete later";
#endif
#if defined(WSAELOOP)
      case WSAELOOP:  /* (Windows Sockets) */
         return "Cannot translate name";
#endif
#if defined(WSAEPROCLIM)
      case WSAEPROCLIM:  /* (Windows Sockets) */
         return "Too many processes";
#endif
#if defined(WSASYSNOTREADY)
      case WSASYSNOTREADY:  /* (Windows Sockets) */
         return "Network subsystem is unavailable";
#endif
#if defined(WSAVERNOTSUPPORTED)
      case WSAVERNOTSUPPORTED:  /* (Windows Sockets) */
         return "Winsock.dll version out of range";
#endif
#if defined(WSANOTINITIALISED)
      case WSANOTINITIALISED:  /* (Windows Sockets) */
         return "Succesful WSAStartup not yet performed";
#endif
#if defined(WSAEDISCON)
      case WSAEDISCON:  /* (Windows Sockets) */
         return "Graceful shutdown in progress";
#endif
#if defined(WSA_E_NO_MORE)
      case WSA_E_NO_MORE:  /* (Windows Sockets) */
#endif
#if defined(WSAENOMORE)
      case WSAENOMORE:  /* (Windows Sockets) */
#endif
#if defined(WSA_E_NO_MORE) || defined(WSAENOMORE)
         return "No more results";
#endif
#if defined(WSAECANCELLED)
      case WSAECANCELLED:  /* (Windows Sockets) */
         return "Call has been canceled";
#endif
#if defined(WSAEINVALIDPROCTABLE)
      case WSAEINVALIDPROCTABLE:  /* (Windows Sockets) */
         return "Procedure call table is invalid";
#endif
#if defined(WSAEINVALIDPROVIDER)
      case WSAEINVALIDPROVIDER:  /* (Windows Sockets) */
         return "Service provider is invalid";
#endif
#if defined(WSAEPROVIDERFAILEDINIT)
      case WSAEPROVIDERFAILEDINIT:  /* (Windows Sockets) */
         return "Service provider failed to initialize";
#endif
#if defined(WSASYSCALLFAILURE)
      case WSASYSCALLFAILURE:  /* (Windows Sockets) */
         return "System call failure";
#endif
#if defined(WSASERVICE_NOT_FOUND)
      case WSASERVICE_NOT_FOUND:  /* (Windows Sockets) */
         return "Service not found";
#endif
#if defined(WSATYPE_NOT_FOUND)
      case WSATYPE_NOT_FOUND:  /* (Windows Sockets) */
         return "Class type not found";
#endif
#if defined(WSA_E_CANCELLED)
      case WSA_E_CANCELLED:  /* (Windows Sockets) */
         return "Call was canceled";
#endif
#if defined(WSAEREFUSED)
      case WSAEREFUSED:  /* (Windows Sockets) */
         return "Database query was refused";
#endif
#if defined(WSAHOST_NOT_FOUND)
      case WSAHOST_NOT_FOUND:  /* (Windows Sockets) */
         return "Host not found";
#endif
#if defined(WSATRY_AGAIN)
      case WSATRY_AGAIN:  /* (Windows Sockets) */
         return "Nonauthoritative host not found";
#endif
#if defined(WSANO_DATA)
      case WSANO_DATA:  /* (Windows Sockets) */
         return "Valid name, no data record of requested type";
#endif

/* END WSA ERRNO ERRORS */

      default:
         return "Unknown error...";
   }
}  /* end errno_description() */

/* end include guard */
#endif
