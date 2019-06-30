/* sock.h  Includes for Berkeley Sockets
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * The Mochimo Project System Software
 *
*/

/* #define FD_SETSIZE  (MAXNODES+1)   * include listening socket */

#ifdef WIN32
#include <winsock.h>
#define EISCONN      WSAEISCONN
#define EINPROGRESS  WSAEINPROGRESS 
#define EALREADY     WSAEALREADY 
#define EWOULDBLOCK  WSAEWOULDBLOCK
#define getsockerr() WSAGetLastError()
#else
#include <fcntl.h>
#include <sys/socket.h>           /* for Unix sockets */
#include <netdb.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <arpa/nameser.h>
#include <arpa/inet.h>
#include <resolv.h>
#include <sys/types.h>
#include <termios.h>     /* for FIONBIO */
#ifndef SOCKET
#define SOCKET int
#endif
#define SOCKET_ERROR (-1)
#define INVALID_SOCKET (-1)
#define FAR
/* was (h_errno) */
#define getsockerr() (errno)
#define WSAGetLastError() (errno)
#define closesocket(_sd) close(_sd)
#define ioctlsocket(_fd, _cmd, _arg) ioctl(_fd, _cmd, _arg)
#endif  /* not WIN32 */
