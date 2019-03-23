/* sock.c  Support functions that require SOCKET and inet types.
 *
 * Copyright (c) 2018 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 9 January 2018
 *
*/


#ifdef WIN32
/* Set socket sd to non-blocking I/O on Win32 */
int nonblock(SOCKET sd)
{
   u_long arg = 1L;

   return ioctlsocket(sd, FIONBIO, (u_long FAR *) &arg);
}

#else
#include <fcntl.h>

/* Set socket sd to non-blocking I/O
 * Returns -1 on error.
 */
int nonblock(SOCKET sd)
{
   int flags;

   flags = fcntl(sd, F_GETFL, 0);
   return fcntl(sd, F_SETFL, flags | O_NONBLOCK);
}

/* Set socket sd to blocking I/O
 * Returns -1 on error.
 */
int blocking(SOCKET sd)
{
   int flags;

   flags = fcntl(sd, F_GETFL, 0);
   return fcntl(sd, F_SETFL, flags & (~O_NONBLOCK));
}

#endif


word32 getsocketip(SOCKET sd)
{
   struct sockaddr_in addr;
   unsigned addrlen;

   addrlen = sizeof(addr);
   if(getpeername(sd, (struct sockaddr *) &addr, &addrlen) == 0) {
#ifdef DEBUG
      if(Trace > 1)
         plog("[%s]  0x%08x", ntoa((byte *) &addr.sin_addr),
              addr.sin_addr.s_addr);
#endif
      return (word32) addr.sin_addr.s_addr;  /* the 32-bit ip */
   }
   return INVALID_SOCKET;
}  /* end getsocketip() */
