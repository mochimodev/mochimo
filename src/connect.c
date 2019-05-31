/* connect.c  Make outgoing connection
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 3 January 2018
*/

SOCKET connectip(word32 ip)
{
   SOCKET sd;
   struct sockaddr_in addr;
   word16 port;
   time_t timeout;

   if((sd = socket(AF_INET, SOCK_STREAM, 0)) == INVALID_SOCKET) {
bad:
      error("connectip(): cannot open socket.");
      return INVALID_SOCKET;
   }

   memset((char *) &addr, 0, sizeof(addr));
   port = DSTPORT;
   addr.sin_addr.s_addr = ip;
   addr.sin_family = AF_INET;  /* AF_UNIX */
   /* Convert short integer to network byte order */
   addr.sin_port = htons(port);

   nonblock(sd);  /* was after connect() v.21 */
   timeout = time(NULL) + 3;
retry:
   if(connect(sd, (struct sockaddr *) &addr, sizeof(struct sockaddr))) {
      if(errno == EISCONN) return sd;
      if((errno == EINPROGRESS || errno == EALREADY)
         && time(NULL) < timeout) goto retry;
      closesocket(sd);
      if(Trace) plog("connectip(): cannot connect(0x%08x):%d.", ip, port);
      return INVALID_SOCKET;
   }
   return sd;
}  /* end connectip() */
