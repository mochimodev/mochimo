/* connect2.c  Make outgoing connection (can use domain names and 1.2.3.4)
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 3 January 2018
*/

SOCKET connectip(word32 ip, char *addrstr)
{
   SOCKET sd;
   struct hostent *host;
   struct sockaddr_in addr;
   word16 port;

   if((sd = socket(AF_INET, SOCK_STREAM, 0)) == INVALID_SOCKET) {
bad:
      error("connectip(): cannot open socket.");
      return INVALID_SOCKET;
   }

   port = DSTPORT;
   memset((char *) &addr, 0, sizeof(addr));
   if(addrstr) {
      if(addrstr[0] < '0' || addrstr[0] > '9') {
         host = gethostbyname(addrstr);
         if(host == NULL) {
            plog("connectip(): gethostbyname() failed");
            return INVALID_SOCKET;
         }
         memcpy((char *) &(addr.sin_addr.s_addr),
                host->h_addr_list[0], host->h_length);
      }
      else
         addr.sin_addr.s_addr = inet_addr(addrstr);
   } else {
      addr.sin_addr.s_addr = ip;
   }  /* end if NULL addrstr */

   addr.sin_family = AF_INET;  /* AF_UNIX */
   /* Convert short integer to network byte order */
   addr.sin_port = htons(port);

   if(connect(sd, (struct sockaddr *) &addr, sizeof(struct sockaddr))) {
      closesocket(sd);
      plog("connectip(): cannot connect() socket.");
      return INVALID_SOCKET;
   }

   nonblock(sd);
   return sd;
}  /* end connectip() */
