/* str2ip.c  Char string to ip address lookup.
 *           (can use domain names and 1.2.3.4)
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 12 February 2018
*/


word32 str2ip(char *addrstr)
{
   struct hostent *host;
   struct sockaddr_in addr;

   if(addrstr == NULL) return 0;

   memset(&addr, 0, sizeof(addr));
   if(addrstr[0] < '0' || addrstr[0] > '9') {
      host = gethostbyname(addrstr);
      if(host == NULL) {
         plog("str2ip(): gethostbyname() failed");
         return 0;
      }
      memcpy((char *) &(addr.sin_addr.s_addr),
             host->h_addr_list[0], host->h_length);
   }
   else
      addr.sin_addr.s_addr = inet_addr(addrstr);

   return addr.sin_addr.s_addr;
}  /* end str2ip() */
