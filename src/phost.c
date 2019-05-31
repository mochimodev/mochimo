/* phost.c
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 2 January 2018
 *
 * Functions to print internet host info.
 *
*/


void phostent(struct hostent *host)
{
    int i;

    if(host->h_name == NULL)
       host->h_name = "unknown";
    printf("    hostname: %s\n", host->h_name);

    printf("    aliases: ");
    for(i = 0; host->h_aliases[i]; i++)
       printf(" %s", host->h_aliases[i]);
    printf("\n");

    printf("    addrlist: ");
    for(i = 0; host->h_addr_list[i]; i++)
       printf(" %s", ntoa((byte *) host->h_addr_list[i]));
    printf("\n\n");
}

/* Print local host info on stdout */
int phostinfo(void)
{
   int result;
   char hostname[100];
   struct hostent *host;

    /*
     * Get local machine name and IP address
     */
    result = gethostname(hostname, sizeof(hostname));
    if(result == SOCKET_ERROR) {
      error("gethostname(): %u", WSAGetLastError());
      return -1;
    }
    host = gethostbyname(hostname);
    if(host == NULL) {
       error("gethostbyname(): %u", WSAGetLastError());
       return -1;
    }
    printf("Local Machine Info\n");
    phostent(host);
    return 0;
}
