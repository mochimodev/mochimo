/* daemon.c  Save us all from the signals and signs of EVIL...
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * The Mochimo Project System Software
 *
 * Date: 12 January 2018
 *
 * Requires data.c for globals: Monitor and Running.
*/

/* include guard */
#ifndef MOCHIMO_DAEMON_C
#define MOCHIMO_DAEMON_C


#include <signal.h>

#include "extinet.h"
#include "extprint.h"

/*
 * Signal handlers
 *
 * Enter monitor on ctrl-C
 */
void ctrlc(int sig)
{
   if(Trace) plog("Got signal %i\n", sig);
   signal(SIGINT, ctrlc);
   Monitor = 1;
}


/*
 * Clear run flag, Running on SIGTERM
 */
void sigterm(int sig)
{
   if(Trace) plog("Got signal %i\n", sig);
   signal(SIGTERM, sigterm);
   Running = 0;
}


void fix_signals(void)
{
   int j;

   /*
    * Ignore all signals.
    */
   for(j = 0; j <= NSIG; j++)
      signal(j, SIG_IGN);

   signal(SIGINT, ctrlc);     /* then install ctrl-C handler */
   signal(SIGTERM, sigterm);  /* ...and software termination */
}


void close_extra(void)
{
   int j;

   for(j = 3; j < 50; j++) sock_close(j);
}

/* end include guard */
#endif
