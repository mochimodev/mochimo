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


#include <execinfo.h>
#include <signal.h>

#include "extinet.h"
#include "extprint.h"

#ifndef NSIG
#define NSIG 23
#endif

/*
 * Signal handlers
 *
 * Enter monitor on ctrl-C
 */
void ctrlc(int sig)
{
   pdebug("Got signal %i\n", sig);
   signal(SIGINT, ctrlc);
   if (Ininit) Running = 0;
   else Monitor = 1;
}


/*
 * Clear run flag, Running on SIGTERM
 */
void sigterm(int sig)
{
   pdebug("Got signal %i\n", sig);
   signal(SIGTERM, sigterm);
   Running = 0;
}

void segfault(int sig) {
  void *array[10];
  size_t size;

  // get void*'s for all entries on the stack
  size = backtrace(array, 10);

  // print out all the frames to stderr
  fprintf(stderr, "Error: signal %d:\n", sig);
  backtrace_symbols_fd(array, size, STDERR_FILENO);
  exit(1);
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
   signal(SIGSEGV, segfault);   // install our handler
}


void close_extra(void)
{
   int j;

   for(j = 3; j < 50; j++) sock_close(j);
}

/* end include guard */
#endif
