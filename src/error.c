/* error.c  Error logging, trace, and fatal()
 *
 * Copyright (c) 2018 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 1 January 2018
 *
 * Error logging and trace functions
 *
*/


FILE *Logfp = NULL;
word32 Nerrors;      /* error counter */
char *Statusarg;     /* Statusarg->"message_string" shows on ps */

#ifndef ERRORFNAME
#define ERRORFNAME "error.log"
#endif
#ifndef LOGFNAME
#define LOGFNAME "mochi.log"
#endif


void log_time(FILE *fp)
{
   time_t curtime;

   time(&curtime);
   fprintf(fp, " on %s", asctime(gmtime(&curtime)));
   fflush(fp);
}


/* Print an error message to error file and/or stdout
 * NOTE: errorfp is opened and closed on each call.
 */
int error(char *fmt, ...)
{
   va_list argp;
   FILE *errorfp;

   if(fmt == NULL) return VERROR;

   Nerrors++;
   errorfp = NULL;
   if(Errorlog) {
      errorfp = fopen(ERRORFNAME, "a");
   }
   if(errorfp != NULL) {
      fprintf(errorfp, "error: ");
      va_start(argp, fmt);
      vfprintf(errorfp, fmt, argp);
      va_end(argp);
      log_time(errorfp);
   }
   if(Logfp != NULL) {
      fprintf(Logfp, "error: ");
      va_start(argp, fmt);
      vfprintf(Logfp, fmt, argp);
      va_end(argp);
      log_time(Logfp);
   }
   if(!Bgflag && errorfp != stdout) {
      fprintf(stdout, "error: ");
      va_start(argp, fmt);
      vfprintf(stdout, fmt, argp);
      va_end(argp);
      log_time(stdout); 
   }
   if(errorfp != NULL) {
      fclose(errorfp);
   }
   return VERROR;
}


/* Print message to log file, Logfp, and/or stdout */
void plog(char *fmt, ...)
{
   va_list argp;

   if(fmt == NULL) return;

   if(Logfp != NULL) {
      va_start(argp, fmt);
      vfprintf(Logfp, fmt, argp);
      va_end(argp);
      log_time(Logfp);
   }
   if(!Bgflag && Logfp != stdout) {
      va_start(argp, fmt);
      vfprintf(stdout, fmt, argp);
      va_end(argp);
      log_time(stdout);
   }
}


/* Kill the miner child */
int stop_miner(void)
{
   int status;

   if(Mpid == 0) return -1;
   kill(Mpid, SIGTERM);
   waitpid(Mpid, &status, 0);
   Mpid = 0;
   return status;
}


/* Display terminal error message
 * and exit with exitcode after reaping zombies.
 */
void fatal2(int exitcode, char *message)
{
   stop_miner();
   if(Sendfound_pid) kill(Sendfound_pid, SIGTERM);
#ifndef EXCLUDE_NODES
   stop_mirror();
#endif
   if(!Bgflag && message) {
      error("%s", message);
      fprintf(stdout, "fatal: %s\n", message);
   }
   /* wait for all children */
   while(waitpid(-1, NULL, 0) != -1);
   exit(exitcode);
}

/* Display terminal error message
 * and exit with NO restart (code 0).
 */
#define fatal(mess) fatal2(0, mess)
#define pause_server() fatal2(0, NULL);

void restart(char *mess)
{
   unlink("epink.lst");
   stop_miner();
   if(Trace && mess != NULL) plog("restart: %s", mess);
   fatal2(1, NULL);
}

char *show(char *state)
{
   char *cp, *sp;

   if(state == NULL) state = "(null)";
   if(Statusarg) strncpy(Statusarg, state, 8);
   return state;
}
