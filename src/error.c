/* error.c  Error logging, trace, and fatal()
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
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
