/* mochimo.h   Master Header
 *
 * Copyright (c) 2018 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 2 January 2018
 *
 * System-wide constants, structures, macros, and typedefs.
 *
 * Comment Suffix Legend:  @=Needs check
*/

#ifndef MOCHIMO_H
#define MOCHIMO_H

#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdarg.h>
#include <time.h>
#include <string.h>
#include <signal.h>

#include <unistd.h>
#include <errno.h>
#include <sys/wait.h>  /* for waitpid() */
#include <sys/file.h>  /* for flock() */
#include <fcntl.h>

#ifndef NSIG
#define NSIG 23
#endif

#include "crypto/sha256.h"

#ifdef DEBUG
#define debug(_x) if(Trace > 1) plog(_x)
#define ifdebug(x) x
#else
#define debug(_x)
#define ifdebug(x)
#endif

/* Function return  codes */
#define VEOK        0      /* No error                    */

/* error codes */
#define VERROR      1      /* General error               */
#define VEBAD       2      /* client was bad              */
#define VEBAD2      3      /* client was naughty          */
#define VETIMEOUT   (-1)   /* socket timeout              */

#define TRUE     1
#define FALSE    0

#ifndef SOCKET
         #define SOCKET unsigned int   /* Borland 32-bit */
         #define INVALID_SOCKET  (SOCKET)(~0)
#endif

#ifndef SOCKET_ERROR
#define SOCKET_ERROR   (-1)
#endif

#ifndef INVALID_SOCKET
#define INVALID_SOCKET (-1)
#endif

#include "crypto/wots/wots.h"
#include "types.h"

#endif  /* MOCHIMO_H */
