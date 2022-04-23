
#define PATCHLEVEL 37
#define VERSIONSTR  "37"   /*   as printable string */

#include "extinet.h"    /* socket support */
#include "extlib.h"     /* general support */
#include "extmath.h"    /* 64-bit math support */
#include "extprint.h"   /* print/logging support */

/* Include everything that we need */
#include "config.h"
#include "mochimo.h"
#include "proto.h"

/* Include global data . . . */
#include "data.c"       /* System wide globals  */

/* Support functions  */
#include "crypto/crc16.c"
#include "crypto/crc32.c"      /* for mirroring          */

/* Server control */
#include "util.c"       /* server support */
#include "pink.c"       /* manage pinklist                 */
#include "ledger.c"
#include "tag.c"        /* address tag support             */
#include "gettx.c"      /* poll and read NODE socket       */
#include "txval.c"      /* validate transactions           */
#include "mirror.c"
#include "execute.c"
#include "monitor.c"    /* system monitor/debugger prompt  */
#include "daemon.c"
#include "miner.c"
#include "pval.c"       /* pseudo-blocks                   */
#include "optf.c"       /* for OP_HASH and OP_TF           */
#include "proof.c"
#include "renew.c"
#include "update.c"
#include "init.c"       /* read Coreplist[] and get_eon()  */
#include "server.c"     /* tcp server */
int main(void)
{
   Running = 1;
   Difficulty=16;

   miner("cblock.dat", "mblock.dat");
   return 0;
}
