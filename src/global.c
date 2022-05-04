/**
 * @private
 * @headerfile global.h <global.h>
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_GLOBAL_C
#define MOCHIMO_GLOBAL_C


#include "global.h"

char *Bcdir = BCDIR;
char *Ngdir = NGDIR;
char *Spdir = SPDIR;

word32 Mfee[2] = { MFEE, 0 };
word32 Myfee[2] = { MFEE, 0 };
word32 Quorum = 4;
word16 Dstport = PORT1;
word16 Port = PORT1;
word8 Cbits = CBITS;
word8 One[8] = { 1 };

/* end include guard */
#endif
