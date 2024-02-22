/**
 * @file tx.h
 * @brief Mochimo transaction support.
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_TX_H
#define MOCHIMO_TX_H


/* internal support */
#include "types.h"
#include "network.h"

/* extended-c support */
#include "extint.h"
#ifndef _WIN32
   #include <sys/file.h>   /* flock() */

#endif

/* C/C++ compatible function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

#ifndef _WIN32
   int lock(char *lockfile, int seconds);
   int unlock(int fd);

#endif

int mtx_val(MTX *mtx, word32 *fee);
int tx_val(TX *tx);
int txcheck(word8 *src_addr);
int txmap(TX *tx, word32 src_ip);
pid_t mgc(word32 ip);
pid_t mirror1(word32 *iplist, int len);
pid_t mirror(void);
int process_tx(NODE *np);

#ifdef __cplusplus
}  /* end extern "C" */
#endif

/* end include guard */
#endif
