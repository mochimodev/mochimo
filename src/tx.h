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
#include "extprint.h"
#include "extint.h"

/* C/C++ compatible function prototypes for wots.c */
#ifdef __cplusplus
extern "C" {
#endif

int mtx_val(MTX *mtx, word32 *fee);
int tx_val(TX *tx);
int txcheck(word8 *src_addr);
int txmap(TX *tx, word32 src_ip);
pid_t mgc(word32 ip);
pid_t mirror1(word32 *iplist, int len);
pid_t mirror(void);
int process_tx(NODE *np);

/* end extern "C" {} for C++ */
#ifdef __cplusplus
}
#endif

/* end include guard */
#endif
