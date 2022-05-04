/**
 * @file sort.h
 * @brief Mochimo quick sorting support.
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_SORT_H
#define MOCHIMO_SORT_H


/* external support */
#include "extint.h"

/* C/C++ compatible function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

word8 *Tx_ids;  /* malloc'd Tx_ids[] Ntx*32 bytes */
word32 *Txidx;  /* malloc'd Txidx[] Ntx*4 bytes */
word32 Ntx;     /* number of transactions in clean TX queue */

int sortlt(char *fname);
int sorttx(char *fname);

/* end extern "C" {} for C++ */
#ifdef __cplusplus
}
#endif

/* end include guard */
#endif
