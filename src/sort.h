/**
 * @file sort.h
 * @brief Mochimo related sorting support.
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
 * @note The original Polymorphic Shell sort algorithm, shell(), was
 * deprecated in favour of qsort().
 * > For more details see <https://godbolt.org/z/YE7j57Po9>
*/

/* include guard */
#ifndef MOCHIMO_SORT_H
#define MOCHIMO_SORT_H


#include "extint.h"

/* C/C++ compatible function prototypes for wots.c */
#ifdef __cplusplus
extern "C" {
#endif

word8 *Tx_ids; /**< malloc'd Tx_ids[]: Ntx * HASHLEN */
word32 *Txidx; /**< malloc'd Txidx[]: Ntx * sizeof(word32) */
word32 Ntx;    /**< number of transaction entries in "txclean" */

int sortlt(char *fname);
int sorttx(char *fname);

/* end extern "C" {} for C++ */
#ifdef __cplusplus
}
#endif

/* end include guard */
#endif
