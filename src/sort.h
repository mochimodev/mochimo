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

/* global variables */
extern word8 *Tx_ids;
extern word32 *Txidx;
extern word32 Ntx;

/* C/C++ compatible function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

int sortlt(char *fname);
int sorttx(char *fname);

#ifdef __cplusplus
}  /* end extern "C" */
#endif

/* end include guard */
#endif
