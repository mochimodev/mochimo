/**
 * @file transaction.h
 * @brief Mochimo transaction support.
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_TRANSACTION_H
#define MOCHIMO_TRANSACTION_H

/* internal support */
#include "types.h"

/* C/C++ compatible function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

int tx_memo_val(TX_MEMO *tx, void *fee);
int tx_val(TX *tx, void *fee);
int txw_mdst_val(TXW_MDST *tx, void *fee);
int txw_val(TXW *tx, void *fee);

#ifdef __cplusplus
}  /* end extern "C" */
#endif

/* end include guard */
#endif
