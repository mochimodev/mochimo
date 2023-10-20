/**
 * @file ledger.h
 * @brief Mochimo ledger support.
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_LEDGER_H
#define MOCHIMO_LEDGER_H


#include "types.h"

/**
 * Legacy ledger entry struct
*/
typedef struct {
   word8 addr[TXWOTSLEN];    /* 2208 */
   word8 balance[TXAMOUNT];  /* 8 */
} LENTRY_W;

/* global variables */
extern word32 Sanctuary;
extern word32 Lastday;

/* C/C++ compatible function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

int le_open(char *ledger, char *fopenmode);
void le_close(void);
int le_find(word8 *addr, LENTRY *le, long *position, word16 len);
int le_extract(char *fname, char *lfile);
int le_renew(void);
int le_txclean(void);
int le_update(void);

/* end extern "C" {} for C++ */
#ifdef __cplusplus
}
#endif

/* end include guard */
#endif
