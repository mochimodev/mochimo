/**
 * @file tag.h
 * @brief Mochimo address tag support.
 * @details ```
 * Ledger Entry: [<---2208 Bytes Address--->][<---8 Bytes Amount--->]
 * Ledger Address: [<---2196 Bytes WOTS+--->][<--12 Bytes Tag-->]
 *
 * Ledger Tag Types...
 * [ 0x00, ... ] = Extended Transaction Tag Types (see below)
 * [ 0x42, ... ] = Default (untagged) Address
 * [ ... ] = Tagged Addresses
 *
 * Extended Transaction Tag Types...
 * [ 0x00, 0x01, <validation bits> ] = Multi-destination Transaction
 * [ 0x00, ... ] = Reserved for future functionality
 *
 * Proposed Transaction Tag Types...
 * [ 0x00, 0x02, <simple message payload> ] = Memo Transaction
 * ```
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
 * @todo Replace the private tag_qfind() in tag.c with a better alternative.
 * Reason: stopping bcon for every new transaction check is not sustainable.
*/

/* include guard */
#ifndef MOCHIMO_TAG_H
#define MOCHIMO_TAG_H


/* external support */
#include <stddef.h>
#include "extint.h"

/* global variables */
extern word8 *Tagidx;
extern word32 Ntagidx;

/* C/C++ compatible function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

void tag_free(void);
int tag_buildidx(void);
int tag_valid(word8 *src_addr, word8 *chg_addr, word8 *dst_addr, word8 *bnum);
int tag_find(word8 *addr, word8 *foundaddr, word8 *balance, size_t len);

#ifdef __cplusplus
}  /* end extern "C" */
#endif

/* end include guard */
#endif

