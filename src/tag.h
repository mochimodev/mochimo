/**
 * @file tag.h
 * @brief Mochimo address tag support.
 * @details ```
 * [<---2196 Bytes Address--->][<--12 Bytes Tag-->]
 * 12-byte tag:  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
 *               ^Type byte: 42 - Default (untagged)
 *                           1 - Multi-destination
 * ```
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
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
int tag_qfind(word8 *addr);
int tag_valid(word8 *src_addr, word8 *chg_addr, word8 *dst_addr, word8 *bnum);
int tag_find(word8 *addr, word8 *foundaddr, word8 *balance, size_t len);

#ifdef __cplusplus
}  /* end extern "C" */
#endif

/* end include guard */
#endif

