/**
 * @file tag.h
 * @brief Mochimo address tag support.
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_TAG_H
#define MOCHIMO_TAG_H


/* external support */
#include <stddef.h>
#include "extint.h"

/* C/C++ compatible function prototypes for wots.c */
#ifdef __cplusplus
extern "C" {
#endif

/* Release tag index */
void tag_free(void);
int tag_buildidx(void);
int tag_qfind(word8 *addr);
int tag_valid(word8 *src_addr, word8 *chg_addr, word8 *dst_addr, word8 *bnum);
int tag_find(word8 *addr, word8 *foundaddr, word8 *balance, size_t len);

/* end extern "C" {} for C++ */
#ifdef __cplusplus
}
#endif

/* end include guard */
#endif

