/**
 * @file bval.h
 * @brief Mochimo block validation support.
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_BVAL_H
#define MOCHIMO_BVAL_H


/* internal support */
#include "types.h"

/* C/C++ compatible prototypes */
#ifdef __cplusplus
extern "C" {
#endif

int p_val(char *fname);
int ng_val(char *fname, word8 *bnum);
int b_val(char *fname);

/* end extern "C" {} for C++ */
#ifdef __cplusplus
}
#endif

/* end include guard */
#endif
