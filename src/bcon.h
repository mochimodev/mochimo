/**
 * @file bcon.h
 * @brief Mochimo block construction and alternate block generation support.
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_BCON_H
#define MOCHIMO_BCON_H


#include "types.h"

/* C/C++ compatible function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

int pseudo(const char *output);
int neogen(const BTRAILER *bt, const char *lefile, const char *output);
int b_con(const char *fname);

#ifdef __cplusplus
}  /* end extern "C" */
#endif

/* end include guard */
#endif
