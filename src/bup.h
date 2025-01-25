/**
 * @file bup.h
 * @brief Mochimo block update support.
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_BUP_H
#define MOCHIMO_BUP_H

#include "types.h"

/* C/C++ compatible function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

void print_bup(BTRAILER *bt);
int b_update(char *fname);

#ifdef __cplusplus
}  /* end extern "C" */
#endif

/* end include guard */
#endif
