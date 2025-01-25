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

/* C/C++ compatible function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

int ng_val(const char *ngfile, const word8 bnum[8]);
int b_val(const char *bcfile, const char *ltfile);

#ifdef __cplusplus
}  /* end extern "C" */
#endif

/* end include guard */
#endif
