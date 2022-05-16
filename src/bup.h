/**
 * @file bup.h
 * @brief Mochimo block update support.
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_BUP_H
#define MOCHIMO_BUP_H


/* C/C++ compatible prototypes */
#ifdef __cplusplus
extern "C" {
#endif

int b_txclean(char *bcfname);
int b_update(char *fname, int mode);

/* end extern "C" {} for C++ */
#ifdef __cplusplus
}
#endif

/* end include guard */
#endif
