/**
 * @file txclean.h
 * @brief Mochimo TX queue cleaning support.
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_TXCLEAN_H
#define MOCHIMO_TXCLEAN_H


/* C/C++ compatible prototypes */
#ifdef __cplusplus
extern "C" {
#endif

int txclean_bc(char *fname);
int txclean_le(void);

/* end extern "C" {} for C++ */
#ifdef __cplusplus
}
#endif

/* end include guard */
#endif
