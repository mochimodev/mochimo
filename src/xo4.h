/**
 * @file xo4.h
 * @brief Crypto support header for shylock.
 * @details --------  XO4 Cipher package  --------
 * <br/>Courtesy Patrick Cargill -- EYES ONLY!
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_XO4_H
#define MOCHIMO_XO4_H


#include <stddef.h>  /* for size_t */
#include "extint.h"  /* for word types */

typedef struct {
   word8 s[64];   /**< Seed for encryption, up to 64 Bytes */
   word8 rnd[32]; /**< Seed hash, as entropy */
   int j;         /**< Seed hash byte iterator */
} XO4_CTX;  /**< XO4 encryption context */

/* C/C++ compatible function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

void xo4_init(XO4_CTX *ctx, void *key, size_t len);
void xo4_crypt(XO4_CTX *ctx, void *input, void *output, size_t len);

#ifdef __cplusplus
}  /* end extern "C" */
#endif

/* end include guard */
#endif
