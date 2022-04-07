/**
 * xo4.h - Crypto support header for shylock.c
 *
 * Copyright (c) 2021 by Adequate Systems, LLC.  All Rights Reserved.
 * For more information, please refer to ../LICENSE
 *
 * Date: 20 September 2021
 * Revised: 26 October 2021
 *
 * --------  XO4 Cipher package  --------
 * Courtesy Patrick Cargill -- EYES ONLY!
 *
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

/* C/C++ compatible function prototypes for wots.c */
#ifdef __cplusplus
extern "C" {
#endif

void xo4_init(XO4_CTX *ctx, void *key, size_t len);
void xo4_crypt(XO4_CTX *ctx, void *input, void *output, size_t len);

/* end extern "C" {} for C++ */
#ifdef __cplusplus
}
#endif

/* end include guard */
#endif
