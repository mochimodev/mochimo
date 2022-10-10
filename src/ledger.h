/**
 * @file ledger.h
 * @brief Mochimo ledger support.
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_LEDGER_H
#define MOCHIMO_LEDGER_H


#include "types.h"

/**
 * Amount of buffer space to use when sequentially reading or writing.
 * Change to balance memory usage against system read/write calls.
*/
#ifndef LERWBUFSZ
   #define LERWBUFSZ ( 1 << 24 )
#endif

/**
 * Ledger merge condition function, where v = next depth of ledger tree.
 * Change to balance ledger depth with ledger compression frequency/scale.
*/
#ifndef LECOMPRESS
   #define LECOMPRESS(v)   ( 1LL << ( 1LL << (v) ) )
#endif

/* global variables */

extern word32 Sanctuary, Lastday;
extern char *Ledger_opt, *Tagidx_opt;

/* C/C++ compatible function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

int le_appendw(char *lfname, char *tfname);
int le_close(int depth);
int le_cmpp(const void *a, const void *b);
int le_cmpw(const void *a, const void *b);
int le_compressw(char *fname, int from, int to);
int le_delete(int depth);
int le_extractw(char *ngfname);
void *le_find(void *addr);
int le_reneww(void *fee);
void le_update(char *filename, size_t count);
int tag_cmp(const void *a, const void *b);
int tag_equal(const void *a, const void *b);
int tag_extractw(char *lfname, char *tfname);
void *tag_find(void *tag);

/* end extern "C" {} for C++ */
#ifdef __cplusplus
}
#endif

/* end include guard */
#endif
