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
 * Max allowable depth of the ledger tree.
 * Change to balance address search time complexity with I/O (re)writes.
*/
#ifndef LEDEPTHMAX
   #define LEDEPTHMAX   8
#endif

#if LEDEPTHMAX < 2
   #error "LEDEPTHMAX cannot be less than 2"
#endif

/**
 * Amount of buffer space to use when sequentially reading or writing.
 * Change to balance memory usage against system read/write calls.
*/
#ifndef LERWBUFSZ
   #define LERWBUFSZ ( 1 << 24 )
#endif

#if LERWBUFSZ < BUFSZ
   #error "LERWBUFSZ cannot be less than BUFSZ"
#endif

/* global variables */
extern word32 Sanctuary_opt;
extern word32 Lastday_opt;
extern char *Lefname_opt;
extern char *Tifname_opt;

/* C/C++ compatible function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

int auto_compression_depth(size_t count);
int le_append(const char *lfname, const char *tfname);
void le_close(int depth);
int le_cmpw(const void *a, const void *b);
int le_cmp(const void *a, const void *b);
int le_compress(const char *filename, int depth, int count);
void le_convert(void *hash, void *wots);
void le_delete(int depth);
int le_extract(const char *ngfname);
LENTRY *le_find(void *addr);
LENTRY *le_findw(void *wots);
int le_renew(void *fee);
int le_splice(const char *filename, int depth, int count);
int tag_cmp(const void *a, const void *b);
int tag_equal(const void *a, const void *b);
int tag_extract(const char *lfname, const char *tfname);
LENTRY *tag_find(void *tag);

/* end extern "C" {} for C++ */
#ifdef __cplusplus
}
#endif

/* end include guard */
#endif
