/**
 * @file ledger.h
 * @brief Mochimo ledger support.
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_LEDGER_H
#define MOCHIMO_LEDGER_H


#define _FILE_OFFSET_BITS  64 /* for 64-bit off_t stdio */
#include "types.h"

#ifndef LEBUFSZ
   /**
    * Amount of buffer space to allocate for large ledger operations.
    * Data exceeding this amount will be split into chunks.
   */
   #define LEBUFSZ ( 1 << 26 ) /* 64M */
#endif

/* global variables */
extern word32 Sanctuary;
extern word32 Lastday;

/* C/C++ compatible function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

int addr_compare(const void * a, const void * b);
void addr_convert(const word8 *wots, word8 *addr);
int addr_tag_compare(const void * a, const void * b);
int addr_tag_equal(const void * a, const void * b);
int le_open(const char *lefile);
void le_close(void);
int le_extract(const char *neogen_file, const char *ledger_file);
int le_find(const word8 *addr, LENTRY *le, word16 len);
int le_renew(void);
int le_update(const char *ltfname);
int tag_compare(const void *a, const void *b);
int tag_equal(const void *a, const void *b);

/* end extern "C" {} for C++ */
#ifdef __cplusplus
}
#endif

/* end include guard */
#endif
