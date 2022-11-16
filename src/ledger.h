/**
 * @file ledger.h
 * @brief Mochimo ledger support.
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
 * @note **The trouble with Tags preceding their addresses...**<br/><br/>
 * The whole situation just becomes far more expensive and unnecessary.
 * Deterministic recovery of a cryptographic address would go from a
 * simple "binary search" to an "exhaustive search" of the ledger.
 * Implementing a complimentary binary search operation would require
 * indexing 32 bytes of definitely not compressible data for every
 * address in the ledger, in comparison to 12 bytes of potentially
 * compressible data containing only tagged addresses.<br/><br/>
 * @note **The trouble with deriving the "OTS Hash Address" from the
 * public "SEED" of the WOTS+ signature scheme...**<br/><br/>
 * If we change the current scheme to one that derives the "OTS Hash
 * Address" from it's public "SEED", then the WOTS+ public key and
 * subsequent Hashed Address also change to an unknowable result.
 * This kind of transition requires all occupants of the ledger to
 * transfer their funds to an address fitting the compliant scheme,
 * before allowing further transactions to take place.
*/

/* include guard */
#ifndef MOCHIMO_LEDGER_H
#define MOCHIMO_LEDGER_H


#include "types.h"

/**
 * Amount of buffer space to allocate for large sorting operations.
 * Data exceeding this amount will be split into manageable chunks.
*/
#ifndef LEBUFSZ
   #define LEBUFSZ ( 1 << 26 )
#endif

/**
 * Max allowable depth of the ledger tree.
 * Change to balance address search time complexity with I/O (re)writes.
*/
#ifndef LEDEPTHMAX
   #define LEDEPTHMAX   8
#endif

/* definition checks */
#if LEBUFSZ < BUFSIZ
   #error "LEBUFSZ cannot be less than BUFSIZ"
#endif

#if LEDEPTHMAX < 2
   #error "LEDEPTHMAX cannot be less than 2"
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

int auto_compression_depth(void);
int le_append(const char *lfname, const char *tfname);
void le_close(int depth);
int le_cmpw(const void *a, const void *b);
int le_cmp(const void *a, const void *b);
int le_compress(const char *filename, int depth, int count);
void le_convert(void *wots, void *hash);
void le_delete(int depth);
int le_extract(const char *ngfname);
LENTRY *le_find(void *addr);
LENTRY *le_findw(void *wots);
int le_renew(void *fee);
int le_splice(const char *filename, int depth, int count);
int le_transpose(void);
int le_update(char *fname);
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
