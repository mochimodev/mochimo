/**
 * @file sync.h
 * @brief Mochimo blockchain synchronization support.
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_SYNC_H
#define MOCHIMO_SYNC_H


/* internal support */
#include "network.h"
#include "types.h"

/* C/C++ compatible prototypes */
#ifdef __cplusplus
extern "C" {
#endif

int reset_chain(void);
int delete_blocks(word8 *matchblock);
int extract_gen(char *lfile);
int catchup(word32 plist[], word32 count);
int resync(word32 quorum[], word32 *qidx, void *highweight, void *highbnum);
int syncup(word32 splitblock, word8 *txcblock, word32 peerip);
int contention(NODE *np);

/* end extern "C" {} for C++ */
#ifdef __cplusplus
}
#endif

/* end include guard */
#endif
