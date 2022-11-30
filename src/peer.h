/**
 * @file peer.h
 * @brief Mochimo peer handling support.
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_PEER_H
#define MOCHIMO_PEER_H


/* internal support */
#include "types.h"

typedef struct {
   word32 *list, idx, len;
   word8 bnum[8], bhash[32], weight[32];
} QUORUM;

extern word32 Cpinklist[CPINKLEN];
extern word32 Cpinkidx;
extern word32 Epinklist[EPINKLEN];
extern word32 Epinkidx;
extern word32 Lplist[LPLISTLEN];
extern word32 Lplistidx;
extern word32 Rplist[RPLISTLEN];
extern word32 Rplistidx;

extern word8 Nopinklist_opt;
extern word8 Noprivate_opt;

/* C/C++ compatible function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

word32 *search32(word32 val, word32 *list, unsigned len);
word32 remove32(word32 bad, word32 *list, unsigned maxlen, word32 *idx);
word32 include32(word32 val, word32 *list, unsigned len, word32 *idx);
void shuffle32(word32 *list, word32 len);
int isprivate(word32 ip);
word32 addpeer(word32 ip, word32 *list, word32 len, word32 *idx);
word32 quorum_addpeer(QUORUM *qp, word32 ip);
int quorum_cleanup(QUORUM *qp);
word32 quorum_drop(QUORUM *qp, word32 ip);
int quorum_update(QUORUM *qp, NODE *np);
void print_ipl(word32 *list, word32 len);
int save_ipl(char *fname, word32 *list, word32 len);
int read_ipl(char *fname, word32 *plist, word32 plistlen, word32 *plistidx);
int pinklisted(word32 ip);
void pinklist(word32 ip);
void epinklist(word32 ip);
void purge_pinklist(void);
void purge_epinklist(void);

#ifdef __cplusplus
}  /* end extern "C" */
#endif

/* end include guard */
#endif
