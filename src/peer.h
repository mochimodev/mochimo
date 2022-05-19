/**
 * @file peer.h
 * @brief Mochimo peer handling support.
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_PEER_H
#define MOCHIMO_PEER_H


#include "global.h"
#include "types.h"

#define addrecent(ip)   addpeer(ip, Rplist, RPLISTLEN, &Rplistidx)

/* global variables */
extern char *Coreipfname;
extern char *Epinkipfname;
extern char *Recentipfname;
extern char *Trustedipfname;
extern word32 Rplist[RPLISTLEN], Rplistidx;
extern word32 Tplist[TPLISTLEN], Tplistidx;
extern word32 Cpinklist[CPINKLEN], Cpinkidx;
extern word32 Lpinklist[LPINKLEN], Lpinkidx;
extern word32 Epinklist[EPINKLEN], Epinkidx;
extern word8 Nopinklist;
extern word8 Noprivate;

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
void print_ipl(word32 *list, word32 len);
int save_ipl(char *fname, word32 *list, word32 len);
word32 read_ipl(char *fname, word32 *plist, word32 plistlen, word32 *plistidx);
int pinklisted(word32 ip);
int cpinklist(word32 ip);
int pinklist(word32 ip);
int lpinklist(word32 ip);
int epinklist(word32 ip);
void mergepinklists(void);
void purge_epoch(void);

#ifdef __cplusplus
}  /* end extern "C" */
#endif

/* end include guard */
#endif
