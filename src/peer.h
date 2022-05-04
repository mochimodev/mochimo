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

/* C/C++ compatible prototypes */
#ifdef __cplusplus
extern "C" {
#endif

char *Coreipfname;
char *Epinkipfname;
char *Recentipfname;
char *Trustedipfname;

word32 Rplist[RPLISTLEN], Rplistidx;  /* Recent peer list */
word32 Tplist[TPLISTLEN], Tplistidx;  /* Trusted peer list - preserved */

/* pink lists of EVIL IP addresses read in from disk */
word32 Cpinklist[CPINKLEN], Cpinkidx;
word32 Lpinklist[LPINKLEN], Lpinkidx;
word32 Epinklist[EPINKLEN], Epinkidx;

word8 Nopinklist;  /* disable pinklist IP's when set */
word8 Noprivate;     /* filter out private IP's when set v.28 */

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

/* end extern "C" {} for C++ */
#ifdef __cplusplus
}
#endif

/* end include guard */
#endif
