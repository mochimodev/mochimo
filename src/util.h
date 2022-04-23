/**
 * util.c - Mochimo specific utilities support
 *
 * Copyright (c) 2018-2021 Adequate Systems, LLC. All Rights Reserved.
 * For more information, please refer to ../LICENSE
 *
 * Date: 2 January 2018
 * Revised: 28 October 2021
 *
*/

/* include guard */
#ifndef MOCHIMO_UTIL_H
#define MOCHIMO_UTIL_H


#include "extint.h"
#include "types.h"

/* bnum is little-endian on disk and core. */
#define weight2hex(_weight)   val2hex(_weight, 32, NULL, 0)

/* C/C++ compatible function prototypes for wots.c */
#ifdef __cplusplus
extern "C" {
#endif

#ifndef _WIN32
   int lock(char *lockfile, int seconds);
   int unlock(int fd);

#endif

void crctx(TX *tx);
int readtrailer(BTRAILER *trailer, char *fname);
char *val2hex64(void *val, char hex[]);
char *bnum2hex(void *bnum);
char *val2hex(void *val, int len, char *buf, int bufsize);
char *addr2str(void *addr);
char *hash2str(word8 *hash);
int moveublock(char *ublock, word8 *newnum);
int read_global(void);
int write_global(void);
void add_weight(word8 *weight, word8 difficulty, word8 *bnum);
void get_mreward(word32 *reward, word32 *bnum);
int append_tfile(char *fname, char *tfile);
word32 set_difficulty(BTRAILER *btp);
int bupdata(void);
int do_neogen(void);

/* end extern "C" {} for C++ */
#ifdef __cplusplus
}
#endif

/* end include guard */
#endif
