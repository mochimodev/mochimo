/**
 * @file chain.h
 * @brief Mochimo chain data support.
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_CHAIN_H
#define MOCHIMO_CHAIN_H


/* internal support */
#include "types.h"

/* C/C++ compatible prototypes */
#ifdef __cplusplus
extern "C" {
#endif

void add_weight(word8 *weight, word8 difficulty, word8 *bnum);
int append_tfile_fp(FILE *fp, const char *tfilename);
int append_tfile(const char *filename, const char *tfilename);
void get_mreward(void *reward, void *bnum);
int get_tfrewards(const char *fname, void *rewards, void *bnum);
word32 next_difficulty(BTRAILER *btp);
void ptrailer(BTRAILER *btp);
int read_bnum_fp(void *bnum, FILE *fp);
int read_bnum(void *bnum, const char *filename);
int read_hdrlen(void *hdrlen, const char *filename);
int read_tfile(void *buffer, void *bnum, int count, const char *tfname);
int read_trailer(BTRAILER *btp, const char *fname);
int trim_tfile(const char *tfname, void *highbnum, void *weight);
int validate_pow(BTRAILER *btp);
int validate_trailer(BTRAILER *btp, BTRAILER *pbtp);
int validate_tfile_fp(FILE *tfp, void *bnum, void *weight, int part);
int validate_tfile(const char *tfname, void *bnum, void *weight);
int weigh_tfile(const char *tfname, void *bnum, void *weight);

#ifdef __cplusplus
}  /* end extern "C" */
#endif

/* end include guard */
#endif
