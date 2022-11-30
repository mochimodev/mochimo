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
int append_tfile_fp(FILE *fp, char *tfilename);
int append_tfile(char *filename, char *tfilename);
void get_mreward(void *reward, void *bnum);
int get_tfrewards(char *fname, void *rewards, void *bnum);
word32 next_difficulty(BTRAILER *btp);
int read_bnum(void *bnum, char *filename);
int read_hdrlen(void *hdrlen, char *filename);
int read_tfile(void *buffer, void *bnum, int count, char *tfname);
int read_trailer(BTRAILER *btp, char *fname);
int trim_tfile(char *tfname, void *highbnum);
int validate_pow(BTRAILER *btp);
int validate_trailer(BTRAILER *btp, BTRAILER *pbtp);
int validate_tfile_fp(FILE *tfp, void *highbnum, void *highweight);
int validate_tfile(char *tfname, void *highbnum, void *highweight);
int weigh_tfile(char *tfname, void *highweight);

#ifdef __cplusplus
}  /* end extern "C" */
#endif

/* end include guard */
#endif
