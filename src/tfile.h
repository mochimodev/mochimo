/**
 * @file tfile.h
 * @brief Mochimo trailer file support.
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_TFILE_H
#define MOCHIMO_TFILE_H


/* internal support */
#include "types.h"

/* Count of trailers that fit in a TX: */
#define NTFTX (TRANLEN / sizeof(BTRAILER))

/* C/C++ compatible prototypes */
#ifdef __cplusplus
extern "C" {
#endif

int get_tfile_rewards(const char *fname, void *rewards, void *bnum);
int read_tfile(void *buffer, void *bnum, int count, const char *tfname);
int read_trailer(BTRAILER *btp, const char *fname);

int readtf(void *buff, word32 bnum, word32 count);
word32 set_difficulty(BTRAILER *btp);
int past_weight(word8 *weight, word32 lownum);
int loadproof(TX *tx);
int checkproof(TX *tx, word32 *splitblock);
int tf_val(char *fname, void *bnum, void *weight, int weight_only);
int trim_tfile(void *highbnum);

int validate_trailer(BTRAILER *btp, BTRAILER *pbtp);

#ifdef __cplusplus
}  /* end extern "C" */
#endif

/* end include guard */
#endif
