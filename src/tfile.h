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

/* number of Tfile entries to load into network packet */
#ifndef NTFTX
   #define NTFTX 54
#endif

/* ensure validity of NTFTX definition */
#define NTFTX_SPACE  ( sizeof(((TX *) NULL)->buffer) / sizeof(BTRAILER) )
STATIC_ASSERT(NTFTX <= NTFTX_SPACE, NTFTX_too_large_for_buffer);

extern word8 TfileShutdown;

/* C/C++ compatible prototypes */
#ifdef __cplusplus
extern "C" {
#endif

void add_weight(word8 weight[32], word8 difficulty);
int append_tfile(const BTRAILER *bt, size_t count, const char *file);
word32 get_bridge(const void *bnum);
void get_mreward(word8 reward[8], const word8 bnum[8]);
int get_tfrewards(const char *tfile, word8 rewards[8], const word8 bnum[8]);
void merkle_root(const word8 *hashlist, size_t count, word8 *root);
size_t read_tfile
   (void *buffer, const word8 bnum[8], size_t count, const char *tfile);
int read_trailer(BTRAILER *bt, const char *file);
word32 next_difficulty(const BTRAILER *bt);
int past_weight(const char *tfile, const word8 bnum[8], word8 weight[32]);
int trim_tfile(const char *tfile, const word8 highbnum[8]);
int validate_pow(const BTRAILER *btp);
int validate_trailer(const BTRAILER *bt, const BTRAILER *prev_bt);
int validate_tfile_fp(FILE *fp, word8 bnum[8], word8 weight[32], int trust);
int validate_tfile_pow_fp(FILE *fp, int trust);
int validate_tfile_pow(const char *tfile, int trust);
int validate_tfile
   (const char *tfile, word8 bnum[8], word8 weight[32], int trust);
int weigh_tfile(const char *tfile, const word8 bnum[8], word8 weight[32]);

#ifdef __cplusplus
}  /* end extern "C" */
#endif

/* end include guard */
#endif
