/**
 * @file block.h
 * @brief Mochimo blockchain generation and validation support.
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_BLOCK_H
#define MOCHIMO_BLOCK_H


/* internal support */
#include "types.h"

/* global variables */
extern const char *Maddr_opt;
extern const char *Bcdir_opt;

/* C/C++ compatible function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

/* block_legacy.c */
int blockw_val_fp(FILE *fp, BTRAILER *btp);
int blockw_val(const char *fname, const char *tfname);
int blockw(const char *fname, TXW *txw_clean, size_t count, const char *tfname);
int neogenw_val_fp(FILE *fp, const char *tfname);
int neogenw_val(const char *fname, const char *tfname);

/* block.c */
int archive_block(const char *filename, const char *dirname);
int block_syncup_fp(FILE *fp, void *next_syncblock);
int block_update(const char *fname);
int generate_neogen(const char *fname, const char *lfname, const char *tfname);
int validate_pseudo_fp(FILE *fp, BTRAILER *prev_btp);
int validate_pseudo(const char *pfname, const char *tfname);
int generate_pseudo(const char *fname, const char *tfname);
int validate_block_fp(FILE *fp, const char *tfname);
int validate_block(const char *bcfname, const char *tfname);

/* end extern "C" {} for C++ */
#ifdef __cplusplus
}
#endif

/* end include guard */
#endif
