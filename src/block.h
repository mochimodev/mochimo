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
extern char *Maddr_opt;
extern char *Bcdir_opt;

/* C/C++ compatible function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

/* block_legacy.c */
int blockw_val_fp(FILE *fp, BTRAILER *btp);
int blockw_val(char *fname, char *tfname);
int blockw(char *fname, TXW *txw_clean, size_t count, char *tfname);
int neogenw_val_fp(FILE *fp, char *tfname);
int neogenw_val(char *fname, char *tfname);

/* block.c */
int neogen(char *fname, char *lfname, char *tfname);
int pseudo_val_fp(FILE *fp, BTRAILER *prev_btp);
int pseudo_val(char *pfname, char *tfname);
int pseudo(char *fname, char *tfname);
int update_block(char *fname);
int validate_block_fp(FILE *fp, char *tfname);
int validate_block(char *bcfname, char *tfname);

/* end extern "C" {} for C++ */
#ifdef __cplusplus
}
#endif

/* end include guard */
#endif
