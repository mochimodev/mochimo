/**
 * @file validate.h
 * @brief Mochimo blockchain validation support.
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
 * @note Use of "Validation Error MACROs assumes @code int ecode; @endcode
 * has been previously declared.
*/

/* include guard */
#ifndef MOCHIMO_VALIDATE_H
#define MOCHIMO_VALIDATE_H


/* extended-c support */
#include "extint.h"     /* integer support */
#include "extprint.h"   /* print/logging support */

/* mochimo support */
#include "types.h"

/* validation error MACROs */

/**
 * Validation error code. Sets @a ecode to given value and jumps to label.
 * Example: @code vEcode(FAIL_LABEL, VETIMEOUT); @endcode
 * @param _lbl Label to jump to
 * @param _e Error code to set ecode to
*/
#define vEcode(_lbl, _e)   { ecode = _e; goto _lbl; }

/**
 * Validation protocol violation. Calls perr(...) with variable arguments,
 * sets @a ecode to VEBAD2 (indicating that a peer is in violation of
 * protocol and may need pinklisting), and jumps to label.
 * Example: @code vEdrop(FAIL_LABEL, "Violation of protocol"); @endcode
 * @param _lbl Label to jump to
 * @param ... arguments passed to perr()
*/
#define vEdrop(_lbl, ...)  { perr(__VA_ARGS__); vEcode(_lbl, VEBAD2); }

/**
 * Validation error w/ error number. Calls perrno(...) with variable
 * arguments, sets @a ecode to VERROR, and jumps to label.
 * Example: @code vErrno(FAIL_LABEL, errno, "Failure message"); @endcode
 * @param _lbl Label to jump to
 * @param ... arguments passed to perr()
*/
#define vErrno(_lbl, ...)  vEcode(_lbl, perrno(__VA_ARGS__));

/**
 * Validation error. Calls perr(...) with variable arguments, sets
 * @a ecode to VERROR, and jumps to label.
 * Example: @code vError(FAIL_LABEL, "Failure message"); @endcode
 * @param _lbl Label to jump to
 * @param ... arguments passed to perr()
*/
#define vError(_lbl, ...)  vEcode(_lbl, perr(__VA_ARGS__));


#define BAIL(m) { message = m; goto bail; }

/* C/C++ compatible function prototypes for wots.c */
#ifdef __cplusplus
extern "C" {
#endif

int mtx_val(MTX *mtx, word32 *fee);
int tag_valid(word8 *src_addr, word8 *chg_addr, word8 *dst_addr, word8 *bnum);
int tx_val(TX *tx);
int p_val(char *fname);
int b_val(char *fname, char *vfname);

/* end extern "C" {} for C++ */
#ifdef __cplusplus
}
#endif

/* end include guard */
#endif

