/**
 * @file wots.h
 * @brief WOTS+ Public address, Signature, and Verification support.
 * @details Mochimo's implementation of WOTS+ is derived from the XMSS
 * reference implementation written by Andreas Huelsing and Joost Rijneveld
 * of the Crypto Forum Research Group:
 * > "XMSS: Extended Hash-Based Signatures"<br />
 * > https://datatracker.ietf.org/doc/draft-irtf-cfrg-xmss-hash-based-signatures/draft-irtf-cfrg-xmss-hash-based-signatures-11
 * > <br />Update 12/11/2018 - the above RFC was moved from Draft status to
 * > published RFC, and can now be found here: <br />
 * > "XMSS: eXtended Merkle Signature Scheme"<br />
 * > <https://datatracker.ietf.org/doc/rfc8391/>
 * The reference implementation is permanently available at
 * <https://github.com/joostrijneveld/xmss-reference>
 * under the CC0 1.0 Universal Public Domain Dedication. For more
 * information on that license, please refer to
 * <http://creativecommons.org/publicdomain/zero/1.0/>
 *
 * Constants inlined from the reference implementation:
 * ```
 # PARAMSN = 32
 # W = 16
 # LOGW = 4
 # LEN2 = 3
 # LEN1 = 8 * PARAMSN / LOGW = 64
 # LEN = LEN1 + LEN2 = 67
 # ```
 * @copyright Adequate Systems LLC, 2018-2024. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_WOTS_H
#define MOCHIMO_WOTS_H


#include "extint.h"  /* for word types */

/* WOTS+ parameters */
#define WOTSSIGBYTES 2144
#define WOTSADDRLEN  2208

/* C/C++ compatible function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

void wots_sign(word8 *sig, const word8 *msg, const word8 *seed,
               const word8 *pub_seed, word32 addr[8]);
void wots_pkgen(word8 *pk, const word8 *seed, const word8 *pub_seed,
               word32 addr[8]);
void wots_pk_from_sig(word8 *pk, const word8 *sig, const word8 *msg,
                      const word8 *pub_seed, word32 addr[8]);

#ifdef __cplusplus
}  /* end extern "C" */
#endif

/* end include guard */
#endif
