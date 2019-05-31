/* wots.h  WOTS+ Public address, Signature, and Verification
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * The Mochimo Project System Software
 *
 * Code in "wots" directory is derived from the XMSS reference implementation
 * written by Andreas Huelsing and Joost Rijneveld of the Crypto Forum
 * Research Group.  See ATTRIBUTION in this directory.
*/

#ifndef WOTS_H
#define WOTS_H


/* Parameters */

#define WOTSW      16
#define WOTSLOGW   4
#define WOTSLEN    (WOTSLEN1 + WOTSLEN2)
#define WOTSLEN1   (8 * PARAMSN / WOTSLOGW)
#define WOTSLEN2   3
#define WOTSSIGBYTES (WOTSLEN * PARAMSN)
#define PARAMSN 32

/* 2144 + 32 + 32 = 2208 */
#define TXSIGLEN   2144
#define TXADDRLEN  2208


/* Prototypes */

/**
 * WOTS key generation. Takes a 32 byte seed for the private key, expands it to
 * a full WOTS private key and computes the corresponding public key.
 * It requires the seed pub_seed (used to generate bitmasks and hash keys)
 * and the address of this WOTS key pair.
 *
 * Writes the computed public key to 'pk'.
 */
void wots_pkgen(byte *pk, const byte *seed,
                const byte *pub_seed, word32 addr[8]);

/**
 * Takes a n-byte message and the 32-byte seed for the private key to compute a
 * signature that is placed at 'sig'.
 */
void wots_sign(byte *sig, const byte *msg,
               const byte *seed, const byte *pub_seed,
               word32 addr[8]);

/**
 * Takes a WOTS signature and an n-byte message, computes a WOTS public key.
 *
 * Writes the computed public key to 'pk'.
 */
void wots_pk_from_sig(byte *pk,
                      const byte *sig, const byte *msg,
                      const byte *pub_seed, word32 addr[8]);

#endif
