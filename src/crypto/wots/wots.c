/* wots.c  WOTS+ Public address, Signature, and Verification
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

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* typedef unsigned long word32;   * for 16-bit compiler */
/* typedef unsigned char byte; */

#define core_hash(out, in, inlen) sha256(in, inlen, out)

#include "../hash/cpu/sha256.h"  /* defines byte and word32 */
#include "wots.h"

/**
 * Converts the value of 'in' to 'outlen' bytes in big-endian byte order.
 */
void ull_to_bytes(byte *out, unsigned int outlen,
                  unsigned long in)
{
    int i;

    /* Iterate over out in decreasing order, for big-endianness. */
    for (i = outlen - 1; i >= 0; i--) {
        out[i] = in & 0xff;
        in = in >> 8;
    }
}

#include "wotshash.c"


/**
 * Helper method for pseudorandom key generation.
 * Expands an n-byte array into a len*n byte array using the `prf` function.
 */
static void expand_seed(byte *outseeds, const byte *inseed)
{
    word32 i;
    byte ctr[32];

    for (i = 0; i < WOTSLEN; i++) {
        ull_to_bytes(ctr, 32, i);
        prf(outseeds + i*PARAMSN, ctr, inseed);
    }
}

/**
 * Computes the chaining function.
 * out and in have to be n-byte arrays.
 *
 * Interprets in as start-th value of the chain.
 * addr has to contain the address of the chain.
 */
static void gen_chain(byte *out, const byte *in,
                      unsigned int start, unsigned int steps,
                      const byte *pub_seed, word32 addr[8])
{
    word32 i;

    /* Initialize out with the value at position 'start'. */
    memcpy(out, in, PARAMSN);

    /* Iterate 'steps' calls to the hash function. */
    for (i = start; i < (start+steps) && i < WOTSW; i++) {
        set_hash_addr(addr, i);
        thash_f(out, out, pub_seed, addr);
    }
}

/**
 * base_w algorithm as described in draft.
 * Interprets an array of bytes as integers in base w.
 * This only works when log_w is a divisor of 8.
 */
static void base_w(int *output, const int out_len, const byte *input)
{
    int in = 0;
    int out = 0;
    byte total;
    int bits = 0;
    int consumed;

    for (consumed = 0; consumed < out_len; consumed++) {
        if (bits == 0) {
            total = input[in];
            in++;
            bits += 8;
        }
        bits -= WOTSLOGW;
        output[out] = (total >> bits) & (WOTSW - 1);
        out++;
    }
}

/* Computes the WOTS+ checksum over a message (in base_w). */
static void wots_checksum(int *csum_base_w, const int *msg_base_w)
{
    int csum = 0;
    byte csum_bytes[(WOTSLEN2 * WOTSLOGW + 7) / 8];
    unsigned int i;

    /* Compute checksum. */
    for (i = 0; i < WOTSLEN1; i++) {
        csum += WOTSW - 1 - msg_base_w[i];
    }

    /* Convert checksum to base_w. */
    /* Make sure expected empty zero bits are the least significant bits. */
    csum = csum << (8 - ((WOTSLEN2 * WOTSLOGW) % 8));
    ull_to_bytes(csum_bytes, sizeof(csum_bytes), csum);
    base_w(csum_base_w, WOTSLEN2, csum_bytes);
}

/* Takes a message and derives the matching chain lengths. */
static void chain_lengths(int *lengths, const byte *msg)
{
    base_w(lengths, WOTSLEN1, msg);
    wots_checksum(lengths + WOTSLEN1, lengths);
}

/**
 * WOTS key generation. Takes a 32 byte seed for the private key, expands it to
 * a full WOTS private key and computes the corresponding public key.
 * It requires the seed pub_seed (used to generate bitmasks and hash keys)
 * and the address of this WOTS key pair.
 *
 * Writes the computed public key to 'pk'.
 */
void wots_pkgen(byte *pk, const byte *seed,
                const byte *pub_seed, word32 addr[8])
{
    word32 i;

    /* The WOTS+ private key is derived from the seed. */
    expand_seed(pk, seed);

    for (i = 0; i < WOTSLEN; i++) {
        set_chain_addr(addr, i);
        gen_chain(pk + i * PARAMSN, pk + i * PARAMSN,
                  0, WOTSW - 1, pub_seed, addr);
    }
}

/**
 * Takes a n-byte message and the 32-byte seed for the private key to compute a
 * signature that is placed at 'sig'.
 */
void wots_sign(byte *sig, const byte *msg,
               const byte *seed, const byte *pub_seed,
               word32 addr[8])
{
    int lengths[WOTSLEN];
    word32 i;

    chain_lengths(lengths, msg);

    /* The WOTS+ private key is derived from the seed. */
    expand_seed(sig, seed);

    for (i = 0; i < WOTSLEN; i++) {
        set_chain_addr(addr, i);
        gen_chain(sig + i * PARAMSN, sig + i * PARAMSN,
                  0, lengths[i], pub_seed, addr);
    }
}

/**
 * Takes a WOTS signature and an n-byte message, computes a WOTS public key.
 *
 * Writes the computed public key to 'pk'.
 */
void wots_pk_from_sig(byte *pk,
                      const byte *sig, const byte *msg,
                      const byte *pub_seed, word32 addr[8])
{
    int lengths[WOTSLEN];
    word32 i;

    chain_lengths(lengths, msg);

    for (i = 0; i < WOTSLEN; i++) {
        set_chain_addr(addr, i);
        gen_chain(pk + i * PARAMSN, sig + i * PARAMSN,
                  lengths[i], WOTSW - 1 - lengths[i], pub_seed, addr);
    }
}
