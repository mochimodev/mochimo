/**
 * @private
 * @headerfile wots.h <wots.h>
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_WOTS_C
#define MOCHIMO_WOTS_C


#include "wots.h"
#include "sha256.h" /* for core_hash hook */
#include <string.h> /* for memory handling */

/**
 * @private
 * Hash padding parameters
*/
#define XMSS_HASH_PADDING_F   0
#define XMSS_HASH_PADDING_PRF 3

/**
 * @private
 * Core hashing function - sha256
*/
#define core_hash(out, in, inlen) sha256(in, inlen, out)

/* These functions are used for OTS addresses. */

static void set_key_and_mask(word32 addr[8], word32 key_and_mask)
{
    addr[7] = key_and_mask;
}

static void set_chain_addr(word32 addr[8], word32 chain)
{
    addr[5] = chain;
}

static void set_hash_addr(word32 addr[8], word32 hash)
{
    addr[6] = hash;
}

/**
 * @private
 * Converts the value of 'in' to 'outlen' bytes in big-endian byte order.
 */
static void ull_to_bytes(word8 *out, unsigned int outlen, unsigned long in)
{
    int i;

    /* Iterate over out in decreasing order, for big-endianness. */
    for (i = outlen - 1; i >= 0; i--) {
        out[i] = in & 0xff;
        in = in >> 8;
    }
}  /* end ull_to_bytes() */

/**
 * @private
 * Converts @a addr to bytes using ull_to_bytes().
 */
static void addr_to_bytes(word8 *bytes, const word32 addr[8])
{
    int i;
    for (i = 0; i < 8; i++) {
        ull_to_bytes(bytes + i*4, 4, addr[i]);
    }
}  /* end addr_to_bytes() */

/**
 * @private
 * Computes PRF(key, in), for a key of PARAMSN bytes, and a 32-byte input.
 */
static int prf(word8 *out, const word8 in[32], const word8 *key)
{
    word8 buf[2 * PARAMSN + 32];

    ull_to_bytes(buf, PARAMSN, XMSS_HASH_PADDING_PRF);
    memcpy(buf + PARAMSN, key, PARAMSN);
    memcpy(buf + (2*PARAMSN), in, 32);
    core_hash(out, buf, (2*PARAMSN) + 32);
    return 0;
}  /* end prf() */

/**
 * @private
 * We assume the left half is in in[0]...in[n-1]
 */
static void thash_f(word8 *out, const word8 *in, const word8 *pub_seed,
                    word32 addr[8])
{
    word8 buf[3 * PARAMSN];
    word8 bitmask[PARAMSN];
    word8 addr_as_bytes[32];
    unsigned int i;

    /* Set the function padding. */
    ull_to_bytes(buf, PARAMSN, XMSS_HASH_PADDING_F);

    /* Generate the n-byte key. */
    set_key_and_mask(addr, 0);
    addr_to_bytes(addr_as_bytes, addr);
    prf(buf + PARAMSN, addr_as_bytes, pub_seed);

    /* Generate the n-byte mask. */
    set_key_and_mask(addr, 1);
    addr_to_bytes(addr_as_bytes, addr);
    prf(bitmask, addr_as_bytes, pub_seed);

    for (i = 0; i < PARAMSN; i++) {
        buf[2*PARAMSN + i] = in[i] ^ bitmask[i];
    }
    core_hash(out, buf, 3 * PARAMSN);
}  /* end thash_f() */

/**
 * @private
 * Helper method for pseudorandom key generation.
 * Expands an n-byte array into a len*n byte array using the `prf` function.
 */
static void expand_seed(word8 *outseeds, const word8 *inseed)
{
    word32 i;
    word8 ctr[32];

    for (i = 0; i < WOTSLEN; i++) {
        ull_to_bytes(ctr, 32, i);
        prf(outseeds + i*PARAMSN, ctr, inseed);
    }
}  /* end expand_seed() */

/**
 * @private
 * Computes the chaining function.
 * out and in have to be n-byte arrays.
 *
 * Interprets in as start-th value of the chain.
 * addr has to contain the address of the chain.
 */
static void gen_chain(word8 *out, const word8 *in,
                      unsigned int start, unsigned int steps,
                      const word8 *pub_seed, word32 addr[8])
{
    word32 i;

    /* Initialize out with the value at position 'start'. */
    memcpy(out, in, PARAMSN);

    /* Iterate 'steps' calls to the hash function. */
    for (i = start; i < (start+steps) && i < WOTSW; i++) {
        set_hash_addr(addr, i);
        thash_f(out, out, pub_seed, addr);
    }
}  /* end gen_chain() */

/**
 * @private
 * base_w algorithm as described in draft.
 * Interprets an array of bytes as integers in base w.
 * This only works when log_w is a divisor of 8.
 */
static void base_w(int *output, const int out_len, const word8 *input)
{
    int in = 0;
    int out = 0;
    word8 total;
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
}  /* end base_w() */

/**
 * @private
 * Computes the WOTS+ checksum over a message (in base_w).
*/
static void wots_checksum(int *csum_base_w, const int *msg_base_w)
{
    int csum = 0;
    word8 csum_bytes[(WOTSLEN2 * WOTSLOGW + 7) / 8];
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
}  /* end wots_checksum() */

/**
 * @private
 * Takes a message and derives the matching chain lengths.
*/
static void chain_lengths(int *lengths, const word8 *msg)
{
    base_w(lengths, WOTSLEN1, msg);
    wots_checksum(lengths + WOTSLEN1, lengths);
}

/**
 * WOTS public key generation. Takes a 32 byte seed for the private key,
 * expands it to a full WOTS private key and computes the corresponding
 * public key. It requires the seed pub_seed (used to generate bitmasks
 * and hash keys) and the address of this WOTS key pair.
 * @param pk Pointer to byte arary to place WOTS+ public key
 * @param seed Pointer to (private) seed to derive private key from
 * @param pub_seed Pointer to seed portion of public key
 * @param addr Pointer to copy of addr portion of public key
 * @warning The @a addr parameter is modified by this function.
*/
void wots_pkgen(word8 *pk, const word8 *seed,
                const word8 *pub_seed, word32 addr[8])
{
    word32 i;

    /* The WOTS+ private key is derived from the seed. */
    expand_seed(pk, seed);

    for (i = 0; i < WOTSLEN; i++) {
        set_chain_addr(addr, i);
        gen_chain(pk + i * PARAMSN, pk + i * PARAMSN,
                  0, WOTSW - 1, pub_seed, addr);
    }
}  /* end wots_pkgen() */

/**
 * WOTS+ signature generation. Takes a n-byte message, @a msg, and the
 * 32-byte @a seed for the private key to compute a signature that is
 * placed in @a sig.
 * @param sig Pointer to byte arary to place WOTS+ Signature
 * @param msg Pointer to message to sign
 * @param seed Pointer to (private) seed to derive private key from
 * @param pub_seed Pointer to seed portion of public key
 * @param addr Pointer to copy of addr portion of public key
 * @warning The @a addr parameter is modified by this function.
*/
void wots_sign(word8 *sig, const word8 *msg,
               const word8 *seed, const word8 *pub_seed,
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
}  /* end wots_sign() */

/**
 * WOTS+ key generation, from a signature. Takes a WOTS signature, @a sig,
 * and an n-byte message, @a msg; computes a WOTS public key at @a pk.
 * @param pk Pointer to byte array to write public key
 * @param sig Pointer to WOTS+ Signature to compute public key from
 * @param msg Pointer to message signed by WOTS+ Signature
 * @param pub_seed Pointer to seed portion of public key
 * @param addr Pointer to copy of addr portion of public key
 * @warning The @a addr parameter is modified by this function.
*/
void wots_pk_from_sig(word8 *pk,
                      const word8 *sig, const word8 *msg,
                      const word8 *pub_seed, word32 addr[8])
{
    int lengths[WOTSLEN];
    word32 i;

    chain_lengths(lengths, msg);

    for (i = 0; i < WOTSLEN; i++) {
        set_chain_addr(addr, i);
        gen_chain(pk + i * PARAMSN, sig + i * PARAMSN,
                  lengths[i], WOTSW - 1 - lengths[i], pub_seed, addr);
    }
}  /* end wots_pk_from_sig() */

/* end include guard */
#endif
