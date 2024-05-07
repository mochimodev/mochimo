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
 * Computes PRF(key, in), for a key of 32 bytes, and a 32-byte input.
 */
static void prf(word8 *out, const word8 in[32], const word8 *key)
{
    word8 buf2[96] = {0};

    buf2[31] = 3;
    memcpy(buf2 + 32, key, 32);
    memcpy(buf2 + 64, in, 32);
    sha256(buf2, 96, out);
}  /* end prf() */

/**
 * @private
 * We assume the left half is in in[0]...in[n-1]
 */
static void thash_f(word8 *out, const word8 *in, const word8 *pub_seed, word32 addr[8]) {
    word8 buf[96] = {0}, bitmask[32], addr_as_bytes[32];
    unsigned long to_bytes;

    for (int i = 0; i < 2; i++) {
        addr[7] = i; /* from spec "set_key_and_mask" */
        for (int j = 0; j < 8; j++) { /* convert addr longs to bytes */
            to_bytes = addr[j];
            for (int k = 3; k >= 0; k--) { /* as big-endian bytes */
                addr_as_bytes[j * 4 + k] = to_bytes & 0xff;
                to_bytes >>= 8;
            }
        }
        if (i == 0) prf(buf + 32, addr_as_bytes, pub_seed);
        else prf(bitmask, addr_as_bytes, pub_seed);
    }
    for (int i = 0; i < 32; i++) buf[64 + i] = in[i] ^ bitmask[i];
    sha256(buf, 96, out);
}

/**
 * @private
 * Helper method for pseudorandom key generation.
 * Expands an n-byte array into a len*n byte array using the `prf` function.
 */
static void expand_seed(word8 *outseeds, const word8 *inseed)
{
    word32 k;
    word8 ctr[32];

    for (int i = 0; i < 67; i++) {
        k = i;
        for (int j = 31; j >= 0; j--) { /* convert i to bytes */
            ctr[j] = k & 0xff; /* fill ctr with i bytes in BE order */
            k = k >> 8;
        }
        prf(outseeds + i*32, ctr, inseed);
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
    /* Initialize out with the value at position 'start'. */
    memcpy(out, in, 32);

    /* Iterate 'steps' calls to the hash function. */
    for (word32 i = start; i < (start+steps) && i < 16; i++) {
        addr[6] = i; /* from spec: set_hash_addr */
        thash_f(out, out, pub_seed, addr);
    }
}  /* end gen_chain() */

/**
 * @private
 * Split each byte from the input array into two 4-bit segments, storing the 
 * higher 4 bits first in the output array (little endian order). 
 */
static void base_w(int *output, const int out_len, const word8 *input) {
    int in = 0;
    for (int i = 0; i < out_len; i++) {
        if (i % 2 == 0) output[i] = (input[in] >> 4) % 16; /* evens */
        else output[i] = input[in++] & 15; /* increment on odds */
    }
}

/**
 * @private
 * Computes the WOTS+ checksum over a message (in base_w).
*/
static void wots_checksum(int *csum_base_w, const int *msg_base_w) {
    int csum = 0;
    word8 csum_bytes[2];

    for (int i = 0; i < 64; i++) csum += 15 - msg_base_w[i];

    csum <<= 4;
    csum_bytes[1] = csum & 0xff;
    csum_bytes[0] = (csum >> 8) & 0xff;
    base_w(csum_base_w, 3, csum_bytes);
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
    /* The WOTS+ private key is derived from the seed. */
    expand_seed(pk, seed);

    for (word32 i = 0; i < 67; i++) {
        addr[5] = i; /* from spec: set_chain_addr */
        gen_chain(pk + i * 32, pk + i * 32, 0, 15, pub_seed, addr);
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

    base_w(lengths, 64, msg);
    wots_checksum(lengths + 64, lengths);

    /* The WOTS+ private key is derived from the seed. */
    expand_seed(sig, seed);

    for (word32 i = 0; i < 67; i++) {
        addr[5] = i; /* from spec: set_chain_addr */
        gen_chain(sig + i * 32, sig + i * 32, 0, lengths[i], pub_seed, addr);
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

    base_w(lengths, 64, msg);
    wots_checksum(lengths + 64, lengths);

    for (word32 i = 0; i < 67; i++) {
        addr[5] = i; /* from spec: set_chain_addr */
        gen_chain(pk + i * 32, sig + i * 32, lengths[i], 15 - lengths[i], pub_seed, addr);
    }
}  /* end wots_pk_from_sig() */

/* end include guard */
#endif
