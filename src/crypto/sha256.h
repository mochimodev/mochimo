/* sha256.h  Application header for SHA2-256 algorithm
 *
 * Copyright (c) 2018 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * sha256.h is based on public domain code by Brad Conte
 * (brad AT bradconte.com).
 * https://raw.githubusercontent.com/B-Con/crypto-algorithms/master/sha256.h
 *
 * NOTE: Max message size is 512M on machines without LONG64 defined.
 *       Compile with LONG64 defined if you have a 64-bit long.
 *
 * Date: 5 January 2018
 *
*/

#ifndef SHA256_H
#define SHA256_H

#include <stddef.h>

#ifndef WORD32
#define WORD32
typedef unsigned char byte;      /* 8-bit byte */
typedef unsigned short word16;   /* 16-bit word */
typedef unsigned int word32;     /* 32-bit word  */
/* for 16-bit machines: */
/* typedef unsigned long word32;  */
#endif  /* WORD32 */

typedef struct {
    byte data[64];
    unsigned datalen;
    unsigned long bitlen;
#ifndef LONG64
    unsigned long bitlen2;
#endif
    word32 state[8];
} SHA256_CTX;

#define SHA256_BLOCK_SIZE 32     /* SHA256 outputs a byte hash[32] digest */

/* Prototypes */
void sha256_init(SHA256_CTX *ctx);
void sha256_update(SHA256_CTX *ctx, const byte data[], unsigned len);
void sha256_final(SHA256_CTX *ctx, byte hash[]);  /* hash is 32 bytes */
void sha256(const byte *in, int inlen, byte *hashout);

#endif   /* SHA256_H */
