/*
 * keccak.h  Implementation of Keccak/SHA3 digest
 *
 * Copyright (c) 2019 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 12 June 2019
 * Revision: 1
 *
 * This file is subject to the license as found in LICENSE.PDF
 *
 */
#include <stdint.h>
#include <string.h>
#include <stdio.h>

#ifndef KECCAK
#define KECCAK

#define KECCAK_ROUND 24
#define KECCAK_STATE_SIZE 25
#define KECCAK_Q_SIZE 192

static const uint64_t KECCAK_CONSTS[] = { 0x0000000000000001, 0x0000000000008082,
0x800000000000808a, 0x8000000080008000, 0x000000000000808b, 0x0000000080000001, 0x8000000080008081,
0x8000000000008009, 0x000000000000008a, 0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
0x000000008000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003, 0x8000000000008002,
0x8000000000000080, 0x000000000000800a, 0x800000008000000a, 0x8000000080008081, 0x8000000000008080,
0x0000000080000001, 0x8000000080008008 };

typedef unsigned char byte;

typedef struct{
	
   byte sha3_flag;
   uint32_t digestbitlen;
   uint64_t rate_bits;
   uint64_t rate_bytes;
   uint64_t absorb_round;

   int64_t state[KECCAK_STATE_SIZE];
   byte q[KECCAK_Q_SIZE];

   uint64_t bits_in_queue;

 } keccak_ctx_t;

void keccak_init(keccak_ctx_t *ctx, uint32_t digestbitlen);
void keccak_sha3_init(keccak_ctx_t *ctx, uint32_t digestbitlen);
void keccak_update(keccak_ctx_t *ctx, byte *in, uint64_t inlen);
void keccak_final(keccak_ctx_t *ctx, byte *out);
uint64_t keccak_ROTL64(uint64_t a, uint64_t b);
int64_t keccak_MIN(int64_t a, int64_t b);
uint64_t keccak_UMIN(uint64_t a, uint64_t b);
uint64_t keccak_leuint64(void *in);
void keccak_absorb(keccak_ctx_t *ctx, byte* in);
void keccak_extract(keccak_ctx_t *ctx);
void keccak_pad(keccak_ctx_t *ctx);
void keccak_permutations(keccak_ctx_t * ctx);

#endif
