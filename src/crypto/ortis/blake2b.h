#include <stdint.h>
#include <string.h>
#include <stdio.h>


typedef unsigned char byte;
typedef unsigned int word32;
#define BLAKE2B_ROUNDS 12
#define BLAKE2B_BLOCK_LENGTH 128
#define BLAKE2B_CHAIN_SIZE 8
#define BLAKE2B_CHAIN_LENGTH (BLAKE2B_CHAIN_SIZE * sizeof(int64_t))
#define BLAKE2B_STATE_SIZE 16
#define BLAKE2B_STATE_LENGTH (BLAKE2B_STATE_SIZE * sizeof(int64_t))


static const uint64_t BLAKE2B_IVS[8] =
{
	0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b,
	0xa54ff53a5f1d36f1, 0x510e527fade682d1, 0x9b05688c2b3e6c1f,
	0x1f83d9abfb41bd6b, 0x5be0cd19137e2179
};

static const unsigned char BLAKE2B_SIGMAS[12][16] =
{
	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
	{ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
	{ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
	{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
	{ 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 }
};

typedef struct
{
	uint32_t digestlen;
	byte *key;
	uint32_t keylen;

	byte buff[BLAKE2B_BLOCK_LENGTH];
	int64_t chain[BLAKE2B_CHAIN_SIZE];
	int64_t state[BLAKE2B_STATE_SIZE];
	
	uint32_t pos;
	uint64_t t0;
	uint64_t t1;
	uint64_t f0;

} blake2b_ctx_t;

void blake2b_init(blake2b_ctx_t *ctx, byte* key, uint32_t keylen, uint32_t digestbitlen);
void blake2b_update(blake2b_ctx_t *ctx, byte* in, uint64_t inlen);
void blake2b_final(blake2b_ctx_t *ctx, byte* out);
void blake2b_init_state(blake2b_ctx_t *ctx);
void blake2b_compress(blake2b_ctx_t *ctx, byte* in, uint32_t inoffset);
uint64_t blake2b_leuint64(byte *in);
uint64_t blake2b_ROTR64(uint64_t a, uint8_t b);
void blake2b_G(blake2b_ctx_t *ctx, int64_t m1, int64_t m2, int32_t a, int32_t b, int32_t c, int32_t d);
