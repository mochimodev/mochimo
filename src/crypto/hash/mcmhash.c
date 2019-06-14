/*********************************************************************
Licensing stuff
*********************************************************************/
#include "mcmhash.h"
#if CUDA_HASH
#if USE_MD2
#include "md2.cuh"
#endif
#if USE_MD5
#include "md5.cuh"
#endif
#if USE_SHA1
#include "sha1.cuh"
#endif
#if USE_SHA256
#include "sha256.cuh"
#endif
#if USE_BLAKE2B
#include "blake2b.cuh"
#endif
#endif

#if USE_MD2
void md2_hash(MD2_CTX* ctx, BYTE * in, WORD inlen, BYTE * out)
{
    md2_init(ctx);
    md2_update(ctx, in, inlen);
    md2_final(ctx, out);
}
#if CUDA_HASH
void cuda_md2_hash_batch(BYTE * in, WORD inlen, BYTE * out, WORD n_batch)
{
    mcm_cuda_md2_hash_batch(in, inlen, out, n_batch);
}
#endif

#endif
#if USE_MD5
void md5_hash(MD5_CTX* ctx, BYTE * in, WORD inlen, BYTE * out)
{
    md5_init(ctx);
    md5_update(ctx, in, inlen);
    md5_final(ctx, out);
}
#if CUDA_HASH
void cuda_md5_hash_batch(BYTE * in, WORD inlen, BYTE * out, WORD n_batch)
{
    mcm_cuda_md5_hash_batch(in, inlen, out, n_batch);
}
#endif
#endif
#if USE_SHA1
void sha1_hash(SHA1_CTX* ctx, BYTE * in, WORD inlen, BYTE * out)
{
    sha1_init(ctx);
    sha1_update(ctx, in, inlen);
    sha1_final(ctx, out);
}
#if CUDA_HASH
void cuda_sha1_hash_batch(BYTE * in, WORD inlen, BYTE * out, WORD n_batch)
{
    mcm_cuda_sha1_hash_batch(in, inlen, out, n_batch);
}
#endif
#endif
#if USE_SHA256
void sha256_hash(SHA256_CTX* ctx, BYTE * in, WORD inlen, BYTE * out)
{
    sha256_init(ctx);
    sha256_update(ctx, in, inlen);
    sha256_final(ctx, out);
}
#if CUDA_HASH
void cuda_sha256_hash_batch(BYTE * in, WORD inlen, BYTE * out, WORD n_batch)
{
    mcm_cuda_sha256_hash_batch(in, inlen, out, n_batch);
}
#endif
#endif

#if USE_BLAKE2B
void blake2b_hash(BLAKE2B_CTX* ctx, BYTE* key, WORD keylen, BYTE * in, WORD inlen, BYTE * out, WORD n_outbit)
{
    blake2b_init(ctx, key, keylen, n_outbit);
    blake2b_update(ctx, in, inlen);
    blake2b_final(ctx, out);
}
#if CUDA_HASH
void cuda_blake2b_hash_batch(BYTE* key, WORD keylen, BYTE * in, WORD inlen, BYTE * out, WORD n_outbit, WORD n_batch)
{
    mcm_cuda_blake2b_hash_batch(key, keylen, in, inlen, out, n_outbit, n_batch);
}
#endif
#endif