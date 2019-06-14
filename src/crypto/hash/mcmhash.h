#ifndef MCMHASHLIB_MCMHASH_H
#define MCMHASHLIB_MCMHASH_H
#include "config.h"

#if USE_MD2
    #include "md2.h"
#endif
#if USE_MD5
    #include "md5.h"
#endif
#if USE_SHA1
    #include "sha1.h"
#endif
#if USE_SHA256
#include "sha256.h"
#endif

#if USE_BLAKE2B
#include "blake2b.h"
#endif


#if OCL_HASH
#endif

// API
#ifdef _WIN32
#  define MCM_Hashlib_API __declspec(dllexport)
#else
#define MCM_Hashlib_API __attribute__((visibility("default")))
#endif

#if USE_MD2
MCM_Hashlib_API void md2_hash(MD2_CTX* ctx, BYTE * in, WORD inlen, BYTE * out);
    #if CUDA_HASH
MCM_Hashlib_API void cuda_md2_hash_batch(BYTE * in, WORD inlen, BYTE * out, WORD n_batch);
    #endif
#endif
#if USE_MD5
MCM_Hashlib_API void md5_hash(MD5_CTX* ctx, BYTE * in, WORD inlen, BYTE * out);
#endif
#if CUDA_HASH
MCM_Hashlib_API void cuda_md5_hash_batch(BYTE * in, WORD inlen, BYTE * out, WORD n_batch);
#endif

#if USE_SHA1
MCM_Hashlib_API void sha1_hash(SHA1_CTX* ctx, BYTE * in, WORD inlen, BYTE * out);
#if CUDA_HASH
MCM_Hashlib_API void cuda_sha1_hash_batch(BYTE * in, WORD inlen, BYTE * out, WORD n_batch);
#endif
#endif
#if USE_SHA256
MCM_Hashlib_API void sha256_hash(SHA256_CTX* ctx, BYTE * in, WORD inlen, BYTE * out);
#if CUDA_HASH
MCM_Hashlib_API void cuda_sha256_hash_batch(BYTE * in, WORD inlen, BYTE * out, WORD n_batch);
#endif
#endif
#if USE_BLAKE2B
MCM_Hashlib_API void blake2b_hash(BLAKE2B_CTX* ctx, BYTE* key, WORD keylen, BYTE * in, WORD inlen, BYTE * out, WORD n_outbit);
#if CUDA_HASH
MCM_Hashlib_API void cuda_blake2b_hash_batch(BYTE* key, WORD keylen, BYTE * in, WORD inlen, BYTE * out, WORD n_outbit, WORD n_batch);
#endif
#endif
#endif //MCMHASHLIB_MCMHASH_H
