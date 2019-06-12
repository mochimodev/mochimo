/*********************************************************************
Licensing stuff
*********************************************************************/
#include "mcmhash.h"
static void test_md2()
{
    // TEST MD2
    BYTE md2_inp[127*1024];
    BYTE md2_oup[1024*MD2_BLOCK_SIZE];
    srand(0);
    for (int i = 0; i < 1024; i++)
    {
        md2_inp[i] = rand() % 256;
    }
    // CPU hash
    for (int i = 0; i < 1024; i++)
    {
        MD2_CTX ctx;
        md2_hash(&ctx, md2_inp + 127 * i, 127, md2_oup + MD2_BLOCK_SIZE * i);
    }

    BYTE md2_cu_oup[1024*MD2_BLOCK_SIZE];
    cuda_md2_hash_batch(md2_inp, 127, md2_cu_oup, 1024);
    if (memcmp(md2_oup, md2_cu_oup, MD2_BLOCK_SIZE * 1024) != 0)
    {
        printf("Failed test MD2 \n");
    }
    else
    {
        printf("Passed test MD2 \n");
    }
}

static void test_md5()
{
    // TEST MD5
    BYTE md5_inp[127*1024];
    BYTE md5_oup[1024*MD5_BLOCK_SIZE];
    srand(0);
    for (int i = 0; i < 1024; i++)
    {
        md5_inp[i] = rand() % 256;
    }
    // CPU hash
    for (int i = 0; i < 1024; i++)
    {
        MD5_CTX ctx;
        md5_hash(&ctx, md5_inp + 127 * i, 127, md5_oup + MD5_BLOCK_SIZE * i);
    }

    BYTE md5_cu_oup[1024*MD5_BLOCK_SIZE];
    cuda_md5_hash_batch(md5_inp, 127, md5_cu_oup, 1024);

    if (memcmp(md5_oup, md5_cu_oup, MD5_BLOCK_SIZE * 1024) != 0)
    {
        printf("Failed test MD5 \n");
    }
    else
    {
        printf("Passed test MD5 \n");
    }
}

static void test_sha1()
{
    // TEST SHA1
    BYTE sha1_inp[127*1024];
    BYTE sha1_oup[1024*SHA1_BLOCK_SIZE];
    srand(0);
    for (int i = 0; i < 1024; i++)
    {
        sha1_inp[i] = rand() % 256;
    }
    // CPU hash
    for (int i = 0; i < 1024; i++)
    {
        SHA1_CTX ctx;
        sha1_hash(&ctx, sha1_inp + 127 * i, 127, sha1_oup + SHA1_BLOCK_SIZE * i);
    }

    BYTE sha1_cu_oup[1024*SHA1_BLOCK_SIZE];
    cuda_sha1_hash_batch(sha1_inp, 127, sha1_cu_oup, 1024);

    if (memcmp(sha1_oup, sha1_cu_oup, SHA1_BLOCK_SIZE * 1024) != 0)
    {
        printf("Failed test SHA1 \n");
    }
    else
    {
        printf("Passed test SHA1 \n");
    }
}

static void test_sha256()
{
    // TEST SHA256
    BYTE sha256_inp[127*1024];
    BYTE sha256_oup[1024*SHA256_BLOCK_SIZE];
    srand(0);
    for (int i = 0; i < 1024; i++)
    {
        sha256_inp[i] = rand() % 256;
    }
    // CPU hash
    for (int i = 0; i < 1024; i++)
    {
        SHA256_CTX ctx;
        sha256_hash(&ctx, sha256_inp + 127 * i, 127, sha256_oup + SHA256_BLOCK_SIZE * i);
    }

    BYTE sha256_cu_oup[1024*SHA256_BLOCK_SIZE];
    cuda_sha256_hash_batch(sha256_inp, 127, sha256_cu_oup, 1024);

    if (memcmp(sha256_oup, sha256_cu_oup, SHA256_BLOCK_SIZE * 1024) != 0)
    {
        printf("Failed test SHA256 \n");
    }
    else
    {
        printf("Passed test SHA256 \n");
    }
}

int main()
{
    test_md2();
    test_md5();
    test_sha1();
    test_sha256();

    return 0;
}