/**
 * @private
 * @headerfile peach.cuh <peach.cuh>
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
 * @note A note on variable naming in this file...
 * [2] suffix implies dual stream variables; for maximum GPU utilisation
 * h_ prefix implies host memory; requires cudaMallocHost()
 * c_ prefix implies (constant) device memory
 * g_ prefix implies (global) device memory
 * d_ prefix implies device memory
 * *h_ pointers require cudaMallocHost();
 * *d_/*g_ pointers require cudaMalloc();
*/

/* include guard */
#ifndef MOCHIMO_PEACH_CU
#define MOCHIMO_PEACH_CU


#include "extint.h"
#include "extmath.h"
#include "extprint.h"

#include "peach.cuh"

/* hashing functions used by Peach's nighthash */
#include "blake2b.cu"
#include "md2.cu"
#include "md5.cu"
#include "sha1.cu"
#include "sha256.cu"
#include "sha3.cu"

/**
 * @private
 * Definitions for embedding strings.
*/
#define cuSTRING(x) #x
#define cuSTR(x) cuSTRING(x)

/**
 * @private
 * Peach specific cuda error checking definition.
 * Destroys/frees resources on error.
*/
#define cuCHK(_cmd, _dev, _exec) \
   do { \
      cudaError_t _cerr = _cmd; \
      if (_cerr != cudaSuccess) { \
         int _n; cudaGetDevice(&_n); \
         const char *_err = cudaGetErrorString(_cerr); \
         pfatal("CUDA#%d->%s: %s", _n, cuSTR(_cmd), _err); \
         if (_dev != NULL) { \
            cudaDeviceSynchronize(); \
            DEVICE_CTX *_d = _dev; \
            _d->status = DEV_FAIL; \
            PEACH_CUDA_CTX *_p = &PeachCudaCTX[_d->id]; \
            if (_p->stream[0]) cudaStreamDestroy(_p->stream[0]); \
            if (_p->stream[1]) cudaStreamDestroy(_p->stream[1]); \
            if (_p->h_solve) cudaFreeHost(_p->h_solve); \
            if (_p->h_ictx) cudaFreeHost(_p->h_ictx); \
            if (_p->d_solve[0]) cudaFree(_p->d_solve[0]); \
            if (_p->d_solve[1]) cudaFree(_p->d_solve[1]); \
            if (_p->d_ictx[0]) cudaFree(_p->d_ictx[0]); \
            if (_p->d_ictx[1]) cudaFree(_p->d_ictx[1]); \
            if (_p->d_map) cudaFree(_p->d_map); \
         } \
         _exec; \
      } \
   } while(0)

/**
 * @private
 * Peach CUDA context. Managed internally by cross referencing parameters
 * of DEVICE_CTX passed to functions.
*/
typedef struct {
   nvmlDevice_t nvml_device;           /**< nvml device for monitoring */
   cudaStream_t stream[2];             /**< asynchronous streams */
   SHA256_CTX *h_ictx[2], *d_ictx[2];  /**< sha256 ictx lists */
   BTRAILER *h_bt[2];                  /**< BTRAILER (current) */
   word8 *h_solve[2], *d_solve[2];     /**< solve seeds */
   word8 *d_map;                       /**< Peach Map */
   int nvml_enabled;                   /**< Flags NVML capable */
} PEACH_CUDA_CTX;


/* pointer to peach CUDA context/s */
static PEACH_CUDA_CTX *PeachCudaCTX;
/* host phash and diff (paged memory txfer) */
static word8 *h_phash, *h_diff;
/* device symbol memory (unique per device) */
__device__ __constant__ static word8 __align__(32) c_phash[SHA256LEN];
__device__ __constant__ static word8 __align__(32) c_diff;

/**
 * @private
 * Perform deterministic (single precision) floating point operations on
 * @a len bytes of @a data (in 4 byte operations).
 * @param data Pointer to data to use in operations
 * @param len Length of @a data to use in operations
 * @param index Peach tile index number
 * @param txf Flag indicates @a data should be transformed by operations
 * @returns 32-bit unsigned operation code for subsequent Peach algo steps
 * @note Operations are guaranteed "deterministic" within the Peach
 * algorithm for all IEEE-754 compliant hardware on "round-to-nearest"
 * rounding mode. This is ensured by the use of CUDA intrinsics:
 * - __fdiv_rn(), __fmul_rn(), __fsub_rn(), __fadd_rn() operations, and
 * - __int2float_rn(), __uint2float_rn() conversions
*/
__device__ static word32 cu_peach_dflops(void *data, size_t len,
   word32 index, int txf)
{
   float *flp, temp, flv;
   word8 *bp, shift;
   int32 operand;
   word32 op;
   unsigned i;

   /* process entire length of input data; limit to 4 byte multiples */
   /* len = len - (len & 3); // uncomment if (len % 4 != 0) is expected */
   for (op = i = 0; i < len; i += 4) {
      bp = &((word8 *) data)[i];
      if (txf) {
         /* input data is modified directly */
         flp = (float *) bp;
      } else {
         /* temp variable is modified, input data is unchanged */
         temp = *((float *) bp);
         flp = &temp;
      }
      /* first byte allocated to determine shift amount */
      shift = ((*bp & 7) + 1) << 1;
      /* remaining bytes are selected for 3 different operations based on
       * the first bytes resulting shift on precomputed contants to...
       * ... 1) determine the floating point operation type */
      op += bp[((WORD32_C(0x26C34) >> shift) & 3)];
      /* ... 2) determine the value of the operand */
      operand = bp[((WORD32_C(0x14198) >> shift) & 3)];
      /* ... 3) determine the upper most bit of the operand
       *        NOTE: must be performed AFTER the allocation of the operand */
      if (bp[((WORD32_C(0x3D6EC) >> shift) & 3)] & 1) {
         operand ^= WORD32_C(0x80000000);
      }
      /* interpret operand as SIGNED integer and cast to float */
      flv = __int2float_rn(operand);
      /* Replace pre-operation NaN with index */
      if (isnan(*flp)) *flp = __uint2float_rn(index);
      /* Perform predetermined floating point operation */
      switch (op & 3) {
         case 3: *flp = __fdiv_rn(*flp, flv);  break;
         case 2: *flp = __fmul_rn(*flp, flv);  break;
         case 1: *flp = __fsub_rn(*flp, flv);  break;
         case 0: *flp = __fadd_rn(*flp, flv);  break;
      }
      /* Replace post-operation NaN with index */
      if (isnan(*flp)) *flp = __uint2float_rn(index);
      /* Add result of the operation to `op` as an array of bytes */
      bp = (word8 *) flp;
      op += bp[0];
      op += bp[1];
      op += bp[2];
      op += bp[3];
   }  /* end for(i = 0; ... */

   return op;
}  /* end cu_peach_dflops() */

/**
 * @private
 * Perform deterministic memory transformations on @a len bytes of @a data.
 * @param data Pointer to data to use in operations
 * @param len Length of @a data to use in operations
 * @param op Operating code from previous Peach algo steps
 * @returns 32-bit unsigned operation code for subsequent Peach algo steps
*/
__device__ static word32 cu_peach_dmemtx(void *data, size_t len, word32 op)
{
   unsigned i, z;
   size_t len16, len32, len64, y;
   word64 *qp;
   word32 *dp;
   word8 *bp, temp;

   /* prepare memory pointers and lengths */
   qp = (word64 *) data;
   dp = (word32 *) data;
   bp = (word8 *) data;
   len64 = len >> 3;
   len32 = len >> 2;
   len16 = len >> 1;
   /* perform memory transformations multiple times */
   for (i = 0; i < PEACHROUNDS; i++) {
      /* determine operation to use for this iteration */
      op += bp[i & 31];
      /* select "random" transformation based on value of `op` */
      switch (op & 7) {
         case 0:  /* flip the first and last bit in every byte */
            for (z = 0; z < len64; z++) qp[z] ^= WORD64_C(0x8181818181818181);
            for (z <<= 1; z < len32; z++) dp[z] ^= WORD32_C(0x81818181);
            break;
         case 1:  /* Swap bytes */
            for (y = len16, z = 0; z < len16; y++, z++) {
               temp = bp[z]; bp[z] = bp[y]; bp[y] = temp;
            }
            break;
         case 2:  /* 1's complement, all bytes */
            for (z = 0; z < len64; z++) qp[z] = ~qp[z];
            for (z <<= 1; z < len32; z++) dp[z] = ~dp[z];
            break;
         case 3:  /* Alternate +1 and -1 on all bytes */
            for (z = 0; z < len; z++) bp[z] += (z & 1) ? -1 : 1;
            break;
         case 4:  /* Alternate -i and +i on all bytes */
            for (z = 0; z < len; z++) bp[z] += (word8) ((z & 1) ? i : -i);
            break;
         case 5:  /* Replace every occurrence of 104 with 72 */ 
            for (z = 0; z < len; z++) if(bp[z] == 104) bp[z] = 72;
            break;
         case 6:  /* If byte a is > byte b, swap them. */
            for (y = len16, z = 0; z < len16; y++, z++) {
               if(bp[z] > bp[y]) {
                  temp = bp[z]; bp[z] = bp[y]; bp[y] = temp;
               }
            }
            break;
         case 7:  /* XOR all bytes */
            for (y = 0, z = 1; z < len; y++, z++) bp[z] ^= bp[y];
            break;
      } /* end switch(op & 7)... */
   } /* end for(i = 0; ... */

   return op;
}  /* end cu_peach_dmemtx() */

/**
 * @private
 * Perform Nighthash on @a inlen bytes of @a in and place result in @a out.
 * Utilizes deterministic float operations and memory transformations.
 * @param in Pointer to input data
 * @param inlen Length of data from @a in, used in non-transform steps
 * @param index Peach tile index number
 * @param txlen Length of data from @a in, used in transform steps
 * @param out Pointer to location to place resulting hash
*/
__device__ static void cu_peach_nighthash(void *in, size_t inlen,
   word32 index, size_t txlen, void *out)
{
   static const word64 key32B[4] = { 0, 0, 0, 0 };
   static const word64 key64B[8] = {
      WORD64_C(0x0101010101010101), WORD64_C(0x0101010101010101),
      WORD64_C(0x0101010101010101), WORD64_C(0x0101010101010101),
      WORD64_C(0x0101010101010101), WORD64_C(0x0101010101010101),
      WORD64_C(0x0101010101010101), WORD64_C(0x0101010101010101),
   };

   /* Perform flops to determine initial algo type.
    * When txlen is non-zero the transformation of input data is enabled,
    * as well as the additional memory transformation process. */
   if (txlen) {
      index = cu_peach_dflops(in, txlen, index, 1);
      index = cu_peach_dmemtx(in, txlen, index);
   } else index = cu_peach_dflops(in, inlen, index, 0);

   /* reduce algorithm selection to 1 of 8 choices */
   switch (index & 7) {
      case 0: cu_blake2b(in, inlen, key32B, 32, out, BLAKE2BLEN256); break;
      case 1: cu_blake2b(in, inlen, key64B, 64, out, BLAKE2BLEN256); break;
      case 2: {
         cu_sha1(in, inlen, out);
         /* SHA1 hash is only 20 bytes long, zero fill remaining... */
         ((word32 *) out)[5] = 0;
         ((word64 *) out)[3] = 0;
         break;
      }
      case 3: cu_sha256(in, inlen, out); break;
      case 4: cu_sha3(in, inlen, out, SHA3LEN256); break;
      case 5: cu_keccak(in, inlen, out, KECCAKLEN256); break;
      case 6: {
         cu_md2(in, inlen, out);
         /* MD2 hash is only 16 bytes long, zero fill remaining... */
         ((word64 *) out)[2] = 0;
         ((word64 *) out)[3] = 0;
         break;
      }
      case 7: {
         cu_md5(in, inlen, out);
         /* MD5 hash is only 16 bytes long, zero fill remaining... */
         ((word64 *) out)[2] = 0;
         ((word64 *) out)[3] = 0;
         break;
      }
   }  /* end switch(algo_type)... */
}  /* end cu_peach_nighthash() */

/**
 * @private
 * Generate a tile of the Peach map.
 * @param index Index number of tile to generate
 * @param tilep Pointer to location to place generated tile
*/
__device__ static void cu_peach_generate(word32 index, word8 *tilep)
{
   int i;

   /* place initial data into seed */
   memcpy(tilep, &index, 4);
   memcpy(tilep + 4, c_phash, SHA256LEN);
   /* perform initial nighthash into first row of tile */
   cu_peach_nighthash(tilep, PEACHGENLEN, index, PEACHGENLEN, tilep);
   /* fill the rest of the tile with the preceding Nighthash result */
   for (i = 0; i < 992; i += 32) {
      memcpy(tilep + i + 32, &index, 4);
      cu_peach_nighthash(&tilep[i], PEACHGENLEN, index, SHA256LEN,
         &tilep[i + 32]);
   }
}  /* end cu_peach_generate() */

/**
 * @private
 * Perform an index jump using the hash result of the Nighthash function.
 * @param index Index number of (current) tile on Peach map
 * @param nonce Nonce for use as entropy in jump direction
 * @param tilep Pointer to tile data at @a index
 * @returns 32-bit unsigned index of next tile
*/
__device__ static void cu_peach_jump(word32 *index, word8 *nonce,
   word8 *tilep)
{
   __align__(32) word32 dhash[SHA256LEN / 4];
   __align__(32) word8 seed[PEACHJUMPLEN];

   /* construct seed for use as Nighthash input for this index on the map */
   memcpy(seed, nonce, HASHLEN);
   memcpy(seed + HASHLEN, index, 4);
   memcpy(seed + HASHLEN + 4, tilep, PEACHTILELEN);
   /* perform nighthash on PEACHJUMPLEN bytes of seed */
   cu_peach_nighthash(seed, PEACHJUMPLEN, *index, 0, dhash);
   /* sum hash as 8x 32-bit unsigned integers */
   *index = (
      dhash[0] + dhash[1] + dhash[2] + dhash[3] +
      dhash[4] + dhash[5] + dhash[6] + dhash[7]
   ) & PEACHCACHELEN_M1;
}  /* end cu_peach_jump() */

/**
 * CUDA kernel for bulk generation of Peach Map tiles.
 * @param d_map Device pointer to location of Peach Map
 * @param offset Index number offset to generate tiles from
 */
__global__ void kcu_peach_build(word8 *d_map, word32 offset)
{
   const word32 index = blockDim.x * blockIdx.x + threadIdx.x + offset;
   if (index < PEACHCACHELEN) {
      cu_peach_generate(index, &d_map[index * PEACHTILELEN]);
   }
}  /* end kcu_peach_build() */

/**
 * CUDA kernel for solving a tokenized haiku as nonce output for Peach proof
 * of work. Combine haiku protocols implemented in the Trigg Algorithm with
 * the memory intensive protocols of the Peach algorithm to generate haiku
 * output as proof of work.
 * @param d_map Device pointer to Peach Map
 * @param d_ictx Device pointer to incomplete hashing contexts
 * @param d_solve Device pointer to location to place nonce on solve
*/
__global__ void kcu_peach_solve(word8 *d_map, SHA256_CTX *d_ictx,
   word8 *d_solve)
{
   word8 __align__(32) hash[SHA256LEN];
   word8 __align__(32) nonce[32];
   SHA256_CTX *ictx;
   word32 *x, mario;
   int i;

   /* shift ictx to appropriate location and extract nonce */
   ictx = &d_ictx[(blockIdx.x * blockDim.x) + threadIdx.x];
   memcpy(nonce, ictx->data + 28, 32);
   /* finalise incomplete sha256 hash */
   cu_sha256_final(ictx, hash);
   /* initialize mario's starting index on the map, bound to PEACHCACHELEN */
   for(mario = hash[0], i = 1; i < SHA256LEN; i++) mario *= hash[i];
   mario &= PEACHCACHELEN_M1;
   /* perform tile jumps to find the final tile x8 */
   cu_peach_jump(&mario, nonce, &d_map[mario * PEACHTILELEN]);
   cu_peach_jump(&mario, nonce, &d_map[mario * PEACHTILELEN]);
   cu_peach_jump(&mario, nonce, &d_map[mario * PEACHTILELEN]);
   cu_peach_jump(&mario, nonce, &d_map[mario * PEACHTILELEN]);
   cu_peach_jump(&mario, nonce, &d_map[mario * PEACHTILELEN]);
   cu_peach_jump(&mario, nonce, &d_map[mario * PEACHTILELEN]);
   cu_peach_jump(&mario, nonce, &d_map[mario * PEACHTILELEN]);
   cu_peach_jump(&mario, nonce, &d_map[mario * PEACHTILELEN]);
   /* hash block trailer with final tile */
   cu_sha256_init(ictx);
   cu_sha256_update(ictx, hash, SHA256LEN);
   cu_sha256_update(ictx, &d_map[mario * PEACHTILELEN], PEACHTILELEN);
   cu_sha256_final(ictx, hash);
   /* Coarse/Fine evaluation checks */
   x = (word32 *) hash;
   for(i = c_diff >> 5; i; i--) if(*(x++) != 0) return;
   if(__clz(__byte_perm(*x, 0, 0x0123)) < (c_diff & 31)) return;

   /* check first to solve with atomic solve handling */
   if(!atomicCAS((int *) d_solve, 0, *((int *) nonce))) {
      memcpy(d_solve, nonce, 32);
   }
}  /* end kcu_peach_solve() */

/**
 * CUDA kernel for checking Peach Proof-of-Work. The haiku must be
 * syntactically correct AND have the right vibe. Also, entropy MUST match
 * difficulty.
 * @param ictx Device pointer to incomplete hashing context
 * @param out Pointer to location to place final hash
 * @param eval Evaluation result: VEOK on success, else VERROR
*/
__global__ void kcu_peach_checkhash(SHA256_CTX *ictx, word8 *out,
   word8 *eval)
{
   __align__(32) word8 tile[PEACHTILELEN] = { 0 };
   __align__(32) word8 hash[SHA256LEN] = { 0 };
   __align__(32) word8 nonce[32] = { 0 };
   word32 *x, mario;
   int i;

   /* restricted to a single thread for debug purposes */
   if ((blockDim.x * blockIdx.x) + threadIdx.x > 0) return;

   /* extract nonce */
   memcpy(nonce, ictx->data + 28, 32);
   /* finalise incomplete sha256 hash */
   cu_sha256_final(ictx, hash);
   /* initialize mario's starting index on the map, bound to PEACHCACHELEN */
   for(mario = hash[0], i = 1; i < SHA256LEN; i++) mario *= hash[i];
   mario &= PEACHCACHELEN_M1;
   /* generate and perform tile jumps to find the final tile x8 */
   cu_peach_generate(mario, tile);
   cu_peach_jump(&mario, nonce, tile);
   cu_peach_generate(mario, tile);
   cu_peach_jump(&mario, nonce, tile);
   cu_peach_generate(mario, tile);
   cu_peach_jump(&mario, nonce, tile);
   cu_peach_generate(mario, tile);
   cu_peach_jump(&mario, nonce, tile);
   cu_peach_generate(mario, tile);
   cu_peach_jump(&mario, nonce, tile);
   cu_peach_generate(mario, tile);
   cu_peach_jump(&mario, nonce, tile);
   cu_peach_generate(mario, tile);
   cu_peach_jump(&mario, nonce, tile);
   cu_peach_generate(mario, tile);
   cu_peach_jump(&mario, nonce, tile);
   /* generate the last tile */
   cu_peach_generate(mario, tile);
   /* hash block trailer with final tile */
   cu_sha256_init(ictx);
   cu_sha256_update(ictx, hash, SHA256LEN);
   cu_sha256_update(ictx, tile, PEACHTILELEN);
   cu_sha256_final(ictx, hash);
   /* pass final hash to out */
   memcpy(out, hash, SHA256LEN);
   /* Coarse/Fine evaluation checks */
   *eval = 1;
   x = (word32 *) hash;
   for(i = c_diff >> 5; i; i--) if(*(x++) != 0) return;
   if(__clz(__byte_perm(*x, 0, 0x0123)) < (c_diff & 31)) return;
   *eval = 0;
}  /* end kcu_peach_checkhash() */

/**
 * Check Peach proof of work algorithm with a CUDA device.
 * @param bt Pointer to block trailer to check
 * @param diff Difficulty to test against entropy of final hash
 * @param out Pointer to location to place final hash, if non-null
 * @returns VEOK on success, else VERROR
*/
int peach_checkhash_cuda(BTRAILER *btp, word8 diff, void *out)
{
   SHA256_CTX *d_ictx, ictx;
   word8 *d_hash, *d_eval;
   word8 eval = 0;
   int count;

   cuCHK(cudaGetDeviceCount(&count), NULL, return (-1));
   if (count < 1) {
      pfatal("No CUDA devices...");
      return -1;
   }
   cuCHK(cudaSetDevice(0), NULL, return (-1));
   cuCHK(cudaMalloc(&d_ictx, sizeof(SHA256_CTX)), NULL, return (-1));
   cuCHK(cudaMalloc(&d_hash, SHA256LEN), NULL, return (-1));
   cuCHK(cudaMalloc(&d_eval, 1), NULL, return (-1));
   cuCHK(cudaMemset(d_eval, 0xff, 1), NULL, return (-1));
   /* prepare intermediate state for next round */
   sha256_init(&ictx);
   sha256_update(&ictx, btp, 124);
   /* transfer phash to device */
   cuCHK(cudaMemcpyToSymbol(c_diff, btp->difficulty, 1, 0,
      cudaMemcpyHostToDevice), NULL, return (-1));
   /* transfer phash to device */
   cuCHK(cudaMemcpyToSymbol(c_phash, btp->phash, SHA256LEN, 0,
      cudaMemcpyHostToDevice), NULL, return (-1));
   /* transfer ictx to device */
   cuCHK(cudaMemcpy(d_ictx, &ictx, sizeof(SHA256_CTX),
      cudaMemcpyHostToDevice), NULL, return (-1));
   /* launch kernel to check Peach */
   kcu_peach_checkhash<<<1, 1>>>(d_ictx, d_hash, d_eval);
   cuCHK(cudaGetLastError(), NULL, return (-1));
   /* retrieve hash/eval data */
   cuCHK(cudaMemcpy(out, d_hash, SHA256LEN,
      cudaMemcpyDeviceToHost), NULL, return (-1));
   cuCHK(cudaMemcpy(&eval, d_eval, 1,
      cudaMemcpyDeviceToHost), NULL, return (-1));
   /* wait for device to finish */
   cuCHK(cudaDeviceSynchronize(), NULL, return (-1));
   /* free memory */
   cuCHK(cudaFree(d_ictx), NULL, return (-1));
   cuCHK(cudaFree(d_hash), NULL, return (-1));
   cuCHK(cudaFree(d_eval), NULL, return (-1));
   /* return */
   return (int) eval;
}  /* end peach_checkhash_cuda() */

/**
 * (re)Initialize a device context with a CUDA device.
 * @param devp Pointer to DEVICE_CTX to initialize
 * @param id Index of CUDA device to initialize to DEVICE_CTX
 * @returns VEOK on success, else VERROR
 * @note The `id` parameter of the DEVICE_CTX must be set to an appropriate
 * CUDA device number. If not performing a re-initialization, recommend
 * using peach_init_cuda() first.
*/
int peach_init_cuda_device(DEVICE_CTX *devp, int id)
{
   static int nvml_initialized = 0;
   static unsigned nvml_count = 0;

   struct cudaDeviceProp props;
   nvmlPciInfo_t pci;
   nvmlDevice_t *nvmlp;
   size_t ictxlen;
   unsigned i, gen, width;

   if (nvml_initialized == 0) {
      /* set nvml initialized */
      nvml_initialized = 1;
      /* initialize nvml */
      if (nvmlInit() != NVML_SUCCESS) {
         nvml_count = pdebug("Unable to initialize NVML");
         plog("No NVML devices detected...");
      } else if (nvmlDeviceGetCount(&nvml_count) != NVML_SUCCESS) {
         nvml_count = pdebug("Unable to obtain NVML count");
         plog("No NVML devices detected...");
      }
   }

   devp->id = id;
   devp->type = CUDA_DEVICE;
   nvmlp = &(PeachCudaCTX[id].nvml_device);
   /* get CUDA properties for verification with nvml device */
   if (cudaGetDeviceProperties(&props, id) != cudaSuccess) {
      perr("cudaGetDeviceProperties(%d)", id);
   } else {
      /* store GPU name - "NVIDIA" may be dropped on LINUX OS */
      if (strncmp(props.name, "NVIDIA", 6)) {
         strncpy(devp->name, props.name, sizeof(devp->name));
      } else strncpy(devp->name, props.name + 7, sizeof(devp->name));
      /* scan nvml devices for match */
      for (i = 0; i < nvml_count; i++) {
         if (nvmlDeviceGetHandleByIndex(i, nvmlp) != NVML_SUCCESS ||
            (nvmlDeviceGetPciInfo(*nvmlp, &pci) != NVML_SUCCESS) ||
            (pci.device != props.pciDeviceID) ||
            (pci.domain != props.pciDomainID) ||
            (pci.bus != props.pciBusID)) {
            /* clear nvmlDev property */
            memset(nvmlp, 0, sizeof(nvmlDevice_t));
            PeachCudaCTX[id].nvml_enabled = 0;
            continue;
         }
         /* obtain link gen/width and add to id */
         if (nvmlDeviceGetCurrPcieLinkGeneration(*nvmlp, &gen)
               != NVML_SUCCESS) gen = 0;
         if (nvmlDeviceGetCurrPcieLinkWidth(*nvmlp, &width)
               != NVML_SUCCESS) width = 0;
         snprintf(devp->linkId, sizeof(devp->linkId), "Gen%ux%02u#%u",
            gen, width, pci.device);
         PeachCudaCTX[id].nvml_enabled = 1;
         break;
      }
   }
   /* set context to CUDA id */
   cuCHK(cudaSetDevice(id), devp, return VERROR);
   /* set CUDA configuration for device */
   if (cudaOccupancyMaxPotentialBlockSize(&(devp->grid), &(devp->block),
         kcu_peach_solve, 0, 0) != cudaSuccess) {
      pdebug("cudaOccupancy~BlockSize(%d) failed...", id);
      pdebug("Using conservative defaults for <<<512/128>>>");
      devp->grid = 512;
      devp->block = 128;
   }
   /* calculate total threads and ictxlist size */
   devp->threads = devp->grid * devp->block;
   ictxlen = sizeof(SHA256_CTX) * devp->threads;
   /* create streams for device */
   cuCHK(cudaStreamCreate(&(PeachCudaCTX[id].stream[0])), devp, return VERROR);
   cuCHK(cudaStreamCreate(&(PeachCudaCTX[id].stream[1])), devp, return VERROR);
   /* allocate pinned host memory for host/device transfers */
   cuCHK(cudaMallocHost(&(PeachCudaCTX[id].h_solve[0]), 32), devp, return VERROR);
   cuCHK(cudaMallocHost(&(PeachCudaCTX[id].h_solve[1]), 32), devp, return VERROR);
   cuCHK(cudaMallocHost(&(PeachCudaCTX[id].h_ictx[0]), ictxlen), devp, return VERROR);
   cuCHK(cudaMallocHost(&(PeachCudaCTX[id].h_ictx[1]), ictxlen), devp, return VERROR);
   cuCHK(cudaMallocHost(&(PeachCudaCTX[id].h_bt[0]), sizeof(BTRAILER)), devp, return VERROR);
   cuCHK(cudaMallocHost(&(PeachCudaCTX[id].h_bt[1]), sizeof(BTRAILER)), devp, return VERROR);
   /* allocate device memory for host/device transfers */
   cuCHK(cudaMalloc(&(PeachCudaCTX[id].d_solve[0]), 32), devp, return VERROR);
   cuCHK(cudaMalloc(&(PeachCudaCTX[id].d_solve[1]), 32), devp, return VERROR);
   cuCHK(cudaMalloc(&(PeachCudaCTX[id].d_ictx[0]), ictxlen), devp, return VERROR);
   cuCHK(cudaMalloc(&(PeachCudaCTX[id].d_ictx[1]), ictxlen), devp, return VERROR);
   /* allocate memory for Peach map on device */
   cuCHK(cudaMalloc(&(PeachCudaCTX[id].d_map), PEACHMAPLEN), devp, return VERROR);
   /* clear device/host allocated memory */
   cudaMemsetAsync(PeachCudaCTX[id].d_ictx[0], 0, ictxlen, cudaStreamDefault);
   cudaMemsetAsync(PeachCudaCTX[id].d_ictx[1], 0, ictxlen, cudaStreamDefault);
   cudaMemsetAsync(PeachCudaCTX[id].d_solve[0], 0, 32, cudaStreamDefault);
   cudaMemsetAsync(PeachCudaCTX[id].d_solve[1], 0, 32, cudaStreamDefault);
   memset(PeachCudaCTX[id].h_bt[0], 0, sizeof(BTRAILER));
   memset(PeachCudaCTX[id].h_bt[1], 0, sizeof(BTRAILER));
   memset(PeachCudaCTX[id].h_ictx[0], 0, ictxlen);
   memset(PeachCudaCTX[id].h_ictx[1], 0, ictxlen);
   memset(PeachCudaCTX[id].h_solve[0], 0, 32);
   memset(PeachCudaCTX[id].h_solve[1], 0, 32);

   return VEOK;
}  /* end peach_init_cuda_device() */

/**
 * Initialize a DEVICE_CTX list with CUDA devices for solving the Peach
 * proof of work algorithm.
 * @param devlist Pointer to DEVICE_CTX list to initialize
 * @param max Maximum number of CUDA devices to initialize
 * @returns number of CUDA devices available for initialization
 * @note It is possible to have "some" CUDA devices fail to initialize.
*/
int peach_init_cuda(DEVICE_CTX devlist[], int max)
{
   static int cuda_initialized = 0;
   static int cuda_num = 0;

   int id;

   /* avoid re-initialization attempts */
   if (cuda_initialized) return cuda_num;

   /* check for cuda driver and devices */
   switch(cudaGetDeviceCount(&cuda_num)) {
      case cudaErrorNoDevice:
         return plog("No CUDA devices detected...");
      case cudaErrorInsufficientDriver:
         pfatal("Insufficient CUDA Driver. Update display drivers...");
         return 0;
      case cudaSuccess:
         if (cuda_num > max) {
            cuda_num = max;
            plog("CUDA Devices: %d (limited)\n", cuda_num);
            perr("CUDA device count EXCEEDED maximum count parameter!");
            pwarn("Some CUDA devices will not be utilized.");
            plog("Please advise developers if this is an issue...");
         }
         break;
      default:
         pfatal("Unknown CUDA initialization error occured...");
         return 0;
   }

   /* set initialized */
   cuda_initialized = 1;
   if (cuda_num < 1) return (cuda_num = 0);

   /* allocate memory for PeachCudaCTX */
   PeachCudaCTX =
      (PEACH_CUDA_CTX *) malloc(sizeof(PEACH_CUDA_CTX) * cuda_num);

   /* allocate pinned host memory for data consistant across devices */
   cuCHK(cudaMallocHost(&h_phash, SHA256LEN), NULL, return (cuda_num = 0));
   cuCHK(cudaMallocHost(&h_diff, 1), NULL, return (cuda_num = 0));
   memset(h_phash, 0, SHA256LEN);
   memset(h_diff, 0, 1);

   /* initialize device contexts for CUDA num devices */
   for(id = 0; id < cuda_num; id++) {
      peach_init_cuda_device(&devlist[id], id);
   }

   return cuda_num;
}  /* end peach_init_cuda() */

/**
 * Try solve for a tokenized haiku as nonce output for Peach proof of work
 * on CUDA devices. Combine haiku protocols implemented in the Trigg
 * Algorithm with the memory intensive protocols of the Peach algorithm to
 * generate haiku output as proof of work.
 * @param dev Pointer to DEVICE_CTX to perform work with
 * @param bt Pointer to block trailer to solve for
 * @param diff Difficulty to test against entropy of final hash
 * @param out Pointer to location to place solved block trailer
 * @returns VEOK on solve, else VERROR
*/
int peach_solve_cuda(DEVICE_CTX *dev, BTRAILER *bt, word8 diff, void *out)
{
   static size_t SHA256CTXLEN = sizeof(SHA256_CTX);
   static size_t BTLEN = sizeof(BTRAILER);
   int i, id, sid, grid, block;
   PEACH_CUDA_CTX *P;
   nvmlReturn_t nr;
   size_t ictxlen;

   /* check for GPU failure */
   if (dev->status == DEV_FAIL) return VERROR;

   id = dev->id;
   P = &PeachCudaCTX[id];
   /* set/check cuda device */
   cuCHK(cudaSetDevice(id), dev, return VERROR);
   cuCHK(cudaGetLastError(), dev, return VERROR);

   /* ensure initialization is complete */
   if (dev->status == DEV_NULL) {
      if (cudaStreamQuery(cudaStreamDefault) != cudaSuccess) return VERROR;
      /* set next action to build Peach map */
      dev->status = DEV_INIT;
      dev->last_work = time(NULL);
      dev->total_work = 0;
      dev->work = 0;
   }

   /* build peach map */
   if (dev->status == DEV_INIT) {
      for(sid = 0; sid < 2; sid++) {
         if (cudaStreamQuery(P->stream[sid]) != cudaSuccess) continue;
         /* build peach map */
         if (dev->work == 0) {
            /* clear any late solves */
            memset(P->h_solve[0], 0, 32);
            memset(P->h_solve[1], 0, 32);
            cudaMemcpyAsync(P->d_solve[0], P->h_solve[0], 32,
               cudaMemcpyHostToDevice, P->stream[sid]);
            cudaMemcpyAsync(P->d_solve[1], P->h_solve[1], 32,
               cudaMemcpyHostToDevice, P->stream[sid]);
            /* update block trailer */
            memcpy(P->h_bt[0], bt, BTLEN);
            memcpy(P->h_bt[1], bt, BTLEN);
            /* ensure phash is set */
            memcpy(h_phash, bt->phash, SHA256LEN);
            /* asynchronous copy to phash and difficulty symbols */
            cudaMemcpyToSymbolAsync(c_phash, h_phash, SHA256LEN, 0,
               cudaMemcpyHostToDevice, P->stream[sid]);
            /* update h_diff from function diff parameter */
            if (*h_diff != diff) *h_diff = diff;
            /* if h_diff is zero (0) or greater than bt, use bt */
            if (*h_diff == 0 || *h_diff > bt->difficulty[0]) {
               cudaMemcpyToSymbolAsync(c_diff, P->h_bt[0]->difficulty,
                  1, 0, cudaMemcpyHostToDevice, P->stream[sid]);
            } else {
               cudaMemcpyToSymbolAsync(c_diff, h_diff,
                  1, 0, cudaMemcpyHostToDevice, P->stream[sid]);
            }
         }
         if (dev->work < PEACHCACHELEN) {
            /* set CUDA configuration for generating peach map */
            if (cudaOccupancyMaxPotentialBlockSize(&grid, &block,
                  kcu_peach_build, 0, 0) != cudaSuccess) {
               pdebug("cudaOccupancy~BlockSize(%d) failed...", id);
               pdebug("Using conservative defaults, <<<128/128>>>");
               grid = 128;
               block = 128;
            }
            /* launch kernel to generate map */
            kcu_peach_build<<<grid, block, 0, P->stream[sid]>>>
               (P->d_map, (word32) dev->work);
            cuCHK(cudaGetLastError(), dev, return VERROR);
            /* update build progress */
            dev->work += grid * block;
         } else {
            /* wait for both streams */
            if (cudaStreamQuery(P->stream[!sid]) != cudaSuccess) continue;
            dev->last_work = time(NULL);
            dev->status = DEV_IDLE;
            dev->work = 0;
            break;
         }
      }
   }

   /* check for solvable work in block trailer */
   if (dev->status == DEV_IDLE && get32(bt->tcount)) {
      dev->status = DEV_WORK;
   } else if (get32(bt->tcount) == 0) {
      dev->status = DEV_IDLE;
   }

   /* solve work in block trailer */
   if (dev->status == DEV_WORK) {
      for(sid = 0; sid < 2; sid++) {
         if (cudaStreamQuery(P->stream[sid]) != cudaSuccess) continue;
         /* check trailer for block update */
         if (cmp256(h_phash, bt->phash)) {
            /* wait for both streams */
            if (cudaStreamQuery(P->stream[!sid]) != cudaSuccess) continue;
            dev->status = DEV_INIT;
            dev->work = 0;
            continue;
         }
         /* check for solvable work in block trailer */
         if (get32(bt->tcount) == 0) {
            dev->status = DEV_IDLE;
            dev->work = 0;
            continue;
         }
         /* check for solves */
         if (*(P->h_solve[sid])) {
            /* move solved nonce */
            memcpy(P->h_bt[sid]->nonce, P->h_solve[sid], 32);
            /* clear solve from host/device */
            memset(P->h_solve[sid], 0, 32);
            cudaMemcpyAsync(P->d_solve[sid], P->h_solve[sid], 32,
               cudaMemcpyHostToDevice, P->stream[sid]);
            /* record stime */
            put32(P->h_bt[sid]->stime, time(NULL));
            memcpy(out, P->h_bt[sid], BTLEN);
            /* return a solve */
            return VEOK;
         }
         /* ensure block trailer is updated */
         memcpy(P->h_bt[sid], bt, BTSIZE);
         /* generate nonce directly into block trailer */
         trigg_generate_fast(P->h_bt[sid]->nonce);
         trigg_generate_fast(P->h_bt[sid]->nonce + 16);
         /* prepare intermediate state for next round */
         sha256_init(P->h_ictx[sid]);
         sha256_update(P->h_ictx[sid], P->h_bt[sid], 124);
         /* duplicate intermediate state with random second seed */
         for(i = 1; i < dev->threads; i++) {
            memcpy(&(P->h_ictx[sid][i]), P->h_ictx[sid], SHA256CTXLEN);
            trigg_generate_fast(P->h_ictx[sid][i].data + 44);
         }
         /* transfer ictx to device */
         ictxlen = dev->threads * SHA256CTXLEN;
         cudaMemcpyAsync(P->d_ictx[sid], P->h_ictx[sid], ictxlen,
            cudaMemcpyHostToDevice, P->stream[sid]);
         /* launch kernel to solve Peach */
         kcu_peach_solve<<<dev->grid, dev->block, 0, P->stream[sid]>>>
            (P->d_map, P->d_ictx[sid], P->d_solve[sid]);
         cuCHK(cudaGetLastError(), dev, return VERROR);
         /* retrieve solve seed */
         cudaMemcpyAsync(P->h_solve[sid], P->d_solve[sid], 32,
            cudaMemcpyDeviceToHost, P->stream[sid]);
         /* increment progress counters */
         dev->total_work += dev->threads;
         dev->work += dev->threads;
      }
   }

   /* power and temperature monitoring (1 second interval) */
   if (P->nvml_device && difftime(time(NULL), dev->last_monitor)) {
      dev->last_monitor = time(NULL);
      /* get GPU device power */
      unsigned int fan;
      nr = nvmlDeviceGetFanSpeed(P->nvml_device, &fan);
      if (nr != NVML_SUCCESS) {
         perr("nvml(%d) fan speed: %s\n", id, nvmlErrorString(nr));
         memset(&(P->nvml_device), 0, sizeof(nvmlDevice_t));
         P->nvml_enabled = 0;
      } else dev->fan = fan;
      /* get GPU device power */
      unsigned int power;
      nr = nvmlDeviceGetPowerUsage(P->nvml_device, &power);
      if (nr != NVML_SUCCESS) {
         perr("nvml(%d) power usage: %s\n", id, nvmlErrorString(nr));
         memset(&(P->nvml_device), 0, sizeof(nvmlDevice_t));
         P->nvml_enabled = 0;
      } else dev->pow = power / 1000;
      /* get GPU device temperature */
      unsigned int temperature;
      nr = nvmlDeviceGetTemperature(P->nvml_device, NVML_TEMPERATURE_GPU,
         &temperature);
      if (nr != NVML_SUCCESS) {
         perr("nvml(%d) temperature: %s\n", id, nvmlErrorString(nr));
         memset(&(P->nvml_device), 0, sizeof(nvmlDevice_t));
         P->nvml_enabled = 0;
      } else dev->temp = temperature;
      /* get GPU device utilization */
      nvmlUtilization_t utilization;
      nr = nvmlDeviceGetUtilizationRates(P->nvml_device, &utilization);
      if (nr != NVML_SUCCESS) {
         perr("nvml(%d) utilization rates: %s\n", id, nvmlErrorString(nr));
         memset(&(P->nvml_device), 0, sizeof(nvmlDevice_t));
         P->nvml_enabled = 0;
      } else dev->util = utilization.gpu;
   }

   return VERROR;
}  /* end peach_solve_cuda() */

/* end include guard */
#endif
