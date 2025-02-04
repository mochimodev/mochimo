

#include "device.h"
#include "error.h"

#include <cuda_runtime.h>
#include <nvml.h>

/* personal preference macros */
#define CUDA_SUCCESS cudaSuccess

#define cuda__log_error(err) \
   do { \
      perr("CUDA ERROR: (%d) %s", (int) err, cudaGetErrorString(err)); \
      set_errno(EMCM_CUDA); \
   } while(0);

int init_cuda_devices(DEVICE_CTX *ctx, int len)
{
   struct cudaDeviceProp props;
   cudaError_t err;
   int cuda_count;
   int cuda_idx;

   /* get cuda devices */
   cuda_count = 0;
   err = cudaGetDeviceCount(&cuda_count);
   if (err != CUDA_SUCCESS) {
      cuda__log_error(err);
      return (-1);
   }

   /* check device context capacity */
   if (cuda_count > len) {
      set_errno(EMCM_CUDA_LIMIT);
      return (-1);
   }

   /* initialize context per device */
   for (cuda_idx = 0; cuda_idx < cuda_count; cuda_idx++) {
      ctx[cuda_idx].id = cuda_idx;
      ctx[cuda_idx].type = CUDA_DEVICE;
      ctx[cuda_idx].status = DEV_NULL;
      ctx[cuda_idx].work = ctx[cuda_idx].total = 0;
      ctx[cuda_idx].last = time(NULL);
      ctx[cuda_idx].peach = NULL;
      snprintf(ctx[cuda_idx].info, sizeof(ctx[cuda_idx].info), "Unknown Device");

      /* cross reference cuda-nvml properties to build device info */
      err = cudaGetDeviceProperties(&props, cuda_idx);
      if (err != CUDA_SUCCESS) {
         cuda__log_error(err);
         continue;
      }

      /* build device info string */
      snprintf(ctx[cuda_idx].info, sizeof(ctx[cuda_idx].info),
         "%04u:%02u:%02u %.128s", props.pciDomainID,
         props.pciDeviceID, props.pciBusID, props.name);
   }

   return cuda_count;
}  /* end init_cuda_devices() */
