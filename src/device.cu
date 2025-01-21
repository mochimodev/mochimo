

#include "device.h"
#include "error.h"

#include <cuda_runtime.h>
#include <nvml.h>

/* personal preference macros */
#define CUDA_SUCCESS cudaSuccess

/* reasons of clarity */
#define cuda_nvml_pci_match(cuda_props, nvml_pci) \
   (cuda_props.pciDomainID == (int) nvml_pci.domain && \
      cuda_props.pciBusID == (int) nvml_pci.bus && \
      cuda_props.pciDeviceID == (int) nvml_pci.device)

#define cuda__log_error(err) \
   do { \
      perr("CUDA ERROR: (%d) %s", (int) err, cudaGetErrorString(err)); \
      set_errno(EMCM_CUDA); \
      return VERROR; \
   } while(0);

#define nvml__log_error(ret) \
   do { \
      perr("NVML ERROR: (%d) %s", (int) ret, nvmlErrorString(ret)); \
      set_errno(EMCM_NVML); \
      return VERROR; \
   } while(0);

int init_cuda_devices(DEVICE_CTX *ctx, int len)
{
   struct cudaDeviceProp props;
   cudaError_t err;
   nvmlReturn_t ret;
   nvmlPciInfo_t pci;
   nvmlDevice_t nvmlp;
   unsigned int nvml_count, gen, width;
   int cuda_count, cuda_idx;
   int nvml_idx, nvml_init;

   /* initialize nvml */
   nvml_init = 1;
   nvml_count = 0;
   ret = nvmlInit();
   if (ret != NVML_SUCCESS && ret != NVML_ERROR_ALREADY_INITIALIZED) {
      nvml__log_error(ret);
      nvml_init = 0;
   } else {
      ret = nvmlDeviceGetCount(&nvml_count);
      if (ret != NVML_SUCCESS) {
         nvml__log_error(ret);
         nvml_init = 0;
      }
   }

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

      /* clear (previous) link gen/width */
      gen = width = 0;

      if (nvml_init) {
         /* iterate over nvml devices to compare against cuda devices */
         for (nvml_idx = 0; nvml_idx < nvml_count; nvml_idx++) {
            /* ? perhaps not all cuda devices could be nvml devices ? */
            ret = nvmlDeviceGetHandleByIndex(nvml_idx, &nvmlp);
            if (ret != NVML_SUCCESS) {
               nvml__log_error(ret);
               continue;
            }
            ret = nvmlDeviceGetPciInfo(nvmlp, &pci);
            if (ret != NVML_SUCCESS) {
               nvml__log_error(ret);
               continue;
            }
            /* check for match */
            if (cuda_nvml_pci_match(props, pci)) {
               /* obtain current link gen/width */
               ret = nvmlDeviceGetMaxPcieLinkGeneration(nvmlp, &gen);
               if (ret != NVML_SUCCESS) {
                  nvml__log_error(ret);
                  continue;
               }
               ret = nvmlDeviceGetMaxPcieLinkWidth(nvmlp, &width);
               if (ret != NVML_SUCCESS) {
                  nvml__log_error(ret);
                  continue;
               }
               break;
            }
         }  /* end for (nvml_idx... */
      }  /* end if (nvml_init) */

      /* build device info string */
      snprintf(ctx[cuda_idx].info, sizeof(ctx[cuda_idx].info),
         "%04u:%02u:%02u %.128s Gen%1ux%02u", props.pciDomainID,
         props.pciDeviceID, props.pciBusID, props.name, gen, width);
   }

   /* shutdown nvml after use */
   if (nvml_init) {
      nvmlShutdown();
   }

   return cuda_count;
}  /* end init_cuda_devices() */
