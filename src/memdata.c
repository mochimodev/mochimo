/**
 * @private
 * @headerfile memdata.h <memdata.h>
 * @copyright Adequate Systems LLC, 2024. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_MEMDATA_C
#define MOCHIMO_MEMDATA_C


#include "memdata.h"

/* external support */
#include "exterrno.h"
#include <string.h>

#ifdef _WIN32
   #include <win32lean.h>  /* for Windows definitions */
   #include <fcntl.h>      /* for _O_RDWR et al. */
   #include <io.h>

   /* Windows does not support fmemopen(); polyfill... */
   FILE *fmemopen(void *buf, size_t len, const char *mode)
   {
      DWORD access = GENERIC_READ | GENERIC_WRITE;
      DWORD flags = FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE;
      HANDLE handle = INVALID_HANDLE_VALUE;
      FILE *fp = NULL;
      int fd = (-1);
      char path[FILENAME_MAX - 13];
      char file[FILENAME_MAX + 1];

      /* create temporary Windows file */
      if (GetTempPathA(sizeof(path), path) == 0) goto FALLBACK;
      if (GetTempFileNameA(path, "mcmtmp", 0, file) == 0) goto FALLBACK;
      handle = CreateFileA(file, access, 0, NULL, CREATE_ALWAYS, flags, NULL);
      if (handle == INVALID_HANDLE_VALUE) goto FALLBACK;

      /* get file descriptor from windows handle */
      fd = _open_osfhandle((intptr_t) handle, _O_RDWR);
      if (fd == -1) goto FALLBACK;

      /* open FILE from descriptor using specified mode */
      fp = _fdopen(fd, mode);
      if (fp == NULL) goto FALLBACK;

   FALLBACK_SUCCESS:
      /* place provided buffer data into FILE */
      if (fwrite(buf, len, 1, fp) != 1) goto FALLBACK;
      rewind(fp);

      return fp;

      /* cleanup / error handling */
   FALLBACK:
      if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle);
      if (fd != -1) _close(fd);
      if (fp != NULL) fclose(fp);
      else {
         fp = tmpfile();
         if (fp == NULL) return NULL;
         goto FALLBACK_SUCCESS;
      }

      return NULL;
   }  /* end fmemopen() */

#endif

/* golden ratio buffer growth strategy (approximated) */
static inline size_t golden_ceil(size_t size)
{
   size = (size * 207) >> 6;
   size = (size >> 1) + (size & 1);

   return size;
}

int memfree(MEM *mp)
{
   /* check invalid parameter */
   if (mp == NULL) {
      set_errno(EINVAL);
      return -1;
   }
   /* free allocated memory */
   if (mp->buf && mp->bufsz) free(mp->buf);
   /* deallocate MEM pointer */
   free(mp);

   return 0;
}  /* end memfree() */

MEM *memdynamic(void *ptr, size_t ptrsz)
{
   MEM *mp;

   /* allocate MEM pointer handler */
   mp = malloc(sizeof(MEM));
   if (mp == NULL) return NULL;
   mp->buf = ptr;
   mp->bufsz = ptrsz;
   mp->bytes = 0;
   mp->position = 0;

   /* allocate memory */
   if (mp->buf == NULL) {
      mp->buf = malloc(ptrsz);
      if (mp->buf == NULL) {
         /* deallocate on error */
         free(mp);
         return NULL;
      }
   }

   return mp;
}  /* end memdynamic() */

MEM *memstatic(void *ptr, size_t ptrsz)
{
   MEM *mp;

   /* allocate MEM pointer handler */
   mp = malloc(sizeof(MEM));
   if (mp == NULL) return NULL;
   mp->buf = ptr;
   mp->bufsz = 0;
   mp->bytes = ptrsz;
   mp->position = 0;

   return mp;
}  /* end memstatic() */

size_t memread(void *buffer, size_t size, size_t count, MEM *mp)
{
   size_t n;

   /* check invalid parameter */
   if (buffer == NULL || size == 0 || count == 0 || mp == NULL) {
      set_errno(EINVAL);
      return 0;
   }

   /* adjust out-of-range count */
   n = (mp->bytes - mp->position) / size;
   if (n < count) count = n;

   /* copy data from buffer, update position */
   memcpy(buffer, mp->buf + mp->position, size * count);
   mp->position += size * count;

   /* return actual count read */
   return count;
}  /* end memread() */

int memseek(MEM *mp, long long offset, int whence)
{
   /* check invalid parameter */
   if (mp == NULL) {
      set_errno(EINVAL);
      return (-1);
   }

   /* adjust position in MEM pointer */
   switch (whence) {
      case SEEK_SET:
         mp->position = offset;
         break;
      case SEEK_CUR:
         mp->position += offset;
         break;
      case SEEK_END:
         mp->position = mp->bytes + offset;
         break;
      default:
         set_errno(EINVAL);
         return (-1);
   }

   return 0;
}  /* end memseek() */

FILE *memstream(MEM *mp)
{
#ifdef _WIN32
   /* Windows does not support fmemopen(), use alternate method... */
   /** @todo implement WIN32 equivalent */
   return NULL;
#else
   return fmemopen(mp->buf, mp->bytes, "r");
#endif
}

size_t memwrite(const void *buffer, size_t size, size_t count, MEM *mp)
{
   size_t required, len, n;
   void *ptr;

   /* check invalid parameter */
   if (buffer == NULL || size == 0 || count == 0 || mp == NULL) {
      set_errno(EINVAL);
      return 0;
   }

   /* do not reallocate for static memory types */
   if (mp->bufsz) {
      /* check required memory space, reallocate if necessary */
      required = mp->position + (size * count);
      if (required > mp->bufsz) {
         /* determine new len using golden growth strategy */
         for (len = mp->bufsz; len < required; len = golden_ceil(len));
         /* reallocate larger space */
         ptr = realloc(mp->buf, len);
         if (ptr == NULL) return 0;
         mp->buf = ptr;
         mp->bufsz = len;
      }
   }

   /* adjust out-of-range count (only applies to static MEM) */
   n = (mp->bytes - mp->position) / size;
   if (n < count) count = n;

   /* copy data, and update position and byte count */
   memcpy(mp->buf + mp->position, buffer, size * count);
   mp->position += size * count;
   if (mp->position > mp->bytes) {
      /* SHOULD only be triggered for dynamic MEM */
      mp->bytes = mp->position;
   }

   return count;
}  /* end memwrite() */

/* end include guard */
#endif
