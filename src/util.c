/**
 * @file util.c
 * @private
 * @headerfile util.h <util.h>
 * @brief Mochimo utility functions.
 * @details File operation utilities not yet available in extended-c.
 * These functions are candidates for migration to extended-c/src/extio.
 * When migrated, remove this file and update includes accordingly.
 * @copyright Adequate Systems LLC, 2025. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
 *
 * @par Migration instructions for extended-c:
 * 1. Add fappend() and rmdir_r() to extended-c/src/extio.c
 * 2. Add prototypes to extended-c/src/extio.h
 * 3. Remove this file (src/util.c) and its header (src/util.h)
 * 4. Update any #include "util.h" to rely on extio.h instead
*/

/* include guard */
#ifndef MOCHIMO_UTIL_C
#define MOCHIMO_UTIL_C


#include "util.h"

#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>

/**
 * Append the contents of one file to another.
 * Opens source for reading and destination for appending.
 * Creates destination if it does not exist.
 * @param srcpath Path of the source file to read from
 * @param dstpath Path of the destination file to append to
 * @return 0 on success, or non-zero on error. Check errno for details.
*/
int fappend(const char *srcpath, const char *dstpath)
{
   char buf[BUFSIZ];
   FILE *sfp, *dfp;
   size_t nBytes;
   int fd, ecode = -1;

   /* open source file */
   sfp = fopen(srcpath, "rb");
   if (sfp != NULL) {
      /* open destination file for append with restricted permissions */
      fd = open(dstpath, O_WRONLY | O_CREAT | O_APPEND, 0600);
      dfp = (fd != -1) ? fdopen(fd, "ab") : NULL;
      if (dfp != NULL) {
         /* transfer bytes in BUFSIZ chunks (set by stdio) */
         while ((nBytes = fread(buf, 1, BUFSIZ, sfp))) {
            if (nBytes > 0 && fwrite(buf, nBytes, 1, dfp) != 1) break;
            if (nBytes < BUFSIZ) break;  /* EOF or ERROR */
         }
         /* if no file errors, set operation success (0) */
         if (ferror(sfp) == 0 && ferror(dfp) == 0) ecode = 0;
         fclose(dfp);
      }
      fclose(sfp);
   }

   return ecode;
}  /* end fappend() */

/**
 * Remove a directory and all files within it (non-recursive).
 * Only removes regular files within the directory; does not
 * descend into subdirectories. Fails if subdirectories exist
 * and are non-empty.
 * @param dirpath Path of the directory to remove
 * @return 0 on success, or non-zero on error. Check errno for details.
*/
int rmdir_r(const char *dirpath)
{
   DIR *dir;
   struct dirent *entry;
   char filepath[BUFSIZ];

   dir = opendir(dirpath);
   if (dir == NULL) return -1;
   while ((entry = readdir(dir)) != NULL) {
      if (strcmp(entry->d_name, ".") == 0) continue;
      if (strcmp(entry->d_name, "..") == 0) continue;
      snprintf(filepath, sizeof(filepath), "%s/%s", dirpath, entry->d_name);
      remove(filepath);
   }
   closedir(dir);

   return rmdir(dirpath);
}  /* end rmdir_r() */

/* end include guard */
#endif
