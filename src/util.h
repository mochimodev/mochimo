/**
 * @file util.h
 * @brief Mochimo utility functions.
 * @details File operation utilities not yet available in extended-c.
 * These functions are candidates for migration to extended-c/src/extio.
 * When migrated, remove this file and update includes accordingly.
 * @copyright Adequate Systems LLC, 2025. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_UTIL_H
#define MOCHIMO_UTIL_H

/* C/C++ compatible function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Append the contents of one file to another.
 * Opens source for reading and destination for appending.
 * Creates destination if it does not exist.
 * @param srcpath Path of the source file to read from
 * @param dstpath Path of the destination file to append to
 * @return 0 on success, or non-zero on error. Check errno for details.
 * @note Migration: add to extended-c/src/extio.h as fappend()
 */
int fappend(const char *srcpath, const char *dstpath);

/**
 * Remove a directory and all files within it (non-recursive).
 * Only removes regular files within the directory; does not
 * descend into subdirectories. Fails if subdirectories exist.
 * @param dirpath Path of the directory to remove
 * @return 0 on success, or non-zero on error. Check errno for details.
 * @note Migration: add to extended-c/src/extio.h as rmdir_r()
 *       Consider adding recursive subdirectory support if needed.
 */
int rmdir_r(const char *dirpath);

#ifdef __cplusplus
}  /* end extern "C" */
#endif

/* end include guard */
#endif
