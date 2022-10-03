/**
 * @file sort.h
 * @brief Mochimo quick sorting support.
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
 * @note The original Polymorphic Shell sort algorithm, shell(),
 * was deprecated in favour of qsort() and adapted to sorting
 * void pointers for better flexibility of compared sizes.
 * > For more details see <https://godbolt.org/z/rM13Kra14>
*/

/* include guard */
#ifndef MOCHIMO_SORT_H
#define MOCHIMO_SORT_H


#include <stdlib.h>

/**
 * Default number of open split files that are merged together
 * during the merge phase of external_merge_sort().
*/
#define DEFAULT_SORT_FILES  8

/**
 * Default amount of memory to use for sorting in "split phase" and
 * chunk reading/writing in "merge phase" of external_merge_sort().
 * Default value, ( 1 << 28 ) = 256MB.
 * NOTE: Configurable at run-time with MaxSortBuffer_opt.
*/
#define DEFAULT_SORT_BUFFER ( 1 << 28 )

/* define global options */

extern size_t MaxSortBuffer_opt;

/* C/C++ compatible function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

int filesort_compare_tagidx(const void *a, const void *b);

int filesort
   (char *filename, size_t size, int (*comp)(const void *, const void *));

#ifdef __cplusplus
}  /* end extern "C" */
#endif

/* end include guard */
#endif
