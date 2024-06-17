/**
 * @file memdata.h
 * @brief Mochimo dynamic memory container support.
 * @copyright Adequate Systems LLC, 2024. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_MEMDATA_H
#define MOCHIMO_MEMDATA_H


/* system support */
#include <stdio.h>
#include <stdlib.h>

/** Memory buffer struct */
typedef struct {
   char *buf;
   size_t bufsz;
   size_t bytes;
   size_t position;
} MEM;

/* C/C++ compatible function prototypes */
#ifdef __cplusplus
extern "C" {
#endif

int memfree(MEM *mp);
MEM *memdynamic(void *ptr, size_t ptrsz);
MEM *memstatic(void *ptr, size_t ptrsz);
size_t memread(void *buffer, size_t size, size_t count, MEM *mp);
int memseek(MEM *mp, long long offset, int whence);
FILE *memstream(MEM *mp);
size_t memwrite(const void *buffer, size_t size, size_t count, MEM *mp);

#ifdef __cplusplus
}  /* end extern "C" */
#endif

/* end include guard */
#endif
