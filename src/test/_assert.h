/** assert.h - Adequate Systems' extended assertion header for C testing
 *
 * Copyright (c) 2021 Adequate Systems, LLC. All Rights Reserved.
 * For more information, please refer to ../../LICENSE
 *
 * Date: 8 October 2021
 * Revised: 11 October 2021
 *
*/

#ifndef _TEST_ASSERT_H_
#define _TEST_ASSERT_H_  /* include guard */


/* standard assertion operations, with or without custom message */
#define ASSERT_EQ(A,B)              ASSERT_OP_MSG(==,A,B,"")
#define ASSERT_EQ2(A,B,C)           ASSERT_OP2_MSG(==,A,B,C,"")
#define ASSERT_GE(A,B)              ASSERT_OP_MSG(>=,A,B,"")
#define ASSERT_GE2(A,B,C)           ASSERT_OP2_MSG(>=,A,B,C,"")
#define ASSERT_GT(A,B)              ASSERT_OP_MSG(>,A,B,"")
#define ASSERT_GT2(A,B,C)           ASSERT_OP2_MSG(>,A,B,C,"")
#define ASSERT_LE(A,B)              ASSERT_OP_MSG(<=,A,B,"")
#define ASSERT_LE2(A,B,C)           ASSERT_OP2_MSG(<=,A,B,C,"")
#define ASSERT_LT(A,B)              ASSERT_OP_MSG(<,A,B,"")
#define ASSERT_LT2(A,B,C)           ASSERT_OP2_MSG(<,A,B,C,"")
#define ASSERT_NE(A,B)              ASSERT_OP_MSG(!=,A,B,"")
#define ASSERT_NE2(A,B,C)           ASSERT_OP2_MSG(!=,A,B,C,"")
#define ASSERT_OP(OP,A,B)           ASSERT_OP_MSG(OP,A,B,"")
#define ASSERT_OP2(OP,A,B,C)        ASSERT_OP2_MSG(OP,A,B,C,"")
#define ASSERT_STR(A,B,LEN)         ASSERT_STR_MSG(A,B,LEN,"")
#define ASSERT_CMP(A,B,LEN)         ASSERT_CMP_MSG(A,B,LEN,"")
#define ASSERT_ASC(A,LEN)           ASSERT_ASC_MSG(A,LEN,"")
#define ASSERT_EQ_MSG(A,B,MSG)      ASSERT_OP_MSG(==,A,B,MSG)
#define ASSERT_EQ2_MSG(A,B,C,MSG)   ASSERT_OP2_MSG(==,A,B,C,MSG)
#define ASSERT_GE_MSG(A,B,MSG)      ASSERT_OP_MSG(>=,A,B,MSG)
#define ASSERT_GE2_MSG(A,B,C,MSG)   ASSERT_OP2_MSG(>=,A,B,C,MSG)
#define ASSERT_GT_MSG(A,B,MSG)      ASSERT_OP_MSG(>,A,B,MSG)
#define ASSERT_GT2_MSG(A,B,C,MSG)   ASSERT_OP2_MSG(>,A,B,C,MSG)
#define ASSERT_LE_MSG(A,B,MSG)      ASSERT_OP_MSG(<=,A,B,MSG)
#define ASSERT_LE2_MSG(A,B,C,MSG)   ASSERT_OP2_MSG(<=,A,B,C,MSG)
#define ASSERT_LT_MSG(A,B,MSG)      ASSERT_OP_MSG(<,A,B,MSG)
#define ASSERT_LT2_MSG(A,B,C,MSG)   ASSERT_OP2_MSG(<,A,B,C,MSG)
#define ASSERT_NE_MSG(A,B,MSG)      ASSERT_OP_MSG(!=,A,B,MSG)
#define ASSERT_NE2_MSG(A,B,C,MSG)   ASSERT_OP2_MSG(!=,A,B,C,MSG)

#ifdef NDEBUG  /* NDEBUG override */
   #define ASSERT(COND) ((void)COND)  /* suppress compiler warnings */
#else /* ! defined(NDEBUG) - redirect ASSERT */
   #include <assert.h>
   #define ASSERT(COND) assert(COND)
#endif

#ifdef DEBUG
   #include <stdio.h>
   #include <string.h>
   #define PRINT(FMT, ...)  printf(FMT, ##__VA_ARGS__)
   #define PRINT_ARRAY(FMT, ARRAY, BYTES, ...) \
      do { \
         int _TYPESIZE = (int) sizeof(ARRAY[0]); \
         printf(FMT "{ ", ##__VA_ARGS__); \
         for (int _i = 0; _i < (int) (BYTES / _TYPESIZE); _i++) { \
            printf("0x%llx, ", (unsigned long long) ARRAY[_i]); } \
         printf("}\n"); \
      } while (0)
   #define ASSERT_OP2_MSG(OP,A,B,C,MSG) \
      do { \
         int _LEN = (int) strlen(#A); \
         if ((int) strlen(#B) > _LEN) _LEN = (int) strlen(#B); \
         if ((int) strlen(#C) > _LEN) _LEN = (int) strlen(#C); \
         unsigned long long a = (unsigned long long) A; \
         unsigned long long b = (unsigned long long) B; \
         unsigned long long c = (unsigned long long) C; \
         PRINT("DEBUG: a = %*s = %llu;\n", _LEN, #A, a); \
         PRINT("DEBUG: b = %*s = %llu;\n", _LEN, #B, b); \
         PRINT("DEBUG: c = %*s = %llu;\n", _LEN, #C, c); \
         PRINT(" TEST: assert(a " #OP " b  && b " #OP " c)\n"); \
         ASSERT(a OP b && b OP c && MSG); \
      } while (0)
   #define ASSERT_OP_MSG(OP,A,B,MSG) \
      do { \
         int _LEN = (int) strlen(#A); \
         if ((int) strlen(#B) > _LEN) _LEN = (int) strlen(#B); \
         unsigned long long a = (unsigned long long) A; \
         unsigned long long b = (unsigned long long) B; \
         PRINT("DEBUG: a = %*s = %llu;\n", _LEN, #A, a); \
         PRINT("DEBUG: b = %*s = %llu;\n", _LEN, #B, b); \
         PRINT(" TEST: assert(a " #OP " b)\n"); \
         ASSERT(a OP b && MSG); \
      } while (0)
   #define ASSERT_STR_MSG(A,B,LEN,MSG) \
      do { \
         char *a = (char *) A; \
         char *b = (char *) B; \
         int _LEN = (int) strlen(#A); \
         if ((int) strlen(#B) > _LEN) _LEN = (int) strlen(#B); \
         PRINT("DEBUG: a = %*s = \"%.*s\";\n", _LEN, #A, (int) LEN, a); \
         PRINT("DEBUG: b = %*s = \"%.*s\";\n", _LEN, #B, (int) LEN, b); \
         PRINT(" TEST: assert(strcmp(a, b) == 0);\n"); \
         ASSERT(strncmp(a, b, LEN) == 0 && MSG); \
      } while (0)
   #define ASSERT_CMP_MSG(A,B,LEN,MSG) \
      do { \
         int _LEN = (int) strlen(#A); \
         if ((int) strlen(#B) > _LEN) _LEN = (int) strlen(#B); \
         PRINT_ARRAY("DEBUG: A = %*s = ", A, LEN, _LEN, #A); \
         PRINT_ARRAY("DEBUG: B = %*s = ", B, LEN, _LEN, #B); \
         PRINT(" TEST: assert(memcmp(A, B, %d) == 0)\n", (int) LEN); \
         ASSERT(memcmp(A, B, LEN) == 0 && MSG); \
      } while (0)
   #define ASSERT_ASC_MSG(A,LEN,MSG) \
      do { \
         PRINT_ARRAY("DEBUG: A = %*s = ", A, LEN, (int) strlen(#A), #A); \
         PRINT(" TEST: for (i = 0; i < %zu; i++) ", (size_t) (LEN - 2)); \
         PRINT("assert(A[i] < A[i+1]);\n"); \
         for (int _i = 0; _i < (LEN-2); _i++) ASSERT(A[_i] < A[_i+1] && MSG); \
      } while (0)
#else  /* end DEBUG */
   #include <assert.h>
   #include <string.h>
   #define ASSERT_OP_MSG(OP,A,B,MSG)    ASSERT(A OP B && MSG)
   #define ASSERT_OP2_MSG(OP,A,B,C,MSG) ASSERT(A OP B && B OP C && MSG)
   #define ASSERT_STR_MSG(A,B,LEN,MSG)  ASSERT(strncmp(A, B, LEN) == 0 && MSG)
   #define ASSERT_CMP_MSG(A,B,LEN,MSG)  ASSERT(memcmp(A, B, LEN) == 0 && MSG)
   #define ASSERT_ASC_MSG(A,LEN,MSG) \
      for (int _i = 0; _i < (LEN-2); _i++) ASSERT(A[_i] < A[_i+1] && MSG)
#endif


#endif  /* end _TEST_ASSERT_H_ */
