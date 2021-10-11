/** assert.h - Adequate Systems' extended assertion header for C testing
 *
 * Copyright (c) 2021 Adequate Systems, LLC. All Rights Reserved.
 * For more information, please refer to ../../LICENSE
 *
 * Date: 8 October 2021
 *
*/

#ifndef _TEST_ASSERT_H_
#define _TEST_ASSERT_H_  /* include guard */


#define ASSERT_EQ(A,B,MSG)      ASSERT_OP(==,A,B,MSG)
#define ASSERT_EQ2(A,B,C,MSG)   ASSERT_OP2(==,A,B,C,MSG)
#define ASSERT_GE(A,B,MSG)      ASSERT_OP(>=,A,B,MSG)
#define ASSERT_GE2(A,B,C,MSG)   ASSERT_OP2(>=,A,B,C,MSG)
#define ASSERT_GT(A,B,MSG)      ASSERT_OP(>,A,B,MSG)
#define ASSERT_GT2(A,B,C,MSG)   ASSERT_OP2(>,A,B,C,MSG)
#define ASSERT_LE(A,B,MSG)      ASSERT_OP(<=,A,B,MSG)
#define ASSERT_LE2(A,B,C,MSG)   ASSERT_OP2(<=,A,B,C,MSG)
#define ASSERT_LT(A,B,MSG)      ASSERT_OP(<,A,B,MSG)
#define ASSERT_LT2(A,B,C,MSG)   ASSERT_OP2(<,A,B,C,MSG)
#define ASSERT_NE(A,B,MSG)      ASSERT_OP(!=,A,B,MSG)
#define ASSERT_NE2(A,B,C,MSG)   ASSERT_OP2(!=,A,B,C,MSG)

#ifdef DEBUG
   #ifdef NDEBUG
      #define ASSERT(COND) ((void) COND)
   #else
      #include <assert.h>
      #define ASSERT(COND) assert(COND)
   #endif
   #include <stdio.h>
   #include <string.h>
   #define PRINT(fmt, ...)  printf(fmt, ##__VA_ARGS__)
   #define PRINT_ARRAY(fmt, array, len, ...) \
      do { \
         printf(fmt "{ ", ##__VA_ARGS__); \
         unsigned char *_B = (unsigned char *) array; \
         for (int _i = 0; _i < len; _i++) { printf("0x%x, ", _B[_i]); } \
         printf("}\n"); \
      } while (0)
   #define ASSERT_OP2(OP,A,B,C,MSG) \
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
   #define ASSERT_OP(OP,A,B,MSG) \
      do { \
         int _LEN = (int) strlen(#A); \
         if ((int) strlen(#B) > _LEN) _LEN = (int) strlen(#B); \
         unsigned long long a = (unsigned long long) A; \
         unsigned long long b = (unsigned long long) B; \
         PRINT("DEBUG: a = %*s = %llu;\n", _LEN, #A, a); \
         PRINT("DEBUG: b = %*s = %llu;\n", _LEN, #B, b); \
         PRINT(" TEST: assert(a " #OP " b)\n"); \
         assert(a OP b && MSG); \
      } while (0)
   #define ASSERT_STR(A,B,LEN,MSG) \
      do { \
         char *a = (char *) A; \
         char *b = (char *) B; \
         int _LEN = (int) strlen(#A); \
         if ((int) strlen(#B) > _LEN) _LEN = (int) strlen(#B); \
         PRINT("DEBUG: a = %*s = \"%.*s\";\n", _LEN, #A, (int) LEN, a); \
         PRINT("DEBUG: b = %*s = \"%.*s\";\n", _LEN, #B, (int) LEN, b); \
         PRINT(" TEST: assert(strcmp(a, b) == 0);\n"); \
         assert(strncmp(a, b, LEN) == 0 && MSG); \
      } while (0)
   #define ASSERT_CMP(A,B,LEN,MSG) \
      do { \
         unsigned char *a = (unsigned char *) A; \
         unsigned char *b = (unsigned char *) B; \
         int _LEN = (int) strlen(#A); \
         if ((int) strlen(#B) > _LEN) _LEN = (int) strlen(#B); \
         PRINT_ARRAY("DEBUG: a = %*s = ", a, LEN, _LEN, #A); \
         PRINT_ARRAY("DEBUG: b = %*s = ", b, LEN, _LEN, #B); \
         PRINT(" TEST: assert(memcmp(a, b, %zu) == 0)\n", (size_t) LEN); \
         assert(memcmp(a, b, LEN) == 0 && MSG); \
      } while (0)
   #define ASSERT_ASC(A,LEN,MSG) \
      do { \
         PRINT_ARRAY("DEBUG: a = %*s = ", A, LEN, (int) strlen(#A), #A); \
         PRINT(" TEST: for (i = 0; i < %zu; i++) ", (size_t) (LEN - 2)); \
         PRINT("assert(a[i] < a[i+1]);\n"); \
         for (int _i = 0; _i < (LEN-2); _i++) assert(A[_i] < A[_i+1] && MSG); \
      } while (0)
#elif ! defined(NDEBUG)
   #include <assert.h>
   #include <string.h>
   #define ASSERT_OP2(OP,A,B,C,MSG) assert(A OP B && B OP C && MSG)
   #define ASSERT_OP(OP,A,B,MSG)    assert(A OP B && MSG)
   #define ASSERT_STR(A,B,LEN,MSG)  assert(strncmp(A, B, LEN) == 0 && MSG)
   #define ASSERT_CMP(A,B,LEN,MSG)  assert(memcmp(A, B, LEN) == 0 && MSG)
   #define ASSERT_ASC(A,LEN,MSG) \
      for (int _i = 0; _i < (LEN-2); _i++) assert(A[_i] < A[_i+1] && MSG)
#else
   #define ASSERT_OP2(OP,A,B,C,MSG) do { (void)A; (void)B; (void)C; } while (0)
   #define ASSERT_OP(OP,A,B,MSG)    do { (void)A; (void)B; } while (0)
   #define ASSERT_STR(A,B,MSG)      do { (void)A; (void)B; } while (0)
   #define ASSERT_CMP(A,B,LEN,MSG)  do { (void)A; (void)B; } while (0)
   #define ASSERT_ASC(A,LEN,MSG)    do { (void)A; } while (0)
#endif


#endif  /* end _TEST_ASSERT_H_ */
