
/* include guard */
#ifndef TEST_ASSERT_H
#define TEST_ASSERT_H


#ifdef NDEBUG
   /* suppress compiler warnings */
   #define ASSERT(COND) ((void)COND)

#else
   #include <assert.h>
   #define ASSERT(COND) assert(COND)

#endif

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

#ifdef DEBUG
   #include <stdarg.h>
   #include <stdio.h>
   #include <string.h>
   #define PRINT_ARRAY(ARRAY, BYTES, ...) \
      do { \
         int _TYPESIZE = (int) sizeof((ARRAY)[0]); \
         printf(__VA_ARGS__); \
         printf("{ "); \
         for (int _i = 0; _i < (int) BYTES / _TYPESIZE; _i++) { \
            printf("0x%llx, ", (long long) (ARRAY)[_i]); \
         } \
         printf("}\n"); \
         fflush(stdout); \
      } while (0)
   #define ASSERT_OP2_MSG(OP,A,B,C,MSG) \
      do { \
         int _LEN = (int) strlen(#A); \
         if ((int) strlen(#B) > _LEN) _LEN = (int) strlen(#B); \
         if ((int) strlen(#C) > _LEN) _LEN = (int) strlen(#C); \
         long long a = (long long) A; \
         long long b = (long long) B; \
         long long c = (long long) C; \
         printf("ASSERT: a = %*s = %llu;\n", _LEN, #A, a); \
         printf("ASSERT: b = %*s = %llu;\n", _LEN, #B, b); \
         printf("ASSERT: c = %*s = %llu;\n", _LEN, #C, c); \
         printf("ASSERT: assert(a " #OP " b  && b " #OP " c); %s\n\n", MSG); \
         ASSERT(a OP b && b OP c && MSG); \
      } while (0)
   #define ASSERT_OP_MSG(OP,A,B,MSG) \
      do { \
         int _LEN = (int) strlen(#A); \
         if ((int) strlen(#B) > _LEN) _LEN = (int) strlen(#B); \
         long long a = (long long) A; \
         long long b = (long long) B; \
         printf("ASSERT: a = %*s = %llu;\n", _LEN, #A, a); \
         printf("ASSERT: b = %*s = %llu;\n", _LEN, #B, b); \
         printf("ASSERT: assert(a " #OP " b); %s\n\n", MSG); \
         ASSERT(a OP b && MSG); \
      } while (0)
   #define ASSERT_STR_MSG(A,B,LEN,MSG) \
      do { \
         char *a = (char *) A; \
         char *b = (char *) B; \
         int _LEN = (int) strlen(#A); \
         if ((int) strlen(#B) > _LEN) _LEN = (int) strlen(#B); \
         printf("ASSERT: a = %*s = \"%.*s\";\n", _LEN, #A, (int) LEN, a); \
         printf("ASSERT: b = %*s = \"%.*s\";\n", _LEN, #B, (int) LEN, b); \
         printf("ASSERT: assert(strcmp(a, b) == 0); %s\n\n", MSG); \
         ASSERT(strncmp(a, b, LEN) == 0 && MSG); \
      } while (0)
   #define ASSERT_CMP_MSG(A,B,LEN,MSG) \
      do { \
         int _LEN = (int) strlen(#A); \
         if ((int) strlen(#B) > _LEN) _LEN = (int) strlen(#B); \
         PRINT_ARRAY(A, LEN, "ASSERT: A = %*s = ", _LEN, #A); \
         PRINT_ARRAY(B, LEN, "ASSERT: B = %*s = ", _LEN, #B); \
         printf("ASSERT: assert(memcmp(A, B, %d) == 0); %s\n\n", \
            (int) LEN, MSG); \
         ASSERT(memcmp(A, B, LEN) == 0 && MSG); \
      } while (0)
   #define ASSERT_ASC_MSG(A,LEN,MSG) \
      do { \
         printf_ARRAY(A, LEN, "ASSERT: A = %*s = ", (int) strlen(#A), #A); \
         printf("ASSERT: for (i = 0; i < %zu; i++) ", (size_t) (LEN - 2)); \
         printf("assert(A[i] < A[i+1]); %s\n\n", MSG); \
         for (int _i = 0; _i < (LEN-2); _i++) { \
            ASSERT((A)[_i] < (A)[_i+1] && MSG); } \
      } while (0)
   static inline void ASSERT_DEBUG(char *fmt, ...)
   {
      va_list args;

      va_start(args, fmt);
      vprintf(fmt, args);
      va_end(args);
   }

/* end DEBUG */
#else
   #include <assert.h>
   #include <string.h>
   #define ASSERT_OP_MSG(OP,A,B,MSG)    ASSERT(A OP B && MSG)
   #define ASSERT_OP2_MSG(OP,A,B,C,MSG) ASSERT(A OP B && B OP C && MSG)
   #define ASSERT_STR_MSG(A,B,LEN,MSG)  ASSERT(strncmp(A, B, LEN) == 0 && MSG)
   #define ASSERT_CMP_MSG(A,B,LEN,MSG)  ASSERT(memcmp(A, B, LEN) == 0 && MSG)
   #define ASSERT_ASC_MSG(A,LEN,MSG) \
      for (int _i = 0; _i < (LEN-2); _i++) ASSERT((A)[_i] < (A)[_i+1] && MSG)
   static inline void ASSERT_DEBUG(char *fmt, ...) { (void)fmt; }

/* end ! defined DEBUG */
#endif

/* end include guard */
#endif
