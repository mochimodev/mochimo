/* rand.c  High speed random number generation.
 *
 * Copyright (c) 2018 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 2 January 2018
 *
*/

#ifdef TESTRAND
#include "config.h"
#include <stdlib.h>
#include <stdio.h>
#endif

/* Default initial seed for random generator */
static word32 Lseed = 1;
static word32 Lseed2 = 1;
static word32 Lseed3 = 362436069;
static word32 Lseed4 = 123456789;

/* Seed the generator */
word32 srand16(word32 x)
{
   word32 r;

   r = Lseed;
   Lseed = x;
   return r;
}

/* Return random seed to caller */
word32 getrand16(void)
{
   return Lseed;
}

void srand2(word32 x, word32 y, word32 z)
{
   Lseed2 = x;
   Lseed3 = y;
   Lseed4 = z;
}

/* Return random seed to caller */
void getrand2(word32 *x, word32 *y, word32 *z)
{
   *x = Lseed2;
   *y = Lseed3;
   *z = Lseed4;
}

/* Period: 2**32 randl4() -- returns 0-65535 */
word32 rand16(void)
{
   Lseed = Lseed * 69069L + 262145L;
   return (Lseed >> 16);
}


/* Based on Dr. Marsaglia's Usenet post */
word32 rand2(void)
{
   Lseed2 = Lseed2 * 69069L + 262145L;  /* LGC */
   if(Lseed3 == 0) Lseed3 = 362436069;
   Lseed3 = 36969 * (Lseed3 & 65535) + (Lseed3 >> 16);  /* MWC */
   if(Lseed4 == 0) Lseed4 = 123456789;
   Lseed4 ^= (Lseed4 << 17);
   Lseed4 ^= (Lseed4 >> 13);
   Lseed4 ^= (Lseed4 << 5);  /* LFSR */
   return (Lseed2 ^ (Lseed3 << 16) ^ Lseed4) >> 16;
}


#ifdef TESTRAND
#include <time.h>

word32 Hist[256];

int main()
{
   word32 j;
   double v, p;

   printf("rand2():\n");
   srand2(time(NULL), 0, 0);

   for(j = 0; j < 1000000L; j++)
      Hist[rand2() % 100]++;
   for(j = 0; j < 100; j++)
      printf("%lu  ", (long) Hist[j]);
      printf("\n");
   v = 0.0;
   p = .01;
   for(j = 0; j < 100; j++)
      v += (Hist[j] * Hist[j] / p);
   v = ((1.0/1000000.0) * v) - 1000000.0;
   printf("Chi-square: %f\n", v);   /* > 77.046  < 123.225 DF=99 a=0.05 */

   printf("\nrand16():\n");
   for(j = 0; j < 10; j++)
      Hist[j] = 0L;
   Lseed = Lseed2 = time(NULL);
   for(j = 0; j < 1000000L; j++)
      Hist[rand16() % 10]++;
   for(j = 0; j < 10; j++)
      printf("%lu  ", (long) Hist[j]);
      printf("\n\n");

   for(j = 0; j < 10; j++)
      printf("%ld  ", (long) Hist[j] - 100000L);
      printf("\n\n");

   exit(0);
}

#endif
