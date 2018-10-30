/* sort.c  Polymorphic Shell sort
 *
 * Copyright (c) 2018 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * The Mochimo Project System Software
 *
 * Date 10 January 2018
*/


#ifndef SHELLFUN
/* For sorting unsigned indexes: a, k, gap, and temp are local to shell() */
#define SHELLFUN  (a[k - *gap] > temp)
/* Example for 32-byte hash value:
 * #define SHELLFUN (memcmp(&Tx_ids[a[k - *gap]*32], &Tx_ids[temp*32], 32) > 0)
 */
#endif


/* Sort an array unsigned array a[0...n-1] of indexes. */
void shell(unsigned *a, int n)
{

   static int gaps[] = {
      90934, 40415, 17962, 7983, 3548, 1577,
        701,   301,   132,   57,   23,   10, 4, 1
   };  /* k = k * 2.25 */

   int *gap, j, k;
   unsigned temp;

   if(n < 1) return;

   gap = gaps;

   for( ; ; gap++ ) {
      for(j = *gap; j < n; j++) {
         temp = a[j];
         for(k = j;
            k >= *gap
            /*           a[k - *gap]  > temp   */
            && SHELLFUN;  /* our comparison function */
            k -= *gap) {
                a[k] = a[k - *gap];
         }  /* end for k */
         a[k] = temp;  /* in correct location */
      }  /* end for j */
      if(*gap <= 1) break;  /* try next gap */
   }  /* end gaps */
}  /* end shell() */
