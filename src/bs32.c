/* bs32.c  Binary search array of word32 indexes
 *
 * Copyright (c) 2018 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * The Mochimo Project System Software
*/

#ifndef BS32FUN
#define BS32FUN  (target - array[mid])
#endif

/* Binary search an array of indexes for target.
 * Returns 1 if found, 0 if not found.
 * If position is non-NULL put the index of found target there,
 * else the index of where to insert target.
 */
int bsearch32(word32 *array, word32 len, word32 target, word32 *position)
{
   int cond, mid, hi, low;

   low = 0;
   hi = len - 1;

   while(low <= hi) {
      mid = (hi + low) / 2;
      cond = BS32FUN;         /* target - array[mid]; * comparison function */
      if(cond == 0) {
         if(position) *position = mid;
         return 1;  /* found target */
      }
      if(cond < 0) hi = mid - 1; else low = mid + 1;
   }  /* end while */
   /* Not found.
    * To add target, move array[position] up and insert target
    * at array[position].
    */
   if(position) *position = low;
   return 0;  /* not found */
}  /* end bsearch32() */
