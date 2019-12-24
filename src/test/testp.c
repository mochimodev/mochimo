/* testp.c  Test functions from proof.c

   int sub_weight(byte *weight, byte difficulty);
   int past_weight(byte *weight, word32 lownum);

   Date: 23 November 2019
*/


#include "../config.h"
#include "../mochimo.h"

#define plog printf
#define BAIL(m) { message = m; goto bail; }
int Trace = 1;
byte Weight[32];
byte Cblocknum[8] = { 100 };

word32 get32(void *buff)
{
   return *((word32 *) buff);
}

/* buff<--val */
void put64(void *buff, void *val)
{
   ((word32 *) buff)[0] = ((word32 *) val)[0];
   ((word32 *) buff)[1] = ((word32 *) val)[1];
}


/* Check if buff is all zeros */
int iszero(void *buff, int len)
{
   byte *bp;

   for(bp = buff; len; bp++, len--)
      if(*bp) return 0;

   return 1;
}


void b2hexch(byte *addr, int len, int lastchar)
{
   int n;

   for(n = 0; len; len--) {
      printf("%02x", *addr++);
      if(++n >= 36) {
         printf("\n");
         n = 0;
      }
   }
   if(lastchar)
      printf("%c", lastchar);
}


#include "../add64.c"
#include "../rand.c"


/***** Dummy test function
 * Return number of records read from tfile.dat.
 */
int readtf(void *buff, word32 bnum, word32 count)
{
   byte *bp;
   word32 n, save;

   save = srand16(bnum);
   for(bp = buff, n = count * sizeof(BTRAILER); n; n--, bp++)
      *bp = rand16();
   if(Trace > 1) plog("readtf() read %u trailers\n", count);
   srand16(save);
   return count;
}  /* end readtf() */


void add_weight2(byte *weight, byte difficulty)
{
   byte temp[32];

   memset(temp, 0, 32);
   /* temp = 2**difficulty */
   temp[difficulty / 8] = 1 << (difficulty % 8);
   multi_add(weight, temp, weight, 32);
}  /* end add_weight2() */


/* Reduce weight based on difficulty
 * Note: Only works above WTRIGGER31.
 */
int sub_weight(byte *weight, byte difficulty)
{
   byte temp[32];

   memset(temp, 0, 32);
   /* temp = 2**difficulty */
   temp[difficulty / 8] = 1 << (difficulty % 8);
   return multi_sub(weight, temp, weight, 32);
}  /* end sub_weight() */


/* Compute our weight at lownum and return in weight[]
 * Return VEOK on success, else error code.
 */
int past_weight(byte *weight, word32 lownum)
{
   word32 cbnum;
   int message;
   BTRAILER bts;

   cbnum = get32(Cblocknum);
   if(lownum >= cbnum) BAIL(1);
   memcpy(weight, Weight, 32);
   for( ; cbnum > lownum; cbnum--) {
      if(readtf(&bts, cbnum, 1) != 1) BAIL(2);
      printf("Trailer difficulty: %3d  weight at block %d:\n",
             bts.difficulty[0], cbnum - 1);  /* debug */
      sub_weight(weight, bts.difficulty[0]);
      b2hexch(weight, 32, '\n');  /* debug */
   }
   return VEOK;
bail:
   memset(weight, 0, 32);
   if(Trace) plog("past_weight(): bail: %d\n", message);
   return message;
}  /* end past_weight() */


int main()
{
   static byte weight[32];
   int j, high, low;
   BTRAILER bts;

   Cblocknum[0] = 0;
   printf("Weight at block %d:\n", Cblocknum[0]);
   b2hexch(Weight, 32, '\n');

   high = 100;
   for(j = 1; j <= high; j++) {
      readtf(&bts, j, 1);
      add_weight2(Weight, bts.difficulty[0]);
   }
   Cblocknum[0] = high;
   printf("Weight at block %d:\n", Cblocknum[0]);
   b2hexch(Weight, 32, '\n');
   memcpy(weight, Weight, 32);

   low = 0;
   past_weight(weight, low);
   printf("weight after regression to block"
          " %d after past_weight() call:\n", low);
   b2hexch(weight, 32, '\n');
   if(low == 0) {
      if(iszero(weight, 32))
         printf("Success!\n"); else printf("Error!\n");
   }
   return 0;
}
