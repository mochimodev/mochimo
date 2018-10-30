/* sha2.c  SHA-256 command
 *
 * SHA-256 code after Brad Conte:
 * https://raw.githubusercontent.com/B-Con/crypto-algorithms/master/sha256.c
 *
 *
 * NOTE:   requires sha256.h and sha256.c
 *         cc -c sha256.c
 *         cc shatest.c sha256.obj
 *         cc sha2.c sha256.obj
 *
 * Copyright (c) 2018 by Adequate Systems, LLC.  All Rights Reserved.
 * See LICENSE.PDF   **** NO WARRANTY ****
 *
 * Date: 17 February 2018
 *
 */

#include <stdio.h>
#include <string.h>
#include "sha256.h"

#ifdef  __TURBOC__   /* or any DOS Compiler */
#include <io.h>
#include <fcntl.h>   /* for setmode() */
#else
#define setmode(fd, mode)
#endif


void printhash(byte *hash, int binaryflag)
{
   int j;

   if(binaryflag) {
      fwrite(hash, 1, 32, stdout);
      return;
   }
   printf("0x");
   for(j = 0; j < 32; j++)
      printf("%02x", *hash++);
   printf("\n");
}


void usage(void)
{
   printf("Display SHA-256 hash of file(s)\n"
          "usage: sha2 [-b] filename...\n"
          "       sha2 [-b] -s string\n"
          "       -b send raw binary output to stdout\n"
   );
   exit(1);
}


int main(int argc, char **argv)
{
   static byte iobuff[30*1024];
   FILE *fp;
   int j;
   static int binaryflag;
   static SHA256_CTX ctx;
   static byte hash[SHA256_BLOCK_SIZE];

   if(argc < 2) usage();

   for( ; argv[1] && argv[1][0] == '-'; argv++) {
      if(argv[1][1] == 's') {
         if(argv[2] == NULL) usage();
         sha256(argv[2], strlen(argv[2]), hash);
         printhash(hash, binaryflag);
         return 0;
      }
      if(argv[1][1] == 'b') {
         setmode(fileno(stdout), O_BINARY);  /* for MS-DOS */
         binaryflag = 1;
         continue;
      }
   }

   for( ; argv[1]; argv++) {

      fp = fopen(argv[1], "rb");
      if(!fp) {
         printf("Cannot open %s\n", argv[1]);
         return 1;
      }

      sha256_init(&ctx);

      for(;;) {
         j = fread(iobuff, 1, sizeof(iobuff), fp);
         if(j < 1) break;
         sha256_update(&ctx, iobuff, j);
      }
      fclose(fp);
      sha256_final(&ctx, hash);
      printhash(hash, binaryflag);
   }  /* end for */
   return 0;
}
