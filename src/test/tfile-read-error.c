/**
 * Unit test for F-23 (issue #99): read_tfile() must return 0 on
 * error, not VERROR (==1), so that callers can distinguish "failed
 * to open file" from "successfully read one trailer".
 */

#include <string.h>
#include <stdio.h>

#include "_assert.h"
#include "tfile.h"
#include "types.h"

int main(void)
{
   BTRAILER bt;
   word8 bnum_zero[8] = {0};
   size_t n;

   /* --- Case 1: read from a file that does not exist ---
    * Before fix: returned VERROR (== 1), indistinguishable from a
    * legitimate one-trailer read.
    * After fix: returns 0, which callers that check != 1 or <= 0
    * correctly identify as an error. */
   (void) remove("nonexistent_tfile.dat");
   n = read_tfile(&bt, bnum_zero, 1, "nonexistent_tfile.dat");
   ASSERT_EQ_MSG(n, 0,
      "read_tfile() on a missing file must return 0 (not VERROR=1)");

   /* --- Case 2: short read (request more than the file contains) ---
    * Write a tfile containing exactly one trailer, then ask for 3.
    * Result: n == 1, which is both a legitimate partial read AND
    * the old VERROR value. Errno differentiates (EMCM_EOF). */
   {
      FILE *fp = fopen("short_tfile.dat", "wb");
      BTRAILER trailer;
      memset(&trailer, 0, sizeof(trailer));
      ASSERT_NE(fp, NULL);
      if (fwrite(&trailer, sizeof(BTRAILER), 1, fp) != 1) {
         fprintf(stderr, "fwrite failed\n");
         fclose(fp);
         return 1;
      }
      fclose(fp);
   }
   n = read_tfile(&bt, bnum_zero, 3, "short_tfile.dat");
   ASSERT_EQ_MSG(n, 1,
      "read_tfile() must return the actual record count on a partial read");
   (void) remove("short_tfile.dat");

   /* --- Case 3: seek past end (bnum beyond file) ---
    * Tfile has 1 trailer (bnum 0); we ask to start reading at bnum 5.
    * fseek succeeds (past-end is allowed), fread returns 0.
    * Before fix: still returned 0 (the happy path), BUT fopen-failure
    * also returned VERROR=1 which collided with the real "1 record"
    * case elsewhere. Case 3 itself exercises the normal-path 0 return
    * which was already correct. */
   {
      FILE *fp = fopen("tiny_tfile.dat", "wb");
      BTRAILER trailer;
      memset(&trailer, 0, sizeof(trailer));
      ASSERT_NE(fp, NULL);
      if (fwrite(&trailer, sizeof(BTRAILER), 1, fp) != 1) {
         fclose(fp);
         return 1;
      }
      fclose(fp);
   }
   {
      word8 bnum_past[8] = { 5, 0, 0, 0, 0, 0, 0, 0 };
      n = read_tfile(&bt, bnum_past, 1, "tiny_tfile.dat");
   }
   ASSERT_EQ_MSG(n, 0,
      "read_tfile() past EOF must return 0");
   (void) remove("tiny_tfile.dat");

   printf("[PASS] tfile-read-error: all 3 cases behave correctly\n");
   return 0;
}
