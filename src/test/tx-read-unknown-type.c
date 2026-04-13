/**
 * Unit test for F-13 (issue #90): tx_read() / tx__init() must reject
 * OP_TX payloads that declare unsupported TXDAT_TYPE or TXDSA_TYPE
 * values without reading uninitialized offset variables.
 *
 * Before fix: tx__init() was void, and a payload with an unknown type
 * byte caused the switch to fall through without setting dsaoff /
 * tlroff, leaving them as uninitialized stack garbage that was then
 * used in pointer arithmetic and size-check computations.
 *
 * After fix: tx__init() returns int; default cases return VERROR;
 * tx_read() propagates the failure; unknown types yield a clean
 * VERROR rejection with no UB.
 */

#include <string.h>
#include <stdio.h>

#include "_assert.h"
#include "tx.h"
#include "types.h"

int main(void)
{
   TXENTRY txe;
   word8 buf[4096];
   int result;

   /* --- Case 1: valid TXDAT_TYPE + valid TXDSA_TYPE ---
    * Establish the baseline. A well-formed header with
    *   options[0] = 0x00 (TXDAT_MDST)
    *   options[1] = 0x00 (TXDSA_WOTS)
    *   options[2] = 0x00 (MDST_COUNT = 1)
    * should reach tx_read's size check and fail at the size check
    * (we are not supplying a full valid TX body), NOT reach any
    * uninitialized-variable path. Both before and after the fix this
    * should return VERROR with errno EMCM_TXINVAL. */
   memset(&txe, 0, sizeof(txe));
   memset(buf, 0, sizeof(buf));
   buf[0] = TXDAT_MDST;
   buf[1] = TXDSA_WOTS;
   buf[2] = 0x00;
   result = tx_read(&txe, buf, sizeof(TXHDR));
   ASSERT_EQ_MSG(result, VERROR,
      "baseline: valid types with header-only buffer should fail size check");

   /* --- Case 2: unknown TXDAT_TYPE ---
    * Before fix: dsaoff uninitialized; tlroff derived from it is
    * also garbage; tx->tx_sz is garbage; size check outcome is
    * platform/compiler-dependent.
    * After fix: tx__init returns VERROR at the default branch of the
    * TXDAT switch; tx_read propagates it. Clean VERROR. */
   memset(&txe, 0, sizeof(txe));
   memset(buf, 0, sizeof(buf));
   buf[0] = 0x01;          /* unknown */
   buf[1] = TXDSA_WOTS;
   buf[2] = 0x00;
   result = tx_read(&txe, buf, sizeof(TXHDR));
   ASSERT_EQ_MSG(result, VERROR,
      "unknown TXDAT_TYPE must yield VERROR");

   /* --- Case 3: unknown TXDSA_TYPE ---
    * Similar to case 2 but with valid TXDAT and invalid TXDSA.
    * Before fix: tlroff uninitialized; tx_sz garbage.
    * After fix: VERROR at the default of the TXDSA switch. */
   memset(&txe, 0, sizeof(txe));
   memset(buf, 0, sizeof(buf));
   buf[0] = TXDAT_MDST;
   buf[1] = 0x01;          /* unknown */
   buf[2] = 0x00;
   result = tx_read(&txe, buf, sizeof(TXHDR));
   ASSERT_EQ_MSG(result, VERROR,
      "unknown TXDSA_TYPE must yield VERROR");

   /* --- Case 4: both type bytes unknown ---
    * Before fix: both offsets uninitialized.
    * After fix: early VERROR on the TXDAT default. */
   memset(&txe, 0, sizeof(txe));
   memset(buf, 0, sizeof(buf));
   buf[0] = 0xFF;
   buf[1] = 0xFF;
   buf[2] = 0x00;
   result = tx_read(&txe, buf, sizeof(TXHDR));
   ASSERT_EQ_MSG(result, VERROR,
      "both unknown type bytes must yield VERROR");

   /* --- Case 5: valid TXDAT_TYPE but inflated MDST_COUNT ---
    * options[2] = 0xFF means MDST_COUNT = 256. dsaoff =
    * sizeof(TXHDR) + (sizeof(MDST) * 256) which is larger than the
    * TX buffer. This should not produce out-of-bounds pointer
    * arithmetic regardless of buffer contents.
    * Before fix: no bounds check; dsaoff/tlroff point outside buffer.
    * After fix: bounds check catches it and returns VERROR. */
   memset(&txe, 0, sizeof(txe));
   memset(buf, 0, sizeof(buf));
   buf[0] = TXDAT_MDST;
   buf[1] = TXDSA_WOTS;
   buf[2] = 0xFF;
   result = tx_read(&txe, buf, sizeof(TXHDR));
   ASSERT_EQ_MSG(result, VERROR,
      "oversized MDST_COUNT must yield VERROR");

   printf("[PASS] tx-read-unknown-type: all 5 cases rejected cleanly\n");
   return 0;
}
