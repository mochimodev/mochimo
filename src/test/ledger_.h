
#include "wots.h"
#include "ledger.h"
#include "error.h"
#include "extio.h"
#include "extlib.h"
#include <string.h>

#include "_assert.h"
#include "_testutils.h"

#define WOTS_SEEDp(addr) ((((word8 *) (addr)) + WOTSSIGBYTES))

LENTRY_W *random_ledgerw(size_t count)
{
   LENTRY_W *le;
   size_t buflen, i;
   word32 ADRS[8];
   word8 *tag;

   /* init */
   buflen = sizeof(*le) * count;
   le = malloc(buflen);

   /* generate WOTS+ addresses */
   for (i = 0; le && i < count; i++) {
      ADRS[0] = i;
      wots_pkgen(le[i].addr, (word8 *) ADRS, WOTS_SEEDp(le[i].addr), ADRS);
      ((word32 *) le[i].balance)[0] = i + MFEE + 1;
      if ((i % 2) == 0) {
         tag = WOTS_TAGp(le[i].addr);
         tag[0] = 0x01;
         *((word32 *) &tag[4]) = i;
      }
   }
   /* sort ledger entries (WOTS+) */
   if (le) qsort(le, count, sizeof(*le), le_cmpw);

   return le;
}  /* end random_ledgerw() */

int random_neogenw(char *fname, LENTRY_W *le, size_t count)
{
   FILE *fp;
   word32 hdrlen;
   BTRAILER bt = { 0 };

   /* init data */
   hdrlen = 4 + (sizeof(*le) * count);

   /* write data */
   if ((fp = fopen(fname, "wb")) == NULL) goto FAIL_IO;
   if (fwrite(&hdrlen, sizeof(hdrlen), 1, fp) != 1) goto FAIL_IO;
   if (fwrite(le, sizeof(*le), count, fp) != count) goto FAIL_IO;
   if (fwrite(&bt, sizeof(bt), 1, fp) != 1) goto FAIL_IO;

   /* cleanup */
   fclose(fp);

   return 0;

FAIL_IO:
   if (fp) fclose(fp);
   return 1;
}  /* end random_neogenw() */