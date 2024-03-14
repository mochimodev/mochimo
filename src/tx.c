/**
 * @private
 * @headerfile tx.h <tx.h>
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
*/

/* include guard */
#ifndef MOCHIMO_TX_C
#define MOCHIMO_TX_C


#include "tx.h"

/* internal support */
#include "wots.h"
#include "trigg.h"
#include "tag.h"
#include "peach.h"
#include "ledger.h"
#include "global.h"
#include "error.h"

/* external support */
#include <sys/wait.h>
#include <string.h>
#include "exttime.h"
#include "extmath.h"
#include "extlib.h"
#include <ctype.h>
#include "crc16.h"

/**
 * @private Transaction Position structure.
 * Contains an ID and file position type pair.
*/
typedef struct {
   word8 id[HASHLEN];
   fpos_t pos;
} TXPOS;

/**
 * @private
 * Comparison function to sort TXPOS objects by id.
*/
static int txpos_compare(const void *va, const void *vb)
{
   TXPOS *a = (TXPOS *) va;
   TXPOS *b = (TXPOS *) vb;

   return memcmp(a->id, b->id, sizeof(a->id));
}

#ifndef _WIN32

   /* Get exclusive lock on lockfile.
   * Returns: -1 if lock not made within 'seconds' or open() failed
   *          else a descriptor to be used with unlock()
   */
   int lock(char *lockfile, int seconds)
   {
      time_t timeout;
      int fd, status;

      timeout = time(NULL) + seconds;
      fd = open(lockfile, O_NONBLOCK | O_RDONLY);
      if(fd == -1) return -1;
      for(;;) {
         status = flock(fd, LOCK_EX | LOCK_NB);
         if(status == 0) return fd;
         if(time(NULL) >= timeout) {
            close(fd);
            return -1;
         }
      }
   }

   /* Unlock a decriptor returned from lock() */
   int unlock(int fd)
   {
      int status;

      status =  flock(fd, LOCK_UN);
      close(fd);
      return status;
   }

#endif

/**
 * Derive Transaction Entry, @a txe, and eXtended Data, @a xdata, parts
 * from a transaction data buffer received from the network.
 * @note DOES NOT COPY tx_nonce and tx_id. This function is designed for
 * use on a transaction buffer received directly from the network, where
 * the nonce and id are NOT generally provided.
 * @param txe Pointer to put Transaction Entry data
 * @param xdata Pointer to put eXtended Data
 * @param buffer Pointer to buffer to derive transaction parts from
 * @param bufsz Length, in bytes, of provided @a buffer
 * @return (int) value representing the result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 */
int tx_data(TXQENTRY *txe, XDATA *xdata, const void *buffer, size_t bufsz)
{
   word8 *xbuffer;
   size_t len, total;

   /* check buffer contains at least TXQENTRY (excl. nonce and id) */
   total = sizeof(TXQENTRY) - 8 - HASHLEN;
   if (bufsz < total) {
      set_errno(EMCM_TXINVAL);
      return VERROR;
   }

   xbuffer = ((word8 *) buffer) + (TXADDRLEN * 2);
   /* determine if eXtended Data is present */
   if (IS_XTX(buffer)) {
      switch (XTX_TYPE(buffer)) {
         /* ... add eXtended Transaction data types here */
         case XTX_MDST:
            /* infer +1 MDST count due to byte limitations */
            len = ((size_t) XTX_COUNT(buffer) + 1) * sizeof(MDST);
            break;
         default:
            /* no eXtended Data available */
            len = 0;
      }
      if (len > 0) {
         /* check buffer contains aditional xdata (len) */
         if (bufsz < (total + len)) {
            set_errno(EMCM_TXINVAL);
            return VERROR;
         }
         /* copy xdata (if provided) and adjust buffer */
         if (xdata != NULL) memcpy(xdata, xbuffer, len);
         xbuffer = xbuffer + len;
      }
   }

   /* copy core transaction data */
   memcpy(txe, buffer, (TXADDRLEN * 2));
   memcpy(txe->chg_addr, xbuffer, total - (TXADDRLEN * 2));

   return VEOK;
}  /* end tx_data() */

/**
 * Read a single Transaction and eXtended Data entry in to the provided
 * buffers, @a txe and @a xdata, from the given input @a stream. The
 * file position indicator is advanced by the size of the whole entry.
 * @param txe Pointer to Transaction Entry buffer
 * @param xdata Pointer to eXtended Data buffer, or NULL
 * @param stream The stream to read from
 * @return (int) value representing the read result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 */
int tx_fread(TXQENTRY *txe, XDATA *xdata, FILE *stream)
{
   size_t len, res;

   /* read up to transaction change address to check for MDST */
   res = fread(txe, (TXADDRLEN * 2), 1, stream);
   if (res != 1) return VERROR;

   /* determine if eXtended Data is present */
   if (IS_XTX(txe)) {
      switch (XTX_TYPE(txe)) {
         /* ... add eXtended Transaction data types here */
         case XTX_MDST:
            /* infer +1 MDST count due to byte limitations */
            len = ((size_t) XTX_COUNT(txe) + 1) * sizeof(MDST);
            break;
         default:
            /* no eXtended Data available */
            len = 0;
      }
      /* read eXtended Data or skip if xdata is NULL */
      if (len > 0) {
         if (xdata == NULL) {
            res = (size_t) fseek64(stream, (long long) len, SEEK_CUR);
            if (res != 0) return VERROR;
         } else {
            res = fread(xdata, len, 1, stream);
            if (res != 1) return VERROR;
         }
      }
   }

   /* read remaining transaction data (from chg_addr) */
   len = sizeof(TXQENTRY) - (TXADDRLEN * 2);
   res = fread(txe->chg_addr, len, 1, stream);
   if (res != 1) return VERROR;

   return VEOK;
}  /* end tx_fread() */

/**
 * Write a single Transaction and eXtended Data entry from the provided
 * buffers, @a txe and @a xdata, to the given input @a stream. The file
 * position indicator is advanced by the size of the whole entry. If the
 * provided Transaction entry does not indicate an eXtended Transaction
 * entry that contains eXtended Data, it MAY be left NULL.
 * @param txe Pointer to Transaction Entry buffer
 * @param xdata Pointer to eXtended Data buffer, or NULL
 * @param stream The stream to read from
 * @return (int) value representing the write result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 * @exception errno=ENODATA Write requires xdata, which was not provided
 */
int tx_fwrite(TXQENTRY *txe, XDATA *xdata, FILE *stream)
{
   size_t len, res;

   /* write transaction data up to possible eXtended Data */
   res = fwrite(txe, (TXADDRLEN * 2), 1, stream);
   if (res != 1) return VERROR;

   /* determine if eXtended Data is present */
   if (IS_XTX(txe)) {
      switch (XTX_TYPE(txe)) {
         /* ... add eXtended Transaction data types here */
         case XTX_MDST:
            /* infer +1 MDST count due to byte limitations */
            len = ((size_t) XTX_COUNT(txe) + 1) * sizeof(MDST);
            break;
         default:
            /* no eXtended Data available */
            len = 0;
      }
      /* read eXtended Data or skip if xdata is NULL */
      if (len > 0) {
         if (xdata == NULL) {
            set_errno(ENODATA);
            return VERROR;
         } else {
            res = fwrite(xdata, len, 1, stream);
            if (res != 1) return VERROR;
         }
      }
   }

   /* write remaining transaction data */
   len = sizeof(TXQENTRY) - (TXADDRLEN * 2);
   res = fwrite(txe->chg_addr, len, 1, stream);
   if (res != 1) return VERROR;

   return VEOK;
}  /* end tx_fwrite() */

/**
 * Hash Transaction Entry, @a txe, and eXtended Data, @a xdata, parts per
 * their intended buffer structure as if found within a blockchain file.
 * @param txe Pointer to Transaction Entry data
 * @param xdata Pointer to eXtended Data
 * @param full Set non-zero for a "full" Transaction ID hash or
 * set zero for a Transaction Signature Message hash.
 * @param out Pointer to place finalized hash
 */
void tx_hash(TXQENTRY *txe, XDATA *xdata, int full, void *out)
{
   SHA256_CTX ctx;
   size_t len;

   sha256_init(&ctx);

   /* update hash with pre-xtx transaction data */
   sha256_update(&ctx, txe->src_addr, TXADDRLEN);
   sha256_update(&ctx, txe->dst_addr, TXADDRLEN);

   /* determine if eXtended Data is present */
   if (IS_XTX(txe)) {
      switch (XTX_TYPE(txe)) {
         /* ... add eXtended Transaction data types here */
         case XTX_MDST:
            /* infer +1 MDST count due to byte limitations */
            len = ((size_t) XTX_COUNT(txe) + 1) * sizeof(MDST);
            break;
         default:
            /* no eXtended Data available */
            len = 0;
      }
      /* update hash with transaction eXtended Data */
      sha256_update(&ctx, xdata, len);
   }

   /* update hash with remaining transaction data */
   if (full) {
      sha256_update(&ctx, txe->chg_addr, txe->tx_id - txe->chg_addr);
   } else sha256_update(&ctx, txe->chg_addr, txe->tx_sig - txe->chg_addr);

   /* finalize */
   sha256_final(&ctx, out);
}  /* end tx_hash() */

/**
 * @private
 * Validate a 32 byte eXtended Transaction reference field.
 * @param buffer Pointer to start of 32 byte reference buffer
 * @return VEOK on success, or VERROR or error; check errno for details
 */
static int xtx_ref_val(void *buffer)
{
   char *reference;
   int j;

   /* define states for stages of validation */
   enum { START, DIGIT_DASH, DIGIT, UPPER_DASH, UPPER, ZERO } state;

   /* Validation format rules (from types.h):
    * - CONTAINS only uppercase [A-Z], digit [0-9], dash [-], null [\0]
    * - SHALL be null terminated with remaining unused bytes zeroed
    *   - (e.g. VALID   `(char[32]) { 'A','-','1','\0','\0','\0', ... }`)
    *   - (e.g. INVALID `(char[32]) { 'A','-','1','\0','B','\0', ... }`)
    * - MAY have multiple uppercase OR digits (NOT both) grouped together
    * - SHALL only contain a dash to separate groups of uppercase or digit
    * - SHALL NOT contain consecutive groups of the same group type
    *   - (e.g. VALID   "AB-00-EF", "123-CDE-789", "ABC", "123")
    *   - (e.g. INVALID "AB-CD-EF", "123-456-789", "ABC-", "-123")
    */

   reference = (char *) buffer;
   /* ensure null termination */
   if (reference[HASHLEN - 1] != '\0') {
      return VERROR;
   }

   /* validate reference format */
   for (state = START, j = 0; j < HASHLEN; j++) {
      /* state determines the next allowed characters */
      switch (state) {
         /* NOTE: "continue" here is associated with for() loop */
         case START:  /* allow either null, digit, or uppercase */
            if (reference[j] == '\0') { state = ZERO; continue; }
            if (isdigit(reference[j])) { state = DIGIT; continue; }
            /* fallthrough */
         case DIGIT_DASH:  /* allow only uppercase (follows "[0-9]-") */
            if (isupper(reference[j])) { state = UPPER; continue; }
            break;  /* switch() */
         case UPPER_DASH:  /* allow only numeric (follows "[A-Z]-") */
            if (isdigit(reference[j])) { state = DIGIT; continue; }
            break;  /* switch() */
         case DIGIT:  /* allow either numeric, dash, or ZERO */
            if (isdigit(reference[j])) continue;  /* for() */
            if (reference[j] == '-') { state = UPPER_DASH; continue; }
            if (reference[j] == '\0') { state = ZERO; continue; }
            break;  /* switch() */
         case UPPER:  /* allow either uppercase, dash, or ZERO */
            if (isupper(reference[j])) continue;  /* for() */
            if (reference[j] == '-') { state = DIGIT_DASH; continue; }
            if (reference[j] == '\0') { state = ZERO; continue; }
            break;  /* switch() */
         case ZERO:  /* allow only ZERO (end of reference) */
            if (reference[j] == '\0') continue;  /* for() */
      }  /* end switch(state) */

      /* no valid character for current state */
      return VERROR;
   }  /* end for() */

   /* reference valid */
   return VEOK;
}  /* end xtx_ref_val() */

/**
 * @private
 * Validate a Multi-Destination Transaction (incl. reference field).
 * @param txe Pointer to Transaction Entry
 * @param mdst Pointer to Multi-Destination array
 * @return (int) value representing the validation result
 * @retval VEBAD2 on bad signature; check errno for details
 * @retval VEBAD on bad transaction; check errno for details
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 */
static int xtx_mdst_val(TXQENTRY *txe, MDST *mdst)
{
   word8 addr[TXADDRLEN];
   word8 total[8], mfees[8];
   int j, count;

   /* check valid reference format */
   if (xtx_ref_val((char *) txe->dst_addr) != VEOK) {
      set_errno(EMCM_XTXREF);
      return VEBAD;
   }

   memset(total, 0, 8);
   memset(mfees, 0, 8);
   memset(addr, 0, TXADDRLEN);
   count = (int) XTX_COUNT(txe) + 1;
   /* Tally each dst[] amount and mfees... */
   for (j = 0; j < count; j++) {
      if (iszero(mdst[j].amount, 8)) {
         set_errno(EMCM_XTXDSTAMOUNT);
         return VEBAD;
      }
      /* no dst to src */
      if (tag_equal(mdst[j].tag, ADDR_TAG_PTR(txe->src_addr))) {
         set_errno(EMCM_XTXTAGMATCH);
         return VEBAD;
      }
      /* tally fees and send_total */
      if (add64(total, mdst[j].amount, total)) {
         set_errno(EMCM_XTXTOTALS);
         return VEBAD;
      }
      if (add64(mfees, Myfee, mfees)) {
         set_errno(EMCM_XTXFEES);
         return VEBAD;
      }
      /* dst tag MUST exist in the ledger */
      memcpy(ADDR_TAG_PTR(addr), mdst[j].tag, TXTAGLEN);
      if (!tag_find(addr, NULL, NULL, TXTAGLEN)) {
         set_errno(EMCM_XTXTAGNOLE);
         return VERROR;
      }
   }
   /* Check tallies... */
   if (cmp64(total, txe->send_total) != 0) {
      set_errno(EMCM_XTXTOTALS);
      return VEBAD;
   }
   if (cmp64(txe->tx_fee, mfees) < 0) {
      set_errno(EMCM_XTXFEES);
      return VEBAD;
   }

   return VEOK;  /* valid */
}  /* end xtx_mdst_val() */

/**
 * Validate a Hashed transaction. Requires an open ledger.
 * @param txe Pointer to transaction entry to validate
 * @param xdata Pointer to eXtended Data to validate
 * @param bnum Pointer to block number to validate against
 * @return (int) value representing validation result
 * @retval VEBAD2 on invalid signature; check errno for details
 * @retval VEBAD on bad transaction data; check errno for details
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 */
int tx_val(TXQENTRY *txe, XDATA *xdata, word8 *bnum)
{
   LENTRY le;
   word8 message[HASHLEN];    /* for signature validation */
   word8 addr_hash[HASHLEN];  /* for signature validation */
   word8 wots[WOTSSIGBYTES];  /* for WOTS signature validation */
   word32 adrs[HASHLEN / 4];  /* for WOTS signature validation */
   word32 total[2];           /* for 64-bit maths */
   int overflow;

   /* only non-zero block-to-live values are checked */
   if (!iszero(txe->tx_btl, 8)) {
      /* prepare block-to-live upper bound */
      if (add64(bnum, (word16[4]) { 0x100 }, total)) {
         set_errno(EMCM_MATH64_OVERFLOW);
         return VERROR;
      }
      /* check block number against block-to-live range */
      if (cmp64(txe->tx_btl, bnum) < 0 || cmp64(txe->tx_btl, total) > 0) {
         set_errno(EMCM_TXBTL);
         return VEBAD;
      }
   }
   /* check block number matches nonce */
   if (cmp64(txe->tx_nonce, bnum) != 0) {
      set_errno(EMCM_TXNONCE);
      return VEBAD;
   }

   /* validate src != chg */
   if (addr_compare(txe->src_addr, txe->chg_addr) == 0) {
      set_errno(EMCM_TXCHG);
      return VEBAD;
   }

   /* validate transaction fee -- implicit MFEE check (Myfee >= MFEE) */
   if (cmp64(txe->tx_fee, Myfee) < 0) {
      set_errno(EMCM_TXFEE);
      return VEBAD;
   }

   /* for XTX type transactions... */
   if (IS_XTX(txe)) {
      /* ... source address MUST be tagged */
      if (!ADDR_HAS_TAG(txe->src_addr)) {
         set_errno(EMCM_XTXSRCNOTAG);
         return VEBAD;
      }
      /* ... source address tag MUST equal change address tag */
      if (!addr_tag_equal(txe->src_addr, txe->chg_addr)) {
         set_errno(EMCM_XTXTAGMISMATCH);
         return VEBAD;
      }
      /* ... change total MUST be > MFEE, so the address is not dissolved */
      /** @todo revisit reasoning for this last check */
      if (cmp64(txe->change_total, Mfee) <= 0) {
         set_errno(EMCM_XTXCHGTOTAL);
         return VEBAD;
      }
   } else {
   /* ... OR, for non-XTX transactions... */

      /* ... check src != dst */
      if (addr_compare(txe->src_addr, txe->dst_addr) == 0) {
         set_errno(EMCM_TXDST);
         return VEBAD;
      }
/* >> BEGIN OLD TAG VALIDATION */
      /* ... validate destination tag */
      if (ADDR_HAS_TAG(txe->dst_addr)) {
         /* check full dst_addr is in ledger, else invalid */
         if (le_find(txe->dst_addr, &le, TXADDRLEN) == 0) {
            set_errno(EMCM_TXDSTNOLE);
            return VERROR;
         }
      }
      /* if change tag exists and source tag != change tag, check... */
      if (ADDR_HAS_TAG(txe->chg_addr)) {
         if (!addr_tag_equal(txe->src_addr, txe->chg_addr)) {
            /* ... if src is not default, tx is invalid */
            if (ADDR_HAS_TAG(txe->src_addr)) {
               set_errno(EMCM_TXTAGSRC);
               return VEBAD;
            }
            /* ... if change tag is in ledger.dat, tx is invalid */
            if (tag_find(txe->chg_addr, NULL, NULL, TXTAGLEN) == 1) {
               set_errno(EMCM_TXTAGCHG);
               return VERROR;
            }
         }
      }
      /* chg_addr tag queue conflict checks that would normally be done
       * here were moved into txq_check() for convenience/efficiency */
/* >> END OLD TAG VALIDATION */
   }  /* end if (IS_XTX(txe))... else... */

   /* generate transaction signature message */
   tx_hash(txe, xdata, 0, message);

   /* determine appropriate DSA type */
   switch (DSA_TYPE(txe)) {
      case DSA_WOTS:
         /* recreate WOTS+ public key from signature */
         memcpy(adrs, txe->tx_adrs, HASHLEN);
         wots_pk_from_sig(wots, txe->tx_sig, message, txe->tx_seed, adrs);
         /* NOTE: next check prevents free floating bytes in tx_adrs */
         if (memcmp(txe->tx_adrs, adrs, HASHLEN) != 0) {
            set_errno(EMCM_TXADRS);
            return VEBAD2;
         }
         /* validate hashed address against source */
         sha256(wots, WOTSSIGBYTES, addr_hash);
         if (memcmp(addr_hash, txe->src_addr, HASHLEN) != 0) {
            set_errno(EMCM_TXWOTS);
            return VEBAD2;
         }
         break;
      default:
         /* includes DSA_NONE */
         set_errno(EMCM_TXDSA);
         return VEBAD2;
   }  /* end switch (DSA_TYPE(txe)) */

   /* look up source address in ledger */
   if (!le_find(txe->src_addr, &le, TXADDRLEN)) {
      set_errno(EMCM_TXSRCLE);
      return VERROR;
   }
   total[0] = total[1] = 0;
   /* use add64() to check for overflow */
   overflow =  add64(txe->send_total, txe->change_total, total);
   overflow += add64(txe->tx_fee, total, total);
   if (overflow) {
      set_errno(EMCM_TXOVERFLOW);
      return VEBAD;
   }
   /* check totals match ledger balance */
   if (cmp64(le.balance, total) != 0) {
      set_errno(EMCM_TXTOTAL);
      return VERROR;
   }

   /* return validation result for eXtended Transaction types... */
   if (IS_XTX(txe)) {
      switch (XTX_TYPE(txe)) {
         case XTX_MDST:
            return xtx_mdst_val(txe, xdata->mdst);
         default:
            /* includes XTX_NONE */
            set_errno(EMCM_XTXUNDEF);
            return VEBAD;
      }
   }

   /* transaction is valid */
   return VEOK;
}  /* end tx_val() */

/**
 * Search txq1.dat and txclean.dat for conflicts with the src_addr or
 * (when applicable) chg_addr tags, of queued transactions.
 * @param src_addr Pointer to source address
 * @param chg_addr Pointer to change address
 * @return (int) value representing the result
 * @retval VERROR on conflict; check errno for details
 * @retval VEOK on success
 */
int txcheck(word8 *src_addr, word8 *chg_addr)
{
   FILE *fp;
   TXQENTRY txe;
   int chg_chk;

   /* determine requirement for additional chg_addr checks */
   chg_chk = ADDR_HAS_TAG(chg_addr) && !addr_tag_equal(src_addr, chg_addr);
   /* NOTE: chg_addr tag conflict checks would not usually be required
    * until the tag validation stage of a transaction validation routine,
    * however it feels particularly convenient/efficient to perform the
    * check at the same time as the src_addr conflict checks
    */

   /* read transaction in txq1 checkiung for conflicts */
   fp = fopen("txq1.dat", "rb");
   if (fp != NULL) {
      while (tx_fread(&txe, NULL, fp) == VEOK) {
         if (memcmp(tx.src_addr, src_addr, TXADDRLEN) == 0) {
            /* source address conflict */
            set_errno(EMCM_TXSRCDUP);
            goto FAIL;
         }
         if (chg_chk) {
            if (addr_tag_equal(txe.chg_addr, chg_addr)) {
               /* change address tag conflict */
               set_errno(EMCM_TXCHGTAGDUP);
               goto FAIL;
            }
         }
      }
      /* error check file and close*/
      if (ferror(fp)) goto FAIL;
      fclose(fp);  /* EOF */
   }  /* end if fp */

   /* duplicate search routine (as above) for txclean */
   fp = fopen("txclean.dat", "rb");
   if (fp != NULL) {
      while (tx_fread(&txe, NULL, fp) == VEOK) {
         if (memcmp(tx.src_addr, src_addr, TXADDRLEN) == 0) {
            /* source address conflict */
            set_errno(EMCM_TXSRCDUP);
            goto FAIL;
         }
         if (chg_chk) {
            if (addr_tag_equal(txe.chg_addr, chg_addr)) {
               /* change address tag conflict */
               set_errno(EMCM_TXCHGTAGDUP);
               goto FAIL;
            }
         }
      }
      /* error check file and close*/
      if (ferror(fp)) goto FAIL;
      fclose(fp);  /* EOF */
   }  /* end if fp */

   /* no conflicts found */
   return VEOK;

   /* cleanup / error handling */
FAIL:
   fclose(fp);

   return VERROR;
}  /* end txcheck() */

/**
 * Clean a Transaction Queue file, @a txfname, of entries that may have
 * been invalidated by an updated Ledger or may have been solved into a
 * recent Blockchain file, @a bcfname. If a Blockchain file is not 
 * provided, a Ledger-only clean will be performed.
 * @param txfname Filename of the clean transaction queue file
 * @param bcfname Filename of the block to clean against, or NULL
 * @return (int) value representing the clean result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 */
int txclean(const char *txfname, const char *bcfname)
{
   TXQENTRY txe, txc;      /* block entry and txclean transactions */
   XDATA xdata;            /* eXtended transaction data (for transfer) */
   FILE *fp, *bfp, *tfp;   /* input, blockchain and temporary files */
   TXPOS *tx;              /* malloc'd transaction positions */
   size_t count, actual;   /* malloc'd and actual element counts */
   size_t j, nout;
   fpos_t pos;             /* file position offset indicator */
   long long offset;       /* file position offset value */
   word32 hdrlen;          /* for block header length */
   int cond;
   char tmpfname[FILENAME_MAX];

   /* ensure ledger is open (required) */
   if (le_open("ledger.dat", "rb") != VEOK) {
      perr("failed to le_open(ledger.dat)");
      return VERROR;
   }

   /* GENERATE SORTED (ASCENDING) TXID REFERENCES FOR COMPARE */

   /* open provided txclean file */
   fp = fopen(txfname, "rb");
   if (fp == NULL) return VERROR;

   /* obtain EOF offset */
   if (fseek64(fp, 0LL, SEEK_END) != 0) goto FAIL_FP;
   offset = ftell64(fp);
   if (offset == EOF) goto FAIL_FP;
   if (offset == 0LL) {
      /* nothing to clean */
      fclose(fp);
      return VEOK;
   }
   if ((size_t) offset < sizeof(TXQENTRY)) {
      /* file contains unknown data */
      set_errno(EMCM_FILEDATA);
      goto FAIL_FP;
   }

   /* malloc required space (approximate) */
   count = (size_t) offset / sizeof(TXQENTRY);
   tx = malloc(count * sizeof(TXPOS));
   if (tx == NULL) goto FAIL_FP;

   /* store txid and associated fpos_t value in arrays */
   for(rewind(fp), actual = 0; actual < count; actual++) {
      /* store position for later use (if tx is read) */
      if (fgetpos(fp, &pos) != 0) goto FAIL_FP_MEM;
      if (tx_fread(&txc, NULL, fp) != VEOK) {
         if (ferror(fp)) goto FAIL_FP_MEM;
         break;  /* EOF */
      }
      /* set txid reference data */
      memcpy(&(tx[actual].id), txc.tx_id, HASHLEN);
      tx[actual].pos = pos;
   }
   /* check for leftover data */
   if (ftell64(fp) < offset) {
      set_errno(EMCM_FILEDATA);
      goto FAIL_FP_MEM;
   }
   /* sort the txid reference array */
   qsort(tx, actual, sizeof(TXPOS), txpos_compare);

   /* PREPARE BLOCKCHAIN FILE FOR TRANSACTION COMPARISON (IF PROVIDED) */

   bfp = NULL;
   /* NOTE: bfp MUST BE initialized NULL for error handling */
   if (bcfname != NULL) {
      /* open validated block file */
      bfp = fopen(bcfname, "rb");
      if (bfp == NULL) goto FAIL_FP_MEM;
      /* read and check fixed length header */
      if (fread(&hdrlen, 4, 1, bfp) != 1) {
         if (!ferror(fp)) set_errno(EMCM_EOF);
         goto FAIL_FP_MEM_BFP;
      }
      if (hdrlen != sizeof(BHEADER)) {
         set_errno(EMCM_HDRLEN);
         goto FAIL_FP_MEM_BFP;
      }
      /* seek to start of Merkle Array */
      if (fseek(bfp, (long) hdrlen, SEEK_SET) != 0) {
         goto FAIL_FP_MEM_BFP;
      }
   }

   /* FILTER OLD QUEUE BASED ON BLOCK AND LEDGER */

   /* generate temporary filename */
   snprintf(tmpfname, sizeof(tmpfname), "txc-%04x.tmp", rand16());
   tfp = fopen(tmpfname, "wb");
   if (tfp == NULL) goto FAIL_FP_MEM_BFP;

   /* Remove TX_ID's from clean TX queue that are in the new block.
    * Remove remaining invalidated (per ledger) transactions.
    * Merkel Array in new block is sorted on TX_ID (checked in bval).
    * Clean queue, txclean.dat, is sorted by reference array above.
    *
    * NOTE: end of file check on Merkel Block (bfp) depends on block
    *       trailer being shorter than a TXQENTRY !
    *       STATIC_ASSERT() prevents this.
    */
   STATIC_ASSERT(sizeof(BTRAILER) >= sizeof(TXQENTRY), "LOOP EOF CHECK");
   for (cond = j = nout = 0; j < actual; j++) {
      if (bfp != NULL) {
         do {
            /* if block transaction ID compared AFTER reference, hold... */
            if (cond <= 0) {
               /* read next transaction from block */
               if (tx_fread(&txe, NULL, bfp) != VEOK) {
                  if (ferror(bfp)) goto FAIL_ALL;
                  /* EOF -- break inner loop */
                  fclose(bfp);
                  bfp = NULL;
                  break;
               }
            }
            /* compare block transaction ID with reference ID */
            cond = memcmp(txe.tx_id, tx[j].id, HASHLEN);
            /* if ID from block compares BEFORE reference, redo... */
         } while (cond < 0);
         /* if ID from block compares EQUAL TO reference, skip... */
         if (bfp != NULL && cond == 0) continue;
      }
      /* .. else; read reference transaction from previously set fpos */
      if (fsetpos(fp, &(tx[j].pos)) != 0) goto FAIL_ALL;
      if (tx_fread(&txc, &xdata, fp) != VEOK) goto FAIL_ALL;
      /* if (re)validation fails, skip... */
      /** @todo: replace tx_val with less wasteful tx_reval process */
      if (tx_val(&txc, &xdata, Cblocknum) != VEOK) continue;
      /* ... else; write clean (valid) transaction to output */
      if (tx_fwrite(&txc, &xdata, tfp) != VEOK) goto FAIL_ALL;
      nout++;
   }  /* end for() */

   /* cleanup */
   fclose(tfp);
   fclose(bfp);
   fclose(fp);
   free(tx);

   /* out with the old, in with the new */
   remove(txfname);
   if (nout > 0) {
      if (rename(tmpfname, txfname) != 0) {
         remove(tmpfname);
         return VERROR;
      }
   }

   /* success */
   return VEOK;

   /* cleanup / error handling */
FAIL_ALL:
   fclose(tfp);
   remove(tmpfname);
FAIL_FP_MEM_BFP:
   if (bfp != NULL) fclose(bfp);
FAIL_FP_MEM:
   free(tx);
FAIL_FP:
   fclose(fp);

   return VERROR;
}  /* end txclean() */

/* Add src_ip to tx address map (weight[])
 * Called from process_tx()
 * Returns VERROR if no space in map, else VEOK.
 */
static int txmap(TX *tx, word32 src_ip)
{
   int j;
   word32 *ipp;

   /* Apply Matt's Algorithm v1.0 to control mirroring... */
   if (tx->version[1] & C_OPTIN) {
      /* try to put src_ip in map */
      for(ipp = (word32 *) tx->weight, j = 0; j < 8; ipp++, j++) {
         if(*ipp == 0) {
            *ipp = src_ip;
            break;
         }
      }
      if(j >= 8) return VERROR;  /* no space in map to mirror() */
   } else memset(tx->weight, 0, 32);  /* clear address map */

   return VEOK;
}  /* end txmap() */

/* Create a grandchild to send TX's in mirror.dat to ip... */
pid_t mgc(word32 ip)
{
   pid_t pid;
   FILE *fp;
   long offset;
   int lockfd, count;
   TX mtx;
   NODE node;

   /* create grandchild */
   pid = fork();
   if(pid < 0) {
      perr("Cannot fork()");
      return 0;  /* to parent */
   }
   if(pid) return pid;  /* to parent */

   /* in (grand) child */
   pdebug("mgc()...");
   show("mgc()");

   fp = fopen("mirror.dat", "rb");
   if(fp == NULL) {
      perr("Cannot open mirror.dat");
      exit(1);
   }
   offset = 0;

   while(Running) {
      lockfd = lock("mq.lck", 20);
      if(lockfd == -1) {
         perr("Cannot lock mq.lck"); fclose(fp); exit(1);
      }
      if(fseek(fp, offset, SEEK_SET)) {
         unlock(lockfd); fclose(fp); exit(1);
      }
      /* read the TX from mirror.dat */
      count = fread(&mtx, 1, sizeof(TX), fp);
      /* preserve seek pos because other mgc()'s may be running */
      offset = ftell(fp);
      unlock(lockfd);
      if(count != sizeof(TX)) break;
      /* if not in -v modes... */
      if(Port == Dstport) {
         /* Skip this TX if ip address is already in map. */
         if(search32(ip, (word32 *) mtx.weight, 8)) continue;
      }
      if(callserver(&node, ip) != VEOK) break;
      put16(node.tx.len, TRANLEN);
      memcpy(TRANBUFF(&node.tx), TRANBUFF(&mtx), TRANLEN);
      /* copy ip address map to outgoing TX */
      memcpy(node.tx.weight, mtx.weight, 32);
      send_op(&node, OP_TX);
      sock_close(node.sd);
   }  /* end while Running */
   fclose(fp);
   exit(0);
}  /* end mgc() */

/* Send tx to all current or recent peers on iplist.
 * Called from server()       --  becomes child
 */
pid_t mirror1(word32 *iplist, int len)
{
   pid_t pid, peer[RPLISTLEN];
   int j;
   word8 busy;

   /* create child */
   pid = fork();
   if(pid < 0) {
      perr("Cannot fork()");
      return 0;
   }
   if(pid) return pid;  /* to parent */

   /* in child */
   pdebug("mirror()...");
   show("mirror");

   shuffle32(iplist, len);  /* NOTE: can create embedded zeros. */
   /* Create up to len mgc() grandchildren */
   for(j = 0; j < len; j++) {
      if(iplist[j] == 0) { peer[j] = 0; continue; }
      peer[j] = mgc(iplist[j]);  /* grandchild */
   }

   /* while Running, wait for grandchildren to finish. */
   while(Running) {
      busy = 0;
      for(j = 0; j < len; j++) {
         if(peer[j] == 0) continue;
         pid = waitpid(peer[j], NULL, WNOHANG);
         if(pid <= 0) busy = 1; else peer[j] = 0;
      }
      if(!busy) exit(0);
      else millisleep(1);
   }  /* end while Running */
   /* got SIGTERM */
   for(j = 0; j < len; j++) {
      if(peer[j]) {
         kill(peer[j], SIGTERM);     /* Kill grandchild */
         waitpid(peer[j], NULL, 0);  /* and burry her. */
      }
   }
   exit(0);
}  /* end mirror1() */

/* Send tx to either current or recent peers
 * Called from server()       --  becomes child
 */
pid_t mirror(void)
{
   word32 Splist[TPLISTLEN + RPLISTLEN] = { 0 };
   int i, num;

   num = 0;
   for (i = 0; i < TPLISTLEN; i++) {
      if (Tplist[i] == 0) break; /* no more trusted peers */
      Splist[num++] = Tplist[i];
   }
   for (i = 0; i < RPLISTLEN; i++) {
      if (Rplist[i] == 0) break; /* no more recent peers */
      Splist[num++] = Rplist[i];
   }

   return mirror1(Splist, num);
}


/* Called by gettx()  -- in parent
 *
 * Validate a TX, write clean TX to txq1.dat, and raw TX to
 * mirror queue, mq.dat.
 * Locks mq.lck while appending mq.dat.
 */
int process_tx(NODE *np)
{
   TX *tx;
   int evilness;
   int count, lockfd;
   int ecode;
   word8 tx_id[HASHLEN];
   FILE *fp;

   pdebug("process_tx()");
   show("tx");

   tx = &np->tx;

   /* Validate addresses, fee, signature, source balance, and total. */
   evilness = tx_val(tx);
   if(evilness) return evilness;

   /* Compute tx_id[] (hash of tx->src_addr) to append to txq1.dat. */
   sha256(tx->src_addr, TXWOTSLEN, tx_id);

   fp = fopen("txq1.dat", "ab");
   if(!fp) {
      perr("Cannot open txq1.dat");
      return 1;
   }

   /* Now write transaction to txq1.dat followed by tx_id */
   ecode = 0;
   /* 3 addresses (TXWOTSLEN*3) + 3 amounts (8*3) + signature (TXSIGLEN) */
   count = fwrite(TRANBUFF(tx), 1, TRANLEN, fp);
   if(count != TRANLEN) ecode = 1;
   /* then append source tx_id */
   count = fwrite(tx_id, 1, HASHLEN, fp);
   if(count != HASHLEN) ecode = 1;
   pdebug("writing TX to txq1.dat");
   fclose(fp);  /* close txq1.dat */
   if(ecode) {
      perr("bad write on txq1.dat");
      return 1;
   }
   else {
      Txcount++;
      pdebug("incrementing Txcount to %d", Txcount);
   }
   Nrec++;  /* total good TX received */

   /* lock mirror file */
   lockfd = lock("mq.lck", 20);
   if(lockfd == -1) {
      perr("Cannot lock mq.lck");  /* should not happen */
      return 1;
   }
   fp = fopen("mq.dat", "ab");
   if(!fp) {
      perr("Cannot open mq.dat");
      unlock(lockfd);
      return 1;
   }
   ecode = 0;
   /* If empty slot in mirror address map, fill it
    * in and then write tx to mirror queue, mq.dat.
    */
   pdebug("before txmap()");
   if(txmap(tx, np->ip) == VEOK) {
      count = fwrite(tx, 1, sizeof(TX), fp);
      if(count != sizeof(TX)) {
         perr("bad write on mq.dat");
         ecode = 1;
      } else Mqcount++;
   }
   pdebug("after txmap()");
   fclose(fp);      /* close mirror queue, mq.dat */
   unlock(lockfd);  /* unlock mirror queue lock, mq.lck */
   pdebug("done %d", ecode);
   return ecode;
}  /* end process_tx() */

/* end include guard */
#endif
