/**
 * @private
 * @headerfile tx.h <tx.h>
 * @copyright Adequate Systems LLC, 2018-2022. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
 * @todo This unit contains duplicated code from src/bcon.c involving
 * TXPOS and txpos_compare(). This code has not been refactored into
 * a common header file or source file due to future planned deprecation.
*/

/* include guard */
#ifndef MOCHIMO_TX_C
#define MOCHIMO_TX_C


#include "tx.h"

/* internal support */
#include "wots.h"
#include "ledger.h"
#include "global.h"
#include "error.h"

/* external support */
#include <sys/wait.h>
#include <string.h>
#include "sha256.h"
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
   word8 src[ADDR_LEN];
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

   return memcmp(a->src, b->src, sizeof(a->src));
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
 * @private
 * Initialize a Transaction Entry structure per the given header options.
 * @param tx Pointer to TXENTRY structure to initialize
 * @param hdr Pointer to TXHDR structure to analyze, or NULL where the
 * Transaction Header is contained within the TXENTRY buffer
 */
static void tx__init(TXENTRY *tx, const void *opts)
{
   size_t dsaoff, tlroff;

   if (opts == NULL) {
      opts = tx->buffer;
   }

   /* determine validation data offset */
   switch (TXDAT_TYPE(opts)) {
      case TXDAT_MDST:
         /* offset depends on destination count (+1) */
         dsaoff = sizeof(TXHDR) + (sizeof(MDST) * MDST_COUNT(opts));
         break;
   }

   /* determine trailer data offset */
   switch (TXDSA_TYPE(opts)) {
      case TXDSA_WOTS:
         tlroff = dsaoff + sizeof(WOTSVAL);
         break;
   }

   /* initialize structure properties */
   tx->hdr = (TXHDR *) tx->buffer;
   tx->dat = (TXDAT *) (tx->buffer + sizeof(TXHDR));
   tx->dsa = (TXDSA *) (tx->buffer + dsaoff);
   tx->tlr = (TXTLR *) (tx->buffer + tlroff);
   tx->tx_sz = tlroff + sizeof(TXTLR);

   /* set convenient pointers */
   tx->options = tx->hdr->options;
   tx->src_addr = tx->hdr->src_addr;
   tx->chg_addr = tx->hdr->chg_addr;
   tx->send_total = tx->hdr->send_total;
   tx->change_total = tx->hdr->change_total;
   tx->tx_fee = tx->hdr->fee_total;
   tx->tx_btl = tx->hdr->blk_to_live;
   tx->mdst = tx->dat->mdst;
   tx->wots = &(tx->dsa->wots);
   tx->tx_nonce = tx->tlr->nonce;
   tx->tx_id = tx->tlr->id;
}  /* end tx__init() */

struct {
   word8 secret[32];
   word8 public[2][32];
   word8 adrs[2][32];
   word8 tag[2][ADDR_TAG_LEN];
   word8 count[8];
   word8 idx[8];
   int active;
} Txbot = {0};

int tx_bot_is_active(void)
{
   return Txbot.active;
}

static void tx_bot_get_secret(word8 *secret, word8 *idx)
{
   SHA256_CTX ctx;

   /* generate WOTS+ address for index... */
   if (iszero(idx, 8)) {
      /* ... copy origin secret */
      memcpy(secret, Txbot.secret, HASHLEN);
   } else {
      /* ... or generate idx secret */
      sha256_init(&ctx);
      sha256_update(&ctx, idx, 8);
      sha256_update(&ctx, Txbot.secret, HASHLEN);
      sha256_final(&ctx, secret);
   }
}

static void tx_bot_get_addr(word8 *addr, word8 *secret, word8 *idx)
{
   /* local copies */
   word8 wots[WOTS_ADDR_LEN];
   word8 private[32];
   word32 adrs[8];
   int item;

   item = (*idx) % 2;

   if (secret == NULL) {
      tx_bot_get_secret(private, idx);
      secret = private;
   }

   /* update sacrificial adrs and generate public key */
   memcpy(adrs, Txbot.adrs[item], HASHLEN);
   wots_pkgen(wots, secret, Txbot.public[item], adrs);
   /* update full wots address and convert to hash based */
   memcpy(wots + WOTS_PK_LEN, Txbot.public[item], HASHLEN);
   memcpy(wots + WOTS_PK_LEN + HASHLEN, Txbot.adrs[item], HASHLEN);
   addr_from_wots(wots, addr);
}

int tx_bot_activate(const char *filename)
{
   struct /* LEGACY_WOTS_EXPORT */ {
      word8 addr[WOTS_ADDR_LEN];
      word8 balance[8];
      word8 secret[32];
   } wots;
   word8 addr[ADDR_LEN];

   if (!fexists("txbot.dat")) {
      if (filename == NULL) return VERROR;
      pdebug("Converting LEGACY address for Txbot...");
      /* obtain Txbot secret and pub_seed/adrs values */
      if (read_data(&wots, sizeof(wots), (char *) filename) != sizeof(wots)) return VERROR;
      if (WOTS_TAG_PTR(wots.addr)[0] != 0x42) {
         perr("Legacy Tagged WOTS+ address not supported");
         return VERROR;
      }
      /* move data to Txbot global */
      memcpy(Txbot.secret, wots.secret, 32);
      memcpy(Txbot.public[0], wots.addr + WOTS_PK_LEN, 32);
      memcpy(Txbot.adrs[0], wots.addr + WOTS_PK_LEN + 32, 32);
      /* generate appropriate tag (item 0) */
      tx_bot_get_addr(addr, NULL, ZERO64);
      memcpy(Txbot.tag[0], ADDR_TAG_PTR(addr), ADDR_TAG_LEN);
      /* expand secondary public and adrs items */
      sha256(Txbot.public[0], 32, Txbot.public[1]);
      sha256(Txbot.adrs[0], 32, Txbot.adrs[1]);
      /* ... force WOTS+ default */
      put32(Txbot.adrs[1] + 20, 0x42);
      put32(Txbot.adrs[1] + 24, 0x0e);
      put32(Txbot.adrs[1] + 28, 0x01);
      /* generate appropriate tag (item 1) */
      tx_bot_get_addr(addr, NULL, ONE64);
      memcpy(Txbot.tag[1], ADDR_TAG_PTR(addr), ADDR_TAG_LEN);
      /* write Txbot file back to disk */
      pdebug("Writing Txbot...");
      if (write_data(&Txbot, sizeof(Txbot), "txbot.dat") != sizeof(Txbot)) return VERROR;
   } else {
      pdebug("Reading existing Txbot...");
      if (read_data(&Txbot, sizeof(Txbot), "txbot.dat") != sizeof(Txbot)) {
         return VEOK;
      }
   }

   /* set Txbot option and continue */
   Txbot.active = 1;

   return VEOK;
}

int tx_bot_process(void)
{
   LENTRY le;
   NODE node;
   TXENTRY tx;
   word8 src_addr[ADDR_LEN];
   word8 chg_addr[ADDR_LEN];
   word8 hash[HASHLEN];
   word8 secret[HASHLEN];
   word32 adrs[8];

   int item;

   /* zero transaction entry */
   memset(&tx, 0, sizeof(TXENTRY));
   item = (*Txbot.idx) % 2;

   /* store source item, and secret for signature */
   tx_bot_get_secret(secret, Txbot.idx);
   /* generate src_addr */
   tx_bot_get_addr(src_addr, secret, Txbot.idx);
   memcpy(ADDR_TAG_PTR(src_addr), Txbot.tag[item], ADDR_TAG_LEN);
   /* check for available funds */
   for (;;) {
      /* ensure available funds in found addresses */
      if (le_find(src_addr, &le, ADDR_LEN)) {
         if (sub64(le.balance, MFEE64, le.balance) == 0) {
            break;
         }
      }
      pdebug("No Txbot funds @ idx#%" P32u "...", get32(Txbot.idx));
      /* decrement index for next process iteration */
      if (sub64(Txbot.idx, ONE64, Txbot.idx)) {
         perr("No available balance for Txbot");
         put64(Txbot.idx, Txbot.count);
         goto DONE;
      }
      return VERROR;
   }

   /* Txbot funds found */
   pdebug("Txbot funds found in %s...", hash2hex32(le.addr, NULL));


   /* generate chg_addr (+2) -- force tag */
   add64(Txbot.idx, CL64_32(2), Txbot.idx);
   tx_bot_get_addr(chg_addr, NULL, Txbot.idx);
   memcpy(ADDR_TAG_PTR(chg_addr), Txbot.tag[item], ADDR_TAG_LEN);
   /* update count if necessary */
   if (cmp64(Txbot.count, Txbot.idx) < 0) {
      put64(Txbot.count, Txbot.idx);
   }

   /* store current Txbot state */
   if (write_data(&Txbot, sizeof(Txbot), "txbot.dat") != sizeof(Txbot)) {
      /* DO NOT PROCEED WITH TRANSACTION */
      perr("write_data() FAILURE");
      goto DONE;
   }

   /* set options and intialize transaction */
   TXDSA_TYPE(tx.buffer) = TXDSA_WOTS;
   TXDAT_TYPE(tx.buffer) = TXDAT_MDST;
   tx__init(&tx, NULL);
   /* build transaction -- le.amount already reduced by fee */
   memcpy(tx.hdr->src_addr, src_addr, ADDR_LEN);
   memcpy(tx.mdst[0].tag, Txbot.tag[item ^ 1], ADDR_TAG_LEN);
   put64(tx.mdst[0].amount, le.balance);
   memcpy(tx.hdr->chg_addr, chg_addr, ADDR_LEN);
   put64(tx.hdr->send_total, le.balance);
   put64(tx.tx_fee, MFEE64);
   /* generate WOTS+ signature */
   tx_hash(&tx, 0, hash);
   memcpy(adrs, Txbot.adrs[item], 32);
   wots_sign(tx.wots->signature, hash, secret, Txbot.public[item], adrs);
   memcpy(tx.wots->pub_seed, Txbot.public[item], 32);
   memcpy(tx.wots->adrs, Txbot.adrs[item], 32);
   /* ... force WOTS+ default */
   put32(tx.wots->adrs + 20, 0x42);
   put32(tx.wots->adrs + 24, 0x0e);
   put32(tx.wots->adrs + 28, 0x01);

   /* place transaction in empty NODE and process */
   memset(&node, 0, sizeof(NODE));
   memcpy(node.tx.buffer, &tx, TXLEN_MIN);
   put16(node.tx.len, TXLEN_MIN);
   if (process_tx(&node) != VEOK) {
      perrno("Txbot transaction was not processed...");
   }

DONE:
   return VEOK;
}

/**
 * Read a single Transaction Entry in to the provided container, @a txe,
 * from the given input @a stream. The file position indicator is
 * advanced by the size of the Transaction Entry (size varies).
 * @param txe Pointer to Transaction Entry container
 * @param stream The stream to read from
 * @return (int) value representing the read result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 */
int tx_fread(TXENTRY *tx, FILE *stream)
{
   size_t len;

   /* prepare transaction container */
   memset(tx, 0, sizeof(TXENTRY));

   /* read transaction header into buffer */
   if (fread(tx->buffer, sizeof(TXHDR), 1, stream) != 1) {
      return VERROR;
   }

   /* initialize transaction entry already containing header */
   tx__init(tx, NULL);

   /* read remaining transaction parts */
   len = tx->tx_sz - sizeof(TXHDR);
   if (fread(tx->buffer + sizeof(TXHDR), len, 1, stream) != 1) {
      return VERROR;
   }

   return VEOK;
}  /* end tx_fread() */

/**
 * Write a single Transaction Entry in to the provided container, @a txe,
 * to the given output @a stream. The file position indicator is
 * advanced by the size of the Transaction Entry (size varies).
 * @param txe Pointer to Transaction Entry container
 * @param stream The stream to write to
 * @return (int) value representing the write result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 */
int tx_fwrite(const TXENTRY *tx, FILE *stream)
{
   /* write buffer to output stream */
   if (fwrite(tx->buffer, tx->tx_sz, 1, stream) != 1) {
      return VERROR;
   }

   return VEOK;
}  /* end tx_fwrite() */

/**
 * Hash a Transaction Entry, @a txe.
 * @param txe Pointer to Transaction Entry data
 * @param full Set non-zero for a "full" Transaction ID hash or
 * set zero for a Transaction Signature Message hash.
 * @param out Pointer to place finalized hash
 */
void tx_hash(const TXENTRY *tx, int full, void *out)
{
   if (full) {
      /* full transaction ID hash */
      sha256(tx->buffer, tx->tx_sz - sizeof(TXTLR), out);
      return;
   }

   /* transaction signature message hash (header + data) */
   sha256(tx->buffer, (size_t) tx->dsa - (size_t) tx->hdr, out);
}  /* end tx_hash() */

/**
 * Read transaction data from a buffer into a Transaction Entry structure.
 * @param tx Pointer to TXENTRY structure to populate
 * @param buf Pointer to buffer containing transaction data
 * @param bufsz Size of buffer containing transaction data
 * @return (int) value representing the result of the operation
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
*/
int tx_read(TXENTRY *tx, const void *buf, size_t bufsz)
{
   /* check bufsz contains header data */
   if (bufsz < sizeof(TXHDR)) {
      set_errno(EMCM_TXINVAL);
      return VERROR;
   }

   /* prepare transaction container */
   memset(tx, 0, sizeof(TXENTRY));

   /* compute offsets */
   tx__init(tx, buf);

   /* check transaction size -- valid with or without trailer data */
   if (bufsz > tx->tx_sz || bufsz < (tx->tx_sz - sizeof(TXTLR))) {
      set_errno(EMCM_TXINVAL);
      return VERROR;
   }

   /* prepare transaction buffer */
   memcpy(tx->buffer, buf, bufsz);

   return VEOK;
}  /* end tx_read() */

/**
 * @private
 * Validate a Multi-Destination Reference field.
 * @param ref Pointer to start of 16 byte reference buffer
 * @return VEOK on success, or VERROR or error; check errno for details
 */
static int mdst_val__reference(const char *reference)
{
   int j;

   /* define states for stages of validation */
   enum { START, DIGIT_DASH, DIGIT, UPPER_DASH, UPPER, ZERO } state;

   /* Validation format rules (from types.h):
    * - CONTAINS only uppercase [A-Z], digit [0-9], dash [-], null [\0]
    * - SHALL have remaining bytes zeroed after first null termination
    *   - (e.g. VALID   `(char[]) { 'A','-','1','\0','\0','\0', ... }`)
    *   - (e.g. INVALID `(char[]) { 'A','-','1','\0','B','\0', ... }`)
    * - MAY have multiple uppercase OR digits (NOT both) grouped together
    * - SHALL only contain a dash to separate groups of uppercase or digit
    * - SHALL NOT contain consecutive groups of the same group type
    *   - (e.g. VALID   "AB-00-EF", "123-CDE-789", "ABC", "123")
    *   - (e.g. INVALID "AB-CD-EF", "123-456-789", "ABC-", "-123")
    */

   /* validate reference format */
   for (state = START, j = 0; j < ADDR_REF_LEN; j++) {
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
            if (reference[j] == '-') { state = DIGIT_DASH; continue; }
            if (reference[j] == '\0') { state = ZERO; continue; }
            break;  /* switch() */
         case UPPER:  /* allow either uppercase, dash, or ZERO */
            if (isupper(reference[j])) continue;  /* for() */
            if (reference[j] == '-') { state = UPPER_DASH; continue; }
            if (reference[j] == '\0') { state = ZERO; continue; }
            break;  /* switch() */
         case ZERO:  /* allow only ZERO (end of reference) */
            if (reference[j] == '\0') continue;  /* for() */
      }  /* end switch(state) */

      /* no valid character for current state */
      return VERROR;
   }  /* end for() */

   /* state machine must end with ZERO, DIGIT or UPPER state */
   if (state == ZERO) return VEOK;
   if (state == DIGIT) return VEOK;
   if (state == UPPER) return VEOK;
   /* ... else reference is invalid */

   return VERROR;
}  /* end mdst_val__reference() */

/**
 * @private
 * Validate a Multi-Destination Transaction (incl. reference field).
 * @param txe Pointer to Transaction Entry to validate
 * @return (int) value representing the validation result
 * @retval VEBAD2 on bad signature; check errno for details
 * @retval VEBAD on bad transaction; check errno for details
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 */
static int mdst_val(const TXENTRY *txe)
{
   MDST *mdst = txe->mdst;
   word8 total[8] = {0};
   word8 mfees[8] = {0};
   int count;
   int j;

   /* obtain multi-destination count */
   count = MDST_COUNT(txe->hdr);

   /* Tally each dst[] amount and mfees... */
   for (j = 0; j < count; j++) {
      if (j > 0) {
         /* check sort -- allow duplicates */
         if (tag_compare(mdst[j].tag, mdst[j - 1].tag) < 0 \
            || memcmp(mdst[j].ref, mdst[j - 1].ref, ADDR_REF_LEN) < 0) {
            set_errno(EMCM_TXMDSTSORT);
            return VEBAD;
         }
      }
      /* no zero amounts */
      if (iszero(mdst[j].amount, 8)) {
         set_errno(EMCM_XTXDSTAMOUNT);
         return VEBAD;
      }
      /* no dst to src */
      if (tag_equal(mdst[j].tag, ADDR_TAG_PTR(txe->src_addr))) {
         set_errno(EMCM_XTXTAGMATCH);
         return VEBAD;
      }
      /* tally amounts and fees -- no overflow */
      if (add64(total, mdst[j].amount, total)) {
         set_errno(EMCM_MATH64_OVERFLOW);
         return VEBAD;
      }
      if (add64(mfees, Myfee, mfees)) {
         set_errno(EMCM_MFEES_OVERFLOW);
         return VEBAD;
      }
      /* dst reference must be valid */
      if (mdst_val__reference(mdst[j].ref) != VEOK) {
         set_errno(EMCM_XTXREF);
         return VEBAD;
      }
   }  /* end for() */
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
}  /* end mdst_val() */

/**
 * @private
 * Validate WOTS+ validation data from a transaction.
 * @param tx Pointer to Transaction Entry to validate
 * @param message Pointer to message hash
 * @return (int) value representing validation result
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 */
static int tx_val__wots(const TXENTRY *tx)
{
   const word32 adrs_[] = { 0x42, 0x0e, 0x01 };

   word8 pk[WOTS_PK_LEN];
   word8 message[HASHLEN];
   word8 addrhash[ADDR_HASH_LEN];
   word32 adrs[8];
   word8 *src_addr;
   WOTSVAL *wots;

   /* dereference relevant pointers */
   src_addr = tx->hdr->src_addr;
   wots = tx->wots;

   /* generate transaction signature message */
   tx_hash(tx, 0, message);

   /* recreate WOTS+ public key from signature */
   memcpy(adrs, wots->adrs, 32);
   wots_pk_from_sig(pk, wots->signature, message, wots->pub_seed, adrs);
   /* ... always modifies adrs[], resulting in { ..., 0x42, 0x0e, 0x01 }.
    * Somewhat unintentionally, a check was included on this result that
    * would have normally been ignored, discovering an issue with WOTS+
    * optimisations/improvements causing validation failures elsewhere.
    * The following checks will remain to prevent future regressions,
    * until the appropriate unit test is updated to cover this case.
    */
   if (memcmp(wots->adrs, adrs, 32) != 0 || /* ... BAD WOTS+ function */
         memcmp(wots->adrs + 20, adrs_, 12) != 0 /* ... BAD TX data */) {
      set_errno(EMCM_TXADRS);
      return VERROR;
   }
   /* validate hashed address against source */
   addr_hash_generate(pk, WOTS_PK_LEN, addrhash);
   if (memcmp(ADDR_HASH_PTR(src_addr), addrhash, ADDR_HASH_LEN) != 0) {
      set_errno(EMCM_TXWOTS);
      return VERROR;
   }

   return VEOK;
}  /* end tx_val__wots() */

/**
 * Validate transaction data, as if received directly from a wallet.
 * DOES NOT validate nonce or id. Requires an open ledger.
 * @param txe Pointer to Transaction Entry to validate
 * @param bnum Pointer to block number to validate against
 * @return (int) value representing validation result
 * @retval VEBAD2 on invalid signature; check errno for details
 * @retval VEBAD on bad transaction data; check errno for details
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 */
int tx_val(const TXENTRY *txe, const void *bnum, const void *mfee)
{
   LENTRY le;
   word8 total[8];
   word8 *src_addr;
   word8 *chg_addr;
   word8 *send_total;
   word8 *change_total;
   int overflow;

   /* derefence header pointers */
   src_addr = txe->hdr->src_addr;
   chg_addr = txe->hdr->chg_addr;
   send_total = txe->hdr->send_total;
   change_total = txe->hdr->change_total;

   /* only non-zero block-to-live values are checked */
   if (!iszero(txe->tx_btl, 8)) {
      /* prepare block-to-live upper bound */
      if (add64(bnum, CL64_32(0x100), total)) {
         set_errno(EMCM_MATH64_OVERFLOW);
         return VERROR;
      }
      /* check block number against block-to-live range */
      if (cmp64(txe->tx_btl, bnum) < 0 || cmp64(txe->tx_btl, total) > 0) {
         set_errno(EMCM_TXBTL);
         return VEBAD;
      }
   }

   /* validate src != chg, but associated TAGs MUST MATCH */
   if (addr_hash_equal(src_addr, chg_addr)) {
      set_errno(EMCM_TXCHG);
      return VEBAD;
   }
   if (!addr_tag_equal(src_addr, chg_addr)) {
      set_errno(EMCM_XTXTAGMISMATCH);
      return VEBAD;
   }

   /* validate transaction fee -- implicit MFEE check (mfee >= MFEE) */
   if (cmp64(txe->tx_fee, mfee) < 0) {
      set_errno(EMCM_TXFEE);
      return VEBAD;
   }

   /* check transaction type... */
   switch (TXDAT_TYPE(txe->hdr)) {
      case TXDAT_MDST:
         /* ... validate destination array */
         if (mdst_val(txe) != VEOK) return VEBAD;
         break;
      default:
         set_errno(EMCM_XTXUNDEF);
         return VEBAD;
   }  /* end switch(TYPE) */

   /* determine appropriate DSA type */
   switch (TXDSA_TYPE(txe->hdr)) {
      case TXDSA_WOTS:
         /* ... validate WOTS+ transaction data */
         if (tx_val__wots(txe) != VEOK) return VEBAD2;
         break;
      default:
         set_errno(EMCM_TXDSA);
         return VEBAD2;
   }  /* end switch(DSA) */

   /* look up source address in ledger */
   if (!le_find(src_addr, &le, ADDR_LEN)) {
      set_errno(EMCM_TXSRCLE);
      return VERROR;
   }
   /* check total amounts match balance */
   memset(total, 0, sizeof(total));
   /* use add64() to check for overflow */
   overflow =  add64(send_total, change_total, total);
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

   /* transaction is valid */
   return VEOK;
}  /* end tx_val() */

/**
 * Validate transaction entry, as stored on chain. Requires an open ledger.
 * @param txe Pointer to transaction entry to validate
 * @param bnum Pointer to block number to validate against
 * @return (int) value representing validation result
 * @retval VEBAD2 on invalid signature; check errno for details
 * @retval VEBAD on bad transaction data; check errno for details
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 */
int txe_val(const TXENTRY *txe, const void *bnum, const void *mfee)
{
   word8 hash[HASHLEN];

   /* check block number matches nonce */
   if (cmp64(txe->tx_nonce, bnum) != 0) {
      set_errno(EMCM_TXNONCE);
      return VEBAD2;
   }

   /* check transaction ID hash is correct */
   tx_hash(txe, 1, hash);
   if (memcmp(txe->tx_id, hash, HASHLEN) != 0) {
      set_errno(EMCM_TXID);
      return VEBAD2;
   }

   /* return result of transaction data validation */
   return tx_val(txe, bnum, mfee);
}  /* end txe_val() */

/**
 * Search txq1.dat and txclean.dat for conflicts with the src_addr
 * of queued transactions.
 * @param src_addr Pointer to source address
 * @return (int) value representing the result
 * @retval VERROR on conflict; check errno for details
 * @retval VEOK on success
 */
int txcheck(const word8 *src_addr)
{
   FILE *fp;
   TXENTRY txe;

   /* read transaction in txq1 checkiung for conflicts */
   fp = fopen("txq1.dat", "rb");
   if (fp != NULL) {
      while (tx_fread(&txe, fp) == VEOK) {
         if (addr_tag_equal(txe.src_addr, src_addr)) {
            /* source address (tag) conflict */
            set_errno(EMCM_TXSRCDUP);
            goto FAIL;
         }
      }
      /* error check file and close*/
      if (ferror(fp)) goto FAIL;
      fclose(fp);  /* EOF */
   }  /* end if fp */

   /* duplicate search routine (as above) for txclean */
   fp = fopen("txclean.dat", "rb");
   if (fp != NULL) {
      while (tx_fread(&txe, fp) == VEOK) {
         if (addr_tag_equal(txe.src_addr, src_addr)) {
            /* source address (tag) conflict */
            set_errno(EMCM_TXSRCDUP);
            goto FAIL;
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
   TXENTRY txe, txc;       /* block entry and txclean transactions */
   FILE *fp, *bfp, *tfp;   /* input, blockchain and temporary files */
   void *ptr;              /* realloc pointer */
   TXPOS *tx;              /* malloc'd transaction positions */
   size_t count, actual;   /* malloc'd and actual tx element counts */
   size_t j, nout;
   fpos_t pos;             /* file position offset indicator */
   long long offset;       /* file position offset value */
   word32 hdrlen;          /* for block header length */
   int cond;

   /* ensure ledger is open (required) */
   if (le_open("ledger.dat") != VEOK) {
      perr("failed to le_open(ledger.dat)");
      return VERROR;
   }

   /* error handling init */
   fp = bfp = tfp = NULL;
   tx = NULL;

   /* GENERATE SORTED (ASCENDING) TXID REFERENCES FOR COMPARE */

   /* open provided txclean file */
   fp = fopen(txfname, "rb");
   if (fp == NULL) return VERROR;

   /* obtain EOF offset */
   if (fseek64(fp, 0LL, SEEK_END) != 0) goto ERROR_CLEANUP;
   offset = ftell64(fp);
   if (offset == (-1)) goto ERROR_CLEANUP;
   if (offset == 0LL) {
      /* nothing to clean */
      fclose(fp);
      return VEOK;
   }
   /* check offset against minimum txentry size on disk */
   if ((size_t) offset < TXLEN_DSK_MIN) {
      /* file contains unknown data */
      set_errno(EMCM_FILEDATA);
      goto ERROR_CLEANUP;
   }

   /* reset file position indicator */
   if (fseek64(fp, 0LL, SEEK_SET) != 0) goto ERROR_CLEANUP;
   /* loop to check allocated space is sufficient (+32 TXs/loop) */
   for (actual = 0, cond = 1, count = 32; cond; count += 32) {
      /* (re)allocate memory space for 32 TXs at a time */
      ptr = realloc(tx, count * sizeof(TXPOS));
      if (ptr == NULL) goto ERROR_CLEANUP;
      tx = ptr;
      /* loop to store source and associated fpos_t value in array */
      while (actual < count) {
         /* store position for later use (if tx is read) */
         if (fgetpos(fp, &pos) != 0) goto ERROR_CLEANUP;
         if (tx_fread(&txc, fp) != VEOK) {
            if (ferror(fp)) goto ERROR_CLEANUP;
            /* set EOF condition */
            cond = 0;
            break;
         }
         /* set source reference data */
         memcpy(&(tx[actual].src), txc.src_addr, ADDR_LEN);
         tx[actual++].pos = pos;
      }  /* end while() */
   }  /* end for() */
   /* check for leftover data */
   if (ftell64(fp) < offset) {
      set_errno(EMCM_FILEDATA);
      goto ERROR_CLEANUP;
   }
   /* sort the txid reference array */
   qsort(tx, actual, sizeof(TXPOS), txpos_compare);

   /* PREPARE BLOCKCHAIN FILE FOR TRANSACTION COMPARISON (IF PROVIDED) */

   /* only if blockchain file is provided */
   if (bcfname != NULL) {
      /* open validated block file */
      bfp = fopen(bcfname, "rb");
      if (bfp == NULL) goto ERROR_CLEANUP;
      /* read and check fixed length header */
      if (fread(&hdrlen, 4, 1, bfp) != 1) {
         if (!ferror(fp)) set_errno(EMCM_EOF);
         goto ERROR_CLEANUP;
      }
      /* allow transaction loop for all non-neogenesis type blocks */
      if (hdrlen != sizeof(BHEADER) && hdrlen != 4) {
         set_errno(EMCM_HDRLEN);
         goto ERROR_CLEANUP;
      }
      /* seek to start of Merkle Array */
      if (fseek(bfp, (long) hdrlen, SEEK_SET) != 0) {
         goto ERROR_CLEANUP;
      }
   }

   /* FILTER OLD QUEUE BASED ON BLOCK AND LEDGER */

   /* generate temporary filename */
   tfp = tmpfile();
   if (tfp == NULL) goto ERROR_CLEANUP;

   /* Remove TX's from clean TX queue that are in the new block.
    * Remove remaining transactions that no longer validate.
    * Merkel Array in new block is sorted on src_addr (checked in bval).
    * Clean queue, txclean.dat, is sorted by reference array above.
    */
   for (cond = j = nout = 0; j < actual; j++) {
      if (bfp != NULL) {
         do {
            /* if src from block compares AFTER reference, hold... */
            if (cond <= 0) {
               /* read next transaction from block */
               if (tx_fread(&txe, bfp) != VEOK) {
                  if (ferror(bfp)) goto ERROR_CLEANUP;
                  /* EOF -- break inner loop */
                  fclose(bfp);
                  bfp = NULL;
                  break;
               }
            }
            /* compare block transaction source with reference source */
            cond = memcmp(txe.src_addr, tx[j].src, HASHLEN);
            /* if src from block compares BEFORE reference, redo... */
         } while (cond < 0);
         /* if src from block compares EQUAL TO reference, skip... */
         if (bfp != NULL && cond == 0) continue;
      }
      /* .. else; read reference transaction from previously set fpos */
      if (fsetpos(fp, &(tx[j].pos)) != 0) goto ERROR_CLEANUP;
      if (tx_fread(&txc, fp) != VEOK) goto ERROR_CLEANUP;
      /* update nonce for validation check */
      add64(Cblocknum, ONE64, txc.tx_nonce);
      /* if (re)validation fails, skip... */
      /** @todo: replace tx_val with less wasteful tx_reval process */
      if (tx_val(&txc, txc.tx_nonce, Myfee) != VEOK) continue;
      /* write clean (valid) transaction to output */
      if (tx_fwrite(&txc, tfp) != VEOK) goto ERROR_CLEANUP;
      nout++;
   }  /* end for() */

   /* cleanup */
   if (bfp) fclose(bfp);
   fclose(fp);
   free(tx);

   /* out with the old, in with the new */
   remove(txfname);
   if (nout > 0) {
      if (fsave(tfp, (char *) txfname) != 0) {
         fclose(tfp);
         return VERROR;
      }
   }

   /* success */
   fclose(tfp);
   return VEOK;

   /* cleanup / error handling */
ERROR_CLEANUP:
   if (tfp) fclose(tfp);
   if (bfp) fclose(bfp);
   if (fp) fclose(fp);
   if (tx) free(tx);

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
      memcpy(node.tx.buffer, mtx.buffer, get16(mtx.len));
      /* copy buffer length and ip address map to outgoing TX */
      memcpy(node.tx.weight, mtx.weight, 32);
      put16(node.tx.len, get16(mtx.len));
      send_op(&node, OP_TX);
      sock_close(node.sd);
   }  /* end while Running */
   fclose(fp);
   exit(0);
}  /* end mgc() */

/* Send tx to all current or recent peers on iplist.
 * Called from server()       --  becomes child
 */
pid_t mirror(void)
{
   pid_t pid, peer[RPLISTLEN];
   int j, len;
   word8 busy;

   /* create child */
   pid = fork();
   if (pid < 0) return 0;
   if(pid) return pid;  /* to parent */

   /* in child */
   pdebug("mirror()...");
   show("mirror");

   /* Create up to len mgc() grandchildren */
   memset(peer, 0, sizeof(peer));
   for (j = len = 0; j < RPLISTLEN; j++) {
      if (Rplist[j] == 0) continue;
      peer[len++] = mgc(Rplist[j]);  /* grandchild */
   }
   pdebug("prepared %d mgc()...", len);

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
   pdebug("... unfinished mgc()");

   /* got SIGTERM */
   for(j = 0; j < len; j++) {
      if(peer[j]) {
         pdebug("killing mgc(%ld)...", (long) peer[j]);
         kill(peer[j], SIGTERM);     /* Kill grandchild */
         waitpid(peer[j], NULL, 0);  /* and burry her. */
         pdebug("... killed mgc(%ld)", (long) peer[j]);
      }
   }
   exit(0);
}  /* end mirror() */

/**
 * Process a transaction received into a NODE structure's TX buffer.
 * Validate a TX, write clean TX to txq1.dat, and raw TX to
 * mirror queue, mq.dat.
 * Locks mq.lck while appending mq.dat.
 * @param np Pointer to NODE containing transaction to process
 * @return (int) value representing the result
 * @retval VEBAD2 on invalid signature; check errno for details
 * @retval VEBAD on bad data; check errno for details
 * @retval VERROR on error; check errno for details
 * @retval VEOK on success
 */
int process_tx(NODE *np)
{
   TXENTRY txe;
   FILE *fp;
   TX *tx;
   int evilness;
   int count, lockfd;
   int ecode;

   show("tx");

   tx = &np->tx;

   /* read transaction entry from buffer */
   ecode = tx_read(&txe, tx->buffer, get16(tx->len));
   if (ecode != VEOK) return VEBAD;

   /* (quick) check for duplicate transactions */
   ecode = txcheck(txe.src_addr);
   if (ecode != VEOK) {
      Ndups++;
      return ecode;
   }

   /* Validate addresses, fee, signature, source balance, and total. */
   evilness = tx_val(&txe, Cblocknum, Myfee);
   if(evilness) return evilness;

   fp = fopen("txq1.dat", "ab");
   if (fp == NULL) return VERROR;

   /* write transaction (incl. nonce and id) to txq1.dat */
   ecode = tx_fwrite(&txe, fp);
   fclose(fp);  /* close txq1.dat */
   if (ecode != VEOK) return VERROR;

   Txcount++;
   Nrec++;  /* total good TX received */

   /* lock mirror file */
   lockfd = lock("mq.lck", 20);
   if (lockfd == -1) return VERROR;
   fp = fopen("mq.dat", "ab");
   if (fp == NULL) {
      unlock(lockfd);
      return VERROR;
   }
   ecode = VEOK;
   /* If empty slot in mirror address map, fill it
    * in and then write tx to mirror queue, mq.dat.
    */
   if(txmap(tx, np->ip) == VEOK) {
      count = fwrite(tx, 1, sizeof(TX), fp);
      if(count != sizeof(TX)) {
         ecode = VERROR;
      } else Mqcount++;
   }
   fclose(fp);      /* close mirror queue, mq.dat */
   unlock(lockfd);  /* unlock mirror queue lock, mq.lck */
   return ecode;
}  /* end process_tx() */

/* end include guard */
#endif
