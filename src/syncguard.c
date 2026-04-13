/**
 * @file syncguard.c
 * @brief Sync subsystem hardening implementation.
 * @copyright Adequate Systems LLC, 2026. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
 */

/* include guard */
#ifndef MOCHIMO_SYNCGUARD_C
#define MOCHIMO_SYNCGUARD_C

#include "syncguard.h"

/* internal support */
#include "error.h"
#include "tfile.h"   /* for NTFTX */

/* external support */
#include "extmath.h"  /* cmp64 */
#include "extlib.h"   /* get32 */
#include "sha256.h"
#include "extio.h"    /* plog/perr */

#include <stdio.h>
#include <string.h>

/* ---- session state -------------------------------------------------- */

typedef struct {
   word8  weight[32];
   word8  hash[HASHLEN];
   int    used;
} sg_bad_chain_t;

typedef struct {
   word8  hash[HASHLEN];
   int    used;
} sg_bad_tfile_t;

typedef struct {
   word32   ip;
   word32   count;
   BTRAILER proof[NTFTX];
   int      used;
} sg_proof_t;

static sg_bad_chain_t  BadChains[SYNCGUARD_BAD_CHAIN_MAX];
static sg_bad_tfile_t  BadTfiles[SYNCGUARD_BAD_TFILE_MAX];
static sg_proof_t      Proofs[SYNCGUARD_PROOF_CACHE_MAX];

/* ---- lifecycle ------------------------------------------------------ */

void sg_session_reset(void)
{
   memset(BadChains, 0, sizeof(BadChains));
   memset(BadTfiles, 0, sizeof(BadTfiles));
   memset(Proofs, 0, sizeof(Proofs));
}

/* ---- bad-chain exclusion list -------------------------------------- */

int sg_bad_chain_check(const word8 weight[32], const word8 hash[HASHLEN])
{
   int i;
   for (i = 0; i < SYNCGUARD_BAD_CHAIN_MAX; i++) {
      if (!BadChains[i].used) continue;
      if (memcmp(BadChains[i].weight, weight, 32) == 0 &&
          memcmp(BadChains[i].hash, hash, HASHLEN) == 0) {
         return 1;  /* match: excluded */
      }
   }
   return 0;
}

void sg_bad_chain_add(const word8 weight[32], const word8 hash[HASHLEN])
{
   int i;
   /* skip if already present */
   if (sg_bad_chain_check(weight, hash)) return;
   /* find first free slot */
   for (i = 0; i < SYNCGUARD_BAD_CHAIN_MAX; i++) {
      if (!BadChains[i].used) {
         memcpy(BadChains[i].weight, weight, 32);
         memcpy(BadChains[i].hash, hash, HASHLEN);
         BadChains[i].used = 1;
         return;
      }
   }
   /* table full: overwrite slot 0 (oldest-style replacement) */
   memcpy(BadChains[0].weight, weight, 32);
   memcpy(BadChains[0].hash, hash, HASHLEN);
   BadChains[0].used = 1;
}

/* ---- bad-tfile cache ------------------------------------------------ */

int sg_bad_tfile_check(const word8 hash[HASHLEN])
{
   int i;
   for (i = 0; i < SYNCGUARD_BAD_TFILE_MAX; i++) {
      if (!BadTfiles[i].used) continue;
      if (memcmp(BadTfiles[i].hash, hash, HASHLEN) == 0) return 1;
   }
   return 0;
}

void sg_bad_tfile_add(const word8 hash[HASHLEN])
{
   int i;
   if (sg_bad_tfile_check(hash)) return;
   for (i = 0; i < SYNCGUARD_BAD_TFILE_MAX; i++) {
      if (!BadTfiles[i].used) {
         memcpy(BadTfiles[i].hash, hash, HASHLEN);
         BadTfiles[i].used = 1;
         return;
      }
   }
   /* replace slot 0 if full */
   memcpy(BadTfiles[0].hash, hash, HASHLEN);
   BadTfiles[0].used = 1;
}

int sg_hash_file(const char *fname, word8 out[HASHLEN])
{
   FILE *fp;
   word8 buf[4096];
   size_t n;
   SHA256_CTX ctx;

   fp = fopen(fname, "rb");
   if (fp == NULL) return VERROR;
   sha256_init(&ctx);
   while ((n = fread(buf, 1, sizeof(buf), fp)) > 0) {
      sha256_update(&ctx, buf, n);
   }
   if (ferror(fp)) { fclose(fp); return VERROR; }
   fclose(fp);
   sha256_final(&ctx, out);
   return VEOK;
}

/* ---- per-peer proof cache ------------------------------------------- */

static sg_proof_t *sg_proof_find(word32 ip)
{
   int i;
   for (i = 0; i < SYNCGUARD_PROOF_CACHE_MAX; i++) {
      if (Proofs[i].used && Proofs[i].ip == ip) return &Proofs[i];
   }
   return NULL;
}

static sg_proof_t *sg_proof_slot(word32 ip)
{
   int i;
   sg_proof_t *existing = sg_proof_find(ip);
   if (existing) return existing;
   for (i = 0; i < SYNCGUARD_PROOF_CACHE_MAX; i++) {
      if (!Proofs[i].used) return &Proofs[i];
   }
   /* table full: evict slot 0 */
   return &Proofs[0];
}

void sg_proof_store(word32 ip, const BTRAILER *proof, word32 count)
{
   sg_proof_t *slot;
   if (proof == NULL || count == 0 || count > NTFTX) return;
   slot = sg_proof_slot(ip);
   slot->ip = ip;
   slot->count = count;
   memcpy(slot->proof, proof, count * sizeof(BTRAILER));
   slot->used = 1;
}

int sg_proof_get(word32 ip, BTRAILER *out, word32 count)
{
   sg_proof_t *p = sg_proof_find(ip);
   if (p == NULL) return VERROR;
   if (count > p->count) count = p->count;
   memcpy(out, p->proof, count * sizeof(BTRAILER));
   return VEOK;
}

void sg_proof_clear_all(void)
{
   memset(Proofs, 0, sizeof(Proofs));
}

/* Compare the cached proof segment for ip against the corresponding
 * range of trailers in the tfile at fname. The proof's first trailer
 * should appear at offset proof[0].bnum * sizeof(BTRAILER) in the
 * tfile (trailers are stored in bnum order starting from the genesis
 * block). The tfile may have advanced past the proof's tip if the
 * peer mined or received new blocks between scan_quorum() and
 * resync(); we only care that the proof's historical trailers match
 * the corresponding positions in the tfile.
 *
 * Returns VEOK on byte-exact match, VERROR on any mismatch, I/O
 * failure, or if the tfile does not contain the proof's range. */
int sg_proof_match_tfile(word32 ip, const char *tfname)
{
   sg_proof_t *p;
   FILE *fp;
   long long len, off;
   BTRAILER bt;
   word32 i;
   word64 first_bnum;

   p = sg_proof_find(ip);
   if (p == NULL) {
      pdebug("sg_proof_match_tfile: no cached proof for peer");
      return VERROR;
   }
   if (p->count == 0) return VERROR;

   /* compute byte offset for proof[0].bnum */
   memcpy(&first_bnum, p->proof[0].bnum, 8);

   fp = fopen(tfname, "rb");
   if (fp == NULL) return VERROR;

   /* check tfile length covers the proof range */
   if (fseek64(fp, 0LL, SEEK_END) != 0) { fclose(fp); return VERROR; }
   len = ftell64(fp);
   off = (long long)(first_bnum * sizeof(BTRAILER));
   if (len < off + (long long)(p->count * sizeof(BTRAILER))) {
      pdebug("sg_proof_match_tfile: tfile too short (len=%lld, need=%lld)",
         len, off + (long long)(p->count * sizeof(BTRAILER)));
      fclose(fp);
      return VERROR;
   }
   if (fseek64(fp, off, SEEK_SET) != 0) { fclose(fp); return VERROR; }

   for (i = 0; i < p->count; i++) {
      if (fread(&bt, sizeof(BTRAILER), 1, fp) != 1) {
         pdebug("sg_proof_match_tfile: fread failed at proof[%u]", i);
         fclose(fp);
         return VERROR;
      }
      if (memcmp(&bt, &p->proof[i], sizeof(BTRAILER)) != 0) {
         word32 tfile_bnum = (word32) get32(bt.bnum);
         word32 proof_bnum = (word32) get32(p->proof[i].bnum);
         pdebug("sg_proof_match_tfile: mismatch at proof[%u] "
            "(tfile bnum=0x%x, proof bnum=0x%x)",
            i, tfile_bnum, proof_bnum);
         fclose(fp);
         return VERROR;
      }
   }
   fclose(fp);
   return VEOK;
}

/* ---- proof chain validation ---------------------------------------- */

/* Structural validation of NTFTX trailers:
 *   - phash of trailer[i+1] must match bhash of trailer[i] (for v3.0+ blocks)
 *   - bnum must increment by 1 across consecutive trailers
 *   - final trailer's bhash must equal advertised_hash
 *   - final trailer's bnum must equal advertised_bnum
 * No PoW check (too expensive for per-peer pre-admission scan).
 */
int sg_validate_proof_chain(const BTRAILER *proof, word32 count,
                             const word8 advertised_hash[HASHLEN],
                             const word8 advertised_bnum[8])
{
   word32 i;
   word8 expect_bnum[8];

   if (proof == NULL || count < 2) return VERROR;

   for (i = 1; i < count; i++) {
      /* bnum must increment by 1 */
      memcpy(expect_bnum, proof[i - 1].bnum, 8);
      if (add64(expect_bnum, (word8[]){1,0,0,0,0,0,0,0}, expect_bnum)) {
         return VERROR;  /* overflow */
      }
      if (memcmp(expect_bnum, proof[i].bnum, 8) != 0) return VERROR;
      /* phash of this trailer must match bhash of previous trailer */
      if (memcmp(proof[i].phash, proof[i - 1].bhash, HASHLEN) != 0) {
         return VERROR;
      }
   }
   /* tip must match advertised */
   if (memcmp(proof[count - 1].bhash, advertised_hash, HASHLEN) != 0) {
      return VERROR;
   }
   if (memcmp(proof[count - 1].bnum, advertised_bnum, 8) != 0) {
      return VERROR;
   }
   return VEOK;
}

/* end include guard */
#endif
