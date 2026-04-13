/**
 * @file syncguard.h
 * @brief Sync subsystem hardening: proof spot-check, bad-tfile cache, chain re-selection.
 * @copyright Adequate Systems LLC, 2026. All Rights Reserved.
 * <br />For license information, please refer to ../LICENSE.md
 *
 * This unit provides defense-in-depth for the node sync flow:
 *   - Per-peer proof segment spot-check (NTFTX trailers) before the peer
 *     is admitted to a quorum in scan_quorum()
 *   - Session-local cache of known-bad tfile hashes to avoid repeated
 *     validation of the same garbage across multiple quorum members
 *   - Session-local cache of known-bad (weight, hash) chain pairs used
 *     as an exclusion list when re-scanning after a quorum fails
 *
 * All caches are in-memory and cleared at the start of each resync()
 * attempt. No on-disk persistence.
 */

/* include guard */
#ifndef MOCHIMO_SYNCGUARD_H
#define MOCHIMO_SYNCGUARD_H

#include "types.h"

/* Sizing constants */
#ifndef SYNCGUARD_BAD_CHAIN_MAX
#define SYNCGUARD_BAD_CHAIN_MAX    8    /**< max excluded chains per session */
#endif
#ifndef SYNCGUARD_BAD_TFILE_MAX
#define SYNCGUARD_BAD_TFILE_MAX    16   /**< max cached bad tfile hashes */
#endif
#ifndef SYNCGUARD_PROOF_CACHE_MAX
#define SYNCGUARD_PROOF_CACHE_MAX  64   /**< max cached peer proof segments */
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Lifecycle */
void sg_session_reset(void);

/* Bad-chain exclusion list (session-local) */
int  sg_bad_chain_check(const word8 weight[32], const word8 hash[HASHLEN]);
void sg_bad_chain_add(const word8 weight[32], const word8 hash[HASHLEN]);

/* Bad-tfile hash cache (session-local) */
int  sg_bad_tfile_check(const word8 hash[HASHLEN]);
void sg_bad_tfile_add(const word8 hash[HASHLEN]);
int  sg_hash_file(const char *fname, word8 out[HASHLEN]);

/* Per-peer proof segment cache (session-local) */
void sg_proof_store(word32 ip, const BTRAILER *proof, word32 count);
int  sg_proof_get(word32 ip, BTRAILER *out, word32 count);
int  sg_proof_match_tfile_tail(word32 ip, const char *tfname, word32 count);
void sg_proof_clear_all(void);

/* Proof validation: structural chain climb + tip hash match against advertised.
 * Does NOT validate PoW (too expensive per-peer; full PoW is validated
 * downstream in validate_tfile_pow()).
 * Returns VEOK on pass, VERROR on fail. */
int  sg_validate_proof_chain(const BTRAILER *proof, word32 count,
                              const word8 advertised_hash[HASHLEN],
                              const word8 advertised_bnum[8]);

#ifdef __cplusplus
}  /* end extern "C" */
#endif

/* end include guard */
#endif
