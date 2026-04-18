/**
 * @file super_debug.h
 * @brief Forensic-grade instrumentation for mochimo.
 *
 * Everything declared here is a no-op at compile time unless
 * -DSUPER_DEBUG is set. To build an instrumented binary:
 *
 *   make clean mochimo CCARGS="-DSUPER_DEBUG"
 *
 * Output layout (controlled by SUPER_DEBUG_ROOT, default
 * "/data/mochimo-debug"):
 *
 *   <root>/debug/YYYY-MM-DD.log          general events (key=value)
 *   <root>/peer_events/YYYY-MM-DD.log    per-packet peer interactions
 *   <root>/tx_trail/YYYY-MM-DD.csv       per-transaction audit trail
 *   <root>/maddr_trail/YYYY-MM-DD.csv    per-accepted-block metadata
 *   <root>/file_ops/YYYY-MM-DD.log       per-write to tfile/ledger/bc/mirror
 *   <root>/state-dumps/<ts>.txt          on-demand & periodic snapshots
 *
 * Files re-open on UTC date rollover. Writes are serialized with a
 * pthread mutex; failures are silent so production behavior is never
 * affected if the log partition disappears.
 *
 * Send SIGUSR1 to the main mochimo process to request an immediate
 * state dump into state-dumps/.
 */

#ifndef MOCHIMO_SUPER_DEBUG_H
#define MOCHIMO_SUPER_DEBUG_H

#include "types.h"

#ifdef SUPER_DEBUG

#include <stddef.h>
#include <stdint.h>

/* Root directory for all debug output. Override at build time via
 * -DSUPER_DEBUG_ROOT=\"/some/path\" if deployment uses a different
 * mount point. */
#ifndef SUPER_DEBUG_ROOT
#define SUPER_DEBUG_ROOT "/data/mochimo-debug"
#endif

/* Max length of a single emitted line before truncation. */
#ifndef SUPER_DEBUG_LINE_MAX
#define SUPER_DEBUG_LINE_MAX 4096
#endif

/* Initialize (creates subdirs if missing, installs SIGUSR1 handler,
 * seeds the monotonic sequence counter). Safe to call more than once;
 * second and subsequent calls are no-ops. */
void super_debug_init(void);

/* Emit a single event line to debug/YYYY-MM-DD.log.
 * @a category is short identifier (e.g. "contention", "block.update");
 * @a fmt is printf-style, should produce key=value tokens. ts, utc,
 * pid, tid, seq prefix is added automatically. */
void super_debug_emit(const char *category, const char *fmt, ...)
   __attribute__((format(printf, 2, 3)));

/* Per-packet peer interaction. result is VEOK/VERROR/VEBAD etc.;
 * note is a short free-form explanation. Any pointer may be NULL. */
void super_debug_peer_event(
   word32 ip, word16 id1, word16 id2,
   int opcode, int status, int result,
   const void *weight32, const void *bnum8,
   const void *hash32, word16 len,
   const char *note);

/* Per-accepted-block record. blktype is one of "real", "pseudo",
 * "neogenesis". bhash32, maddr20 may be NULL if not applicable. */
void super_debug_maddr_trail(
   const void *bnum8, const void *bhash32,
   const void *maddr20, word32 difficulty,
   word32 tcount, size_t block_size,
   const char *blktype, int elapsed_ms);

/* Per-transaction audit trail. Called on every TX we see, regardless
 * of accept/reject. result is VEOK/VEBAD/VEBAD2/VERROR. */
void super_debug_tx_trail(
   word32 peer_ip, const void *tx_id32,
   const void *tx_fee8, int result,
   const char *reason);

/* File operation record. @a before/@a after are sizes in bytes.
 * op is a short verb: "append", "rewrite", "rename", "remove". */
void super_debug_file_op(
   const char *path, const char *op,
   long long before, long long after,
   const char *note);

/* Dump full state to state-dumps/<ts>.txt. reason is free-form. */
void super_debug_state_dump(const char *reason);

/* 60s heartbeat: emits a heartbeat event and writes a concise state
 * line to debug.log. Not a full dump (use state_dump for that). */
void super_debug_heartbeat(void);

/* Caller-side convenience: log a read_tfile() call with caller PC
 * and parameters. */
void super_debug_read_tfile(
   const void *caller_pc, const void *bnum8,
   size_t count, size_t returned, int errnum);

#define SDEBUG(cat, ...)   super_debug_emit((cat), __VA_ARGS__)
#define SDEBUG_INIT()      super_debug_init()
#define SDEBUG_HEARTBEAT() super_debug_heartbeat()

#else  /* !SUPER_DEBUG */

#define SDEBUG(cat, ...)           ((void)0)
#define SDEBUG_INIT()              ((void)0)
#define SDEBUG_HEARTBEAT()         ((void)0)

static inline void super_debug_peer_event(
   word32 ip, word16 id1, word16 id2,
   int opcode, int status, int result,
   const void *weight32, const void *bnum8,
   const void *hash32, word16 len,
   const char *note)
{
   (void)ip; (void)id1; (void)id2; (void)opcode; (void)status;
   (void)result; (void)weight32; (void)bnum8; (void)hash32;
   (void)len; (void)note;
}

static inline void super_debug_maddr_trail(
   const void *bnum8, const void *bhash32,
   const void *maddr20, word32 difficulty,
   word32 tcount, size_t block_size,
   const char *blktype, int elapsed_ms)
{
   (void)bnum8; (void)bhash32; (void)maddr20; (void)difficulty;
   (void)tcount; (void)block_size; (void)blktype; (void)elapsed_ms;
}

static inline void super_debug_tx_trail(
   word32 peer_ip, const void *tx_id32,
   const void *tx_fee8, int result, const char *reason)
{
   (void)peer_ip; (void)tx_id32; (void)tx_fee8;
   (void)result; (void)reason;
}

static inline void super_debug_file_op(
   const char *path, const char *op,
   long long before, long long after,
   const char *note)
{
   (void)path; (void)op; (void)before; (void)after; (void)note;
}

static inline void super_debug_state_dump(const char *reason) { (void)reason; }

static inline void super_debug_read_tfile(
   const void *caller_pc, const void *bnum8,
   size_t count, size_t returned, int errnum)
{
   (void)caller_pc; (void)bnum8; (void)count; (void)returned; (void)errnum;
}

#endif  /* SUPER_DEBUG */

#endif  /* MOCHIMO_SUPER_DEBUG_H */
