/**
 * @file super_debug.c
 * @brief Implementation of the SUPER_DEBUG forensic instrumentation.
 *
 * Compiled in only when -DSUPER_DEBUG is set. Every entrypoint here
 * is fail-silent: if the log partition goes away, if a write fails,
 * if the process is under memory pressure, we swallow the error and
 * let production code keep running unaffected.
 */

#ifdef SUPER_DEBUG

#define _GNU_SOURCE

#include "super_debug.h"
#include "global.h"
#include "types.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

/* =======================================================================
 * Static state
 * ======================================================================= */

enum {
   CAT_DEBUG       = 0,
   CAT_PEER        = 1,
   CAT_TX          = 2,
   CAT_MADDR       = 3,
   CAT_FILE_OPS    = 4,
   CAT_COUNT
};

struct log_channel {
   const char *subdir;
   const char *suffix;     /* "log" or "csv" */
   const char *csv_header; /* NULL for non-CSV */
   FILE       *fp;
   char        open_date[11]; /* YYYY-MM-DD\0 */
};

static struct log_channel g_channels[CAT_COUNT] = {
   [CAT_DEBUG]    = { "debug",       "log", NULL, NULL, "" },
   [CAT_PEER]     = { "peer_events", "log", NULL, NULL, "" },
   [CAT_TX]       = { "tx_trail",    "csv",
      "ts_utc,peer_ip,tx_id,tx_fee,result,reason\n",                       NULL, "" },
   [CAT_MADDR]    = { "maddr_trail", "csv",
      "ts_utc,bnum,bhash,maddr,difficulty,tx_count,block_size,type,elapsed_ms\n",
                                                                            NULL, "" },
   [CAT_FILE_OPS] = { "file_ops",    "log", NULL, NULL, "" }
};

static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static volatile int    g_initialized = 0;
static volatile int    g_sigusr1_pending = 0;
static _Atomic uint64_t g_seq = 0;
static pid_t           g_pid = 0;

/* =======================================================================
 * Small helpers
 * ======================================================================= */

static uint64_t monotonic_us(void)
{
   struct timespec ts;
   if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0;
   return (uint64_t)ts.tv_sec * 1000000ull + (uint64_t)(ts.tv_nsec / 1000);
}

static void utc_now(struct timespec *out)
{
   if (clock_gettime(CLOCK_REALTIME, out) != 0) {
      out->tv_sec = 0; out->tv_nsec = 0;
   }
}

static void utc_iso(char *buf, size_t buflen)
{
   struct timespec ts;
   struct tm tm;
   utc_now(&ts);
   gmtime_r(&ts.tv_sec, &tm);
   snprintf(buf, buflen,
      "%04d-%02d-%02dT%02d:%02d:%02d.%06ldZ",
      tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
      tm.tm_hour, tm.tm_min, tm.tm_sec, (long)(ts.tv_nsec / 1000));
}

static void utc_date(char *buf, size_t buflen)
{
   struct timespec ts;
   struct tm tm;
   utc_now(&ts);
   gmtime_r(&ts.tv_sec, &tm);
   snprintf(buf, buflen, "%04d-%02d-%02d",
      tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday);
}

static pid_t gettid_compat(void)
{
#ifdef SYS_gettid
   return (pid_t) syscall(SYS_gettid);
#else
   return getpid();
#endif
}

static void hex_encode(const void *src, size_t srclen, char *dst, size_t dstlen)
{
   static const char hex[] = "0123456789abcdef";
   const unsigned char *s = (const unsigned char *) src;
   size_t i, n = srclen;
   if (dstlen < n * 2 + 1) n = (dstlen - 1) / 2;
   for (i = 0; i < n; i++) {
      dst[i*2]   = hex[(s[i] >> 4) & 0xf];
      dst[i*2+1] = hex[s[i] & 0xf];
   }
   dst[n * 2] = '\0';
}

static void sd_ensure_dir(const char *path)
{
   char buf[512];
   size_t n = strlen(path);
   if (n >= sizeof(buf)) return;
   memcpy(buf, path, n + 1);
   /* extio's mkdir_p takes non-const char*; reuse rather than
    * reimplement recursive-mkdir logic. */
   mkdir_p(buf);
}

/* Open (or reopen) a channel's log file for the current UTC date.
 * Called with g_mu held. Returns 0 on success, -1 on failure. */
static int channel_open_locked(struct log_channel *ch)
{
   char date[11];
   char path[512];
   int new_file = 0;

   utc_date(date, sizeof(date));

   /* Already open and date still current? */
   if (ch->fp != NULL && strcmp(date, ch->open_date) == 0) return 0;

   /* Close stale handle if open. */
   if (ch->fp != NULL) {
      fclose(ch->fp);
      ch->fp = NULL;
   }

   snprintf(path, sizeof(path), "%s/%s/%s.%s",
      SUPER_DEBUG_ROOT, ch->subdir, date, ch->suffix);

   /* If the target file doesn't already exist and we're a CSV, we'll
    * need to write the header on open. */
   if (ch->csv_header != NULL) {
      struct stat st;
      if (stat(path, &st) != 0) new_file = 1;
   }

   ch->fp = fopen(path, "a");
   if (ch->fp == NULL) return -1;

   /* Line-buffered so tail -f is useful. */
   setvbuf(ch->fp, NULL, _IOLBF, 0);

   if (new_file && ch->csv_header != NULL) {
      fputs(ch->csv_header, ch->fp);
   }

   memcpy(ch->open_date, date, sizeof(ch->open_date));
   return 0;
}

/* Format the standard event prefix into buf. Returns length. */
static int fmt_prefix(char *buf, size_t buflen, const char *category)
{
   char iso[40];
   uint64_t seq = __atomic_fetch_add(&g_seq, 1, __ATOMIC_RELAXED);
   utc_iso(iso, sizeof(iso));
   return snprintf(buf, buflen,
      "ts=%" PRIu64 " utc=%s pid=%d tid=%d seq=%" PRIu64 " ev=%s ",
      monotonic_us(), iso, (int) g_pid, (int) gettid_compat(),
      seq, category ? category : "?");
}

/* =======================================================================
 * SIGUSR1 handler
 * ======================================================================= */

static void sigusr1_handler(int sig)
{
   (void) sig;
   /* async-signal-safe: just flip the flag and let the main loop dump. */
   g_sigusr1_pending = 1;
}

/* =======================================================================
 * Public API
 * ======================================================================= */

void super_debug_init(void)
{
   int i;
   struct sigaction sa;
   char root[sizeof(SUPER_DEBUG_ROOT) + 32];

   if (g_initialized) return;

   g_pid = getpid();

   /* Ensure all subdirectories exist. mkdir_p tolerates races. */
   sd_ensure_dir(SUPER_DEBUG_ROOT);
   for (i = 0; i < CAT_COUNT; i++) {
      snprintf(root, sizeof(root), "%s/%s",
         SUPER_DEBUG_ROOT, g_channels[i].subdir);
      sd_ensure_dir(root);
   }
   snprintf(root, sizeof(root), "%s/state-dumps", SUPER_DEBUG_ROOT);
   sd_ensure_dir(root);

   /* Install SIGUSR1 handler without disturbing existing mochimo
    * signal setup. SA_RESTART so syscalls resume. */
   memset(&sa, 0, sizeof(sa));
   sa.sa_handler = sigusr1_handler;
   sigemptyset(&sa.sa_mask);
   sa.sa_flags = SA_RESTART;
   sigaction(SIGUSR1, &sa, NULL);

   g_initialized = 1;

   super_debug_emit("super_debug.init",
      "root=%s pid=%d version=%s",
      SUPER_DEBUG_ROOT, (int) g_pid,
#ifdef VERSION
      VERSION
#else
      "unknown"
#endif
   );
   super_debug_state_dump("init");
}

void super_debug_emit(const char *category, const char *fmt, ...)
{
   char line[SUPER_DEBUG_LINE_MAX];
   int pre_n, body_n;
   va_list ap;

   if (!g_initialized) return;

   pre_n = fmt_prefix(line, sizeof(line), category);
   if (pre_n < 0) return;
   if ((size_t) pre_n >= sizeof(line)) pre_n = sizeof(line) - 1;

   va_start(ap, fmt);
   body_n = vsnprintf(line + pre_n, sizeof(line) - pre_n, fmt, ap);
   va_end(ap);
   if (body_n < 0) body_n = 0;

   pthread_mutex_lock(&g_mu);
   if (channel_open_locked(&g_channels[CAT_DEBUG]) == 0) {
      fputs(line, g_channels[CAT_DEBUG].fp);
      fputc('\n', g_channels[CAT_DEBUG].fp);
   }
   pthread_mutex_unlock(&g_mu);
}

void super_debug_peer_event(
   word32 ip, word16 id1, word16 id2,
   int opcode, int status, int result,
   const void *weight32, const void *bnum8,
   const void *hash32, word16 len,
   const char *note)
{
   char line[SUPER_DEBUG_LINE_MAX];
   char weight_hex[65] = "";
   char bnum_hex[17] = "";
   char hash_hex[65] = "";
   char ipstr[16];
   int pre_n, body_n;
   unsigned char *ipb = (unsigned char *) &ip;

   if (!g_initialized) return;

   if (weight32 != NULL) hex_encode(weight32, 32, weight_hex, sizeof(weight_hex));
   if (bnum8   != NULL) hex_encode(bnum8,    8,  bnum_hex,   sizeof(bnum_hex));
   if (hash32  != NULL) hex_encode(hash32,  32,  hash_hex,   sizeof(hash_hex));
   snprintf(ipstr, sizeof(ipstr), "%u.%u.%u.%u",
      ipb[0], ipb[1], ipb[2], ipb[3]);

   pre_n = fmt_prefix(line, sizeof(line), "peer");
   if (pre_n < 0) return;
   if ((size_t) pre_n >= sizeof(line)) pre_n = sizeof(line) - 1;

   body_n = snprintf(line + pre_n, sizeof(line) - pre_n,
      "peer=%s id1=%04x id2=%04x op=%d status=%d result=%d "
      "len=%u weight=%s bnum=%s hash=%s note=%s",
      ipstr, id1, id2, opcode, status, result, (unsigned) len,
      weight_hex, bnum_hex, hash_hex, note ? note : "");
   (void) body_n;

   pthread_mutex_lock(&g_mu);
   if (channel_open_locked(&g_channels[CAT_PEER]) == 0) {
      fputs(line, g_channels[CAT_PEER].fp);
      fputc('\n', g_channels[CAT_PEER].fp);
   }
   pthread_mutex_unlock(&g_mu);
}

void super_debug_maddr_trail(
   const void *bnum8, const void *bhash32,
   const void *maddr20, word32 difficulty,
   word32 tcount, size_t block_size,
   const char *blktype, int elapsed_ms)
{
   char iso[40];
   char bnum_hex[17] = "";
   char hash_hex[65] = "";
   char maddr_hex[41] = "";

   if (!g_initialized) return;
   utc_iso(iso, sizeof(iso));
   if (bnum8   != NULL) hex_encode(bnum8,    8,  bnum_hex,   sizeof(bnum_hex));
   if (bhash32 != NULL) hex_encode(bhash32, 32,  hash_hex,   sizeof(hash_hex));
   if (maddr20 != NULL) hex_encode(maddr20, 20,  maddr_hex,  sizeof(maddr_hex));

   pthread_mutex_lock(&g_mu);
   if (channel_open_locked(&g_channels[CAT_MADDR]) == 0) {
      fprintf(g_channels[CAT_MADDR].fp,
         "%s,%s,%s,%s,%u,%u,%zu,%s,%d\n",
         iso, bnum_hex, hash_hex, maddr_hex,
         (unsigned) difficulty, (unsigned) tcount, block_size,
         blktype ? blktype : "", elapsed_ms);
   }
   pthread_mutex_unlock(&g_mu);
}

void super_debug_tx_trail(
   word32 peer_ip, const void *tx_id32,
   const void *tx_fee8, int result,
   const char *reason)
{
   char iso[40];
   char ipstr[16];
   char id_hex[65] = "";
   char fee_hex[17] = "";
   unsigned char *ipb = (unsigned char *) &peer_ip;

   if (!g_initialized) return;
   utc_iso(iso, sizeof(iso));
   if (tx_id32  != NULL) hex_encode(tx_id32,  32, id_hex,  sizeof(id_hex));
   if (tx_fee8  != NULL) hex_encode(tx_fee8,  8,  fee_hex, sizeof(fee_hex));
   snprintf(ipstr, sizeof(ipstr), "%u.%u.%u.%u",
      ipb[0], ipb[1], ipb[2], ipb[3]);

   pthread_mutex_lock(&g_mu);
   if (channel_open_locked(&g_channels[CAT_TX]) == 0) {
      /* Escape commas/newlines in reason by replacing them. */
      char safe[256];
      size_t i;
      if (reason == NULL) reason = "";
      for (i = 0; i < sizeof(safe) - 1 && reason[i]; i++) {
         char c = reason[i];
         safe[i] = (c == ',' || c == '\n' || c == '\r') ? ' ' : c;
      }
      safe[i] = '\0';
      fprintf(g_channels[CAT_TX].fp,
         "%s,%s,%s,%s,%d,%s\n",
         iso, ipstr, id_hex, fee_hex, result, safe);
   }
   pthread_mutex_unlock(&g_mu);
}

void super_debug_file_op(
   const char *path, const char *op,
   long long before, long long after,
   const char *note)
{
   char line[SUPER_DEBUG_LINE_MAX];
   int pre_n, body_n;

   if (!g_initialized) return;

   pre_n = fmt_prefix(line, sizeof(line), "file_op");
   if (pre_n < 0) return;
   if ((size_t) pre_n >= sizeof(line)) pre_n = sizeof(line) - 1;

   body_n = snprintf(line + pre_n, sizeof(line) - pre_n,
      "path=%s op=%s before=%lld after=%lld delta=%lld note=%s",
      path ? path : "", op ? op : "", before, after,
      after - before, note ? note : "");
   (void) body_n;

   pthread_mutex_lock(&g_mu);
   if (channel_open_locked(&g_channels[CAT_FILE_OPS]) == 0) {
      fputs(line, g_channels[CAT_FILE_OPS].fp);
      fputc('\n', g_channels[CAT_FILE_OPS].fp);
   }
   pthread_mutex_unlock(&g_mu);
}

void super_debug_read_tfile(
   const void *caller_pc, const void *bnum8,
   size_t count, size_t returned, int errnum)
{
   char bnum_hex[17] = "";
   if (!g_initialized) return;
   if (bnum8 != NULL) hex_encode(bnum8, 8, bnum_hex, sizeof(bnum_hex));
   super_debug_emit("tfile.read",
      "caller=%p bnum=%s count=%zu returned=%zu errno=%d errmsg=\"%s\"",
      caller_pc, bnum_hex, count, returned, errnum,
      errnum ? strerror(errnum) : "");
}

/* =======================================================================
 * State dump + heartbeat
 * ======================================================================= */

static void append_file_stat(FILE *out, const char *label, const char *path)
{
   struct stat st;
   if (stat(path, &st) == 0) {
      fprintf(out, "  %s: size=%lld mtime=%ld mode=%o uid=%u gid=%u\n",
         label, (long long) st.st_size, (long) st.st_mtime,
         (unsigned) (st.st_mode & 07777),
         (unsigned) st.st_uid, (unsigned) st.st_gid);
   } else {
      fprintf(out, "  %s: stat_failed errno=%d (%s)\n",
         label, errno, strerror(errno));
   }
}

static void append_disk_free(FILE *out, const char *label, const char *path)
{
   struct statvfs vfs;
   if (statvfs(path, &vfs) == 0) {
      unsigned long long free_bytes = (unsigned long long) vfs.f_bavail * vfs.f_frsize;
      unsigned long long total_bytes = (unsigned long long) vfs.f_blocks * vfs.f_frsize;
      fprintf(out, "  %s: free=%llu total=%llu (%llu%% used)\n",
         label, free_bytes, total_bytes,
         total_bytes == 0 ? 0 :
         100ULL - (free_bytes * 100ULL / total_bytes));
   } else {
      fprintf(out, "  %s: statvfs_failed errno=%d\n", label, errno);
   }
}

void super_debug_state_dump(const char *reason)
{
   char path[512];
   char iso[40];
   char bnum_hex[17], hash_hex[65], weight_hex[65], prevhash_hex[65];
   struct timespec ts;
   struct tm tm;
   FILE *out;

   if (!g_initialized) return;

   utc_now(&ts);
   gmtime_r(&ts.tv_sec, &tm);
   snprintf(path, sizeof(path),
      "%s/state-dumps/%04d%02d%02dT%02d%02d%02dZ.%06ld.txt",
      SUPER_DEBUG_ROOT,
      tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
      tm.tm_hour, tm.tm_min, tm.tm_sec,
      (long) (ts.tv_nsec / 1000));

   out = fopen(path, "w");
   if (out == NULL) {
      super_debug_emit("state_dump.fail",
         "path=%s errno=%d errmsg=\"%s\"",
         path, errno, strerror(errno));
      return;
   }

   utc_iso(iso, sizeof(iso));
   hex_encode(Cblocknum, 8, bnum_hex, sizeof(bnum_hex));
   hex_encode(Cblockhash, 32, hash_hex, sizeof(hash_hex));
   hex_encode(Weight, 32, weight_hex, sizeof(weight_hex));
   hex_encode(Prevhash, 32, prevhash_hex, sizeof(prevhash_hex));

   fprintf(out, "# mochimo super_debug state dump\n");
   fprintf(out, "utc: %s\n", iso);
   fprintf(out, "reason: %s\n", reason ? reason : "unspecified");
   fprintf(out, "pid: %d\n", (int) g_pid);
#ifdef VERSION
   fprintf(out, "version: %s\n", VERSION);
#endif
   fprintf(out, "\n## chain state\n");
   fprintf(out, "  Cblocknum: %s\n", bnum_hex);
   fprintf(out, "  Cblockhash: %s\n", hash_hex);
   fprintf(out, "  Prevhash: %s\n", prevhash_hex);
   fprintf(out, "  Weight: %s\n", weight_hex);
   fprintf(out, "  Difficulty: %u\n", (unsigned) Difficulty);
   fprintf(out, "  Time0: %u\n", (unsigned) Time0);
   fprintf(out, "  Bgflag: %u\n", (unsigned) Bgflag);
   fprintf(out, "  Insyncup: %u\n", (unsigned) Insyncup);
   fprintf(out, "  Mqcount: %d\n", Mqcount);
   fprintf(out, "  Mqpid: %d\n", (int) Mqpid);
   fprintf(out, "  Ngen: %u\n", (unsigned) Ngen);
   fprintf(out, "  Nlogins: %u\n", (unsigned) Nlogins);
   fprintf(out, "  Nrec: %u\n", (unsigned) Nrec);
   fprintf(out, "  Ndups: %u\n", (unsigned) Ndups);
   fprintf(out, "  Nbadlogs: %u\n", (unsigned) Nbadlogs);
   fprintf(out, "  Ntimeouts: %u\n", (unsigned) Ntimeouts);
   fprintf(out, "  Nspace: %u\n", (unsigned) Nspace);
   fprintf(out, "  Nbalance: %u\n", (unsigned) Nbalance);
   fprintf(out, "  Nupdated: %u\n", (unsigned) Nupdated);
   fprintf(out, "  Txcount: %u\n", (unsigned) Txcount);
   fprintf(out, "  Nonline: %d\n", Nonline);
   fprintf(out, "  Blockfound: %u\n", (unsigned) Blockfound);

   fprintf(out, "\n## file state (d/)\n");
   append_file_stat(out, "tfile.dat",  "tfile.dat");
   append_file_stat(out, "ledger.dat", "ledger.dat");
   append_file_stat(out, "ltran.dat",  "ltran.dat");
   append_file_stat(out, "mirror.dat", "mirror.dat");
   append_file_stat(out, "mq.dat",     "mq.dat");
   append_file_stat(out, "cblock.dat", "cblock.dat");
   append_file_stat(out, "mblock.dat", "mblock.dat");

   fprintf(out, "\n## disk usage\n");
   append_disk_free(out, "cwd",             ".");
   append_disk_free(out, "debug_root",      SUPER_DEBUG_ROOT);

   fclose(out);

   super_debug_emit("state_dump.written", "path=%s reason=%s",
      path, reason ? reason : "");
}

void super_debug_heartbeat(void)
{
   static uint64_t last_heartbeat_us = 0;
   uint64_t now_us;
   char bnum_hex[17], weight_hex[65];

   if (!g_initialized) return;

   /* If SIGUSR1 arrived since the last heartbeat, dump now and clear. */
   if (g_sigusr1_pending) {
      g_sigusr1_pending = 0;
      super_debug_state_dump("sigusr1");
   }

   now_us = monotonic_us();
   if (last_heartbeat_us != 0 && (now_us - last_heartbeat_us) < 60000000ull) {
      return;
   }
   last_heartbeat_us = now_us;

   hex_encode(Cblocknum, 8, bnum_hex, sizeof(bnum_hex));
   hex_encode(Weight,   32, weight_hex, sizeof(weight_hex));

   super_debug_emit("heartbeat",
      "bnum=%s weight=%s diff=%u insyncup=%u mqcount=%d mqpid=%d "
      "ngen=%u nlogins=%u nrec=%u ndups=%u nupdated=%u "
      "nbalance=%u ntimeouts=%u nonline=%d txcount=%u blockfound=%u",
      bnum_hex, weight_hex,
      (unsigned) Difficulty, (unsigned) Insyncup,
      Mqcount, (int) Mqpid,
      (unsigned) Ngen, (unsigned) Nlogins,
      (unsigned) Nrec, (unsigned) Ndups, (unsigned) Nupdated,
      (unsigned) Nbalance, (unsigned) Ntimeouts,
      Nonline, (unsigned) Txcount, (unsigned) Blockfound);

   /* Every 10 heartbeats (~10 min), also write a state dump. */
   {
      static unsigned hb_counter = 0;
      if (++hb_counter % 10 == 0) {
         super_debug_state_dump("periodic");
      }
   }
}

#endif  /* SUPER_DEBUG */

/* ISO C forbids an empty translation unit. When -DSUPER_DEBUG is not
 * set, everything above compiles out; emit a harmless typedef so the
 * object file is still well-formed and -Wpedantic stays quiet. */
typedef int super_debug_translation_unit_marker;

