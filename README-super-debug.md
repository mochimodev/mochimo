# super_debug — forensic instrumentation build

A compile-time-opt-in instrumentation layer for mochimo, designed so we never have to guess again about what happened to a node or what network event caused a problem.

## Build

```bash
make clean mochimo CCARGS="-DSUPER_DEBUG"
```

Everything is gated by `#ifdef SUPER_DEBUG`. Without the flag every hook is a no-op, every header function becomes a `static inline` empty body, and the implementation object is an empty marker typedef — production builds are unaffected.

Optional overrides:
- `-DSUPER_DEBUG_ROOT=\"/some/other/path\"` — change the log output directory (default `/data/mochimo-debug`).
- `-DSUPER_DEBUG_LINE_MAX=8192` — extend per-event line length cap (default 4096).

## Runtime layout

All files live under `/data/mochimo-debug/` (or the override above). Subdirectories are created at startup.

| Subdir | Filename pattern | Rotation | Purpose |
|---|---|---|---|
| `debug/` | `YYYY-MM-DD.log` | UTC day | General structured events: enter/exit for every instrumented function, gate decisions, heartbeats |
| `peer_events/` | `YYYY-MM-DD.log` | UTC day | One line per received packet after handshake |
| `tx_trail/` | `YYYY-MM-DD.csv` | UTC day | Every TX seen: accepted, duplicate, evil, or fail-to-write |
| `maddr_trail/` | `YYYY-MM-DD.csv` | UTC day | Every accepted block with miner address, difficulty, tx count, size |
| `file_ops/` | `YYYY-MM-DD.log` | UTC day | Reserved: wire up as needed for tfile/ledger/mirror writes |
| `state-dumps/` | `YYYYMMDDTHHMMSSZ.uuuuuu.txt` | per invocation | Full state snapshot |

Files reopen on UTC date rollover. All writes are serialized through a single `pthread_mutex`. Writes fail silently if the disk is unavailable — mochimo keeps running.

## Event schema (debug/*.log)

Every line begins with a fixed prefix:

```
ts=<monotonic_us> utc=<YYYY-MM-DDThh:mm:ss.uuuZ> pid=<N> tid=<N> seq=<N> ev=<category> <key=value ...>
```

Fields:
- `ts` — monotonic microseconds since boot. Use for ordering and measuring intervals; not a wall clock.
- `utc` — UTC wall clock in ISO 8601. Use for correlating with external logs (journalctl, API events).
- `pid` / `tid` — process and thread id. OpenMP threads have distinct `tid`.
- `seq` — monotonic counter across all events (process-global). Gaps indicate missed writes (disk unavailable); non-monotonic jumps never happen.
- `ev` — dotted category (e.g. `contention.enter`, `b_update.accepted`).
- Remaining key-value tokens are category-specific. See below.

## CSV schemas

### tx_trail/*.csv

```
ts_utc,peer_ip,tx_id,tx_fee,result,reason
```

- `tx_id` / `tx_fee` are hex-encoded without a `0x` prefix. Empty string if unknown at reject point.
- `result` is the integer VEOK/VEBAD/VEBAD2/VERROR.
- `reason` is a short human-readable token (`accepted`, `duplicate`, `tx_val_evil errno=24597`, `tx_read_failed`, `tx_fwrite_fail`, etc).

### maddr_trail/*.csv

```
ts_utc,bnum,bhash,maddr,difficulty,tx_count,block_size,type,elapsed_ms
```

- `bnum` — little-endian hex (mochimo native on-wire format).
- `bhash` — 32-byte block hash, full hex.
- `maddr` — 20-byte miner address. Empty string for pseudo / neogen blocks.
- `type` ∈ {`update`, `pseudo`, `neogen`}.
- `elapsed_ms` — time spent in `b_update()` from enter to accepted event.

## Event categories (non-exhaustive, grows with hooks added)

| Event | Where | Key fields |
|---|---|---|
| `super_debug.init` | `super_debug_init()` | `root` `version` |
| `heartbeat` | main loop ~60s | `bnum` `weight` `diff` `mqcount` `nrec` |
| `state_dump.written` | `super_debug_state_dump()` | `path` `reason` |
| `tfile.read` | every `read_tfile()` exit | `caller` `bnum` `count` `returned` `errno` |
| `contention.enter` | `contention()` top | `peer` `peer_bnum` `our_bnum` |
| `contention.gate` | every short-circuit in contention() | `peer` `gate` `result` |
| `contention.fastpath` | eligibility for 1-block catchup | `peer` `gap` `eligible` |
| `contention.catchup_try` / `catchup_result` | the fast-path inner call | `peer` `result` |
| `contention.proof_reject` | proof-scan rejection | `peer` `step` `reason` |
| `contention.proof_ok` | proof accepted, calling syncup | `peer` `splitblock` |
| `contention.syncup_fail` | `syncup()` returned non-VEOK | `peer` `splitblock` |
| `contention.done` | exit on success | `peer` |
| `catchup.enter` / `.exit` | enter/exit | `initial_peers` / `result` `remaining` |
| `catchup.get_file_fail` / `.b_update_fail` / `.b_update.ok` | per-iteration outcomes | `peer` `fname` |
| `syncup.enter` / `.exit` / `.bad` | syncup lifecycle | `peer` `splitblock` `result` |
| `gettx.op_found` / `.op_found.result` | OP_FOUND dispatch | `peer` `blockfound_already` `contention` |
| `gettx.op_tx.result` | OP_TX dispatch | `peer` `process_tx` |
| `gettx.epinklist` / `.pinklist` | pinklisting attribution | `peer` `opcode` |
| `recv_file.*` | file download lifecycle | `peer` `fname` `total` |
| `send_found.begin` / `.read_tfile_fail` | OP_FOUND broadcast | `bnum` `count` |
| `scan_quorum.result` | per-scan outcome | `qcount` `high_bnum` `high_weight` `high_hash_pre` |
| `b_update.enter` / `.accepted` / `.bval_fail` / `.le_update_fail` / `.exit` | block acceptance | `fname` `bnum` `type` `diff` `tcount` `size` |
| `b_val.reject` / `ng_val.reject` | block-validation rejects | `result` `errno` |
| `le_update.ok` / `.reject` | ledger update | `ltfile` `result` `errno` |
| `peer` (from `super_debug_peer_event`) | every received packet after handshake | `peer` `id1` `id2` `op` `status` `weight` `bnum` `hash` `len` |

Expect more events to appear as instrumentation is extended. The schema is append-only; we never rename or remove keys.

## Operational controls

### SIGUSR1 on-demand state dump

```bash
kill -USR1 $(pgrep -f '^../mochimo')
```

Dumps all globals, file sizes, disk-free stats to `/data/mochimo-debug/state-dumps/<ts>.txt`. Cheap; does not pause the main loop for long. Also happens automatically:
- once at startup
- every 10 heartbeats (~10 min)
- on `syncup()` entry

### Analysis recipes

**All events for one peer since a point in time:**

```bash
awk -v since='2026-04-18T18:53:00' '$0 ~ "peer=35.211.14.30" && $2 >= "utc=" since' /data/mochimo-debug/debug/2026-04-18.log
```

**Block-production distribution by miner, last 24h:**

```bash
awk -F, 'NR>1 {print $4}' /data/mochimo-debug/maddr_trail/2026-04-18.csv \
  | sort | uniq -c | sort -rn
```

**Every read_tfile call that returned 0 (error):**

```bash
grep -E 'ev=tfile.read .*returned=0' /data/mochimo-debug/debug/2026-04-18.log
```

**Contention gate-failure distribution:**

```bash
grep 'ev=contention.gate' /data/mochimo-debug/debug/2026-04-18.log \
  | grep -oE 'gate=[a-z_]+' | sort | uniq -c | sort -rn
```

**Rejected TXs grouped by reason:**

```bash
awk -F, 'NR>1 && $5 != 0 {print $6}' /data/mochimo-debug/tx_trail/2026-04-18.csv \
  | sort | uniq -c | sort -rn
```

**Peer weights observed at OP_FOUND, plus ours, over an hour:**

```bash
grep '^ts=.*ev=peer.*op=4' /data/mochimo-debug/peer_events/2026-04-18.log \
  | awk '{for(i=1;i<=NF;i++) if ($i~/^peer=|^weight=|^bnum=/) printf "%s ", $i; print ""}'
```

## Storage budget

With every hook active on a healthy-synced node, rough rate at full mainnet traffic is ~5-20 GB/day uncompressed. With gzip of files ≥7 days old, expect ~1 GB/day compressed. A 2 TB log disk holds ~years of data.

There is no built-in rotation beyond UTC-day filename sharding. Compress old logs with logrotate or a cron script when convenient; they can be removed at any time without affecting the running node.

## Performance

All log writes happen inside a single mutex. At high event rates (tens of thousands per second during block acceptance bursts) the mutex can become a contention point. In practice mochimo's event rate is modest (~100-500 events per second under normal load) and this is not an issue. If it ever becomes one, replace the mutex with a lock-free MPMC queue drained by a dedicated writer thread — the event-emission API is already lock-owning on its output so this change is localized.

Cost per event when enabled is roughly:
- 1 `clock_gettime(CLOCK_MONOTONIC)` call
- 1 `clock_gettime(CLOCK_REALTIME)` call
- 1 `snprintf` of ~200 bytes
- 1 `pthread_mutex_lock` / `unlock` pair
- 1 `fputs` / `fputc` (line-buffered)

Empirically, ~1-3 µs per event on the hardware we've tested. A 500 events/sec workload costs under 0.2% CPU.

## Deployment notes

- Deploy on a small number of nodes (we're using `test-instance` / `35.211.126.143`), not on reference / public-serving nodes.
- The log disk (`/data/mochimo-debug`) is a separate mount; detach if you ever want to ship logs off-box without interrupting the service.
- `systemctl restart mochimo` re-emits the init state dump, giving you a clean marker for each run.
- Never run a super-debug build on a host that also serves normal traffic if disk headroom is tight; the log volume can accumulate faster than logrotate.
