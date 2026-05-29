#!/bin/bash
# Per-rank %CPU + RES timeline of a monofonIC run, bucketed by pipeline stage.
#
# Reads /proc/PID/{stat,status} directly (top rounds RES to 0.1 GB, hiding
# small rank-to-rank differences). Stage boundaries are detected from rank-0
# stdout markers like "Generating white noise field", "Computing phi(2) term",
# etc.; each top-style sample is bucketed to the most recent stage by epoch
# timestamp.
#
# Usage:
#   tests/scripts/measure_top_stages.sh <monofonIC-binary> <np> <conf>
#
# Env vars:
#   SAMPLE_DT  sampling interval in seconds (default 0.3)
#   I_MPI_PIN_DOMAIN=omp recommended when NumThreads>1 with Intel MPI
#
# Outputs (in current working directory):
#   run_<tag>.log   timestamped monofonIC stdout
#   proc_<tag>.log  raw per-PID samples (epoch pid %cpu rss_kb)
# plus a per-stage per-rank table printed to stdout.
set -u

BIN=${1:?monofonIC binary required}
NP=${2:?np required}
CONF=${3:?conf required}
TAG=$(basename "${CONF%.conf}")_np${NP}
LOG="run_${TAG}.log"
PROCLOG="proc_${TAG}.log"
SAMPLE_DT=${SAMPLE_DT:-0.3}

rm -f "$LOG" "$PROCLOG"

mpirun -np "$NP" "$BIN" "$CONF" > >(
    awk '{ cmd="date +%s.%N"; cmd | getline ts; close(cmd);
           print ts " " $0; fflush() }' >"$LOG"
) 2>&1 &
MPIRUN_PID=$!

# Find descendant monofonIC PIDs (rank order = ascending PID under hydra
# on a single node).
PIDS=""
for _ in $(seq 1 100); do
    PIDS=$(pstree -p "$MPIRUN_PID" 2>/dev/null \
           | grep -oE 'monofonIC\([0-9]+\)' \
           | grep -oE '[0-9]+' | sort -n | head -"$NP" | paste -sd' ')
    NPID=$(echo $PIDS | wc -w)
    [ "$NPID" -ge "$NP" ] && break
    sleep 0.1
done
[ -z "$PIDS" ] && { echo "FAIL: no monofonIC procs" >&2; wait $MPIRUN_PID; exit 1; }
echo "  PIDs: $PIDS" >&2

HZ=$(getconf CLK_TCK)

python3 - "$MPIRUN_PID" "$PROCLOG" "$SAMPLE_DT" "$HZ" $PIDS <<'PY' &
import os, sys, time
mpirun_pid = int(sys.argv[1]); outpath = sys.argv[2]
dt = float(sys.argv[3]); hz = int(sys.argv[4])
pids = [int(p) for p in sys.argv[5:]]

def alive(pid):
    try: os.kill(pid, 0); return True
    except OSError: return False

def rss_kb(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except FileNotFoundError: pass
    return None

def jiffies(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            s = f.read()
        rp = s.rindex(')')
        rest = s[rp+2:].split()
        return int(rest[11]) + int(rest[12])    # utime + stime
    except (FileNotFoundError, ValueError, IndexError):
        return None

prev = {p: None for p in pids}
prev_t = None
out = open(outpath, "w")
while alive(mpirun_pid):
    t = time.time()
    rows = []
    for p in pids:
        j = jiffies(p); r = rss_kb(p)
        if j is None or r is None: continue
        if prev[p] is not None and prev_t is not None and t > prev_t:
            cpu_pct = 100.0 * (j - prev[p]) / hz / (t - prev_t)
        else:
            cpu_pct = float('nan')
        rows.append((p, cpu_pct, r))
        prev[p] = j
    prev_t = t
    for p, c, r in rows:
        if c == c:
            out.write(f"{t:.3f} {p} {c:.2f} {r}\n")
    out.flush()
    time.sleep(dt)
out.close()
PY
PROBE_PID=$!

wait $MPIRUN_PID 2>/dev/null || true
wait $PROBE_PID  2>/dev/null || true

python3 - "$LOG" "$PROCLOG" <<'PY'
import sys, re, collections, statistics, bisect
log_path, proc_path = sys.argv[1], sys.argv[2]

stage_markers = [
    (r"Initializing plugins",                    "init_plugins"),
    (r"Generating white noise field",            "white_noise"),
    (r"Generating LPT fields",                   "lpt_setup"),
    (r"Computing phi\(1\) term",                 "lpt_phi1"),
    (r"Computing phi\(2\) term",                 "lpt_phi2"),
    (r"Computing phi\(3a\) term",                "lpt_phi3a"),
    (r"Computing phi\(3b\) term",                "lpt_phi3b"),
    (r"Computing A\(3\) term",                   "lpt_A3"),
    (r"Computing ICs for species . Dark matter", "ic_dm"),
    (r"Computing ICs for species . Baryons",     "ic_bar"),
    (r"K-section particle partition active",     "ksec_partition"),
    (r"Finalizing",                              "finalize"),
]
markers = [(re.compile(p), n) for p, n in stage_markers]

transitions = []
with open(log_path, errors="replace") as f:
    for line in f:
        clean = re.sub(r'\x1b\[[0-9;]*m', '', line)
        m = re.match(r"^([0-9.]+)\s+(.*)$", clean)
        if not m: continue
        ts = float(m.group(1)); rest = m.group(2)
        for pat, name in markers:
            if pat.search(rest):
                transitions.append((ts, name)); break

samples = []
with open(proc_path) as f:
    for line in f:
        parts = line.split()
        if len(parts) < 4: continue
        try:
            ts = float(parts[0]); pid = parts[1]
            cpu = float(parts[2]); rss_kb = int(parts[3])
        except ValueError:
            continue
        samples.append((ts, pid, cpu, rss_kb / 1024.0))

if not transitions: print("no stages"); sys.exit(1)
if not samples: print("no samples"); sys.exit(1)

all_pids = sorted({s[1] for s in samples}, key=int)
rank_of = {pid: i for i, pid in enumerate(all_pids)}

trans_ts = [t for t, _ in transitions]
trans_nm = [n for _, n in transitions]
def stage_at(ts):
    i = bisect.bisect_right(trans_ts, ts) - 1
    return trans_nm[i] if i >= 0 else "pre_init"

cpu_by = collections.defaultdict(list)
mem_by = collections.defaultdict(list)
for ts, pid, cpu, mem in samples:
    rk = rank_of[pid]; st = stage_at(ts)
    cpu_by[(st, rk)].append(cpu)
    mem_by[(st, rk)].append(mem)

seen=set(); stage_order=[]
for _, n in transitions:
    if n not in seen: seen.add(n); stage_order.append(n)

rn = list(range(len(all_pids)))
hdr_cpu = " | ".join(f"r{r} %CPU" for r in rn)
hdr_mem = " | ".join(f"r{r} RES MB" for r in rn)
print(f"{'stage':<16} | n  | " + hdr_cpu + " || " + hdr_mem)
print("-" * (24 + 11*len(rn) + 13*len(rn)))

for st in stage_order:
    nsamp = max((len(cpu_by.get((st, r), [])) for r in rn), default=0)
    if nsamp == 0: continue
    cpu_row, mem_row = [], []
    for r in rn:
        c = cpu_by.get((st, r), [])
        m = mem_by.get((st, r), [])
        cpu_row.append(f"{statistics.mean(c):6.1f}" if c else "  --  ")
        mem_row.append(f"{max(m):8.1f}" if m else "    --  ")
    print(f"{st:<16} | {nsamp:>2d} | " + " | ".join(cpu_row) + " || " + " | ".join(mem_row))

print()
print("OVERALL  (mean %CPU, peak RES MB):")
for r in rn:
    c = [x for (st, rr), v in cpu_by.items() if rr == r for x in v]
    m = [x for (st, rr), v in mem_by.items() if rr == r for x in v]
    if c:
        print(f"  rank {r}: %CPU mean={statistics.mean(c):6.1f}  "
              f"peak={max(c):6.1f}   "
              f"RES peak={max(m):8.1f} MB  (n={len(c)})")
PY
