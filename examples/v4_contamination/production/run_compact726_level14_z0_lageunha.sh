#!/bin/bash
set -eo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
SOURCE_ROOT=$(git -C "$HERE" rev-parse --show-toplevel)
V4_ROOT=${V4_WORKROOT:-/gpfs/kjhan/VoidSim/v4_parent_n512}
HANDOFF="$V4_ROOT/compact726_level14_pilot"
RUNDIR="$V4_ROOT/compact726_level14_z0"
HANDOFF_OUTPUT="$HANDOFF/ramses/output_00001"
RAMSES_SOURCE="$HANDOFF/ramses/ramses_final3d"
EXPECTED_RAMSES_SHA256=ce3e5a14c22639fd14e3c70092694eeb2b6fb3a009aefafdd7e20fbf585a592e

if [ "$(hostname | tr '[:upper:]' '[:lower:]')" != "lageunha" ]; then
    echo "this launcher must run on LagEunha, not $(hostname)" >&2
    exit 2
fi
if [ ! -e "$HANDOFF/.complete" ] || \
    ! grep -q "V4 COMPACT RANK-726 LEVEL-14 DMO HAND-OFF PASSED" \
        "$HANDOFF/verify_ramses.log"; then
    echo "the compact rank-726 level-14 hand-off gate has not passed" >&2
    exit 2
fi
for required in \
    "$HANDOFF_OUTPUT/COMPLETE" \
    "$HANDOFF_OUTPUT/info_00001.txt" \
    "$RAMSES_SOURCE"; do
    if [ ! -e "$required" ]; then
        echo "required restart input is absent: $required" >&2
        exit 2
    fi
done
if [ ! -x "$RAMSES_SOURCE" ]; then
    echo "RAMSES restart executable is not executable: $RAMSES_SOURCE" >&2
    exit 2
fi
actual_ramses_sha256=$(sha256sum "$RAMSES_SOURCE" | awk '{print $1}')
if [ "$actual_ramses_sha256" != "$EXPECTED_RAMSES_SHA256" ]; then
    echo "validated RAMSES binary checksum mismatch: $actual_ramses_sha256" >&2
    exit 2
fi
available_kib=$(df -Pk "$V4_ROOT" | awk 'NR==2 {print $4}')
required_kib=$((500 * 1024 * 1024))
if [ "$available_kib" -lt "$required_kib" ]; then
    echo "the z=0 evolution requires at least 500 GiB free below $V4_ROOT" >&2
    exit 2
fi

if [ ! -e "$RUNDIR" ]; then
    mkdir -p "$RUNDIR"
    install -m 0644 "$HERE/ramses_compact726_level14_z0.nml" \
        "$RUNDIR/ramses.nml"
    install -m 0755 "$HERE/verify_compact_level14_z0.py" \
        "$RUNDIR/verify_compact_level14_z0.py"
    install -m 0644 "$SOURCE_ROOT/examples/v2_hop_id_file/hop_to_genetic_id.py" \
        "$RUNDIR/hop_to_genetic_id.py"
    install -m 0755 "$RAMSES_SOURCE" "$RUNDIR/ramses_final3d"
    ln -s "$HANDOFF_OUTPUT" "$RUNDIR/output_00001"
    {
        echo "started_at=$(date --iso-8601=seconds)"
        echo "host=$(hostname)"
        echo "target=compact_rank_726"
        echo "restart_output=$HANDOFF_OUTPUT"
        echo "laggenetic_head=$(git -C "$SOURCE_ROOT" rev-parse HEAD)"
        echo "ramses_binary_sha256=$actual_ramses_sha256"
        echo "ranks=64"
        echo "openmp_threads=1"
        echo "void_refine=false"
        sha256sum "$RUNDIR/ramses.nml" "$RUNDIR/ramses_final3d" \
            "$RUNDIR/verify_compact_level14_z0.py" \
            "$RUNDIR/hop_to_genetic_id.py"
    } > "$RUNDIR/provenance.txt"
elif [ -e "$RUNDIR/.complete" ]; then
    echo "compact rank-726 level-14 z=0 DMO evolution is already complete"
    exit 0
fi

if [ ! -L "$RUNDIR/output_00001" ] || \
    [ "$(readlink -f "$RUNDIR/output_00001")" != "$(readlink -f "$HANDOFF_OUTPUT")" ]; then
    echo "output_00001 is not the verified hand-off snapshot" >&2
    exit 2
fi
if [ -s "$RUNDIR/launcher.pid" ]; then
    prior_pid=$(cat "$RUNDIR/launcher.pid")
    if [[ "$prior_pid" =~ ^[0-9]+$ ]] && kill -0 "$prior_pid" 2>/dev/null; then
        echo "launcher PID $prior_pid is already active" >&2
        exit 2
    fi
    echo "stale launcher.pid is present; refusing an ambiguous restart" >&2
    exit 2
fi

latest_restart=0
for output in "$RUNDIR"/output_[0-9][0-9][0-9][0-9][0-9]; do
    [ -e "$output" ] || continue
    if [ ! -f "$output/COMPLETE" ]; then
        echo "incomplete snapshot exists; refusing to overwrite it: $output" >&2
        exit 2
    fi
    suffix=${output##*_}
    index=$((10#$suffix))
    if [ "$index" -gt "$latest_restart" ]; then
        latest_restart=$index
    fi
done
if [ "$latest_restart" -lt 1 ] || [ "$latest_restart" -gt 6 ]; then
    echo "invalid latest restart output: $latest_restart" >&2
    exit 2
fi

if [ -e "$RUNDIR/.ramses.complete" ]; then
    if [ "$latest_restart" -ne 6 ] || ! grep -q "Run completed" "$RUNDIR/ramses.log"; then
        echo "RAMSES completion marker is inconsistent with its output" >&2
        exit 2
    fi
elif [ -e "$RUNDIR/.failed" ]; then
    retry_stamp=$(date +%Y%m%dT%H%M%S)
    resume_reason=recorded_runtime_failure
    if [ "$latest_restart" -eq 1 ] && \
        grep -q "You need to set up namelist &INIT_PARAMS" \
            "$RUNDIR/ramses.log" 2>/dev/null; then
        mv "$RUNDIR/ramses.nml" \
            "$RUNDIR/ramses.missing_init.failed.$retry_stamp.nml"
        install -m 0644 "$HERE/ramses_compact726_level14_z0.nml" \
            "$RUNDIR/ramses.nml"
        resume_reason=missing_init_params_corrected
    elif ! cmp -s "$HERE/ramses_compact726_level14_z0.nml" \
        "$RUNDIR/ramses.nml"; then
        echo "staged namelist differs from the source; refusing an untracked resume" >&2
        exit 2
    fi
    for artifact in ramses.log ramses.time; do
        if [ -e "$RUNDIR/$artifact" ]; then
            mv "$RUNDIR/$artifact" \
                "$RUNDIR/${artifact%.*}.restart${latest_restart}.failed.$retry_stamp.${artifact##*.}"
        fi
    done
    mv "$RUNDIR/.failed" "$RUNDIR/.failed_before_resume_$retry_stamp"
    {
        echo "resumed_at=$(date --iso-8601=seconds)"
        echo "restart_index=$latest_restart"
        echo "resume_reason=$resume_reason"
        echo "resume_laggenetic_head=$(git -C "$SOURCE_ROOT" rev-parse HEAD)"
        sha256sum "$RUNDIR/ramses.nml"
    } >> "$RUNDIR/provenance.txt"
elif [ -e "$RUNDIR/ramses.log" ]; then
    echo "RAMSES log exists without a completion or failure marker" >&2
    exit 2
fi

echo "$$" > "$RUNDIR/launcher.pid"
status=failed
cleanup() {
    if [ "$status" != complete ]; then
        touch "$RUNDIR/.failed"
    fi
    rm -f "$RUNDIR/launcher.pid"
}
trap cleanup EXIT

source /opt/ohpc/pub/intel/oneapi/setvars.sh >/dev/null
set -u
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export I_MPI_PIN=1
export I_MPI_PIN_DOMAIN=core
export I_MPI_PIN_PROCESSOR_LIST=0-63
export I_MPI_FABRICS=shm
ulimit -s unlimited

if [ ! -e "$RUNDIR/.ramses.complete" ]; then
    echo "[1/2] evolve compact rank 726 from output_$(printf '%05d' "$latest_restart") to a=1"
    (
        TIMEFORMAT=$'real_seconds=%R\nuser_seconds=%U\nsystem_seconds=%S'
        cd "$RUNDIR"
        time mpirun -np 64 ./ramses_final3d ramses.nml "$latest_restart" \
            > ramses.log 2>&1
    ) 2> "$RUNDIR/ramses.time"
    if ! grep -q "Run completed" "$RUNDIR/ramses.log" || \
        [ ! -f "$RUNDIR/output_00006/COMPLETE" ]; then
        echo "RAMSES returned without a complete a=1 snapshot" >&2
        exit 2
    fi
    touch "$RUNDIR/.ramses.complete"
fi

echo "[2/2] verify output sequence and exact initial/final particle-ID equality"
(
    TIMEFORMAT=$'real_seconds=%R\nuser_seconds=%U\nsystem_seconds=%S'
    time python3 "$RUNDIR/verify_compact_level14_z0.py" "$RUNDIR" \
        --ranks 64 --outputs 6
) 2> "$RUNDIR/verify_z0.time" | tee "$RUNDIR/verify_z0.log"
{
    echo "completed_at=$(date --iso-8601=seconds)"
    du -sb "$RUNDIR"
} >> "$RUNDIR/provenance.txt"
touch "$RUNDIR/.complete"
status=complete
echo "V4 COMPACT RANK-726 LEVEL-14 Z=0 DMO EVOLUTION PASSED"
