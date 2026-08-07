#!/bin/bash
set -eo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
SOURCE_ROOT=$(git -C "$HERE" rev-parse --show-toplevel)
V4_ROOT=${V4_WORKROOT:-/gpfs/kjhan/VoidSim/v4_parent_n512}
RUNDIR="$V4_ROOT/compact729_level14_pilot"
CANDIDATE_ROOT="$V4_ROOT/dmo_z0/candidate_void_comparison"
TARGET_ID="$CANDIDATE_ROOT/comparison_06_compact_rank_000729_halo_000729.id"
GENETIC_BIN=${GENETIC_BIN:-$SOURCE_ROOT/genetIC/genetIC/genetIC}
RAMSES_SOURCE="$V4_ROOT/dmo_z0/ramses_final3d"
CAMB_ROOT=${CAMB_ROOT:-/home/kjhan/BACKUP/lagCAMB_validation/lagCAMB}
CAMB_BIN=${CAMB_BIN:-/home/kjhan/BACKUP/lagCAMB_validation/venv/bin/camb}
EXPECTED_RAMSES_SHA256=ce3e5a14c22639fd14e3c70092694eeb2b6fb3a009aefafdd7e20fbf585a592e

if [ "$(hostname | tr '[:upper:]' '[:lower:]')" != "lageunha" ]; then
    echo "this launcher must run on LagEunha, not $(hostname)" >&2
    exit 2
fi
for gate in \
    "$V4_ROOT/target1_level11_validation/.complete" \
    "$V4_ROOT/inverted_parent_z0/hop_catalog/.complete" \
    "$V4_ROOT/inverted_parent_z0/watershed_grid256/watershed_properties.json" \
    "$V4_ROOT/inverted_parent_z0/halo_yield_grid256/watershed_halo_counts.json"; do
    if [ ! -e "$gate" ]; then
        echo "required compact-pilot gate is absent: $gate" >&2
        exit 2
    fi
done
for executable in "$GENETIC_BIN" "$RAMSES_SOURCE" "$CAMB_BIN"; do
    if [ ! -x "$executable" ]; then
        echo "executable is absent: $executable" >&2
        exit 2
    fi
done
if [ ! -f "$CAMB_ROOT/inifiles/planck_2018.ini" ]; then
    echo "CAMB Planck base ini is absent below $CAMB_ROOT" >&2
    exit 2
fi
if [ ! -s "$TARGET_ID" ] || [ "$(wc -l < "$TARGET_ID")" -ne 3490 ]; then
    echo "rank-729 target file is absent or does not contain 3490 IDs" >&2
    exit 2
fi
actual_ramses_sha256=$(sha256sum "$RAMSES_SOURCE" | awk '{print $1}')
if [ "$actual_ramses_sha256" != "$EXPECTED_RAMSES_SHA256" ]; then
    echo "validated RAMSES binary checksum mismatch: $actual_ramses_sha256" >&2
    exit 2
fi
available_kib=$(df -Pk "$V4_ROOT" | awk 'NR==2 {print $4}')
required_kib=$((150 * 1024 * 1024))
if [ "$available_kib" -lt "$required_kib" ]; then
    echo "compact pilot requires at least 150 GiB free below $V4_ROOT" >&2
    exit 2
fi

if [ ! -e "$RUNDIR" ]; then
    mkdir -p "$RUNDIR/camb" "$RUNDIR/genetic" "$RUNDIR/ramses"
    install -m 0644 "$TARGET_ID" "$RUNDIR/compact729.id"
    install -m 0644 "$HERE/genetic_compact729_level14_inverted.txt" \
        "$RUNDIR/genetic/genetic.txt"
    install -m 0644 "$HERE/ramses_compact729_level14_pilot.nml" \
        "$RUNDIR/ramses/ramses.nml"
    install -m 0644 "$CAMB_ROOT/inifiles/planck_2018.ini" \
        "$RUNDIR/camb/planck_2018_base.ini"
    install -m 0644 "$HERE/lagcamb_z49_level14.ini" \
        "$RUNDIR/camb/lagcamb_z49_level14.ini"
    install -m 0755 "$HERE/prepare_transfer.py" \
        "$RUNDIR/camb/prepare_transfer.py"
    install -m 0755 "$HERE/verify_compact_level14.py" \
        "$RUNDIR/verify_compact_level14.py"
    install -m 0644 "$HERE/verify_target1_zoom.py" \
        "$RUNDIR/verify_target1_zoom.py"
    install -m 0755 "$RAMSES_SOURCE" "$RUNDIR/ramses/ramses_final3d"
    ln -s ../camb/camb_transfer_z49_level14.dat \
        "$RUNDIR/genetic/camb_transfer_z49.dat"
    ln -s "$V4_ROOT/genetic/wn_level0.npy" "$RUNDIR/genetic/wn_level0.npy"
    ln -s ../compact729.id "$RUNDIR/genetic/compact729.id"
    {
        echo "started_at=$(date --iso-8601=seconds)"
        echo "host=$(hostname)"
        echo "target=compact_rank_729"
        echo "target_parent_particles=3490"
        echo "dense_grafic_cells=1361313792"
        echo "laggenetic_head=$(git -C "$SOURCE_ROOT" rev-parse HEAD)"
        echo "ramses_binary_sha256=$actual_ramses_sha256"
        sha256sum "$GENETIC_BIN" "$TARGET_ID" \
            "$RUNDIR/camb/lagcamb_z49_level14.ini" \
            "$RUNDIR/genetic/genetic.txt" "$RUNDIR/ramses/ramses.nml"
    } > "$RUNDIR/provenance.txt"
elif [ -e "$RUNDIR/.complete" ]; then
    echo "compact rank-729 level-14 pilot is already complete"
    exit 0
elif [ -e "$RUNDIR/.failed" ]; then
    mv "$RUNDIR/.failed" \
        "$RUNDIR/.failed_before_resume_$(date +%Y%m%dT%H%M%S)"
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
export OMP_PROC_BIND=close
export OMP_PLACES=cores
ulimit -s unlimited

if [ ! -e "$RUNDIR/.camb.complete" ]; then
    if [ -e "$RUNDIR/camb/lagcamb_transfer_z49_level14_raw.dat" ] || \
        [ -e "$RUNDIR/camb/camb_transfer_z49_level14.dat" ]; then
        echo "partial level-14 CAMB output exists; refusing to overwrite it" >&2
        exit 2
    fi
    echo "[1/5] build a transfer table beyond the level-14 grid wave number"
    (
        TIMEFORMAT=$'real_seconds=%R\nuser_seconds=%U\nsystem_seconds=%S'
        cd "$RUNDIR/camb"
        time "$CAMB_BIN" lagcamb_z49_level14.ini > camb.log 2>&1
    ) 2> "$RUNDIR/camb/camb.time"
    python3 "$RUNDIR/camb/prepare_transfer.py" \
        "$RUNDIR/camb/lagcamb_transfer_z49_level14_raw.dat" \
        "$RUNDIR/camb/camb_transfer_z49_level14.dat" \
        --minimum-kmax 220 > "$RUNDIR/camb/prepare_transfer.log" 2>&1
    touch "$RUNDIR/.camb.complete"
fi

if [ ! -e "$RUNDIR/.genetic.complete" ]; then
    if find "$RUNDIR/genetic" -maxdepth 1 -type d \
        -name 'v4_compact729_inverted.grafic_*' | grep -q .; then
        echo "partial compact GRAFIC output exists; refusing to overwrite it" >&2
        exit 2
    fi
    echo "[2/5] generate the inverted compact rank-729 level-14 IC"
    (
        TIMEFORMAT=$'real_seconds=%R\nuser_seconds=%U\nsystem_seconds=%S'
        cd "$RUNDIR/genetic"
        time taskset -c 16-31 env OMP_NUM_THREADS=16 \
            "$GENETIC_BIN" genetic.txt > genetic.log 2>&1
    ) 2> "$RUNDIR/genetic/genetic.time"
    touch "$RUNDIR/.genetic.complete"
fi

if [ ! -e "$RUNDIR/.ic_verify.complete" ]; then
    echo "[3/5] verify all six GRAFIC levels and the rank-729 mask"
    (
        TIMEFORMAT=$'real_seconds=%R\nuser_seconds=%U\nsystem_seconds=%S'
        time python3 "$RUNDIR/verify_compact_level14.py" \
            "$RUNDIR/genetic" "$RUNDIR/compact729.id"
    ) 2> "$RUNDIR/verify_ic.time" | tee "$RUNDIR/verify_ic.log"
    touch "$RUNDIR/.ic_verify.complete"
fi

if [ ! -e "$RUNDIR/.ramses.complete" ]; then
    if find "$RUNDIR/ramses" -maxdepth 1 -type d \
        -name 'output_[0-9][0-9][0-9][0-9][0-9]' | grep -q .; then
        echo "partial compact RAMSES output exists; refusing to overwrite it" >&2
        exit 2
    fi
    echo "[4/5] ingest the level-9 through level-14 hierarchy on 64 MPI ranks"
    export OMP_NUM_THREADS=1
    export I_MPI_PIN=1
    export I_MPI_PIN_DOMAIN=core
    export I_MPI_PIN_PROCESSOR_LIST=0-63
    export I_MPI_FABRICS=shm
    (
        TIMEFORMAT=$'real_seconds=%R\nuser_seconds=%U\nsystem_seconds=%S'
        cd "$RUNDIR/ramses"
        time mpirun -np 64 ./ramses_final3d ramses.nml > ramses.log 2>&1
    ) 2> "$RUNDIR/ramses/ramses.time"
    touch "$RUNDIR/.ramses.complete"
fi

echo "[5/5] verify the AMR hierarchy and global particle IDs"
(
    TIMEFORMAT=$'real_seconds=%R\nuser_seconds=%U\nsystem_seconds=%S'
    time python3 "$RUNDIR/verify_compact_level14.py" \
        "$RUNDIR/genetic" "$RUNDIR/compact729.id" \
        --ramses-case "$RUNDIR/ramses" --ranks 64
) 2> "$RUNDIR/verify_ramses.time" | tee "$RUNDIR/verify_ramses.log"
{
    echo "completed_at=$(date --iso-8601=seconds)"
    du -sb "$RUNDIR"
} >> "$RUNDIR/provenance.txt"
touch "$RUNDIR/.complete"
status=complete
echo "V4 COMPACT RANK-729 LEVEL-14 DMO PILOT PASSED"
