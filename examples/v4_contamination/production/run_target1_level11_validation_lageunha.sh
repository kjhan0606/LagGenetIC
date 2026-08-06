#!/bin/bash
set -eo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
SOURCE_ROOT=$(git -C "$HERE" rev-parse --show-toplevel)
V4_ROOT=${V4_WORKROOT:-/gpfs/kjhan/VoidSim/v4_parent_n512}
PRIOR_GATE="$V4_ROOT/target1_zoom_validation/.complete"
TARGET_ROOT="$V4_ROOT/dmo_z0/parent_targets"
RUNDIR="$V4_ROOT/target1_level11_validation"
GENETIC_BIN=${GENETIC_BIN:-$SOURCE_ROOT/genetIC/genetIC/genetIC}
RAMSES_SOURCE="$V4_ROOT/dmo_z0/ramses_final3d"
EXPECTED_RAMSES_SHA256=ce3e5a14c22639fd14e3c70092694eeb2b6fb3a009aefafdd7e20fbf585a592e

if [ "$(hostname | tr '[:upper:]' '[:lower:]')" != "lageunha" ]; then
    echo "this launcher must run on LagEunha, not $(hostname)" >&2
    exit 2
fi
if [ ! -e "$PRIOR_GATE" ]; then
    echo "required level-10 validation gate is absent: $PRIOR_GATE" >&2
    exit 2
fi
if [ -e "$RUNDIR" ]; then
    echo "$RUNDIR already exists; refusing to overwrite it" >&2
    exit 2
fi
for executable in "$GENETIC_BIN" "$RAMSES_SOURCE"; do
    if [ ! -x "$executable" ]; then
        echo "executable is absent: $executable" >&2
        exit 2
    fi
done
actual_ramses_sha256=$(sha256sum "$RAMSES_SOURCE" | awk '{print $1}')
if [ "$actual_ramses_sha256" != "$EXPECTED_RAMSES_SHA256" ]; then
    echo "validated RAMSES binary checksum mismatch: $actual_ramses_sha256" >&2
    exit 2
fi
mapfile -t target_files < <(find "$TARGET_ROOT" -maxdepth 1 -type f \
    -name 'target_01_halo_*.id' | sort)
if [ "${#target_files[@]}" -ne 1 ]; then
    echo "expected exactly one target-1 ID file, found ${#target_files[@]}" >&2
    exit 2
fi
target_id=${target_files[0]}

mkdir -p "$RUNDIR/normal" "$RUNDIR/inverted" "$RUNDIR/ramses"
install -m 0644 "$target_id" "$RUNDIR/target1.id"
install -m 0644 "$HERE/genetic_target1_level11_normal.txt" \
    "$RUNDIR/normal/genetic.txt"
install -m 0644 "$HERE/genetic_target1_level11_inverted.txt" \
    "$RUNDIR/inverted/genetic.txt"
install -m 0644 "$HERE/ramses_target1_level11_ingest.nml" \
    "$RUNDIR/ramses/ramses.nml"
install -m 0755 "$HERE/verify_target1_zoom.py" "$RUNDIR/verify_target1_zoom.py"
install -m 0755 "$RAMSES_SOURCE" "$RUNDIR/ramses/ramses_final3d"
for mode in normal inverted; do
    ln -s "$V4_ROOT/camb/camb_transfer_z49.dat" \
        "$RUNDIR/$mode/camb_transfer_z49.dat"
    ln -s "$V4_ROOT/genetic/wn_level0.npy" "$RUNDIR/$mode/wn_level0.npy"
    ln -s ../target1.id "$RUNDIR/$mode/target1.id"
done

{
    echo "started_at=$(date --iso-8601=seconds)"
    echo "host=$(hostname)"
    echo "laggenetic_head=$(git -C "$SOURCE_ROOT" rev-parse HEAD)"
    echo "ramses_binary_sha256=$actual_ramses_sha256"
    sha256sum "$GENETIC_BIN" "$target_id" \
        "$RUNDIR/normal/genetic.txt" "$RUNDIR/inverted/genetic.txt" \
        "$RUNDIR/ramses/ramses.nml"
} > "$RUNDIR/provenance.txt"

source /opt/ohpc/pub/intel/oneapi/setvars.sh >/dev/null
set -u
export OMP_PROC_BIND=close
export OMP_PLACES=cores
ulimit -s unlimited

echo "[1/4] generate the matched normal and inverted level-11 ICs"
for mode in normal inverted; do
    (cd "$RUNDIR/$mode" && {
        TIMEFORMAT=$'real_seconds=%R\nuser_seconds=%U\nsystem_seconds=%S'
        time taskset -c 16-31 env OMP_NUM_THREADS=16 \
            "$GENETIC_BIN" genetic.txt > genetic.log 2>&1
    } 2> genetic.time)
done

echo "[2/4] verify the multilevel sign pair and target-1 mask"
{
    TIMEFORMAT=$'real_seconds=%R\nuser_seconds=%U\nsystem_seconds=%S'
    time python3 "$RUNDIR/verify_target1_zoom.py" \
        "$RUNDIR/normal" "$RUNDIR/inverted" "$RUNDIR/target1.id"
} 2> "$RUNDIR/verify_ic.time" | tee "$RUNDIR/verify_ic.log"

echo "[3/4] ingest three GRAFIC levels with lagRamses on 16 MPI ranks"
export OMP_NUM_THREADS=1
export I_MPI_PIN=1
export I_MPI_PIN_DOMAIN=core
export I_MPI_PIN_PROCESSOR_LIST=16-31
export I_MPI_FABRICS=shm
(cd "$RUNDIR/ramses" && {
    TIMEFORMAT=$'real_seconds=%R\nuser_seconds=%U\nsystem_seconds=%S'
    time mpirun -np 16 ./ramses_final3d ramses.nml > ramses.log 2>&1
} 2> ramses.time)

echo "[4/4] verify every refined mesh and globally unique particle IDs"
{
    TIMEFORMAT=$'real_seconds=%R\nuser_seconds=%U\nsystem_seconds=%S'
    time python3 "$RUNDIR/verify_target1_zoom.py" \
        "$RUNDIR/normal" "$RUNDIR/inverted" "$RUNDIR/target1.id" \
        --ramses-case "$RUNDIR/ramses" --ranks 16
} 2> "$RUNDIR/verify_zoom.time" | tee "$RUNDIR/verify_zoom.log"
{
    echo "completed_at=$(date --iso-8601=seconds)"
    du -sb "$RUNDIR"
} >> "$RUNDIR/provenance.txt"
touch "$RUNDIR/.complete"
echo "V4 TARGET-1 LEVEL-11 SCALE HAND-OFF PASSED"
