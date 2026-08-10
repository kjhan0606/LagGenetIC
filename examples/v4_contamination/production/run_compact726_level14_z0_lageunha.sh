#!/bin/bash
set -eo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
SOURCE_ROOT=$(git -C "$HERE" rev-parse --show-toplevel)
V4_ROOT=${V4_WORKROOT:-/gpfs/kjhan/VoidSim/v4_parent_n512}
V4_ROOT=$(realpath -m "$V4_ROOT")
HANDOFF="$V4_ROOT/compact726_level14_pilot"
GRAFIC_SOURCE="$HANDOFF/genetic"
DEFAULT_RUNDIR="$V4_ROOT/compact726_level14_z0"
RUNDIR=${V4_Z0_RUNDIR:-$DEFAULT_RUNDIR}
RUNDIR=$(realpath -m "$RUNDIR")
NGRIDMAX=${V4_Z0_NGRIDMAX:-3000000}
RAMSES_SOURCE="$HANDOFF/ramses/ramses_final3d"
EXPECTED_RAMSES_SHA256=ce3e5a14c22639fd14e3c70092694eeb2b6fb3a009aefafdd7e20fbf585a592e

case "$V4_ROOT" in
    /gpfs/kjhan/*) ;;
    *) echo "V4_WORKROOT must be below /gpfs/kjhan" >&2; exit 2 ;;
esac
case "$RUNDIR" in
    "$V4_ROOT"/compact726_level14_z0*) ;;
    *) echo "V4_Z0_RUNDIR must be a compact726_level14_z0 run below V4_WORKROOT" >&2; exit 2 ;;
esac
case "$NGRIDMAX" in
    ''|*[!0-9]*) echo "V4_Z0_NGRIDMAX must be an integer" >&2; exit 2 ;;
esac
if [ "$NGRIDMAX" -lt 3000000 ]; then
    echo "V4_Z0_NGRIDMAX must not be smaller than the verified 3000000-grid pool" >&2
    exit 2
fi
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
for level in 512 1024 2048 4096 8192 16384; do
    grafic="$GRAFIC_SOURCE/v4_compact726_inverted.grafic_$level"
    if [ ! -s "$grafic/ic_particle_ids" ]; then
        echo "verified GRAFIC particle IDs are absent: $grafic/ic_particle_ids" >&2
        exit 2
    fi
done
if [ ! -x "$RAMSES_SOURCE" ]; then
    echo "validated RAMSES executable is absent: $RAMSES_SOURCE" >&2
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

if [ -e "$RUNDIR/.complete" ]; then
    echo "compact rank-726 level-14 z=0 DMO evolution is already complete"
    exit 0
fi
if [ -e "$RUNDIR" ] && find "$RUNDIR" -maxdepth 1 \
        \( -name 'output_[0-9]*' -o -name 'ramses.log' -o -name '.failed' \) \
        -print -quit | grep -q .; then
    echo "$RUNDIR contains an earlier run; archive it before a fresh launch" >&2
    exit 2
fi

mkdir -p "$RUNDIR"
install -m 0644 "$HERE/ramses_compact726_level14_z0.nml" \
    "$RUNDIR/ramses.nml"
sed -i "s/^ngridmax=.*/ngridmax=$NGRIDMAX/" "$RUNDIR/ramses.nml"
if ! grep -qx "ngridmax=$NGRIDMAX" "$RUNDIR/ramses.nml"; then
    echo "failed to stage the requested ngridmax=$NGRIDMAX" >&2
    exit 2
fi
install -m 0755 "$HERE/verify_compact_level14_z0.py" \
    "$RUNDIR/verify_compact_level14_z0.py"
install -m 0644 "$SOURCE_ROOT/examples/v2_hop_id_file/hop_to_genetic_id.py" \
    "$RUNDIR/hop_to_genetic_id.py"
install -m 0755 "$RAMSES_SOURCE" "$RUNDIR/ramses_final3d"
if [ -L "$RUNDIR/genetic" ]; then
    if [ "$(readlink -f "$RUNDIR/genetic")" != "$(readlink -f "$GRAFIC_SOURCE")" ]; then
        echo "staged GRAFIC link does not select the verified hierarchy" >&2
        exit 2
    fi
elif [ -e "$RUNDIR/genetic" ]; then
    echo "$RUNDIR/genetic exists but is not a symbolic link" >&2
    exit 2
else
    ln -s "$GRAFIC_SOURCE" "$RUNDIR/genetic"
fi
{
    echo "started_at=$(date --iso-8601=seconds)"
    echo "host=$(hostname)"
    echo "target=compact_rank_726"
    echo "initialization=fresh_verified_grafic"
    echo "grafic_source=$GRAFIC_SOURCE"
    echo "handoff_verification=$HANDOFF/verify_ramses.log"
    echo "laggenetic_head=$(git -C "$SOURCE_ROOT" rev-parse HEAD)"
    echo "ramses_binary_sha256=$actual_ramses_sha256"
    echo "ranks=64"
    echo "openmp_threads=1"
    echo "ngridmax=$NGRIDMAX"
    echo "void_refine=false"
    sha256sum "$HERE/run_compact726_level14_z0_lageunha.sh"
    sha256sum "$RUNDIR/ramses.nml" "$RUNDIR/ramses_final3d" \
        "$RUNDIR/verify_compact_level14_z0.py" \
        "$RUNDIR/hop_to_genetic_id.py"
} > "$RUNDIR/provenance.txt"

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

echo "[1/2] evolve compact rank 726 continuously from verified GRAFIC ICs to a=1"
(
    TIMEFORMAT=$'real_seconds=%R\nuser_seconds=%U\nsystem_seconds=%S'
    cd "$RUNDIR"
    time mpirun -np 64 ./ramses_final3d ramses.nml > ramses.log 2>&1
) 2> "$RUNDIR/ramses.time"
if ! grep -q "Run completed" "$RUNDIR/ramses.log" || \
    [ ! -f "$RUNDIR/output_00006/COMPLETE" ]; then
    echo "RAMSES returned without a complete a=1 snapshot" >&2
    exit 2
fi
touch "$RUNDIR/.ramses.complete"

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
echo "V4 COMPACT RANK-726 LEVEL-14 Z=0 DMO EVOLUTION PASSED (ngridmax=$NGRIDMAX)"
