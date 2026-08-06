#!/bin/bash
set -eo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
V4_ROOT=${V4_WORKROOT:-/gpfs/kjhan/VoidSim/v4_parent_n512}
ORIGINAL_DMO="$V4_ROOT/dmo_z0"
INVERTED_PARENT="$V4_ROOT/inverted_parent_z0"
TARGET_ROOT="$ORIGINAL_DMO/parent_targets"
RUNDIR="$V4_ROOT/target1_zoom_validation"
GENETIC_BIN=${GENETIC_BIN:-/home/kjhan/BACKUP/VoidSim/code/LagGenetIC/genetIC/genetIC/genetIC}
RAMSES_SOURCE="$ORIGINAL_DMO/ramses_final3d"
EXPECTED_RAMSES_SHA256=ce3e5a14c22639fd14e3c70092694eeb2b6fb3a009aefafdd7e20fbf585a592e

if [ "$(hostname | tr '[:upper:]' '[:lower:]')" != "lageunha" ]; then
    echo "this launcher must run on LagEunha, not $(hostname)" >&2
    exit 2
fi
for gate in "$ORIGINAL_DMO/hop_catalog/.complete" "$INVERTED_PARENT/.complete"; do
    if [ ! -e "$gate" ]; then
        echo "required validation gate is absent: $gate" >&2
        exit 2
    fi
done
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
python3 - "$TARGET_ROOT/parent_target_candidates.json" <<'PY'
import json
import sys

catalogue = json.load(open(sys.argv[1]))
target = next(item for item in catalogue["candidates"] if item.get("target_rank") == 1)
span = max(target["lagrangian_width"]) * 512.0
if span + 8.0 > 64.0:
    raise SystemExit(
        f"target-1 span {span} Mpc/h plus buffer does not fit the 64 Mpc/h fine grid"
    )
print(f"target-1 maximum Lagrangian span: {span:.3f} Mpc/h")
PY

mkdir -p "$RUNDIR/normal" "$RUNDIR/inverted" "$RUNDIR/ramses"
install -m 0644 "$target_id" "$RUNDIR/target1.id"
install -m 0644 "$HERE/genetic_target1_zoom_normal.txt" \
    "$RUNDIR/normal/genetic.txt"
install -m 0644 "$HERE/genetic_target1_zoom_inverted.txt" \
    "$RUNDIR/inverted/genetic.txt"
install -m 0644 "$HERE/ramses_target1_zoom_ingest.nml" \
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
    echo "laggenetic_head=$(git -C /home/kjhan/BACKUP/VoidSim/code/LagGenetIC rev-parse HEAD)"
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

echo "[1/4] generate the matched normal and inverted one-level zoom ICs"
for mode in normal inverted; do
    (cd "$RUNDIR/$mode" && taskset -c 16-31 env OMP_NUM_THREADS=16 \
        "$GENETIC_BIN" genetic.txt > genetic.log 2>&1)
done

echo "[2/4] verify exact sign reversal and the target-1 refinement mask"
python3 "$RUNDIR/verify_target1_zoom.py" \
    "$RUNDIR/normal" "$RUNDIR/inverted" "$RUNDIR/target1.id" \
    | tee "$RUNDIR/verify_ic.log"

echo "[3/4] ingest both GRAFIC levels with lagRamses on 16 MPI ranks"
export OMP_NUM_THREADS=1
export I_MPI_PIN=1
export I_MPI_PIN_DOMAIN=core
export I_MPI_PIN_PROCESSOR_LIST=16-31
export I_MPI_FABRICS=shm
(cd "$RUNDIR/ramses" && mpirun -np 16 ./ramses_final3d ramses.nml \
    > ramses.log 2>&1)

echo "[4/4] verify the level-10 mesh and globally unique particle IDs"
python3 "$RUNDIR/verify_target1_zoom.py" \
    "$RUNDIR/normal" "$RUNDIR/inverted" "$RUNDIR/target1.id" \
    --ramses-case "$RUNDIR/ramses" --ranks 16 \
    | tee "$RUNDIR/verify_zoom.log"
touch "$RUNDIR/.complete"
echo "V4 TARGET-1 ONE-LEVEL ZOOM HAND-OFF PASSED"
