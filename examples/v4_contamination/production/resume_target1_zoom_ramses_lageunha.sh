#!/bin/bash
set -eo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
V4_ROOT=${V4_WORKROOT:-/gpfs/kjhan/VoidSim/v4_parent_n512}
RUNDIR="$V4_ROOT/target1_zoom_validation"
EXPECTED_RAMSES_SHA256=ce3e5a14c22639fd14e3c70092694eeb2b6fb3a009aefafdd7e20fbf585a592e

if [ "$(hostname | tr '[:upper:]' '[:lower:]')" != "lageunha" ]; then
    echo "this launcher must run on LagEunha, not $(hostname)" >&2
    exit 2
fi
if [ -e "$RUNDIR/.complete" ]; then
    echo "$RUNDIR is already complete; refusing to rerun it" >&2
    exit 2
fi
if ! grep -q "V4 TARGET-1 ONE-LEVEL ZOOM PASSED" "$RUNDIR/verify_ic.log"; then
    echo "the exact normal/inverted IC gate has not passed" >&2
    exit 2
fi
if ! grep -q "Increase ngridmax" "$RUNDIR/ramses/ramses.log"; then
    echo "the prior RAMSES run did not fail at the expected ngridmax gate" >&2
    exit 2
fi
if find "$RUNDIR/ramses" -maxdepth 1 -type d -name 'output_[0-9]*' | grep -q .; then
    echo "a RAMSES output already exists; refusing an ambiguous resume" >&2
    exit 2
fi
actual_ramses_sha256=$(sha256sum "$RUNDIR/ramses/ramses_final3d" | awk '{print $1}')
if [ "$actual_ramses_sha256" != "$EXPECTED_RAMSES_SHA256" ]; then
    echo "validated RAMSES binary checksum mismatch: $actual_ramses_sha256" >&2
    exit 2
fi
failed_log="$RUNDIR/ramses/ramses.ngridmax1000000.failed.log"
if [ -e "$failed_log" ]; then
    echo "$failed_log already exists; refusing to replace it" >&2
    exit 2
fi
mv "$RUNDIR/ramses/ramses.log" "$failed_log"
install -m 0644 "$HERE/ramses_target1_zoom_ingest.nml" \
    "$RUNDIR/ramses/ramses.nml"
install -m 0755 "$HERE/verify_target1_zoom.py" "$RUNDIR/verify_target1_zoom.py"
{
    echo "resumed_at=$(date --iso-8601=seconds)"
    echo "resume_laggenetic_head=$(git -C /home/kjhan/BACKUP/VoidSim/code/LagGenetIC rev-parse HEAD)"
    sha256sum "$RUNDIR/ramses/ramses.nml" "$failed_log"
} >> "$RUNDIR/provenance.txt"

source /opt/ohpc/pub/intel/oneapi/setvars.sh >/dev/null
set -u
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export I_MPI_PIN=1
export I_MPI_PIN_DOMAIN=core
export I_MPI_PIN_PROCESSOR_LIST=16-31
export I_MPI_FABRICS=shm
ulimit -s unlimited

echo "[resume 1/2] ingest the verified GRAFIC pair with ngridmax=1600000"
(cd "$RUNDIR/ramses" && mpirun -np 16 ./ramses_final3d ramses.nml \
    > ramses.log 2>&1)

echo "[resume 2/2] verify the level-10 mesh and globally unique particle IDs"
python3 "$RUNDIR/verify_target1_zoom.py" \
    "$RUNDIR/normal" "$RUNDIR/inverted" "$RUNDIR/target1.id" \
    --ramses-case "$RUNDIR/ramses" --ranks 16 \
    | tee "$RUNDIR/verify_zoom.log"
touch "$RUNDIR/.complete"
echo "V4 TARGET-1 ONE-LEVEL ZOOM HAND-OFF PASSED"
