#!/bin/bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
WORKROOT=${V4_WORKROOT:-/gpfs/kjhan/VoidSim/v4_parent_n512}
WORKROOT=$(realpath -m "$WORKROOT")
RUNDIR="$WORKROOT/dmo_z0"
RAMSES_BIN=${RAMSES_BIN:-/home/kjhan/BACKUP/lagRamses/bin/ramses_final3d}
EXPECTED_BINARY_SHA256=ce3e5a14c22639fd14e3c70092694eeb2b6fb3a009aefafdd7e20fbf585a592e

case "$WORKROOT" in
    /gpfs/kjhan/*) ;;
    *) echo "V4_WORKROOT must be below /gpfs/kjhan" >&2; exit 2 ;;
esac
if [ ! -x "$RAMSES_BIN" ]; then
    echo "RAMSES executable not found: $RAMSES_BIN" >&2
    exit 2
fi
if [ ! -s "$WORKROOT/genetic/v4_parent.grafic_512/ic_particle_ids" ]; then
    echo "validated level-9 GRAFIC input is absent under $WORKROOT/genetic" >&2
    exit 2
fi
if squeue -h -u "${USER:?}" -n void_dmo512 | grep -q .; then
    echo "an active void_dmo512 job already exists" >&2
    exit 2
fi
if [ -d "$RUNDIR" ] && find "$RUNDIR" -maxdepth 1 \
        \( -name 'output_[0-9]*' -o -name 'ramses.log' \) -print -quit | grep -q .; then
    echo "$RUNDIR already contains simulation output; refusing to overwrite it" >&2
    exit 2
fi

actual_sha256=$(sha256sum "$RAMSES_BIN" | awk '{print $1}')
if [ "$actual_sha256" != "$EXPECTED_BINARY_SHA256" ]; then
    echo "RAMSES binary changed: $actual_sha256" >&2
    exit 2
fi

mkdir -p "$RUNDIR"
install -m 0644 "$HERE/ramses_dmo_z0.nml" "$RUNDIR/ramses_dmo_z0.nml"
install -m 0644 "$HERE/submit_dmo_z0.slurm" "$RUNDIR/submit_dmo_z0.slurm"
install -m 0755 "$HERE/verify_dmo_z0.py" "$RUNDIR/verify_dmo_z0.py"
install -m 0644 "$HERE/../../v2_hop_id_file/hop_to_genetic_id.py" \
    "$RUNDIR/hop_to_genetic_id.py"
install -m 0755 "$RAMSES_BIN" "$RUNDIR/ramses_final3d"

{
    echo "submitted_at=$(date --iso-8601=seconds)"
    echo "host=$(hostname)"
    echo "source_binary=$RAMSES_BIN"
    echo "binary_sha256=$actual_sha256"
    echo "lagramses_head=$(git -C /home/kjhan/BACKUP/lagRamses rev-parse HEAD)"
    echo "laggenetic_head=$(git -C "$HERE/../../.." rev-parse HEAD)"
} > "$RUNDIR/provenance.txt"

job_id=$(cd "$RUNDIR" && sbatch --parsable submit_dmo_z0.slurm)
echo "submitted VoidSim V4 DMO job $job_id"
echo "run directory: $RUNDIR"
