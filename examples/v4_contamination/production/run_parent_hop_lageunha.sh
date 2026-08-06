#!/bin/bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
RUNDIR=${V4_DMO_RUNDIR:-/gpfs/kjhan/VoidSim/v4_parent_n512/dmo_z0}
FINAL_OUTPUT="$RUNDIR/output_00006"
HOP_BUILD="$RUNDIR/hop_build"
HOP_DIR="$RUNDIR/hop_catalog"
HOP_ROOT="$HOP_DIR/parent_hop"
GROUP_ROOT="$HOP_DIR/parent_groups"
TARGET_DIR="$RUNDIR/parent_targets"

if [ "$(hostname | tr '[:upper:]' '[:lower:]')" != "lageunha" ]; then
    echo "this launcher must run on LagEunha, not $(hostname)" >&2
    exit 2
fi
if ! grep -q "V4 Z=0 DMO PARENT PASSED" "$RUNDIR/verify_dmo_z0.log"; then
    echo "the V4 z=0 DMO verification gate has not passed" >&2
    exit 2
fi
if [ ! -d "$FINAL_OUTPUT" ]; then
    echo "final output is absent: $FINAL_OUTPUT" >&2
    exit 2
fi
for executable in "$HOP_BUILD/hop" "$HOP_BUILD/regroup" "$HOP_BUILD/poshalo"; do
    if [ ! -x "$executable" ]; then
        echo "HOP executable is absent: $executable" >&2
        exit 2
    fi
done
if [ -e "$HOP_DIR/.complete" ] || [ -e "$HOP_ROOT.tag" ] || [ -e "$GROUP_ROOT.tag" ]; then
    echo "HOP catalogue output already exists; refusing to overwrite it" >&2
    exit 2
fi
if [ -e "$TARGET_DIR" ]; then
    echo "target output already exists; refusing to overwrite it" >&2
    exit 2
fi

mkdir -p "$HOP_DIR"
echo "$$" > "$HOP_DIR/launcher.pid"
cleanup() {
    rm -f "$HOP_DIR/launcher.pid"
}
trap cleanup EXIT
ulimit -s unlimited

final_number=${FINAL_OUTPUT##*_}
{
    echo "started_at=$(date --iso-8601=seconds)"
    echo "host=$(hostname)"
    echo "final_output=$FINAL_OUTPUT"
    echo "lagramses_head=$(git -C /home/kjhan/BACKUP/lagRamses rev-parse HEAD)"
    echo "laggenetic_head=$(git -C /home/kjhan/BACKUP/VoidSim/code/LagGenetIC rev-parse HEAD)"
    sha256sum "$HOP_BUILD/hop" "$HOP_BUILD/regroup" "$HOP_BUILD/poshalo"
    sha256sum "$HERE/select_parent_targets.py"
} > "$HOP_DIR/provenance.txt"

"$HOP_BUILD/hop" \
    -in "$FINAL_OUTPUT/part_${final_number}.out" -p 1. -o "$HOP_ROOT" \
    > "$HOP_DIR/hop.log" 2>&1
"$HOP_BUILD/regroup" \
    -root "$HOP_ROOT" -douter 80. -dsaddle 200. -dpeak 240. \
    -f77 -o "$GROUP_ROOT" > "$HOP_DIR/regroup.log" 2>&1
"$HOP_BUILD/poshalo" \
    -inp "$FINAL_OUTPUT" -pre "$GROUP_ROOT" > "$HOP_DIR/poshalo.log" 2>&1

python3 "$HERE/select_parent_targets.py" \
    "$FINAL_OUTPUT" "$GROUP_ROOT.tag" "$GROUP_ROOT.pos" "$TARGET_DIR" \
    --grid-size 512 --candidate-count 32 --target-count 3 \
    --min-particles 3000 --max-lagrangian-width 0.5 \
    --edge-buffer-cells 2 --min-centre-separation 0.1 \
    > "$HOP_DIR/select_targets.log" 2>&1
touch "$HOP_DIR/.complete"
echo "V4 parent HOP catalogue and target selection passed"
