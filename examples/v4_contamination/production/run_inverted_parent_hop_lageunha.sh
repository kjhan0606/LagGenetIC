#!/bin/bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${V4_ROOT:-/gpfs/kjhan/VoidSim/v4_parent_n512}
FINAL_OUTPUT="$ROOT/inverted_parent_z0/evolution/output_00006"
HOP_BUILD="$ROOT/dmo_z0/hop_build"
HOP_DIR="$ROOT/inverted_parent_z0/hop_catalog"
HOP_ROOT="$HOP_DIR/inverted_hop"
GROUP_ROOT="$HOP_DIR/inverted_groups"

if [ "$(hostname | tr '[:upper:]' '[:lower:]')" != "lageunha" ]; then
    echo "this launcher must run on LagEunha, not $(hostname)" >&2
    exit 2
fi
if ! grep -q "V4 Z=0 DMO PARENT PASSED" \
    "$ROOT/inverted_parent_z0/evolution/verify_dmo_z0.log"; then
    echo "the inverted-parent verification gate has not passed" >&2
    exit 2
fi
if [ ! -d "$FINAL_OUTPUT" ]; then
    echo "final inverted output is absent: $FINAL_OUTPUT" >&2
    exit 2
fi
for executable in "$HOP_BUILD/hop" "$HOP_BUILD/regroup" "$HOP_BUILD/poshalo"; do
    if [ ! -x "$executable" ]; then
        echo "HOP executable is absent: $executable" >&2
        exit 2
    fi
done
if [ -e "$HOP_DIR" ]; then
    echo "inverted HOP output already exists; refusing to overwrite it" >&2
    exit 2
fi

mkdir -p "$HOP_DIR"
echo "$$" > "$HOP_DIR/launcher.pid"
status=failed
cleanup() {
    if [ "$status" != complete ]; then
        touch "$HOP_DIR/.failed"
    fi
    rm -f "$HOP_DIR/launcher.pid"
}
trap cleanup EXIT
ulimit -s unlimited

{
    echo "started_at=$(date --iso-8601=seconds)"
    echo "host=$(hostname)"
    echo "final_output=$FINAL_OUTPUT"
    echo "laggenetic_head=$(git -C "$HERE/../../.." rev-parse HEAD)"
    sha256sum "$HOP_BUILD/hop" "$HOP_BUILD/regroup" "$HOP_BUILD/poshalo"
    sha256sum "$HERE/run_inverted_parent_hop_lageunha.sh"
} > "$HOP_DIR/provenance.txt"

{
    TIMEFORMAT=$'real_seconds=%R\nuser_seconds=%U\nsystem_seconds=%S'
    time "$HOP_BUILD/hop" \
        -in "$FINAL_OUTPUT/part_00006.out" -p 1. -o "$HOP_ROOT" \
        > "$HOP_DIR/hop.log" 2>&1
} 2> "$HOP_DIR/hop.time"
{
    TIMEFORMAT=$'real_seconds=%R\nuser_seconds=%U\nsystem_seconds=%S'
    time "$HOP_BUILD/regroup" \
        -root "$HOP_ROOT" -douter 80. -dsaddle 200. -dpeak 240. \
        -f77 -o "$GROUP_ROOT" > "$HOP_DIR/regroup.log" 2>&1
} 2> "$HOP_DIR/regroup.time"
{
    TIMEFORMAT=$'real_seconds=%R\nuser_seconds=%U\nsystem_seconds=%S'
    time "$HOP_BUILD/poshalo" \
        -inp "$FINAL_OUTPUT" -pre "$GROUP_ROOT" \
        > "$HOP_DIR/poshalo.log" 2>&1
} 2> "$HOP_DIR/poshalo.time"

for output in "$GROUP_ROOT.tag" "$GROUP_ROOT.size" "$GROUP_ROOT.pos"; do
    if [ ! -s "$output" ]; then
        echo "required inverted HOP output is empty or absent: $output" >&2
        exit 2
    fi
done
{
    echo "completed_at=$(date --iso-8601=seconds)"
    sha256sum "$GROUP_ROOT.size" "$GROUP_ROOT.pos"
} >> "$HOP_DIR/provenance.txt"
touch "$HOP_DIR/.complete"
status=complete
echo "V4 inverted-parent HOP catalogue passed"
