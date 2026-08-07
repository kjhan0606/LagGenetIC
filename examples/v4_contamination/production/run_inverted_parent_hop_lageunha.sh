#!/bin/bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${V4_ROOT:-/gpfs/kjhan/VoidSim/v4_parent_n512}
FINAL_OUTPUT="$ROOT/inverted_parent_z0/evolution/output_00006"
HOP_BUILD="$ROOT/dmo_z0/hop_build"
HOP_DIR="$ROOT/inverted_parent_z0/hop_catalog"
HOP_ROOT="$HOP_DIR/inverted_hop"
GROUP_ROOT="$HOP_DIR/inverted_groups"
resume_density=${V4_HOP_RESUME_DENSITY:-no}
resume_groups=${V4_HOP_RESUME_GROUPS:-no}

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
if [ "$resume_density" != no ] && [ "$resume_density" != yes ]; then
    echo "V4_HOP_RESUME_DENSITY must be yes or no" >&2
    exit 2
fi
if [ "$resume_groups" != no ] && [ "$resume_groups" != yes ]; then
    echo "V4_HOP_RESUME_GROUPS must be yes or no" >&2
    exit 2
fi
if [ "$resume_density" = yes ] && [ "$resume_groups" = yes ]; then
    echo "density and group resumes are mutually exclusive" >&2
    exit 2
fi
new_run=yes
if [ -e "$HOP_DIR" ]; then
    if [ "$resume_density" != yes ] && [ "$resume_groups" != yes ]; then
        echo "inverted HOP output already exists; refusing to overwrite it" >&2
        exit 2
    fi
    new_run=no
    if [ ! -e "$HOP_DIR/.failed" ] || [ ! -s "$HOP_ROOT.den" ]; then
        echo "resume requires a failed run with a nonempty .den file" >&2
        exit 2
    fi
    if [ "$resume_density" = yes ] && \
        { [ -e "$HOP_ROOT.hop" ] || [ -e "$HOP_ROOT.gbound" ]; }; then
        echo "density resume refuses to replace existing HOP group outputs" >&2
        exit 2
    fi
    if [ "$resume_groups" = yes ]; then
        if [ ! -s "$HOP_ROOT.hop" ] || [ ! -s "$HOP_ROOT.gbound" ]; then
            echo "group resume requires complete .hop and .gbound files" >&2
            exit 2
        fi
        if [ -e "$GROUP_ROOT.tag" ] || [ -e "$GROUP_ROOT.size" ] || \
            [ -e "$GROUP_ROOT.pos" ]; then
            echo "group resume refuses to replace regrouped outputs" >&2
            exit 2
        fi
    fi
    density_particles=$(od -An -td4 -N4 "$HOP_ROOT.den" | tr -d ' ')
    if [ "$density_particles" != 134217728 ]; then
        echo "density file declares $density_particles particles, expected 134217728" >&2
        exit 2
    fi
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

if [ "$new_run" = yes ]; then
    {
        echo "started_at=$(date --iso-8601=seconds)"
        echo "host=$(hostname)"
        echo "final_output=$FINAL_OUTPUT"
        echo "laggenetic_head=$(git -C "$HERE/../../.." rev-parse HEAD)"
        sha256sum "$HOP_BUILD/hop" "$HOP_BUILD/regroup" "$HOP_BUILD/poshalo"
        sha256sum "$HERE/run_inverted_parent_hop_lageunha.sh"
    } > "$HOP_DIR/provenance.txt"
else
    if [ "$resume_groups" = yes ]; then
        mv "$HOP_DIR/.failed" "$HOP_DIR/.failed_after_hop_output"
        if [ -e "$HOP_DIR/regroup.log" ]; then
            mv "$HOP_DIR/regroup.log" "$HOP_DIR/regroup_failed_absolute_root.log"
        fi
        if [ -e "$HOP_DIR/regroup.time" ]; then
            mv "$HOP_DIR/regroup.time" "$HOP_DIR/regroup_failed_absolute_root.time"
        fi
    else
        mv "$HOP_DIR/.failed" "$HOP_DIR/.failed_after_density_stage"
    fi
    {
        echo "resumed_at=$(date --iso-8601=seconds)"
        echo "resume_laggenetic_head=$(git -C "$HERE/../../.." rev-parse HEAD)"
        sha256sum "$HERE/run_inverted_parent_hop_lageunha.sh"
    } >> "$HOP_DIR/provenance.txt"
fi

if [ -e "$HOP_DIR/snapshot" ] || [ -L "$HOP_DIR/snapshot" ]; then
    if [ "$(readlink -f "$HOP_DIR/snapshot")" != "$FINAL_OUTPUT" ]; then
        echo "snapshot link does not resolve to the inverted final output" >&2
        exit 2
    fi
else
    ln -s "$FINAL_OUTPUT" "$HOP_DIR/snapshot"
fi

if [ "$resume_groups" != yes ]; then
    hop_time="$HOP_DIR/hop.time"
    if [ "$resume_density" = yes ]; then
        hop_time="$HOP_DIR/hop_resume.time"
    fi
    (
        TIMEFORMAT=$'real_seconds=%R\nuser_seconds=%U\nsystem_seconds=%S'
        cd "$HOP_DIR"
        # The legacy reader uses 80-byte input and output buffers.  Both names
        # must remain short before the reader appends rank and product suffixes.
        if [ "$resume_density" = yes ]; then
            time "$HOP_BUILD/hop" -in "snapshot/part_00006.out" \
                -den "inverted_hop.den" -p 1. -o "inverted_hop" \
                > "$HOP_DIR/hop_resume.log" 2>&1
        else
            time "$HOP_BUILD/hop" -in "snapshot/part_00006.out" \
                -p 1. -o "inverted_hop" > "$HOP_DIR/hop.log" 2>&1
        fi
    ) 2> "$hop_time"
fi
(
    TIMEFORMAT=$'real_seconds=%R\nuser_seconds=%U\nsystem_seconds=%S'
    cd "$HOP_DIR"
    time "$HOP_BUILD/regroup" \
        -root "inverted_hop" -douter 80. -dsaddle 200. -dpeak 240. \
        -f77 -o "inverted_groups" > "$HOP_DIR/regroup.log" 2>&1
) 2> "$HOP_DIR/regroup.time"
(
    TIMEFORMAT=$'real_seconds=%R\nuser_seconds=%U\nsystem_seconds=%S'
    cd "$HOP_DIR"
    time "$HOP_BUILD/poshalo" \
        -inp "$FINAL_OUTPUT" -pre "inverted_groups" \
        > "$HOP_DIR/poshalo.log" 2>&1
) 2> "$HOP_DIR/poshalo.time"

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
