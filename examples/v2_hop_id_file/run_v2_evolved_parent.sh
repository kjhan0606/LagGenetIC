#!/bin/bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
MONOFONIC_BIN=${MONOFONIC_BIN:-"$HERE/../../monofonIC/build/monofonIC"}
GENETIC_BIN=${GENETIC_BIN:-"$HERE/../../genetIC/genetIC/genetIC"}
RAMSES_ROOT=${RAMSES_ROOT:-}
RAMSES_BIN=${RAMSES_BIN:-}
MPIEXEC=${MPIEXEC:-mpirun}
NP=${NP:-16}
RAMSES_NP=${RAMSES_NP:-4}
KEEP_V2_WORKDIR=${KEEP_V2_WORKDIR:-0}

if [ -z "$RAMSES_ROOT" ]; then
    echo "RAMSES_ROOT must point to the lagRamses source directory" >&2
    exit 2
fi
if [ -z "$RAMSES_BIN" ]; then
    RAMSES_BIN="$RAMSES_ROOT/bin/ramses_final3d"
fi
HOP_SOURCE="$RAMSES_ROOT/utils/f90/hop_ramses"
for exe in "$MONOFONIC_BIN" "$GENETIC_BIN" "$RAMSES_BIN"; do
    if [ ! -x "$exe" ]; then
        echo "executable not found: $exe" >&2
        exit 2
    fi
done
if [ ! -f "$HOP_SOURCE/Makefile" ]; then
    echo "HOP source directory not found: $HOP_SOURCE" >&2
    exit 2
fi

CREATED_WORKDIR=0
if [ -n "${V2_WORKDIR:-}" ]; then
    WORKDIR=$(realpath -m "$V2_WORKDIR")
    if [ -e "$WORKDIR" ]; then
        echo "V2_WORKDIR already exists; refusing to overwrite: $WORKDIR" >&2
        exit 2
    fi
    mkdir -p "$WORKDIR"
else
    WORKDIR=$(mktemp -d -t voidsim-v2-XXXXXX)
    CREATED_WORKDIR=1
fi

cleanup() {
    if [ "$CREATED_WORKDIR" -eq 1 ] && [ "$KEEP_V2_WORKDIR" != "1" ]; then
        case "$WORKDIR" in
            /tmp/voidsim-v2-*) rm -rf -- "$WORKDIR" ;;
            *) echo "refusing to remove unexpected work directory: $WORKDIR" >&2 ;;
        esac
    else
        echo "V2 artifacts kept at $WORKDIR"
    fi
}
trap cleanup EXIT

mkdir -p "$WORKDIR/mono" "$WORKDIR/parent" "$WORKDIR/hop_build"
cp "$HERE/parent.conf" "$WORKDIR/mono/parent.conf"
cp "$HERE/genetic.txt" "$WORKDIR/parent/genetic.txt"
cp "$HERE/ramses.nml" "$WORKDIR/parent/ramses.nml"
cp "$HERE/../../genetIC/genetIC/tests/camb_transfer_kmax40_z0.dat" \
   "$WORKDIR/parent/camb.dat"
cp "$HOP_SOURCE/Makefile" "$HOP_SOURCE/hop.c" "$HOP_SOURCE/hop_input.c" \
   "$HOP_SOURCE/kd.c" "$HOP_SOURCE/kd.h" "$HOP_SOURCE/smooth.c" \
   "$HOP_SOURCE/smooth.h" "$HOP_SOURCE/regroup.c" "$HOP_SOURCE/slice.c" \
   "$HOP_SOURCE/slice.h" "$HOP_SOURCE/poshalo.f90" "$WORKDIR/hop_build/"

echo "V2 work directory: $WORKDIR"

echo "[1/8] build bundled HOP tools"
(cd "$WORKDIR/hop_build" && \
    make CFLAGS='-O -mcmodel=medium -DHUGE=1.0e30' \
    hop regroup poshalo > build.log 2>&1)

echo "[2/8] monofonIC 64^3 parent white noise on $NP MPI ranks"
(cd "$WORKDIR/mono" && OMP_NUM_THREADS=1 "$MPIEXEC" -np "$NP" \
    "$MONOFONIC_BIN" parent.conf > monofonic.log 2>&1)

echo "[3/8] convert white noise and generate GRAFIC ICs"
python3 "$HERE/../monofonic_to_genetic_zoom/monofonic_to_genetic_wn.py" \
    "$WORKDIR/mono/wn.h5" "$WORKDIR/parent/wn.npy" --expect-n 64
(cd "$WORKDIR/parent" && OMP_NUM_THREADS=4 \
    "$GENETIC_BIN" genetic.txt > genetic.log 2>&1)

echo "[4/8] evolve DMO parent to a=1 on $RAMSES_NP MPI ranks"
(cd "$WORKDIR/parent" && OMP_NUM_THREADS=1 "$MPIEXEC" -np "$RAMSES_NP" \
    "$RAMSES_BIN" ramses.nml > ramses.log 2>&1)
mapfile -t SNAPSHOTS < <(
    find "$WORKDIR/parent" -maxdepth 1 -type d -name 'output_[0-9][0-9][0-9][0-9][0-9]' | sort
)
if [ "${#SNAPSHOTS[@]}" -lt 2 ]; then
    echo "lagRamses did not write initial and final snapshots" >&2
    exit 1
fi
INITIAL_OUTPUT=${SNAPSHOTS[0]}
FINAL_INDEX=$((${#SNAPSHOTS[@]} - 1))
FINAL_OUTPUT=${SNAPSHOTS[$FINAL_INDEX]}
FINAL_NAME=$(basename "$FINAL_OUTPUT")
FINAL_NUMBER=${FINAL_NAME#output_}

echo "[5/8] HOP and regroup the evolved DMO parent"
HOP_ROOT="$WORKDIR/parent_hop"
GROUP_ROOT="$WORKDIR/parent_groups"
"$WORKDIR/hop_build/hop" \
    -in "$FINAL_OUTPUT/part_${FINAL_NUMBER}.out" -p 1. -o "$HOP_ROOT" \
    > "$WORKDIR/hop.log" 2>&1
"$WORKDIR/hop_build/regroup" \
    -root "$HOP_ROOT" -douter 80. -dsaddle 200. -dpeak 240. \
    -f77 -o "$GROUP_ROOT" > "$WORKDIR/regroup.log" 2>&1
"$WORKDIR/hop_build/poshalo" \
    -inp "$FINAL_OUTPUT" -pre "$GROUP_ROOT" > "$WORKDIR/poshalo.log" 2>&1

echo "[6/8] select the most populated HOP halo"
HALO_ID=$(awk 'BEGIN {count=-1} $1 !~ /^#/ && NF>=2 {if ($2>count) {count=$2; id=$1}} END {print id}' "$GROUP_ROOT.pos")
if [ -z "$HALO_ID" ]; then
    echo "no HOP halo was reported in $GROUP_ROOT.pos" >&2
    exit 1
fi
echo "selected one-based HOP halo ID $HALO_ID"

echo "[7/8] convert halo membership to a GenetIC id_file"
python3 "$HERE/hop_to_genetic_id.py" \
    "$FINAL_OUTPUT" "$GROUP_ROOT.tag" "$WORKDIR/largest_halo.id" \
    --halo-id "$HALO_ID" --grid-size 64

echo "[8/8] verify the initial Lagrangian envelope and GenetIC round trip"
python3 "$HERE/verify_evolved_mask.py" \
    "$INITIAL_OUTPUT" "$GROUP_ROOT.pos" "$WORKDIR/largest_halo.id" \
    --halo-id "$HALO_ID" --grid-size 64 --genetic-binary "$GENETIC_BIN"

echo "V2 EVOLVED-PARENT REGRESSION PASSED"
