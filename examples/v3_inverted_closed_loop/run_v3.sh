#!/bin/bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
V2_SOURCE=${1:-${V2_WORKDIR:-}}
GENETIC_BIN=${GENETIC_BIN:-"$HERE/../../genetIC/genetIC/genetIC"}
RAMSES_BIN=${RAMSES_BIN:-}
MPIEXEC=${MPIEXEC:-mpirun}
RAMSES_NP=${RAMSES_NP:-4}
HALO_ID=${HALO_ID:-21}
KEEP_V3_WORKDIR=${KEEP_V3_WORKDIR:-0}

if [ -z "$V2_SOURCE" ]; then
    echo "pass the completed V2 work directory as the first argument" >&2
    exit 2
fi
if [ -z "$RAMSES_BIN" ]; then
    echo "RAMSES_BIN must point to ramses_final3d" >&2
    exit 2
fi
for exe in "$GENETIC_BIN" "$RAMSES_BIN"; do
    if [ ! -x "$exe" ]; then
        echo "executable not found: $exe" >&2
        exit 2
    fi
done

V2_SOURCE=$(realpath "$V2_SOURCE")
PARENT_DIR=${V2_PARENT_DIR:-"$V2_SOURCE/parent"}
GROUP_TAG=${V2_GROUP_TAG:-"$V2_SOURCE/parent_groups.tag"}
WN_FILE=${V2_WN_FILE:-"$PARENT_DIR/wn.npy"}
for path in "$PARENT_DIR" "$GROUP_TAG" "$WN_FILE"; do
    if [ ! -e "$path" ]; then
        echo "required V2 artifact not found: $path" >&2
        exit 2
    fi
done
mapfile -t PARENT_OUTPUTS < <(
    find "$PARENT_DIR" -maxdepth 1 -type d -name 'output_[0-9][0-9][0-9][0-9][0-9]' | sort
)
if [ "${#PARENT_OUTPUTS[@]}" -lt 1 ]; then
    echo "no evolved RAMSES output found under $PARENT_DIR" >&2
    exit 2
fi
FINAL_INDEX=$((${#PARENT_OUTPUTS[@]} - 1))
FINAL_OUTPUT=${PARENT_OUTPUTS[$FINAL_INDEX]}

CREATED_WORKDIR=0
if [ -n "${V3_WORKDIR:-}" ]; then
    WORKDIR=$(realpath -m "$V3_WORKDIR")
    if [ -e "$WORKDIR" ]; then
        echo "V3_WORKDIR already exists; refusing to overwrite: $WORKDIR" >&2
        exit 2
    fi
    mkdir -p "$WORKDIR"
else
    WORKDIR=$(mktemp -d -t voidsim-v3-XXXXXX)
    CREATED_WORKDIR=1
fi

cleanup() {
    if [ "$CREATED_WORKDIR" -eq 1 ] && [ "$KEEP_V3_WORKDIR" != "1" ]; then
        case "$WORKDIR" in
            /tmp/voidsim-v3-*) rm -rf -- "$WORKDIR" ;;
            *) echo "refusing to remove unexpected work directory: $WORKDIR" >&2 ;;
        esac
    else
        echo "V3 artifacts kept at $WORKDIR"
    fi
}
trap cleanup EXIT

mkdir -p "$WORKDIR/normal" "$WORKDIR/inverted"

echo "V3 work directory: $WORKDIR"
echo "[1/4] build the non-wrapping HOP halo mask"
python3 "$HERE/../v2_hop_id_file/hop_to_genetic_id.py" \
    "$FINAL_OUTPUT" "$GROUP_TAG" "$WORKDIR/halo.id" \
    --halo-id "$HALO_ID" --grid-size 64

echo "[2/4] generate matched normal and sign-inverted zoom ICs"
for mode in normal inverted; do
    cp "$WORKDIR/halo.id" "$WORKDIR/$mode/halo.id"
    cp "$WN_FILE" "$WORKDIR/$mode/wn.npy"
    cp "$HERE/../../genetIC/genetIC/tests/camb_transfer_kmax40_z0.dat" \
       "$WORKDIR/$mode/camb.dat"
    cp "$HERE/genetic_${mode}.txt" "$WORKDIR/$mode/genetic.txt"
    (cd "$WORKDIR/$mode" && OMP_NUM_THREADS=4 \
        "$GENETIC_BIN" genetic.txt > genetic.log 2>&1)
done

echo "[3/4] ingest the inverted two-level GRAFIC IC with lagRamses"
cp "$HERE/ramses_inverted.nml" "$WORKDIR/inverted/ramses.nml"
(cd "$WORKDIR/inverted" && OMP_NUM_THREADS=1 "$MPIEXEC" -np "$RAMSES_NP" \
    "$RAMSES_BIN" ramses.nml > ramses.log 2>&1)

echo "[4/4] verify sign inversion, masked refinement, and particle IDs"
python3 "$HERE/verify_v3.py" \
    "$WORKDIR/normal" "$WORKDIR/inverted" "$WORKDIR/inverted" \
    --halo-id-file "$WORKDIR/halo.id" --grid-size 64

echo "V3 CLOSED-LOOP REGRESSION PASSED"
