#!/bin/bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
MONOFONIC_BIN=${MONOFONIC_BIN:-"$HERE/../../monofonIC/build/monofonIC"}
GENETIC_BIN=${GENETIC_BIN:-"$HERE/../../genetIC/genetIC/genetIC"}
RAMSES_BIN=${RAMSES_BIN:-}
MPIEXEC=${MPIEXEC:-mpirun}
NP=${NP:-4}
KEEP_V1_WORKDIR=${KEEP_V1_WORKDIR:-0}

if [ -z "$RAMSES_BIN" ]; then
    echo "RAMSES_BIN must point to ramses_final3d" >&2
    exit 2
fi
for exe in "$MONOFONIC_BIN" "$GENETIC_BIN" "$RAMSES_BIN"; do
    if [ ! -x "$exe" ]; then
        echo "executable not found: $exe" >&2
        exit 2
    fi
done

CREATED_WORKDIR=0
if [ -n "${V1_WORKDIR:-}" ]; then
    WORKDIR=$(realpath -m "$V1_WORKDIR")
    if [ -e "$WORKDIR" ]; then
        echo "V1_WORKDIR already exists; refusing to overwrite: $WORKDIR" >&2
        exit 2
    fi
    mkdir -p "$WORKDIR"
else
    WORKDIR=$(mktemp -d -t voidsim-v1-XXXXXX)
    CREATED_WORKDIR=1
fi

cleanup() {
    if [ "$CREATED_WORKDIR" -eq 1 ] && [ "$KEEP_V1_WORKDIR" != "1" ]; then
        case "$WORKDIR" in
            /tmp/voidsim-v1-*) rm -rf -- "$WORKDIR" ;;
            *) echo "refusing to remove unexpected work directory: $WORKDIR" >&2 ;;
        esac
    else
        echo "V1 artifacts kept at $WORKDIR"
    fi
}
trap cleanup EXIT

mkdir -p "$WORKDIR/mono_slab" "$WORKDIR/mono_ksec" "$WORKDIR/slab" "$WORKDIR/ksec"
cp "$HERE/parent_slab.conf" "$WORKDIR/mono_slab/parent.conf"
cp "$HERE/parent_ksec.conf" "$WORKDIR/mono_ksec/parent.conf"
for mode in slab ksec; do
    cp "$HERE/genetic.txt" "$WORKDIR/$mode/genetic.txt"
    cp "$HERE/ramses.nml" "$WORKDIR/$mode/ramses.nml"
    cp "$HERE/../../genetIC/genetIC/tests/camb_transfer_kmax40_z0.dat" \
       "$WORKDIR/$mode/camb.dat"
done

echo "V1 work directory: $WORKDIR"

echo "[1/7] monofonIC slab parent"
(cd "$WORKDIR/mono_slab" && OMP_NUM_THREADS=2 "$MPIEXEC" -np "$NP" \
    "$MONOFONIC_BIN" parent.conf > monofonic.log 2>&1)

echo "[2/7] monofonIC k-section parent"
(cd "$WORKDIR/mono_ksec" && OMP_NUM_THREADS=2 "$MPIEXEC" -np "$NP" \
    "$MONOFONIC_BIN" parent.conf > monofonic.log 2>&1)
if ! grep -q "K-section particle partition active" "$WORKDIR/mono_ksec/monofonic.log"; then
    echo "k-section path was not activated" >&2
    exit 1
fi

echo "[3/7] white-noise conversion"
python3 "$HERE/../monofonic_to_genetic_zoom/monofonic_to_genetic_wn.py" \
    "$WORKDIR/mono_slab/wn.h5" "$WORKDIR/slab/wn.npy" --expect-n 128
python3 "$HERE/../monofonic_to_genetic_zoom/monofonic_to_genetic_wn.py" \
    "$WORKDIR/mono_ksec/wn.h5" "$WORKDIR/ksec/wn.npy" --expect-n 128

echo "[4/7] GenetIC slab-derived grafic"
(cd "$WORKDIR/slab" && OMP_NUM_THREADS=4 "$GENETIC_BIN" genetic.txt > genetic.log 2>&1)

echo "[5/7] GenetIC k-section-derived grafic"
(cd "$WORKDIR/ksec" && OMP_NUM_THREADS=4 "$GENETIC_BIN" genetic.txt > genetic.log 2>&1)

echo "[6/7] lagRamses initial snapshots"
for mode in slab ksec; do
    (cd "$WORKDIR/$mode" && OMP_NUM_THREADS=1 "$MPIEXEC" -np "$NP" \
        "$RAMSES_BIN" ramses.nml > ramses.log 2>&1)
    if [ ! -d "$WORKDIR/$mode/output_00001" ]; then
        echo "lagRamses did not write $mode/output_00001" >&2
        exit 1
    fi
done

echo "[7/7] invariant checks"
python3 "$HERE/verify_v1.py" "$WORKDIR"

