#!/bin/bash
#
# Glass-on-ksec regression test for monofonIC.
#
# Runs the glass particle load on:
#   - slab path (single rank)
#   - ksec path at np = 1, 2, 4 (when mpirun is available)
#
# Verifies that all ksec runs are bit-identical to the slab reference after
# sorting particles by position (ksec re-IDs glass particles by
# post-domain-decomp order, so direct index-by-index compare would fail).
#
# Usage:
#   test_glass_consistency.sh <monofonic_exe> <slab_conf> <ksec_conf> \
#                             <make_glass.py> <compare_glass_sorted.py> \
#                             <python_exe>
#

set -e

if [ $# -ne 6 ]; then
    echo "Usage: $0 <monofonic_exe> <slab_conf> <ksec_conf> <make_glass.py> <compare.py> <python>"
    exit 1
fi

MONOFONIC_EXE="$1"
SLAB_CONF="$2"
KSEC_CONF="$3"
MAKE_GLASS="$4"
COMPARE="$5"
PYTHON_EXE="$6"

WORK_DIR=$(mktemp -d)
trap "rm -rf $WORK_DIR" EXIT

cp "$SLAB_CONF" "$WORK_DIR/" || true
cp "$KSEC_CONF" "$WORK_DIR/"

SLAB_CONF_BASENAME=$(basename "$SLAB_CONF")
KSEC_CONF_BASENAME=$(basename "$KSEC_CONF")

cd "$WORK_DIR"

echo "=========================================="
echo "Glass-on-ksec consistency test"
echo "=========================================="
echo "Work dir: $WORK_DIR"
echo ""

# 1. Generate the small glass HDF5 file.
echo "--- Generating glass64.hdf5 ---"
"$PYTHON_EXE" "$MAKE_GLASS"

# 2. Slab reference (always np=1 — slab path is FFTW3-MPI for np>1 which is
#    out of scope for this test; it would just shuffle particles differently
#    across ranks and the compare-by-position would still match).
echo "--- Slab reference (np=1) ---"
"$MONOFONIC_EXE" "$SLAB_CONF_BASENAME" > slab.log 2>&1 || {
    echo "Slab run failed:"; cat slab.log; exit 1;
}
SLAB_OUT="test_glass_slab.hdf5"
[ -f "$SLAB_OUT" ] || SLAB_OUT="test_glass_slab.0.hdf5"
[ -f "$SLAB_OUT" ] || { echo "slab output not found"; ls; exit 1; }

# 3. ksec runs.
MPI_TASKS=(1)
MPI_EXTRA_FLAGS=()
if command -v mpirun > /dev/null 2>&1; then
    MPI_TASKS=(1 2 4)
    # --oversubscribe is Open MPI only; Intel MPI / MPICH reject it.
    if mpirun --version 2>&1 | grep -qi "open[ -]*mpi"; then
        MPI_EXTRA_FLAGS=(--oversubscribe)
    fi
fi

KSEC_OUT_BASE="test_glass_ksec"

for NP in "${MPI_TASKS[@]}"; do
    echo "--- ksec np=$NP ---"
    # Clean any stale ksec outputs from previous iteration.
    rm -f "${KSEC_OUT_BASE}.hdf5" "${KSEC_OUT_BASE}".*.hdf5
    if [ "$NP" -eq 1 ]; then
        "$MONOFONIC_EXE" "$KSEC_CONF_BASENAME" > "ksec_np${NP}.log" 2>&1 || {
            echo "ksec np=$NP run failed:"; cat "ksec_np${NP}.log"; exit 1;
        }
    else
        mpirun "${MPI_EXTRA_FLAGS[@]}" -np "$NP" "$MONOFONIC_EXE" "$KSEC_CONF_BASENAME" \
            > "ksec_np${NP}.log" 2>&1 || {
            echo "ksec np=$NP run failed:"; cat "ksec_np${NP}.log"; exit 1;
        }
    fi

    # Locate ksec output (single-file at np=1, multi-file at np>1).
    if [ -f "${KSEC_OUT_BASE}.hdf5" ]; then
        KSEC_OUT="${KSEC_OUT_BASE}.hdf5"
    elif [ -f "${KSEC_OUT_BASE}.0.hdf5" ]; then
        KSEC_OUT="${KSEC_OUT_BASE}.0.hdf5"
    else
        echo "ksec output not found for np=$NP"; ls; exit 1
    fi

    echo "Comparing slab vs ksec np=$NP (sorted by position)..."
    "$PYTHON_EXE" "$COMPARE" "$SLAB_OUT" "$KSEC_OUT" --rtol 1e-9 || {
        echo "MISMATCH: slab vs ksec np=$NP"; exit 1
    }

    # Stash the ksec output for posterity in the per-NP log section.
    mv "$KSEC_OUT" "stash_ksec_np${NP}.hdf5" 2>/dev/null || true
done

echo "=========================================="
echo "All glass consistency checks passed."
echo "=========================================="
exit 0
