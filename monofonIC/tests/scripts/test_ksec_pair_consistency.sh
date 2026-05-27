#!/bin/bash
#
# Generic slab-vs-ksec pair regression test for monofonIC.
#
# Runs the slab path (single rank) and the k-section path at np in
# {1, 2, 4} on a config pair that differs only in
# `UseKSectionParticles`. Asserts bit-identity (after sorting particles
# by position because the ksec path may re-ID particles).
#
# Designed for any ParticleLoad that has both a slab and a k-section
# implementation: SC/BCC/FCC/RSC, masked, glass (glass has its own
# wrapper that also generates the glass HDF5 first, see
# test_glass_consistency.sh).
#
# Usage:
#   test_ksec_pair_consistency.sh <monofonic_exe> <slab_conf> <ksec_conf> \
#                                 <compare_glass_sorted.py> <python_exe>
#

set -e

if [ $# -ne 5 ]; then
    echo "Usage: $0 <monofonic_exe> <slab_conf> <ksec_conf> <compare.py> <python>"
    exit 1
fi

MONOFONIC_EXE="$1"
SLAB_CONF="$2"
KSEC_CONF="$3"
COMPARE="$4"
PYTHON_EXE="$5"

WORK_DIR=$(mktemp -d)
trap "rm -rf $WORK_DIR" EXIT

cp "$SLAB_CONF" "$WORK_DIR/"
cp "$KSEC_CONF" "$WORK_DIR/"

SLAB_CONF_BASENAME=$(basename "$SLAB_CONF")
KSEC_CONF_BASENAME=$(basename "$KSEC_CONF")

cd "$WORK_DIR"

# Extract output basenames from each config (single line each, value after `=`).
SLAB_OUT_BASE=$(grep "^filename" "$SLAB_CONF_BASENAME" | awk '{print $3}' | sed 's/\.hdf5$//')
KSEC_OUT_BASE=$(grep "^filename" "$KSEC_CONF_BASENAME" | awk '{print $3}' | sed 's/\.hdf5$//')

if [ -z "$SLAB_OUT_BASE" ] || [ -z "$KSEC_OUT_BASE" ]; then
    echo "Could not extract output filenames from configs"
    exit 1
fi

echo "=========================================="
echo "K-section pair consistency test"
echo "=========================================="
echo "Slab conf: $SLAB_CONF_BASENAME (output: ${SLAB_OUT_BASE}.hdf5)"
echo "Ksec conf: $KSEC_CONF_BASENAME (output: ${KSEC_OUT_BASE}.hdf5)"
echo "Work dir:  $WORK_DIR"
echo ""

# 1. Slab reference (np=1; slab MPI determinism is covered by test_mpi_consistency).
echo "--- Slab reference (np=1) ---"
"$MONOFONIC_EXE" "$SLAB_CONF_BASENAME" > slab.log 2>&1 || {
    echo "Slab run failed:"; cat slab.log; exit 1;
}
SLAB_OUT="${SLAB_OUT_BASE}.hdf5"
[ -f "$SLAB_OUT" ] || SLAB_OUT="${SLAB_OUT_BASE}.0.hdf5"
[ -f "$SLAB_OUT" ] || { echo "slab output not found"; ls; exit 1; }

# 2. Ksec runs.
MPI_TASKS=(1)
MPI_EXTRA_FLAGS=()
if command -v mpirun > /dev/null 2>&1; then
    MPI_TASKS=(1 2 4)
    # --oversubscribe is Open MPI only; Intel MPI / MPICH reject it.
    if mpirun --version 2>&1 | grep -qi "open[ -]*mpi"; then
        MPI_EXTRA_FLAGS=(--oversubscribe)
    fi
fi

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

    mv "$KSEC_OUT" "stash_ksec_np${NP}.hdf5" 2>/dev/null || true
done

echo "=========================================="
echo "All slab-vs-ksec checks passed."
echo "=========================================="
exit 0
