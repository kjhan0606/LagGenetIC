#!/bin/bash
#
# Hessian cache consistency regression for LPT3.
#
# Runs the LPT3 base config in three modes and asserts:
#   1. cache=on (unlimited budget) is bit-identical to cache=off  → C1
#   2. cache=on with a tight HessianCacheBudgetGB (forces evictions)
#      matches cache=off within ≤1e-12 rtol — the documented 1-ULP FP drift
#      introduced when FFTW rebuilds evicted padded grids from scratch     → C2
#
# Usage:
#   test_hessian_cache_consistency.sh <monofonic_exe> <base_conf> \
#                                      <compare.py> <python>
#
set -e

if [ $# -ne 4 ]; then
    echo "Usage: $0 <monofonic_exe> <base_conf> <compare.py> <python>"
    exit 1
fi

abspath() { python3 -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$1"; }
MONOFONIC_EXE=$(abspath "$1")
BASE_CONF=$(abspath "$2")
COMPARE=$(abspath "$3")
PYTHON_EXE="$4"

WORK_DIR=$(mktemp -d)
trap "rm -rf $WORK_DIR" EXIT

# Generate variants by overriding the [setup] EnableHessianCache /
# HessianCacheBudgetGB lines (already present in base config so [setup]
# is the active section) and the [output] filename.
make_variant() {
    local label="$1"
    local enable="$2"
    local budget_gb="$3"
    local conf="$WORK_DIR/${label}.conf"
    sed \
        -e "s|^EnableHessianCache.*=.*|EnableHessianCache   = ${enable}|" \
        -e "s|^HessianCacheBudgetGB.*=.*|HessianCacheBudgetGB = ${budget_gb}|" \
        -e "s|^filename.*=.*|filename   = ${label}.hdf5|" \
        "$BASE_CONF" > "$conf"
    echo "$conf"
}

CONF_OFF=$(make_variant cache_off    no  0.0)
CONF_ON=$(make_variant  cache_on     yes 0.0)
# 1 MB budget — well under a single N=32 padded Hessian (3/2*32)^3 * 8B
# ≈ 880 kB; even one cached entry exceeds budget, so every miss evicts.
CONF_BUDGET=$(make_variant cache_budget yes 0.001)

cd "$WORK_DIR"

echo "=========================================="
echo "Hessian cache consistency test (LPT3)"
echo "=========================================="

run_variant() {
    local label="$1"
    local conf="$2"
    echo "--- $label ---" >&2
    "$MONOFONIC_EXE" "$conf" > "${label}.log" 2>&1 || {
        echo "$label run failed:" >&2; cat "${label}.log" >&2; exit 1
    }
    local out="${label}.hdf5"
    [ -f "$out" ] || out="${label}.0.hdf5"
    [ -f "$out" ] || { echo "$label output not found" >&2; ls >&2; exit 1; }
    echo "$out"
}

OUT_OFF=$(run_variant cache_off    "$CONF_OFF")
OUT_ON=$(run_variant  cache_on     "$CONF_ON")
OUT_BUDGET=$(run_variant cache_budget "$CONF_BUDGET")

echo ""
echo "--- C1: cache=on vs cache=off (bit-exact, rtol=0) ---"
"$PYTHON_EXE" "$COMPARE" "$OUT_OFF" "$OUT_ON" --rtol 0 --atol 0 || {
    echo "MISMATCH: unlimited Hessian cache must be bit-identical to no cache"
    exit 1
}

echo ""
echo "--- C2: cache=on+budget vs cache=off (rtol≤1e-12, FP drift OK) ---"
"$PYTHON_EXE" "$COMPARE" "$OUT_OFF" "$OUT_BUDGET" --rtol 1e-12 --atol 1e-14 || {
    echo "MISMATCH: bounded Hessian cache drifted beyond documented 1-ULP tolerance"
    exit 1
}

# Sanity: confirm the budget run actually evicted (otherwise the test is vacuous).
if grep -q "evicts=0" cache_budget.log; then
    if ! grep -E "evicts=[1-9]" cache_budget.log > /dev/null; then
        echo "WARNING: HessianCacheBudgetGB=0.001 did not produce any evictions."
        echo "         The eviction-tolerance check is not exercising the LRU path."
        grep -E "\[conv" cache_budget.log || true
        exit 1
    fi
fi

echo "=========================================="
echo "Hessian cache consistency checks passed."
echo "=========================================="
exit 0
