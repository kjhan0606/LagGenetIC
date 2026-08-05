#!/bin/bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
WORKROOT=${V4_WORKROOT:-/gpfs/kjhan/VoidSim/v4_parent_n512}
WORKROOT=$(realpath -m "$WORKROOT")
STAGE=${1:-all}
MPI_RANKS=${MPI_RANKS:-16}
MONO_THREADS=${MONO_THREADS:-4}
GENETIC_THREADS=${GENETIC_THREADS:-16}

CAMB_ROOT=${CAMB_ROOT:-/home/kjhan/BACKUP/lagCAMB_validation/lagCAMB}
CAMB_BIN=${CAMB_BIN:-/home/kjhan/BACKUP/lagCAMB_validation/venv/bin/camb}
MONOFONIC_BIN=${MONOFONIC_BIN:-$HERE/../../../monofonIC/build/monofonIC}
GENETIC_BIN=${GENETIC_BIN:-$HERE/../../../genetIC/genetIC/genetIC}
RAMSES_BIN=${RAMSES_BIN:-/home/kjhan/BACKUP/lagRamses/bin/ramses_final3d}
MPIEXEC=${MPIEXEC:-mpirun}
CONVERTER=${CONVERTER:-$HERE/../../monofonic_to_genetic_zoom/monofonic_to_genetic_wn.py}

for executable in "$CAMB_BIN" "$MONOFONIC_BIN" "$GENETIC_BIN" "$RAMSES_BIN"; do
    if [ ! -x "$executable" ]; then
        echo "executable not found: $executable" >&2
        exit 2
    fi
done
if [ ! -f "$CAMB_ROOT/inifiles/planck_2018.ini" ]; then
    echo "CAMB Planck base ini not found under $CAMB_ROOT" >&2
    exit 2
fi
case "$WORKROOT" in
    /gpfs/kjhan/*) ;;
    *) echo "V4_WORKROOT must be a dedicated directory below /gpfs/kjhan" >&2; exit 2 ;;
esac

mkdir -p "$WORKROOT"/{camb,monofonic,genetic,ramses_initial}

stage_requested() {
    [ "$STAGE" = "all" ] || [ "$STAGE" = "$1" ]
}

run_checked() {
    local name=$1
    shift
    local marker="$WORKROOT/.${name}.complete"
    local failed="$WORKROOT/.${name}.failed"
    if [ -f "$marker" ]; then
        echo "[$name] already complete"
        return
    fi
    rm -f "$failed"
    echo "[$name] starting"
    if "$@"; then
        touch "$marker"
        echo "[$name] complete"
    else
        touch "$failed"
        echo "[$name] failed; inspect its log under $WORKROOT" >&2
        return 1
    fi
}

run_camb() {
    install -m 0644 "$CAMB_ROOT/inifiles/planck_2018.ini" \
        "$WORKROOT/camb/planck_2018_base.ini"
    install -m 0644 "$HERE/lagcamb_z49.ini" "$WORKROOT/camb/lagcamb_z49.ini"
    (cd "$WORKROOT/camb" && "$CAMB_BIN" lagcamb_z49.ini > camb.log 2>&1)
    test -s "$WORKROOT/camb/lagcamb_transfer_z49_raw.dat"
    python3 "$HERE/prepare_transfer.py" \
        "$WORKROOT/camb/lagcamb_transfer_z49_raw.dat" \
        "$WORKROOT/camb/camb_transfer_z49.dat" \
        > "$WORKROOT/camb/prepare_transfer.log" 2>&1
    test -s "$WORKROOT/camb/camb_transfer_z49.dat"
}

run_monofonic() {
    test -s "$WORKROOT/camb/camb_transfer_z49.dat"
    install -m 0644 "$HERE/monofonic_parent.conf" \
        "$WORKROOT/monofonic/monofonic_parent.conf"
    cp "$WORKROOT/camb/camb_transfer_z49.dat" "$WORKROOT/monofonic/"
    (cd "$WORKROOT/monofonic" && \
        OMP_NUM_THREADS="$MONO_THREADS" I_MPI_PIN_DOMAIN=omp \
        "$MPIEXEC" -np "$MPI_RANKS" "$MONOFONIC_BIN" monofonic_parent.conf \
        > monofonic.log 2>&1)
    test -s "$WORKROOT/monofonic/wn.h5"
}

run_convert() {
    test -s "$WORKROOT/monofonic/wn.h5"
    python3 "$CONVERTER" "$WORKROOT/monofonic/wn.h5" \
        "$WORKROOT/genetic/wn_level0.npy" --expect-n 512 \
        > "$WORKROOT/genetic/convert.log" 2>&1
    test -s "$WORKROOT/genetic/wn_level0.npy"
}

run_genetic() {
    test -s "$WORKROOT/genetic/wn_level0.npy"
    install -m 0644 "$HERE/genetic_parent.txt" "$WORKROOT/genetic/genetic_parent.txt"
    cp "$WORKROOT/camb/camb_transfer_z49.dat" "$WORKROOT/genetic/"
    (cd "$WORKROOT/genetic" && OMP_NUM_THREADS="$GENETIC_THREADS" \
        "$GENETIC_BIN" genetic_parent.txt > genetic.log 2>&1)
    test -s "$WORKROOT/genetic/v4_parent.grafic_512/ic_particle_ids"
}

run_ramses() {
    test -s "$WORKROOT/genetic/v4_parent.grafic_512/ic_particle_ids"
    install -m 0644 "$HERE/ramses_initial.nml" \
        "$WORKROOT/ramses_initial/ramses_initial.nml"
    (cd "$WORKROOT/ramses_initial" && OMP_NUM_THREADS=1 \
        "$MPIEXEC" -np "$MPI_RANKS" "$RAMSES_BIN" ramses_initial.nml \
        > ramses.log 2>&1)
    test -s "$WORKROOT/ramses_initial/output_00001/info_00001.txt"
}

run_verify() {
    python3 "$HERE/verify_parent.py" "$WORKROOT" --grid-size 512 \
        --ranks "$MPI_RANKS" | tee "$WORKROOT/verify_parent.log"
}

if stage_requested camb; then run_checked camb run_camb; fi
if stage_requested monofonic; then run_checked monofonic run_monofonic; fi
if stage_requested convert; then run_checked convert run_convert; fi
if stage_requested genetic; then run_checked genetic run_genetic; fi
if stage_requested ramses; then run_checked ramses run_ramses; fi
if stage_requested verify; then run_checked verify run_verify; fi

case "$STAGE" in
    all|camb|monofonic|convert|genetic|ramses|verify) ;;
    *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac

echo "V4 work root: $WORKROOT"
