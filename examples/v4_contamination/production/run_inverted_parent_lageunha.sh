#!/bin/bash
set -eo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
V4_ROOT=${V4_WORKROOT:-/gpfs/kjhan/VoidSim/v4_parent_n512}
ORIGINAL_DMO="$V4_ROOT/dmo_z0"
RUNDIR="$V4_ROOT/inverted_parent_z0"
GENETIC_BIN=${GENETIC_BIN:-/home/kjhan/BACKUP/VoidSim/code/LagGenetIC/genetIC/genetIC/genetIC}
RAMSES_SOURCE="$ORIGINAL_DMO/ramses_final3d"
EXPECTED_RAMSES_SHA256=ce3e5a14c22639fd14e3c70092694eeb2b6fb3a009aefafdd7e20fbf585a592e

if [ "$(hostname | tr '[:upper:]' '[:lower:]')" != "lageunha" ]; then
    echo "this launcher must run on LagEunha, not $(hostname)" >&2
    exit 2
fi
if ! grep -q "V4 Z=0 DMO PARENT PASSED" "$ORIGINAL_DMO/verify_dmo_z0.log"; then
    echo "the original parent verification gate has not passed" >&2
    exit 2
fi
if [ ! -e "$ORIGINAL_DMO/hop_catalog/.complete" ]; then
    echo "the original parent target catalogue is incomplete" >&2
    exit 2
fi
if [ -e "$RUNDIR" ]; then
    echo "$RUNDIR already exists; refusing to overwrite it" >&2
    exit 2
fi
for executable in "$GENETIC_BIN" "$RAMSES_SOURCE"; do
    if [ ! -x "$executable" ]; then
        echo "executable is absent: $executable" >&2
        exit 2
    fi
done
actual_ramses_sha256=$(sha256sum "$RAMSES_SOURCE" | awk '{print $1}')
if [ "$actual_ramses_sha256" != "$EXPECTED_RAMSES_SHA256" ]; then
    echo "validated RAMSES binary checksum mismatch: $actual_ramses_sha256" >&2
    exit 2
fi

mkdir -p "$RUNDIR/genetic" "$RUNDIR/evolution"
install -m 0644 "$HERE/genetic_inverted_parent.txt" "$RUNDIR/genetic/genetic.txt"
install -m 0644 "$HERE/ramses_inverted_parent_z0.nml" "$RUNDIR/evolution/ramses.nml"
install -m 0755 "$HERE/verify_inverted_parent_ic.py" "$RUNDIR/verify_inverted_parent_ic.py"
install -m 0755 "$HERE/verify_dmo_z0.py" "$RUNDIR/evolution/verify_dmo_z0.py"
install -m 0644 "$HERE/../../v2_hop_id_file/hop_to_genetic_id.py" \
    "$RUNDIR/evolution/hop_to_genetic_id.py"
install -m 0755 "$RAMSES_SOURCE" "$RUNDIR/evolution/ramses_final3d"
ln -s "$V4_ROOT/camb/camb_transfer_z49.dat" "$RUNDIR/genetic/camb_transfer_z49.dat"
ln -s "$V4_ROOT/genetic/wn_level0.npy" "$RUNDIR/genetic/wn_level0.npy"

{
    echo "started_at=$(date --iso-8601=seconds)"
    echo "host=$(hostname)"
    echo "laggenetic_head=$(git -C /home/kjhan/BACKUP/VoidSim/code/LagGenetIC rev-parse HEAD)"
    echo "binary_sha256=$actual_ramses_sha256"
    sha256sum "$GENETIC_BIN" "$RUNDIR/genetic/genetic.txt" "$RUNDIR/evolution/ramses.nml"
} > "$RUNDIR/provenance.txt"

source /opt/ohpc/pub/intel/oneapi/setvars.sh
set -u
export OMP_PROC_BIND=close
export OMP_PLACES=cores
ulimit -s unlimited

echo "[1/4] generate the globally inverted level-9 parent IC"
(cd "$RUNDIR/genetic" && taskset -c 16-31 env OMP_NUM_THREADS=16 \
    "$GENETIC_BIN" genetic.txt > genetic.log 2>&1)

echo "[2/4] verify the exact normal/inverted IC pair"
python3 "$RUNDIR/verify_inverted_parent_ic.py" \
    "$V4_ROOT/genetic/v4_parent.grafic_512" \
    "$RUNDIR/genetic/v4_parent_inverted.grafic_512" --grid-size 512 \
    | tee "$RUNDIR/verify_inverted_parent_ic.log"

echo "[3/4] evolve the inverted parent to z=0 on 64 MPI ranks"
export OMP_NUM_THREADS=1
export I_MPI_PIN=1
export I_MPI_PIN_DOMAIN=core
export I_MPI_PIN_PROCESSOR_LIST=0-63
export I_MPI_FABRICS=shm
set +e
(cd "$RUNDIR/evolution" && mpirun -np 64 ./ramses_final3d ramses.nml \
    > ramses.log 2>&1)
run_status=$?
set -e
if [ "$run_status" -ne 0 ]; then
    echo "inverted parent RAMSES failed with status $run_status" >&2
    exit "$run_status"
fi

echo "[4/4] verify the z=0 inverted parent"
python3 "$RUNDIR/evolution/verify_dmo_z0.py" "$RUNDIR/evolution" \
    --grid-size 512 --ranks 64 | tee "$RUNDIR/evolution/verify_dmo_z0.log"
touch "$RUNDIR/.complete"
echo "V4 INVERTED PARENT Z=0 PASSED"
