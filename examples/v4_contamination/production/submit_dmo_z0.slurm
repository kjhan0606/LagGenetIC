#!/bin/bash
#SBATCH --job-name=void_dmo512
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --mem=480G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=grammar007,grammar012,grammar023,grammar081,grammar112
#SBATCH --output=dmo-%j.out
#SBATCH --error=dmo-%j.err

set -euo pipefail

RUNDIR=/gpfs/kjhan/VoidSim/v4_parent_n512/dmo_z0
EXPECTED_BINARY_SHA256=ce3e5a14c22639fd14e3c70092694eeb2b6fb3a009aefafdd7e20fbf585a592e

source /opt/ohpc/pub/intel/oneapi-2025.03/setvars.sh >/dev/null 2>&1
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so
export I_MPI_PIN=1
export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close
ulimit -s unlimited

cd "$RUNDIR"
actual_sha256=$(sha256sum ramses_final3d | awk '{print $1}')
if [ "$actual_sha256" != "$EXPECTED_BINARY_SHA256" ]; then
    echo "RAMSES binary checksum mismatch: $actual_sha256" >&2
    exit 2
fi
if [ ! -s ../genetic/v4_parent.grafic_512/ic_particle_ids ]; then
    echo "validated level-9 GRAFIC input is absent" >&2
    exit 2
fi

echo "VoidSim V4 DMO start=$(date --iso-8601=seconds)"
echo "job=$SLURM_JOB_ID host=$(hostname) nodes=$SLURM_NNODES ranks=$SLURM_NTASKS"
echo "binary_sha256=$actual_sha256"

set +e
srun --kill-on-bad-exit=1 --mpi=pmi2 --cpu-bind=cores \
    ./ramses_final3d ramses_dmo_z0.nml > ramses.log 2>&1
run_status=$?
set -e
echo "RAMSES exit=$run_status end=$(date --iso-8601=seconds)"
if [ "$run_status" -ne 0 ]; then
    exit "$run_status"
fi

python3 verify_dmo_z0.py "$RUNDIR" --grid-size 512 --ranks "$SLURM_NTASKS" \
    | tee verify_dmo_z0.log
