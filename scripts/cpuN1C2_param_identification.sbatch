#!/bin/bash

# 48hrs min walltime:
#SBATCH --time 48:00:00

# 1 node:
#SBATCH --nodes 1

# 1 slot for MPI ranks per node:
#SBATCH --ntasks-per-node 1

# 4 CPUs per MPI rank:
#SBATCH --cpus-per-task 2

# 4GB of RAM per CPU:
#SBATCH --mem-per-cpu=4000

# jobname:
#SBATCH -J infectio-param

# stdouts
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
j
# Notification
# Mail alert at BEGIN|END|FAIL|ALL
# #SBATCH --mail-type=END
# #SBATCH --mail-user=a.mokaria-forooshani@hzdr.de

cleanup ()
{
	echo "Catch TERM SIGNAL | Clean up"
	rm -rf $LOCATION_TMPFS/$SLURM_JOB_ID
	exit
}
trap 'cleanup' TERM

LOG_FILE=./output/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log

function Log {
	local level=$1
	local msg=$2
	echo $(date --rfc-3339=seconds):${level} ${msg} >> ${LOG_FILE}
}

Log INFO "JOB START"
Log INFO "JOB NAME = ${SLURM_JOB_NAME}"

echo "Show CPU ids visible to the job:"
numactl --show

echo -e "\nallocated GPUs are exclusively visible to this job, other GPUs \
are not visible:" nvidia-smi
echo -e "\nmodules already loaded:"
module list

# List modules you want to load
Log INFO "loading modules"
Log INFO "Loading module python ..."
module load git python/3.11 >> ${LOG_FILE} 2>&1

echo -e "\nmodule loaded for batch job:"
module list

Log INFO "changing to directory: ${SLURM_SUBMIT_DIR}"
cd $SLURM_SUBMIT_DIR

NODES=$(scontrol show hostname $SLURM_JOB_NODELIST | paste -d, -s)
Log INFO "allocated nodes ${NODES}"

source ./venv/bin/activate >> ${LOG_FILE} 2>&1

# Capture all arguments passed to this script
python ./infectio/models/vacv/run.py "$@" >> ${LOG_FILE} 2>&1

Log INFO "JOB FINISH"