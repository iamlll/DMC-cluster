#!/bin/bash
#SBATCH -J Ncut10
#SBATCH -n 1
#SBATCH -t 1-23:59:58 # max time limit of 2 days
#SBATCH -p genx,gen,ccq # partition from which slurm will select the requested amt of nodes
#SBATCH --mail-type=BEGIN,END #Mail when job starts and ends
#SBATCH --mail-user=iamlll@mit.edu #email recipient

cd $SLURM_SUBMIT_DIR

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated = $SLURM_NTASKS"

module load pyscf
module load python
module list

date
rss=(1 2 3 5 8 10 12 15 18 20 25 30 40 50 60 70 80 90 100)
Ncut=15
Ls=(1 2 5 10 20 50)
eta=$(bc <<< 0.2)
# want eta = 0,0.01,0.05,0.1,0.2)
etas=(0 0.01 0.05 0.1 0.2)
fetas=(bc <<< "${etas[*]}")

for eta in ${etas[@]};
do
    echo $eta
    for ll in ${Ls[@]};
    do
        D=phasedia1/eta$eta\_U$((ll*2))
        echo $D
    done
done
