#!/bin/bash
#SBATCH -J diffusion
#SBATCH -n 12
#SBATCH --mem-per-cpu 7gb
#SBATCH -t 23:59:58 # max time limit of 2 days
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

# setting Coulomb interaction in wffiles/egas.py to 0 (jellium w/o e-e interaction, i.e. pure diffusion) -- looking for increasing energy as fxn of sqrt(tau)

module load pyscf
module load python
module list

date
rs=90
taus=(0.01 0.05 0.1 0.2 0.4 0.6 1 1.5 2 3 5 10)
proj=128
nconfig=512

date
for tau in ${taus[@]};
do
    jfilename=diffusion/diff.out
    joutdir=diffusion/diff_rs$rs\_nconfig$nconfig\_data
    if [ $(bc <<< "$tau<1") == 1 ]; then
        tproj=$proj
    else
        tproj=$(bc<<<"$proj*$tau")
    fi
    python -u testphonons.py --eta 0 --l 1 --Ncut 5 --nconf $nconfig --rs $rs --tproj $tproj --tau $tau --ph 0 --outdir $joutdir > $jfilename & #forces command to run in background
done

wait
date
