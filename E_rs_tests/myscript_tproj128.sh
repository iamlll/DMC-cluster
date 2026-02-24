#!/bin/bash
#SBATCH -J eta0.2-U50-Ncut123
#SBATCH -n 8
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
python -u testphonons.py --eta 0.2 --l 50 --Ncut 10 --nconf 128 --rs 4 --tproj 128 --tau 40 --outdir tproj128 > rs1.out & #forces command to run in backgro
python -u testphonons.py --eta 0.2 --l 50 --Ncut 20 --nconf 128 --rs 4 --tproj 128 --tau 40 --outdir tproj128 > rs1.out & #forces command to run in backgro
python -u testphonons.py --eta 0.2 --l 50 --Ncut 30 --nconf 128 --rs 4 --tproj 128 --tau 40 --outdir tproj128 > rs1.out & #forces command to run in backgro
python -u testphonons.py --eta 0.2 --l 50 --Ncut 40 --nconf 128 --rs 4 --tproj 128 --tau 40 --outdir tproj128 > rs1.out & #forces command to run in backgro
wait
date
