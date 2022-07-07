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
#python -u testphonons.py --eta 0.2 --l 50 --Ncut 10 --nconf 32 --rs 1 --tproj 64 --tau 20 > rs1.out & #forces command to run in backgro
#python -u testphonons.py --eta 0.2 --l 50 --Ncut 10 --nconf 32 --rs 2 --tproj 64 --tau 20 > rs2.out & #forces command to run in backgro
#python -u testphonons.py --eta 0.2 --l 50 --Ncut 10 --nconf 32 --rs 3 --tproj 64 --tau 20 > rs3.out &
#python -u testphonons.py --eta 0.2 --l 50 --Ncut 10 --nconf 32 --rs 4 --tproj 64 --tau 20 > rs4.out &
python -u testphonons.py --eta 0.2 --l 50 --Ncut 10 --nconf 32 --rs 5 --tproj 64 --tau 20 > rs5.out &
python -u testphonons.py --eta 0.2 --l 50 --Ncut 10 --nconf 32 --rs 6 --tproj 64 --tau 20 > rs6.out &
python -u testphonons.py --eta 0.2 --l 50 --Ncut 10 --nconf 32 --rs 7 --tproj 64 --tau 20 > rs7.out &
python -u testphonons.py --eta 0.2 --l 50 --Ncut 10 --nconf 32 --rs 8 --tproj 64 --tau 20 > rs8.out &

python -u dmc_reblock.py data/DMC_*.csv
wait
date
