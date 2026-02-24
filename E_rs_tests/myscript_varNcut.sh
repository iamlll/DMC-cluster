#!/bin/bash
#SBATCH -J Ncut10
#SBATCH -n 16
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
Ncuts=(15)
#Ncuts=(5 10 15 20 25)
l=100
eta=0.2

for N in ${Ncuts[@]};
do
    echo "Starting loop for Ncut=$Ncut:"
    for rv in ${rss[@]};
    do
        filename=N$N\_U$((l*2))\_eta$eta\_rs$rv\.out
        outdir=Ncut$N\_U$((l*2))\_eta$eta\_data
        python -u testphonons.py --eta $eta --l $l --Ncut $N --nconf 32 --rs $rv --tproj 64 --tau 20 --outdir $outdir > $filename & #forces command to run in backgro
    done
done

wait
date
