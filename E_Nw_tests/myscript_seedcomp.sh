#!/bin/bash
#SBATCH -J comptests
#SBATCH -n 18
#SBATCH --mem-per-cpu 7gb
#SBATCH -t 1-23:59:58 # max time limit of 2 days
#SBATCH -p genx,gen,ccq # partition from which slurm will select the requested amt of nodes

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

rs=90 #(4 90)
eta=0
l=5
nstep=30000 #4000 for rs=90
Nw=512
seeds=(6 7 8 9 10)

date
#for seed in ${seeds[@]};
for seed in {0..5};
do
    tau=$(bc <<< "scale=2; $rs/40")
    echo $tau
    jfilename=seedtest.out
    joutdir=rs$rs\_nconfig$Nw\_nstep$nstep\_data
    python -u testphonons.py --eta $eta --l $l --Ncut 15 --seed $seed --nconf $Nw --gth 0 --rs $rs --Nstep $nstep --tau $tau --ph 1 --resume 0 --outdir $joutdir > $jfilename & #forces command to run in background
    python -u testphonons.py --eta $eta --l $l --Ncut 15 --seed $seed --nconf $Nw --gth 0 --rs $rs --Nstep $nstep --tau $tau --ph 0 --resume 0 --outdir $joutdir > $jfilename & #jellium
    python -u testphonons.py --eta $eta --l $l --Ncut 15 --seed $seed --nconf $Nw --gth 0 --rs $rs --Nstep $nstep --tau $tau --diffusion 1 --resume 0 --outdir $joutdir > $jfilename & #jellium

done

wait
date
