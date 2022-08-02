#!/bin/bash
#SBATCH -J bindingtest
#SBATCH -n 8
#SBATCH --mem-per-cpu 7gb
#SBATCH -t 23:59:58 # max time limit of 2 days
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

rs=30
eta=0
l=5
nstep=2500 #4000 for rs=90
nconfigs=(2500)

date
for Nw in ${nconfigs[@]};
do
    tau=$(bc <<< "scale=2; $rs/40")
    echo $tau
    jfilename=Nwtest.out
    joutdir=rs$rs\_nconfig$Nw\_nstep$nstep\_data
    #python -u testphonons.py --eta $eta --l $l --Ncut 15 --nconf $Nw --gth 0 --rs $rs --Nstep $nstep --tau $tau --ph 0 --savestep 1 --outdir $joutdir > $jfilename & #forces command to run in background
    python -u testphonons.py --eta $eta --l $l --Ncut 15 --nconf $Nw --gth 0 --rs $rs --Nstep $nstep --tau 2 --ph 0 --savestep 1 --outdir $joutdir > $jfilename & #jellium
    #python -u testphonons.py --eta $eta --l $l --Ncut 15 --nconf $Nw --gth 0 --rs $rs --Nstep $nstep --tau $tau --ph 0 --popstep 2 --savestep 1 --outdir $joutdir > $jfilename & #jellium
    #python -u testphonons.py --eta $eta --l $l --Ncut 15 --nconf $Nw --gth 0 --rs $rs --Nstep $nstep --tau $tau --ph 0 --popstep 20 --savestep 1 --outdir $joutdir > $jfilename & #jellium
    #python -u testphonons.py --eta $eta --l $l --Ncut 15 --nconf $Nw --gth 0 --rs $rs --Nstep $nstep --tau $tau --diffusion 1 --outdir $joutdir > $jfilename & #jellium

done

wait
date
