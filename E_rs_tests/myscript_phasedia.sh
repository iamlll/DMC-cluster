#!/bin/bash
#SBATCH -J PD0
#SBATCH -n 128
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

module load pyscf
module load python
module list

date
rss=(1 2 3 5 8 10 12 15 18 20 25 30 40 50 60 70 80 90 100)
Ncut=15
etas=(0.2)
Ls=(50)
tproj=128

for eta in ${etas[@]};
do
    echo $eta
    for ll in ${Ls[@]};
    do
        D=phasedia1/eta$eta\_U$((ll*2))
        echo $D
        if ! [[ -d "$D" ]]; then
            mkdir $D
        fi
    
        echo "Starting loop for Ncut=$Ncut:"
        for rv in ${rss[@]};
        do
            filename=$D\/N$Ncut\_U$((ll*2))\_eta$eta\_rs$rv\.out
            outdir=$D\/Ncut$Ncut\_U$((ll*2))\_eta$eta\_data
            tau=$(bc<<<"$rv/0.1")
            python -u testphonons.py --eta $eta --l $ll --Ncut $Ncut --nconf 32 --rs $rv --tproj $tproj --tau $tau --ph 1 --outdir $outdir > $filename & #forces command to run in background
        done
    done
done

wait
date
