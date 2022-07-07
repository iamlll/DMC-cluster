#!/bin/bash
#SBATCH -J jell
#SBATCH -n 4
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
rss=(4 90)
tproj=128
nconfig=512

date
for rv in ${rss[@]};
do
    jfilename=comp_rs/comp_rs.out
    joutdir=comp_rs/rs$rv\_tproj$tproj\_nconfig$nconfig\_data
    tau=0.1 #$(bc<<<"$rv/80")
    python -u testphonons.py --eta 0 --l 5 --Ncut 15 --nconf $nconfig --rs $rv --tproj $tproj --tau $tau --ph 0 --outdir $joutdir > $jfilename & #forces command to run in background
    python -u testphonons.py --eta 0.2 --l 50 --Ncut 15 --nconf $nconfig --rs $rv --tproj $tproj --tau $tau --ph 0 --outdir $joutdir > $jfilename & #forces command to run in background
done

wait
date
