#!/bin/bash
#SBATCH -J comptests
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

rss=(30)
l=5
proj=128
nconfig=512

date
for rv in ${rss[@]};
do
    tau=$(bc <<< "scale=2; $rv/40")
    echo $tau
    if [ $(bc <<< "$tau < 1") == 1 ]; then
        tproj=$proj
    else
        tproj=$(bc <<< "2*$proj*$tau")
        tproj=${tproj%.*}
    fi
    echo $tproj
    jfilename=comp_rs_bind/comp_rs.out
    joutdir=comp_rs_bind/rs$rv\_nconfig$nconfig\_data
    python -u testphonons.py --eta 0 --l $l --Ncut 15 --nconf $nconfig --gth 0 --rs $rv --tproj $tproj --tau $tau --ph 1 --outdir $joutdir > $jfilename & #forces command to run in background
    python -u testphonons.py --eta 0 --l $l --Ncut 15 --init bind --nconf $nconfig --gth 0 --rs $rv --tproj $tproj --tau $tau --ph 0 --outdir $joutdir > $jfilename & #jellium run for comparison
    python -u testphonons.py --eta 0 --l $l --Ncut 15 --nconf $nconfig --rs $rv --tproj $tproj --gth 0 --tau $tau --diffusion 1 --init "bind" --outdir $joutdir > $jfilename & #pure diffusion run for comparison
done

wait
date
