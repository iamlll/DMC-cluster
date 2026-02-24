#!/bin/bash
#SBATCH --job-name=multi
#SBATCH -n 100
#SBATCH --mem-per-cpu 7gb #memory per processor to be used for each task
#SBATCH -p genx,gen,ccq # partition from which slurm will select the requested amt of nodes
#SBATCH -t 1-23:59:58

cd $SLURM_SUBMIT_DIR

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated = $SLURM_NTASKS"

source /etc/profile #commands in the file will be executed as they were executed in command line instead of in a new subshell

# ORDER OF STEPS:
# 1. Generate simulation parameters with generateparams.py
# e.g. python generateparams.py --xvar Dg_ratio --xb 0 1 25 --yvar theta_c0_ratio --yb 0.01 0.99 25 --outdir test

# 2. Simulate system using these parameters - make sure that [total number of jobs being submitted (--array)] x [number of tasks per job (per_task)] = total number of different parameter combos. You can check the file length of the parameter file with "wc -l [parameter file path]". Subtract 1 from the final count to account for the header row.
# NOTE: per_task = 25 takes around 15 min to run
# e.g. sbatch myscript.sh

#run simulation function
start=`date +%s.%N`
joutdir='/mnt/home/llin1/scratch/data_pol_30k'
paramfile=$joutdir/data_eta_l_params_lin_el1_ph1_coul0.csv
n=$(wc -l < $paramfile) #find param file length
echo $n
ID='dat'

END=$(( $n-1 ))

for ((i=1;i<=END;i++)); #read each line of the parameter file and submit to separate node since it looks like
do
    start_num=$(( $i-1 ))
    end_num=$i
    #echo $start_num
    #echo $end_num
    
    python -u jobarray.py --outdir $joutdir --startnum $start_num --endnum $end_num --paramfile $paramfile > $joutdir/$ID-$i.out & 
done

wait
end=`date +%s.%N`

runtime=$(echo "$end - $start" | bc -l)
seconds=$(bc <<< "scale=2; $runtime"); 
echo "Runtime: $seconds seconds"
