#!/bin/bash
#SBATCH -J bindtest
#SBATCH -n 1
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

module load pyscf/2.1.0
module load python
module list
source $VENVDIR/rice/bin/activate

rs=30
eta=0.9
l=10 #5.56
Nw=512
seeds=(0)
popstep=50 #200
arrstep=200 #200
savestep=5 #5
savephonons=0
resume=1
init=bind
#init='/mnt/home/llin1/scratch/E_Nw_tests/rs30_nconfig512_data_eta0_l20_Econstraint_tau75/DMC_bind_diffusion0_el1_ph1_rs30_popsize512_seed0_N15_eta0.00_l20.00_nstep80000_popstep200_arrstep200_tau0.75.h5'

nstep=100000 #4000 for rs=90
#tau=$(bc <<< "scale=2; $rs/40")
tau=0.75
echo $tau
#jfilename=elph.out
#joutdir=rs$rs\_nconfig$Nw\_data_eta$eta\_l$l\_noconstraint
joutdir=rs$rs\_nconfig$Nw\_data_eta$eta\_l$l\_Econstraint_tau$tau
echo $joutdir
if [ ! -d "$joutdir" ]; then
  echo "$joutdir does not exist. Creating directory..."
  mkdir $joutdir
fi

date
for seed in ${seeds[@]};
#for seed in {1..10};
do
    # el + ph
    #python -u testphonons.py --savestep $savestep --diffusion 0 --eta $eta --l $l --Ncut 15 --seed $seed --nconf $Nw --gth 0 --rs $rs --Nstep $nstep --tau $tau --ph 1 --resume $resume --outdir $joutdir --arrstep $arrstep --popstep $popstep --savephonons $savephonons > elphtest_rs$rs\_eta$eta\_l$l\_seed$seed\_diff0\_popstep$popstep\_arrstep$arrstep\_saveph$savephonons\_tau$tau.out & 
    # no ph (i.e. jellium)
    python -u testphonons.py --savestep $savestep --diffusion 0 --init $init --eta $eta --l $l --Ncut 15 --seed $seed --nconf $Nw --gth 0 --rs $rs --Nstep $nstep --tau $tau --ph 0 --resume $resume --outdir $joutdir --arrstep $arrstep --popstep $popstep --savephonons $savephonons > elphtest_rs$rs\_eta$eta\_l$l\_seed$seed\_diff0\_popstep$popstep\_arrstep$arrstep\_saveph$savephonons\_tau$tau\_ph$ph.out & 

    python -u testphonons.py --savestep $savestep --diffusion 0 --init $init --eta $eta --l $l --Ncut 15 --seed $seed --nconf $Nw --gth 0 --rs $rs --Nstep $nstep --tau $tau --ph 1 --resume $resume --outdir $joutdir --arrstep $arrstep --popstep $popstep --savephonons $savephonons > elphtest_rs$rs\_eta$eta\_l$l\_seed$seed\_diff0\_popstep$popstep\_arrstep$arrstep\_saveph$savephonons\_tau$tau\_ph$ph.out & 

    # command line command for debugging
    # jellium
    #python testphonons.py --savestep 10 --diffusion 0 --init bind --eta 0 --l 10 --Ncut 15 --seed 0 --nconf 5 --gth 0 --rs 30 --Nstep 5000 --tau 0.75 --ph 0 --resume 0 --outdir rs30_nconfig5_tau0.75_jellium --arrstep 100 --popstep 50 --savephonons 100 > debug_rs30_nconfig5_tau0.75_popstep50_jellium.out
    # diffusion
    #python testphonons.py --savestep 10 --diffusion 1 --init bind --eta 0 --l 10 --Ncut 15 --seed 0 --nconf 5 --gth 0 --rs 30 --Nstep 5000 --tau 0.75 --ph 0 --resume 0 --outdir rs30_nconfig5_tau0.75_diffusion --arrstep 100 --popstep 50 --savephonons 100 > debug_rs30_nconfig5_tau0.75_popstep50_diffusion.out

    #python -u testphonons.py --eta $eta --l $l --Ncut 15 --seed $seed --nconf $Nw --gth 0 --rs $rs --Nstep $nstep --tau $tau --diffusion 1 --resume 1 --outdir $joutdir > $jfilename & #pure diffusion

# for interactive (command line) mode
#python -u testphonons.py --eta 0 --l 5 --Ncut 15 --seed 0 --nconf 512 --gth 0 --rs 30 --Nstep 1000 --tau 0.75 --ph 1 --resume 0 --outdir '.' --arrstep 200 --popstep 100
done

wait
date
