#!/bin/bash -x

#SBATCH --output=inference-cpu_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=47:59:00
#SBATCH --mem=128GB
#SBATCH --job-name=inference-cpu
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=bf996@nyu.edu

module purge;

#debug flags
echo $SLURM_JOB_NAME

#env vars
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
export MASTER_ADDR="$(hostname -s).hpc.nyu.edu"
echo "MASTER_ADDR="$MASTER_ADDR

#run command
srun --cpu_bind=v --accel-bind=v \
    /bin/bash src/script/run-singularity-infds.bash \
    /bin/bash -c \
    'export PYTHONPATH="$PYTHONPATH:/scratch/bf996/open_clip/src"; python -u /scratch/bf996/open_clip/src/training/main.py --imagenet-val "/imagenet/val/" --zeroshot-frequency=1 --workers=16 --model=RN50; python -u /scratch/bf996/open_clip/src/training/main.py --imagenet-val "/imagenet/val/" --zeroshot-frequency=1 --workers=16 --model=RN50 --pretrained=yfcc15m  --precision=fp32; python -u /scratch/bf996/open_clip/src/training/main.py --imagenet-val "/imagenet/val/" --zeroshot-frequency=1 --workers=16 --model=RN50; python -u /scratch/bf996/open_clip/src/training/main.py --imagenet-val "/imagenet/val/" --imagenet-v2 "/scratch/bf996/datasets/" --imagenet-a "/scratch/bf996/datasets/imagenet-a" --imagenet-s "/scratch/bf996/datasets/imagenet-sketch" --zeroshot-frequency=1 --workers=16 --model=RN50 --pretrained=openai  --precision=fp32'