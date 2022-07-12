#!/bin/bash -x
#SBATCH --output=yfcc_subset_train_%A_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=47:59:00
#SBATCH --mem=128GB
#SBATCH --gpus-per-node=1
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=yfcc_subset_train
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=bf996@nyu.edu

module purge

singularity exec --nv \
    $(for sqf in /vast/work/public/ml-datasets/yfcc15m/data/*.sqf; do echo "--overlay $sqf:ro"; done) \
	--overlay /scratch/bf996/singularity_containers/openclip_env_cuda.ext3:ro \
    --overlay /scratch/bf996/datasets/imagenet-r.sqf:ro \
    --overlay /scratch/bf996/datasets/imagenet-a.sqf:ro \
    --overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro \
	/scratch/work/public/singularity/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif \
	/bin/bash -c 'source /ext3/env.sh; export PYTHONPATH="$PYTHONPATH:$PWD/src";\
    python -u src/training/main.py \
    --train-data="yfcc-subsets/yfcc_in1k.csv" \
    --csv-separator "," \
    --save-frequency 1 \
    --report-to wandb \
    --imagenet-val "/imagenet/val/" \
    --imagenet-a "/imagenet-a/" \
    --imagenet-r "/imagenet-r/" \
    --zeroshot-frequency=2 \
    --warmup 2000 \
    --batch-size=256 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs=32 \
    --workers=8 \
    --model=RN50'