#!/bin/bash
#SBATCH --nodes=1
#SBATCH -J imdbwiki-baseline                                               # job name
#SBATCH --workdir=/home/bkargi/regression-uncertainty/  		                    # working directory
#SBATCH --gres=gpu:1          # or you can specify gpu type: --gres=gpu:rtx2080ti:1					# how many gpu(s) will be reserved
#SBATCH --output=/home/bkargi/regression-uncertainty/slurm_outs/%j.out	              # output file
#SBATCH --error=/home/bkargi/regression-uncertainty/slurm_outs/%j.err	              # error file
#SBATCH --time=4-20:00:00				                                # max job time. (currently no need to use)
#SBATCH --nodelist=dl2

########### Cuda paths, you may not need that #############
export PATH=/usr/local/cuda-10.0/bin:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-10.0/lib64
#########################################################

echo $PATH
source /home/bkargi/miniconda3/bin/activate dl             # you can activate your conda env
WANDB__SERVICE_WAIT=300 python tools/train.py --config configs/imdb_wiki_dataset.yaml --device cuda --checkpoint checkpoints/ensembles_fixed \
    --only_estimators --seed -1

echo "JOB DONE!!!"

