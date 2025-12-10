#!/bin/bash

# Usage:
# bash launch_experiment.sh
# This script sets up the environment and submits the Slurm job.

# Root and PYTHONPATH
export ROOT_PATH='/cluster/tufts/hugheslab/abaran03/SSL-vs-SSL-benchmark'
export PYTHONPATH="${PYTHONPATH}:$ROOT_PATH/src"

if [[ -z $1 ]]; then
    ACTION_NAME='list'
else
    ACTION_NAME=$1
fi

# Experiment settings
export method='PseudoLabeling'
#export dataset_name='IDRID' 
export dataset_name='FullCheXpertEffusion'
export data_seed=0
export training_seed=0
export train_dir="/cluster/tufts/hugheslab/sslbench/experiments/$dataset_name/pl_ft_fixed/data_seed$data_seed/training_seed$training_seed/$method/pretrained"
mkdir -p $train_dir

# Model / training hyperparameters
export arch='resnet18'
export train_epoch=200
export start_epoch=0
export total_hour=10
export use_pretrained=""
export num_workers=8
export labeledtrain_batchsize=32
export unlabeledtrain_batchsize=128
export optimizer_type='Adam'
export lr_warmup_epochs=0
export lr_schedule_type='CosineLR'
export lr_cycle_epochs=$train_epoch

# Data paths
export l_train_dataset_path='/cluster/tufts/hugheslab/datasets/CheXpert-v1.0-small/splits_effusion/train_labeled.csv'
export val_dataset_path='/cluster/tufts/hugheslab/datasets/CheXpert-v1.0-small/splits_effusion/val_labeled.csv'
export test_dataset_path='/cluster/tufts/hugheslab/datasets/CheXpert-v1.0-small/splits_effusion/test_labeled.csv'
export l_root_dataset_path='/cluster/tufts/hugheslab/datasets'
export u_train_dataset_path='/cluster/tufts/hugheslab/datasets/CheXpert-v1.0-small/splits_effusion/unlabeled.csv'
export u_root_dataset_path='/cluster/tufts/hugheslab/datasets'


# export l_train_dataset_path='/cluster/tufts/hugheslab/datasets/IDRID/Labels/train.csv'
# export val_dataset_path='/cluster/tufts/hugheslab/datasets/IDRID/Labels/validation.csv'
# export test_dataset_path='/cluster/tufts/hugheslab/datasets/IDRID/Labels/test.csv'
# export l_root_dataset_path='/cluster/tufts/hugheslab/datasets/IDRID/Images'
# export u_train_dataset_path='/cluster/tufts/hugheslab/datasets/AIROGS/train_0.csv'
# export u_root_dataset_path='/cluster/tufts/hugheslab/datasets/AIROGS/Images_0/0'



if [[ $ACTION_NAME == 'submit' ]]; then
    ## Use this line to submit the experiment to the batch scheduler
    sbatch ./do_experiment.slurm  # Fixed from `< ./do_experiment.slurm`
    
elif [[ $ACTION_NAME == 'run_here' ]]; then
    ## Use this line to run the script interactively
    bash ./do_experiment.slurm
fi
